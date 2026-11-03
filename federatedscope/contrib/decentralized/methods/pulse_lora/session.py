"""PULSE-LoRA session policy: manifests, the eligible set, byte-budgeted
selection, per-edge ages and the planning log.

``core/pair_session.py`` runs the bilateral protocol: it charges both manifests
and both acknowledgements as control bytes before computing the payload
credit, asks each endpoint to plan on its own, and admits the payloads only if
the larger of the two frames fits. This module implements the hooks a
PULSE-LoRA peer answers with.

Both endpoints plan alone. There is no negotiation between planning and
sending, so the two peers must reach the same component set from the two
manifests, the shared per-edge state and the budget; the scores are symmetric
in the endpoints (the exact oracle up to rounding) and ties are broken by
component id.

Manifest bytes are charged per direction before the payload credit is
computed, so a larger manifest can leave less room for payload. Sets travel as hex
bitmaps over ``P`` rather than as id lists, and the disagreement sketches, the
only large field, are sent only under ``sketch_disagreement``.

``plan_payload`` runs on both peers even when the session then sends nothing,
so it records a pending decision; ``acknowledge``, called once per endpoint on
every path that ends a session, applies it to the selector state.
"""
import json
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

from federatedscope.contrib.decentralized.core.wire import encoded_size, payload_frame
from federatedscope.contrib.decentralized.methods.pulse_lora import scoring, selection, units
from federatedscope.contrib.decentralized.methods.pulse_lora.ages import (
    SelectorMemory, edge_key, exact_int,
)

#: Version of the manifest format; ``consume_manifest`` accepts only this one.
MANIFEST_SCHEMA_VERSION = 2

#: The counters a policy keeps, and exactly what a checkpoint must carry for an
#: exact resume. Module-level so callers can list them without constructing a
#: policy.
METRIC_NAMES = (
    "sessions_planned", "sessions_committed", "units_selected", "units_selectable",
    "units_eligible", "sessions_budget_bound",
    "units_committed", "exploration_picks", "utility_picks", "no_fit_sessions",
    "no_candidate_sessions", "clamped_scores", "failed_sessions",
)


class SessionError(ValueError):
    pass


def _finite(value, where: str) -> float:
    """A numeric manifest field as a float, or a SessionError naming it.

    Rejects booleans (``bool`` is an ``int`` subclass, so ``True`` would pass as
    ``1.0``), NaN and the infinities. NaN compares false with everything, so a
    candidate scored NaN could sort differently at the two endpoints.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SessionError(f"manifest {where} is {value!r}, not a number")
    number = float(value)
    if number != number:
        raise SessionError(f"manifest {where} is NaN")
    if number in (float("inf"), float("-inf")):
        raise SessionError(f"manifest {where} is {number}, not finite")
    return number


# -- bitmaps ---------------------------------------------------------------------

def pack_bitmap(components: Sequence[int], r_max: int) -> str:
    """A set over ``P`` as hex. 64 components cost 16 characters, not ~200."""
    width = (int(r_max) + 7) // 8
    raw = bytearray(width)
    for component in components:
        component = int(component)
        if not 0 <= component < int(r_max):
            raise SessionError(f"component {component} is outside P = 0..{r_max - 1}")
        raw[component // 8] |= 1 << (component % 8)
    return raw.hex()


def unpack_bitmap(blob: str, r_max: int) -> Tuple[int, ...]:
    width = (int(r_max) + 7) // 8
    try:
        raw = bytes.fromhex(str(blob))
    except ValueError as exc:
        raise SessionError(f"inventory bitmap is not hex: {blob!r}") from exc
    if len(raw) != width:
        raise SessionError(
            f"inventory bitmap is {len(raw)} bytes but r_max={r_max} needs {width}; "
            f"the peers disagree about the component space")
    out = [c for c in range(int(r_max)) if raw[c // 8] >> (c % 8) & 1]
    # A bit set past r_max inside the final byte would silently vanish above.
    for bit in range(int(r_max), width * 8):
        if raw[bit // 8] >> (bit % 8) & 1:
            raise SessionError(
                f"inventory bitmap sets bit {bit}, outside P = 0..{r_max - 1}")
    return tuple(out)


# -- pending decision -------------------------------------------------------------

@dataclass
class PendingDecision:
    """What this session would do, held until ``acknowledge`` says what happened."""
    edge: Tuple[int, int]
    eligible: Tuple[int, ...] = ()
    selected: Tuple[int, ...] = ()
    reserved: int = 0
    could_fit: bool = False
    slots: int = 0
    #: the peer's state version when this plan was computed, so staging can
    #: refuse a mix built against a state that has since moved
    version: int = 0
    planned_bytes: int = 0
    reason: str = ""
    #: Which of the selected components the exploration reservation served.
    explored: Tuple[int, ...] = ()


# -- the privileged oracle ----------------------------------------------------------

class OracleRegistry:
    """Process-local peer directory for the ``exact_disagreement`` oracle.

    The exact score reads the partner's tensors directly, which no manifest
    carries and only the single-process simulation can do. The factory creates a
    registry only when that selector is configured, and the policy refuses the
    selector without one.
    """

    def __init__(self):
        self._peers: Dict[int, object] = {}

    def register(self, peer_id: int, peer) -> None:
        self._peers[int(peer_id)] = peer

    def get(self, peer_id: int):
        return self._peers.get(int(peer_id))


# -- the policy ----------------------------------------------------------------------

class PulseSessionPolicy:
    """A PULSE-LoRA peer's answers to the pair-session hooks, and the selector
    state, metrics and planning log behind them."""

    def __init__(self, peer_id: int, components, state, selector: str,
                 exploration_fraction: float, sketch_dim: int,
                 includes_auxiliary: bool, checksum: bool = True,
                 oracle: OracleRegistry = None, seed: int = 0,
                 version_source=None):
        if selector not in selection.SELECTORS:
            raise SessionError(
                f"unknown selector {selector!r}; expected one of {list(selection.SELECTORS)}")
        if selector in selection.ORACLE_SELECTORS and oracle is None:
            raise SessionError(
                f"selector {selector!r} is a privileged simulation-only oracle and "
                f"needs an OracleRegistry; it is not a deployable protocol")
        self.peer_id = int(peer_id)
        self.components = components
        self.state = state
        self.selector = selector
        self.exploration_fraction = float(exploration_fraction)
        self.sketch_dim = int(sketch_dim)
        self.includes_auxiliary = bool(includes_auxiliary)
        self.checksum = bool(checksum)
        self.oracle = oracle
        # The peer owns the monotone state version; the policy only reports it.
        self.version_source = version_source or (lambda: 0)
        self.memory = SelectorMemory(seed=seed)
        self._pending: Dict[Tuple[int, int], PendingDecision] = {}
        #: The components this peer last decoded from the partner's payload, per
        #: edge (set during staging). Not checkpointed.
        self._observed: Dict[Tuple[int, int], Tuple[int, ...]] = {}
        #: What planning decided, one entry per committed session, per edge. The
        #: exploration quota is computed from the affordable slots, not from the
        #: eligible-set size, so it is recorded per session rather than inferred
        #: from the selector memory afterwards.
        self.planning_log: Dict[Tuple[int, int], list] = {}
        self.metrics = dict.fromkeys(METRIC_NAMES, 0)

    # -- stage 1: the manifest ------------------------------------------------------

    def build_manifest(self, session) -> bytes:
        """This peer's manifest for ``session``, as compact JSON.

        ``v`` format version, ``m`` method, ``id`` component-space identity,
        ``sel`` selector, ``aux`` auxiliary-state flag, ``sv`` state version,
        ``mat`` and ``sup`` the materialized set and the trainable support as
        bitmaps, ``sd`` this edge's selector-state digest, ``u`` bytes per
        component. The selector adds, for every materialized component, ``sc``
        (local-update magnitudes) or ``nm`` and ``sk`` (norms and sketches, with
        the sketch width ``d``).
        """
        edge = self._edge(session)
        declaration = {
            "v": MANIFEST_SCHEMA_VERSION,
            "m": "pulse_lora",
            "id": self.components.identity_digest()[:16],
            "sel": self.selector,
            "aux": self.includes_auxiliary,
            "sv": int(self.version_source()),
            "mat": pack_bitmap(self.state.materialised, self.components.r_max),
            "sup": pack_bitmap(self.state.support, self.components.r_max),
            "sd": self.memory.selector_digest(*edge),
            "u": units.unit_cost_bytes(self.components),
        }
        inventory = tuple(self.state.materialised)
        if self.selector in selection.SCORE_PUBLISHING_SELECTORS:
            declaration["sc"] = {
                str(c): scoring.local_update_magnitude_for(
                    self.components, c, self.state.gather(c))
                for c in inventory
            }
        elif self.selector == selection.SELECTOR_SKETCH:
            declaration["nm"] = {}
            declaration["sk"] = {}
            for component in inventory:
                block = self.state.gather(component)
                declaration["nm"][str(component)] = [
                    list(pair) for pair in scoring.component_norms(self.components, block)]
                declaration["sk"][str(component)] = [
                    [list(pa), list(qb)]
                    for pa, qb in scoring.component_sketch(
                        self.components, block, self.sketch_dim)]
            declaration["d"] = self.sketch_dim
        return json.dumps(declaration, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")

    # -- stage 2: the partner's manifest ----------------------------------------------

    def consume_manifest(self, remote_manifest: bytes, session) -> Mapping[str, object]:
        """Validate the partner's manifest and return it, with its bitmaps parsed
        into ``_materialised`` and ``_support``.

        Raises SessionError when the partner disagrees on the format, the method,
        the component space, the selector, the auxiliary-state policy, the unit
        cost, this edge's selector state or the sketch width, or when a field is
        malformed. Changes no state: the runtime stops at the first rejection, so
        a change here could leave the two endpoints out of step.
        """
        try:
            remote = json.loads(remote_manifest.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SessionError(f"unreadable manifest from the peer: {exc}") from exc
        if not isinstance(remote, dict):
            raise SessionError(
                f"manifest is a {type(remote).__name__}, not an object")

        if remote.get("v") != MANIFEST_SCHEMA_VERSION:
            raise SessionError(
                f"manifest schema version {remote.get('v')!r} is not "
                f"{MANIFEST_SCHEMA_VERSION}")
        if remote.get("m") != "pulse_lora":
            raise SessionError(f"peer runs method {remote.get('m')!r}, not pulse_lora")

        mine = self.components.identity_digest()[:16]
        if remote.get("id") != mine:
            raise SessionError(
                f"component-space disagreement: peer {str(remote.get('id'))[:12]}... "
                f"vs local {mine[:12]}.... The namespace covers structural hash, "
                f"r_max, scale and seed; components are not comparable across it.")
        if remote.get("sel") != self.selector:
            raise SessionError(
                f"selector disagreement: peer {remote.get('sel')!r} vs local "
                f"{self.selector!r}; the two endpoints would rank candidates "
                f"differently and select different sets")
        if remote.get("aux") != self.includes_auxiliary:
            raise SessionError(
                f"auxiliary-state policy disagreement: peer aux={remote.get('aux')}, "
                f"local={self.includes_auxiliary}")
        if int(remote.get("u", -1)) != units.unit_cost_bytes(self.components):
            raise SessionError(
                f"unit cost disagreement: peer {remote.get('u')} vs local "
                f"{units.unit_cost_bytes(self.components)} bytes per component")

        edge = self._edge(session)
        if remote.get("sd") != self.memory.selector_digest(*edge):
            raise SessionError(
                f"per-edge selector state has diverged between the endpoints "
                f"(peer {remote.get('sd')!r} vs local "
                f"{self.memory.selector_digest(*edge)!r}); they would rank and "
                f"explore differently. The digest covers the selector seed, the "
                f"ages, the cursor and the session counter.")

        # Parse the inventories here so a malformed bitmap is a manifest rejection
        # rather than an exception escaping plan_payload, which the runtime does
        # not catch.
        remote["_materialised"] = unpack_bitmap(remote.get("mat", ""),
                                                self.components.r_max)
        remote["_support"] = unpack_bitmap(remote.get("sup", ""),
                                           self.components.r_max)
        if self.selector == selection.SELECTOR_SKETCH:
            if int(remote.get("d", -1)) != self.sketch_dim:
                raise SessionError(
                    f"sketch width disagreement: peer {remote.get('d')} vs local "
                    f"{self.sketch_dim}")
        self._validate_scoring_payload(remote)
        return remote

    def _validate_scoring_payload(self, remote: Mapping[str, object]) -> None:
        """Check ``sc``, or ``nm`` and ``sk``, against the partner's declared
        materialized set.

        Each field needs exactly one entry per component the partner declares
        materialized, keyed by its canonical decimal id, with finite values of
        the expected shape. A missing entry would make this endpoint score the
        component as untouched while the partner scores its real value, and the
        two would select different sets; the initial-value fallback in
        ``_utility`` is only for components the partner does not hold.

        The checks run here rather than in ``plan_payload`` because the runtime
        turns an exception from ``consume_manifest`` into a rejected session,
        while one from ``plan_payload`` aborts the run.
        """
        declared = frozenset(remote["_materialised"])
        modules = len(self.components.lora.modules)

        if self.selector in selection.SCORE_PUBLISHING_SELECTORS:
            scores = self._require_mapping(remote.get("sc"), "sc")
            self._require_exact_components(scores, declared, "sc")
            for key, value in scores.items():
                _finite(value, f"score for component {key}")

        if self.selector != selection.SELECTOR_SKETCH:
            return

        norms = self._require_mapping(remote.get("nm"), "nm")
        sketches = self._require_mapping(remote.get("sk"), "sk")
        self._require_exact_components(norms, declared, "nm")
        self._require_exact_components(sketches, declared, "sk")

        for key in sorted(norms, key=int):
            per_module = norms[key]
            if not isinstance(per_module, list) or len(per_module) != modules:
                raise SessionError(
                    f"manifest norms for component {key} cover "
                    f"{len(per_module) if isinstance(per_module, list) else '?'} "
                    f"modules, expected {modules}")
            for pair in per_module:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise SessionError(
                        f"manifest norms for component {key} are not (aa, bb) pairs")
                for value in pair:
                    _finite(value, f"norm for component {key}")

            block = sketches[key]
            if not isinstance(block, list) or len(block) != modules:
                raise SessionError(
                    f"manifest sketch for component {key} covers "
                    f"{len(block) if isinstance(block, list) else '?'} modules, "
                    f"expected {modules}")
            for sides in block:
                if not isinstance(sides, list) or len(sides) != 2:
                    raise SessionError(
                        f"manifest sketch for component {key} is not an (A, B) pair")
                for vector in sides:
                    if not isinstance(vector, list) or len(vector) != self.sketch_dim:
                        raise SessionError(
                            f"manifest sketch for component {key} has width "
                            f"{len(vector) if isinstance(vector, list) else '?'}, "
                            f"expected {self.sketch_dim}")
                    for value in vector:
                        _finite(value, f"sketch for component {key}")

    def _require_mapping(self, payload, name: str) -> Mapping[str, object]:
        if not isinstance(payload, dict):
            raise SessionError(
                f"manifest {name!r} is {type(payload).__name__}, not an object; "
                f"selector {self.selector!r} requires one")
        return payload

    def _require_exact_components(self, payload, declared, name: str) -> None:
        """Require the raw keys of ``payload`` to be exactly the canonical decimal
        ids of ``declared``.

        The comparison is on the raw strings, not on ``int(key)``: planning looks
        entries up by ``str(component)``, so an alias that ``int()`` accepts
        (``"01"``, ``"+1"``, ``" 1"``, ``"-0"``) would pass an integer comparison
        and then fail the lookup in ``plan_payload``, and a canonical key next to
        its alias would give one component two entries.
        """
        expected = {str(int(component)) for component in declared}
        provided = set()
        for key in payload:
            if not isinstance(key, str):
                raise SessionError(
                    f"manifest {name!r} has a {type(key).__name__} key {key!r}; "
                    f"component keys are canonical decimal strings")
            provided.add(key)

        missing = sorted(expected - provided, key=int)
        extra = sorted(provided - expected)
        if not (missing or extra):
            return

        # Report an alias of a declared id as such, rather than as one missing
        # and one unexpected key.
        aliases = []
        for key in extra:
            try:
                component = int(key)
            except (TypeError, ValueError):
                continue
            if str(component) in expected:
                aliases.append((key, str(component)))
        if aliases:
            rendered = ", ".join(f"{raw!r} for {canon!r}" for raw, canon in aliases)
            raise SessionError(
                f"manifest {name!r} uses non-canonical component keys: {rendered}; "
                f"planning looks entries up by the canonical decimal string")

        raise SessionError(
            f"manifest {name!r} does not match the components the peer declares "
            f"materialised: missing={missing}, unexpected={extra}. A declared "
            f"component with no entry would score as if untouched, and the two "
            f"endpoints would then select different sets.")

    # -- stage 3: the payload plan ----------------------------------------------------

    def plan_payload(self, session, budget, remote_view) -> Optional[bytes]:
        """Plan this endpoint's payload: the encoded components, or None.

        Takes the eligible set (``S_i & S_j`` under ``survivor_only``), counts
        the complete components whose frame fits the session's payload credit
        and the run's wire-cost model (``slots``), and lets the selector choose
        ``min(slots, |Q|)`` of them:
        the exploration quota goes to the oldest candidates, the rest by score.
        The plan is kept as a pending decision for ``acknowledge``. Returns None
        when nothing is eligible or nothing fits.
        """
        if remote_view is None:
            return None
        edge = self._edge(session)
        pending = PendingDecision(edge=edge, version=int(self.version_source()))
        self._pending[edge] = pending
        self.metrics["sessions_planned"] += 1

        candidates = selection.candidate_components(
            self.selector,
            self.state.materialised, remote_view["_materialised"],
            self.state.support, remote_view["_support"],
        )
        pending.eligible = candidates
        if not candidates:
            pending.reason = "no candidate components"
            self.metrics["no_candidate_sessions"] += 1
            return None

        overhead = self._frame_overhead()
        # The count that fits the ledger's payload credit is an upper bound; the
        # run's wire-cost model (``logical`` or ``socket_v1``) can only lower it.
        # Under the logical model the two counts are equal.
        ceiling = units.affordable_units(self.components, budget.payload_credit(), overhead)
        slots = budget.affordable(
            lambda k: units.units_cost_bytes(self.components, k) + overhead, ceiling)
        pending.slots = slots
        pending.could_fit = slots > 0
        if slots == 0:
            pending.reason = "not one complete component fits after control overhead"
            self.metrics["no_fit_sessions"] += 1
            return None

        scores = selection.score_candidates(
            self.selector, self.memory, edge, candidates,
            self._utility(candidates, remote_view, edge))
        chosen, explored = selection.select_with_provenance(
            self.selector, self.memory, edge, candidates, scores, slots,
            self.exploration_fraction)
        pending.explored = explored
        if not chosen:
            pending.reason = "selector chose nothing"
            return None

        pending.selected = chosen
        pending.reserved = selection.exploration_slots(
            self.exploration_fraction, min(slots, len(candidates)))
        self.metrics["units_selected"] += len(chosen)
        # How many components this session could send: min(slots, |Q|). Every
        # selector fills all of them, so this equals units_selected.
        self.metrics["units_selectable"] += min(slots, len(candidates))
        # The budget binds when fewer components fit than are eligible.
        self.metrics["units_eligible"] += len(candidates)
        self.metrics["sessions_budget_bound"] += int(slots < len(candidates))
        self.metrics["exploration_picks"] += min(pending.reserved, len(chosen))
        self.metrics["utility_picks"] += max(0, len(chosen) - pending.reserved)

        body = units.encode_units(self.components,
                                  {c: self.state.gather(c) for c in chosen})
        predicted = units.units_cost_bytes(self.components, len(chosen))
        if len(body) != predicted:
            raise SessionError(
                f"unit cost model is wrong: predicted {predicted} bytes for "
                f"{len(chosen)} components, encoded {len(body)}")
        if len(body) + overhead > budget.payload_credit():
            raise SessionError(
                f"planned {len(body)} + {overhead} bytes against a credit of "
                f"{budget.payload_credit()}; the capacity arithmetic is wrong")
        pending.planned_bytes = len(body)
        return body

    # -- stage 4: received components, identities for the runtime, acknowledge -----

    def note_received(self, edge, components) -> None:
        """Record what arrived, as decoded from the partner's payload bytes."""
        self._observed[edge_key(*edge)] = tuple(sorted(int(c) for c in components))

    def received_identities(self, session):
        """The components this peer decoded from the partner's payload in this
        session, for the ledger."""
        return list(self._observed.get(edge_key(*session.endpoints), ()))

    def payload_identities(self, session):
        """The components in the payload just planned, for the ledger row.

        The runtime asks while the payload frame is debited, when the pending
        decision is still live and before ``acknowledge`` writes the planning
        log, so the ledger holds its own record of which components moved.
        """
        pending = self._pending.get(edge_key(*session.endpoints))
        if pending is None:
            return None
        return [int(c) for c in pending.selected]

    def planning_identities(self, session) -> dict:
        """The plan for this session, reported to the runtime at commit.

        Read from the pending decision: the executor asks before ``acknowledge``
        writes the plan to the planning log, and stores the answer in the
        session record. It includes this peer's materialized set and support,
        from which each session's eligible set ``Q_ij = M_i | M_j`` can be
        recomputed from the two endpoints' records.
        """
        edge = edge_key(*session.endpoints)
        pending = self._pending.get(edge)
        if pending is None:
            return {}
        return {
            # The canonical edge and this edge's session index, the same index
            # as the ``session`` field of the planning-log entry that
            # ``acknowledge`` writes for this session.
            "edge": [int(edge[0]), int(edge[1])],
            "session": int(self.memory.edge(*edge).sessions),
            "eligible": [int(c) for c in pending.eligible],
            "selected": [int(c) for c in pending.selected],
            "explored": [int(c) for c in pending.explored],
            "materialised": [int(c) for c in self.state.materialised],
            "support": [int(c) for c in self.state.support],
        }

    def acknowledge(self, session, committed: bool, reason: str = "") -> None:
        """Apply the pending decision for this session if it committed.

        The selector state (ages, cursor, session counter) changes on a commit
        and on nothing else, so a rolled-back session leaves it as it was, like
        the tensors. Both endpoints see the same ``committed`` flag, so they
        stay in step; a rule keyed on how far this peer got before a failure
        could diverge when the engine fails one endpoint and not the other.

        On a commit, every eligible component ages by one if at least one
        component could fit, the selected components reset to zero, the
        planning log gains an entry, the session counter advances and the
        round-robin cursor moves by the number of reserved slots. Ages
        therefore count sessions in which a component was eligible and one
        complete component fit, restricted to committed sessions.
        """
        edge = self._edge(session)
        pending = self._pending.pop(edge, None)      # transient, always cleared
        if pending is None:
            return

        if not committed:
            # Telemetry only. Nothing here feeds selection or the manifest
            # digest, which covers seed, ages, cursor and sessions.
            self.metrics["failed_sessions"] += 1
            return

        state = self.memory.edge(*edge)
        # The ages before and after the update are both logged: serving a
        # component resets its age to zero, so the final ages alone do not show
        # how long it waited.
        ages_before = {int(c): int(state.age_of(c)) for c in pending.eligible}
        cursor_before, session_index = int(state.cursor), int(state.sessions)
        if pending.could_fit:
            state.advance(pending.eligible)
        if pending.selected:
            state.reset(pending.selected)
            self.metrics["units_committed"] += len(pending.selected)
        self.metrics["sessions_committed"] += 1
        self.planning_log.setdefault(edge, []).append({
            "session": session_index,
            "eligible": len(pending.eligible),
            "eligible_components": [int(c) for c in pending.eligible],
            "affordable_slots": int(pending.slots),
            "selectable": min(int(pending.slots), len(pending.eligible)),
            "reserved_slots": int(pending.reserved),
            "selected": len(pending.selected),
            "selected_components": [int(c) for c in pending.selected],
            "explored_components": [int(c) for c in pending.explored],
            "could_fit": bool(pending.could_fit),
            "ages_before": ages_before,
            "ages_after": {int(c): int(state.age_of(c)) for c in pending.eligible},
            "cursor_before": cursor_before,
        })
        state.sessions += 1
        if pending.reserved:
            state.cursor = (state.cursor + pending.reserved) % max(1, len(pending.eligible))

    # -- helpers ------------------------------------------------------------------------

    def _utility(self, candidates, remote_view, edge) -> Dict[int, float]:
        """The per-candidate utility, before the mode turns it into a score."""
        if self.selector in selection.SCORE_FREE_SELECTORS:
            return {}

        if self.selector == selection.SELECTOR_EXACT:
            partner = self.oracle.get(self._partner_id(edge))
            if partner is None:
                raise SessionError(
                    "the exact-disagreement oracle has no record of the partner peer")
            # One orientation on both endpoints (the lower peer id first), so both compute
            # bit-identical scores and rank the candidates the same way.
            first, second = ((self.state, partner.state)
                             if int(self.peer_id) < int(self._partner_id(edge))
                             else (partner.state, self.state))
            out = {}
            for component in candidates:
                report = {}
                out[component] = scoring.exact_disagreement(
                    self.components, first.gather(component), second.gather(component), report)
                self.metrics["clamped_scores"] += bool(report.get("clamped"))
            return out

        if self.selector == selection.SELECTOR_SKETCH:
            remote_norms = remote_view.get("nm") or {}
            remote_sketch = remote_view.get("sk") or {}
            held = frozenset(remote_view["_materialised"])
            out = {}
            for component in candidates:
                block = self.state.gather(component)
                key = str(component)
                report = {}
                # Whether the partner holds the component is read from its
                # bitmap, not from a missing entry: consume_manifest has checked
                # that every declared component has one. A component the partner
                # does not hold is at its initial value, computed locally.
                if component in held:
                    partner_norms = remote_norms[key]
                    partner_sketch = [tuple(sides) for sides in remote_sketch[key]]
                else:
                    initial = self.components.initial_component(component)
                    partner_norms = scoring.component_norms(self.components, initial)
                    partner_sketch = scoring.component_sketch(
                        self.components, initial, self.sketch_dim)
                out[component] = scoring.sketch_disagreement(
                    self.components,
                    scoring.component_norms(self.components, block),
                    scoring.component_sketch(self.components, block, self.sketch_dim),
                    partner_norms, partner_sketch, report,
                )
                self.metrics["clamped_scores"] += bool(report.get("clamped"))
            return out

        # local_update, shuffled_score and survivor_only: each peer publishes its
        # own local-update magnitude and the two are folded symmetrically.
        local = {c: scoring.local_update_magnitude_for(
                    self.components, c, self.state.gather(c))
                 for c in candidates}
        # A component the partner holds has a published score (checked by
        # consume_manifest); one it does not hold is at its initial value and
        # contributes nothing.
        published = remote_view.get("sc") or {}
        held = frozenset(remote_view["_materialised"])
        remote = {c: float(published[str(c)]) for c in candidates if c in held}
        return selection.combine_scores(self.selector, candidates, local, remote)

    def _partner_id(self, edge: Tuple[int, int]) -> int:
        """The endpoint that is not me. The oracle reads that peer's state directly."""
        others = [p for p in edge if p != self.peer_id]
        if len(others) != 1:
            raise SessionError(
                f"peer {self.peer_id} is not exactly one endpoint of edge {edge}")
        return others[0]

    def _frame_overhead(self) -> int:
        """The framing bytes a payload frame adds to its body: the encoded size of
        an empty ``unit`` frame with this session's checksum setting (the frame
        header has a fixed width)."""
        return encoded_size(payload_frame("unit", self.peer_id, self.peer_id, 0, b"",
                                          checksum=self.checksum))

    def edge_of(self, session) -> Tuple[int, int]:
        """The unordered edge for a session; the peer's staging step uses it too."""
        return self._edge(session)

    def pending_for(self, edge: Tuple[int, int]):
        """The decision planning recorded for this edge, or None."""
        return self._pending.get(edge)

    def _edge(self, session) -> Tuple[int, int]:
        endpoints = tuple(getattr(session, "endpoints", ()) or ())
        if len(endpoints) != 2:
            raise SessionError(
                f"a pair session must have exactly two endpoints, got {endpoints!r}")
        return edge_key(*endpoints)

    # -- checkpoint -----------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {"memory": self.memory.state_dict(), "metrics": dict(self.metrics),
                "planning_log": {f"{a}-{b}": list(entries)
                                 for (a, b), entries in sorted(self.planning_log.items())}}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Validate the whole checkpoint (selector memory, metrics, planning
        log), then assign it. A rejected checkpoint leaves the policy unchanged.
        """
        if not isinstance(state, dict):
            raise SessionError(
                f"session policy checkpoint is {type(state).__name__}, not a mapping")

        seed, edges = SelectorMemory._parse(state.get("memory") or {})

        if "metrics" not in state:
            raise SessionError(
                "session policy checkpoint is missing 'metrics'; these counters are "
                "part of exact resume, and defaulting them to zero would make a "
                "resumed run report different totals from the uninterrupted one")
        stored = state["metrics"]
        if not isinstance(stored, dict):
            raise SessionError(
                f"session policy metrics is {type(stored).__name__}, not a mapping")
        unknown = sorted(set(stored) - set(self.metrics))
        if unknown:
            raise SessionError(
                f"session policy metrics carries unknown counters {unknown}; the "
                f"checkpoint was written by a different build")
        metrics = {}
        for key in self.metrics:
            if key not in stored:
                raise SessionError(
                    f"session policy metrics is missing {key!r}")
            try:
                value = exact_int(stored[key], f"session policy metric {key!r}")
            except ValueError as exc:
                raise SessionError(str(exc)) from exc
            if value < 0:
                raise SessionError(
                    f"session policy metric {key!r} is negative: {value}")
            metrics[key] = value

        # ``None``, like any other non-mapping, is rejected rather than read as an
        # empty history.
        if "planning_log" not in state:
            raise SessionError("session policy checkpoint has no planning_log")
        restored = _parse_planning_log(state["planning_log"], edges, metrics)

        # --- commit; the first mutation happens here ---------------------------
        self.memory.seed, self.memory._edges = seed, edges
        self.metrics.update(metrics)
        self.planning_log = restored


def _endpoint(text: str, where: str) -> int:
    """A peer id spelled canonically: no sign, no padding, no whitespace."""
    if not text.isdigit() or (len(text) > 1 and text.startswith("0")):
        raise SessionError(f"{where} is {text!r}, not a canonical peer id")
    return int(text)


def _parse_planning_log(raw, edges, metrics) -> Dict[Tuple[int, int], list]:
    """Validate a checkpointed planning log and return it keyed by edge.

    Checks every entry's fields and their consistency (counts against
    identities, selected within eligible, explored within selected and within
    the reserved slots, ages over exactly the eligible set), and that the log is
    complete: one entry per committed session, numbered from 0 on every edge of
    the restored selector memory, in agreement with the restored metrics.
    Changes nothing, so the caller can still reject the checkpoint.
    """
    if not isinstance(raw, dict):
        raise SessionError(
            f"session policy planning_log is {type(raw).__name__}, not a mapping")

    fields = ("session", "eligible", "affordable_slots", "selectable",
              "reserved_slots", "selected")
    identity_fields = ("eligible_components", "selected_components",
                       "explored_components")
    restored: Dict[Tuple[int, int], list] = {}
    committed = 0
    for raw_key, entries in raw.items():
        where = f"session policy planning_log[{raw_key!r}]"
        if not isinstance(raw_key, str):
            raise SessionError(f"{where} key is {type(raw_key).__name__}, not a string")
        left, sep, right = raw_key.partition("-")
        if not sep:
            raise SessionError(f"{where} key is not of the form '<a>-<b>'")
        endpoints = tuple(_endpoint(text, f"{where} {side} endpoint")
                          for text, side in ((left, "left"), (right, "right")))
        key = edge_key(*endpoints)
        if raw_key != f"{key[0]}-{key[1]}":
            # Canonical form only. '2-1' and '1-2' name one edge, so accepting
            # both lets two spellings collide into one entry, last write winning.
            raise SessionError(
                f"{where} key is not the canonical form '{key[0]}-{key[1]}'")
        if key in restored:
            raise SessionError(f"{where} names an edge already present in the log")
        if key not in edges:
            raise SessionError(
                f"{where} names an edge absent from the restored selector memory; "
                f"the log and the memory describe different runs")

        if not isinstance(entries, list):
            raise SessionError(
                f"{where} is {type(entries).__name__}, not a list of sessions")
        parsed, previous = [], None
        for index, entry in enumerate(entries):
            at = f"{where}[{index}]"
            if not isinstance(entry, dict):
                raise SessionError(f"{at} is {type(entry).__name__}, not a mapping")
            record = {}
            for field in fields:
                if field not in entry:
                    raise SessionError(f"{at} is missing {field!r}")
                try:
                    value = exact_int(entry[field], f"{at} {field!r}")
                except ValueError as exc:
                    raise SessionError(str(exc)) from exc
                if value < 0:
                    raise SessionError(f"{at} {field!r} is negative: {value}")
                record[field] = value
            if previous is not None and record["session"] <= previous:
                raise SessionError(
                    f"{at} session {record['session']} does not follow "
                    f"{previous}; committed sessions are strictly increasing")
            previous = record["session"]

            for field in identity_fields:
                if field not in entry:
                    raise SessionError(f"{at} is missing {field!r}")
                value = entry[field]
                if not isinstance(value, (list, tuple)):
                    raise SessionError(
                        f"{at} {field!r} is {type(value).__name__}, not a list")
                ids = []
                for item in value:
                    try:
                        ids.append(exact_int(item, f"{at} {field!r} component"))
                    except ValueError as exc:
                        raise SessionError(str(exc)) from exc
                if len(set(ids)) != len(ids):
                    raise SessionError(f"{at} {field!r} repeats a component")
                record[field] = ids
            for field in ("ages_before", "ages_after"):
                if field not in entry:
                    raise SessionError(f"{at} is missing {field!r}")
                value = entry[field]
                if not isinstance(value, dict):
                    raise SessionError(
                        f"{at} {field!r} is {type(value).__name__}, not a mapping")
                ages = {}
                for component, age in value.items():
                    try:
                        cid = exact_int(component, f"{at} {field!r} component")
                        parsed_age = exact_int(age, f"{at} {field!r} age")
                    except ValueError as exc:
                        raise SessionError(str(exc)) from exc
                    if parsed_age < 0:
                        raise SessionError(
                            f"{at} {field!r} age for component {cid} is negative")
                    if cid in ages:
                        raise SessionError(f"{at} {field!r} repeats component {cid}")
                    ages[cid] = parsed_age
                record[field] = ages
            if "could_fit" not in entry:
                raise SessionError(f"{at} is missing 'could_fit'")
            if not isinstance(entry["could_fit"], bool):
                raise SessionError(f"{at} 'could_fit' is not a boolean")
            record["could_fit"] = entry["could_fit"]
            if record["could_fit"] != (record["affordable_slots"] > 0):
                # plan_payload sets could_fit = slots > 0, so the two fields
                # cannot disagree in a log this code wrote.
                raise SessionError(
                    f"{at} records could_fit={record['could_fit']} against "
                    f"{record['affordable_slots']} affordable slot(s); the two "
                    f"cannot disagree")
            if not record["could_fit"] and record["selected"]:
                raise SessionError(
                    f"{at} shipped {record['selected']} component(s) in a session "
                    f"that reported nothing could fit")
            if "cursor_before" in entry:
                try:
                    record["cursor_before"] = exact_int(
                        entry["cursor_before"], f"{at} 'cursor_before'")
                except ValueError as exc:
                    raise SessionError(str(exc)) from exc
                if record["cursor_before"] < 0:
                    raise SessionError(f"{at} 'cursor_before' is negative")

            # Counts, identities and slots must describe one consistent session.
            if len(record["eligible_components"]) != record["eligible"]:
                raise SessionError(f"{at} eligible identities contradict the count")
            if len(record["selected_components"]) != record["selected"]:
                raise SessionError(f"{at} selected identities contradict the count")
            if record["selectable"] != min(record["affordable_slots"],
                                           record["eligible"]):
                raise SessionError(
                    f"{at} 'selectable' is not min(affordable_slots, eligible)")
            if record["selected"] > record["selectable"]:
                raise SessionError(f"{at} selected more components than selectable")
            if record["reserved_slots"] > record["selectable"]:
                raise SessionError(f"{at} reserved more slots than were selectable")
            if not set(record["selected_components"]) <= set(record["eligible_components"]):
                raise SessionError(f"{at} selected a component it deemed ineligible")
            if not set(record["explored_components"]) <= set(record["selected_components"]):
                raise SessionError(f"{at} explored a component it did not select")
            if len(record["explored_components"]) > record["reserved_slots"]:
                raise SessionError(f"{at} explored more components than it reserved")
            for field in ("ages_before", "ages_after"):
                if set(record[field]) != set(record["eligible_components"]):
                    raise SessionError(
                        f"{at} {field!r} does not cover exactly the eligible set")
            parsed.append(record)
        committed += len(parsed)
        restored[key] = parsed

    # -- completeness ---------------------------------------------------------
    # The log holds exactly one entry per committed session: its totals must
    # match the restored metrics and every edge's session counter, and each
    # edge's entries are numbered 0, 1, 2, ... without gaps.
    expected_sessions = int(metrics.get("sessions_committed", 0))
    if committed != expected_sessions:
        raise SessionError(
            f"session policy planning_log records {committed} committed "
            f"session(s) but the restored metrics report {expected_sessions}; "
            f"the history is incomplete")
    for key, state in edges.items():
        entries = restored.get(key, [])
        if len(entries) != int(state.sessions):
            raise SessionError(
                f"session policy planning_log has {len(entries)} entr(ies) for "
                f"edge {key[0]}-{key[1]} but its restored selector state reports "
                f"{state.sessions} committed session(s)")
        indices = [int(e["session"]) for e in entries]
        if indices != list(range(len(entries))):
            raise SessionError(
                f"session policy planning_log for edge {key[0]}-{key[1]} does not "
                f"run contiguously from session 0: {indices}")
    selected_total = sum(len(e["selected_components"])
                         for entries in restored.values() for e in entries)
    expected_units = int(metrics.get("units_committed", 0))
    if selected_total != expected_units:
        raise SessionError(
            f"session policy planning_log accounts for {selected_total} committed "
            f"component(s) but the restored metrics report {expected_units}")
    return restored
