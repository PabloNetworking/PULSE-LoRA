"""Per-edge eligible ages and the deterministic exploration state.

A component's age on an edge is the number of committed sessions on that
edge, since the component was last selected there, in which it was eligible
and at least one complete component fit. A session that cannot afford any
component ages nothing, a selected component's age resets to zero when the
session commits, and a failed or rolled-back session changes no age.

``plan_payload`` therefore does not touch ages: the runtime plans on both
endpoints even when the session then ends without a payload
(``core/pair_session.py`` plans on both peers with no early exit). Planning
records a pending decision, and ``PulseSessionPolicy.acknowledge``, which the
runtime calls once per endpoint on every path that ends a session, applies it
if the session committed.

Both endpoints must hold identical selector state, or they can select different
component sets. They see the same eligible set and the same outcome, and each
manifest carries :meth:`SelectorMemory.selector_digest`, so a divergence is
rejected instead of producing a mismatched exchange. The digest covers the
seed and the session counter as well as the ages and the cursor, because
``draw`` depends on all four.

Pseudo-random draws are counter-based: a hash of (seed, purpose, edge, session
counter, component), not a stateful RNG. A draw does not depend on how many
draws came before it, so the two endpoints stay in step without consuming a
stream in lockstep, a resumed run draws exactly what the uninterrupted one
would, and no other random stream is affected.
"""
import hashlib
import struct
from typing import Dict, Iterable, Mapping, Sequence, Tuple


#: Version of the selector-memory checkpoint format. ``load_state_dict``
#: requires the field and accepts only this version; a missing version is not
#: read as the current one.
SELECTOR_CHECKPOINT_VERSION = 1


def exact_int(value, where: str) -> int:
    """``value`` as an int, or a ValueError naming ``where``.

    Accepts ints and integral finite floats. Rejects booleans (``bool`` is an
    ``int`` subclass), non-integral or non-finite floats and anything else,
    where ``int()`` would truncate or coerce them into a different counter.
    """
    if isinstance(value, bool):
        raise ValueError(f"{where} is {value!r}; a boolean is not a counter")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{where} is {value!r}, not a finite integer")
        if not value.is_integer():
            raise ValueError(
                f"{where} is {value!r}, which is not an exact integer; refusing "
                f"to truncate it")
        return int(value)
    raise ValueError(f"{where} is {value!r}, not an integer")


def require(mapping, field: str, where: str):
    """``mapping[field]``; a missing field raises instead of defaulting to zero."""
    if field not in mapping:
        raise ValueError(
            f"{where} is missing {field!r}; the selector checkpoint format "
            f"requires it")
    return mapping[field]


def edge_key(a: int, b: int) -> Tuple[int, int]:
    """Unordered: an edge is the same edge whichever endpoint names it."""
    return (int(a), int(b)) if int(a) <= int(b) else (int(b), int(a))


class EdgeSelectorState:
    """Component ages, the round-robin cursor and the committed-session counter
    of one unordered edge."""

    __slots__ = ("ages", "cursor", "sessions")

    def __init__(self, ages: Mapping[int, int] = None, cursor: int = 0,
                 sessions: int = 0):
        self.ages: Dict[int, int] = {int(k): int(v) for k, v in (ages or {}).items()}
        self.cursor = int(cursor)
        self.sessions = int(sessions)

    def age_of(self, component: int) -> int:
        return self.ages.get(int(component), 0)

    def advance(self, eligible: Iterable[int]) -> None:
        for component in eligible:
            component = int(component)
            self.ages[component] = self.ages.get(component, 0) + 1

    def reset(self, components: Iterable[int]) -> None:
        for component in components:
            self.ages[int(component)] = 0

    def state_dict(self) -> dict:
        return {"ages": dict(sorted(self.ages.items())),
                "cursor": self.cursor, "sessions": self.sessions}

    def __repr__(self) -> str:      # pragma: no cover - debugging aid
        return (f"EdgeSelectorState(ages={self.state_dict()['ages']}, "
                f"cursor={self.cursor}, sessions={self.sessions})")


class SelectorMemory:
    """A peer's per-edge selector state and the seed of its draws, checkpointed
    with the peer."""

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self._edges: Dict[Tuple[int, int], EdgeSelectorState] = {}

    # -- access ---------------------------------------------------------------

    def edge(self, a: int, b: int) -> EdgeSelectorState:
        """The mutable state for an edge, created on first use. Write path only."""
        key = edge_key(a, b)
        if key not in self._edges:
            self._edges[key] = EdgeSelectorState()
        return self._edges[key]

    def view(self, a: int, b: int) -> EdgeSelectorState:
        """The state of an edge, or an empty one; never creates an entry.

        ``selector_digest`` is called from both manifest hooks, and
        ``consume_manifest`` must not change state: the runtime stops at the
        first rejection, so creating an entry there could leave the two
        endpoints with different checkpoints. Entries are created only through
        :meth:`edge`, when a session commits.
        """
        return self._edges.get(edge_key(a, b)) or EdgeSelectorState()

    def known_edges(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(sorted(self._edges))

    def max_age(self) -> int:
        return max((max(s.ages.values(), default=0) for s in self._edges.values()),
                   default=0)

    # -- agreement -------------------------------------------------------------

    def selector_digest(self, a: int, b: int) -> str:
        """Digest of everything on this edge that a selection depends on.

        Covers the seed, the ages, the cursor and the session counter. The seed
        and the counter matter because ``draw``, which drives the ``random`` and
        ``shuffled_score`` selectors, is keyed on ``(seed, purpose, edge,
        sessions, component)``.

        Twelve hex characters: enough to detect an accidental divergence, and
        cheap to carry in every manifest, whose bytes are charged before the
        payload credit is computed.
        """
        state = self.view(a, b)
        material = (f"{self.seed}|{edge_key(a, b)}|{sorted(state.ages.items())}|"
                    f"{state.cursor}|{state.sessions}")
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]

    # -- deterministic pseudo-randomness ----------------------------------------

    def draw(self, a: int, b: int, component: int, purpose: str) -> float:
        """A reproducible value in [0, 1) keyed on checkpointed state alone.

        A hash of ``(seed, purpose, edge, sessions, component)`` rather than a
        stream, so it does not depend on how many times it has been called and
        survives a resume exactly.
        """
        state = self.view(a, b)
        material = (f"{self.seed}|{purpose}|{edge_key(a, b)}|{state.sessions}|"
                    f"{int(component)}")
        digest = hashlib.blake2b(material.encode("utf-8"), digest_size=8).digest()
        return struct.unpack("!Q", digest)[0] / float(1 << 64)

    # -- checkpoint --------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "version": SELECTOR_CHECKPOINT_VERSION,
            "seed": self.seed,
            # JSON-safe and order-stable: an edge is a two-int key, so persist it
            # as a string rather than relying on tuple keys surviving a round trip.
            "edges": {f"{a}-{b}": s.state_dict()
                      for (a, b), s in sorted(self._edges.items())},
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Validate the whole checkpoint, then assign it.

        A rejected checkpoint leaves the memory unchanged, whoever the caller
        is, so a peer never ends up with the seed of one checkpoint and the
        edges of another.
        """
        parsed = self._parse(state)
        self.seed, self._edges = parsed          # the first mutation is here

    @staticmethod
    def _parse(state: Mapping[str, object]):
        """Validate a checkpoint and return ``(seed, edges)``, changing nothing."""
        if not isinstance(state, dict):
            raise ValueError(
                f"selector memory checkpoint is {type(state).__name__}, not a mapping")

        version = exact_int(
            require(state, "version", "selector memory checkpoint"),
            "selector memory checkpoint version")
        if version != SELECTOR_CHECKPOINT_VERSION:
            raise ValueError(
                f"selector memory checkpoint is version {version}, and this code "
                f"reads version {SELECTOR_CHECKPOINT_VERSION}; another version needs "
                f"an explicit migration")

        seed = exact_int(require(state, "seed", "selector memory checkpoint"),
                         "selector memory seed")

        raw_edges = require(state, "edges", "selector memory checkpoint")
        if not isinstance(raw_edges, dict):
            raise ValueError(
                f"selector memory edges is {type(raw_edges).__name__}, not a mapping")

        edges: Dict[Tuple[int, int], EdgeSelectorState] = {}
        for raw_key, raw in raw_edges.items():
            try:
                left, sep, right = str(raw_key).partition("-")
                if not sep:
                    raise ValueError(raw_key)
                key = edge_key(int(left), int(right))
            except ValueError as exc:
                raise ValueError(
                    f"selector memory has an unreadable edge key {raw_key!r}") from exc
            if str(raw_key) != f"{key[0]}-{key[1]}":
                raise ValueError(
                    f"selector memory edge key {raw_key!r} is not the canonical "
                    f"form {key[0]}-{key[1]!r}; two spellings of one edge would "
                    f"silently overwrite each other")
            if key in edges:
                raise ValueError(
                    f"selector memory has colliding identifiers for edge {key}")
            if not isinstance(raw, dict):
                raise ValueError(
                    f"selector memory edge {key} is {type(raw).__name__}, not a mapping")

            where = f"selector memory edge {key}"
            ages = require(raw, "ages", where)
            if not isinstance(ages, dict):
                raise ValueError(
                    f"{where} ages is {type(ages).__name__}, not a mapping")
            parsed_ages: Dict[int, int] = {}
            for component, value in ages.items():
                try:
                    component_id = int(component)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{where} has a non-numeric component key "
                        f"{component!r}") from exc
                if str(component) != str(component_id):
                    raise ValueError(
                        f"{where} age key {component!r} is not the canonical form "
                        f"{str(component_id)!r}")
                if component_id in parsed_ages:
                    raise ValueError(
                        f"{where} has colliding identifiers for component "
                        f"{component_id}")
                age = exact_int(value, f"{where} age for component {component_id}")
                if age < 0:
                    raise ValueError(
                        f"{where} has a negative age for component "
                        f"{component_id}: {age}")
                parsed_ages[component_id] = age

            counters = {}
            for field in ("cursor", "sessions"):
                counters[field] = exact_int(require(raw, field, where),
                                            f"{where} {field}")
                if counters[field] < 0:
                    raise ValueError(f"{where} has a negative {field}")

            edges[key] = EdgeSelectorState(ages=parsed_ages, **counters)
        return seed, edges
