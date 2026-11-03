"""PULSE-LoRA peer and factory.

The peer trains its active support through the shared persistent trainer and
prunes it with its local controller. It answers the pair-session hooks with a
byte-budgeted component selection, whose policy lives in ``session.py``, and
applies an agreed exchange transactionally through ``mixing.py``: ``stage``
validates and snapshots, ``commit`` writes, ``rollback`` restores exactly, and
``discard`` releases. This class wires both to the runtime's hook names.
"""
import logging
from typing import Mapping, Optional

import torch

from federatedscope.contrib.decentralized.core.interfaces import PeerContext
from federatedscope.contrib.decentralized.lora.adapter_bridge import AdapterBridge
from federatedscope.contrib.decentralized.lora.canonical_init import (
    EXTRAS_KEY as COMMON_CONSTRUCTION_KEY,
)
from federatedscope.contrib.decentralized.lora.auxiliary_state import (
    apply_policy,
    communicates_auxiliary,
)
from federatedscope.contrib.decentralized.lora.optimizer_state import clear_moment_slices
from federatedscope.contrib.decentralized.lora.schema import build_schema
from federatedscope.contrib.decentralized.methods.pulse_lora import config as pulse_config
from federatedscope.contrib.decentralized.methods.pulse_lora.components import (
    build_component_schema,
)
from federatedscope.contrib.decentralized.methods.pulse_lora.pruning import build_controller
from federatedscope.contrib.decentralized.methods.pulse_lora.selection import ORACLE_SELECTORS
from federatedscope.contrib.decentralized.methods.pulse_lora.mixing import (
    MixEngine, MixError,
)
from federatedscope.contrib.decentralized.methods.pulse_lora.session import (
    OracleRegistry, PulseSessionPolicy,
)
from federatedscope.contrib.decentralized.methods.pulse_lora.state import RELAY_CARRIED, PulsePeerState
from federatedscope.contrib.decentralized.methods.pulse_lora.units import (
    decode_units, unit_cost_bytes,
)

logger = logging.getLogger(__name__)


class PulseLoRAPeer:
    """One PULSE learner: heterogeneous support plus a relay plane."""

    def __init__(self, context: PeerContext, r_max: int, seed: int,
                 controller: str, controller_kwargs: Mapping[str, object],
                 auxiliary_policy: str, interaction: str,
                 initial_support: Optional[list] = None,
                 scale: Optional[float] = None,
                 initial_rank: Optional[int] = None,
                 selector: str = "local_update",
                 exploration_fraction: float = 0.5,
                 sketch_dim: int = 16,
                 checksum: bool = True,
                 oracle: Optional[OracleRegistry] = None,
                 gamma: float = 0.5,
                 relay_content: str = RELAY_CARRIED):
        self.peer_id = context.peer_id
        self.context = context

        self.model = context.extras.get("model") or self._build_model(context)
        self.lora_schema = build_schema(self.model)
        self.bridge = AdapterBridge(self.model, self.lora_schema)

        # An untouched component's value is the run's canonical value, the one the active
        # rows are constructed with (lora/canonical_init.py).
        construction = context.extras.get(COMMON_CONSTRUCTION_KEY)
        if construction is None:
            raise ValueError(f"peer {self.peer_id}: PULSE-LoRA needs the run's common "
                             f"construction in context.extras[{COMMON_CONSTRUCTION_KEY!r}]")
        init_rule = construction.bind(self.lora_schema)
        self.components = build_component_schema(self.lora_schema, r_max=r_max, seed=seed,
                                                 scale=scale, init_rule=init_rule)
        self.state = PulsePeerState(self.bridge, self.components,
                                    relay_content=relay_content)
        self._apply_initial_support(initial_rank, initial_support)
        self.controller = build_controller(controller, **dict(controller_kwargs))

        self.includes_auxiliary = communicates_auxiliary(auxiliary_policy, interaction)
        apply_policy(self.model, self.lora_schema.auxiliary_keys, auxiliary_policy)

        self.trainer = context.extras.get("trainer")
        if self.trainer is None and context.data is not None:
            self.trainer = self._build_trainer(context)

        self._version = 0
        self.episodes = 0
        self._initial_digest = None

        # -- session policy and mixing ---------------------------------------------
        self.policy = PulseSessionPolicy(
            peer_id=self.peer_id,
            components=self.components,
            state=self.state,
            selector=selector,
            exploration_fraction=exploration_fraction,
            sketch_dim=sketch_dim,
            includes_auxiliary=self.includes_auxiliary,
            checksum=checksum,
            oracle=oracle,
            # The selector's draws hash the shared component seed with the edge,
            # the session counter and the component, so both endpoints of an edge
            # draw the same values and no other random stream is consumed.
            seed=seed,
            version_source=self.state_version,
        )
        # The factory passes decentralized.comparison.gamma, the coefficient Dec-LoRA
        # also reads, so methods compared on one trace mix with the same strength.
        self.mixer = MixEngine(
            state=self.state, bridge=self.bridge, components_schema=self.components,
            gamma=gamma,
            optimizer_source=lambda: (
                getattr(self.trainer, "optimizer", None) if self.trainer else None),
        )
        if oracle is not None:
            oracle.register(self.peer_id, self)

    def _apply_initial_support(self, initial_rank, initial_support) -> None:
        """Narrow the built adapter to the configured initial support.

        The support is ``fixed_support`` when given, else the first
        ``initial_rank`` components. Narrowing compacts the untouched adapter; a
        support wider than the built adapter is a configuration error, since
        supports never grow.
        """
        built = len(self.state.support)

        if initial_support:
            wanted = [int(c) for c in initial_support]
            if initial_rank is not None and len(wanted) != int(initial_rank):
                raise ValueError(
                    f"peer {self.peer_id}: fixed_support has {len(wanted)} components "
                    f"but initial_rank is {initial_rank}; they must agree")
        elif initial_rank is not None:
            wanted = list(range(int(initial_rank)))
        else:
            return

        if len(wanted) > built:
            raise ValueError(
                f"peer {self.peer_id}: an initial support of {len(wanted)} components "
                f"was configured, but the adapter was built at rank {built}. PULSE "
                f"never grows a support - build the adapter at least that wide "
                f"(glue.adapter.max_rank) or lower initial_rank."
            )

        self.state.restrict_initial_support(wanted)

        if len(self.state.support) != len(wanted) or \
                self.bridge.schema.rank != len(wanted):
            raise ValueError(
                f"peer {self.peer_id}: resolved support {wanted} was not applied; "
                f"the adapter holds rank {self.bridge.schema.rank}")

    # -- construction ---------------------------------------------------------

    def _build_model(self, context: PeerContext):
        from federatedscope.core.auxiliaries.model_builder import get_model

        return get_model(context.config, local_data=context.data)

    def _build_trainer(self, context: PeerContext):
        from federatedscope.core.auxiliaries.trainer_builder import get_trainer
        from federatedscope.core.monitors.monitor import Monitor

        import federatedscope.contrib.trainer.glue_decentralized_trainer  # noqa: F401

        return get_trainer(
            model=self.model, data=context.data, device=context.device,
            config=context.config, monitor=Monitor(context.config, monitored_object=self),
        )

    # -- PeerAlgorithm -----------------------------------------------------------

    def local_episode(self, steps: int) -> Mapping[str, object]:
        """Train the active support, then let the controller retire components.

        The controller records its low set before training, so it can tell which
        components' scores decreased, and prunes after.
        """
        self.controller.before_episode(self.state)

        if self.trainer is None:
            metrics = {"optimizer_steps": 0, "requested_steps": steps,
                       "skipped_batches": 0, "trained": False}
        else:
            steps_before = self.trainer.optimizer_steps
            skipped_before = self.trainer.skipped_batches
            _, _, trained = self.trainer.train()
            performed = self.trainer.optimizer_steps - steps_before
            skipped = self.trainer.skipped_batches - skipped_before
            if performed + skipped != steps:
                raise RuntimeError(
                    f"peer {self.peer_id}: performed {performed} optimizer steps and "
                    f"skipped {skipped} batches, which does not account for the {steps} "
                    f"requested")
            metrics = dict(trained or {}, optimizer_steps=performed,
                           requested_steps=steps, skipped_batches=skipped, trained=True)

        doomed = self.controller.select_for_pruning(self.state)
        if doomed:
            optimizer = getattr(self.trainer, "optimizer", None) if self.trainer else None
            self.state.prune(doomed, optimizer=optimizer)
            self._version += 1

        self.episodes += 1
        return dict(metrics, pruned=list(doomed), active_rank=len(self.state.support))

    def common_construction_targets(self):
        """``common_canonical_v1``: the bridge and the component ids of its compact slots."""
        return self.bridge, tuple(self.state.support)

    def evaluate(self, split: str = "val") -> Mapping[str, object]:
        if self.trainer is None:
            return self.support_fingerprint()
        metrics = dict(self.trainer.evaluate(target_data_split_name=split) or {})
        metrics.update(self.support_fingerprint())
        return metrics

    def support_fingerprint(self) -> Mapping[str, float]:
        state = self.bridge.read(self.lora_schema.lora_keys())
        flat = torch.cat([t.reshape(-1).to(torch.float64) for t in state.values()])
        return {"adapter_mean": float(flat.mean()), "adapter_norm": float(flat.norm())}

    def state_version(self) -> int:
        return self._version

    def state_dict(self) -> Mapping[str, object]:
        state = {
            "version": self._version,
            "episodes": self.episodes,
            "adapter": self.bridge.read(),
            "pulse_state": self.state.state_dict(),
            "session_policy": self.policy.state_dict(),
            "structural_hash": self.lora_schema.structural_hash,
        }
        if self.trainer is not None:
            state["trainer"] = self.trainer.trainer_state()
        return state

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Validate the entire checkpoint, then apply it. Never the other way round.

        The component plan and the adapter mapping are validated together before
        either is applied, because applying the plan can compact the model and
        migrate optimizer state. The adapter is therefore checked against the rank
        of the planned support, derived without resizing, and every check below
        runs before the first mutation.
        """
        if state.get("structural_hash") != self.lora_schema.structural_hash:
            raise ValueError("checkpoint structure does not match this peer's adapter")

        # --- preflight, mutating nothing --------------------------------------
        plan = self.state.plan_restore(state["pulse_state"])
        target_rank = len(plan["support"])
        self.bridge.validate_state(
            state["adapter"], self.bridge.adapter_contract(rank=target_rank))

        version = self._validate_counter(state, "version")
        episodes = self._validate_counter(state, "episodes")
        record = self._validate_trainer_record(state.get("trainer"))
        policy_state = self._validate_session_policy(state.get("session_policy"))

        # --- apply; the first mutation happens here ----------------------------
        optimizer = getattr(self.trainer, "optimizer", None) if self.trainer else None
        self.state.apply_restore(plan, optimizer=optimizer)
        self.bridge.write(state["adapter"],
                          expected_keys=self.bridge.expected_adapter_keys())
        self.state.enforce_scale()

        if self.trainer is not None and record is not None:
            loader = getattr(self.trainer, "load_trainer_state", None)
            if callable(loader):
                loader(record)

        if policy_state is not None:
            self.policy.load_state_dict(policy_state)
        self._version = version
        self.episodes = episodes

    def _validate_session_policy(self, policy_state):
        """Parse the selector state into a throwaway, so a bad one mutates nothing.

        Loading it into the live policy would replace the per-edge ages, and a peer
        that rejected a checkpoint half way through would carry a partly restored
        selector into the next session. It would then select differently from its
        partners, and the selector digest in the manifest would reject its sessions.
        """
        # `state_dict` always writes this section, so a checkpoint without it is
        # malformed.
        if policy_state is None:
            raise ValueError(
                "checkpoint is missing the 'session_policy' section; the "
                "selector history cannot be restored, and skipping it would "
                "silently resume with a history that never happened")
        probe = PulseSessionPolicy(
            peer_id=self.peer_id, components=self.components, state=self.state,
            selector=self.policy.selector,
            exploration_fraction=self.policy.exploration_fraction,
            sketch_dim=self.policy.sketch_dim,
            includes_auxiliary=self.policy.includes_auxiliary,
            checksum=self.policy.checksum, oracle=self.policy.oracle,
            seed=self.policy.memory.seed, version_source=self.state_version,
        )
        probe.load_state_dict(policy_state)
        return policy_state

    @staticmethod
    def _validate_counter(state: Mapping[str, object], field: str) -> int:
        if field not in state:
            raise ValueError(f"checkpoint is missing the {field!r} field")
        try:
            value = int(state[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"checkpoint {field!r} is {state[field]!r}, not an integer") from exc
        if value < 0:
            raise ValueError(f"checkpoint {field!r} is negative: {value}")
        return value

    @staticmethod
    def _validate_trainer_record(record):
        """Check the trainer record's shape before anything is applied."""
        if record is None:
            return None
        if not isinstance(record, dict):
            raise ValueError(
                f"checkpoint trainer record is {type(record).__name__}, not a mapping")
        for field in ("optimizer_steps", "skipped_batches", "local_episodes"):
            if field not in record:
                raise ValueError(f"checkpoint trainer record is missing {field!r}")
            try:
                if int(record[field]) < 0:
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"checkpoint trainer record has an invalid {field!r}: "
                    f"{record[field]!r}") from exc
        return record

    # -- optimizer reaction (a method policy, not a trainer behaviour) -------------

    def clear_moments_for(self, components) -> int:
        """Zero Adam moments for exactly these components' own slices.

        A canonical component owns one row of each A and one column of each B, so
        clearing whole parameters would reset momentum for every other component
        sharing the tensor. Returns the number of buffer slices zeroed.

        Inactive relay components have no optimizer state at all - they never
        receive gradients - so they are skipped rather than treated as an error.
        """
        optimizer = getattr(self.trainer, "optimizer", None) if self.trainer else None
        if optimizer is None:
            return 0

        slices = []
        for component in components:
            if not self.state.is_active(component):
                continue
            position = self.state.compact_index(component)
            for spec in self.bridge.schema.modules:
                slices.append((self.bridge._params[spec.a_key], 0, position))
                slices.append((self.bridge._params[spec.b_key], 1, position))
        if not slices:
            return 0
        return clear_moment_slices(optimizer, slices)

    # -- PairwiseAlgorithm -------------------------------------------------------------
    #
    # Thin by design: the runtime's hook names are fixed and positional, and all
    # of the policy lives in session.py where it can be tested without a runtime.

    def build_manifest(self, session) -> bytes:
        return self.policy.build_manifest(session)

    def consume_manifest(self, remote_manifest: bytes, session):
        return self.policy.consume_manifest(remote_manifest, session)

    def plan_payload(self, session, budget, remote_view) -> Optional[bytes]:
        return self.policy.plan_payload(session, budget, remote_view)

    def acknowledge(self, session, committed: bool, reason: str = "") -> None:
        self.policy.acknowledge(session, committed, reason)

    def received_identities(self, session):
        """The components this peer decoded from the partner's payload.

        The executor records them in the ledger beside the sender's
        ``payload_identities`` for the same frame.
        """
        return self.policy.received_identities(session)

    def identity_bound(self) -> int:
        """The most component ids one identity list can carry: ``r_max``.

        The executor derives the widest id from it, and the wire-cost model uses
        both to bound the identity lists it cannot size exactly."""
        return int(self.components.r_max)

    def payload_identities(self, session):
        """What this peer's outgoing payload carries, for the ledger row.

        Called by the executor when it prices the payload on the wire and when it
        debits the payload frame, so the ledger, which the method never writes,
        records which components moved and not only how many bytes did.
        """
        return self.policy.payload_identities(session)

    def planning_identities(self, session) -> dict:
        """What this peer planned for ``session``, for the runtime to record.

        Called by the executor on a committed session, before ``acknowledge``, so
        the runtime's event log holds its own copy of the identities rather than
        only the policy's later report.
        """
        return self.policy.planning_identities(session)

    # -- transactional mixing ----------------------------------------------------------

    def stage(self, incoming: bytes, session) -> int:
        """Validate the partner's payload completely, then stage a reversible mix.

        Every rejection here happens before a single tensor is written. The
        engine turns a raised exception into a failed transaction and unwinds
        whatever else was staged, and the bytes already sent stay charged - a
        rollback cannot un-send.
        """
        edge = self.policy.edge_of(session)
        pending = self.policy.pending_for(edge)
        if pending is None or not pending.selected:
            raise MixError(
                "a payload arrived for a session this peer planned nothing for; "
                "the two endpoints did not agree on an exchange")

        remote_blocks = decode_units(self.components, incoming)
        # Recorded from the decoded payload, not from any plan, so the receiver's
        # record is independent of the sender's account of the same frame.
        self.policy.note_received(edge, sorted(remote_blocks))

        agreed = frozenset(pending.selected)
        received = frozenset(remote_blocks)
        if received != agreed:
            raise MixError(
                f"the payload carries components {sorted(received)} but this session "
                f"agreed on {sorted(agreed)}. Both endpoints apply the same set or "
                f"neither does; a one-sided set would leave the pair inconsistent.")

        if len(received) > pending.slots:
            raise MixError(
                f"the payload carries {len(received)} components but only "
                f"{pending.slots} were affordable in this session")

        if pending.version != self._version:
            raise MixError(
                f"this peer's state moved from version {pending.version} to "
                f"{self._version} between planning and staging; the mix was computed "
                f"against a state that no longer exists")

        return self.mixer.stage(pending.selected, remote_blocks, self._version)

    def commit(self, token: int) -> None:
        """Apply the mix. The snapshot is kept - the partner can still fail."""
        self.mixer.commit(token)
        self._version += 1

    def rollback(self, token: int) -> None:
        was_committed = token in self.mixer.open_tokens and self.mixer.was_committed(token)
        self.mixer.rollback(token)
        if was_committed:
            self._version -= 1

    def discard(self, token: int) -> None:
        self.mixer.discard(token)


    def initial_state_digest(self) -> str:
        """Digest of the effective update this peer started from.

        Hashes the live scale and ``scale * B @ A`` of every injected module.
        Captured once, on the first call: the runner calls it right after building
        its peers, before the first frame, and the run manifest records it. It is
        not taken in ``__init__`` because the common construction is applied to
        the peer after it is built (``simulation.scenario.construct_peer``).
        """
        if self._initial_digest is None:
            self._initial_digest = self._capture_initial_digest()
        return self._initial_digest


    def _capture_initial_digest(self) -> str:
        import hashlib
        parts = []
        scale = self.bridge.live_scale()
        for spec in self.bridge.schema.modules:
            state = self.bridge.read([spec.a_key, spec.b_key])
            update = scale * (state[spec.b_key].detach().to(torch.float64).cpu()
                              @ state[spec.a_key].detach().to(torch.float64).cpu())
            parts.append(f"{spec.module}:{update.numpy().tobytes().hex()}")
        material = f"{scale}|" + "|".join(parts)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]

    def value_precision(self) -> str:
        """The dtype this peer exchanges component values at.

        Recorded in the run manifest: precision changes both the bytes per
        component and the round-off, so runs at different precision are not
        directly comparable.
        """
        return self.lora_schema.dtype

    # -- MethodMetricProvider ---------------------------------------------------------

    def method_metrics(self) -> Mapping[str, object]:
        metrics = dict(self.state.metrics())
        metrics.update(self.controller.metrics())
        metrics.update({
            "pulse_r_max": self.components.r_max,
            "pulse_component_cost_bytes": self.components.component_cost_bytes(),
            "pulse_unit_bytes": unit_cost_bytes(self.components),
            "pulse_identity": self.components.identity_digest()[:16],
            "pulse_episodes": self.episodes,
            "pulse_selector": self.policy.selector,
            "pulse_selector_is_oracle": self.policy.selector in ORACLE_SELECTORS,
            "pulse_exploration_fraction": self.policy.exploration_fraction,
            "pulse_max_eligible_age": self.policy.memory.max_age(),
            "pulse_known_edges": len(self.policy.memory.known_edges()),
            # Fraction of P that is materialised (active or in relay storage).
            "pulse_materialisation_coverage": (
                len(self.state.materialised) / float(self.components.r_max)),
            # Relay components whose value differs from the canonical initial one.
            "pulse_relay_informative": len(self.state.informative_relay_components()),
        })
        metrics.update({f"pulse_{k}": v for k, v in self.policy.metrics.items()})
        return metrics


class PulseLoRAFactory:
    """Constructed with cfg, like every other decentralized method factory.

    Run-level values are read once from the global config; everything else is
    resolved per peer from that peer's own resolved config, so a `client_<id>`
    overlay can give peers different supports, ranks and controller settings.
    """

    name = "pulse_lora"

    def __init__(self, cfg):
        settings = cfg.decentralized.methods.pulse_lora
        self.cfg = cfg
        self.interaction_form = pulse_config.INTERACTION
        # run-level invariants
        self.r_max = int(settings.r_max)
        self.seed = int(settings.component_seed)
        self.scale = float(settings.scale) or None
        self.auxiliary_policy = cfg.decentralized.auxiliary_state.policy
        # Run-level too: both endpoints must rank candidates identically.
        self.selector = str(settings.selector)
        self.exploration_fraction = float(settings.exploration_fraction)
        self.sketch_dim = int(settings.sketch_dim)
        # Run-level: with one endpoint scrubbing and the other carrying, the two
        # peers would keep different content for the same exchange.
        self.relay_content = str(settings.relay_content)
        self.checksum = bool(cfg.decentralized.transport.checksum)
        # The mixing coefficient shared with the other methods on the trace; PULSE
        # has no gamma key of its own.
        self.gamma = float(cfg.decentralized.comparison.gamma)
        # The registry lets the exact_disagreement oracle read its partner's tensors,
        # so it exists only when an oracle selector is configured.
        self.oracle = OracleRegistry() if self.selector in ORACLE_SELECTORS else None

    @staticmethod
    def validate_config(cfg) -> None:
        pulse_config.validate(cfg)

    def _peer_settings(self, context: PeerContext):
        """Resolve this peer's settings, refusing any change to a run-level setting."""
        config = context.config if context.config is not None else self.cfg
        settings = config.decentralized.methods.pulse_lora

        for name, run_level in (("r_max", self.r_max),
                                ("component_seed", self.seed),
                                ("scale", self.scale),
                                ("selector", self.selector),
                                ("exploration_fraction", self.exploration_fraction),
                                ("sketch_dim", self.sketch_dim),
                                ("relay_content", self.relay_content)):
            local = getattr(settings, name)
            if name == "scale":
                local = float(local) or None
            elif name in ("selector", "relay_content"):
                local = str(local)
            elif name == "exploration_fraction":
                local = float(local)
            else:
                local = int(local)
            if local != run_level:
                raise ValueError(
                    f"peer {context.peer_id}: decentralized.methods.pulse_lora.{name}="
                    f"{local!r} disagrees with the run-level value {run_level!r}. "
                    f"{name} is a run-level setting (component namespace, selection "
                    f"rule or relay content), so every peer must agree on it; vary "
                    f"initial_rank or fixed_support instead."
                )

        # Validate the peer's resolved configuration too, not just the run-level
        # one: an overlay can set rank_min, prune_gamma, controller or
        # fixed_support to an invalid value, which must fail while peers are built
        # rather than during the run.
        try:
            pulse_config.validate(config)
        except ValueError as exc:
            raise ValueError(f"peer {context.peer_id}: {exc}") from exc

        return settings

    def create_peer(self, context: PeerContext) -> PulseLoRAPeer:
        settings = self._peer_settings(context)
        fixed_support = [int(c) for c in settings.fixed_support] or None
        controller = settings.controller
        controller_kwargs = {"gamma": float(settings.prune_gamma),
                             "rank_min": int(settings.rank_min)}
        return PulseLoRAPeer(
            context=context,
            r_max=self.r_max,
            seed=self.seed,
            controller=controller,
            controller_kwargs=controller_kwargs,
            auxiliary_policy=self.auxiliary_policy,
            interaction=self.interaction_form,
            scale=self.scale,
            initial_rank=int(settings.initial_rank),
            initial_support=fixed_support,
            selector=self.selector,
            exploration_fraction=self.exploration_fraction,
            sketch_dim=self.sketch_dim,
            checksum=self.checksum,
            oracle=self.oracle,
            gamma=self.gamma,
            relay_content=self.relay_content,
        )


from federatedscope.contrib.decentralized.registry import register_decentralized_method

register_decentralized_method("pulse_lora", PulseLoRAFactory)
