"""TAD-LoRA peer and factory.

Built on the shared runtime and LoRA layer. TAD-LoRA adds two policies and the
state they need:

* an alternating local update: one factor group trains at a time, because
  ``B @ A`` is convex in each factor with the other fixed and jointly non-convex;
* a topology-aware switching interval: hold a factor longer when recent
  connectivity is poor, so the alternation does not outrun the communication
  that has to carry its result.

Mixing is joint: one pre-round snapshot barrier, both factors weighted by the
same ``W_t``, whether that ``W_t`` is the paper's fixed matrix or the one the
runner compiles from each frame of the contact trace. Alternation is a local
policy and does not reach into mixing: a peer that mixed only the group it
happened to be training would make its state depend on its own phase.

This module imports no other method package.
"""
import logging
from typing import Mapping, Optional

import torch

from federatedscope.contrib.decentralized.core.interfaces import PeerContext
from federatedscope.contrib.decentralized.lora.adapter_bridge import AdapterBridge
from federatedscope.contrib.decentralized.lora.auxiliary_state import (
    apply_policy,
    communicates_auxiliary,
)
from federatedscope.contrib.decentralized.lora.factor_groups import (
    FACTOR_GROUPS,
    group_keys,
    set_trainable_group,
    trainable_groups,
)
from federatedscope.contrib.decentralized.lora.mixing.full_state import weighted_mix
from federatedscope.contrib.decentralized.lora.payloads import (
    PayloadSchema,
    decode_state,
    encode_state,
)
from federatedscope.contrib.decentralized.lora.schema import build_schema
from federatedscope.contrib.decentralized.methods.tad_lora import config as tad_config
from federatedscope.contrib.decentralized.methods.tad_lora.phases import (
    AlternationSchedule,
)

logger = logging.getLogger(__name__)


class TADLoRAPeer:
    """One TAD-LoRA learner. Owns its model, its schema, and its phase state."""

    def __init__(self, context: PeerContext, profile: str, gamma: float,
                 auxiliary_policy: str, interaction: str,
                 schedule: AlternationSchedule,
                 expected_rank: Optional[int] = None):
        self.peer_id = context.peer_id
        self.context = context
        self.profile = profile
        self.gamma = float(gamma)
        self.interaction = interaction
        self.schedule = schedule

        self.model = context.extras.get("model") or self._build_model(context)
        self.schema = build_schema(self.model)
        self.bridge = AdapterBridge(self.model, self.schema)

        # TAD-LoRA is common-rank. As for Dec-LoRA, the configured method rank
        # must be the rank of the adapter that was built, or a run would report
        # one rank and mix another.
        if expected_rank is not None and self.schema.rank != expected_rank:
            raise ValueError(
                f"peer {self.peer_id}: decentralized.methods.tad_lora.rank="
                f"{expected_rank} but the adapter that was built has rank "
                f"{self.schema.rank}. TAD-LoRA is common-rank; reconcile the "
                f"method rank with the adapter configuration "
                f"(glue.adapter.max_rank / args.r).")

        self.includes_auxiliary = communicates_auxiliary(auxiliary_policy, interaction)
        apply_policy(self.model, self.schema.auxiliary_keys, auxiliary_policy)
        self.payload_schema = PayloadSchema.from_bridge(
            self.bridge, include_auxiliary=self.includes_auxiliary)

        #: The runner's topology history, read for the recent contact degree.
        #: Every method is handed the same history.
        self.topology = context.extras.get("topology")

        self.trainer = context.extras.get("trainer")
        if self.trainer is None and context.data is not None:
            self.trainer = self._build_trainer(context)

        self._version = 0
        self._staged = {}
        self._next_token = 1
        self._initial_digest = None
        self.mixing_rounds = 0
        self.episodes = 0
        self.group_episodes = {group: 0 for group in FACTOR_GROUPS}

        # Put the model into the starting phase now, so a peer that is inspected
        # before its first episode already reports the phase it will train in.
        set_trainable_group(self.model, self.schema, self.schedule.group)

    # -- construction ------------------------------------------------------

    def common_construction_targets(self):
        """``common_canonical_v1``: the bridge and its component ids, ``0..r-1`` - the common
        rank's components are the canonical universe's first ``r``."""
        return self.bridge, tuple(range(int(self.schema.rank)))

    def _build_model(self, context: PeerContext):
        from federatedscope.core.auxiliaries.model_builder import get_model

        return get_model(context.config, local_data=context.data)

    def _build_trainer(self, context: PeerContext):
        from federatedscope.core.auxiliaries.trainer_builder import get_trainer
        from federatedscope.core.monitors.monitor import Monitor

        import federatedscope.contrib.trainer.glue_decentralized_trainer  # noqa: F401

        return get_trainer(
            model=self.model, data=context.data, device=context.device,
            config=context.config,
            monitor=Monitor(context.config, monitored_object=self),
        )

    # -- phase policy ------------------------------------------------------

    def _recent_degree(self) -> Optional[int]:
        """This peer's contact count over the topology window, or None.

        ``None`` when the peer was given no topology history; the schedule
        treats that as no reason to stretch. Under the runner the history
        already holds the current frame, so the first episode sees that frame's
        contacts (0 if it has none).
        """
        if self.topology is None:
            return None
        degrees = getattr(self.topology, "degree", None)
        if not callable(degrees):
            return None
        return int(degrees().get(self.peer_id, 0))

    # -- PeerAlgorithm -----------------------------------------------------

    def local_episode(self, steps: int) -> Mapping[str, object]:
        """Train the currently active factor group for ``steps`` steps.

        The group is applied to the model before training and the schedule is
        advanced after, so the episode that triggers a switch is trained under
        the old group and the next one under the new.
        """
        group = self.schedule.group
        set_trainable_group(self.model, self.schema, group)

        if self.trainer is None:
            outcome = self.schedule.advance(self._recent_degree())
            return {"optimizer_steps": 0, "requested_steps": steps,
                    "skipped_batches": 0, "trained": False,
                    "factor_group": group, "switched": outcome["switched"],
                    "alternation_interval": outcome["interval"]}

        steps_before = self.trainer.optimizer_steps
        skipped_before = self.trainer.skipped_batches
        _, _, metrics = self.trainer.train()
        performed = self.trainer.optimizer_steps - steps_before
        skipped = self.trainer.skipped_batches - skipped_before

        self.episodes += 1
        self.group_episodes[group] += 1
        outcome = self.schedule.advance(self._recent_degree())

        result = dict(metrics or {})
        result.update({
            "optimizer_steps": performed,
            "requested_steps": steps,
            "skipped_batches": skipped,
            "trained": True,
            "factor_group": group,
            "switched": outcome["switched"],
            "alternation_interval": outcome["interval"],
        })
        return result

    def evaluate(self, split: str = "val") -> Mapping[str, object]:
        if self.trainer is None:
            return self.consensus_fingerprint()
        metrics = dict(self.trainer.evaluate(target_data_split_name=split) or {})
        metrics.update(self.consensus_fingerprint())
        return metrics

    def consensus_fingerprint(self) -> Mapping[str, float]:
        state = self.bridge.read(self.schema.lora_keys())
        flat = torch.cat([t.reshape(-1).to(torch.float64) for t in state.values()])
        return {"adapter_mean": float(flat.mean()),
                "adapter_norm": float(flat.norm())}

    def state_version(self) -> int:
        return self._version

    def state_dict(self) -> Mapping[str, object]:
        """Adapter, trainer state and the alternation state.

        A resumed run that restarted the alternation, or resumed in the wrong
        group, would train a different sequence from the uninterrupted one.
        """
        state = {
            "version": self._version,
            "adapter": self.bridge.read(),
            "schema_hash": self.schema.compatibility_hash,
            "alternation": dict(self.schedule.state_dict()),
            "mixing_rounds": self.mixing_rounds,
            "episodes": self.episodes,
            "group_episodes": dict(self.group_episodes),
        }
        if self.trainer is not None:
            state["trainer"] = self.trainer.trainer_state()
        return state

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state["schema_hash"] != self.schema.compatibility_hash:
            raise ValueError(
                "checkpoint schema does not match this peer's adapter schema")
        self.bridge.write(state["adapter"],
                          expected_keys=self.bridge.expected_adapter_keys())
        self._version = int(state["version"])
        self.schedule.load_state_dict(state["alternation"])
        self.mixing_rounds = int(state.get("mixing_rounds", 0))
        self.episodes = int(state.get("episodes", 0))
        self.group_episodes = {group: int(state.get("group_episodes", {}).get(group, 0))
                               for group in FACTOR_GROUPS}
        # Put the model in the checkpointed phase too: restoring the schedule
        # alone would leave the wrong factor group trainable.
        set_trainable_group(self.model, self.schema, self.schedule.group)

        loader = getattr(self.trainer, "load_trainer_state", None)
        if self.trainer is not None and callable(loader):
            loader(state.get("trainer"))

    # -- staging ------------------------------------------------------------

    def _current_payload(self):
        return self.bridge.read(self.payload_schema.keys)

    def _stage(self, new_state) -> int:
        token = self._next_token
        self._next_token += 1
        self._staged[token] = {
            "previous": self.bridge.read(self.payload_schema.keys),
            "next": new_state,
            "committed": False,
        }
        return token

    def commit(self, token: int) -> None:
        staged = self._staged[token]
        self.bridge.write(staged["next"], expected_keys=self.payload_schema.keys)
        staged["committed"] = True
        self._version += 1
        self.mixing_rounds += 1

    def rollback(self, token: int) -> None:
        staged = self._staged.pop(token)
        self.bridge.write(staged["previous"], expected_keys=self.payload_schema.keys)
        if staged["committed"]:
            self._version -= 1
            self.mixing_rounds -= 1

    def discard(self, token: int) -> None:
        self._staged.pop(token, None)

    # -- GraphAlgorithm -----------------------------------------------------

    def snapshot(self) -> bytes:
        """The complete pre-round state: both factors, whatever the phase.

        The mixing weight applies to a peer's whole state; sending only the
        trainable group would make the exchanged quantity depend on the sender's
        phase.
        """
        return encode_state(self._current_payload())

    def stage_mix(self, weights, snapshots) -> int:
        decoded = {}
        for peer_id, blob in snapshots.items():
            if weights.get(peer_id, 0.0) == 0.0:
                continue
            state = decode_state(blob)
            self.payload_schema.validate(state)
            decoded[peer_id] = state
        return self._stage(weighted_mix(weights, decoded))

    def auxiliary_payload_bytes(self) -> int:
        if not self.includes_auxiliary or not self.schema.auxiliary_keys:
            return 0
        return len(encode_state(self.bridge.read(self.schema.auxiliary_keys)))

    # -- run manifest -------------------------------------------------------

    def initial_state_digest(self) -> str:
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
        return self.schema.dtype

    # -- MethodMetricProvider ------------------------------------------------

    def method_metrics(self) -> Mapping[str, object]:
        live = trainable_groups(self.model, self.schema)
        return {
            "tad_lora_rank": self.schema.rank,
            "tad_lora_scaling": self.schema.scaling,
            "tad_lora_profile": self.profile,
            "tad_lora_communicates_auxiliary": self.includes_auxiliary,
            "tad_lora_payload_keys": len(self.payload_schema.keys),
            "tad_lora_factor_group": self.schedule.group,
            # Read back from the model's parameters, not from the schedule.
            "tad_lora_trainable_groups": "".join(
                group for group in FACTOR_GROUPS if live.get(group)),
            "tad_lora_alternation_switches": self.schedule.switches,
            "tad_lora_alternation_interval": self.schedule.base_interval,
            "tad_lora_topology_aware": self.schedule.topology_aware,
            "tad_lora_episodes": self.episodes,
            "tad_lora_episodes_group_a": self.group_episodes[FACTOR_GROUPS[0]],
            "tad_lora_episodes_group_b": self.group_episodes[FACTOR_GROUPS[1]],
            "tad_lora_mixing_rounds": self.mixing_rounds,
            "tad_lora_factor_keys_per_group": len(group_keys(self.schema,
                                                             FACTOR_GROUPS[0])),
        }


class TADLoRAFactory:
    """Constructed with cfg so the profile can decide how ``W_t`` is obtained."""

    name = "tad_lora"

    def __init__(self, cfg):
        settings = cfg.decentralized.methods.tad_lora
        self.cfg = cfg
        self.profile = settings.profile
        self.gamma = float(cfg.decentralized.comparison.gamma)
        self.rank = int(settings.rank)
        self.interaction_form = tad_config.PROFILE_INTERACTION[self.profile]
        self.auxiliary_policy = cfg.decentralized.auxiliary_state.policy
        self.fixed_wt_spec = settings.fixed_wt
        self.alternation_interval = int(settings.alternation_interval)
        self.max_alternation_interval = int(settings.max_alternation_interval)
        self.topology_aware = bool(settings.topology_aware)
        self.start_group = settings.start_group
        self.topology_window = int(cfg.decentralized.topology.history_window)

    @staticmethod
    def validate_config(cfg) -> None:
        tad_config.validate(cfg)

    def fixed_mixing_matrix(self, peer_ids):
        """The paper's fixed ``W_t``, or None to use the runner's per-frame ``W_t``.

        With None (``common_matching_trace``) the runner compiles ``W_t`` from
        each frame of the contact trace: pair blocks at ``comparison.gamma`` over
        a matching frame (``graph_mixing: pair_block``), or Metropolis weights
        over the frame's available contacts (``dynamic_metropolis``).
        """
        if self.profile != tad_config.PROFILE_PAPER_WT:
            return None
        from federatedscope.contrib.decentralized.core.mixing_matrix import resolve_fixed_q

        return resolve_fixed_q(self.fixed_wt_spec, peer_ids, self.gamma)

    def create_peer(self, context: PeerContext) -> TADLoRAPeer:
        schedule = AlternationSchedule(
            base_interval=self.alternation_interval,
            topology_aware=self.topology_aware,
            window=self.topology_window,
            max_interval=self.max_alternation_interval,
            group=self.start_group,
        )
        return TADLoRAPeer(
            context=context,
            profile=self.profile,
            gamma=self.gamma,
            auxiliary_policy=self.auxiliary_policy,
            interaction=self.interaction_form,
            schedule=schedule,
            expected_rank=self.rank,
        )


from federatedscope.contrib.decentralized.registry import register_decentralized_method

register_decentralized_method("tad_lora", TADLoRAFactory)
