"""Dec-LoRA peer and factory.

The peer implements the shared pairwise and graph interfaces
(``core/interfaces.py``) on top of the shared LoRA layer: adapter bridge,
payload codec and mixing kernels.
"""
import json
import logging
from typing import Mapping, Optional

import torch

from federatedscope.contrib.decentralized.core.interfaces import PeerContext
from federatedscope.contrib.decentralized.lora.adapter_bridge import AdapterBridge
from federatedscope.contrib.decentralized.lora.auxiliary_state import (
    apply_policy,
    communicates_auxiliary,
)
from federatedscope.contrib.decentralized.lora.mixing.factor_gossip import bilateral_gossip
from federatedscope.contrib.decentralized.lora.mixing.full_state import weighted_mix
from federatedscope.contrib.decentralized.core.wire import encoded_size, payload_frame
from federatedscope.contrib.decentralized.lora.payloads import (
    PayloadSchema,
    decode_state,
    encode_state,
)
from federatedscope.contrib.decentralized.lora.schema import build_schema
from federatedscope.contrib.decentralized.methods.dec_lora import config as dec_config

logger = logging.getLogger(__name__)


class DecLoRAPeer:
    """One Dec-LoRA learner. Owns its model, its schema, and its optimizer."""

    def __init__(self, context: PeerContext, profile: str, gamma: float,
                 auxiliary_policy: str, interaction: str,
                 expected_rank: Optional[int] = None):
        self.peer_id = context.peer_id
        self.context = context
        self.profile = profile
        self.gamma = float(gamma)
        self.interaction = interaction

        self.model = context.extras.get("model") or self._build_model(context)
        self.schema = build_schema(self.model)
        self.bridge = AdapterBridge(self.model, self.schema)

        # The configured method rank must be the rank of the adapter that was built.
        if expected_rank is not None and self.schema.rank != expected_rank:
            raise ValueError(
                f"peer {self.peer_id}: decentralized.methods.dec_lora.rank="
                f"{expected_rank} but the adapter that was built has rank "
                f"{self.schema.rank}. Dec-LoRA is common-rank; reconcile the method "
                f"rank with the adapter configuration (glue.adapter.max_rank / args.r)."
            )

        self.includes_auxiliary = communicates_auxiliary(auxiliary_policy, interaction)
        apply_policy(self.model, self.schema.auxiliary_keys, auxiliary_policy)
        # from_bridge records the live tensor shapes, so a payload of the wrong
        # shape or dtype is rejected when it is decoded, before it is mixed.
        self.payload_schema = PayloadSchema.from_bridge(
            self.bridge, include_auxiliary=self.includes_auxiliary)

        self.trainer = context.extras.get("trainer")
        if self.trainer is None and context.data is not None:
            self.trainer = self._build_trainer(context)

        self._version = 0
        self._staged = {}
        self._next_token = 1
        self._initial_digest = None
        self.last_outcome = None
        self.committed_sessions = 0
        self.unsuccessful_sessions = 0

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

        # Importing the module registers 'gluedecentralizedtrainer'.
        import federatedscope.contrib.trainer.glue_decentralized_trainer  # noqa: F401

        return get_trainer(
            model=self.model, data=context.data, device=context.device,
            config=context.config, monitor=Monitor(context.config, monitored_object=self),
        )

    # -- PeerAlgorithm --------------------------------------------------------

    def local_episode(self, steps: int) -> Mapping[str, object]:
        """Train locally for exactly ``steps`` optimizer steps.

        The count is read back from the trainer. Config validation makes
        ``train.local_update_steps`` equal ``decentralized.execution.local_steps``,
        so the trainer runs ``steps`` batches; a NaN-loss batch is skipped without
        an optimizer step, so steps performed plus batches skipped must equal
        ``steps``, or this raises.

        A peer with no data trains nothing and only takes part in mixing.
        """
        if self.trainer is None:
            return {"optimizer_steps": 0, "requested_steps": steps,
                    "skipped_batches": 0, "trained": False}

        steps_before = self.trainer.optimizer_steps
        skipped_before = self.trainer.skipped_batches
        _, _, metrics = self.trainer.train()
        performed = self.trainer.optimizer_steps - steps_before
        skipped = self.trainer.skipped_batches - skipped_before

        if performed + skipped != steps:
            raise RuntimeError(
                f"peer {self.peer_id}: local episode performed {performed} optimizer "
                f"steps and skipped {skipped} batches, which does not account for the "
                f"{steps} requested. decentralized.execution.local_steps is "
                f"authoritative; check train.local_update_steps and train.batch_or_epoch."
            )

        return dict(metrics or {}, optimizer_steps=performed, requested_steps=steps,
                    skipped_batches=skipped, trained=True)

    def evaluate(self, split: str = "val") -> Mapping[str, object]:
        if self.trainer is None:
            return self.consensus_fingerprint()
        metrics = dict(self.trainer.evaluate(target_data_split_name=split) or {})
        metrics.update(self.consensus_fingerprint())
        return metrics

    def consensus_fingerprint(self) -> Mapping[str, float]:
        """Scalar summary of this peer's adapter, for consensus diagnostics."""
        state = self.bridge.read(self.schema.lora_keys())
        flat = torch.cat([tensor.reshape(-1).to(torch.float64) for tensor in state.values()])
        return {
            "adapter_mean": float(flat.mean()),
            "adapter_norm": float(flat.norm()),
        }

    def state_version(self) -> int:
        return self._version

    def state_dict(self) -> Mapping[str, object]:
        """Everything needed to resume identically, not just the adapter.

        The trainer's optimizer and scheduler state are included: restoring the
        tensors alone would restart the Adam moments and the LR schedule, and the
        resumed run would diverge from an uninterrupted one.
        """
        state = {
            "version": self._version,
            "adapter": self.bridge.read(),
            "schema_hash": self.schema.compatibility_hash,
            "committed_sessions": self.committed_sessions,
            "unsuccessful_sessions": self.unsuccessful_sessions,
        }
        if self.trainer is not None:
            state["trainer"] = self.trainer.trainer_state()
        return state

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state["schema_hash"] != self.schema.compatibility_hash:
            raise ValueError(
                "checkpoint schema does not match this peer's adapter schema"
            )
        # Expected keys come from the live adapter, never from the incoming
        # mapping, so a checkpoint missing a tensor cannot pass by defining a
        # reduced expectation.
        self.bridge.write(state["adapter"],
                          expected_keys=self.bridge.expected_adapter_keys())
        self._version = int(state["version"])
        self.committed_sessions = int(state.get("committed_sessions", 0))
        self.unsuccessful_sessions = int(state.get("unsuccessful_sessions", 0))

        # The trainer restores its own optimizer, scheduler and counters.
        loader = getattr(self.trainer, "load_trainer_state", None)
        if self.trainer is not None and callable(loader):
            loader(state.get("trainer"))

    # -- payload helpers --------------------------------------------------------

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
        # The snapshot is kept: this endpoint must stay undoable until the whole
        # transaction finishes, in case the other endpoint fails afterwards.
        staged = self._staged[token]
        self.bridge.write(staged["next"], expected_keys=self.payload_schema.keys)
        staged["committed"] = True
        self._version += 1

    def rollback(self, token: int) -> None:
        staged = self._staged.pop(token)
        self.bridge.write(staged["previous"], expected_keys=self.payload_schema.keys)
        if staged["committed"]:
            self._version -= 1

    def discard(self, token: int) -> None:
        self._staged.pop(token, None)

    # -- PairwiseAlgorithm --------------------------------------------------------

    MANIFEST_SCHEMA_VERSION = 1

    def build_manifest(self, session) -> bytes:
        """Declare compatibility and cost, not the payload itself.

        The schema hash stands in for the key list, which would cost kilobytes
        of control bytes. ``payload_bytes`` lets the partner plan against the
        size of what it is about to receive.
        """
        payload = self._current_payload()
        declaration = {
            "v": self.MANIFEST_SCHEMA_VERSION,
            "method": "dec_lora",
            "schema_hash": self.schema.compatibility_hash,
            "rank": self.schema.rank,
            "scaling": self.schema.scaling,
            "payload_key_count": len(self.payload_schema.keys),
            "payload_bytes": len(encode_state(payload)),
            "includes_auxiliary": self.includes_auxiliary,
            "state_version": self._version,
        }
        return json.dumps(declaration, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")

    def consume_manifest(self, remote_manifest: bytes, session) -> Mapping[str, object]:
        """Validate the peer is compatible before any payload is planned."""
        try:
            remote = json.loads(remote_manifest.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"unreadable manifest from the peer: {exc}") from exc

        if remote.get("v") != self.MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"manifest schema version {remote.get('v')!r} is not "
                f"{self.MANIFEST_SCHEMA_VERSION}")
        if remote.get("schema_hash") != self.schema.compatibility_hash:
            raise ValueError(
                f"incompatible LoRA schema: peer {remote.get('schema_hash', '?')[:12]}... "
                f"vs local {self.schema.compatibility_hash[:12]}...")
        if remote.get("includes_auxiliary") != self.includes_auxiliary:
            raise ValueError(
                f"auxiliary-state policy disagreement: peer includes_auxiliary="
                f"{remote.get('includes_auxiliary')}, local={self.includes_auxiliary}")
        if remote.get("payload_key_count") != len(self.payload_schema.keys):
            raise ValueError(
                f"payload key count disagreement: peer {remote.get('payload_key_count')} "
                f"vs local {len(self.payload_schema.keys)}")
        return remote

    def plan_payload(self, session, budget, remote_view) -> Optional[bytes]:
        """Plan from local state, the remote declaration, and the budget.

        Dec-LoRA's payload is one indivisible full-state unit, so the plan is
        all-or-nothing. The remote's declared size is used so both endpoints reach
        the same decision independently: the session cost is the larger of the two
        directions.
        """
        if remote_view is None:
            return None
        body = encode_state(self._current_payload())
        remote_bytes = int(remote_view.get("payload_bytes", len(body)))
        overhead = self._frame_overhead(session)
        needed = max(len(body), remote_bytes) + overhead
        # The payload must fit both the ledger's payload credit and the run's wire-cost
        # model; a whole-adapter payload is sent whole or not at all.
        if needed > budget.payload_credit() or not budget.wire_fits(needed):
            return None
        return body

    def _frame_overhead(self, session) -> int:
        """Encoded wire-frame overhead on top of the body, measured on an empty frame."""
        probe = payload_frame("unit", self.peer_id, self.peer_id, 0, b"")
        return encoded_size(probe)

    def acknowledge(self, session, committed: bool, reason: str = "") -> None:
        self.last_outcome = {
            "committed": bool(committed),
            "reason": reason,
            "endpoints": list(getattr(session, "endpoints", ())),
        }
        if committed:
            self.committed_sessions += 1
        else:
            self.unsuccessful_sessions += 1

    def stage(self, incoming: bytes, session) -> int:
        remote = decode_state(incoming)
        self.payload_schema.validate(remote)
        local = self._current_payload()
        return self._stage(bilateral_gossip(local, remote, self.gamma))

    # -- GraphAlgorithm -------------------------------------------------------------

    def snapshot(self) -> bytes:
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
        """How many encoded bytes of the atomic payload are auxiliary state.

        Reported separately as a breakdown, not a second budget: it is still part
        of the one indivisible payload. Zero under the ``local`` and ``frozen``
        policies.
        """
        if not self.includes_auxiliary or not self.schema.auxiliary_keys:
            return 0
        return len(encode_state(self.bridge.read(self.schema.auxiliary_keys)))


    def initial_state_digest(self) -> str:
        """Digest of the effective update ``scale * B @ A`` this peer started from.

        The scale is this method's alpha / rank. Captured the first time it is
        asked for; the runner asks right after building its peers (after the
        common construction, before the first session) and records it in the
        run manifest.
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

        The run manifest records it as one of the axes two runs are compared on:
        runs at different precision spend different bytes on the same payload
        and accumulate different round-off.
        """
        return self.schema.dtype

    # -- MethodMetricProvider ----------------------------------------------------------

    def method_metrics(self) -> Mapping[str, object]:
        return {
            "dec_lora_rank": self.schema.rank,
            "dec_lora_scaling": self.schema.scaling,
            "dec_lora_profile": self.profile,
            "dec_lora_communicates_auxiliary": self.includes_auxiliary,
            "dec_lora_payload_keys": len(self.payload_schema.keys),
            "dec_lora_committed_sessions": self.committed_sessions,
            "dec_lora_unsuccessful_sessions": self.unsuccessful_sessions,
        }


class DecLoRAFactory:
    """Constructed with cfg so the profile can decide the interaction form."""

    name = "dec_lora"

    def __init__(self, cfg):
        settings = cfg.decentralized.methods.dec_lora
        self.cfg = cfg
        self.profile = settings.profile
        # comparison.gamma is authoritative; config validation refuses a
        # methods.dec_lora.gamma that differs from it.
        self.gamma = float(cfg.decentralized.comparison.gamma)
        self.rank = int(settings.rank)
        self.interaction_form = dec_config.PROFILE_INTERACTION[self.profile]
        self.auxiliary_policy = cfg.decentralized.auxiliary_state.policy
        self.fixed_q_spec = settings.fixed_q

    @staticmethod
    def validate_config(cfg) -> None:
        dec_config.validate(cfg)

    def fixed_mixing_matrix(self, peer_ids):
        """Supply the paper-native fixed Q, or None if this profile has none.

        A duck-typed hook, so the runtime asks any graph method the same question
        without knowing which method it is talking to.
        """
        if self.profile != dec_config.PROFILE_PAPER_FIXED_Q:
            return None
        from federatedscope.contrib.decentralized.core.mixing_matrix import resolve_fixed_q

        return resolve_fixed_q(self.fixed_q_spec, peer_ids, self.gamma)

    def create_peer(self, context: PeerContext) -> DecLoRAPeer:
        return DecLoRAPeer(
            context=context,
            profile=self.profile,
            gamma=self.gamma,
            auxiliary_policy=self.auxiliary_policy,
            interaction=self.interaction_form,
            expected_rank=self.rank,
        )


from federatedscope.contrib.decentralized.registry import register_decentralized_method

register_decentralized_method("dec_lora", DecLoRAFactory)
