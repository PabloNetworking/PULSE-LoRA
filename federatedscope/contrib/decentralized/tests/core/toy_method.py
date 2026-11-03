"""A scalar toy method for the runtime tests.

It holds one float and gossips it, so the runner, transaction engine, graph
executor, accounting and checkpointing can be exercised without any LoRA
concept. It implements both interaction forms, so the pairwise and graph paths
are covered by the same double.
"""
import struct

from federatedscope.contrib.decentralized.core.interfaces import (
    INTERACTION_PAIRWISE,
)

_PACK = struct.Struct("!d")


class ToyAlgorithm:
    def __init__(self, peer_id, value, gamma=0.5, drift=0.0):
        self.peer_id = peer_id
        self.value = float(value)
        self.gamma = float(gamma)
        self.drift = float(drift)
        self._version = 0
        self._staged = {}
        self._next_token = 1
        self.acknowledgements = []

    # -- PeerAlgorithm ----------------------------------------------------
    def local_episode(self, steps):
        self.value += self.drift * steps
        # The toy has no optimizer, so it reports the requested count as performed.
        return {"value": self.value, "optimizer_steps": steps,
                "requested_steps": steps, "skipped_batches": 0}

    def evaluate(self, split="val"):
        return {"value": self.value}

    def state_version(self):
        return self._version

    def state_dict(self):
        return {"value": self.value, "version": self._version, "peer_id": self.peer_id}

    def load_state_dict(self, state):
        self.value = float(state["value"])
        self._version = int(state["version"])

    # -- PairwiseAlgorithm: the four-stage negotiation -----------------------
    def build_manifest(self, session):
        return _PACK.pack(float(self._version))

    def consume_manifest(self, remote_manifest, session):
        (remote_version,) = _PACK.unpack(remote_manifest)
        return {"state_version": remote_version}

    def plan_payload(self, session, budget, remote_view):
        body = _PACK.pack(self.value)
        if budget is not None and len(body) > budget.payload_credit():
            return None
        return body

    def acknowledge(self, session, committed, reason=""):
        self.acknowledgements.append((bool(committed), reason))

    def stage(self, incoming, session):
        (other,) = _PACK.unpack(incoming)
        token = self._next_token
        self._next_token += 1
        # Snapshots live here, in the plugin, keyed by an inert scalar token.
        self._staged[token] = {
            "previous": self.value,
            "next": (1.0 - self.gamma) * self.value + self.gamma * other,
        }
        return token

    def commit(self, token):
        # Keep the snapshot: this endpoint must remain undoable until the whole
        # transaction finishes, in case the other endpoint fails afterwards.
        staged = self._staged[token]
        self.value = staged["next"]
        staged["committed"] = True
        self._version += 1

    def rollback(self, token):
        staged = self._staged.pop(token)
        self.value = staged["previous"]
        if staged.get("committed"):
            self._version -= 1

    def discard(self, token):
        self._staged.pop(token, None)

    # -- GraphAlgorithm -----------------------------------------------------
    def snapshot(self):
        return _PACK.pack(self.value)

    def stage_mix(self, weights, snapshots):
        total = 0.0
        for peer_id, weight in weights.items():
            (value,) = _PACK.unpack(snapshots[peer_id])
            total += weight * value
        token = self._next_token
        self._next_token += 1
        self._staged[token] = {"previous": self.value, "next": total}
        return token

    def method_metrics(self):
        return {"toy_value": self.value}


class ToyFactory:
    """Factory double. Mirrors the shape a real plugin's factory has."""

    def __init__(self, cfg=None, interaction_form=None, gamma=0.5,
                 initial=None, drift=0.0):
        self.name = "toy_method"
        if interaction_form is None:
            interaction_form = (cfg.decentralized.execution.interaction
                                if cfg is not None else INTERACTION_PAIRWISE)
        self.interaction_form = interaction_form
        self.gamma = gamma
        self.initial = initial or {}
        self.drift = drift

    @staticmethod
    def validate_config(cfg):
        return None

    def create_peer(self, context):
        value = self.initial.get(context.peer_id, float(context.peer_id))
        return ToyAlgorithm(context.peer_id, value, gamma=self.gamma, drift=self.drift)
