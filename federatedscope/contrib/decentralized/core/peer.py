"""Generic peer shell.

Hosts exactly one method plugin instance and adds bookkeeping. It owns no
algorithm semantics: every method-specific decision is delegated.
"""
from typing import Any, Mapping

from federatedscope.contrib.decentralized.core.interfaces import PeerContext
from federatedscope.contrib.decentralized.core.types import PeerId


class Peer:
    def __init__(self, peer_id: PeerId, algorithm: Any, context: PeerContext):
        self.peer_id = peer_id
        self.algorithm = algorithm
        self.context = context
        self.local_episodes = 0
        #: Optimizer steps the plugin reports performing (``optimizer_steps``); the
        #: requested count is used only when the plugin reports none.
        self.local_steps = 0
        self.skipped_batches = 0
        self.sessions = 0

    def local_episode(self, steps: int) -> Mapping[str, Any]:
        metrics = self.algorithm.local_episode(steps) or {}
        self.local_episodes += 1
        self.local_steps += int(metrics.get("optimizer_steps", steps))
        self.skipped_batches += int(metrics.get("skipped_batches", 0))
        return metrics

    def evaluate(self, split: str = "val") -> Mapping[str, Any]:
        return self.algorithm.evaluate(split) or {}

    def state_dict(self) -> dict:
        return {
            "peer_id": self.peer_id,
            "local_episodes": self.local_episodes,
            "local_steps": self.local_steps,
            "skipped_batches": self.skipped_batches,
            "sessions": self.sessions,
            "algorithm": self.algorithm.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.local_episodes = state["local_episodes"]
        self.local_steps = state["local_steps"]
        self.skipped_batches = state.get("skipped_batches", 0)
        self.sessions = state["sessions"]
        self.algorithm.load_state_dict(state["algorithm"])

    def __repr__(self) -> str:
        return f"Peer(id={self.peer_id}, algorithm={type(self.algorithm).__name__})"
