"""Topology history over offered communication.

Records which endpoint pairs each frame *offered* (not which sessions succeeded,
which is a method outcome) over a window of recent frames, so that methods can
consult recent contact structure without any service owning learning state.
Peers receive it as ``PeerContext.extras["topology"]``; a method that adapts to
recent contact degree reads ``degree()``.
"""
from collections import deque
from typing import Deque, Dict, Iterable, List, Set, Tuple

from federatedscope.contrib.decentralized.core.types import PeerId


class TopologyHistory:
    def __init__(self, peer_ids: Iterable[PeerId], window: int = 16):
        self.peer_ids = tuple(sorted(peer_ids))
        self.window = int(window)
        self._frames: Deque[Tuple[int, Tuple[Tuple[PeerId, PeerId], ...]]] = deque(
            maxlen=self.window if self.window > 0 else None
        )

    def record_frame(self, frame_index: int, edges: Iterable[Tuple[PeerId, PeerId]]) -> None:
        normalised = tuple(sorted(tuple(sorted(edge)) for edge in edges))
        self._frames.append((frame_index, normalised))

    def recent_edges(self) -> List[Tuple[PeerId, PeerId]]:
        seen: List[Tuple[PeerId, PeerId]] = []
        for _, edges in self._frames:
            for edge in edges:
                if edge not in seen:
                    seen.append(edge)
        return seen

    def degree(self) -> Dict[PeerId, int]:
        counts = {peer: 0 for peer in self.peer_ids}
        for _, edges in self._frames:
            for a, b in edges:
                counts[a] = counts.get(a, 0) + 1
                counts[b] = counts.get(b, 0) + 1
        return counts

    def union_is_connected(self) -> bool:
        """Is the union of recorded edges connected over the retained window?"""
        if len(self.peer_ids) <= 1:
            return True
        adjacency: Dict[PeerId, Set[PeerId]] = {p: set() for p in self.peer_ids}
        for _, edges in self._frames:
            for a, b in edges:
                adjacency[a].add(b)
                adjacency[b].add(a)
        start = self.peer_ids[0]
        seen = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return len(seen) == len(self.peer_ids)

    def state_dict(self) -> dict:
        return {"window": self.window, "frames": [list(f) for f in self._frames]}

    def load_state_dict(self, state: dict) -> None:
        self.window = int(state["window"])
        self._frames = deque(
            [(idx, tuple(tuple(e) for e in edges)) for idx, edges in state["frames"]],
            maxlen=self.window if self.window > 0 else None,
        )
