"""Deterministic event queue.

Ordering is total and reproducible: ties at equal simulated time are broken by
event kind, then by a structural key (endpoint pair, peer id, frame index or
round index), then by insertion sequence. The sequence number also guarantees
the heap never compares two event objects, which are not orderable.
"""
import heapq
from typing import Any, List, Optional, Tuple

from federatedscope.contrib.decentralized.core.types import EventKind


def structural_key(event) -> Tuple[int, ...]:
    """A deterministic integer key so equal-time events never depend on hashing."""
    endpoints = getattr(event, "endpoints", None)
    if endpoints is not None:
        return (int(endpoints[0]), int(endpoints[1]))
    peer_id = getattr(event, "peer_id", None)
    if peer_id is not None:
        return (int(peer_id), 0)
    frame_index = getattr(event, "frame_index", None)
    if frame_index is not None:
        return (int(frame_index), 0)
    return (int(getattr(event, "round_index", 0)), 0)


class EventQueue:
    def __init__(self):
        self._heap: List[Tuple] = []
        self._sequence = 0

    def push(self, event: Any) -> None:
        kind = int(getattr(event, "kind", EventKind.LOCAL_COMPUTE))
        heapq.heappush(
            self._heap,
            (int(event.time_us), kind, structural_key(event), self._sequence, event),
        )
        self._sequence += 1

    def pop(self) -> Optional[Any]:
        if not self._heap:
            return None
        return heapq.heappop(self._heap)[-1]

    def peek_time(self) -> Optional[int]:
        return self._heap[0][0] if self._heap else None

    def drain(self):
        while self._heap:
            yield heapq.heappop(self._heap)[-1]

    def pop_until(self, horizon: Optional[int]):
        """Yield events strictly before ``horizon``; ``None`` drains everything.

        The horizon keeps dispatch times globally nondecreasing: draining the
        whole queue at the end of each frame would dispatch an event dated beyond
        the next frame before that frame's own events had been created.
        """
        while self._heap:
            if horizon is not None and self._heap[0][0] >= horizon:
                return
            yield heapq.heappop(self._heap)[-1]

    def state_dict(self) -> dict:
        # Under event-driven scheduling an event can outlive the frame that created
        # it, so pending events are part of the run state and are checkpointed.
        return {"sequence": self._sequence,
                "entries": [(t, k, tuple(sk), seq, ev)
                            for (t, k, sk, seq, ev) in self._heap]}

    def load_state_dict(self, state: dict) -> None:
        self._sequence = int(state["sequence"])
        self._heap = [tuple(entry) for entry in state["entries"]]
        heapq.heapify(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)
