"""Model-free simulation coordinator.

Holds the opportunity trace, the RNG service, the topology history, the event
queue and the event-id counter, the simulated clock, the frame cursor and the
metric sink.

It holds no ``torch.nn.Module``, adapter tensor, optimizer, method state or
global model dictionary, and no peer objects either, since a held peer (or a
callback closing over one) would reach tensors just as well. It stores peer
ids; the runner owns the peer table.
"""
from typing import Iterable, Optional

from federatedscope.contrib.decentralized.core.events import EventQueue
from federatedscope.contrib.decentralized.core.metrics import MetricSink
from federatedscope.contrib.decentralized.core.rng import RngService
from federatedscope.contrib.decentralized.core.topology import TopologyHistory
from federatedscope.contrib.decentralized.core.traces import CommunicationOpportunityTrace
from federatedscope.contrib.decentralized.core.types import PeerId


class DecentralizedSimulationCoordinator:
    def __init__(self, trace: CommunicationOpportunityTrace, rng: RngService,
                 peer_ids: Iterable[PeerId], sink: MetricSink,
                 topology_window: int = 16):
        self.trace = trace
        self.rng = rng
        self.peer_ids = tuple(sorted(int(p) for p in peer_ids))  # ids only
        self.sink = sink
        self.topology = TopologyHistory(self.peer_ids, window=topology_window)
        self.queue = EventQueue()
        self.clock_us = 0
        self.frame_cursor = 0
        self.next_event_id = 0

    def allocate_event_id(self) -> int:
        value = self.next_event_id
        self.next_event_id += 1
        return value

    def advance_clock(self, time_us: int) -> None:
        self.clock_us = max(self.clock_us, int(time_us))

    def state_dict(self) -> dict:
        return {
            "clock_us": self.clock_us,
            "frame_cursor": self.frame_cursor,
            "next_event_id": self.next_event_id,
            "trace_id": self.trace.trace_id,
            "rng": self.rng.state_dict(),
            "topology": self.topology.state_dict(),
            "sink": self.sink.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state["trace_id"] != self.trace.trace_id:
            raise ValueError(
                f"checkpoint was taken against trace {state['trace_id'][:12]}... but the "
                f"configured trace is {self.trace.trace_id[:12]}...; refusing to resume"
            )
        self.clock_us = state["clock_us"]
        self.frame_cursor = state["frame_cursor"]
        self.next_event_id = state["next_event_id"]
        self.rng.load_state_dict(state["rng"])
        self.topology.load_state_dict(state["topology"])
        self.sink.load_state_dict(state["sink"])
