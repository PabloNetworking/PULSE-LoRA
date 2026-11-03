"""Method-independent metrics: events, per-peer records, and run summaries.

Offered communication and realized communication are reported separately
throughout: a session can be offered credit that it never uses.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from federatedscope.contrib.decentralized.core.types import CONTROL_ONLY_STATUSES


#: Substrings marking a metric where a LOWER value is better. Everything else is
#: treated as higher-is-better, which covers accuracy, f1, mcc, pearson, spearman.
_LOWER_IS_BETTER = ("loss", "error", "perplexity", "ppl")


def metric_lower_is_better(key: str) -> bool:
    lowered = str(key).lower()
    return any(marker in lowered for marker in _LOWER_IS_BETTER)


def _reason_counts(events) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for event in events:
        status = event.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


@dataclass
class MetricSink:
    """Collects event records and optionally streams them to JSONL."""

    path: Optional[str] = None
    events: List[dict] = field(default_factory=list)
    peers: Dict[int, dict] = field(default_factory=dict)
    _handle: Any = None

    def open(self) -> None:
        """Open the event log, re-emitting anything already in memory.

        On a resumed run ``self.events`` holds the history restored from the
        checkpoint; writing it first makes the resumed log match that of an
        uninterrupted run. For a fresh run the history is empty.
        """
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._handle = open(self.path, "w", encoding="utf-8")
        for record in self.events:
            self._write(record)

    def _write(self, record: dict) -> None:
        self._handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self._handle.flush()

    def record_event(self, record: dict) -> None:
        self.events.append(record)
        if self._handle is not None:
            self._write(record)

    def record_peer(self, peer_id: int, record: dict) -> None:
        self.peers.setdefault(int(peer_id), {}).update(record)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    # -- summaries ---------------------------------------------------------

    def summary(self) -> dict:
        offered = [e for e in self.events if e.get("event") == "pair_session"]
        graph = [e for e in self.events if e.get("event") == "graph_round"]
        committed = [e for e in offered if e.get("status") == "committed"]
        control_only = [e for e in offered
                        if e.get("status") in CONTROL_ONLY_STATUSES]
        failed = [e for e in offered if e.get("status") == "failed"]
        return {
            "offered_sessions": len(offered),
            "offered_graph_rounds": len(graph),
            "offered_capacity_bytes": sum(e.get("offered_payload_credit", 0) for e in offered),
            "completed_sessions": len(committed),
            "control_only_sessions": len(control_only),
            "failed_transactions": len(failed),
            "control_only_reason_counts": _reason_counts(control_only),
            # Directional as well as total: an asymmetric link or an asymmetric
            # payload policy is invisible in a single number.
            "realized_control_bytes": sum(e.get("control_bytes", 0) for e in offered + graph),
            "realized_control_bytes_forward": sum(
                e.get("control_bytes_forward", 0) for e in offered + graph),
            "realized_control_bytes_reverse": sum(
                e.get("control_bytes_reverse", 0) for e in offered + graph),
            "realized_payload_bytes": sum(e.get("payload_bytes", 0) for e in offered + graph),
            "realized_payload_bytes_forward": sum(
                e.get("payload_bytes_forward", 0) for e in offered + graph),
            "realized_payload_bytes_reverse": sum(
                e.get("payload_bytes_reverse", 0) for e in offered + graph),
            "realized_auxiliary_bytes": sum(e.get("auxiliary_bytes", 0) for e in offered + graph),
            "realized_auxiliary_bytes_forward": sum(
                e.get("auxiliary_bytes_forward", 0) for e in offered + graph),
            "realized_auxiliary_bytes_reverse": sum(
                e.get("auxiliary_bytes_reverse", 0) for e in offered + graph),
        }

    def state_dict(self) -> dict:
        return {"events": list(self.events), "peers": dict(self.peers)}

    def load_state_dict(self, state: dict) -> None:
        self.events = list(state["events"])
        self.peers = dict(state["peers"])
