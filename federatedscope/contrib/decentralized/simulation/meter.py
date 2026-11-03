"""Serialized bytes per session and per peer, as the run's wire-cost model prices them.

The runner's ledger counts logical bytes (encoded frames). The wire-cost model in force (in a
scenario ``socket_v1`` by default, wire_quote.py) decides admission in serialized bytes.
:class:`WireMeter` prices every pair session and every graph-round edge again from the wire
context its record carries, per direction:

  - ``fixed``: the control messages, which a pair session pays even when it moves no payload;
  - ``payload``: the messages that grow with the payload, for a committed exchange only;
  - ``reserve``: the bytes held back until the final commit (``socket_v1.session_reserve``).
    Admission requires ``fixed + payload + reserve <= offered``. The reserve is reported
    separately and not counted as sent.

A peer's upload is the ``fixed + payload`` bytes of the directions it sends, its download
those of the directions it receives. A pair session refused after it was priced counts its
fixed bytes; a refused graph-round edge counts no bytes (its row shows the quote if the edge
was priced). These are quotes from the model, not measurements.
"""
from typing import Any, Dict, List, Mapping

from federatedscope.contrib.decentralized.core.wire_cost import (
    FORWARD,
    REVERSE,
    GraphEdgeWireContext,
    SessionWireContext,
)


class MeterError(RuntimeError):
    """A committed exchange that its own wire quote does not admit."""


class WireMeter:
    """Prices the sessions of one run; use as (part of) the runner's observer."""

    def __init__(self, peers, pair_model, graph_model):
        self.pair_model = pair_model
        self.graph_model = graph_model
        self.sent = {int(p): 0 for p in peers}
        self.received = {int(p): 0 for p in peers}
        self.reserve = {int(p): 0 for p in peers}
        self.rows: List[Dict[str, Any]] = []

    def on_event(self, runner, event, record) -> None:
        if record.get("event") == "pair_session":
            self.price_session(record)
        elif record.get("event") == "graph_round":
            self.price_round(record, int(event.event_id))

    def _count(self, sender: int, receiver: int, priced: int, reserve: int) -> None:
        self.sent[sender] += priced
        self.received[receiver] += priced
        self.reserve[sender] += reserve

    def price_session(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        lo, hi = (int(p) for p in record["endpoints"])
        planning = record.get("planning") or {}
        selected = {str(p): len((planning.get(str(p)) or {}).get("selected") or []) for p in (lo, hi)}
        eligible = {str(p): len((planning.get(str(p)) or {}).get("eligible") or []) for p in (lo, hi)}
        committed = record.get("status") == "committed"
        units = selected[str(lo)] if committed else 0
        row = {"kind": "pair_session", "frame_index": int(record["frame_index"]),
               "event_id": int(record["event_id"]), "endpoints": [lo, hi],
               "status": record.get("status"), "reason": record.get("reason") or "",
               "selected": selected, "eligible": eligible,
               "budget_bound": bool(committed and selected[str(lo)] < eligible[str(lo)]),
               "directions": {}}
        raw = record.get("wire_context")
        if raw is None:      # refused before the wire was priced: control exceeded the credit
            self.rows.append(row)
            return row
        model, ctx = self.pair_model, SessionWireContext.from_dict(raw)
        for direction, sender, receiver in ((FORWARD, lo, hi), (REVERSE, hi, lo)):
            logical = int(record.get(f"payload_bytes_{direction}") or 0) if committed else 0
            fixed = int(model.fixed(ctx, direction))
            payload = int(model.payload(ctx, direction, logical, units)) if logical else 0
            reserve = int(model.reserve(ctx, direction))
            offered = int(record[f"offered_{direction}_bytes"])
            if committed and fixed + payload + reserve > offered:
                raise MeterError(f"event {row['event_id']} {direction}: {fixed}+{payload}+{reserve} "
                                 f"bytes committed against a credit of {offered}")
            row["directions"][direction] = {
                "offered": offered, "fixed": fixed, "payload": payload, "reserve": reserve,
                "priced": fixed + payload, "logical_payload": logical,
                "logical_control": int(record.get(f"control_bytes_{direction}") or 0)}
            self._count(sender, receiver, fixed + payload, reserve)
        self.rows.append(row)
        return row

    def price_round(self, record: Mapping[str, Any], event_id: int) -> List[Dict[str, Any]]:
        admitted = {tuple(int(x) for x in e) for e in record.get("affordable_edges") or []}
        skipped = {tuple(int(x) for x in s["endpoints"]): s for s in record.get("skipped_edges") or []}
        contexts = {}
        for raw in record.get("wire_contexts") or []:
            ctx = GraphEdgeWireContext.from_dict(raw)
            contexts[(int(ctx.lo), int(ctx.hi))] = ctx
        rows = []
        for edge in record.get("offered_edges") or []:
            lo, hi = (int(x) for x in edge)
            committed = (lo, hi) in admitted
            row = {"kind": "graph_edge", "frame_index": int(record["frame_index"]),
                   "event_id": int(event_id), "endpoints": [lo, hi],
                   "status": "committed" if committed else "refused",
                   "reason": (skipped.get((lo, hi)) or {}).get("reason", ""), "directions": {}}
            ctx = contexts.get((lo, hi))
            if committed and ctx is None:
                raise MeterError(f"admitted edge {(event_id, lo, hi)} carries no wire context")
            if ctx is not None:
                model = self.graph_model
                for direction, sender, receiver in ((FORWARD, lo, hi), (REVERSE, hi, lo)):
                    offered = int(ctx.gross_bytes[direction])
                    fixed = int(model.fixed(ctx, direction))
                    payload = int(model.payload(ctx, direction, int(ctx.snapshot_bytes[direction]), None))
                    reserve = int(model.reserve(ctx, direction))
                    if committed and fixed + payload + reserve > offered:
                        raise MeterError(f"edge {(event_id, lo, hi)} {direction}: {fixed}+{payload}+"
                                         f"{reserve} bytes committed against a credit of {offered}")
                    row["directions"][direction] = {
                        "offered": offered, "fixed": fixed, "payload": payload if committed else 0,
                        "needed_payload": payload, "reserve": reserve,
                        "priced": fixed + payload if committed else 0,
                        "logical_snapshot": int(ctx.snapshot_bytes[direction])}
                    if committed:
                        self._count(sender, receiver, fixed + payload, reserve)
            rows.append(row)
        self.rows.extend(rows)
        return rows

    def traffic(self, peer_id: int) -> Dict[str, int]:
        return {"upload_bytes": self.sent[int(peer_id)], "download_bytes": self.received[int(peer_id)],
                "reserve_bytes": self.reserve[int(peer_id)]}
