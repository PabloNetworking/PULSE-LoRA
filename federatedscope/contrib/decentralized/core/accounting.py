"""Directional byte ledger and per-session budgets.

Offered capacity and realized traffic are kept separate throughout: a session
may offer credit that no payload ever uses, and bytes for frames that were
actually sent stay counted even when the learning state is rolled back - a
rollback cannot un-send a packet.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from federatedscope.contrib.decentralized.core.types import DirectionalCredits, PeerId
from federatedscope.contrib.decentralized.core.wire import Frame, encoded_size
from federatedscope.contrib.decentralized.core.wire_cost import (
    DIRECTIONS as WIRE_DIRECTIONS,
    FORWARD,
    OVER_CAP,
    REVERSE,
    LogicalWireCost,
    SessionWireContext,
    WireCostModel,
)

CATEGORY_CONTROL = "control"
CATEGORY_PAYLOAD = "payload"
CATEGORY_AUXILIARY = "auxiliary"


@dataclass
class ByteLedger:
    """Every debit is directional and categorised. Nothing is ever un-debited."""

    records: List[dict] = field(default_factory=list)

    def debit(self, sender: PeerId, receiver: PeerId, category: str, num_bytes: int,
              session_id: Optional[int] = None, transaction_id: Optional[int] = None,
              kind: str = "", identities: Optional[Sequence[int]] = None) -> int:
        if num_bytes < 0:
            raise ValueError("cannot debit negative bytes")
        self.records.append({
            "sender": int(sender), "receiver": int(receiver), "category": category,
            "bytes": int(num_bytes),
            # Always joinable to the session that produced it; joinable to a
            # transaction only when one existed at the time of the debit.
            "session_id": session_id,
            "transaction_id": transaction_id,
            "kind": kind,
            # The identities of the units this frame carried, as the sending
            # endpoint reports them when the bytes are debited. Byte counts alone
            # cannot tell one unit from another (every unit costs the same), so a
            # schedule that keeps serving one unit and one that alternates would
            # otherwise look the same.
            "identities": None if identities is None
                          else [int(c) for c in identities],
            # Filled in after a committed mix with the identities the receiving
            # endpoint decoded from the frame (``observe_identities``).
            "identities_observed": None,
        })
        return num_bytes

    def debit_frame(self, frame: Frame, category: str,
                    transaction_id: Optional[int] = None,
                    identities: Optional[Sequence[int]] = None) -> int:
        """Debit the frame's encoded length (``wire.encoded_size``)."""
        return self.debit(frame.sender, frame.receiver, category, encoded_size(frame),
                          session_id=frame.session_id, transaction_id=transaction_id,
                          kind=frame.kind, identities=identities)

    def observe_identities(self, session_id: int, sender: PeerId, receiver: PeerId,
                           category: str, identities: Sequence[int]) -> None:
        """Record what the receiver decoded, against the row that carried it.

        Raises ValueError unless the session has exactly one ``category`` row from
        ``sender`` to ``receiver`` and that row is not annotated yet.
        """
        matches = [r for r in self.records
                   if r["session_id"] == session_id and r["category"] == category
                   and int(r["sender"]) == int(sender)
                   and int(r["receiver"]) == int(receiver)]
        if len(matches) != 1:
            raise ValueError(
                f"session {session_id}: {len(matches)} {category} rows from "
                f"{sender} to {receiver}; cannot attribute what was received")
        if matches[0]["identities_observed"] is not None:
            raise ValueError(
                f"session {session_id}: the {category} row from {sender} to "
                f"{receiver} is already annotated")
        matches[0]["identities_observed"] = [int(c) for c in identities]

    def rows_for_session(self, session_id: int):
        return [r for r in self.records if r["session_id"] == session_id]

    def rows_for_transaction(self, transaction_id: int):
        return [r for r in self.records if r["transaction_id"] == transaction_id]

    def total(self, category: Optional[str] = None) -> int:
        return sum(r["bytes"] for r in self.records
                   if category is None or r["category"] == category)

    def by_direction(self) -> Dict[Tuple[PeerId, PeerId], int]:
        out: Dict[Tuple[PeerId, PeerId], int] = defaultdict(int)
        for r in self.records:
            out[(r["sender"], r["receiver"])] += r["bytes"]
        return dict(out)

    def sent_by(self, peer: PeerId, category: Optional[str] = None) -> int:
        return sum(r["bytes"] for r in self.records
                   if r["sender"] == peer and (category is None or r["category"] == category))

    def received_by(self, peer: PeerId, category: Optional[str] = None) -> int:
        return sum(r["bytes"] for r in self.records
                   if r["receiver"] == peer and (category is None or r["category"] == category))

    def state_dict(self) -> dict:
        return {"records": list(self.records)}

    def load_state_dict(self, state: dict) -> None:
        self.records = list(state["records"])


class SessionBudget:
    """Gross credits, control debits, and the conservative common payload credit."""

    def __init__(self, endpoints: Tuple[PeerId, PeerId], credits: DirectionalCredits):
        self.endpoints = endpoints
        self.gross = credits
        self._control_forward = 0
        self._control_reverse = 0
        self._payload_spent = 0
        #: The wire-cost model in force: the logical model until the executor attaches the
        #: run's model and the session's wire context (``attach_wire``) after charging the
        #: control frames, so every quote prices the context both endpoints share.
        self.wire: WireCostModel = LogicalWireCost()
        self.wire_context: Optional[SessionWireContext] = None

    def _is_forward(self, sender: PeerId) -> bool:
        return sender == self.endpoints[0]

    def charge_control(self, sender: PeerId, num_bytes: int) -> None:
        if self._is_forward(sender):
            self._control_forward += num_bytes
        else:
            self._control_reverse += num_bytes

    def control_overspent(self) -> bool:
        """Did the handshake alone exceed the offered credit in either direction?

        remaining_*() clamps at zero, which is right for computing payload credit
        but hides infeasibility: a session whose control frames do not fit cannot
        complete a handshake at all, and must not be reported as merely
        "no payload fitted".
        """
        return (self._control_forward > self.gross.forward_bytes
                or self._control_reverse > self.gross.reverse_bytes)

    def control_deficit(self) -> int:
        return max(self._control_forward - self.gross.forward_bytes,
                   self._control_reverse - self.gross.reverse_bytes, 0)

    def remaining_forward(self) -> int:
        return max(0, self.gross.forward_bytes - self._control_forward)

    def remaining_reverse(self) -> int:
        return max(0, self.gross.reverse_bytes - self._control_reverse)

    def payload_credit(self) -> int:
        """Conservative minimum of the two directional remainders, less what is spent."""
        return max(0, min(self.remaining_forward(), self.remaining_reverse()) - self._payload_spent)

    def can_afford(self, num_bytes: int) -> bool:
        return num_bytes <= self.payload_credit()

    # -- the wire view ---------------------------------------------------------------------

    def attach_wire(self, model: WireCostModel, context: SessionWireContext) -> None:
        self.wire = model
        self.wire_context = context

    def _gross(self, direction: str) -> int:
        """The wire credit of one hop: the attached context's ``gross_bytes`` when it has
        them, otherwise the trace's offer. The ledger's credit (``payload_credit``) is
        separate."""
        ctx = self.wire_context
        if ctx is not None and getattr(ctx, "gross_bytes", None):
            return int(ctx.gross_bytes[direction])
        return int(self.gross.forward_bytes if direction == FORWARD else self.gross.reverse_bytes)

    def wire_quote(self, logical_forward: int, logical_reverse: Optional[int] = None,
                   units: Optional[int] = None) -> Dict[str, Dict[str, int]]:
        """What a payload of these logical frame sizes costs on each directed hop, and
        whether it fits. ``fits`` is ``fixed + payload + reserve + spent <= offered`` with
        nothing priced at OVER_CAP, where ``spent`` is the priced size of the payload already
        charged in this session. ``logical_reverse`` defaults to ``logical_forward``.

        These are quotes from the wire-cost model, not measurements."""
        ctx = self.wire_context
        if ctx is None:
            raise ValueError("no wire context attached; the executor attaches one before "
                             "any payload is planned")
        logical = {FORWARD: int(logical_forward),
                   REVERSE: int(logical_forward if logical_reverse is None else logical_reverse)}
        quote: Dict[str, Dict[str, int]] = {}
        for direction in WIRE_DIRECTIONS:
            fixed = int(self.wire.fixed(ctx, direction))
            payload = int(self.wire.payload(ctx, direction, logical[direction], units))
            reserve = int(self.wire.reserve(ctx, direction))
            spent = (int(self.wire.payload(ctx, direction, self._payload_spent, units))
                     if self._payload_spent else 0)
            offered = self._gross(direction)
            total = fixed + payload + reserve + spent
            # A model prices a message larger than the envelope in force at OVER_CAP;
            # such a quote is flagged and never fits.
            over_cap = max(fixed, payload, spent) >= OVER_CAP
            quote[direction] = {"offered": offered, "logical": logical[direction],
                                "fixed": fixed, "payload": payload, "reserve": reserve,
                                "spent": spent, "total": total,
                                "headroom": offered - total,
                                "fits": (not over_cap) and total <= offered,
                                "over_cap": over_cap}
        return quote

    def wire_fits(self, logical_forward: int, logical_reverse: Optional[int] = None,
                  units: Optional[int] = None) -> bool:
        """Does a payload of these logical sizes fit the offered credit in both
        directions, as the model in force prices it? Without a wire context this is
        ``can_afford`` of the larger size; under the logical model it agrees with
        ``can_afford`` for a payload of the same size in both directions."""
        if self.wire_context is None:
            return self.can_afford(max(int(logical_forward),
                                       int(logical_forward if logical_reverse is None
                                           else logical_reverse)))
        quote = self.wire_quote(logical_forward, logical_reverse, units)
        return all(quote[d]["fits"] for d in WIRE_DIRECTIONS)

    def affordable(self, cost_of, upper: int) -> int:
        """The largest k in [0, upper] whose frame of ``cost_of(k)`` logical bytes both
        the ledger credit admits and the wire admits, with ``k`` identifiers crossing.

        A binary search, so admission must be monotone in k. ``upper`` is the caller's
        count of what the ledger credit affords, which the wire can only lower. Under
        the logical model the wire adds no constraint beyond the ledger credit."""
        lo, hi = 0, int(upper)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            need = int(cost_of(mid))
            if self.can_afford(need) and self.wire_fits(need, need, units=mid):
                lo = mid
            else:
                hi = mid - 1
        return lo

    def charge_payload(self, num_bytes: int) -> None:
        if not self.can_afford(num_bytes):
            raise ValueError(
                f"payload of {num_bytes}B exceeds credit {self.payload_credit()}B on "
                f"session {self.endpoints}"
            )
        self._payload_spent += num_bytes

    @property
    def control_bytes(self) -> int:
        return self._control_forward + self._control_reverse

    @property
    def payload_bytes(self) -> int:
        return self._payload_spent
