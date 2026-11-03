"""Immutable records shared by the decentralized runtime.

Nothing here may reference a LoRA tensor, a component id, a trainable support,
or any method-specific concept. Method payloads are opaque bytes to the runtime.

Unit policy: durations are integer microseconds and goodput is integer bits per
second, per direction. Every credit computation is integer arithmetic so that
byte accounting is exactly reproducible and a digest over normalised values is
stable.
"""
from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from typing import Mapping, Optional, Tuple

PeerId = int
FrameIndex = int
EventId = int
TransactionId = int

BITS_PER_BYTE = 8
MICROSECONDS_PER_SECOND = 1_000_000

#: Multipliers into microseconds.
DURATION_UNITS: Mapping[str, int] = {"s": 1_000_000, "ms": 1_000, "us": 1}
#: Multipliers into bits per second.
GOODPUT_UNITS: Mapping[str, int] = {
    "bps": 1,
    "kbps": 1_000,
    "Mbps": 1_000_000,
    "Gbps": 1_000_000_000,
}

#: Transport is a label only: two sessions with identical end-to-end fields behave
#: identically whatever their transport label.
TRANSPORT_DIRECT = "direct"
TRANSPORT_INFRASTRUCTURE = "infrastructure"
KNOWN_TRANSPORTS = (TRANSPORT_DIRECT, TRANSPORT_INFRASTRUCTURE)


class UnitError(ValueError):
    """A declared value does not normalise exactly to the integer base unit."""


def normalise_exact(value, unit: str, table: Mapping[str, int], what: str) -> int:
    """Convert ``value`` in ``unit`` to the integer base unit, or refuse.

    Rounding is never applied: a value that does not land exactly on an integer
    is a malformed trace, not something to silently approximate.
    """
    if unit not in table:
        raise UnitError(
            f"{what}: unknown unit {unit!r}; expected one of {sorted(table)}"
        )
    scaled = Fraction(str(value)) * table[unit]
    if scaled.denominator != 1:
        raise UnitError(
            f"{what}: {value} {unit} does not normalise to an integer "
            f"(got {scaled}); refusing to round"
        )
    return int(scaled)


def gross_credit_bytes(goodput_bps: int, duration_us: int) -> int:
    """Gross directional byte credit. Integer, floored, never over-credits."""
    if goodput_bps < 0 or duration_us < 0:
        raise ValueError("goodput and duration must be non-negative")
    return (goodput_bps * duration_us) // (BITS_PER_BYTE * MICROSECONDS_PER_SECOND)


class EventKind(IntEnum):
    """Ordering rank for deterministic tie-breaking at equal simulated time."""

    LOCAL_COMPUTE = 0
    PAIR_SESSION = 1
    GRAPH_MIXING = 2
    EVALUATION = 3
    STOP = 4


@dataclass(frozen=True)
class DirectionalCredits:
    """Gross byte credits for one session, per direction.

    ``forward`` is ``endpoints[0] -> endpoints[1]``; ``reverse`` is the other way.
    """

    forward_bytes: int
    reverse_bytes: int

    def minimum(self) -> int:
        return min(self.forward_bytes, self.reverse_bytes)


@dataclass(frozen=True)
class CommunicationOpportunity:
    """One offered endpoint-to-endpoint link in one trace frame.

    Purely exogenous: it carries no method name, no selected payload, no success
    flag, and no method-generated edge.
    """

    endpoints: Tuple[PeerId, PeerId]
    duration_us: int
    goodput_bps_forward: int
    goodput_bps_reverse: int
    transport: str = TRANSPORT_DIRECT
    available: bool = True

    def gross_credits(self) -> DirectionalCredits:
        return DirectionalCredits(
            forward_bytes=gross_credit_bytes(self.goodput_bps_forward, self.duration_us),
            reverse_bytes=gross_credit_bytes(self.goodput_bps_reverse, self.duration_us),
        )


@dataclass(frozen=True)
class OpportunityFrame:
    index: FrameIndex
    time_us: int
    opportunities: Tuple[CommunicationOpportunity, ...]


@dataclass(frozen=True)
class PairSessionEvent:
    """A bilateral endpoint session compiled from one offered opportunity."""

    event_id: EventId
    time_us: int
    frame_index: FrameIndex
    endpoints: Tuple[PeerId, PeerId]
    duration_us: int
    credits: DirectionalCredits
    transport: str

    kind: EventKind = EventKind.PAIR_SESSION


@dataclass(frozen=True)
class GraphEdgeInfo:
    endpoints: Tuple[PeerId, PeerId]
    duration_us: int
    credits: DirectionalCredits
    transport: str


@dataclass(frozen=True)
class GraphMixingEvent:
    """A synchronous mixing round: every peer mixes from the same pre-round snapshots."""

    event_id: EventId
    time_us: int
    frame_index: FrameIndex
    peer_order: Tuple[PeerId, ...]
    matrix: Tuple[Tuple[float, ...], ...]
    edges: Tuple[GraphEdgeInfo, ...]
    #: Nonzero off-diagonal weights in ``matrix`` for which this frame offered no
    #: available opportunity. They must be resolved by the executor's skipped-edge
    #: policy: leaving their weight in place would mix state over a link that
    #: carried no bytes.
    unavailable_edges: Tuple[Tuple[PeerId, PeerId], ...] = ()

    kind: EventKind = EventKind.GRAPH_MIXING

    def weight(self, i: PeerId, j: PeerId) -> float:
        return self.matrix[self.peer_order.index(i)][self.peer_order.index(j)]


@dataclass(frozen=True)
class LocalComputeEvent:
    event_id: EventId
    time_us: int
    peer_id: PeerId
    episode: int

    kind: EventKind = EventKind.LOCAL_COMPUTE


@dataclass(frozen=True)
class EvaluationEvent:
    event_id: EventId
    time_us: int
    round_index: int

    kind: EventKind = EventKind.EVALUATION


@dataclass(frozen=True)
class MethodEnvelope:
    """Opaque method payload. The runtime never decodes ``body``."""

    method: str
    schema_version: int
    unit_type: str
    body: bytes
    digest: Optional[str] = None


#: Every status the pair-session executor can stamp on a record. Use these names
#: rather than string literals, so that the executor and the code that reads its
#: records cannot drift apart.
STATUS_CONTROL_ONLY = "control_only"              # handshake done, no payload
STATUS_CONTROL_EXCEEDS_CAPACITY = "control_exceeds_capacity"  # handshake did not fit
STATUS_COMMITTED = "committed"                    # committed at both endpoints
STATUS_FAILED = "failed"                          # opened and did not commit

PAIR_SESSION_STATUSES = frozenset({
    STATUS_CONTROL_ONLY,
    STATUS_CONTROL_EXCEEDS_CAPACITY,
    STATUS_COMMITTED,
    STATUS_FAILED,
})

#: The outcomes in which no payload moved.
CONTROL_ONLY_STATUSES = frozenset({STATUS_CONTROL_ONLY,
                                   STATUS_CONTROL_EXCEEDS_CAPACITY})
