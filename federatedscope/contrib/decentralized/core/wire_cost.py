"""Wire denomination: what one byte of offered credit is, and what fits inside it.

A session's gross credit is directional and is counted in serialized application-layer
bytes, without TCP/IP or link-layer headers. Everything a session sends counts against
it: fixed message envelopes, manifests, masks, sketches, acknowledgements, checksums and
other non-factor bytes are control bytes; base64/JSON expansion and framing that grow with a
selected component are part of that component's wire cost; and the reserve for the
completion or abort messages must fit inside the credit too, so that a session can
always finish or fail cleanly.

The ledger (``core/accounting.ByteLedger``) keeps logical bytes: the encoded length of
every runtime frame, independent of any wire format. Admission, and so the number of
components a session carries, is decided by a ``WireCostModel``, which prices the same
session in wire bytes, per direction; the ledger's own credit stays an upper bound
(``SessionBudget.affordable``).

Models are registered by name, and the runner resolves the one named by
``decentralized.transport.wire_cost``. ``logical`` (defined here) is the identity: wire
bytes equal ledger bytes and the reserve is zero, so admission reduces to
``SessionBudget.can_afford``. ``socket_v1`` (``simulation/wire_quote.py``) is registered
when ``federatedscope.contrib.decentralized.simulation`` is imported. The run manifest
records the model in force; runs priced by different models are not comparable.
"""
import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

FORWARD = "forward"      # lower peer id -> higher peer id (endpoints[0] -> endpoints[1])
REVERSE = "reverse"
DIRECTIONS = (FORWARD, REVERSE)

#: The identity model: wire bytes equal ledger bytes.
LOGICAL = "logical"

#: The envelope: the largest single message a session may send. A wire-cost model prices a
#: session with a larger message at OVER_CAP (``socket_v1`` compares each message body,
#: without its length prefix). 64 MiB by default; ``decentralized.transport.envelope_bytes``
#: may raise it up to ``MAX_ENVELOPE_BYTES``. A wire context includes the envelope, and so
#: its digest covers it, only when it differs from the default.
DEFAULT_ENVELOPE_BYTES = 64 * 1024 * 1024
MAX_ENVELOPE_BYTES = 512 * 1024 * 1024
#: What a model returns for a payload that cannot be carried under the envelope in force:
#: larger than any credit, so admission says "does not fit" - never an exception.
OVER_CAP = 1 << 62


def validate_envelope_bytes(value, wire_cost: str = "socket_v1") -> int:
    """Return the configured envelope. Raises ValueError if it is not an integer, is below
    the default or above ``MAX_ENVELOPE_BYTES``, or differs from the default under the
    logical model, which prices no envelope."""
    if type(value) is bool or not isinstance(value, int):
        raise ValueError(f"decentralized.transport.envelope_bytes must be an integer number of "
                         f"bytes, got {value!r}")
    value = int(value)
    if value < DEFAULT_ENVELOPE_BYTES or value > MAX_ENVELOPE_BYTES:
        raise ValueError(f"decentralized.transport.envelope_bytes={value} is outside "
                         f"[{DEFAULT_ENVELOPE_BYTES}, {MAX_ENVELOPE_BYTES}]: 64 MiB is the "
                         f"default and 512 MiB the largest allowed")
    if value != DEFAULT_ENVELOPE_BYTES and str(wire_cost) == LOGICAL:
        raise ValueError(f"decentralized.transport.envelope_bytes={value} was configured under "
                         f"the {LOGICAL!r} wire cost, which prices no envelope; a larger "
                         f"envelope needs a model that prices one (socket_v1)")
    return value


@dataclass(frozen=True)
class SessionWireContext:
    """Everything a model needs to price one pair session. Both endpoints know it
    before either plans, so both price the session identically."""

    lo: int
    hi: int
    event_id: int
    checksum: bool
    #: Encoded (logical) bytes of the manifest frame sent in each direction.
    manifest_bytes: Dict[str, int]
    #: Encoded (logical) bytes of the acknowledgement frame in each direction.
    ack_bytes: Dict[str, int]
    #: The offered gross bytes in each direction. The plan call of the reference protocol
    #: carries them in its budget state, so they also set that message's size.
    gross_bytes: Dict[str, int] = None  # type: ignore[assignment]
    #: Whether the endpoint sending each direction reports auxiliary payload bytes (its
    #: plan reply then carries an "aux" count). Presence only: the model prices the count
    #: at a bound, since learning its value would take another message.
    auxiliary_present: Dict[str, bool] = None  # type: ignore[assignment]
    #: Decimal digits of the widest component identifier that can cross, and the most
    #: identifiers any one list can hold. Zero means no identifiers cross at all.
    identity_width: int = 0
    identity_count_bound: int = 0
    #: The envelope in force (the model's ``envelope_bytes``). Part of ``as_dict``, and so
    #: of the digest, only when it is not the default.
    envelope_bytes: int = DEFAULT_ENVELOPE_BYTES

    def as_dict(self) -> Dict[str, object]:
        out = self._as_dict()
        if int(self.envelope_bytes) != DEFAULT_ENVELOPE_BYTES:
            out["envelope_bytes"] = int(self.envelope_bytes)
        return out

    def _as_dict(self) -> Dict[str, object]:
        return {"lo": int(self.lo), "hi": int(self.hi), "event_id": int(self.event_id),
                "checksum": bool(self.checksum),
                "manifest_bytes": {d: int(self.manifest_bytes[d]) for d in DIRECTIONS},
                "ack_bytes": {d: int(self.ack_bytes[d]) for d in DIRECTIONS},
                "gross_bytes": ({d: int(self.gross_bytes[d]) for d in DIRECTIONS}
                                if self.gross_bytes else None),
                "auxiliary_present": ({d: bool(self.auxiliary_present.get(d))
                                       for d in DIRECTIONS} if self.auxiliary_present else None),
                "identity_width": int(self.identity_width),
                "identity_count_bound": int(self.identity_count_bound)}

    def digest(self) -> str:
        """The first sixteen hex characters of the SHA-256 of ``as_dict()``. The plan call
        of the reference protocol carries this digest in place of the context, so that the
        two endpoints can check that they price the same session."""
        material = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_dict(cls, raw: Dict[str, object]) -> "SessionWireContext":
        return cls(lo=int(raw["lo"]), hi=int(raw["hi"]), event_id=int(raw["event_id"]),
                   checksum=bool(raw["checksum"]),
                   manifest_bytes={d: int(raw["manifest_bytes"][d]) for d in DIRECTIONS},
                   ack_bytes={d: int(raw["ack_bytes"][d]) for d in DIRECTIONS},
                   gross_bytes=({d: int(raw["gross_bytes"][d]) for d in DIRECTIONS}
                                if raw.get("gross_bytes") else None),
                   auxiliary_present=({d: bool(raw["auxiliary_present"].get(d))
                                       for d in DIRECTIONS} if raw.get("auxiliary_present") else None),
                   identity_width=int(raw.get("identity_width", 0)),
                   identity_count_bound=int(raw.get("identity_count_bound", 0)),
                   envelope_bytes=int(raw.get("envelope_bytes", DEFAULT_ENVELOPE_BYTES)))


@dataclass(frozen=True)
class GraphEdgeWireContext:
    """Everything a model needs to price one edge of a graph round, known to both
    endpoints, so both price the edge identically.

    ``execute_graph_round`` builds it for each edge once both ``graph_manifest`` frames
    are debited and both snapshot frames are encoded, after the ledger has admitted the
    edge.
    """

    lo: int
    hi: int
    event_id: int
    checksum: bool
    #: Encoded (logical) bytes of the empty ``graph_manifest`` frame in each direction.
    manifest_bytes: Dict[str, int]
    #: Encoded (logical) bytes of the ``graph_snapshot`` frame in each direction: the
    #: forward one carries the lower peer's snapshot, the reverse one the higher's.
    snapshot_bytes: Dict[str, int]
    #: The offered gross bytes of each direction of the edge.
    gross_bytes: Dict[str, int]
    #: Whether the endpoint sending each direction reports auxiliary payload bytes (its
    #: snapshot size reply then carries an "aux" count). Presence only.
    auxiliary_present: Dict[str, bool]
    identity_width: int = 0
    identity_count_bound: int = 0
    #: A graph edge sends no acknowledgement frames. Zero, so that the logical model
    #: prices both forms by one rule (fixed = manifest + ack, the control the ledger
    #: charged). Not part of ``as_dict`` or the digest.
    ack_bytes: Dict[str, int] = field(default_factory=lambda: {FORWARD: 0, REVERSE: 0})
    #: The envelope in force; part of ``as_dict``, and so of the digest, only when it is
    #: not the default.
    envelope_bytes: int = DEFAULT_ENVELOPE_BYTES

    def as_dict(self) -> Dict[str, object]:
        out = self._as_dict()
        if int(self.envelope_bytes) != DEFAULT_ENVELOPE_BYTES:
            out["envelope_bytes"] = int(self.envelope_bytes)
        return out

    def _as_dict(self) -> Dict[str, object]:
        return {"form": "graph_round",
                "lo": int(self.lo), "hi": int(self.hi), "event_id": int(self.event_id),
                "checksum": bool(self.checksum),
                "manifest_bytes": {d: int(self.manifest_bytes[d]) for d in DIRECTIONS},
                "snapshot_bytes": {d: int(self.snapshot_bytes[d]) for d in DIRECTIONS},
                "gross_bytes": {d: int(self.gross_bytes[d]) for d in DIRECTIONS},
                "auxiliary_present": {d: bool(self.auxiliary_present.get(d))
                                      for d in DIRECTIONS},
                "identity_width": int(self.identity_width),
                "identity_count_bound": int(self.identity_count_bound)}

    def digest(self) -> str:
        """Sixteen hex characters over the canonical context; ``form`` is inside it, so
        a graph edge and a pair session can never reproduce each other's digest."""
        material = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_dict(cls, raw: Dict[str, object]) -> "GraphEdgeWireContext":
        return cls(lo=int(raw["lo"]), hi=int(raw["hi"]), event_id=int(raw["event_id"]),
                   checksum=bool(raw["checksum"]),
                   manifest_bytes={d: int(raw["manifest_bytes"][d]) for d in DIRECTIONS},
                   snapshot_bytes={d: int(raw["snapshot_bytes"][d]) for d in DIRECTIONS},
                   gross_bytes={d: int(raw["gross_bytes"][d]) for d in DIRECTIONS},
                   auxiliary_present={d: bool(raw["auxiliary_present"].get(d))
                                      for d in DIRECTIONS},
                   identity_width=int(raw.get("identity_width", 0)),
                   identity_count_bound=int(raw.get("identity_count_bound", 0)),
                   envelope_bytes=int(raw.get("envelope_bytes", DEFAULT_ENVELOPE_BYTES)))


class WireCostModel:
    """What a pair session or a graph-round edge costs on each directed hop, in
    serialized bytes.

    ``fixed``   every pre-terminal byte that does not scale with the payload;
    ``payload`` the bytes that carry a payload frame of ``logical_frame_bytes`` in that
                direction, with ``units`` selected components (for identifier lists);
    ``reserve`` the success/abort reserve that must stay free until the final commit.
    Admission for a payload is ``fixed + payload + reserve <= gross`` in both directions.
    """

    name = "abstract"
    #: The envelope this instance prices under; set by the resolvers from the run's
    #: configuration. A payload that cannot be carried under it is priced at OVER_CAP.
    envelope_bytes = DEFAULT_ENVELOPE_BYTES

    def fixed(self, ctx: SessionWireContext, direction: str) -> int:
        raise NotImplementedError

    def payload(self, ctx: SessionWireContext, direction: str,
                logical_frame_bytes: int, units: Optional[int]) -> int:
        raise NotImplementedError

    def reserve(self, ctx: SessionWireContext, direction: str) -> int:
        raise NotImplementedError


class LogicalWireCost(WireCostModel):
    """Wire bytes equal ledger bytes: fixed is the control the ledger charged (manifest +
    acknowledgement), payload is the frame's encoded length, reserve is 0, and there is no
    envelope. Admission under this model agrees with ``SessionBudget.can_afford``."""

    name = LOGICAL

    def fixed(self, ctx, direction):
        return int(ctx.manifest_bytes[direction]) + int(ctx.ack_bytes[direction])

    def payload(self, ctx, direction, logical_frame_bytes, units):
        return int(logical_frame_bytes)

    def reserve(self, ctx, direction):
        return 0


_REGISTRY: Dict[str, Callable[[], WireCostModel]] = {LOGICAL: LogicalWireCost}


def register_wire_cost(name: str, factory: Callable[[], WireCostModel]) -> None:
    """Make a model resolvable by name. Registering a different factory under a name that
    is already taken raises ValueError, so one name always means one model; registering
    the same factory again does nothing."""
    if name in _REGISTRY and _REGISTRY[name] is not factory:
        raise ValueError(f"wire cost model {name!r} is already registered")
    _REGISTRY[name] = factory


def registered_wire_costs():
    return tuple(sorted(_REGISTRY))


def resolve_wire_cost(name: str, envelope_bytes: int = DEFAULT_ENVELOPE_BYTES) -> WireCostModel:
    """The model registered as ``name``, set to the given envelope. An unregistered name
    raises ValueError instead of falling back to the logical model, which would size the
    run's sessions in the wrong denomination."""
    name = str(name)
    if name not in _REGISTRY:
        raise ValueError(
            f"wire cost model {name!r} is not registered in this process; registered: "
            f"{registered_wire_costs()}. socket_v1 is registered by importing "
            f"federatedscope.contrib.decentralized.simulation (wire_quote).")
    model = _REGISTRY[name]()
    model.envelope_bytes = validate_envelope_bytes(envelope_bytes, name)
    return model


# -- graph rounds: the same denomination, priced for an edge -----------------------------------
#
# A run names one denomination (``decentralized.transport.wire_cost``); each interaction
# form resolves that name in its own registry, because a graph edge and a pair session
# send different messages. The logical model is registered in both. A name registered
# for pair sessions only is refused for graph rounds rather than priced by a model built
# for another message sequence.

_GRAPH_REGISTRY: Dict[str, Callable[[], WireCostModel]] = {LOGICAL: LogicalWireCost}


def register_graph_wire_cost(name: str, factory: Callable[[], WireCostModel]) -> None:
    if name in _GRAPH_REGISTRY and _GRAPH_REGISTRY[name] is not factory:
        raise ValueError(f"graph-round wire cost model {name!r} is already registered")
    _GRAPH_REGISTRY[name] = factory


def registered_graph_wire_costs():
    return tuple(sorted(_GRAPH_REGISTRY))


def resolve_graph_wire_cost(name: str, envelope_bytes: int = DEFAULT_ENVELOPE_BYTES) -> WireCostModel:
    """The graph-round model registered as ``name``, set to the given envelope. An
    unregistered name raises ValueError."""
    name = str(name)
    if name not in _GRAPH_REGISTRY:
        raise ValueError(
            f"graph rounds under the {name!r} denomination have no wire cost model: the "
            f"name is not registered for graph rounds in this process; registered: "
            f"{registered_graph_wire_costs()}. socket_v1 registers its graph-round model "
            f"beside its pair-session model when federatedscope.contrib.decentralized."
            f"simulation is imported (wire_quote).")
    model = _GRAPH_REGISTRY[name]()
    model.envelope_bytes = validate_envelope_bytes(envelope_bytes, name)
    return model
