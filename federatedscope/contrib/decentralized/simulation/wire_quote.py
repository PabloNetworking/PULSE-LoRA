"""``socket_v1``: what a pair session or a graph-round edge costs in serialized bytes.

The model prices the message sequence a committed session exchanges in the reference
peer-to-peer protocol (``socket_v1.py``): length prefixes, envelopes, JSON call and reply
skeletons, the base64 expansion of every frame carried inside a reply, and the nested
``{"__bytes__": ...}`` encoding of a frame inside a reply value. The sizes come from the
protocol's own encoders, with a payload of known length added arithmetically, so a quote of a
large frame never builds it.

Every message whose body is known when the session is planned is priced exactly: both
manifests, the budget state the plan call carries, the fetch call, the unit frames, the
digest. Four items are only known after the plan is chosen and are priced at a deterministic
upper bound: the identity lists (k identifiers at the widest identifier's width), the two
token-bearing messages (an int below the transaction engine's 10^19 bound), the per-edge
session index in the commit reply (ten digits) and the auxiliary byte count.

``reserve`` is ``socket_v1.session_reserve``: the bytes that must stay free until the final
commit for the abort or completion messages. Admission is
``fixed + payload + reserve <= offered`` in both directions.

Registered under the name ``socket_v1`` when this module is imported.
"""
from typing import Dict, Optional

from federatedscope.contrib.decentralized.core.wire import encoded_size, payload_frame
from federatedscope.contrib.decentralized.simulation.socket_v1 import (
    encode_call,
    encode_reply,
    message_id,
    predicted_reply_socket_bytes_for,
    request_socket_bytes_for,
    session_reserve,
)
from federatedscope.contrib.decentralized.core.wire_cost import (
    DIRECTIONS,
    FORWARD,
    OVER_CAP,
    REVERSE,
    GraphEdgeWireContext,
    SessionWireContext,
    WireCostModel,
    register_graph_wire_cost,
    register_wire_cost,
)

SOCKET_V1 = "socket_v1"

#: A mixing weight in the stage_mix call is a JSON float. The widest ``repr`` of a
#: double in [0, 1] is 23 characters (17 significant digits, a point, ``e-`` and a
#: three-digit exponent); every weight is priced at one more. A deterministic upper
#: bound, never an average, because the realized row is not fixed until every edge of
#: the round is decided.
WEIGHT_CHARS = 24

#: A staging token is an int below the engine's bound (transaction._check_token); the
#: widest such int has this many decimal digits.
TOKEN_DIGITS = 19
#: The per-edge session index in the commit reply, bounded at ten digits.
SESSION_INDEX_DIGITS = 10


#: The auxiliary byte count in a plan reply, bounded at ten digits.
AUX_DIGITS = 10


def budget_state(lo: int, hi: int, gross: Dict[str, int], control: Dict[str, int],
                 spent: int, wire_cost: str, context_digest: Optional[str]) -> Dict[str, object]:
    """The plan call's budget state. The wire context travels as its digest."""
    return {
        "a": int(lo), "b": int(hi),
        "gross_forward": int(gross[FORWARD]), "gross_reverse": int(gross[REVERSE]),
        "control_forward": int(control[FORWARD]), "control_reverse": int(control[REVERSE]),
        "payload_spent": int(spent),
        "wire_cost": str(wire_cost),
        "wire_context_digest": context_digest,
    }


def _widest(width: int) -> int:
    return int("9" * width) if width > 0 else 0


#: The socket length prefix; a message body is a row's socket bytes minus this.
_PREFIX = 4


def _over_envelope(rows, envelope_bytes: int) -> bool:
    """Does any single message of the sequence exceed the envelope in force? Such a
    session is not refused with an exception: the model prices it at OVER_CAP, which no
    credit admits."""
    return any(int(b) - _PREFIX > int(envelope_bytes) for _, _, b, _ in rows)


def _sized(body) -> int:
    """A message body given either as its bytes or as its length."""
    return int(body) if isinstance(body, int) else len(body)


class SocketV1WireCost(WireCostModel):
    name = SOCKET_V1

    # -- the pieces -----------------------------------------------------------------------

    @staticmethod
    def _frame_overhead(ctx: SessionWireContext) -> int:
        return encoded_size(payload_frame("unit", ctx.lo, ctx.hi, ctx.event_id, b"",
                                          checksum=ctx.checksum))

    def _messages(self, ctx: SessionWireContext, logical_frame_bytes: Optional[int],
                  units: Optional[int]):
        """The committed pre-terminal sequence, one row per message: (label, direction,
        socket_bytes, scales_with_payload). Terminal traffic is the reserve's."""
        lo, hi = int(ctx.lo), int(ctx.hi)
        width = int(ctx.identity_width)
        count = int(units if units is not None else ctx.identity_count_bound)
        # Every value-dependent item is priced at its declared bound.
        widest = _widest(width)
        ids = [widest] * count if width else []
        L = int(logical_frame_bytes or 0)
        body_len = max(0, L - self._frame_overhead(ctx)) if L else 0
        token = _widest(TOKEN_DIGITS)
        session_index = _widest(SESSION_INDEX_DIGITS)
        rows = []

        # Sizes from the protocol's size functions: the encoders run over every message
        # skeleton and a payload of known length is added arithmetically
        # (test_socket_v1 pins them to the encoders).
        def request(counter, label, body, scales):
            mid = message_id(ctx.event_id, counter, label)
            rows.append((label, FORWARD, request_socket_bytes_for(mid, _sized(body)), scales))
            return mid

        def reply(mid, label, result, scales):
            rows.append((label + ":reply", REVERSE,
                         predicted_reply_socket_bytes_for(mid, lo, hi, hi, _sized(result)),
                         scales))

        # 1. build_manifest: a call; the responder's manifest frame rides back as the result
        env = request(1, "build_manifest", encode_call("build_manifest"), False)
        reply(env, "build_manifest", int(ctx.manifest_bytes[REVERSE]), False)
        # 2. consume_manifest: this endpoint's manifest frame, raw; a view handle back
        env = request(2, "consume_manifest", int(ctx.manifest_bytes[FORWARD]), False)
        reply(env, "consume_manifest", encode_reply(True, {"view": 1}), False)
        # 3. plan_payload: the budget state, exactly as shipped; size/digest/identities back
        gross = dict(ctx.gross_bytes or {FORWARD: 0, REVERSE: 0})
        control = {d: int(ctx.manifest_bytes[d]) + int(ctx.ack_bytes[d]) for d in DIRECTIONS}
        state = budget_state(lo, hi, gross, control, 0, self.name, ctx.digest())
        env = request(3, "plan_payload", encode_call("plan_payload", {"budget": state, "view": 1}),
                      False)
        meta = {"size": body_len, "digest": "f" * 64, "payload_identities": ids}
        if (ctx.auxiliary_present or {}).get(REVERSE):
            meta["aux"] = _widest(AUX_DIGITS)
        reply(env, "plan_payload", encode_reply(True, meta), True)
        # 4. fetch: a call; the responder's unit frame rides back as the result
        env = request(4, "fetch", encode_call("fetch", {"what": "plan"}), False)
        reply(env, "fetch", L, True)
        # 5. stage: this endpoint's unit frame, raw; the responder's token back
        env = request(5, "stage", L, True)
        reply(env, "stage", encode_reply(True, token), False)
        # 6. commit: the token; the responder's planning and received identities back
        env = request(6, "commit", encode_call("commit", {"token": token}), False)
        full = [widest] * int(ctx.identity_count_bound)
        planning = {"edge": [lo, hi], "session": session_index,
                    "eligible": full, "selected": ids, "explored": ids,
                    "materialised": full, "support": full}
        final = {"planning_identities": planning if width else {},
                 "received_identities": ids}
        reply(env, "commit", encode_reply(True, final), True)
        return rows

    # -- the model ------------------------------------------------------------------------

    def fixed(self, ctx: SessionWireContext, direction: str) -> int:
        rows = self._messages(ctx, None, None)
        if _over_envelope(rows, self.envelope_bytes):
            return OVER_CAP
        return sum(b for _, d, b, scales in rows if d == direction and not scales)

    def payload(self, ctx: SessionWireContext, direction: str, logical_frame_bytes: int,
                units: Optional[int]) -> int:
        rows = self._messages(ctx, logical_frame_bytes, units)
        if _over_envelope(rows, self.envelope_bytes):
            return OVER_CAP
        return sum(b for _, d, b, scales in rows if d == direction and scales)

    def reserve(self, ctx: SessionWireContext, direction: str) -> int:
        return int(session_reserve(int(ctx.event_id), int(ctx.lo), int(ctx.hi), True,
                                   bool(ctx.checksum))[direction])


register_wire_cost(SOCKET_V1, SocketV1WireCost)


def graph_fetch_args(state: Dict[str, object], snapshot_bytes: int) -> Dict[str, object]:
    """The fetch call of an admitted graph edge: the budget state (as ``budget_state``
    shapes it) and the size of the coordinator's snapshot frame."""
    return {"what": "snapshot", "budget": dict(state), "snapshot_bytes": int(snapshot_bytes)}


class SocketV1GraphWireCost(WireCostModel):
    """``socket_v1`` for one edge of a graph round, priced by the same encoders.

    Per edge, coordinator ``lo`` to responder ``hi`` and back: the snapshot size
    request and its reply (size, digest, aux); the fetch, carrying the admission state,
    and its reply carrying the responder's snapshot frame; the coordinator's snapshot
    frame pushed raw (``receive_snapshot``) and its length back; the mixing row
    (``stage_mix``) and the token back; the commit and its reply. Terminal traffic -
    the discard that carries the debited manifest frames, or an abort - is the
    reserve's: ``session_reserve`` for a graph round.

    Exact: the snapshot request, the fetch (given both snapshot sizes, which the context
    carries), the pushed frame and its length reply, the fetch reply, the snapshot reply's
    size and digest, a commit reply for a method that reports no identities. Bounded: the two
    mixing weights (``WEIGHT_CHARS`` each), the two token-bearing messages, the auxiliary byte
    count, and identity lists where a method reports them.
    """

    name = SOCKET_V1

    @staticmethod
    def _frame_overhead(ctx: GraphEdgeWireContext) -> int:
        return encoded_size(payload_frame("graph_snapshot", ctx.lo, ctx.hi, ctx.event_id, b"",
                                          checksum=ctx.checksum))

    def _messages(self, ctx: GraphEdgeWireContext, logical_forward: Optional[int],
                  logical_reverse: Optional[int]):
        """The committed pre-terminal sequence of one edge, one row per message:
        (label, direction, socket_bytes, scales_with_payload)."""
        lo, hi = int(ctx.lo), int(ctx.hi)
        overhead = self._frame_overhead(ctx)
        L_fwd, L_rev = int(logical_forward or 0), int(logical_reverse or 0)
        body_fwd = max(0, L_fwd - overhead) if L_fwd else 0
        body_rev = max(0, L_rev - overhead) if L_rev else 0
        token = _widest(TOKEN_DIGITS)
        width = int(ctx.identity_width)
        rows = []

        # Sizes from the protocol's size functions: the encoders run over every message
        # skeleton and a payload of known length is added arithmetically
        # (test_socket_v1 pins them to the encoders).
        def request(counter, label, body, scales):
            mid = message_id(ctx.event_id, counter, label)
            rows.append((label, FORWARD, request_socket_bytes_for(mid, _sized(body)), scales))
            return mid

        def reply(mid, label, result, scales):
            rows.append((label + ":reply", REVERSE,
                         predicted_reply_socket_bytes_for(mid, lo, hi, hi, _sized(result)),
                         scales))

        # 1. snapshot: a size request; the responder's size, digest (and aux) back.
        env = request(1, "snapshot", encode_call("snapshot"), False)
        meta = {"size": body_rev, "digest": "f" * 64}
        if (ctx.auxiliary_present or {}).get(REVERSE):
            meta["aux"] = _widest(AUX_DIGITS)
        reply(env, "snapshot", encode_reply(True, meta), True)
        # 2. fetch: the admission state and this endpoint's snapshot frame size, exactly
        #    as shipped; the responder's snapshot frame rides back as the result.
        gross = dict(ctx.gross_bytes or {FORWARD: 0, REVERSE: 0})
        control = {d: int(ctx.manifest_bytes[d]) for d in DIRECTIONS}
        state = budget_state(lo, hi, gross, control, 0, self.name, ctx.digest())
        env = request(2, "fetch", encode_call("fetch", graph_fetch_args(state, L_fwd)), True)
        reply(env, "fetch", L_rev, True)
        # 3. receive_snapshot: this endpoint's snapshot frame, raw; its length back.
        env = request(3, "receive_snapshot", L_fwd, True)
        reply(env, "receive_snapshot", encode_reply(True, body_fwd), True)
        # 4. stage_mix: the responder's row, two weights at a bounded width; a token back.
        base = encode_call("stage_mix", {"weights": {str(lo): 0.0, str(hi): 0.0}})
        padding = 2 * (WEIGHT_CHARS - len("0.0"))
        env = request(4, "stage_mix", base + b" " * padding, False)
        reply(env, "stage_mix", encode_reply(True, token), False)
        # 5. commit: the token; the responder's identities back where its method reports them.
        env = request(5, "commit", encode_call("commit", {"token": token}), False)
        if width:
            count = int(ctx.identity_count_bound)
            widest = _widest(width)
            ids = [widest] * count
            final = {"planning_identities": {"edge": [lo, hi], "session": _widest(
                SESSION_INDEX_DIGITS), "eligible": ids,
                "selected": ids, "explored": ids, "materialised": ids, "support": ids},
                     "received_identities": ids}
            reply(env, "commit", encode_reply(True, final), False)
        else:
            reply(env, "commit", encode_reply(True, None), False)
        return rows

    def _sizes(self, ctx: GraphEdgeWireContext, direction: str, logical_frame_bytes):
        """Both snapshot sizes, the priced direction's from the call and the other's
        from the context - which carries both, so the quote of one hop can price the
        digits that depend on the other hop's frame exactly."""
        forward = int(logical_frame_bytes if direction == FORWARD
                      else ctx.snapshot_bytes[FORWARD])
        reverse = int(logical_frame_bytes if direction == REVERSE
                      else ctx.snapshot_bytes[REVERSE])
        return forward, reverse

    def fixed(self, ctx: GraphEdgeWireContext, direction: str) -> int:
        rows = self._messages(ctx, None, None)
        if _over_envelope(rows, self.envelope_bytes):
            return OVER_CAP
        return sum(b for _, d, b, scales in rows if d == direction and not scales)

    def payload(self, ctx: GraphEdgeWireContext, direction: str, logical_frame_bytes: int,
                units: Optional[int]) -> int:
        forward, reverse = self._sizes(ctx, direction, logical_frame_bytes)
        rows = self._messages(ctx, forward, reverse)
        if _over_envelope(rows, self.envelope_bytes):
            return OVER_CAP
        return sum(b for _, d, b, scales in rows if d == direction and scales)

    def reserve(self, ctx: GraphEdgeWireContext, direction: str) -> int:
        return int(session_reserve(int(ctx.event_id), int(ctx.lo), int(ctx.hi), False,
                                   bool(ctx.checksum))[direction])


register_graph_wire_cost(SOCKET_V1, SocketV1GraphWireCost)
