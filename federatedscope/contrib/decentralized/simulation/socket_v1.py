"""Message encoders of the reference peer-to-peer session protocol, ``socket_v1``.

``socket_v1`` prices a session in the bytes a socket would carry when the two endpoints run
on separate devices. One endpoint (the lower peer id) drives the session and calls the
other endpoint's hooks:

  - every request is an envelope (``encode_envelope``) behind a 4-byte length prefix. Its
    payload is either a JSON hook call (``encode_call``) or a raw runtime frame;
  - every reply is a JSON document (``build_reply``) behind a 4-byte length prefix, whose
    ``result`` is the base64 of either a JSON reply value (``encode_reply``) or a raw
    runtime frame;
  - bytes inside a call or reply value are carried as ``{"__bytes__": base64}``.

The functions below build these messages, or compute their length without building a
payload of known size. ``wire_quote`` uses them to price every message of a pair session
or graph-round edge.
"""
import base64
import hashlib
import json
import struct
import types
import uuid
from typing import Any, Dict, Optional

from federatedscope.contrib.decentralized.core.pair_session import ACK_COMMITTED, ACK_FAILED
from federatedscope.contrib.decentralized.core.wire import control_frame, encode_frame, payload_frame
from federatedscope.contrib.decentralized.core.wire_cost import FORWARD, REVERSE

#: The socket length prefix of every message.
_LENGTH = struct.Struct("!I")

#: Envelope header: magic, version, source, destination, message-id length, payload length.
ENVELOPE_VERSION = 1
ENVELOPE_MAGIC = b"DCX1"
_HEADER = struct.Struct("!4sBIIHI")
_DIGEST_BYTES = 32
MAX_MESSAGE_ID = 128

DELIVERED = "delivered"

#: Frames the executors build as control frames. Everything else is payload.
CONTROL_KINDS = ("manifest", "ack", "graph_manifest")

#: Every label a session message can carry. The longest one sizes the message id used
#: when a reply to a not-yet-known message has to be reserved.
LABELS = ("build_manifest", "consume_manifest", "plan_payload", "payload_identities",
          "auxiliary_payload_bytes", "stage", "fetch", "snapshot", "receive_snapshot",
          "stage_mix", "commit", "discard", "rollback", "ack_reason", "acknowledge",
          "planning_identities", "received_identities", "state_version",
          "session_abort", "session_complete", "session_skip")
_LONGEST_LABEL = max(len(label) for label in LABELS)
#: Upper bound of a staging token (``transaction._check_token``).
_TOKEN_BOUND = 10 ** 19

#: An abort carries its reason truncated to this many characters.
MAX_ABORT_REASON = 160
#: The error text of a refused reply.
REFUSAL_ERROR = "socket_byte_cap"


class EncodingError(ValueError):
    """A value or envelope the protocol cannot carry."""


def message_id(event_id: int, counter: int, label: str) -> str:
    """A message id. The counter has a fixed width, so within a session the id's length
    depends on the label alone and a later message's size can be computed before it is sent."""
    return f"s{int(event_id)}n{int(counter):04d}{label}"


def _frame_encoder(checksum: bool):
    """Encodes a runtime frame exactly as the executors build it."""

    def encode(kind, sender, receiver, body, session):
        build = control_frame if kind in CONTROL_KINDS else payload_frame
        return encode_frame(build(kind, int(sender), int(receiver), int(session.event_id),
                                  bytes(body or b""), checksum=checksum))
    return encode


# -- hook calls and reply values ---------------------------------------------------------------

def _encode(value: Any) -> Any:
    """A call argument or reply value as JSON data. Only inert scalars, bytes, UUID tokens
    and small containers can be carried."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, uuid.UUID):
        return {"__uuid__": str(value)}
    if isinstance(value, (bytes, bytearray)):
        return {"__bytes__": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, dict):
        return {"__dict__": [[_encode(k), _encode(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"__list__": [_encode(v) for v in value]}
    raise EncodingError(f"a {type(value).__name__} cannot be carried in a session message")


def encode_call(hook: str, args: Any = None) -> bytes:
    return json.dumps({"hook": hook, "args": _encode(args)},
                      separators=(",", ":"), sort_keys=True).encode("utf-8")


def encode_reply(ok: bool, value: Any = None, error: str = "") -> bytes:
    return json.dumps({"ok": bool(ok), "value": _encode(value), "error": error},
                      separators=(",", ":"), sort_keys=True).encode("utf-8")


# -- envelopes and replies ---------------------------------------------------------------------

def payload_digest(payload: bytes) -> str:
    """sha256 of the payload bytes."""
    if type(payload) is not bytes:
        raise EncodingError(f"payload is a {type(payload).__name__}, not bytes")
    return hashlib.sha256(payload).hexdigest()


class Envelope:
    """Routing metadata around an opaque payload: message id, source, destination and the
    payload's digest."""

    __slots__ = ("message_id", "source", "destination", "digest", "payload")

    def __init__(self, message_id: str, source: int, destination: int, payload: bytes):
        if type(message_id) is not str or not message_id or len(message_id) > MAX_MESSAGE_ID:
            raise EncodingError(f"message id must be a non-empty string of at most "
                                f"{MAX_MESSAGE_ID} characters")
        self.message_id = message_id
        self.source = int(source)
        self.destination = int(destination)
        self.payload = payload
        self.digest = payload_digest(payload)


def encode_envelope(envelope: Envelope) -> bytes:
    """Header, payload digest, message id, then the payload verbatim."""
    ident = envelope.message_id.encode("utf-8")
    header = _HEADER.pack(ENVELOPE_MAGIC, ENVELOPE_VERSION, envelope.source,
                          envelope.destination, len(ident), len(envelope.payload))
    return header + bytes.fromhex(envelope.digest) + ident + envelope.payload


def envelope_encoded_size(message_id: str, payload_len: int) -> int:
    """``len(encode_envelope(e))`` for an envelope with this id and a payload of this many
    bytes, without building the payload."""
    return (_HEADER.size + _DIGEST_BYTES + len(str(message_id).encode("utf-8"))
            + int(payload_len))


class Acknowledgement:
    """The delivery receipt a reply carries."""

    __slots__ = ("message_id", "source", "destination", "digest", "issuer", "status")

    def __init__(self, message_id: str, source: int, destination: int, digest: str,
                 issuer: int, status: str = DELIVERED):
        self.message_id = message_id
        self.source = source
        self.destination = destination
        self.digest = digest
        self.issuer = issuer
        self.status = status


def build_reply(ack: Acknowledgement, outcome: str, result: Optional[bytes] = None) -> bytes:
    """The reply bytes an endpoint writes."""
    reply = {
        "message_id": ack.message_id, "source": ack.source,
        "destination": ack.destination, "digest": ack.digest,
        "issuer": ack.issuer, "status": ack.status, "outcome": outcome,
    }
    if result is not None:
        reply["result"] = base64.b64encode(bytes(result)).decode("ascii")
    return json.dumps(reply).encode("utf-8")


def request_socket_bytes(envelope: Envelope) -> int:
    """Bytes a request occupies on its hop: length prefix plus the envelope."""
    return _LENGTH.size + len(encode_envelope(envelope))


def request_socket_bytes_for(message_id: str, payload_len: int) -> int:
    """``request_socket_bytes`` of an envelope with this id and payload length, without
    building the payload."""
    return _LENGTH.size + envelope_encoded_size(message_id, payload_len)


def _base64_chars(n: int) -> int:
    return 4 * ((int(n) + 2) // 3)


def predicted_reply_socket_bytes_for(message_id: str, source: int, destination: int,
                                     responder_id: int, result_len: Optional[int],
                                     outcome: str = "accepted") -> int:
    """``predicted_reply_socket_bytes`` without building the result: the reply over an empty
    result plus the base64 length of the result (base64 needs no JSON escaping). The digest
    field is 64 hex characters whatever the payload."""
    ack = Acknowledgement(str(message_id), int(source), int(destination), "0" * 64,
                          issuer=int(responder_id), status=DELIVERED)
    if result_len is None:
        return _LENGTH.size + len(build_reply(ack, outcome, None))
    return _LENGTH.size + len(build_reply(ack, outcome, b"")) + _base64_chars(result_len)


def predicted_reply_socket_bytes(envelope: Envelope, responder_id: int,
                                 result: Optional[bytes] = None,
                                 outcome: str = "accepted") -> int:
    """Bytes the destination's reply to ``envelope`` occupies on its hop."""
    ack = Acknowledgement(envelope.message_id, envelope.source, envelope.destination,
                          envelope.digest, issuer=int(responder_id), status=DELIVERED)
    return _LENGTH.size + len(build_reply(ack, outcome, result))


# -- the session reserve -----------------------------------------------------------------------

def session_reserve(event_id: int, lo: int, hi: int, pairwise: bool,
                    checksum: bool) -> Dict[str, int]:
    """Bytes that must stay free, per direction, until the final commit.

    The larger of the abort path (one abort message, its reply, and a refusal reply to
    whatever message was refused) and the completion path after the final commit (the
    committed acknowledgement for a pair session, the discard for a graph round). Both
    endpoints compute it identically.
    """
    encode = _frame_encoder(checksum)
    session = types.SimpleNamespace(event_id=event_id)
    ack_forward = encode("ack", lo, hi, ACK_FAILED, session) if pairwise else None
    ack_reverse = encode("ack", hi, lo, ACK_FAILED, session) if pairwise else None
    manifest_forward = None if pairwise else encode("graph_manifest", lo, hi, b"", session)
    manifest_reverse = None if pairwise else encode("graph_manifest", hi, lo, b"", session)

    abort = Envelope(message_id(event_id, 9999, "session_abort"), lo, hi,
                     encode_call("session_abort", {"reason": "x" * MAX_ABORT_REASON,
                                                   "ack": ack_forward,
                                                   "manifest": manifest_forward,
                                                   # false encodes one byte longer than true
                                                   "acknowledge": False}))
    anything = Envelope(message_id(event_id, 9999, "x" * _LONGEST_LABEL), lo, hi, b"x")
    abort_forward = request_socket_bytes(abort)
    abort_reverse = (predicted_reply_socket_bytes(anything, hi,
                                                  encode_reply(False, error=REFUSAL_ERROR))
                     + predicted_reply_socket_bytes(
                         abort, hi, encode_reply(True, {"ack": ack_reverse,
                                                        "manifest": manifest_reverse})))
    if pairwise:
        done = Envelope(message_id(event_id, 9999, "acknowledge"), lo, hi,
                        encode("ack", lo, hi, ACK_COMMITTED, session))
        done_reply = encode_reply(True, {"ack": encode("ack", hi, lo, ACK_COMMITTED, session)})
    else:
        done = Envelope(message_id(event_id, 9999, "discard"), lo, hi,
                        encode_call("discard", {"token": _TOKEN_BOUND,
                                                "manifest": manifest_forward}))
        done_reply = encode_reply(True, {"manifest": manifest_reverse})
    return {FORWARD: max(abort_forward, request_socket_bytes(done)),
            REVERSE: max(abort_reverse, predicted_reply_socket_bytes(done, hi, done_reply))}
