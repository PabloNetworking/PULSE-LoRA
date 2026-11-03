"""Versioned wire frames. The encoder is the sole byte authority.

What the ledger debits for a frame is exactly ``len(encode_frame(frame))``, never
an estimate of in-memory object size.

The runtime treats method payloads as opaque bytes. Tensor serialisation belongs
to the LoRA layer, which hands ``body`` down already encoded.
"""
import struct
import zlib
from dataclasses import dataclass, field
from typing import Optional

MAGIC = b"FLDC"
WIRE_VERSION = 1

FRAME_CONTROL = 0
FRAME_PAYLOAD = 1

_HEADER = struct.Struct("!4sBBBxIIQI")  # magic, version, type, kindlen, pad, snd, rcv, session, bodylen


class WireError(ValueError):
    pass


@dataclass(frozen=True)
class Frame:
    """One wire frame.

    The identifier is the **session** id, not a transaction id. Control traffic
    is sent before any transaction is opened, and a control-only session never
    opens one, so a transaction id is attached separately, on the ledger rows
    that have one.
    """

    frame_type: int
    kind: str
    sender: int
    receiver: int
    session_id: int
    body: bytes = b""
    checksum: bool = True

    def __post_init__(self):
        if self.frame_type not in (FRAME_CONTROL, FRAME_PAYLOAD):
            raise WireError(f"unknown frame_type {self.frame_type}")
        if len(self.kind.encode("utf-8")) > 255:
            raise WireError("frame kind must encode to at most 255 bytes")


def encode_frame(frame: Frame) -> bytes:
    kind = frame.kind.encode("utf-8")
    header = _HEADER.pack(
        MAGIC, WIRE_VERSION, frame.frame_type, len(kind),
        int(frame.sender), int(frame.receiver), int(frame.session_id), len(frame.body),
    )
    payload = header + kind + frame.body
    if frame.checksum:
        payload += struct.pack("!I", zlib.crc32(payload) & 0xFFFFFFFF)
    return payload


def decode_frame(raw: bytes, checksum: bool = True) -> Frame:
    if len(raw) < _HEADER.size:
        raise WireError("frame shorter than header")
    magic, version, frame_type, kind_len, sender, receiver, session, body_len = _HEADER.unpack(
        raw[: _HEADER.size]
    )
    if magic != MAGIC:
        raise WireError(f"bad magic {magic!r}")
    if version != WIRE_VERSION:
        raise WireError(f"unsupported wire version {version}; expected {WIRE_VERSION}")

    offset = _HEADER.size
    kind = raw[offset: offset + kind_len].decode("utf-8")
    offset += kind_len
    body = raw[offset: offset + body_len]
    offset += body_len

    if checksum:
        if len(raw) < offset + 4:
            raise WireError("frame is missing its checksum")
        (expected,) = struct.unpack("!I", raw[offset: offset + 4])
        actual = zlib.crc32(raw[:offset]) & 0xFFFFFFFF
        if expected != actual:
            raise WireError(f"checksum mismatch: expected {expected}, computed {actual}")

    return Frame(frame_type=frame_type, kind=kind, sender=sender, receiver=receiver,
                 session_id=session, body=body, checksum=checksum)


def encoded_size(frame: Frame) -> int:
    """The number of bytes the ledger debits for ``frame``: its encoded length."""
    return len(encode_frame(frame))


def control_frame(kind: str, sender: int, receiver: int, session_id: int,
                  body: bytes = b"", checksum: bool = True) -> Frame:
    return Frame(FRAME_CONTROL, kind, sender, receiver, session_id, body, checksum)


def payload_frame(kind: str, sender: int, receiver: int, session_id: int,
                  body: bytes, checksum: bool = True) -> Frame:
    return Frame(FRAME_PAYLOAD, kind, sender, receiver, session_id, body, checksum)
