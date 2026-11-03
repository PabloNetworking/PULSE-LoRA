"""Payload schemas and a deterministic tensor codec.

A payload is validated against the key set it is supposed to carry: the LoRA
keys, plus the auxiliary keys when the auxiliary policy communicates them. Under
``local`` and ``frozen`` a payload without auxiliary keys is valid and one with
them is rejected.
"""
import struct
from collections import OrderedDict
from dataclasses import dataclass
from typing import FrozenSet, Mapping, Optional

import torch

_DTYPE_CODES = {
    torch.float32: 1,
    torch.float64: 2,
    torch.float16: 3,
}
_CODE_DTYPES = {code: dtype for dtype, code in _DTYPE_CODES.items()}

_HEAD = struct.Struct("!I")
_ENTRY = struct.Struct("!HBB")


class PayloadError(ValueError):
    pass


@dataclass(frozen=True)
class PayloadSchema:
    """The exact key set of one payload, with its shapes and dtype.

    Shapes (when recorded) and dtype are checked as well as keys, before the
    payload is mixed: a correctly named tensor of the wrong shape or dtype would
    otherwise be caught only later in the bridge, or be cast by the mixing
    kernel.
    """

    keys: FrozenSet[str]
    includes_auxiliary: bool
    shapes: Optional[Mapping[str, tuple]] = None
    dtype: Optional[str] = None

    @classmethod
    def for_full_state(cls, schema, include_auxiliary: bool) -> "PayloadSchema":
        return cls(keys=schema.payload_keys(include_auxiliary),
                   includes_auxiliary=include_auxiliary, dtype=schema.dtype)

    @classmethod
    def from_bridge(cls, bridge, include_auxiliary: bool) -> "PayloadSchema":
        """Keys and tensor shapes from the live model."""
        keys = bridge.schema.payload_keys(include_auxiliary)
        state = bridge.read(keys)
        return cls(
            keys=frozenset(keys),
            includes_auxiliary=include_auxiliary,
            shapes={key: tuple(tensor.shape) for key, tensor in state.items()},
            dtype=bridge.schema.dtype,
        )

    def validate(self, state: Mapping[str, torch.Tensor]) -> None:
        provided = frozenset(state)
        if provided != self.keys:
            missing = sorted(self.keys - provided)
            extra = sorted(provided - self.keys)
            raise PayloadError(
                f"payload does not match its schema; missing={missing}, "
                f"unexpected={extra}"
            )
        if self.shapes is not None:
            for key in sorted(self.keys):
                expected = self.shapes.get(key)
                actual = tuple(state[key].shape)
                if expected is not None and actual != expected:
                    raise PayloadError(
                        f"{key}: shape mismatch, payload {actual} vs expected {expected}")
        if self.dtype is not None:
            for key in sorted(self.keys):
                actual = str(state[key].dtype)
                if actual != self.dtype:
                    raise PayloadError(
                        f"{key}: dtype mismatch, payload {actual} vs expected {self.dtype}")


def encode_state(state: Mapping[str, torch.Tensor]) -> bytes:
    """Deterministic: keys in sorted order, dtype and shape tagged explicitly."""
    chunks = [_HEAD.pack(len(state))]
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        if tensor.dtype not in _DTYPE_CODES:
            raise PayloadError(
                f"{key}: dtype {tensor.dtype} is not supported by the wire codec; "
                f"supported: {sorted(str(d) for d in _DTYPE_CODES)}"
            )
        raw_key = key.encode("utf-8")
        chunks.append(_ENTRY.pack(len(raw_key), _DTYPE_CODES[tensor.dtype], tensor.dim()))
        chunks.append(raw_key)
        chunks.append(struct.pack(f"!{tensor.dim()}I", *tensor.shape))
        chunks.append(tensor.numpy().tobytes())
    return b"".join(chunks)


def decode_state(blob: bytes) -> "OrderedDict[str, torch.Tensor]":
    import numpy as np

    offset = 0
    (count,) = _HEAD.unpack_from(blob, offset)
    offset += _HEAD.size

    out = OrderedDict()
    for _ in range(count):
        key_len, dtype_code, ndim = _ENTRY.unpack_from(blob, offset)
        offset += _ENTRY.size
        key = blob[offset: offset + key_len].decode("utf-8")
        offset += key_len
        shape = struct.unpack_from(f"!{ndim}I", blob, offset)
        offset += 4 * ndim

        dtype = _CODE_DTYPES.get(dtype_code)
        if dtype is None:
            raise PayloadError(f"unknown dtype code {dtype_code}")
        numel = 1
        for dim in shape:
            numel *= dim
        nbytes = numel * torch.empty(0, dtype=dtype).element_size()
        buffer = blob[offset: offset + nbytes]
        offset += nbytes

        array = np.frombuffer(buffer, dtype=np.dtype(str(dtype).replace("torch.", "")))
        out[key] = torch.from_numpy(array.copy()).reshape(tuple(shape))
    if offset != len(blob):
        raise PayloadError(f"trailing bytes in payload: consumed {offset} of {len(blob)}")
    return out
