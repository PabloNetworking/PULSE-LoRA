"""Typed complete-component exchange units and their exact wire cost.

A PULSE pair unit is one canonical component: the ``p``-th row of every LoRA
``A`` and the ``p``-th column of every matching ``B``, across every injected
module. Never ``A`` without its paired ``B``, never one layer without the
others: the complete cross-layer block is the atomic payload and commit unit,
so a unit is encoded whole or not at all.

Cost is derived from key names, shapes and dtype, never from values, so a
selector can price every candidate without materialising a tensor. The
component id is zero-padded to a fixed width, so every unit costs the same
number of bytes: without padding, component 9 and component 10 would have
different key lengths and capacity would not be a simple multiplication.

The tests check ``unit_cost_bytes`` and ``units_cost_bytes`` against
``encode_state``.
"""
from typing import Dict, Mapping

import torch

from federatedscope.contrib.decentralized.lora.payloads import (
    PayloadError, decode_state, encode_state,
)
from federatedscope.contrib.decentralized.methods.pulse_lora.components import (
    SIDE_A, SIDE_B, CanonicalComponentSchema,
)

#: ``struct.Struct("!I")`` - the count header ``encode_state`` writes once.
_BLOB_HEADER_BYTES = 4
#: ``struct.Struct("!HBB")`` - key length, dtype code, ndim, per entry.
_ENTRY_HEADER_BYTES = 4
#: Component slices are one-dimensional, so one 4-byte extent each.
_SHAPE_BYTES = 4


class UnitError(ValueError):
    """A unit is malformed: incomplete, mis-shaped, or not decodable."""


def id_width(schema: CanonicalComponentSchema) -> int:
    """Decimal width of the largest component id, so every key is equal length."""
    return len(str(schema.r_max - 1))


def unit_key(schema: CanonicalComponentSchema, component: int, module: str,
             side: str) -> str:
    """``c007/<module>|A``.

    The id is a prefix: :func:`decode_units` splits it off at the first ``/``,
    and the rest is the block's ``<module>|<side>`` slot.
    """
    return f"c{component:0{id_width(schema)}d}/{module}|{side}"


def unit_cost_bytes(schema: CanonicalComponentSchema) -> int:
    """Exact encoded bytes one complete component adds to a unit blob.

    Excludes the blob's single count header - see :func:`units_cost_bytes`.
    """
    element = torch.empty(0, dtype=_dtype(schema)).element_size()
    width = id_width(schema)
    total = 0
    for spec in schema.lora.modules:
        for side, numel in ((SIDE_A, spec.in_features), (SIDE_B, spec.out_features)):
            key = f"c{0:0{width}d}/{spec.module}|{side}"
            total += (_ENTRY_HEADER_BYTES + len(key.encode("utf-8"))
                      + _SHAPE_BYTES + numel * element)
    return total


def units_cost_bytes(schema: CanonicalComponentSchema, count: int) -> int:
    """Exact encoded size of a blob carrying ``count`` complete components."""
    if count < 0:
        raise UnitError(f"unit count cannot be negative: {count}")
    return _BLOB_HEADER_BYTES + count * unit_cost_bytes(schema)


def affordable_units(schema: CanonicalComponentSchema, credit: int,
                     frame_overhead: int) -> int:
    """How many complete components fit in ``credit`` bytes, framing included.

    Floors at zero and never over-credits. ``credit`` is inclusive, matching
    ``SessionBudget.can_afford``: a plan costing exactly the credit is allowed.
    """
    per_unit = unit_cost_bytes(schema)
    fixed = _BLOB_HEADER_BYTES + frame_overhead
    if credit < fixed + per_unit:
        return 0
    return (credit - fixed) // per_unit


def encode_units(schema: CanonicalComponentSchema,
                 blocks: Mapping[int, Mapping[str, torch.Tensor]]) -> bytes:
    """Encode complete components. Incompleteness is an error, never a truncation."""
    flat: Dict[str, torch.Tensor] = {}
    for component in sorted(blocks):
        block = blocks[component]
        _validate_complete(schema, component, block)
        for spec in schema.lora.modules:
            for side in (SIDE_A, SIDE_B):
                tensor = block[f"{spec.module}|{side}"]
                # gather() returns active slices on the model's device and relay or
                # untouched slices on the CPU. encode_state would move them to the
                # CPU anyway; doing it here makes the move explicit.
                flat[unit_key(schema, component, spec.module, side)] = \
                    tensor.detach().to("cpu")
    return encode_state(flat)


def decode_units(schema: CanonicalComponentSchema,
                 blob: bytes) -> Dict[int, Dict[str, torch.Tensor]]:
    """Decode and re-validate. A partial component in the wire form is rejected."""
    try:
        flat = decode_state(blob)
    except (PayloadError, Exception) as exc:      # decode_state raises several kinds
        if isinstance(exc, UnitError):
            raise
        raise UnitError(f"unit blob is not decodable: {exc}") from exc

    width = id_width(schema)
    blocks: Dict[int, Dict[str, torch.Tensor]] = {}
    for key, tensor in flat.items():
        head, sep, rest = key.partition("/")
        module, bar, side = rest.rpartition("|")
        if not sep or not bar or not head.startswith("c"):
            raise UnitError(f"unit key {key!r} is not of the form 'c<id>/<module>|<side>'")
        digits = head[1:]
        if len(digits) != width or not digits.isdigit():
            raise UnitError(
                f"unit key {key!r} has a {len(digits)}-digit id; this schema uses "
                f"{width}-digit ids, so the peers disagree about r_max")
        blocks.setdefault(int(digits), {})[f"{module}|{side}"] = tensor

    for component in sorted(blocks):
        _validate_complete(schema, component, blocks[component])
    return blocks


def _validate_complete(schema: CanonicalComponentSchema, component: int,
                       block: Mapping[str, torch.Tensor]) -> None:
    """Every module, both sides, right shape, right dtype - or nothing."""
    if component not in range(schema.r_max):
        raise UnitError(
            f"unit component {component} is outside P = 0..{schema.r_max - 1}")

    expected = {}
    for spec in schema.lora.modules:
        expected[f"{spec.module}|{SIDE_A}"] = (spec.in_features,)
        expected[f"{spec.module}|{SIDE_B}"] = (spec.out_features,)

    provided = set(block)
    if provided != set(expected):
        missing = sorted(set(expected) - provided)
        extra = sorted(provided - set(expected))
        raise UnitError(
            f"component {component} is not a complete cross-layer block: "
            f"missing={missing}, unexpected={extra}. A and B travel together "
            f"across every injected module or not at all.")

    dtype = _dtype(schema)
    for key in sorted(expected):
        tensor = block[key]
        if not torch.is_tensor(tensor):
            raise UnitError(
                f"component {component} entry {key!r} is "
                f"{type(tensor).__name__}, not a tensor")
        if tuple(tensor.shape) != expected[key]:
            raise UnitError(
                f"component {component} entry {key!r}: shape {tuple(tensor.shape)} "
                f"vs expected {expected[key]}")
        if tensor.dtype != dtype:
            raise UnitError(
                f"component {component} entry {key!r}: dtype {tensor.dtype} "
                f"vs expected {dtype}")


def _dtype(schema: CanonicalComponentSchema) -> torch.dtype:
    return getattr(torch, schema.lora.dtype.replace("torch.", ""))
