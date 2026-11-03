"""Per-frame payload schemas and the deterministic tensor codec."""
import pytest
import torch

from federatedscope.contrib.decentralized.lora.adapter_bridge import AdapterBridge
from federatedscope.contrib.decentralized.lora.payloads import (
    PayloadError,
    PayloadSchema,
    decode_state,
    encode_state,
)
from federatedscope.contrib.decentralized.tests.fixtures import build_toy_lora_model


def state():
    return AdapterBridge(build_toy_lora_model()).read()


def test_codec_round_trips_bitwise():
    original = state()
    restored = decode_state(encode_state(original))
    assert set(restored) == set(original)
    assert all(torch.equal(original[k], restored[k]) for k in original)


def test_encoding_is_deterministic():
    assert encode_state(state()) == encode_state(state())


def test_encoding_is_insertion_order_independent():
    original = state()
    shuffled = {k: original[k] for k in reversed(list(original))}
    assert encode_state(original) == encode_state(shuffled)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_supported_dtypes_round_trip(dtype):
    payload = {"a": torch.arange(6, dtype=dtype).reshape(2, 3)}
    restored = decode_state(encode_state(payload))
    assert restored["a"].dtype == dtype
    assert torch.equal(restored["a"], payload["a"])


def test_unsupported_dtype_is_refused_loudly():
    with pytest.raises(PayloadError, match="not supported by the wire codec"):
        encode_state({"a": torch.zeros(2, dtype=torch.bfloat16)})


def test_trailing_bytes_are_detected():
    with pytest.raises(PayloadError, match="trailing bytes"):
        decode_state(encode_state(state()) + b"\x00")


def test_payload_schema_is_per_frame_not_the_whole_lora_schema():
    bridge = AdapterBridge(build_toy_lora_model())
    with_aux = PayloadSchema.for_full_state(bridge.schema, include_auxiliary=True)
    without = PayloadSchema.for_full_state(bridge.schema, include_auxiliary=False)
    assert without.keys <= with_aux.keys
    assert without.includes_auxiliary is False


def test_payload_schema_rejects_missing_and_extra_keys():
    schema = PayloadSchema.for_full_state(
        AdapterBridge(build_toy_lora_model()).schema, include_auxiliary=False)
    good = state()
    schema.validate(good)

    short = dict(good)
    short.pop(next(iter(short)))
    with pytest.raises(PayloadError, match="missing="):
        schema.validate(short)

    long = dict(good)
    long["surprise"] = torch.zeros(1)
    with pytest.raises(PayloadError, match="unexpected="):
        schema.validate(long)
