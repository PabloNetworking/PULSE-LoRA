"""The write path never drops a key, and never uses a lenient load."""
import pytest
import torch

from federatedscope.contrib.decentralized.lora.adapter_bridge import AdapterBridge, BridgeError
from federatedscope.contrib.decentralized.tests.fixtures import build_toy_lora_model


def bridge(**kwargs):
    return AdapterBridge(build_toy_lora_model(**kwargs))


def test_read_returns_detached_clones():
    br = bridge()
    state = br.read()
    key = next(iter(state))
    state[key].add_(1.0)
    assert not torch.equal(state[key], br.read()[key]), "read must not alias live params"


def test_write_round_trips_exactly():
    source = bridge(seed=1)
    target = bridge(seed=2)
    payload = source.read()

    target.write(payload, expected_keys=payload.keys())

    after = target.read()
    assert all(torch.equal(payload[k], after[k]) for k in payload)


def test_missing_key_is_a_hard_error():
    br = bridge()
    state = br.read()
    state.pop(next(iter(state)))
    with pytest.raises(BridgeError, match="missing="):
        br.write(state, expected_keys=br.schema.payload_keys(True))


def test_unexpected_key_is_a_hard_error():
    br = bridge()
    state = br.read()
    state["not.a.real.key"] = torch.zeros(1)
    with pytest.raises(BridgeError, match="unexpected="):
        br.write(state, expected_keys=br.schema.payload_keys(True))


def test_shape_mismatch_is_a_hard_error():
    br = bridge()
    state = br.read()
    key = next(iter(state))
    state[key] = torch.zeros(3, 3)
    with pytest.raises(BridgeError, match="shape mismatch"):
        br.write(state, expected_keys=br.schema.payload_keys(True))


def test_nothing_is_written_when_validation_fails():
    """A rejected payload must not leave the model half-updated."""
    br = bridge(seed=1)
    before = br.read()
    bad = br.read()
    for key in bad:
        bad[key] = torch.full_like(bad[key], 9.0)
    bad.pop(next(iter(bad)))

    with pytest.raises(BridgeError):
        br.write(bad, expected_keys=br.schema.payload_keys(True))

    after = br.read()
    assert all(torch.equal(before[k], after[k]) for k in before), "partial write leaked"


def test_write_preserves_requires_grad_and_identity():
    br = bridge()
    params_before = {k: id(v) for k, v in br._params.items()}
    br.write(bridge(seed=7).read(), expected_keys=br.schema.payload_keys(True))
    assert {k: id(v) for k, v in br._params.items()} == params_before, \
        "write must copy in place, not rebind parameters"
    assert all(p.requires_grad for p in br.parameters(br.schema.lora_keys()))


def test_effective_update_uses_the_recorded_scale():
    br = bridge(rank=4)
    spec = br.schema.modules[0]
    a = br.read([spec.a_key])[spec.a_key]
    b = br.read([spec.b_key])[spec.b_key]
    expected = br.schema.scaling * (b @ a)
    assert torch.allclose(br.effective_update(spec.module), expected)


def test_bridge_does_not_call_load_state_dict():
    """GLUEAdapterModel.load_state_dict forces strict=False; we must not use it."""
    import ast
    import inspect

    source = inspect.getsource(AdapterBridge)
    tree = ast.parse(source.lstrip())
    calls = [n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert "load_state_dict" not in calls
