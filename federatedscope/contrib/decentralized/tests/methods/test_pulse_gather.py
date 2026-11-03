"""PULSE's component gather copies one row of each A and one column of each B, not whole factors.

Cloning both whole factors to keep one row and one column would copy k x rank x (in + out)
values for k components. The tests compare the gather with a reference that does exactly that
and check that the results are bit-identical, share no storage with the live factors, and that
one gather clones only sum(in + out) values over the modules.
"""
import pytest
import torch

from federatedscope.contrib.decentralized.lora.adapter_bridge import AdapterBridge
from federatedscope.contrib.decentralized.methods.pulse_lora.components import SIDE_A, SIDE_B
from federatedscope.contrib.decentralized.methods.pulse_lora.state import PulsePeerState, _slot
from federatedscope.contrib.decentralized.tests.fixtures import toy_component_schema, build_toy_lora_model

SEED = 1234


def trained_state(rank, r_max, seed=0):
    """A peer whose factors all hold distinct, non-initial values (B is zero at init)."""
    model = build_toy_lora_model(rank=rank, seed=seed)
    bridge = AdapterBridge(model)
    generator = torch.Generator().manual_seed(seed + 7)
    with torch.no_grad():
        for key in bridge.schema.payload_keys(False):
            param = bridge._params[key]
            param.copy_(torch.randn(param.shape, generator=generator, dtype=param.dtype))
    schema = toy_component_schema(bridge.schema, r_max=r_max, seed=SEED)
    return PulsePeerState(bridge, schema), bridge


def reference_gather(state, bridge, component):
    """Reference gather: clone both whole factors, then slice and clone one row and one column."""
    position = state._active[component]
    out = {}
    for spec in bridge.schema.modules:
        read = bridge.read([spec.a_key, spec.b_key])
        out[_slot(spec.module, SIDE_A)] = read[spec.a_key][position, :].detach().clone()
        out[_slot(spec.module, SIDE_B)] = read[spec.b_key][:, position].detach().clone()
    return out


@pytest.mark.parametrize("rank", [8, 64, 200])
def test_the_gather_is_bit_identical_to_the_whole_factor_reference(rank):
    state, bridge = trained_state(rank, r_max=max(rank, 8))
    for component in state.support:
        got, want = state.gather(component), reference_gather(state, bridge, component)
        assert list(got) == list(want)
        for slot in want:
            assert got[slot].dtype == want[slot].dtype and got[slot].shape == want[slot].shape
            assert got[slot].stride() == want[slot].stride()
            assert got[slot].is_contiguous() == want[slot].is_contiguous()
            assert torch.equal(got[slot], want[slot])
            assert not got[slot].requires_grad and got[slot].grad_fn is None


@pytest.mark.parametrize("rank", [8, 64, 200])
def test_a_gathered_component_shares_no_storage_with_the_live_factors(rank):
    state, bridge = trained_state(rank, r_max=max(rank, 8))
    component = state.support[-1]
    block = state.gather(component)
    with torch.no_grad():
        for key in bridge.schema.payload_keys(False):
            bridge._params[key].add_(1.0)
    again = reference_gather(state, bridge, component)
    for slot in block:
        assert not torch.equal(block[slot], again[slot]), "the gathered block aliases the model"


@pytest.mark.parametrize("rank", [8, 64, 200])
def test_gathering_one_component_copies_one_row_and_one_column_per_module(rank, monkeypatch):
    state, bridge = trained_state(rank, r_max=max(rank, 8))
    cloned = []
    real = torch.Tensor.clone

    def counting(self, *args, **kwargs):
        cloned.append(self.numel())
        return real(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", counting)
    state.gather(state.support[0])
    monkeypatch.setattr(torch.Tensor, "clone", real)
    per_component = sum(bridge._params[s.a_key].shape[1] + bridge._params[s.b_key].shape[0]
                        for s in bridge.schema.modules)
    assert sum(cloned) == per_component, (sum(cloned), per_component)
