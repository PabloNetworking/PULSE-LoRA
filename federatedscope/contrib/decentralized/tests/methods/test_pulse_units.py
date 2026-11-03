"""Complete-component exchange units and their exact wire cost.

Run: python -m pytest federatedscope/contrib/decentralized/tests/methods/test_pulse_units.py -q -p no:cacheprovider

Per-direction capacity checks are only as exact as ``unit_cost_bytes``, so the
predicted costs are checked against the bytes ``encode_state`` actually writes.
"""
import pytest
import torch

from federatedscope.contrib.decentralized.lora.payloads import encode_state
from federatedscope.contrib.decentralized.methods.pulse_lora import units
from federatedscope.contrib.decentralized.methods.pulse_lora.components import SIDE_A, SIDE_B
from federatedscope.contrib.decentralized.methods.pulse_lora.units import UnitError
from federatedscope.contrib.decentralized.tests.methods.test_pulse_peer import (
    make_peer, pulse_cfg,
)


# -- the cost model is exact, not approximate ---------------------------------------

@pytest.mark.parametrize("count", [0, 1, 2, 3, 4])
def test_predicted_unit_cost_equals_a_real_encoding(count):
    peer = make_peer(1)
    blocks = {c: peer.state.gather(c) for c in list(peer.state.materialised)[:count]}
    blob = units.encode_units(peer.components, blocks)
    assert len(blob) == units.units_cost_bytes(peer.components, len(blocks))


def test_cost_is_linear_in_the_component_count():
    peer = make_peer(1)
    per_unit = units.unit_cost_bytes(peer.components)
    sizes = [units.units_cost_bytes(peer.components, k) for k in range(5)]
    assert [b - a for a, b in zip(sizes, sizes[1:])] == [per_unit] * 4


def test_every_component_costs_the_same_despite_differing_id_widths():
    """The zero-padded id is what makes capacity arithmetic a multiplication.

    Without it, component 9 and component 10 would encode to different sizes and
    'k components fit' would stop being well defined.
    """
    cfg = pulse_cfg(**{"decentralized.methods.pulse_lora.r_max": 16,
                       "decentralized.methods.pulse_lora.initial_rank": 4})
    peer = make_peer(1, cfg=cfg)
    sizes = set()
    for component in (0, 1, 9, 10, 15):
        block = peer.components.initial_component(component)
        sizes.add(len(units.encode_units(peer.components, {component: block})))
    assert len(sizes) == 1, f"unit size varies with the component id: {sorted(sizes)}"
    assert sizes.pop() == units.units_cost_bytes(peer.components, 1)


def test_cost_counts_every_a_and_b_slice_across_all_injected_layers():
    """The cost covers every module's A and B slices, plus keys and headers."""
    peer = make_peer(1)
    schema = peer.components
    element = torch.empty(0, dtype=torch.float32).element_size()
    payload_only = sum((spec.in_features + spec.out_features) * element
                       for spec in schema.lora.modules)
    assert len(schema.lora.modules) == 3, "the toy adapter injects three modules"
    overhead = units.unit_cost_bytes(schema) - payload_only
    assert overhead > 0, "keys and shape tags are charged too"
    # The tensor payload is exactly one component's elements.
    assert payload_only == schema.component_numel() * element


def test_cost_does_not_depend_on_the_values():
    peer = make_peer(1)
    zeros = {k: torch.zeros_like(v) for k, v in peer.state.gather(0).items()}
    wild = {k: torch.full_like(v, 1e30) for k, v in peer.state.gather(0).items()}
    assert (len(units.encode_units(peer.components, {0: zeros}))
            == len(units.encode_units(peer.components, {0: wild}))
            == units.units_cost_bytes(peer.components, 1))


# -- affordability boundaries -------------------------------------------------------

def test_affordable_units_at_the_exact_boundary_and_one_byte_short():
    peer = make_peer(1)
    schema, overhead = peer.components, 36
    one = units.units_cost_bytes(schema, 1) + overhead
    assert units.affordable_units(schema, one, overhead) == 1
    assert units.affordable_units(schema, one - 1, overhead) == 0, "one byte short"
    two = units.units_cost_bytes(schema, 2) + overhead
    assert units.affordable_units(schema, two, overhead) == 2
    assert units.affordable_units(schema, two - 1, overhead) == 1


def test_affordable_units_never_over_credits():
    peer = make_peer(1)
    schema, overhead = peer.components, 36
    for credit in range(0, units.units_cost_bytes(schema, 3) + overhead + 5):
        k = units.affordable_units(schema, credit, overhead)
        assert units.units_cost_bytes(schema, k) + overhead <= credit or k == 0


def test_zero_and_negative_credit_afford_nothing():
    peer = make_peer(1)
    assert units.affordable_units(peer.components, 0, 36) == 0
    assert units.affordable_units(peer.components, -100, 36) == 0


# -- completeness is enforced, never truncated ---------------------------------------

def test_a_round_trip_is_exact():
    peer = make_peer(1)
    blocks = {c: peer.state.gather(c) for c in (0, 2)}
    back = units.decode_units(peer.components, units.encode_units(peer.components, blocks))
    assert sorted(back) == [0, 2]
    for component, block in blocks.items():
        assert all(torch.equal(back[component][k], v) for k, v in block.items())


@pytest.mark.parametrize("drop", ["A", "B"])
def test_a_component_missing_one_factor_side_is_rejected(drop):
    """A component missing the A or the B slice of one module is rejected."""
    peer = make_peer(1)
    block = dict(peer.state.gather(0))
    key = next(k for k in block if k.endswith(f"|{drop}"))
    block.pop(key)
    with pytest.raises(UnitError, match="not a complete cross-layer block"):
        units.encode_units(peer.components, {0: block})


def test_a_component_missing_a_whole_layer_is_rejected():
    peer = make_peer(1)
    block = dict(peer.state.gather(0))
    victim = peer.components.lora.modules[1].module
    for side in (SIDE_A, SIDE_B):
        block.pop(f"{victim}|{side}")
    with pytest.raises(UnitError, match="not a complete cross-layer block"):
        units.encode_units(peer.components, {0: block})


def test_an_unexpected_key_is_rejected():
    peer = make_peer(1)
    block = dict(peer.state.gather(0))
    block["surprise|A"] = torch.zeros(4)
    with pytest.raises(UnitError, match="unexpected="):
        units.encode_units(peer.components, {0: block})


@pytest.mark.parametrize("mangle,match", [
    (lambda t: torch.zeros(t.numel() + 1, dtype=t.dtype), "shape"),
    (lambda t: t.to(torch.float64), "dtype"),
    (lambda t: 1.0, "not a tensor"),
])
def test_a_malformed_slice_is_rejected(mangle, match):
    peer = make_peer(1)
    block = dict(peer.state.gather(0))
    key = sorted(block)[0]
    block[key] = mangle(block[key])
    with pytest.raises(UnitError, match=match):
        units.encode_units(peer.components, {0: block})


def test_a_component_outside_p_is_rejected():
    peer = make_peer(1)
    with pytest.raises(UnitError, match="outside P"):
        units.encode_units(peer.components,
                           {99: peer.components.initial_component(0)})


def test_a_blob_from_a_peer_with_a_different_r_max_is_rejected():
    """Different r_max means a different id width - and a different namespace."""
    narrow = make_peer(1)
    wide = make_peer(1, cfg=pulse_cfg(**{
        "decentralized.methods.pulse_lora.r_max": 16,
        "decentralized.methods.pulse_lora.initial_rank": 4}))
    blob = units.encode_units(narrow.components, {0: narrow.state.gather(0)})
    with pytest.raises(UnitError, match="disagree about r_max"):
        units.decode_units(wide.components, blob)


def test_a_truncated_blob_is_rejected_rather_than_partly_decoded():
    peer = make_peer(1)
    blob = units.encode_units(peer.components, {0: peer.state.gather(0)})
    with pytest.raises(UnitError, match="not decodable"):
        units.decode_units(peer.components, blob[:-8])


def test_decode_rejects_a_partial_component_in_the_wire_form():
    """Completeness is re-checked on the way in, not assumed from the sender."""
    peer = make_peer(1)
    schema = peer.components
    block = peer.state.gather(0)
    flat = {units.unit_key(schema, 0, spec.module, side): block[f"{spec.module}|{side}"]
            for spec in schema.lora.modules for side in (SIDE_A, SIDE_B)}
    flat.pop(sorted(flat)[0])
    with pytest.raises(UnitError, match="not a complete cross-layer block"):
        units.decode_units(schema, encode_state(flat))
