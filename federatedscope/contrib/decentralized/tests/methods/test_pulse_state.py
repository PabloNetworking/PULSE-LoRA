"""PULSE-LoRA canonical component space and the two-plane peer state."""
import pytest
import torch

from federatedscope.contrib.decentralized.core.interfaces import PeerContext
from federatedscope.contrib.decentralized.lora.adapter_bridge import AdapterBridge
from federatedscope.contrib.decentralized.lora.schema import build_schema
from federatedscope.contrib.decentralized.methods.pulse_lora.components import (
    SIDE_A,
    SIDE_B,
    ComponentError,
)
from federatedscope.contrib.decentralized.methods.pulse_lora.state import (
    PulsePeerState,
    StateError,
)
from federatedscope.contrib.decentralized.tests.fixtures import toy_component_schema, build_toy_lora_model

R_MAX = 8
SEED = 1234


def peer_state(rank=4, r_max=R_MAX, support=None, seed=0):
    model = build_toy_lora_model(rank=rank, seed=seed)
    bridge = AdapterBridge(model)
    schema = toy_component_schema(bridge.schema, r_max=r_max, seed=SEED)
    return PulsePeerState(bridge, schema, support=support), schema


# --- canonical component space --------------------------------------------------

def test_component_ids_are_immutable_and_span_r_max():
    _, schema = peer_state()
    assert schema.component_ids == tuple(range(R_MAX))


def test_a_component_spans_every_injected_layer():
    """Flat across the adapter, not one component per layer."""
    _, schema = peer_state()
    block = schema.initial_component(3)
    modules = [spec.module for spec in schema.modules]
    assert len(modules) == 3
    for module in modules:
        assert f"{module}|{SIDE_A}" in block
        assert f"{module}|{SIDE_B}" in block
    assert len(block) == 2 * len(modules)


def test_initialisation_is_deterministic_across_independently_built_peers():
    """Two peers that never touched p must agree on it without communicating."""
    _, first = peer_state(seed=1)
    _, second = peer_state(seed=99)
    for component in (0, 5, 7):
        left, right = first.initial_component(component), second.initial_component(component)
        assert set(left) == set(right)
        assert all(torch.equal(left[k], right[k]) for k in left)


def test_different_components_initialise_differently():
    _, schema = peer_state()
    a3 = schema.initial_component(3)
    a4 = schema.initial_component(4)
    a_key = next(k for k in a3 if k.endswith(f"|{SIDE_A}"))
    assert not torch.equal(a3[a_key], a4[a_key])


def test_an_untouched_component_contributes_nothing():
    """B is zero, so an untouched component is a genuine no-op in the forward pass."""
    _, schema = peer_state()
    block = schema.initial_component(6)
    assert all(t.abs().sum() == 0 for k, t in block.items() if k.endswith(f"|{SIDE_B}"))
    assert any(t.abs().sum() > 0 for k, t in block.items() if k.endswith(f"|{SIDE_A}"))


def test_identity_is_independent_of_a_peers_active_rank():
    """PULSE peers hold different ranks and must still share one namespace."""
    _, low = peer_state(rank=2, support=[0, 1])
    _, high = peer_state(rank=8, support=list(range(8)))
    assert low.identity_digest() == high.identity_digest()


def test_identity_changes_with_r_max_scale_or_seed():
    _, base = peer_state()
    bridge_schema = base.lora
    variants = [
        toy_component_schema(bridge_schema, r_max=16, seed=SEED),
        toy_component_schema(bridge_schema, r_max=R_MAX, seed=SEED + 1),
        toy_component_schema(bridge_schema, r_max=R_MAX, seed=SEED, scale=1.5),
    ]
    for variant in variants:
        assert variant.identity_digest() != base.identity_digest()


def test_component_cost_counts_every_layer_and_both_sides():
    _, schema = peer_state()
    expected_numel = sum(s.in_features + s.out_features for s in schema.modules)
    assert schema.component_numel() == expected_numel
    assert schema.component_cost_bytes(identifier_bytes=4) == expected_numel * 4 + 4


def test_r_max_must_cover_the_adapter_rank():
    model = build_toy_lora_model(rank=8)
    with pytest.raises(ComponentError, match="smaller than the adapter's rank"):
        toy_component_schema(build_schema(model), r_max=4, seed=SEED)


def test_a_component_outside_p_is_refused():
    _, schema = peer_state()
    with pytest.raises(ComponentError, match="outside P"):
        schema.initial_slice(schema.modules[0].module, SIDE_A, R_MAX)


# --- the two planes ----------------------------------------------------------------

def test_initial_materialisation_equals_the_support():
    state, _ = peer_state(rank=4)
    assert state.support == (0, 1, 2, 3)
    assert state.materialised == (0, 1, 2, 3)
    assert state.relay_components == ()
    assert state.relay_memory_bytes() == 0, "untouched components must occupy nothing"


def test_support_is_a_subset_of_materialised_after_pruning():
    state, _ = peer_state(rank=4)
    state.prune([1, 2])
    assert set(state.support) <= set(state.materialised)
    assert state.support == (0, 3)
    assert state.materialised == (0, 1, 2, 3)
    state.check_invariants()


def test_the_mapping_stays_a_bijection_after_repeated_pruning():
    state, _ = peer_state(rank=8, support=list(range(8)))
    for doomed in ([7], [0], [3, 4]):
        state.prune(doomed)
        state.check_invariants()
        positions = sorted(state.compact_index(c) for c in state.support)
        assert positions == list(range(len(state.support)))
    assert state.support == (1, 2, 5, 6)


def test_a_component_never_has_two_authoritative_copies():
    state, _ = peer_state(rank=4)
    state.prune([1])
    assert 1 in state.relay_components
    assert not state.is_active(1)
    with pytest.raises(StateError, match="second authoritative copy"):
        state.materialise(0, state.gather(0))   # 0 is still active


def test_pruning_retains_the_latest_value_in_relay_storage():
    state, _ = peer_state(rank=4)
    # move the component away from its initial value first
    block = state.gather(2)
    for key in block:
        block[key] = block[key] + 1.5
    state.scatter(2, block)

    state.prune([2])

    recovered = state.gather(2)
    assert all(torch.allclose(recovered[k], block[k]) for k in block), \
        "the value at the moment of pruning must be what relay storage holds"


def test_compaction_preserves_the_surviving_components():
    state, schema = peer_state(rank=4)
    before = {c: state.gather(c) for c in (0, 3)}
    before_updates = state.bridge.effective_updates()

    state.prune([1, 2])

    for component, block in before.items():
        after = state.gather(component)
        assert all(torch.allclose(after[k], block[k]) for k in block), \
            f"survivor {component} changed under compaction"
    # B is zero on this freshly built adapter, so the effective updates are zero before
    # and after; what is checked here is that every module still reports one
    assert set(state.bridge.effective_updates()) == set(before_updates)


def test_the_fixed_scale_survives_a_rank_change():
    """peft recomputes alpha/r on rebuild; PULSE must pin it back."""
    state, schema = peer_state(rank=4)
    assert set(state.bridge.current_scales().values()) == {schema.scale}
    state.prune([1, 2])
    state.bridge.assert_fixed_scale(schema.scale)
    assert len(state.support) == 2, "rank really did change"


def test_an_inactive_component_can_be_materialised_without_activating_it():
    state, schema = peer_state(rank=4)
    incoming = schema.initial_component(6)
    for key in incoming:
        incoming[key] = incoming[key] + 0.25

    state.materialise(6, incoming)

    assert 6 in state.materialised and not state.is_active(6)
    assert state.support == (0, 1, 2, 3), "the trainable support must not change"
    assert len(state.bridge.parameters(state.bridge.schema.lora_keys())) == 6


def test_gather_falls_back_to_deterministic_initialisation():
    state, schema = peer_state(rank=4)
    assert not state.is_materialised(7)
    assert all(torch.equal(state.gather(7)[k], schema.initial_component(7)[k])
               for k in schema.initial_component(7))


def test_an_incomplete_component_block_is_refused():
    """A is never accepted without its paired B, nor for a subset of layers."""
    state, _ = peer_state(rank=4)
    block = state.gather(0)
    block.pop(next(iter(block)))
    with pytest.raises(StateError, match="incomplete"):
        state.scatter(0, block)


def test_pruning_cannot_empty_the_support():
    state, _ = peer_state(rank=4)
    with pytest.raises(StateError, match="empty support"):
        state.prune([0, 1, 2, 3])


def test_pruning_an_inactive_component_is_refused():
    state, _ = peer_state(rank=4)
    state.prune([1])
    with pytest.raises(StateError, match="not active"):
        state.prune([1])


def test_pruning_is_monotone_and_never_evicts_relay_state():
    state, _ = peer_state(rank=4)
    state.prune([1])
    state.prune([2])
    assert state.relay_components == (1, 2), "earlier relay state must survive"
    assert set(state.materialised) == {0, 1, 2, 3}


def test_state_round_trips_through_a_checkpoint():
    state, schema = peer_state(rank=4)
    state.prune([1])
    payload = state.state_dict()

    restored, _ = peer_state(rank=3, support=[0, 2, 3])
    restored.load_state_dict(payload)
    assert restored.support == state.support
    assert restored.relay_components == state.relay_components
