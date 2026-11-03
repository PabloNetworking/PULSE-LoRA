"""PULSE-LoRA local-compute controllers (``fixed``, ``adas_c``) and optimizer-state migration
at pruning."""
import pytest
import torch

from federatedscope.contrib.decentralized.lora.adapter_bridge import AdapterBridge
from federatedscope.contrib.decentralized.methods.pulse_lora.pruning import (
    AdaSCController,
    FixedSupportController,
    PruningError,
    build_controller,
    component_scores,
    low_set,
)
from federatedscope.contrib.decentralized.methods.pulse_lora.state import PulsePeerState
from federatedscope.contrib.decentralized.tests.fixtures import toy_component_schema, build_toy_lora_model


def peer_state(rank=8):
    bridge = AdapterBridge(build_toy_lora_model(rank=rank))
    schema = toy_component_schema(bridge.schema, r_max=8, seed=1234)
    return PulsePeerState(bridge, schema)


def energise(state, seed=0):
    """Give every component nonzero A and B values.

    peft initialises B to zero, so a freshly built adapter scores 0 on every
    component and no score can decrease. Pruning tests need a peer whose
    components carry trained-like values; this stands in for training.
    """
    generator = torch.Generator().manual_seed(seed)
    for component in state.support:
        block = state.gather(component)
        state.scatter(component, {
            key: torch.rand(value.shape, generator=generator) + 0.5
            for key, value in block.items()
        })
    return state


def optimizer_for(state):
    params = state.bridge.parameters(state.bridge.schema.lora_keys())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    for param in params:
        param.grad = torch.randn_like(param)
    optimizer.step()
    return optimizer


# --- scoring -----------------------------------------------------------------

def test_scores_are_summed_over_every_injected_layer():
    """A component spans all layers, so scoring one layer scores a different thing."""
    state = energise(peer_state(rank=4))
    scores = component_scores(state)
    assert set(scores) == set(state.support)

    bridge = state.bridge
    component = state.support[0]
    position = state.compact_index(component)
    manual = sum(
        float(torch.norm(bridge._params[s.a_key][position, :], p=2))
        * float(torch.norm(bridge._params[s.b_key][:, position], p=2))
        for s in bridge.schema.modules
    )
    assert scores[component] == pytest.approx(manual)
    assert len(bridge.schema.modules) == 3, "the sum really is over several layers"


def test_low_set_size_follows_the_rule():
    scores = {c: float(c) for c in range(10)}          # rank 10, ascending
    # m = r - max(rank_min, floor(r*gamma))
    assert len(low_set(scores, gamma=0.9, rank_min=1)) == 10 - 9
    assert len(low_set(scores, gamma=0.5, rank_min=1)) == 10 - 5
    assert len(low_set(scores, gamma=0.5, rank_min=8)) == 10 - 8
    assert low_set(scores, gamma=1.0, rank_min=1) == ()


def test_low_set_takes_the_lowest_scores_with_deterministic_tie_breaking():
    scores = {0: 5.0, 1: 1.0, 2: 1.0, 3: 9.0}
    assert low_set(scores, gamma=0.5, rank_min=1) == (1, 2)


def test_low_set_rejects_an_invalid_gamma():
    with pytest.raises(PruningError, match="gamma"):
        low_set({0: 1.0}, gamma=0.0, rank_min=1)


# --- fixed support --------------------------------------------------------------

def test_the_fixed_controller_never_prunes():
    """Fixed supports: the support is assigned from outside and never pruned."""
    state = peer_state(rank=4)
    controller = FixedSupportController()
    controller.before_episode(state)
    assert controller.select_for_pruning(state) == ()


# --- AdaS-LoRA-C conservative rule ------------------------------------------------

def test_a_freshly_initialised_adapter_scores_zero_and_prunes_nothing():
    """peft sets B = 0, so no score has decreased and the conservative rule prunes nothing."""
    state = peer_state(rank=4)
    assert set(component_scores(state).values()) == {0.0}
    controller = AdaSCController(gamma=0.5, rank_min=1)
    controller.before_episode(state)
    assert controller.select_for_pruning(state) == ()


def test_only_low_set_components_whose_score_decreased_are_pruned():
    """The conservative half: small-but-growing components must survive."""
    state = energise(peer_state(rank=4))
    controller = AdaSCController(gamma=0.5, rank_min=1)
    controller.before_episode(state)
    low = controller._low_set
    assert len(low) == 2

    # shrink one low-set component, grow the other
    shrink, grow = low[0], low[1]
    for component, factor in ((shrink, 0.1), (grow, 4.0)):
        block = state.gather(component)
        state.scatter(component, {k: v * factor for k, v in block.items()})

    pruned = controller.select_for_pruning(state)

    assert shrink in pruned
    assert grow not in pruned, "a component that grew must not be pruned"
    assert controller.last_decision["candidates"] == 2
    assert controller.last_decision["decreased"] == 1


def test_nothing_is_pruned_when_no_low_set_component_decreased():
    state = energise(peer_state(rank=4))
    controller = AdaSCController(gamma=0.5, rank_min=1)
    controller.before_episode(state)
    for component in controller._low_set:
        block = state.gather(component)
        state.scatter(component, {k: v * 3.0 for k, v in block.items()})
    assert controller.select_for_pruning(state) == ()


def test_the_rank_floor_bounds_pruning():
    state = energise(peer_state(rank=4))
    controller = AdaSCController(gamma=0.25, rank_min=3)
    controller.before_episode(state)
    for component in controller._low_set:
        block = state.gather(component)
        state.scatter(component, {k: v * 0.01 for k, v in block.items()})

    pruned = controller.select_for_pruning(state)

    assert len(pruned) <= len(state.support) - 3
    if controller.last_decision["decreased"] > len(pruned):
        assert controller.last_decision["truncated_by_rank_min"]


def test_an_unknown_controller_is_refused():
    with pytest.raises(PruningError, match="unknown controller"):
        build_controller("intuition")


# --- optimizer migration -----------------------------------------------------------

def test_optimizer_moments_follow_the_surviving_components():
    """Dropping the state entirely would reset momentum for components never pruned."""
    state = peer_state(rank=4)
    optimizer = optimizer_for(state)
    spec = state.bridge.schema.modules[0]
    a_param = state.bridge._params[spec.a_key]
    before = optimizer.state[a_param]["exp_avg"].clone()

    survivors = [c for c in state.support if c != 1]
    keep_rows = [state.compact_index(c) for c in survivors]
    expected = before.index_select(0, torch.tensor(keep_rows))

    state.prune([1], optimizer=optimizer)

    new_param = state.bridge._params[state.bridge.schema.modules[0].a_key]
    migrated = optimizer.state[new_param]["exp_avg"]
    assert migrated.shape[0] == len(survivors)
    assert torch.allclose(migrated, expected), \
        "surviving components must keep the moments they had"


def test_optimizer_state_matches_the_new_rank_on_both_sides():
    state = peer_state(rank=4)
    optimizer = optimizer_for(state)
    state.prune([0, 2], optimizer=optimizer)

    for spec in state.bridge.schema.modules:
        a_param = state.bridge._params[spec.a_key]
        b_param = state.bridge._params[spec.b_key]
        assert optimizer.state[a_param]["exp_avg"].shape == a_param.shape
        assert optimizer.state[b_param]["exp_avg"].shape == b_param.shape
        assert a_param.shape[0] == 2 and b_param.shape[1] == 2


def test_pruning_without_an_optimizer_is_allowed():
    state = peer_state(rank=4)
    state.prune([1])          # consensus-only peers have no optimizer at all
    assert state.support == (0, 2, 3)


def test_relay_components_receive_no_gradients_or_moments():
    """An inactive component is not in the forward pass, so it has no optimizer state."""
    state = peer_state(rank=4)
    optimizer = optimizer_for(state)
    state.prune([1], optimizer=optimizer)

    assert 1 in state.relay_components
    tracked = {id(p) for p in optimizer.state}
    live = {id(p) for p in state.bridge.parameters(state.bridge.schema.lora_keys())}
    assert tracked <= live, "optimizer state must not outlive the parameters it belongs to"

    for block in state.state_dict()["relay"].values():
        for tensor in block.values():
            assert not tensor.requires_grad
            assert tensor.device.type == "cpu", "relay storage lives on the CPU"
