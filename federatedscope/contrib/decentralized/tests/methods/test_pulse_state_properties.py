"""Properties of the PULSE-LoRA component space and peer state, one section per property.

The structural hash is shared across active ranks; gather and scatter are exact; an
untouched component contributes zero; the scale alpha/r_max survives every path; relay
components stay out of the forward pass and the optimizer; S_i is a subset of M_i;
pruning keeps the latest value; moments follow the surviving components; peers of
different rank share one representation; and session planning and mixing keep these
invariants. test_pulse_state.py and test_pulse_pruning.py unit-test the same mechanisms.
"""
import pathlib

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from federatedscope.contrib.decentralized.core.accounting import ByteLedger
from federatedscope.contrib.decentralized.core.interactions import compile_pair_sessions
from federatedscope.contrib.decentralized.core.traces import load_trace
from federatedscope.contrib.decentralized.core.transaction import TransactionEngine
from federatedscope.contrib.decentralized.core.pair_session import execute_pair_session
from federatedscope.contrib.decentralized.lora.adapter_bridge import AdapterBridge, BridgeError
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
from federatedscope.contrib.decentralized.tests.fixtures import toy_component_schema, ToyBackbone, build_toy_lora_model
from federatedscope.contrib.decentralized.tests.methods.test_pulse_peer import make_peer

R_MAX, SEED = 8, 1234


def state_at(rank=4, seed=0, support=None):
    bridge = AdapterBridge(build_toy_lora_model(rank=rank, seed=seed))
    schema = toy_component_schema(bridge.schema, r_max=R_MAX, seed=SEED)
    return PulsePeerState(bridge, schema, support=support)


def energise(state, seed=0):
    generator = torch.Generator().manual_seed(seed)
    for component in state.support:
        block = state.gather(component)
        state.scatter(component, {k: torch.rand(v.shape, generator=generator) + 0.5
                                  for k, v in block.items()})
    return state


def expected_scale(schema):
    return schema.scale


# =====================================================================
# structural_hash: agrees across active ranks, rejects real structural change
# =====================================================================

def test_structural_hash_agrees_across_different_active_ranks():
    hashes = {build_schema(build_toy_lora_model(rank=r)).structural_hash
              for r in (2, 4, 8)}
    assert len(hashes) == 1, "peers of different active rank must share the namespace"


def test_structural_hash_rejects_a_different_module_set():
    two = build_schema(build_toy_lora_model(target_modules=("lin_a", "lin_b")))
    three = build_schema(build_toy_lora_model(target_modules=("lin_a", "lin_b", "lin_c")))
    assert two.structural_hash != three.structural_hash


def test_structural_hash_rejects_different_dimensions():
    class Wider(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin_a = nn.Linear(16, 16, bias=False)

        def forward(self, x):
            return self.lin_a(x)

    narrow = get_peft_model(ToyBackbone(), LoraConfig(
        r=4, lora_alpha=16, lora_dropout=0.0, target_modules=["lin_a"], bias="none"))
    wide = get_peft_model(Wider(), LoraConfig(
        r=4, lora_alpha=16, lora_dropout=0.0, target_modules=["lin_a"], bias="none"))
    assert build_schema(narrow).structural_hash != build_schema(wide).structural_hash


def test_structural_hash_rejects_a_different_backbone_shape():
    class Deeper(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin_a = nn.Linear(8, 8, bias=False)
            self.lin_b = nn.Linear(8, 8, bias=False)
            self.lin_c = nn.Linear(8, 8, bias=False)
            self.lin_d = nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.lin_d(self.lin_c(self.lin_b(self.lin_a(x))))

    deeper = get_peft_model(Deeper(), LoraConfig(
        r=4, lora_alpha=16, lora_dropout=0.0,
        target_modules=["lin_a", "lin_b", "lin_c", "lin_d"], bias="none"))
    assert build_schema(deeper).structural_hash != \
        build_schema(build_toy_lora_model()).structural_hash


def test_component_ordering_is_canonical_not_insertion_dependent():
    """Ordering is by canonical module name, so it cannot depend on traversal order."""
    schema = build_schema(build_toy_lora_model())
    names = [spec.module for spec in schema.modules]
    assert names == sorted(names)


def test_the_component_namespace_inherits_that_agreement():
    low = toy_component_schema(build_schema(build_toy_lora_model(rank=2)), R_MAX, SEED)
    high = toy_component_schema(build_schema(build_toy_lora_model(rank=8)), R_MAX, SEED)
    assert low.identity_digest() == high.identity_digest()


# =====================================================================
# component extraction and restoration round-trip exactly
# =====================================================================

def test_gather_scatter_round_trips_bitwise():
    state = energise(state_at(rank=4))
    for component in state.support:
        original = state.gather(component)
        state.scatter(component, original)
        recovered = state.gather(component)
        assert all(torch.equal(original[k], recovered[k]) for k in original)


def test_a_component_carries_its_complete_cross_layer_block():
    state = state_at(rank=4)
    modules = [spec.module for spec in state.bridge.schema.modules]
    block = state.gather(0)
    assert set(block) == {f"{m}|{side}" for m in modules for side in (SIDE_A, SIDE_B)}
    for spec in state.bridge.schema.modules:
        assert block[f"{spec.module}|{SIDE_A}"].shape == (spec.in_features,)
        assert block[f"{spec.module}|{SIDE_B}"].shape == (spec.out_features,)


def test_compaction_preserves_each_survivor_scaled_contribution():
    """The scaled rank-1 contribution of every surviving component is the same (to 1e-6)
    before and after compaction, and the bridge diagnostic uses the live model scale."""
    state = energise(state_at(rank=4))
    survivors = [c for c in state.support if c != 1]
    before = {
        (component, spec.module): state.bridge.component_contribution(
            spec.module, state.compact_index(component)).clone()
        for component in survivors for spec in state.bridge.schema.modules
    }

    state.prune([1])

    assert state.bridge.live_scale() == pytest.approx(state.schema.scale)
    assert state.bridge.schema.scaling == pytest.approx(state.bridge.live_scale()), \
        "the reported scale must equal the one the forward pass applies"
    for component in survivors:
        for spec in state.bridge.schema.modules:
            after = state.bridge.component_contribution(
                spec.module, state.compact_index(component))
            assert torch.allclose(after, before[(component, spec.module)], atol=1e-6), \
                f"component {component} in {spec.module} changed under compaction"


def test_restoring_one_component_leaves_the_others_bitwise_unchanged():
    state = energise(state_at(rank=4))
    others = {c: state.gather(c) for c in state.support if c != 1}
    mutated = {k: v * -3.0 for k, v in state.gather(1).items()}

    state.scatter(1, mutated)

    for component, block in others.items():
        after = state.gather(component)
        assert all(torch.equal(after[k], block[k]) for k in block), f"component {component}"


def test_a_partial_block_is_refused_rather_than_partially_applied():
    state = energise(state_at(rank=4))
    before = state.gather(0)
    partial = dict(before)
    partial.pop(next(iter(partial)))
    with pytest.raises(StateError, match="incomplete"):
        state.scatter(0, partial)
    after = state.gather(0)
    assert all(torch.equal(before[k], after[k]) for k in before)


# =====================================================================
# an untouched component is deterministic and contributes exactly zero
# =====================================================================

def test_untouched_component_has_zero_b_and_zero_outer_product():
    schema = toy_component_schema(build_schema(build_toy_lora_model()), R_MAX, SEED)
    block = schema.initial_component(6)
    for spec in schema.modules:
        a = block[f"{spec.module}|{SIDE_A}"]
        b = block[f"{spec.module}|{SIDE_B}"]
        assert torch.count_nonzero(b) == 0, "B must be exactly zero"
        contribution = torch.outer(b, a)
        assert torch.count_nonzero(contribution) == 0, \
            "the rank-1 update this component contributes must be exactly zero"


def test_an_untouched_component_occupying_a_slot_contributes_exactly_zero():
    """Writing an untouched component's value into an active slot makes that slot
    contribute exactly zero, and each module's total update drops by what the slot
    contributed before."""
    state = energise(state_at(rank=4))
    position = state.compact_index(0)
    before_total = {m: t.clone() for m, t in state.bridge.effective_updates().items()}
    before_slot = {spec.module: state.bridge.component_contribution(spec.module, position)
                   for spec in state.bridge.schema.modules}

    state.scatter(0, state.schema.initial_component(7))

    after_total = state.bridge.effective_updates()
    for spec in state.bridge.schema.modules:
        module = spec.module
        contribution = state.bridge.component_contribution(module, position)
        assert torch.count_nonzero(contribution) == 0, "the slot must contribute nothing"
        assert torch.allclose(after_total[module],
                              before_total[module] - before_slot[module], atol=1e-6), \
            "the total must drop by exactly what that slot previously contributed"


def test_initialisation_is_deterministic_and_component_specific():
    first = toy_component_schema(build_schema(build_toy_lora_model(seed=1)), R_MAX, SEED)
    second = toy_component_schema(build_schema(build_toy_lora_model(seed=42)), R_MAX, SEED)
    for component in range(R_MAX):
        left, right = first.initial_component(component), second.initial_component(component)
        assert all(torch.equal(left[k], right[k]) for k in left)
    a_key = f"{first.modules[0].module}|{SIDE_A}"
    assert not torch.equal(first.initial_component(0)[a_key],
                           first.initial_component(1)[a_key])


# =====================================================================
# the common scale alpha/r_max survives every path
# =====================================================================

def test_scale_is_alpha_over_r_max_not_alpha_over_local_rank():
    for rank in (2, 4, 8):
        state = state_at(rank=rank)
        assert state.schema.scale == pytest.approx(16.0 / R_MAX)
        state.bridge.assert_fixed_scale(state.schema.scale)


def test_scale_survives_construction_pruning_writing_and_resume():
    state = energise(state_at(rank=4))
    scale = expected_scale(state.schema)

    state.bridge.assert_fixed_scale(scale)                      # construction

    state.scatter(0, state.gather(0))                            # writing
    state.bridge.assert_fixed_scale(scale)

    state.prune([1])                                             # pruning + rebuild
    state.bridge.assert_fixed_scale(scale)

    payload = state.state_dict()
    resumed = state_at(rank=4)
    resumed.load_state_dict(payload)                             # checkpoint resume
    resumed.bridge.assert_fixed_scale(scale)
    assert resumed.support == state.support


def test_a_peft_rebuild_that_restores_alpha_over_r_is_caught():
    """assert_fixed_scale fails when a peft layer rebuild sets the rank-dependent alpha/r."""
    state = state_at(rank=4)
    state.bridge.model.base_model.model.lin_a.update_layer("default", 2, 16, 0.0, True)
    with pytest.raises(BridgeError, match="not holding the required fixed scale"):
        state.bridge.assert_fixed_scale(state.schema.scale)


def test_peers_of_different_rank_hold_the_same_scale():
    scales = {state_at(rank=r).schema.scale for r in (2, 4, 8)}
    assert len(scales) == 1


# =====================================================================
# relay-only components get no gradients and do not reach the forward pass
# =====================================================================

def test_relay_components_are_absent_from_the_executed_adapter():
    state = energise(state_at(rank=4))
    inputs = torch.randn(3, 8, generator=torch.Generator().manual_seed(0))
    state.prune([1])
    baseline = state.bridge.model(inputs).detach().clone()

    # overwrite the relay copy with very different values
    relayed = state.gather(1)
    state.scatter(1, {k: v * 1e6 + 123.0 for k, v in relayed.items()})

    assert torch.equal(state.bridge.model(inputs), baseline), \
        "relay content must not reach the forward pass"


def test_relay_components_have_no_parameters_and_no_optimizer_state():
    state = energise(state_at(rank=4))
    params = state.bridge.parameters(state.bridge.schema.lora_keys())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    for param in params:
        param.grad = torch.randn_like(param)
    optimizer.step()

    state.prune([1], optimizer=optimizer)

    live = {id(p) for p in state.bridge.parameters(state.bridge.schema.lora_keys())}
    assert {id(p) for p in optimizer.state} <= live
    for block in state.state_dict()["relay"].values():
        for tensor in block.values():
            assert not tensor.requires_grad
            assert tensor.grad is None


def test_an_active_component_does_reach_the_forward_pass():
    """Changing an active component changes the output, so the relay test above is not vacuous."""
    state = energise(state_at(rank=4))
    inputs = torch.randn(3, 8, generator=torch.Generator().manual_seed(0))
    baseline = state.bridge.model(inputs).detach().clone()
    block = state.gather(0)
    state.scatter(0, {k: v * 5.0 for k, v in block.items()})
    assert not torch.equal(state.bridge.model(inputs), baseline)


# =====================================================================
# S_i subset of M_i across every path
# =====================================================================

def test_support_is_contained_in_materialised_across_every_path():
    state = energise(state_at(rank=4))

    def holds(label):
        assert set(state.support) <= set(state.materialised), label
        state.check_invariants()

    holds("initialisation")
    state.scatter(0, state.gather(0));                      holds("state transfer, active")
    state.materialise(6, state.schema.initial_component(6)); holds("state transfer, relay")
    state.prune([1]);                                        holds("pruning")

    payload = state.state_dict();                            holds("serialisation")
    resumed = state_at(rank=4)
    resumed.load_state_dict(payload)
    assert set(resumed.support) <= set(resumed.materialised), "restoration"
    resumed.check_invariants()


@pytest.mark.parametrize("bad,error,match", [
    (lambda s: s.prune([0, 1, 2, 3]), StateError, "empty support"),
    (lambda s: s.prune([99]), StateError, "not active"),
    (lambda s: s.materialise(0, s.gather(0)), StateError, "second authoritative copy"),
    (lambda s: s.scatter(0, {}), StateError, "incomplete"),
    (lambda s: s.scatter(99, s.gather(0)), ComponentError, "outside P"),
    (lambda s: s.restrict_initial_support([0, 0]), StateError, "duplicate"),
])
def test_invariants_hold_after_a_rejected_operation(bad, error, match):
    """A failure path must not leave the state half-changed - and must raise the
    specific error, not merely something."""
    state = energise(state_at(rank=4))
    before = (state.support, state.materialised)
    with pytest.raises(error, match=match):
        bad(state)
    assert (state.support, state.materialised) == before
    state.check_invariants()


# =====================================================================
# pruning preserves the latest trained values
# =====================================================================

def test_pruning_stores_the_value_at_the_moment_of_pruning():
    state = energise(state_at(rank=4))
    params = state.bridge.parameters(state.bridge.schema.lora_keys())
    optimizer = torch.optim.AdamW(params, lr=0.5)
    for param in params:
        param.grad = torch.randn_like(param)
    optimizer.step()                       # component 1 is trained by this step

    trained = state.gather(1)
    state.prune([1], optimizer=optimizer)

    assert all(torch.equal(state.gather(1)[k], trained[k]) for k in trained), \
        "relay must hold the trained value, not the value at construction"


def test_the_stored_value_differs_from_the_initial_one():
    """The value kept at pruning is not the initial one, so the test above is not vacuous."""
    state = state_at(rank=4)
    initial = state.gather(1)
    energise(state, seed=5)
    state.prune([1])
    assert any(not torch.equal(state.gather(1)[k], initial[k]) for k in initial)


# =====================================================================
# optimizer moments migrate to the right rebuilt parameter
# =====================================================================

def test_moments_follow_the_surviving_components_not_slot_positions():
    state = energise(state_at(rank=4))
    params = state.bridge.parameters(state.bridge.schema.lora_keys())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    for param in params:
        param.grad = torch.randn_like(param)
    optimizer.step()

    spec = state.bridge.schema.modules[0]
    before = optimizer.state[state.bridge._params[spec.a_key]]["exp_avg"].clone()
    # remember each surviving component's own moment row, by component id
    survivors = [c for c in state.support if c != 1]
    expected = {c: before[state.compact_index(c)].clone() for c in survivors}

    state.prune([1], optimizer=optimizer)

    migrated = optimizer.state[state.bridge._params[
        state.bridge.schema.modules[0].a_key]]["exp_avg"]
    for component in survivors:
        row = migrated[state.compact_index(component)]
        assert torch.equal(row, expected[component]), \
            f"component {component} got the wrong moment row after compaction"


def test_moments_are_not_reset_by_compaction():
    state = energise(state_at(rank=4))
    params = state.bridge.parameters(state.bridge.schema.lora_keys())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    for param in params:
        param.grad = torch.randn_like(param)
    optimizer.step()

    state.prune([1], optimizer=optimizer)

    for spec in state.bridge.schema.modules:
        moment = optimizer.state[state.bridge._params[spec.a_key]]["exp_avg"]
        assert torch.count_nonzero(moment) > 0, "momentum was silently reset"


def test_moment_shapes_track_the_new_rank_on_both_factor_sides():
    state = energise(state_at(rank=4))
    params = state.bridge.parameters(state.bridge.schema.lora_keys())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    for param in params:
        param.grad = torch.randn_like(param)
    optimizer.step()

    state.prune([0, 2], optimizer=optimizer)

    for spec in state.bridge.schema.modules:
        for key in (spec.a_key, spec.b_key):
            param = state.bridge._params[key]
            assert optimizer.state[param]["exp_avg"].shape == param.shape


# =====================================================================
# rank-heterogeneous peers share one namespace and one representation
# =====================================================================

def test_a_component_transfers_between_peers_of_different_rank():
    wide = energise(state_at(rank=8, seed=1), seed=1)
    narrow = state_at(rank=4, seed=2)
    assert wide.schema.identity_digest() == narrow.schema.identity_digest()

    block = wide.gather(2)                      # active on both
    narrow.scatter(2, block)
    assert all(torch.equal(narrow.gather(2)[k], block[k]) for k in block)


def test_a_component_active_on_one_peer_relays_on_another_without_reinterpretation():
    wide = energise(state_at(rank=8, seed=1), seed=1)
    narrow = state_at(rank=4, seed=2)

    block = wide.gather(6)                      # active on wide, absent on narrow
    narrow.materialise(6, block)

    assert not narrow.is_active(6) and narrow.is_materialised(6)
    assert all(torch.equal(narrow.gather(6)[k], block[k]) for k in block), \
        "the same component id must mean the same values on both peers"


def test_component_slices_have_the_same_shape_regardless_of_active_rank():
    wide, narrow = state_at(rank=8), state_at(rank=4)
    for key, tensor in wide.gather(0).items():
        assert tensor.shape == narrow.gather(0)[key].shape, \
            "a component's shape must not depend on the holder's active rank"


# =====================================================================
# sessions keep these invariants, and core/ names neither pulse nor tad_lora
# =====================================================================

def test_planning_a_payload_leaves_support_materialisation_and_scale_unchanged():
    """Planning a session payload changes neither support, materialisation nor scale."""
    peer = make_peer(1)
    support, materialised = peer.state.support, peer.state.materialised
    scale = peer.bridge.live_scale()
    peer.state.check_invariants()

    other = make_peer(2)
    event = _pair_event(1, 2, duration=1000)
    view = peer.consume_manifest(other.build_manifest(event), event)
    peer.plan_payload(event, _budget(event), view)

    assert peer.state.support == support
    assert peer.state.materialised == materialised
    assert peer.bridge.live_scale() == pytest.approx(scale)
    peer.state.check_invariants()


def test_core_names_neither_pulse_nor_tad_lora():
    """core/ is method-neutral: no core module mentions pulse or tad_lora, even in prose."""
    import pathlib

    core = pathlib.Path(__file__).resolve().parents[2].joinpath("core").resolve()
    for module in sorted(core.glob("*.py")):
        text = module.read_text().lower()
        assert "pulse" not in text, f"core/{module.name} names pulse"
        assert "tad_lora" not in text, f"core/{module.name} names tad_lora"


def test_a_committed_pair_session_keeps_support_and_the_fixed_scale():
    """A session that commits a mix leaves the support, the state invariants and the
    fixed scale intact."""
    algorithms = {1: make_peer(1), 2: make_peer(2)}
    frame = load_trace({
        "trace_schema_version": 1, "time_unit": "ms", "duration_unit": "ms",
        "goodput_unit": "bps", "peers": [1, 2],
        "frames": [{"index": 0, "time": 0, "edges": [
            {"endpoints": [1, 2], "duration": 1000, "goodput": 1_000_000}]}],
    }).frame(0)
    before = {pid: a.state.gather(0) for pid, a in algorithms.items()}

    ledger = ByteLedger()
    record = execute_pair_session(compile_pair_sessions(frame)[0], algorithms,
                                  TransactionEngine(), ledger)

    assert record["status"] == "committed"
    assert record["payload_bytes"] > 0
    assert record["transaction_id"] is not None, "a payload did open a transaction"
    for algorithm in algorithms.values():
        # Mixing changes component values only: support, the state invariants and
        # the fixed scale must hold afterwards.
        assert algorithm.state.support == (0, 1, 2, 3)
        algorithm.state.check_invariants()
        algorithm.bridge.assert_fixed_scale(algorithm.components.scale)


def _pair_event(a: int, b: int, duration: int):
    frame = load_trace({
        "trace_schema_version": 1, "time_unit": "ms", "duration_unit": "ms",
        "goodput_unit": "bps", "peers": [a, b],
        "frames": [{"index": 0, "time": 0, "edges": [
            {"endpoints": [a, b], "duration": duration, "goodput": 1_000_000}]}],
    }).frame(0)
    return compile_pair_sessions(frame)[0]


def _budget(event):
    """The session budget as the executor builds it, before the handshake is charged."""
    from federatedscope.contrib.decentralized.core.accounting import SessionBudget
    return SessionBudget(event.endpoints, event.credits)
