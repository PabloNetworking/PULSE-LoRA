"""PULSE-LoRA checkpoint validation, live-scale resolution and peer overlays.

- a malformed state checkpoint is rejected before anything mutates, and planning a
  restore is separate from applying it;
- adapter keys are expected from the live model, never from the checkpoint, and a
  wrong shape or dtype is rejected rather than cast;
- every schema module resolves to exactly one live module by full name, so a scale
  disagreement in any layer is found even when layers share leaf names;
- per-peer overlays (rank_min, prune_gamma, controller, support) are validated while
  the peers are built;
- the trainer counts optimizer steps without replacing ``optimizer.step``.
"""
import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from federatedscope.contrib.decentralized.core.runner import DecentralizedRunner
from federatedscope.contrib.decentralized.lora.adapter_bridge import AdapterBridge, BridgeError
from federatedscope.contrib.decentralized.methods.pulse_lora.components import (
    ComponentError,
)
from federatedscope.contrib.decentralized.methods.pulse_lora.state import (
    PulsePeerState,
    StateError,
)
from federatedscope.contrib.decentralized.tests.fixtures import toy_component_schema, build_toy_lora_model
from federatedscope.contrib.decentralized.tests.methods.test_pulse_peer import (
    ToyPulseFactory,
    pulse_cfg,
)
from federatedscope.core.configs.config import CN


def state_at(rank=4):
    bridge = AdapterBridge(build_toy_lora_model(rank=rank))
    return PulsePeerState(bridge, toy_component_schema(bridge.schema, r_max=8, seed=1234))


def energise(state, seed=0):
    generator = torch.Generator().manual_seed(seed)
    for component in state.support:
        block = state.gather(component)
        state.scatter(component, {k: (torch.rand(v.shape, generator=generator) + 0.5)
                                  for k, v in block.items()})
    return state


def trained_optimizer(state, seed=0):
    params = state.bridge.parameters(state.bridge.schema.lora_keys())
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    inputs = torch.randn(4, 8, generator=torch.Generator().manual_seed(seed))
    state.bridge.model(inputs).pow(2).mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return optimizer


def full_snapshot(state, optimizer=None):
    return {
        "support": state.support,
        "materialised": state.materialised,
        "rank": state.bridge.schema.rank,
        "adapter": {k: v.clone() for k, v in state.bridge.read().items()},
        "relay": {c: {k: v.clone() for k, v in b.items()}
                  for c, b in state.state_dict()["relay"].items()},
        "scale": state.bridge.live_scale(),
        "moments": None if optimizer is None else sorted(
            float(s["exp_avg"].norm()) for s in optimizer.state.values() if "exp_avg" in s),
    }


def assert_untouched(state, before, optimizer=None):
    after = full_snapshot(state, optimizer)
    assert after["support"] == before["support"]
    assert after["materialised"] == before["materialised"]
    assert after["rank"] == before["rank"], "the adapter was resized by a rejected load"
    assert set(after["adapter"]) == set(before["adapter"])
    for key, tensor in before["adapter"].items():
        assert torch.equal(after["adapter"][key], tensor), f"{key} was written"
    assert set(after["relay"]) == set(before["relay"])
    assert after["scale"] == pytest.approx(before["scale"])
    if optimizer is not None:
        assert after["moments"] == pytest.approx(before["moments"])


# --- validate before mutating ----------------------------------------------------

def good_checkpoint():
    source = energise(state_at(rank=4))
    source.prune([1])
    return source.state_dict()


@pytest.mark.parametrize("corrupt,error,match", [
    (lambda ck: ck["relay"].__setitem__(
        1, {k: v.to(torch.float64) for k, v in ck["relay"][1].items()}), StateError, "dtype"),
    (lambda ck: ck["relay"].__setitem__(
        1, {k: v for i, (k, v) in enumerate(ck["relay"][1].items()) if i}),
     StateError, "incomplete"),
    (lambda ck: ck["relay"].__setitem__(99, ck["relay"][1]), ComponentError, "outside P"),
    (lambda ck: ck["relay"].__setitem__(0, ck["relay"][1]), StateError,
     "two authoritative copies"),
    (lambda ck: ck.__setitem__("active", {}), StateError, "empty or malformed"),
    (lambda ck: ck.__setitem__("active", {0: 0, 2: 0, 3: 1}), StateError, "bijection"),
    (lambda ck: ck.__setitem__("active", {0: 0, 2: 1, 3: 5}), StateError, "bijection"),
    (lambda ck: ck.__setitem__("active", {0: 0, 2: 1, 3: -1}), StateError, "negative"),
    (lambda ck: ck.__setitem__("active", {0: 0, 2: 1, 99: 2}), ComponentError, "outside"),
    (lambda ck: ck.__setitem__("active", {0: 0, 5: 1, 3: 2}), StateError, "does not hold"),
    (lambda ck: ck.__setitem__("pruned_history", [0]), StateError, "also holding"),
    (lambda ck: ck.__setitem__("pruned_history", [99]), ComponentError, "outside P"),
    (lambda ck: ck.__setitem__("identity", "nope"), StateError, "identity"),
])
def test_a_malformed_checkpoint_changes_nothing(corrupt, error, match):
    """A rejected checkpoint leaves the state, the adapter and the optimizer moments unchanged."""
    victim = energise(state_at(rank=4), seed=3)
    optimizer = trained_optimizer(victim, seed=4)
    before = full_snapshot(victim, optimizer)

    payload = good_checkpoint()
    corrupt(payload)

    with pytest.raises(error, match=match):
        victim.load_state_dict(payload, optimizer=optimizer)

    assert_untouched(victim, before, optimizer)


def test_a_valid_checkpoint_restores():
    """A valid checkpoint is restored, so the validation does not reject everything."""
    payload = good_checkpoint()
    victim = energise(state_at(rank=4))
    victim.load_state_dict(payload)
    assert victim.support == (0, 2, 3)
    assert victim.relay_components == (1,)
    victim.check_invariants()


def test_planning_is_separate_from_applying():
    victim = energise(state_at(rank=4))
    before = full_snapshot(victim)
    plan = victim.plan_restore(good_checkpoint())
    assert plan["support"] == [0, 2, 3]
    assert_untouched(victim, before)
    victim.apply_restore(plan)
    assert victim.support == (0, 2, 3)


# --- adapter validation is not self-fulfilling --------------------------------------

def test_expected_keys_come_from_the_model_not_the_payload():
    state = state_at(rank=4)
    expected = state.bridge.expected_adapter_keys()
    assert expected == state.bridge.schema.payload_keys(True)
    assert len(expected) == len(state.bridge.read())


@pytest.mark.parametrize("mangle,match", [
    (lambda full: {k: v for i, (k, v) in enumerate(full.items()) if i}, "missing="),
    (lambda full: dict(full, surprise=torch.zeros(1)), "unexpected="),
    (lambda full: {k: (1.0 if i == 0 else v) for i, (k, v) in enumerate(full.items())},
     "not a tensor"),
    (lambda full: {k: (v[:-1] if i == 0 else v) for i, (k, v) in enumerate(full.items())},
     "shape mismatch"),
    (lambda full: {k: v.to(torch.float64) for k, v in full.items()}, "dtype mismatch"),
])
def test_a_bad_adapter_checkpoint_is_rejected_and_writes_nothing(mangle, match):
    state = energise(state_at(rank=4))
    before = full_snapshot(state)
    with pytest.raises(BridgeError, match=match):
        state.bridge.write(mangle(state.bridge.read()),
                           expected_keys=state.bridge.expected_adapter_keys())
    assert_untouched(state, before)


@pytest.mark.parametrize("method_name", ["dec_lora", "pulse_lora"])
def test_both_methods_reject_a_reduced_adapter_checkpoint(method_name):
    """Neither method may let the checkpoint define its own expectation."""
    if method_name == "pulse_lora":
        from federatedscope.contrib.decentralized.tests.methods.test_pulse_peer import make_peer

        peer = make_peer(1)
        payload = peer.state_dict()
    else:
        from federatedscope.contrib.decentralized.tests.methods.test_dec_lora import make_peer

        peer = make_peer(1)
        payload = dict(peer.state_dict())

    payload["adapter"] = {k: v for i, (k, v) in enumerate(payload["adapter"].items()) if i}
    with pytest.raises(BridgeError, match="missing="):
        peer.load_state_dict(payload)


def test_pulse_expected_shapes_follow_the_checkpoint_support_rank():
    source = energise(state_at(rank=4))
    source.prune([1])
    payload = source.state_dict()

    victim = state_at(rank=4)
    victim.load_state_dict(payload)
    # after restore the contract is the rank-3 one
    for key in victim.bridge.expected_adapter_keys():
        shape = victim.bridge._params[key].shape
        assert 3 in tuple(shape), f"{key} has shape {tuple(shape)}, expected rank 3"


# --- one-to-one live-scale resolution ------------------------------------------------

class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(8, 8, bias=False)
        self.value = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.value(self.query(x))


class _Stack(nn.Module):
    """Repeated leaf names across layers, as every real transformer has."""

    def __init__(self, depth=3):
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def stacked_bridge(depth=3):
    model = get_peft_model(_Stack(depth), LoraConfig(
        r=4, lora_alpha=16, lora_dropout=0.0, target_modules=["query", "value"],
        bias="none"))
    return AdapterBridge(model)


def test_every_schema_module_resolves_to_exactly_one_live_module():
    bridge = stacked_bridge(depth=3)
    resolved = bridge.resolve_modules()
    assert len(resolved) == len(bridge.schema.modules) == 6
    assert len(set(id(m) for m in resolved.values())) == 6, "no two schema modules may alias"


def test_current_scales_reports_one_entry_per_module():
    bridge = stacked_bridge(depth=3)
    scales = bridge.current_scales()
    assert set(scales) == {spec.module for spec in bridge.schema.modules}
    assert len(scales) == 6


@pytest.mark.parametrize("layer_index", [0, 1, 2])
def test_a_scale_disagreement_in_any_layer_is_detected(layer_index):
    """Layers share leaf names, and a scale changed in any one of them is reported."""
    bridge = stacked_bridge(depth=3)
    target = dict(bridge.model.named_modules())[
        f"base_model.model.layers.{layer_index}.query"]
    target.scaling["default"] = 99.0

    with pytest.raises(BridgeError, match="disagree on scale"):
        bridge.live_scale()
    with pytest.raises(BridgeError):
        bridge.assert_fixed_scale(4.0)


def test_detection_does_not_depend_on_traversal_order():
    """A scale changed in the first layer is reported even though later layers share
    its leaf name."""
    bridge = stacked_bridge(depth=4)
    first = dict(bridge.model.named_modules())["base_model.model.layers.0.value"]
    first.scaling["default"] = 0.125
    observed = sorted(set(bridge.current_scales().values()))
    assert observed == [0.125, 4.0], f"the altered layer must appear: {observed}"


def test_an_unresolvable_schema_module_is_an_error():
    bridge = stacked_bridge(depth=2)
    import dataclasses

    bogus = dataclasses.replace(bridge.schema.modules[0], module="nowhere.at.all")
    bridge.schema = dataclasses.replace(
        bridge.schema, modules=(bogus,) + bridge.schema.modules[1:])
    with pytest.raises(BridgeError, match="missing="):
        bridge.resolve_modules()


# --- peer overlays are validated at construction ---------------------------------------

def overlay(**fields):
    cfgs = CN()
    cfgs.client_1 = CN()
    cfgs.client_1.decentralized = CN()
    cfgs.client_1.decentralized.methods = CN()
    cfgs.client_1.decentralized.methods.pulse_lora = CN()
    for key, value in fields.items():
        setattr(cfgs.client_1.decentralized.methods.pulse_lora, key, value)
    return cfgs


@pytest.mark.parametrize("fields,match", [
    ({"rank_min": 0}, "rank_min must be at least 1"),
    ({"prune_gamma": 1.5}, "prune_gamma"),
    ({"controller": "vibes"}, "expected one of"),
    ({"fixed_support": [0, 1], "initial_rank": 4}, "must agree"),
    ({"fixed_support": [0, 99], "initial_rank": 2}, "outside"),
    ({"fixed_support": [1, 1], "initial_rank": 2}, "duplicates"),
    ({"rank_min": 6, "initial_rank": 4}, "exceeds initial_rank"),
])
def test_an_invalid_peer_overlay_fails_while_peers_are_built(fields, match):
    cfg = pulse_cfg(client_num=2, frames=1)
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyPulseFactory(cfg),
                                 client_configs=overlay(**fields))
    with pytest.raises(ValueError, match=match):
        runner._build_peers()


def test_the_error_names_the_offending_peer():
    cfg = pulse_cfg(client_num=2, frames=1)
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyPulseFactory(cfg),
                                 client_configs=overlay(rank_min=0))
    with pytest.raises(ValueError, match="peer 1:"):
        runner._build_peers()


def test_a_valid_overlay_builds():
    cfg = pulse_cfg(client_num=2, frames=1)
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyPulseFactory(cfg),
                                 client_configs=overlay(initial_rank=2, rank_min=1))
    runner._build_peers()
    assert len(runner.peers[1].algorithm.state.support) == 2
    assert len(runner.peers[2].algorithm.state.support) == 4


# --- optimizer step counting ----------------------------------------------------------

def test_step_counting_does_not_replace_the_optimizer_step_method():
    """Replacing step() makes torch warn that it was overridden after the scheduler
    bound to the optimizer."""
    import types

    from federatedscope.contrib.trainer.glue_decentralized_trainer import (
        GLUEDecentralizedTrainer,
    )
    from federatedscope.core.configs.config import global_cfg
    from federatedscope.core.trainers.enums import MODE

    trainer = object.__new__(GLUEDecentralizedTrainer)
    trainer._persistent_optimizer = None
    trainer._persistent_scheduler = None
    trainer.local_episodes = 0
    trainer.optimizer_steps = 0
    trainer.skipped_batches = 0
    ctx = types.SimpleNamespace(model=build_toy_lora_model(), device="cpu",
                                cur_mode=MODE.TRAIN, cfg=global_cfg.clone())
    trainer._hook_on_fit_start_init(ctx)

    optimizer = ctx.optimizer
    assert optimizer.step.__func__ is type(optimizer).step, \
        "step() must not be replaced; a post-step hook counts instead"

    for param in ctx.model.parameters():
        if param.requires_grad:
            param.grad = torch.ones_like(param)
    optimizer.step()
    optimizer.step()
    assert trainer.optimizer_steps == 2, "the hook must still count every step"
