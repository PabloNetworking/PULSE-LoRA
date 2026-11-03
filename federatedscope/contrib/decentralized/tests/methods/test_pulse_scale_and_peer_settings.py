"""PULSE-LoRA peer state: scale, component validation, pruning, moments and per-peer settings.

- the scale the schema reports is the one the forward pass applies, on every path;
- component blocks and checkpoints must match the schema exactly (dtype, shape, keys);
- train, prune and train again works on CPU and CUDA, with moments migrated;
- clearing moments zeroes only the named components' slices;
- the canonical A rows lie inside peft's kaiming-uniform bound;
- per-peer ranks and supports are honoured, and recorded in the manifest without
  becoming comparability axes.
"""
import pytest
import torch

from federatedscope.contrib.decentralized import registry
from federatedscope.contrib.decentralized.core.interfaces import PeerContext
from federatedscope.contrib.decentralized.core.manifest import assert_comparable, build_manifest
from federatedscope.contrib.decentralized.core.runner import DecentralizedRunner
from federatedscope.contrib.decentralized.lora.adapter_bridge import AdapterBridge
from federatedscope.contrib.decentralized.methods.pulse_lora.components import (
    SIDE_A,
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

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def state_at(rank=4, device=None):
    model = build_toy_lora_model(rank=rank)
    if device is not None:
        model = model.to(device)
    bridge = AdapterBridge(model)
    return PulsePeerState(bridge, toy_component_schema(bridge.schema, r_max=8, seed=1234))


def energise(state, seed=0):
    generator = torch.Generator().manual_seed(seed)
    for component in state.support:
        block = state.gather(component)
        state.scatter(component, {
            k: (torch.rand(v.shape, generator=generator) + 0.5).to(v.dtype).to(v.device)
            for k, v in block.items()})
    return state


def train_step(state, device=None, seed=0):
    """A real forward/backward/step over the compact adapter."""
    params = state.bridge.parameters(state.bridge.schema.lora_keys())
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = torch.randn(4, 8, generator=generator)
    if device is not None:
        inputs = inputs.to(device)
    loss = state.bridge.model(inputs).pow(2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return optimizer


# --- one authoritative live scale ------------------------------------------------

def test_reported_scale_equals_the_scale_the_forward_pass_applies():
    """alpha=16, rank=4, r_max=8: the model applies 2, so metadata must say 2."""
    state = state_at(rank=4)
    assert state.schema.scale == pytest.approx(2.0)
    assert state.bridge.live_scale() == pytest.approx(2.0)
    assert state.bridge.schema.scaling == pytest.approx(2.0), \
        "cached metadata must report alpha/r_max = 2, not alpha/r = 4"


@pytest.mark.parametrize("action", ["construct", "write", "prune", "resume"])
def test_scale_metadata_stays_live_on_every_path(action):
    state = energise(state_at(rank=4))
    if action == "write":
        state.scatter(0, state.gather(0))
    elif action == "prune":
        state.prune([1])
    elif action == "resume":
        payload = state.state_dict()
        state = state_at(rank=4)
        state.load_state_dict(payload)

    assert state.bridge.live_scale() == pytest.approx(state.schema.scale)
    assert state.bridge.schema.scaling == pytest.approx(state.bridge.live_scale())
    for spec in state.bridge.schema.modules:
        assert spec.scaling == pytest.approx(state.bridge.live_scale())


def test_effective_update_uses_the_live_scale():
    state = energise(state_at(rank=4))
    spec = state.bridge.schema.modules[0]
    factors = state.bridge.read([spec.a_key, spec.b_key])
    raw = factors[spec.b_key] @ factors[spec.a_key]
    assert torch.allclose(state.bridge.effective_update(spec.module),
                          state.bridge.live_scale() * raw, atol=1e-6)


# --- schema-exact component state and device safety ------------------------------

@pytest.mark.parametrize("mangle,match", [
    (lambda b: {k: v.to(torch.float64) for k, v in b.items()}, "dtype"),
    (lambda b: {k: 1.0 for k in b}, "not a tensor"),
    (lambda b: {k: v[:-1] for k, v in b.items()}, "shape"),
])
def test_a_component_block_must_match_the_schema_exactly(mangle, match):
    state = energise(state_at(rank=4))
    with pytest.raises(StateError, match=match):
        state.scatter(0, mangle(state.gather(0)))


def test_relay_storage_is_detached_cpu_and_canonical_dtype():
    state = energise(state_at(rank=4))
    state.prune([1])
    for tensor in state.state_dict()["relay"][1].values():
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"
        assert not tensor.requires_grad


@pytest.mark.parametrize("corrupt,error", [
    (lambda ck: ck["relay"].__setitem__(
        1, {k: v.to(torch.float64) for k, v in ck["relay"][1].items()}), StateError),
    (lambda ck: ck["relay"].__setitem__(99, ck["relay"][1]), ComponentError),
    (lambda ck: ck["relay"].__setitem__(
        1, {k: v for i, (k, v) in enumerate(ck["relay"][1].items()) if i}), StateError),
])
def test_a_malformed_checkpoint_is_rejected(corrupt, error):
    source = energise(state_at(rank=4))
    source.prune([1])
    payload = source.state_dict()
    corrupt(payload)
    with pytest.raises(error):
        state_at(rank=4).load_state_dict(payload)


def test_train_prune_train_again_on_cpu():
    """A real training step after pruning, which would raise on stale gradients or on
    optimizer moments of the wrong shape."""
    state = energise(state_at(rank=4))
    optimizer = train_step(state, seed=1)
    state.prune([1], optimizer=optimizer)
    state.check_invariants()

    inputs = torch.randn(4, 8, generator=torch.Generator().manual_seed(2))
    loss = state.bridge.model(inputs).pow(2).mean()
    loss.backward()
    optimizer.step()                      # must not raise on shape or staleness

    state.check_invariants()
    assert len(state.support) == 3
    state.bridge.assert_fixed_scale(state.schema.scale)


@CUDA
def test_compaction_and_moment_migration_on_cuda():
    device = torch.device("cuda:0")
    state = energise(state_at(rank=4, device=device))
    optimizer = train_step(state, device=device, seed=3)

    spec = state.bridge.schema.modules[0]
    before = optimizer.state[state.bridge._params[spec.a_key]]["exp_avg"].clone()
    survivors = [c for c in state.support if c != 1]
    expected = {c: before[state.compact_index(c)].clone() for c in survivors}

    state.prune([1], optimizer=optimizer)     # index tensors must be on the GPU

    migrated = optimizer.state[
        state.bridge._params[state.bridge.schema.modules[0].a_key]]["exp_avg"]
    assert migrated.device.type == "cuda"
    for component in survivors:
        assert torch.equal(migrated[state.compact_index(component)], expected[component])
    for tensor in state.state_dict()["relay"][1].values():
        assert tensor.device.type == "cpu", "relay must come back to the host"


@CUDA
def test_train_prune_train_again_on_cuda():
    device = torch.device("cuda:0")
    state = energise(state_at(rank=4, device=device))
    optimizer = train_step(state, device=device, seed=4)
    state.prune([1], optimizer=optimizer)

    inputs = torch.randn(4, 8, generator=torch.Generator().manual_seed(5)).to(device)
    state.bridge.model(inputs).pow(2).mean().backward()
    optimizer.step()
    state.check_invariants()


# --- exact moment clearing ---------------------------------------------------------

def test_only_the_named_components_slices_are_zeroed():
    state = energise(state_at(rank=4))
    optimizer = train_step(state, seed=6)

    class Holder:
        pass

    peer = Holder()
    spec = state.bridge.schema.modules[0]
    a_param = state.bridge._params[spec.a_key]
    before = optimizer.state[a_param]["exp_avg"].clone()

    from federatedscope.contrib.decentralized.lora.optimizer_state import clear_moment_slices

    target = state.compact_index(1)
    slices = []
    for module_spec in state.bridge.schema.modules:
        slices.append((state.bridge._params[module_spec.a_key], 0, target))
        slices.append((state.bridge._params[module_spec.b_key], 1, target))
    touched = clear_moment_slices(optimizer, slices)

    after = optimizer.state[a_param]["exp_avg"]
    assert touched == len(slices) * 2, "both first and second moments per slice"
    assert float(after[target].abs().sum()) == 0.0
    for row in range(after.shape[0]):
        if row != target:
            assert torch.equal(after[row], before[row]), "unselected rows must be intact"


def test_the_peer_api_clears_by_component_identity():
    from federatedscope.contrib.decentralized.tests.methods.test_pulse_peer import make_peer

    peer = make_peer(1)
    energise(peer.state)
    optimizer = train_step(peer.state, seed=7)

    class Trainer:
        pass

    trainer = Trainer()
    trainer.optimizer = optimizer
    peer.trainer = trainer

    spec = peer.bridge.schema.modules[0]
    a_param = peer.bridge._params[spec.a_key]
    before = optimizer.state[a_param]["exp_avg"].clone()
    target = peer.state.compact_index(2)

    touched = peer.clear_moments_for([2])

    after = optimizer.state[a_param]["exp_avg"]
    assert touched > 0
    assert float(after[target].abs().sum()) == 0.0
    for row in range(after.shape[0]):
        if row != target:
            assert torch.equal(after[row], before[row])


def test_a_relay_only_component_creates_no_optimizer_state():
    from federatedscope.contrib.decentralized.tests.methods.test_pulse_peer import make_peer

    peer = make_peer(1)
    energise(peer.state)
    optimizer = train_step(peer.state, seed=8)

    class Trainer:
        pass

    trainer = Trainer()
    trainer.optimizer = optimizer
    peer.trainer = trainer
    peer.state.prune([1], optimizer=optimizer)

    assert peer.clear_moments_for([1]) == 0, "an inactive component has nothing to clear"
    assert peer.clear_moments_for([7]) == 0, "an untouched component likewise"


# --- the deterministic initialisation bound ------------------------------------------

def test_the_a_bound_matches_pefts_kaiming_uniform():
    import math

    schema = toy_component_schema(
        AdapterBridge(build_toy_lora_model()).schema, r_max=8, seed=1234)
    spec = schema.modules[0]
    samples = torch.cat([schema.initial_slice(spec.module, SIDE_A, c)
                         for c in range(schema.r_max)])

    bound = 1.0 / math.sqrt(spec.in_features)     # kaiming_uniform_(a=sqrt(5))
    observed = float(samples.abs().max())
    assert observed <= bound + 1e-6, "samples must lie inside peft's bound"
    assert observed > bound * 0.5, "and must not be degenerately small"
    # a range sqrt(3) times wider would put the largest of these samples above this
    assert observed <= bound * math.sqrt(3) / 1.5


# --- peer-local configuration is applied ----------------------------------------------

def heterogeneous_client_cfgs():
    cfgs = CN()
    for peer_id, rank, support in ((1, 4, []), (2, 2, [1, 3])):
        section = CN()
        section.decentralized = CN()
        section.decentralized.methods = CN()
        section.decentralized.methods.pulse_lora = CN()
        section.decentralized.methods.pulse_lora.initial_rank = rank
        section.decentralized.methods.pulse_lora.fixed_support = support
        setattr(cfgs, f"client_{peer_id}", section)
    return cfgs


def test_peers_are_constructed_at_their_configured_ranks_and_supports():
    cfg = pulse_cfg(client_num=2, frames=1)
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyPulseFactory(cfg),
                                 client_configs=heterogeneous_client_cfgs())
    runner._build_peers()

    first, second = runner.peers[1].algorithm, runner.peers[2].algorithm
    assert first.state.support == (0, 1, 2, 3)
    assert second.state.support == (1, 3), "a non-prefix fixed support must be honoured"
    assert first.bridge.schema.rank == 4
    assert second.bridge.schema.rank == 2, "the adapter must really hold that rank"


def test_heterogeneous_peers_share_one_component_identity():
    cfg = pulse_cfg(client_num=2, frames=1)
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyPulseFactory(cfg),
                                 client_configs=heterogeneous_client_cfgs())
    runner._build_peers()
    identities = {p.algorithm.components.identity_digest() for p in runner.peers.values()}
    assert len(identities) == 1
    scales = {p.algorithm.components.scale for p in runner.peers.values()}
    assert len(scales) == 1, "the fixed scale is run-level, not per peer"


def test_a_peer_may_not_redefine_the_component_namespace():
    cfg = pulse_cfg(client_num=2, frames=1)
    rogue = CN()
    rogue.client_1 = CN()
    rogue.client_1.decentralized = CN()
    rogue.client_1.decentralized.methods = CN()
    rogue.client_1.decentralized.methods.pulse_lora = CN()
    rogue.client_1.decentralized.methods.pulse_lora.r_max = 16

    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyPulseFactory(cfg),
                                 client_configs=rogue)
    with pytest.raises(ValueError, match="disagrees with the run-level value"):
        runner._build_peers()


def test_a_support_wider_than_the_built_adapter_is_rejected():
    cfg = pulse_cfg(client_num=2, frames=1,
                    **{"decentralized.methods.pulse_lora.initial_rank": 8})
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyPulseFactory(cfg))
    with pytest.raises(ValueError, match="never grows a support"):
        runner._build_peers()


@pytest.mark.parametrize("override,expected", [
    ({"decentralized.methods.pulse_lora.fixed_support": [0, 0, 1, 2]}, "duplicates"),
    ({"decentralized.methods.pulse_lora.fixed_support": [0, 99]}, "outside"),
    ({"decentralized.methods.pulse_lora.fixed_support": [0, 1]}, "must agree"),
])
def test_an_invalid_fixed_support_is_rejected_at_config_time(override, expected):
    with pytest.raises(ValueError, match=expected):
        pulse_cfg(**override)


# --- heterogeneity is recorded, not made a comparability axis -----------------------------

def test_the_manifest_records_per_peer_applied_settings():
    cfg = pulse_cfg(client_num=2, frames=1)
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyPulseFactory(cfg),
                                 client_configs=heterogeneous_client_cfgs())
    runner._build_peers()
    manifest = build_manifest(runner)

    per_peer = manifest["peer_method_settings"]
    assert set(per_peer) == {"1", "2"}
    assert per_peer["1"]["pulse_active_rank"] == 4
    assert per_peer["2"]["pulse_active_rank"] == 2
    assert manifest["applied_method_settings"]["pulse_active_rank"] == "<heterogeneous>"
    assert manifest["applied_method_settings"]["pulse_r_max"] == 8, \
        "a value every peer agrees on is still reported concretely"


def test_method_heterogeneity_does_not_block_comparability():
    """Differing method policy is the thing under comparison, not a fairness axis."""
    cfg = pulse_cfg(client_num=2, frames=1)
    # Both runners build their peers first: before that a manifest reports
    # value_precision None, which is a comparability axis, so an unbuilt manifest
    # does not match a built one.
    homogeneous_runner = DecentralizedRunner(
        data={}, config=cfg, factory=ToyPulseFactory(cfg))
    homogeneous_runner._build_peers()
    homogeneous = build_manifest(homogeneous_runner)
    heterogeneous_runner = DecentralizedRunner(
        data={}, config=pulse_cfg(client_num=2, frames=1),
        factory=ToyPulseFactory(cfg), client_configs=heterogeneous_client_cfgs())
    heterogeneous_runner._build_peers()
    heterogeneous = build_manifest(heterogeneous_runner)

    assert homogeneous["applied_method_settings"] != \
        heterogeneous["applied_method_settings"]
    assert_comparable(homogeneous, heterogeneous)   # shared axes still match


def test_a_manifest_taken_before_the_peers_exist_reports_unknown_precision():
    """Before the peers are built the manifest reports value_precision None, so it is
    not comparable with a manifest taken after the build."""
    from federatedscope.contrib.decentralized.core.manifest import assert_comparable

    cfg = pulse_cfg(client_num=2, frames=1)
    unbuilt = DecentralizedRunner(data={}, config=cfg, factory=ToyPulseFactory(cfg))
    before = build_manifest(unbuilt)
    assert before["value_precision"] is None

    unbuilt._build_peers()
    after = build_manifest(unbuilt)
    assert after["value_precision"] == "torch.float32"

    with pytest.raises(ValueError, match="value_precision"):
        assert_comparable(before, after)
