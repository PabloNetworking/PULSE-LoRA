"""PULSE-LoRA peer restore: the whole checkpoint is validated before anything is applied.

pruned_history is fully constrained (no repeats, no active or out-of-range entries, a
relay value for every entry), and the active and relay mappings may not name one
component both as 1 and "1".

These load a pruned rank-3 checkpoint into a trained rank-4 peer, so a rejection that
resized the peer would show in every recorded quantity.
"""
import copy

import pytest
import torch

from federatedscope.contrib.decentralized.lora.adapter_bridge import BridgeError
from federatedscope.contrib.decentralized.methods.pulse_lora.components import ComponentError
from federatedscope.contrib.decentralized.methods.pulse_lora.state import StateError
from federatedscope.contrib.decentralized.tests.methods.test_pulse_peer import make_peer


def energise(peer, seed=0):
    generator = torch.Generator().manual_seed(seed)
    for component in peer.state.support:
        block = peer.state.gather(component)
        peer.state.scatter(component, {k: torch.rand(v.shape, generator=generator) + 0.5
                                       for k, v in block.items()})
    return peer


def trained_victim(seed=1):
    """A rank-4 peer that has actually trained, with live optimizer moments."""
    peer = energise(make_peer(1), seed=seed)
    params = peer.bridge.parameters(peer.bridge.schema.lora_keys())
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    inputs = torch.randn(4, 8, generator=torch.Generator().manual_seed(seed + 1))
    peer.bridge.model(inputs).pow(2).mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    class Trainer:
        pass

    trainer = Trainer()
    trainer.optimizer = optimizer
    peer.trainer = trainer
    peer._version = 7
    peer.episodes = 3
    return peer, optimizer


def pruned_checkpoint(seed=5):
    """A genuine rank-3 checkpoint: one component retired, value in relay."""
    source = energise(make_peer(1), seed=seed)
    source.state.prune([1])
    assert len(source.state.support) == 3
    assert source.state.relay_components == (1,)
    return source.state_dict()


def snapshot(peer, optimizer):
    return {
        "support": peer.state.support,
        "materialised": peer.state.materialised,
        "rank": peer.bridge.schema.rank,
        "adapter": {k: v.clone() for k, v in peer.bridge.read().items()},
        "relay": {c: {k: v.clone() for k, v in b.items()}
                  for c, b in peer.state.state_dict()["relay"].items()},
        "scale": peer.bridge.live_scale(),
        "moments": sorted(float(s["exp_avg"].norm())
                          for s in optimizer.state.values() if "exp_avg" in s),
        "version": peer._version,
        "episodes": peer.episodes,
    }


def assert_untouched(peer, optimizer, before):
    after = snapshot(peer, optimizer)
    assert after["support"] == before["support"], "S_i changed"
    assert after["materialised"] == before["materialised"], "M_i changed"
    assert after["rank"] == before["rank"], "the compact adapter was resized"
    assert set(after["adapter"]) == set(before["adapter"])
    for key, tensor in before["adapter"].items():
        assert torch.equal(after["adapter"][key], tensor), f"{key} was written"
    assert set(after["relay"]) == set(before["relay"]), "relay state changed"
    assert after["scale"] == pytest.approx(before["scale"]), "scale changed"
    assert after["moments"] == pytest.approx(before["moments"]), "optimizer moments changed"
    assert after["version"] == before["version"], "version counter changed"
    assert after["episodes"] == before["episodes"], "episode counter changed"


# --- a rejected checkpoint must not leave the peer resized -----------------------

def drop_one(adapter):
    return {k: v for i, (k, v) in enumerate(sorted(adapter.items())) if i}


def add_one(adapter):
    return dict(adapter, surprise=torch.zeros(2))


def make_non_tensor(adapter):
    out = dict(adapter)
    out[sorted(out)[0]] = 1.0
    return out


def wrong_shape(adapter):
    out = dict(adapter)
    key = sorted(out)[0]
    out[key] = torch.zeros(9, 9)
    return out


def wrong_dtype(adapter):
    return {k: v.to(torch.float64) for k, v in adapter.items()}


@pytest.mark.parametrize("mangle,match", [
    (drop_one, "missing="),
    (add_one, "unexpected="),
    (make_non_tensor, "not a tensor"),
    (wrong_shape, "shape mismatch"),
    (wrong_dtype, "dtype mismatch"),
])
def test_a_malformed_adapter_leaves_a_trained_peer_completely_unchanged(mangle, match):
    peer, optimizer = trained_victim()
    before = snapshot(peer, optimizer)

    payload = dict(pruned_checkpoint())
    payload["adapter"] = mangle(payload["adapter"])

    with pytest.raises(BridgeError, match=match):
        peer.load_state_dict(payload)

    assert_untouched(peer, optimizer, before)


@pytest.mark.parametrize("field,value,match", [
    ("version", "seven", "not an integer"),
    ("version", -1, "negative"),
    ("episodes", None, "not an integer"),
    ("episodes", -3, "negative"),
])
def test_an_invalid_counter_is_rejected_before_any_mutation(field, value, match):
    peer, optimizer = trained_victim()
    before = snapshot(peer, optimizer)

    payload = dict(pruned_checkpoint())
    payload[field] = value

    with pytest.raises(ValueError, match=match):
        peer.load_state_dict(payload)

    assert_untouched(peer, optimizer, before)


@pytest.mark.parametrize("record,match", [
    ("not-a-mapping", "not a mapping"),
    ({"optimizer_steps": 1, "skipped_batches": 0}, "missing 'local_episodes'"),
    ({"optimizer_steps": -1, "skipped_batches": 0, "local_episodes": 0}, "invalid"),
    ({"optimizer_steps": "x", "skipped_batches": 0, "local_episodes": 0}, "invalid"),
])
def test_an_invalid_trainer_record_is_rejected_before_any_mutation(record, match):
    peer, optimizer = trained_victim()
    before = snapshot(peer, optimizer)

    payload = dict(pruned_checkpoint())
    payload["trainer"] = record

    with pytest.raises(ValueError, match=match):
        peer.load_state_dict(payload)

    assert_untouched(peer, optimizer, before)


def test_a_bad_component_plan_is_rejected_before_any_mutation():
    peer, optimizer = trained_victim()
    before = snapshot(peer, optimizer)

    payload = copy.deepcopy(pruned_checkpoint())
    payload["pulse_state"]["active"] = {0: 0, 2: 1, 3: 5}   # not a bijection

    with pytest.raises(StateError, match="bijection"):
        peer.load_state_dict(payload)

    assert_untouched(peer, optimizer, before)


def test_a_valid_pruned_checkpoint_restores_rank_three_exactly():
    """A valid checkpoint is restored, so the validation does not refuse everything."""
    peer, optimizer = trained_victim()
    payload = pruned_checkpoint()

    peer.load_state_dict(payload)

    assert peer.state.support == (0, 2, 3)
    assert peer.state.relay_components == (1,)
    assert peer.bridge.schema.rank == 3, "the adapter really was compacted"
    peer.state.check_invariants()
    peer.bridge.assert_fixed_scale(peer.components.scale)

    for key, tensor in payload["adapter"].items():
        assert torch.equal(peer.bridge.read()[key], tensor), f"{key} was not restored"
    assert peer._version == payload["version"]
    assert peer.episodes == payload["episodes"]


def test_restoring_twice_is_idempotent():
    peer, _ = trained_victim()
    payload = pruned_checkpoint()
    peer.load_state_dict(payload)
    first = {k: v.clone() for k, v in peer.bridge.read().items()}
    peer.load_state_dict(payload)
    assert all(torch.equal(peer.bridge.read()[k], v) for k, v in first.items())
    assert peer.state.support == (0, 2, 3)


# --- pruned_history is fully constrained -------------------------------------------

@pytest.mark.parametrize("corrupt,error,match", [
    (lambda ck: ck["pulse_state"].__setitem__("pruned_history", [1, 1]),
     StateError, "repeats components"),
    (lambda ck: ck["pulse_state"].__setitem__("pruned_history", [1, 5]),
     StateError, "holds no relay value"),
    (lambda ck: ck["pulse_state"].__setitem__("pruned_history", [1, 0]),
     StateError, "also holding"),
    (lambda ck: ck["pulse_state"].__setitem__("pruned_history", [99]),
     ComponentError, "outside P"),
    (lambda ck: ck["pulse_state"]["relay"].__setitem__(
        "1", ck["pulse_state"]["relay"][1]), StateError, "colliding identifiers"),
    (lambda ck: ck["pulse_state"]["active"].__setitem__("0", 0),
     StateError, "colliding identifiers"),
])
def test_history_and_identifier_rules_reject_without_mutating(corrupt, error, match):
    peer, optimizer = trained_victim()
    before = snapshot(peer, optimizer)

    payload = copy.deepcopy(pruned_checkpoint())
    corrupt(payload)

    with pytest.raises(error, match=match):
        peer.load_state_dict(payload)

    assert_untouched(peer, optimizer, before)


def test_a_component_cannot_be_pruned_twice():
    """The invariant the history rule encodes: pruning is monotone."""
    peer = energise(make_peer(1))
    peer.state.prune([1])
    with pytest.raises(StateError, match="not active"):
        peer.state.prune([1])
    assert peer.state.state_dict()["pruned_history"] == [1]


def test_every_history_entry_has_a_retained_relay_value():
    peer = energise(make_peer(1))
    peer.state.prune([1])
    peer.state.prune([2])
    payload = peer.state.state_dict()
    assert payload["pruned_history"] == [1, 2]
    assert set(payload["relay"]) == {1, 2}, "relay is never evicted"
