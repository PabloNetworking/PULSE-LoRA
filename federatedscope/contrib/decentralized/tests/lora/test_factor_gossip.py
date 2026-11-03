"""Mixing kernels: mean preservation, contraction, and the graph/pair identity."""
import pytest
import torch

from federatedscope.contrib.decentralized.lora.mixing.factor_gossip import (
    GossipError,
    bilateral_gossip,
)
from federatedscope.contrib.decentralized.lora.mixing.full_state import MixError, weighted_mix


def a_state(scale=1.0):
    return {"x": torch.tensor([1.0, 2.0]) * scale, "y": torch.tensor([[3.0]]) * scale}


def b_state():
    return {"x": torch.tensor([9.0, -4.0]), "y": torch.tensor([[7.0]])}


@pytest.mark.parametrize("gamma", [0.1, 0.25, 0.5])
def test_gossip_preserves_the_pair_sum(gamma):
    left, right = a_state(), b_state()
    new_left = bilateral_gossip(left, right, gamma)
    new_right = bilateral_gossip(right, left, gamma)
    for key in left:
        assert torch.allclose(new_left[key] + new_right[key], left[key] + right[key])


@pytest.mark.parametrize("gamma", [0.1, 0.25, 0.5])
def test_gossip_contracts_disagreement(gamma):
    left, right = a_state(), b_state()
    before = (left["x"] - right["x"]).abs().sum()
    new_left = bilateral_gossip(left, right, gamma)
    new_right = bilateral_gossip(right, left, gamma)
    after = (new_left["x"] - new_right["x"]).abs().sum()
    assert after < before


def test_gamma_one_half_reaches_exact_agreement_in_one_step():
    left, right = a_state(), b_state()
    new_left = bilateral_gossip(left, right, 0.5)
    new_right = bilateral_gossip(right, left, 0.5)
    assert all(torch.allclose(new_left[k], new_right[k]) for k in left)


@pytest.mark.parametrize("gamma", [0.0, -0.5, 0.51, 1.0])
def test_gossip_rejects_gamma_outside_the_valid_range(gamma):
    with pytest.raises(GossipError, match="gamma"):
        bilateral_gossip(a_state(), b_state(), gamma)


def test_gossip_requires_identical_key_sets():
    partial = {"x": torch.zeros(2)}
    with pytest.raises(GossipError, match="identical key sets"):
        bilateral_gossip(a_state(), partial, 0.5)


def test_gossip_rejects_shape_mismatch():
    wrong = {"x": torch.zeros(3), "y": torch.zeros(1, 1)}
    with pytest.raises(GossipError, match="shape mismatch"):
        bilateral_gossip(a_state(), wrong, 0.5)


# --- the identity that underpins matching equivalence -------------------------

@pytest.mark.parametrize("gamma", [0.1, 0.25, 0.5])
def test_pair_block_weighted_mix_equals_bilateral_gossip(gamma):
    """From identical pre-event snapshots the two formulations must agree."""
    snapshots = {1: a_state(), 2: b_state()}

    for me, other in ((1, 2), (2, 1)):
        graph = weighted_mix({me: 1.0 - gamma, other: gamma}, snapshots)
        pairwise = bilateral_gossip(snapshots[me], snapshots[other], gamma)
        for key in graph:
            assert torch.allclose(graph[key], pairwise[key])


def test_weighted_mix_needs_every_referenced_snapshot():
    with pytest.raises(MixError, match="missing pre-round snapshots"):
        weighted_mix({1: 0.5, 2: 0.5}, {1: a_state()})


def test_weighted_mix_rejects_mismatched_snapshot_keys():
    with pytest.raises(MixError, match="different key set"):
        weighted_mix({1: 0.5, 2: 0.5}, {1: a_state(), 2: {"x": torch.zeros(2)}})


def test_weighted_mix_rejects_empty_weights():
    with pytest.raises(MixError, match="no weights"):
        weighted_mix({}, {1: a_state()})
