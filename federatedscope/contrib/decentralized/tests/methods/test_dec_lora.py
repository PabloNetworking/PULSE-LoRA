"""Dec-LoRA: peer construction, pairwise and graph mixing, end-to-end runs through the
shared runner, and configuration checks.

Runs on the toy adapter so the whole file stays CPU-only and needs no download. Also
defines the toy factory and config helpers that other tests import.

Run::

    python -m pytest federatedscope/contrib/decentralized/tests/methods/test_dec_lora.py
"""
import pytest
import torch

from federatedscope.contrib.decentralized import registry
from federatedscope.contrib.decentralized.core.interfaces import (
    INTERACTION_GRAPH,
    INTERACTION_PAIRWISE,
    PeerContext,
)
from federatedscope.contrib.decentralized.core.runner import DecentralizedRunner
from federatedscope.contrib.decentralized.lora.payloads import PayloadError, encode_state
from federatedscope.contrib.decentralized.methods.dec_lora import DecLoRAFactory
from federatedscope.contrib.decentralized.methods.dec_lora.config import (
    PROFILE_COMMON_TRACE_PAIRWISE,
    PROFILE_PAPER_FIXED_Q,
)
from federatedscope.contrib.decentralized.tests.fixtures import build_toy_lora_model
from federatedscope.core.configs.config import global_cfg


class ToyLoRAFactory(DecLoRAFactory):
    """Injects a toy adapter so tests need no model download."""

    def create_peer(self, context):
        context.extras["model"] = build_toy_lora_model(seed=context.peer_id)
        return super().create_peer(context)


def dec_cfg(profile=PROFILE_COMMON_TRACE_PAIRWISE, client_num=4, frames=4,
            auxiliary="local", **overrides):
    # the toy factory replaces the real one; conftest.py restores the registry
    # after each test
    registry.register_decentralized_method("dec_lora", ToyLoRAFactory, override=True)
    interaction = (INTERACTION_PAIRWISE if profile == PROFILE_COMMON_TRACE_PAIRWISE
                   else INTERACTION_GRAPH)
    cfg = global_cfg.clone()
    settings = {
        "federate.method": "dec_lora",
        "federate.client_num": client_num,
        "decentralized.enabled": True,
        "decentralized.seed": 11,
        "decentralized.execution.interaction": interaction,
        "decentralized.execution.total_frames": frames,
        "decentralized.execution.local_steps": 1,
        "decentralized.execution.eval_every_frames": 0,
        "decentralized.auxiliary_state.policy": auxiliary,
        "decentralized.methods.dec_lora.profile": profile,
        # the toy adapter is rank 4, and the configured rank is checked against the
        # adapter that gets built
        "decentralized.methods.dec_lora.rank": 4,
        # unit tests must not write run artefacts into the repo
        "outdir": "",
    }
    settings.update(overrides)
    if settings["decentralized.methods.dec_lora.profile"] == PROFILE_PAPER_FIXED_Q:
        # a paper-native fixed graph needs a real Q; there is no implicit default
        settings.setdefault("decentralized.methods.dec_lora.fixed_q", "ring")
        # a fixed graph gives peers degree > 1, so it is not a matching
        settings.setdefault("decentralized.comparison.profile", "paper_native")
    flat = []
    for key, value in settings.items():
        flat.extend([key, value])
    cfg.merge_from_list(flat)
    return cfg


def make_peer(peer_id, cfg=None, seed=None):
    cfg = cfg or dec_cfg()
    factory = ToyLoRAFactory(cfg)
    context = PeerContext(peer_id=peer_id, config=cfg,
                          extras={"model": build_toy_lora_model(seed=seed or peer_id)})
    return factory.create_peer(context)


# --- construction --------------------------------------------------------------

def test_peer_is_common_rank_with_no_method_specific_state():
    peer = make_peer(1)
    assert peer.schema.rank == 4
    # Dec-LoRA owns nothing beyond common-rank factors: no relay, no support,
    # no component ages, no selector.
    for forbidden in ("relay", "support", "component_ages", "selector", "materialised"):
        assert not hasattr(peer, forbidden), f"Dec-LoRA leaked {forbidden}"


def test_payload_excludes_auxiliary_under_local_policy():
    peer = make_peer(1, dec_cfg(auxiliary="local"))
    assert peer.includes_auxiliary is False
    assert peer.payload_schema.keys == frozenset(peer.schema.lora_keys())


def test_payload_includes_auxiliary_under_pairwise_full():
    peer = make_peer(1, dec_cfg(auxiliary="pairwise_full"))
    assert peer.includes_auxiliary is True


# --- pairwise mixing --------------------------------------------------------------

def test_bilateral_session_preserves_the_pair_sum_and_contracts_disagreement():
    left, right = make_peer(1), make_peer(2)
    before_left = left.bridge.read(left.payload_schema.keys)
    before_right = right.bridge.read(right.payload_schema.keys)
    spread_before = sum(float((before_left[k] - before_right[k]).abs().sum())
                        for k in before_left)

    token_left = left.stage(encode_state(before_right), None)
    token_right = right.stage(encode_state(before_left), None)
    left.commit(token_left)
    right.commit(token_right)
    left.discard(token_left)
    right.discard(token_right)

    after_left = left.bridge.read(left.payload_schema.keys)
    after_right = right.bridge.read(right.payload_schema.keys)

    for key in after_left:
        assert torch.allclose(after_left[key] + after_right[key],
                              before_left[key] + before_right[key], atol=1e-6)
    spread_after = sum(float((after_left[k] - after_right[k]).abs().sum())
                       for k in after_left)
    assert spread_after < spread_before


def test_rollback_restores_the_adapter_bitwise_and_the_version():
    peer = make_peer(1)
    other = make_peer(2)
    before = peer.bridge.read()
    version = peer.state_version()

    token = peer.stage(encode_state(other.bridge.read(other.payload_schema.keys)), None)
    peer.commit(token)
    assert peer.state_version() == version + 1
    peer.rollback(token)

    after = peer.bridge.read()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert peer.state_version() == version


def test_a_payload_with_the_wrong_key_set_is_rejected():
    peer = make_peer(1)
    partial = peer.bridge.read(peer.payload_schema.keys)
    partial.pop(next(iter(partial)))
    with pytest.raises(PayloadError):
        peer.stage(encode_state(partial), None)


# --- graph mixing ----------------------------------------------------------------

def test_graph_mix_equals_direct_multiplication_by_the_weights():
    cfg = dec_cfg(profile=PROFILE_PAPER_FIXED_Q, auxiliary="graph_full")
    peers = {pid: make_peer(pid, cfg) for pid in (1, 2)}
    snapshots = {pid: peer.snapshot() for pid, peer in peers.items()}
    pre = {pid: peer.bridge.read(peer.payload_schema.keys) for pid, peer in peers.items()}

    weights = {1: 0.75, 2: 0.25}
    token = peers[1].stage_mix(weights, snapshots)
    peers[1].commit(token)
    peers[1].discard(token)

    after = peers[1].bridge.read(peers[1].payload_schema.keys)
    for key in after:
        expected = 0.75 * pre[1][key] + 0.25 * pre[2][key]
        assert torch.allclose(after[key], expected, atol=1e-6)


# --- end to end ----------------------------------------------------------------------

@pytest.mark.parametrize("profile,auxiliary", [
    (PROFILE_COMMON_TRACE_PAIRWISE, "pairwise_full"),
    (PROFILE_PAPER_FIXED_Q, "graph_full"),
])
def test_dec_lora_runs_end_to_end_through_the_shared_runner(profile, auxiliary):
    cfg = dec_cfg(profile=profile, auxiliary=auxiliary, frames=4)
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyLoRAFactory(cfg))
    results = runner.run()

    assert set(results["per_peer"]) == {1, 2, 3, 4}, "every peer must be evaluated"
    assert results["method"] == "dec_lora"
    assert results["comparison_profile"] in ("common_matching_trace", "paper_native")
    assert results["total_bytes"] > 0
    assert "adapter_mean" in results["distribution"]


@pytest.mark.parametrize("profile,auxiliary", [
    (PROFILE_COMMON_TRACE_PAIRWISE, "pairwise_full"),
    (PROFILE_PAPER_FIXED_Q, "graph_full"),
])
def test_repeated_runs_are_byte_identical(profile, auxiliary):
    import json

    def once():
        cfg = dec_cfg(profile=profile, auxiliary=auxiliary)
        runner = DecentralizedRunner(data={}, config=cfg, factory=ToyLoRAFactory(cfg))
        runner.run()
        return json.dumps(runner.sink.events, sort_keys=True)

    assert once() == once()


def test_consensus_improves_over_frames():
    cfg = dec_cfg(frames=10, auxiliary="local")
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyLoRAFactory(cfg))

    runner._build_peers()
    spread_before = _spread(runner)
    runner.run()
    assert _spread(runner) < spread_before


def _spread(runner):
    means = [peer.algorithm.consensus_fingerprint()["adapter_mean"]
             for peer in runner.peers.values()]
    return max(means) - min(means)


# --- configuration ---------------------------------------------------------------------

def test_profile_and_interaction_must_agree():
    with pytest.raises(ValueError, match="needs interaction"):
        dec_cfg(profile=PROFILE_PAPER_FIXED_Q,
                **{"decentralized.execution.interaction": INTERACTION_PAIRWISE})


def test_auxiliary_policy_must_match_the_interaction_form():
    with pytest.raises(ValueError, match="does not match interaction"):
        dec_cfg(profile=PROFILE_COMMON_TRACE_PAIRWISE, auxiliary="graph_full")


@pytest.mark.parametrize("override,expected", [
    ({"decentralized.methods.dec_lora.rank": 0}, "rank must be positive"),
    ({"decentralized.methods.dec_lora.gamma": 0.9}, "gamma"),
    ({"decentralized.methods.dec_lora.profile": "handwave"}, "expected one of"),
])
def test_invalid_dec_lora_settings_are_rejected(override, expected):
    with pytest.raises(ValueError, match=expected):
        dec_cfg(**override)


def test_dec_lora_declares_no_selector_key():
    """Dec-LoRA has no selector setting; yacs rejects a key the schema does not define."""
    cfg = dec_cfg()
    with pytest.raises(Exception):
        cfg.merge_from_list(["decentralized.methods.dec_lora.selector", "oracle"])


# --- one source of truth for shared settings -------------------------------------

def test_a_method_gamma_disagreeing_with_comparison_gamma_is_rejected():
    """comparison.gamma is authoritative, so a run reports the gamma it applies."""
    with pytest.raises(ValueError, match="disagrees with the authoritative"):
        dec_cfg(**{"decentralized.comparison.gamma": 0.5,
                   "decentralized.methods.dec_lora.gamma": 0.25})


def test_the_authoritative_gamma_is_what_the_peer_applies():
    cfg = dec_cfg(**{"decentralized.comparison.gamma": 0.25,
                     "decentralized.methods.dec_lora.gamma": 0.25})
    assert ToyLoRAFactory(cfg).gamma == 0.25
    assert make_peer(1, cfg).gamma == 0.25


def test_a_configured_rank_that_does_not_match_the_adapter_is_rejected():
    """The configured rank must equal the rank of the adapter that was built."""
    cfg = dec_cfg(**{"decentralized.methods.dec_lora.rank": 64})
    with pytest.raises(ValueError, match="but the adapter that was built has rank 4"):
        make_peer(1, cfg)


def test_applied_values_are_recorded_for_the_manifest():
    from federatedscope.contrib.decentralized.core.manifest import build_manifest

    cfg = dec_cfg()
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyLoRAFactory(cfg))
    runner._build_peers()
    manifest = build_manifest(runner)

    applied = manifest["applied_method_settings"]
    assert applied["dec_lora_rank"] == 4
    assert applied["dec_lora_scaling"] == pytest.approx(4.0)
    assert manifest["skipped_edge_policy"] == "self_weight"
    assert manifest["fixed_mixing_digest"] is None, "no fixed Q on a pairwise profile"


def test_the_manifest_records_the_fixed_q_identity():
    from federatedscope.contrib.decentralized.core.manifest import build_manifest

    cfg = dec_cfg(profile=PROFILE_PAPER_FIXED_Q, auxiliary="graph_full")
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyLoRAFactory(cfg))
    runner._build_peers()
    manifest = build_manifest(runner)

    assert manifest["fixed_mixing_digest"] and len(manifest["fixed_mixing_digest"]) == 64
    assert manifest["fixed_mixing_source"] == "topology:ring:gamma=0.5"


def test_paper_fixed_q_requires_a_q_and_refuses_the_matching_label():
    with pytest.raises(ValueError, match="requires"):
        dec_cfg(profile=PROFILE_PAPER_FIXED_Q, auxiliary="graph_full",
                **{"decentralized.methods.dec_lora.fixed_q": ""})

    with pytest.raises(ValueError, match="not a matching"):
        dec_cfg(profile=PROFILE_PAPER_FIXED_Q, auxiliary="graph_full",
                **{"decentralized.comparison.profile": "common_matching_trace"})
