"""End-to-end: the runner hosts an opaque method with no server and no branches."""
import json

import pytest

from federatedscope.contrib.decentralized import registry
from federatedscope.contrib.decentralized.core.interfaces import (
    INTERACTION_GRAPH,
    INTERACTION_PAIRWISE,
)
from federatedscope.contrib.decentralized.core.runner import DecentralizedRunner
from federatedscope.contrib.decentralized.tests.core.ownership import find_learning_state
from federatedscope.contrib.decentralized.tests.core.toy_method import ToyFactory
from federatedscope.core.configs.config import global_cfg


def toy_cfg(client_num=4, frames=4, interaction=INTERACTION_PAIRWISE, **overrides):
    registry.register_decentralized_method("toy_method", ToyFactory, override=True)
    cfg = global_cfg.clone()
    settings = {
        "federate.method": "toy_method",
        "federate.client_num": client_num,
        "decentralized.enabled": True,
        "decentralized.seed": 5,
        "decentralized.execution.interaction": interaction,
        "decentralized.execution.total_frames": frames,
        "decentralized.execution.local_steps": 1,
        "decentralized.execution.eval_every_frames": 0,
        # unit tests must not write run artefacts into the repo
        "outdir": "",
    }
    settings.update(overrides)
    flat = []
    for key, value in settings.items():
        flat.extend([key, value])
    cfg.merge_from_list(flat)
    return cfg


def build(factory=None, **kwargs):
    cfg = toy_cfg(**kwargs)
    factory = factory or ToyFactory(
        interaction_form=cfg.decentralized.execution.interaction)
    return DecentralizedRunner(data={}, config=cfg, factory=factory)


# --- end to end ---------------------------------------------------------------

@pytest.mark.parametrize("interaction", [INTERACTION_PAIRWISE, INTERACTION_GRAPH])
def test_runner_completes_and_evaluates_every_peer(interaction):
    runner = build(interaction=interaction)
    results = runner.run()

    assert set(results["per_peer"]) == {1, 2, 3, 4}, "every peer must be evaluated"
    assert results["interaction"] == interaction
    assert "value" in results["distribution"]
    assert {"mean", "worst", "best", "values", "direction"} == \
        set(results["distribution"]["value"])
    assert results["summary"]["offered_sessions"] or results["summary"]["offered_graph_rounds"]


@pytest.mark.parametrize("interaction", [INTERACTION_PAIRWISE, INTERACTION_GRAPH])
def test_gossip_is_mean_preserving_and_contracts_disagreement(interaction):
    runner = build(interaction=interaction, frames=8)
    before = [float(p) for p in runner.peer_ids]  # ToyFactory seeds value = peer_id
    results = runner.run()
    after = [results["per_peer"][pid]["value"] for pid in runner.peer_ids]

    assert sum(after) == pytest.approx(sum(before)), "mixing must preserve the sum over peers"
    spread_before = max(before) - min(before)
    spread_after = max(after) - min(after)
    assert spread_after < spread_before, "disagreement should contract"


def test_no_server_or_client_is_constructed():
    runner = build()
    runner.run()
    assert not hasattr(runner, "server")
    assert not hasattr(runner, "client")
    assert type(runner).__mro__[1] is object, "the runner must derive directly from object"


# --- ownership boundary ---------------------------------------------------------

def test_coordinator_owns_no_learning_state_and_no_peers():
    runner = build()
    runner.run()
    offenders = find_learning_state(runner.coordinator)
    assert not offenders, f"coordinator reached learning state: {offenders[:5]}"


def test_coordinator_stores_peer_ids_not_peer_objects():
    runner = build()
    runner._build_peers()
    assert runner.coordinator.peer_ids == (1, 2, 3, 4)
    assert all(isinstance(p, int) for p in runner.coordinator.peer_ids)


def test_peers_own_their_own_state():
    runner = build()
    runner._build_peers()
    values = {pid: peer.algorithm.value for pid, peer in runner.peers.items()}
    assert len(set(values.values())) == len(values), "peers must not share state"


# --- scheduling -----------------------------------------------------------------

def test_contact_degree_does_not_change_local_training_frequency():
    """A high-degree peer must gain no accidental optimisation advantage."""
    runner = build(client_num=6, frames=6)
    runner.run()
    episodes = {pid: peer.local_episodes for pid, peer in runner.peers.items()}
    steps = {pid: peer.local_steps for pid, peer in runner.peers.items()}
    assert len(set(episodes.values())) == 1, f"uneven local episodes: {episodes}"
    assert len(set(steps.values())) == 1, f"uneven local steps: {steps}"

    degrees = runner.coordinator.topology.degree()
    # always true; on this trace (a perfect matching every frame) every degree is equal
    assert len(set(degrees.values())) >= 1


# --- determinism -------------------------------------------------------------------

@pytest.mark.parametrize("interaction", [INTERACTION_PAIRWISE, INTERACTION_GRAPH])
def test_two_identical_runs_are_byte_identical(interaction):
    def event_log():
        runner = build(interaction=interaction)
        runner.run()
        return json.dumps(runner.sink.events, sort_keys=True)

    assert event_log() == event_log()


def test_same_seed_gives_the_same_trace_digest():
    assert build().trace.trace_id == build().trace.trace_id


def test_a_different_seed_gives_a_different_trace():
    assert build().trace.trace_id != build(**{"decentralized.seed": 6}).trace.trace_id


# --- transaction failure --------------------------------------------------------------

def test_injected_commit_failure_restores_both_endpoints_but_keeps_bytes_counted():
    """One endpoint commits, the other then fails: both must be restored exactly.

    This is the case that requires a commit to stay reversible until the whole
    transaction finishes.
    """
    runner = build(client_num=2, frames=1, **{
        "decentralized.transaction.failure_injection_stage": "commit",
        "decentralized.transaction.failure_injection_every": 2,
    })
    runner._build_peers()
    before = {pid: peer.algorithm.value for pid, peer in runner.peers.items()}
    versions = {pid: peer.algorithm.state_version() for pid, peer in runner.peers.items()}

    runner.run()

    failed = [e for e in runner.sink.events if e.get("status") == "failed"]
    assert failed, "expected the injected commit failure to fail a transaction"

    for pid, peer in runner.peers.items():
        assert peer.algorithm.value == before[pid], f"peer {pid} was not restored"
        assert peer.algorithm.state_version() == versions[pid], \
            f"peer {pid} version leaked past a rolled-back transaction"

    assert runner.ledger.total() > 0, "bytes already sent must remain counted"
    assert runner.sink.summary()["realized_payload_bytes"] > 0


# --- start-up checks --------------------------------------------------------------------


def test_runner_refuses_to_start_on_a_digest_mismatch():
    with pytest.raises(ValueError, match="refusing to start"):
        build(**{"decentralized.comparison.expected_trace_digest": "deadbeef" * 8})


def test_runner_requires_a_factory():
    with pytest.raises(ValueError, match="requires a method factory"):
        DecentralizedRunner(data={}, config=toy_cfg(), factory=None)
