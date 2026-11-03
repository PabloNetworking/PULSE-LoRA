"""Pair sessions and the pair-block graph round agree on a matching frame.

For a disjoint matching frame and coefficient gamma, a full-state bilateral
session and the compiled pair-block ``W_t`` graph update are numerically
equivalent from identical pre-event snapshots, so one trace describes the same
interactions for pairwise and graph-round methods.
"""
import pytest
import torch

from federatedscope.contrib.decentralized.core.accounting import ByteLedger
from federatedscope.contrib.decentralized.core.graph_mixing import execute_graph_round
from federatedscope.contrib.decentralized.core.interactions import (
    compile_graph_round,
    compile_pair_sessions,
)
from federatedscope.contrib.decentralized.core.pair_session import execute_pair_session
from federatedscope.contrib.decentralized.core.transaction import TransactionEngine
from federatedscope.contrib.decentralized.core.traces import load_trace
from federatedscope.contrib.decentralized.methods.dec_lora.config import (
    PROFILE_COMMON_TRACE_PAIRWISE,
    PROFILE_PAPER_FIXED_Q,
)
from federatedscope.contrib.decentralized.tests.methods.test_dec_lora import (
    dec_cfg,
    make_peer,
)

PEERS = (1, 2, 3, 4)


def matching_frame(goodput_bps=1_000_000):
    return load_trace({
        "trace_schema_version": 1, "time_unit": "ms", "duration_unit": "ms",
        "goodput_unit": "bps", "peers": list(PEERS),
        "frames": [{"index": 0, "time": 0, "edges": [
            {"endpoints": [1, 2], "duration": 1000, "goodput": goodput_bps},
            {"endpoints": [3, 4], "duration": 1000, "goodput": goodput_bps},
        ]}],
    })


def peers_for(profile, auxiliary):
    cfg = dec_cfg(profile=profile, auxiliary=auxiliary)
    return {pid: make_peer(pid, cfg) for pid in PEERS}


def adapters(peers):
    return {pid: peer.bridge.read(peer.payload_schema.keys) for pid, peer in peers.items()}


@pytest.mark.parametrize("gamma", [0.1, 0.25, 0.5])
def test_pairwise_and_graph_agree_from_identical_snapshots(gamma):
    pairwise = peers_for(PROFILE_COMMON_TRACE_PAIRWISE, "pairwise_full")
    graph = peers_for(PROFILE_PAPER_FIXED_Q, "graph_full")
    for peer in list(pairwise.values()) + list(graph.values()):
        peer.gamma = gamma

    before_pairwise = adapters(pairwise)
    before_graph = adapters(graph)
    for pid in PEERS:
        for key in before_pairwise[pid]:
            assert torch.equal(before_pairwise[pid][key], before_graph[pid][key]), \
                "the two runs must start from identical snapshots"

    frame = matching_frame().frame(0)

    engine, ledger = TransactionEngine(), ByteLedger()
    for event in compile_pair_sessions(frame):
        record = execute_pair_session(
            event, {pid: p for pid, p in pairwise.items()}, engine, ledger)
        assert record["status"] == "committed", record["reason"]

    graph_event = compile_graph_round(frame, PEERS, gamma)
    report = execute_graph_round(graph_event, graph, ByteLedger())
    assert not report["skipped_edges"]

    after_pairwise = adapters(pairwise)
    after_graph = adapters(graph)
    for pid in PEERS:
        for key in after_pairwise[pid]:
            assert torch.allclose(after_pairwise[pid][key], after_graph[pid][key],
                                  atol=1e-6), f"peer {pid} key {key} diverged"


def test_pair_sums_are_conserved_in_both_forms():
    for profile, auxiliary in ((PROFILE_COMMON_TRACE_PAIRWISE, "pairwise_full"),
                               (PROFILE_PAPER_FIXED_Q, "graph_full")):
        peers = peers_for(profile, auxiliary)
        before = adapters(peers)
        frame = matching_frame().frame(0)

        if profile == PROFILE_COMMON_TRACE_PAIRWISE:
            engine, ledger = TransactionEngine(), ByteLedger()
            for event in compile_pair_sessions(frame):
                execute_pair_session(event, peers, engine, ledger)
        else:
            execute_graph_round(compile_graph_round(frame, PEERS, 0.5), peers, ByteLedger())

        after = adapters(peers)
        for left, right in ((1, 2), (3, 4)):
            for key in after[left]:
                assert torch.allclose(after[left][key] + after[right][key],
                                      before[left][key] + before[right][key], atol=1e-6)


@pytest.mark.parametrize("goodput_bps,expected_status", [
    # costs at rank 4 on the toy adapter: 294B of control per direction,
    # 1144B for one payload frame
    (200, "control_exceeds_capacity"),   # 25B per direction: the handshake does not fit
    (8_000, "control_only"),             # 1000B per direction: handshake fits, payload does not
])
def test_insufficient_capacity_leaves_state_unchanged_and_the_trace_intact(
        goodput_bps, expected_status):
    """Offered capacity is never raised to make a session succeed.

    The two shortfalls are reported distinctly: a session that cannot complete a
    handshake is not the same outcome as one that handshakes and then finds no
    payload fits.
    """
    tiny = matching_frame(goodput_bps=goodput_bps)
    digest_before = tiny.trace_id
    frame = tiny.frame(0)

    pairwise = peers_for(PROFILE_COMMON_TRACE_PAIRWISE, "pairwise_full")
    before = adapters(pairwise)

    engine, ledger = TransactionEngine(), ByteLedger()
    for event in compile_pair_sessions(frame):
        record = execute_pair_session(event, pairwise, engine, ledger)
        assert record["status"] == expected_status, record["reason"]
        assert record["payload_bytes"] == 0

    for pid in PEERS:
        for key in before[pid]:
            assert torch.equal(adapters(pairwise)[pid][key], before[pid][key])

    assert ledger.total("control") > 0, "control that was sent stays counted"
    assert matching_frame(goodput_bps=goodput_bps).trace_id == digest_before, \
        "affordability must not alter the offered trace"


def test_graph_form_also_leaves_state_unchanged_under_insufficient_capacity():
    tiny = matching_frame(goodput_bps=200)
    graph = peers_for(PROFILE_PAPER_FIXED_Q, "graph_full")
    before = adapters(graph)

    report = execute_graph_round(compile_graph_round(tiny.frame(0), PEERS, 0.5),
                                 graph, ByteLedger())

    assert report["skipped_edges"]
    assert report["realized_matrix"] == [
        [1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    for pid in PEERS:
        for key in before[pid]:
            assert torch.equal(adapters(graph)[pid][key], before[pid][key])
