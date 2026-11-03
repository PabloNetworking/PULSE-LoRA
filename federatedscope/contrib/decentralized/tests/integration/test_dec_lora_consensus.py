"""Consensus trajectories must match a closed-form reference, not just 'converge'."""
import pytest
import torch

from federatedscope.contrib.decentralized.core.accounting import ByteLedger
from federatedscope.contrib.decentralized.core.interactions import compile_pair_sessions
from federatedscope.contrib.decentralized.core.pair_session import execute_pair_session
from federatedscope.contrib.decentralized.core.traces import load_trace
from federatedscope.contrib.decentralized.core.transaction import TransactionEngine
from federatedscope.contrib.decentralized.tests.methods.test_dec_lora import dec_cfg, make_peer


def frame_with(edges, peers):
    return load_trace({
        "trace_schema_version": 1, "time_unit": "ms", "duration_unit": "ms",
        "goodput_unit": "bps", "peers": sorted(peers),
        "frames": [{"index": 0, "time": 0, "edges": [
            {"endpoints": list(e), "duration": 1000, "goodput": 1_000_000} for e in edges]}],
    }).frame(0)


def run_frame(peers, edges, gamma):
    for peer in peers.values():
        peer.gamma = gamma
    engine, ledger = TransactionEngine(), ByteLedger()
    for event in compile_pair_sessions(frame_with(edges, list(peers))):
        record = execute_pair_session(event, peers, engine, ledger)
        assert record["status"] == "committed", record["reason"]
    return ledger


@pytest.mark.parametrize("gamma", [0.1, 0.3, 0.5])
def test_two_peer_trajectory_matches_the_closed_form(gamma):
    cfg = dec_cfg(auxiliary="frozen")
    peers = {1: make_peer(1, cfg), 2: make_peer(2, cfg)}
    keys = peers[1].payload_schema.keys

    x = {pid: peer.bridge.read(keys) for pid, peer in peers.items()}
    for _ in range(3):
        run_frame(peers, [(1, 2)], gamma)
        # closed form: one synchronous symmetric gossip step
        x = {
            1: {k: (1 - gamma) * x[1][k] + gamma * x[2][k] for k in keys},
            2: {k: (1 - gamma) * x[2][k] + gamma * x[1][k] for k in keys},
        }

    for pid in (1, 2):
        actual = peers[pid].bridge.read(keys)
        for key in keys:
            assert torch.allclose(actual[key], x[pid][key], atol=1e-6)


def test_three_peer_chain_matches_the_closed_form_over_alternating_frames():
    """1-2 then 2-3: peer 1's state must reach peer 3 only via peer 2."""
    cfg = dec_cfg(auxiliary="frozen")
    peers = {pid: make_peer(pid, cfg) for pid in (1, 2, 3)}
    keys = peers[1].payload_schema.keys
    gamma = 0.5

    x = {pid: peer.bridge.read(keys) for pid, peer in peers.items()}

    run_frame(peers, [(1, 2)], gamma)
    x[1], x[2] = ({k: 0.5 * x[1][k] + 0.5 * x[2][k] for k in keys},
                  {k: 0.5 * x[2][k] + 0.5 * x[1][k] for k in keys})

    run_frame(peers, [(2, 3)], gamma)
    x[2], x[3] = ({k: 0.5 * x[2][k] + 0.5 * x[3][k] for k in keys},
                  {k: 0.5 * x[3][k] + 0.5 * x[2][k] for k in keys})

    for pid in (1, 2, 3):
        actual = peers[pid].bridge.read(keys)
        for key in keys:
            assert torch.allclose(actual[key], x[pid][key], atol=1e-6)


def test_mean_is_conserved_across_the_whole_population():
    cfg = dec_cfg(auxiliary="frozen")
    peers = {pid: make_peer(pid, cfg) for pid in (1, 2, 3, 4)}
    keys = peers[1].payload_schema.keys

    def total():
        acc = None
        for peer in peers.values():
            state = peer.bridge.read(keys)
            flat = torch.cat([state[k].reshape(-1).to(torch.float64) for k in sorted(keys)])
            acc = flat if acc is None else acc + flat
        return acc

    before = total()
    run_frame(peers, [(1, 2), (3, 4)], 0.5)
    run_frame(peers, [(1, 3), (2, 4)], 0.5)
    assert torch.allclose(total(), before, atol=1e-8)


def test_unmixed_peer_is_bitwise_unchanged():
    cfg = dec_cfg(auxiliary="frozen")
    peers = {pid: make_peer(pid, cfg) for pid in (1, 2, 3)}
    keys = peers[3].payload_schema.keys
    before = peers[3].bridge.read(keys)

    run_frame(peers, [(1, 2)], 0.5)

    after = peers[3].bridge.read(keys)
    assert all(torch.equal(before[k], after[k]) for k in keys)
    assert peers[3].state_version() == 0
