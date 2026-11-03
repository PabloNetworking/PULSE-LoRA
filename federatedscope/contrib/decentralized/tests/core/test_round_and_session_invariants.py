"""Graph-round and pair-session invariants.

  1. A fixed-Q edge that the frame does not offer, or offers as unavailable,
     carries no bytes and moves no state.
  2. Acknowledgements reach the two endpoints of a session and no other peer.
  3. Every ledger row of a session carries the session id, and its payload rows
     the transaction id as well, so rows join to both.
  4. Event-driven compute latency is a configured duration, independent of the
     simulation clock: shifting every timestamp by a constant shifts dispatch by
     the same constant.
"""
import pytest

from federatedscope.contrib.decentralized.core.accounting import ByteLedger
from federatedscope.contrib.decentralized.core.graph_mixing import (
    SKIPPED_EDGE_IDENTITY_ROUND,
    execute_graph_round,
)
from federatedscope.contrib.decentralized.core.interactions import (
    CompilationError,
    compile_fixed_graph_round,
    compile_pair_sessions,
)
from federatedscope.contrib.decentralized.core.mixing_matrix import resolve_fixed_q
from federatedscope.contrib.decentralized.core.pair_session import execute_pair_session
from federatedscope.contrib.decentralized.core.runner import DecentralizedRunner
from federatedscope.contrib.decentralized.core.traces import load_trace
from federatedscope.contrib.decentralized.core.transaction import TransactionEngine
from federatedscope.contrib.decentralized.tests.core.test_runner import toy_cfg
from federatedscope.contrib.decentralized.tests.core.toy_method import ToyAlgorithm, ToyFactory

PATH_PEERS = [1, 2, 3]
#: path graph 1-2-3, so peer 2 has degree two and peers 1 and 3 degree one
PATH_Q = [[0.75, 0.25, 0.0], [0.25, 0.5, 0.25], [0.0, 0.25, 0.75]]


def frame_offering(edges, peers=PATH_PEERS):
    return load_trace({
        "trace_schema_version": 1, "time_unit": "ms", "duration_unit": "ms",
        "goodput_unit": "bps", "peers": peers,
        "frames": [{"index": 0, "time": 0, "edges": edges}],
    }).frame(0)


AVAILABLE_12 = {"endpoints": [1, 2], "duration": 1000, "goodput": 1_000_000}
AVAILABLE_23 = {"endpoints": [2, 3], "duration": 1000, "goodput": 1_000_000}
UNAVAILABLE_23 = {"endpoints": [2, 3], "duration": 1000, "goodput": 0, "available": False}


# --- 1. no opportunity, no mixing ----------------------------------------------

@pytest.mark.parametrize("edges,label", [
    ([AVAILABLE_12], "edge absent from the frame"),
    ([AVAILABLE_12, UNAVAILABLE_23], "edge present but available=False"),
])
def test_a_q_edge_without_an_opportunity_does_not_move_state(edges, label):
    matrix, _, _, _ = resolve_fixed_q(PATH_Q, PATH_PEERS, 0.25)
    event = compile_fixed_graph_round(frame_offering(edges), PATH_PEERS, matrix)
    algorithms = {p: ToyAlgorithm(p, float(p) * 10) for p in PATH_PEERS}
    before = {p: a.value for p, a in algorithms.items()}

    ledger = ByteLedger()
    report = execute_graph_round(event, algorithms, ledger)

    assert algorithms[3].value == before[3], f"peer 3 moved with {label}"
    assert [list(e) for e in [s["endpoints"] for s in report["skipped_edges"]]] == [[2, 3]]
    assert report["skipped_edges"][0]["reason"].startswith("no available opportunity")


@pytest.mark.parametrize("edges", [[AVAILABLE_12], [AVAILABLE_12, UNAVAILABLE_23]])
def test_the_realized_matrix_stays_valid_when_an_edge_is_unoffered(edges):
    matrix, _, _, _ = resolve_fixed_q(PATH_Q, PATH_PEERS, 0.25)
    event = compile_fixed_graph_round(frame_offering(edges), PATH_PEERS, matrix)
    report = execute_graph_round(event, {p: ToyAlgorithm(p, float(p)) for p in PATH_PEERS},
                                 ByteLedger())
    realized = report["realized_matrix"]
    for row in realized:
        assert sum(row) == pytest.approx(1.0)
    assert realized[1][2] == 0.0 and realized[2][1] == 0.0


@pytest.mark.parametrize("edges", [[AVAILABLE_12], [AVAILABLE_12, UNAVAILABLE_23]])
def test_no_bytes_are_charged_on_an_unoffered_edge(edges):
    """State and accounting must agree: no traffic, no mixing."""
    matrix, _, _, _ = resolve_fixed_q(PATH_Q, PATH_PEERS, 0.25)
    event = compile_fixed_graph_round(frame_offering(edges), PATH_PEERS, matrix)
    ledger = ByteLedger()
    execute_graph_round(event, {p: ToyAlgorithm(p, float(p)) for p in PATH_PEERS}, ledger)

    on_23 = [r for r in ledger.records if {r["sender"], r["receiver"]} == {2, 3}]
    assert not on_23, f"charged {sum(r['bytes'] for r in on_23)}B on an unoffered edge"
    assert [r for r in ledger.records if {r["sender"], r["receiver"]} == {1, 2}]


def test_state_change_agrees_with_the_realized_matrix():
    matrix, _, _, _ = resolve_fixed_q(PATH_Q, PATH_PEERS, 0.25)
    event = compile_fixed_graph_round(frame_offering([AVAILABLE_12]), PATH_PEERS, matrix)
    algorithms = {p: ToyAlgorithm(p, float(p) * 10) for p in PATH_PEERS}
    before = {p: a.value for p, a in algorithms.items()}

    report = execute_graph_round(event, algorithms, ByteLedger())
    realized = report["realized_matrix"]

    for position, pid in enumerate(event.peer_order):
        expected = sum(realized[position][j] * before[event.peer_order[j]]
                       for j in range(len(event.peer_order)))
        assert algorithms[pid].value == pytest.approx(expected), f"peer {pid}"


def test_a_fully_offered_frame_is_unaffected():
    matrix, _, _, _ = resolve_fixed_q(PATH_Q, PATH_PEERS, 0.25)
    event = compile_fixed_graph_round(
        frame_offering([AVAILABLE_12, AVAILABLE_23]), PATH_PEERS, matrix)
    assert event.unavailable_edges == ()
    report = execute_graph_round(event, {p: ToyAlgorithm(p, float(p)) for p in PATH_PEERS},
                                 ByteLedger())
    assert not report["skipped_edges"]
    assert report["realized_matrix"] == [list(row) for row in matrix]


@pytest.mark.parametrize("edges", [[AVAILABLE_12], [AVAILABLE_12, UNAVAILABLE_23]])
def test_strict_mode_rejects_an_incomplete_frame(edges):
    matrix, _, _, _ = resolve_fixed_q(PATH_Q, PATH_PEERS, 0.25)
    with pytest.raises(CompilationError, match="no available opportunity"):
        compile_fixed_graph_round(frame_offering(edges), PATH_PEERS, matrix,
                                  require_complete_frame=True)


def test_identity_round_policy_also_covers_an_unoffered_edge():
    matrix, _, _, _ = resolve_fixed_q(PATH_Q, PATH_PEERS, 0.25)
    event = compile_fixed_graph_round(frame_offering([AVAILABLE_12]), PATH_PEERS, matrix)
    algorithms = {p: ToyAlgorithm(p, float(p)) for p in PATH_PEERS}
    report = execute_graph_round(event, algorithms, ByteLedger(),
                                 skipped_edge_policy=SKIPPED_EDGE_IDENTITY_ROUND)
    assert report["realized_matrix"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert [algorithms[p].value for p in PATH_PEERS] == [1.0, 2.0, 3.0]


# --- 2. acknowledgements reach only the endpoints --------------------------------

def four_peers():
    return {p: ToyAlgorithm(p, float(p)) for p in (1, 2, 3, 4)}


def session_event(goodput=1_000_000, first_event_id=0):
    frame = frame_offering(
        [{"endpoints": [1, 2], "duration": 1000, "goodput": goodput}], peers=[1, 2, 3, 4])
    return compile_pair_sessions(frame, first_event_id=first_event_id)[0]


@pytest.mark.parametrize("goodput,expected_status", [
    (1_000_000, "committed"),
    (200, "control_exceeds_capacity"),
])
def test_only_the_endpoints_are_acknowledged(goodput, expected_status):
    algorithms = four_peers()
    record = execute_pair_session(session_event(goodput), algorithms,
                                  TransactionEngine(), ByteLedger())
    assert record["status"] == expected_status

    assert len(algorithms[1].acknowledgements) == 1
    assert len(algorithms[2].acknowledgements) == 1
    assert algorithms[3].acknowledgements == [], "peer 3 took no part in this session"
    assert algorithms[4].acknowledgements == [], "peer 4 took no part in this session"


def test_only_the_endpoints_are_acknowledged_on_transaction_failure():
    from federatedscope.contrib.decentralized.core.transaction import FailureInjector

    algorithms = four_peers()
    record = execute_pair_session(
        session_event(), algorithms, TransactionEngine(), ByteLedger(),
        failure_injector=FailureInjector("commit", every=1))

    assert record["status"] == "failed"
    assert all(len(algorithms[p].acknowledgements) == 1 for p in (1, 2))
    assert all(algorithms[p].acknowledgements == [] for p in (3, 4))


def test_only_the_endpoints_are_acknowledged_on_a_rejected_manifest():
    algorithms = four_peers()

    def refuse(blob, session):
        raise ValueError("incompatible")

    algorithms[1].consume_manifest = refuse
    record = execute_pair_session(session_event(), algorithms,
                                  TransactionEngine(), ByteLedger())

    assert record["status"] == "control_only"
    assert all(len(algorithms[p].acknowledgements) == 1 for p in (1, 2))
    assert all(algorithms[p].acknowledgements == [] for p in (3, 4))


# --- 3. session and transaction identifiers -----------------------------------------

def desynchronised_engine(advance=7):
    engine = TransactionEngine()
    for _ in range(advance):
        engine.begin((1, 2))
    return engine


def test_every_ledger_row_joins_to_its_session_and_transaction():
    """The event and transaction counters are set apart, so a mix-up shows."""
    algorithms = four_peers()
    engine = desynchronised_engine()
    ledger = ByteLedger()
    event = session_event(first_event_id=100)

    record = execute_pair_session(event, algorithms, engine, ledger)

    assert record["session_id"] == 100
    assert record["transaction_id"] == 8, "the engine issued seven transactions before this one"
    assert record["session_id"] != record["transaction_id"]

    assert len(ledger.rows_for_session(100)) == len(ledger.records) > 0, \
        "every row must join to its session"
    payload_rows = [r for r in ledger.records if r["category"] == "payload"]
    assert ledger.rows_for_transaction(8) == payload_rows
    for row in ledger.records:
        if row["category"] == "control":
            assert row["transaction_id"] is None, \
                "control precedes the transaction and must not claim one"


def test_a_control_only_session_allocates_no_transaction():
    algorithms = four_peers()
    engine = desynchronised_engine()
    ledger = ByteLedger()

    record = execute_pair_session(session_event(goodput=200), algorithms, engine, ledger)

    assert record["status"] == "control_exceeds_capacity"
    assert record["transaction_id"] is None
    assert all(r["transaction_id"] is None for r in ledger.records)
    assert all(r["session_id"] == record["session_id"] for r in ledger.records)


def test_frames_carry_the_session_id():
    from federatedscope.contrib.decentralized.core.wire import control_frame

    frame = control_frame("manifest", 1, 2, 4242)
    assert frame.session_id == 4242
    assert not hasattr(frame, "transaction_id"), \
        "a frame is sent before any transaction exists, so it names none"


# --- 4. compute latency -------------------------------------------------------------

def latency_cfg(frames=8, shift_ms=0, base=250_000, jitter=100_000):
    return toy_cfg(frames=frames, **{
        "decentralized.execution.scheduling": "event_driven",
        "decentralized.execution.compute_latency_us": base,
        "decentralized.execution.compute_latency_jitter_us": jitter,
    })


def dispatch_times(cfg, time_shift_us=0):
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyFactory(cfg))
    if time_shift_us:
        shifted = {
            **runner.trace.normalised,
            "frames": [{**f, "time_us": f["time_us"] + time_shift_us}
                       for f in runner.trace.normalised["frames"]],
        }
        from federatedscope.contrib.decentralized.core.traces import _trace_from_normalised

        runner.trace = _trace_from_normalised(shifted)
        runner.coordinator.trace = runner.trace
    seen = []
    original = runner._dispatch

    def spy(event, algorithms, steps):
        seen.append(event.time_us)
        return original(event, algorithms, steps)

    runner._dispatch = spy
    runner.run()
    return seen, runner


def test_latency_is_a_duration_not_an_absolute_age():
    """Shifting every timestamp by a constant must shift results by that constant."""
    shift = 10 ** 9
    base_times, base_runner = dispatch_times(latency_cfg())
    shifted_times, shifted_runner = dispatch_times(latency_cfg(), time_shift_us=shift)

    assert [t + shift for t in base_times] == shifted_times, \
        "dispatch order and spacing must be translation-invariant"
    assert [p.algorithm.value for p in base_runner.peers.values()] == \
           [p.algorithm.value for p in shifted_runner.peers.values()], \
        "learning outcome must not depend on absolute clock offset"


def test_latency_does_not_grow_with_the_run():
    """The latency stays within the base plus the jitter, however long the run."""
    cfg = latency_cfg(frames=20)
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyFactory(cfg))
    ceiling = (int(cfg.decentralized.execution.compute_latency_us)
               + int(cfg.decentralized.execution.compute_latency_jitter_us))
    for _ in range(50):
        for peer_id in runner.peer_ids:
            assert 0 <= runner._compute_latency(peer_id) <= ceiling


def test_zero_jitter_is_exactly_deterministic():
    cfg = latency_cfg(jitter=0)
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyFactory(cfg))
    assert {runner._compute_latency(p) for p in runner.peer_ids} == {250_000}


@pytest.mark.parametrize("frames", [8, 20, 40])
def test_a_long_trace_leaves_no_backlog_and_stays_monotonic(frames):
    times, runner = dispatch_times(latency_cfg(frames=frames))
    assert times == sorted(times), "dispatch must stay globally nondecreasing"
    assert len(runner.coordinator.queue) == 0, "no event may be left pending"
    episodes = {pid: peer.local_episodes for pid, peer in runner.peers.items()}
    assert set(episodes.values()) == {frames}


def test_a_latency_longer_than_the_frame_interval_still_drains():
    """Latency exceeding the interval backlogs events; they must still all fire."""
    cfg = latency_cfg(frames=10, base=2_500_000, jitter=0)  # 2.5x the 1s interval
    times, runner = dispatch_times(cfg)
    assert times == sorted(times)
    assert len(runner.coordinator.queue) == 0
    assert {p.local_episodes for p in runner.peers.values()} == {10}


@pytest.mark.parametrize("bad", [
    {"decentralized.execution.compute_latency_us": -1},
    {"decentralized.execution.compute_latency_jitter_us": -5},
])
def test_negative_latency_is_rejected(bad):
    with pytest.raises(ValueError, match="must be >= 0"):
        toy_cfg(**bad)
