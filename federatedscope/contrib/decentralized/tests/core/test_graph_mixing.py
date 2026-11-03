"""Graph rounds mix from one pre-round barrier, and price the complete payload."""
import pytest

from federatedscope.contrib.decentralized.core.accounting import ByteLedger
from federatedscope.contrib.decentralized.core.graph_mixing import execute_graph_round
from federatedscope.contrib.decentralized.core.interactions import compile_graph_round
from federatedscope.contrib.decentralized.core.traces import load_trace
from federatedscope.contrib.decentralized.tests.core.toy_method import ToyAlgorithm


def frame(goodput_bps=1_000_000, peers=(1, 2, 3, 4)):
    """One matching frame. Goodput is given in bps so budgets can be made tight.

    Reference sizes with the toy's 8-byte payload: a control frame encodes to 46B
    and a snapshot frame to 54B, so a direction needs >= 100B gross to carry one
    complete payload after the handshake.
    """
    edges = []
    ordered = list(peers)
    for position in range(0, len(ordered) - 1, 2):
        edges.append({"endpoints": [ordered[position], ordered[position + 1]],
                      "duration": 1000, "goodput": goodput_bps})
    return load_trace({
        "trace_schema_version": 1, "time_unit": "ms", "duration_unit": "ms",
        "goodput_unit": "bps", "peers": sorted(ordered),
        "frames": [{"index": 0, "time": 0, "edges": edges}],
    }).frame(0)


def algorithms(values, gamma=0.5):
    return {pid: ToyAlgorithm(pid, value, gamma=gamma) for pid, value in values.items()}


def test_result_equals_direct_multiplication_by_w_t():
    values = {1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0}
    algos = algorithms(values)
    event = compile_graph_round(frame(), [1, 2, 3, 4], 0.25)

    execute_graph_round(event, algos, ByteLedger())

    for pid in values:
        expected = sum(event.weight(pid, other) * values[other] for other in values)
        assert algos[pid].value == pytest.approx(expected)


def test_mixing_uses_pre_round_snapshots_not_sequentially_updated_values():
    """Sequential in-place mixing is a different algorithm: every update reads the
    pre-round snapshots."""
    values = {1: 10.0, 2: 20.0}
    algos = algorithms(values, gamma=0.25)
    event = compile_graph_round(frame(peers=(1, 2)), [1, 2], 0.25)

    execute_graph_round(event, algos, ByteLedger())

    assert algos[1].value == pytest.approx(0.75 * 10.0 + 0.25 * 20.0)
    # If peer 2 had consumed peer 1's *updated* value this would be 0.75*20 + 0.25*12.5
    assert algos[2].value == pytest.approx(0.75 * 20.0 + 0.25 * 10.0)


def test_mixing_is_mean_preserving():
    values = {1: 3.0, 2: 11.0, 3: 29.0, 4: 7.0}
    algos = algorithms(values)
    execute_graph_round(compile_graph_round(frame(), [1, 2, 3, 4], 0.5), algos, ByteLedger())
    assert sum(a.value for a in algos.values()) == pytest.approx(sum(values.values()))


def test_unaffordable_edge_gets_an_identity_block_and_changes_nothing():
    values = {1: 10.0, 2: 20.0}
    algos = algorithms(values)
    # 200 bps for 1 s = 25 bytes gross, less than the 46B handshake alone
    event = compile_graph_round(frame(goodput_bps=200, peers=(1, 2)), [1, 2], 0.5)

    report = execute_graph_round(event, algos, ByteLedger())

    assert report["skipped_edges"], "expected the edge to be skipped"
    assert report["realized_matrix"] == [[1.0, 0.0], [0.0, 1.0]], "expected identity"
    assert algos[1].value == 10.0 and algos[2].value == 20.0, "state must be untouched"
    assert report["payload_bytes"] == 0
    assert report["control_bytes"] > 0, "control that was sent is still counted"


def test_offered_matrix_is_preserved_even_when_the_realized_one_is_identity():
    event = compile_graph_round(frame(goodput_bps=200, peers=(1, 2)), [1, 2], 0.5)
    report = execute_graph_round(event, algorithms({1: 1.0, 2: 2.0}), ByteLedger())

    assert report["offered_matrix"] == [[0.5, 0.5], [0.5, 0.5]]
    assert report["realized_matrix"] != report["offered_matrix"]
    assert report["offered_edges"] == [[1, 2]]


def test_unmatched_peer_consumes_no_bytes_and_does_not_mix():
    values = {1: 10.0, 2: 20.0, 3: 99.0}
    algos = algorithms(values)
    event = compile_graph_round(frame(peers=(1, 2)), [1, 2, 3], 0.5)

    report = execute_graph_round(event, algos, ByteLedger())

    assert algos[3].value == 99.0, "a self-weight-one peer must not change"
    assert 3 not in report["mixed_peers"]


def test_ledger_records_every_routed_payload_directionally():
    ledger = ByteLedger()
    event = compile_graph_round(frame(peers=(1, 2)), [1, 2], 0.5)
    execute_graph_round(event, algorithms({1: 1.0, 2: 2.0}), ledger)

    directions = ledger.by_direction()
    assert (1, 2) in directions and (2, 1) in directions
    assert ledger.total("payload") > 0


def test_graph_rounds_report_traffic_directionally():
    """The run summary sums the directional keys of every record, so a graph round
    reports its traffic per direction."""
    ledger = ByteLedger()
    event = compile_graph_round(frame(peers=(1, 2)), [1, 2], 0.5)
    report = execute_graph_round(event, algorithms({1: 1.0, 2: 2.0}), ledger)

    for field in ("control_bytes", "payload_bytes"):
        assert report[f"{field}_forward"] > 0
        assert report[f"{field}_reverse"] > 0
        assert report[field] == report[f"{field}_forward"] + report[f"{field}_reverse"]


def test_a_skipped_edge_reports_control_but_no_payload_in_either_direction():
    event = compile_graph_round(frame(goodput_bps=200, peers=(1, 2)), [1, 2], 0.5)
    report = execute_graph_round(event, algorithms({1: 1.0, 2: 2.0}), ByteLedger())
    assert report["control_bytes_forward"] > 0 and report["control_bytes_reverse"] > 0
    assert report["payload_bytes_forward"] == 0 and report["payload_bytes_reverse"] == 0
