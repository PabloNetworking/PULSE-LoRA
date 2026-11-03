"""Fixed mixing matrices: named and explicit Q, the configured Q applied unchanged
every round, and a valid realized matrix when an edge is skipped at degree > 1.
"""
import pytest

from federatedscope.contrib.decentralized.core.accounting import ByteLedger
from federatedscope.contrib.decentralized.core.graph_mixing import (
    SKIPPED_EDGE_IDENTITY_ROUND,
    SKIPPED_EDGE_SELF_WEIGHT,
    GraphMixingError,
    execute_graph_round,
)
from federatedscope.contrib.decentralized.core.interactions import compile_fixed_graph_round
from federatedscope.contrib.decentralized.core.mixing_matrix import (
    MixingMatrixError,
    build_named_q,
    q_digest,
    resolve_fixed_q,
    validate_q,
)
from federatedscope.contrib.decentralized.core.traces import generate_fixed_graph_trace
from federatedscope.contrib.decentralized.core.types import (
    DirectionalCredits,
    GraphEdgeInfo,
    GraphMixingEvent,
)
from federatedscope.contrib.decentralized.tests.core.toy_method import ToyAlgorithm

PEERS = (1, 2, 3, 4)


# --- the provider ---------------------------------------------------------------

@pytest.mark.parametrize("topology", ["ring", "complete", "star"])
def test_named_topologies_are_symmetric_doubly_stochastic_and_connected(topology):
    matrix, edges = build_named_q(topology, PEERS, 0.5)
    validate_q(matrix, PEERS)          # raises if any property fails
    assert edges


def test_an_explicit_matrix_is_accepted():
    explicit = [[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5]]
    with pytest.raises(MixingMatrixError, match="not connected"):
        resolve_fixed_q(explicit, PEERS, 0.5)

    connected = [[0.5, 0.25, 0.25, 0.0], [0.25, 0.5, 0.0, 0.25],
                 [0.25, 0.0, 0.5, 0.25], [0.0, 0.25, 0.25, 0.5]]
    matrix, edges, digest, source = resolve_fixed_q(connected, PEERS, 0.5)
    assert source == "explicit"
    assert len(edges) == 4
    assert len(digest) == 64


@pytest.mark.parametrize("bad,expected", [
    ([[0.5, 0.6], [0.4, 0.5]], "not symmetric"),
    ([[0.7, 0.3], [0.3, 0.5]], "sums to"),
    ([[1.5, -0.5], [-0.5, 1.5]], "negative weight"),
])
def test_invalid_matrices_are_rejected(bad, expected):
    with pytest.raises(MixingMatrixError, match=expected):
        validate_q(bad, (1, 2))


def test_a_missing_q_is_refused_rather_than_defaulted():
    with pytest.raises(MixingMatrixError, match="none was supplied"):
        resolve_fixed_q("", PEERS, 0.5)


def test_digest_is_stable_and_discriminating():
    ring, _ = build_named_q("ring", PEERS, 0.5)
    star, _ = build_named_q("star", PEERS, 0.5)
    assert q_digest(ring, PEERS) == q_digest(ring, PEERS)
    assert q_digest(ring, PEERS) != q_digest(star, PEERS)


# --- the configured Q controls the update ------------------------------------------

def run_fixed_round(matrix, edges, values, goodput_bps=1_000_000):
    trace = generate_fixed_graph_trace(PEERS, edges, 1, goodput_kbps=goodput_bps // 1000)
    event = compile_fixed_graph_round(trace.frame(0), PEERS, matrix)
    algorithms = {pid: ToyAlgorithm(pid, value) for pid, value in values.items()}
    report = execute_graph_round(event, algorithms, ByteLedger())
    return algorithms, report, event


def test_the_configured_q_numerically_controls_the_update():
    """Each peer's new value is its row of Q times the pre-round values."""
    values = {1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0}
    matrix, edges, _digest, _source = resolve_fixed_q("ring", PEERS, 0.5)

    algorithms, _report, event = run_fixed_round(matrix, edges, values)

    for position, pid in enumerate(event.peer_order):
        expected = sum(matrix[position][j] * values[event.peer_order[j]]
                       for j in range(len(event.peer_order)))
        assert algorithms[pid].value == pytest.approx(expected), f"peer {pid}"


def test_a_different_q_produces_a_different_update():
    """If Q were ignored, these two would agree."""
    values = {1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0}
    ring, ring_edges, _, _ = resolve_fixed_q("ring", PEERS, 0.5)
    star, star_edges, _, _ = resolve_fixed_q("star", PEERS, 0.5)

    ring_algos, _, _ = run_fixed_round(ring, ring_edges, values)
    star_algos, _, _ = run_fixed_round(star, star_edges, values)

    assert [ring_algos[p].value for p in PEERS] != [star_algos[p].value for p in PEERS]


def test_q_is_used_unchanged_across_rounds():
    matrix, edges, _, _ = resolve_fixed_q("ring", PEERS, 0.5)
    trace = generate_fixed_graph_trace(PEERS, edges, 5)
    matrices = {compile_fixed_graph_round(frame, PEERS, matrix).matrix
                for frame in trace.frames}
    assert len(matrices) == 1, "the fixed graph must not vary between rounds"
    assert matrices.pop() == matrix


def test_fixed_mixing_is_mean_preserving():
    values = {1: 3.0, 2: 11.0, 3: 29.0, 4: 7.0}
    matrix, edges, _, _ = resolve_fixed_q("complete", PEERS, 0.5)
    algorithms, _, _ = run_fixed_round(matrix, edges, values)
    assert sum(a.value for a in algorithms.values()) == pytest.approx(sum(values.values()))


# --- partial affordability at degree > 1 --------------------------------------------

def degree_two_event(second_edge_credits):
    """Path graph 1-2-3: peer 2 has degree two."""
    gamma, order = 0.25, (1, 2, 3)
    matrix = ((1 - gamma, gamma, 0.0),
              (gamma, 1 - 2 * gamma, gamma),
              (0.0, gamma, 1 - gamma))
    edges = (
        GraphEdgeInfo((1, 2), 1_000_000, DirectionalCredits(125000, 125000), "direct"),
        GraphEdgeInfo((2, 3), 1_000_000, second_edge_credits, "direct"),
    )
    return GraphMixingEvent(0, 0, 0, order, matrix, edges), matrix, order


def test_partial_affordability_keeps_the_realized_matrix_stochastic():
    """Skipping one of peer 2's two edges must not leave its row summing below one."""
    event, _matrix, order = degree_two_event(DirectionalCredits(20, 20))
    algorithms = {p: ToyAlgorithm(p, float(p)) for p in order}

    report = execute_graph_round(event, algorithms, ByteLedger(),
                                 skipped_edge_policy=SKIPPED_EDGE_SELF_WEIGHT)

    assert report["skipped_edges"], "the second edge should have been skipped"
    for row in report["realized_matrix"]:
        assert sum(row) == pytest.approx(1.0)
    for position in range(len(order)):
        column = sum(r[position] for r in report["realized_matrix"])
        assert column == pytest.approx(1.0)


def test_self_weight_policy_preserves_the_mean():
    event, _matrix, order = degree_two_event(DirectionalCredits(20, 20))
    algorithms = {p: ToyAlgorithm(p, float(p)) for p in order}
    before = sum(a.value for a in algorithms.values())

    execute_graph_round(event, algorithms, ByteLedger(),
                        skipped_edge_policy=SKIPPED_EDGE_SELF_WEIGHT)

    assert sum(a.value for a in algorithms.values()) == pytest.approx(before)


def test_skipped_weight_is_reassigned_to_both_diagonals():
    event, matrix, order = degree_two_event(DirectionalCredits(20, 20))
    algorithms = {p: ToyAlgorithm(p, float(p)) for p in order}

    report = execute_graph_round(event, algorithms, ByteLedger(),
                                 skipped_edge_policy=SKIPPED_EDGE_SELF_WEIGHT)
    realized = report["realized_matrix"]
    weight = matrix[1][2]

    assert realized[1][2] == 0.0 and realized[2][1] == 0.0
    assert realized[1][1] == pytest.approx(matrix[1][1] + weight)
    assert realized[2][2] == pytest.approx(matrix[2][2] + weight)
    assert report["skipped_edges"][0]["reassigned_weight"] == pytest.approx(weight)


def test_identity_round_policy_cancels_the_whole_round():
    event, _matrix, order = degree_two_event(DirectionalCredits(20, 20))
    algorithms = {p: ToyAlgorithm(p, float(p)) for p in order}

    report = execute_graph_round(event, algorithms, ByteLedger(),
                                 skipped_edge_policy=SKIPPED_EDGE_IDENTITY_ROUND)

    assert report["realized_matrix"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert [algorithms[p].value for p in order] == [1.0, 2.0, 3.0], "no state may change"


def test_a_fully_affordable_round_realizes_the_offered_matrix():
    event, matrix, order = degree_two_event(DirectionalCredits(125000, 125000))
    algorithms = {p: ToyAlgorithm(p, float(p)) for p in order}
    report = execute_graph_round(event, algorithms, ByteLedger())
    assert not report["skipped_edges"]
    assert report["realized_matrix"] == [list(row) for row in matrix]


def test_an_unknown_policy_is_rejected():
    event, _, _ = degree_two_event(DirectionalCredits(125000, 125000))
    with pytest.raises(GraphMixingError, match="unknown skipped-edge policy"):
        execute_graph_round(event, {}, ByteLedger(), skipped_edge_policy="improvise")
