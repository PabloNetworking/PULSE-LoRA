"""Pair and graph compilation preserve the same offered conditions."""
import pytest

from federatedscope.contrib.decentralized.core.interactions import (
    CompilationError,
    assert_doubly_stochastic,
    compile_graph_round,
    compile_pair_sessions,
)
from federatedscope.contrib.decentralized.core.rng import RngService
from federatedscope.contrib.decentralized.core.traces import generate_matching_trace, load_trace

TRACE = {
    "trace_schema_version": 1, "time_unit": "ms", "duration_unit": "ms",
    "goodput_unit": "kbps", "peers": [1, 2, 3, 4],
    "frames": [{"index": 0, "time": 0, "edges": [
        {"endpoints": [1, 2], "duration": 1000, "goodput": 1000, "transport": "direct"},
        {"endpoints": [3, 4], "duration": 2000, "goodput": 500, "transport": "infrastructure"},
    ]}],
}


def frame():
    return load_trace(TRACE).frame(0)


def test_compilation_is_deterministic():
    a = compile_pair_sessions(frame())
    b = compile_pair_sessions(frame())
    assert a == b
    assert compile_graph_round(frame(), [1, 2, 3, 4], 0.5) == \
           compile_graph_round(frame(), [1, 2, 3, 4], 0.5)


def test_both_forms_preserve_the_same_exogenous_facts():
    pairs = compile_pair_sessions(frame())
    graph = compile_graph_round(frame(), [1, 2, 3, 4], 0.5)
    assert len(pairs) == len(graph.edges)
    for pair_event, edge in zip(pairs, graph.edges):
        assert pair_event.endpoints == edge.endpoints
        assert pair_event.duration_us == edge.duration_us
        assert pair_event.credits == edge.credits
        assert pair_event.transport == edge.transport


def test_pair_block_matrix_is_symmetric_and_doubly_stochastic():
    graph = compile_graph_round(frame(), [1, 2, 3, 4], 0.25)
    assert_doubly_stochastic(graph.matrix)
    assert graph.weight(1, 2) == 0.25
    assert graph.weight(1, 1) == 0.75
    assert graph.weight(1, 3) == 0.0


def test_unmatched_peer_keeps_self_weight_one():
    graph = compile_graph_round(frame(), [1, 2, 3, 4, 9], 0.5)
    assert graph.weight(9, 9) == 1.0


def test_unavailable_edges_are_not_compiled():
    payload = {**TRACE}
    payload["frames"] = [{"index": 0, "time": 0, "edges": [
        {"endpoints": [1, 2], "duration": 1000, "goodput": 0, "available": False},
        {"endpoints": [3, 4], "duration": 1000, "goodput": 1000},
    ]}]
    only = load_trace(payload).frame(0)
    assert [e.endpoints for e in compile_pair_sessions(only)] == [(3, 4)]
    assert len(compile_graph_round(only, [1, 2, 3, 4], 0.5).edges) == 1


@pytest.mark.parametrize("gamma", [0.0, -0.1, 0.51, 1.0])
def test_graph_compiler_rejects_gamma_outside_the_valid_range(gamma):
    with pytest.raises(CompilationError, match="gamma"):
        compile_graph_round(frame(), [1, 2, 3, 4], gamma)


def test_graph_compiler_rejects_a_non_matching_frame():
    payload = {**TRACE}
    payload["frames"] = [{"index": 0, "time": 0, "edges": [
        {"endpoints": [1, 2], "duration": 1000, "goodput": 1000},
        {"endpoints": [2, 3], "duration": 1000, "goodput": 1000},
    ]}]
    with pytest.raises(CompilationError, match="disjoint edges"):
        compile_graph_round(load_trace(payload).frame(0), [1, 2, 3], 0.5)


def test_every_frame_of_a_generated_trace_compiles_both_ways():
    trace = generate_matching_trace([1, 2, 3, 4, 5, 6], 6,
                                    RngService(21).stream("opportunities"))
    for frame_obj in trace.frames:
        pairs = compile_pair_sessions(frame_obj)
        graph = compile_graph_round(frame_obj, trace.peer_ids, 0.5)
        assert_doubly_stochastic(graph.matrix)
        assert len(pairs) == len(graph.edges)
