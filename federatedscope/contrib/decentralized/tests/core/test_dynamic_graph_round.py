"""Dynamic graph rounds: a per-frame renormalised ``W_t`` at any degree.

``compile_dynamic_graph_round`` derives ``W_t`` from the available edges of each
frame by Metropolis-Hastings weights, the time-varying semantics of TAD-LoRA and
D-PSGD. The pair-block compiler (``compile_graph_round``) refuses a peer on two
edges, and ``compile_fixed_graph_round`` applies the configured matrix to every
frame. Covered here: the rule, the executor's matrix-vector product, the
configuration, the manifest and the run label.
"""
import json
from fractions import Fraction

import pytest

# The global config first: it imports every contrib config (the decentralized extension
# included) before building global_cfg. Imported after the core modules, a circular
# import builds global_cfg without the decentralized keys.
import federatedscope.core.configs.config  # noqa: F401,E402
from federatedscope.contrib.decentralized.core.accounting import ByteLedger
from federatedscope.contrib.decentralized.core.graph_mixing import execute_graph_round
from federatedscope.contrib.decentralized.core.interactions import (
    GRAPH_MIXING_DYNAMIC,
    GRAPH_MIXING_MODES,
    GRAPH_MIXING_PAIR_BLOCK,
    METROPOLIS_RULE,
    CompilationError,
    assert_doubly_stochastic,
    compile_dynamic_graph_round,
    compile_fixed_graph_round,
    compile_graph_round,
    metropolis_hastings_matrix,
)
from federatedscope.contrib.decentralized.core.manifest import (
    assert_comparable,
    build_manifest,
    run_label,
)
from federatedscope.contrib.decentralized.core.mixing_matrix import resolve_fixed_q
from federatedscope.contrib.decentralized.core.traces import load_trace
from federatedscope.contrib.decentralized.tests.core.toy_method import ToyAlgorithm

PEERS = (1, 2, 3, 4)


def document(frames, peers=PEERS, goodput=1000):
    """A trace document. ``frames`` is a list of edge lists; an edge is ``(a, b)`` or
    ``(a, b, available)``."""
    out = []
    for index, edges in enumerate(frames):
        entries = []
        for edge in edges:
            a, b = edge[0], edge[1]
            entry = {"endpoints": [a, b], "duration": 1000, "goodput": goodput}
            if len(edge) > 2:
                entry["available"] = bool(edge[2])
                if not edge[2]:
                    entry["goodput"] = 0
            entries.append(entry)
        out.append({"index": index, "time": index * 1000, "edges": entries})
    return {"trace_schema_version": 1, "time_unit": "ms", "duration_unit": "ms",
            "goodput_unit": "kbps", "peers": list(peers), "frames": out}


#: Six peers over six frames, each a different graph; degrees from 0 to 5.
TIME_VARYING = document([
    [(1, 2), (1, 3), (2, 3), (3, 4)],
    [(1, 4), (2, 5), (5, 6), (1, 6)],
    [(2, 3)],
    [],
    [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6)],
    [(1, 2), (3, 4), (5, 6), (2, 3, False)],
], peers=(1, 2, 3, 4, 5, 6))


def frame_of(doc, index=0):
    return load_trace(doc).frame(index)


# -- the three graph modes at degree above one ---------------------------------------------

def test_only_the_dynamic_mode_renormalises_w_t_at_degree_above_one():
    """The pair-block compiler refuses a peer on two edges, the fixed compiler hands back
    the configured matrix unchanged on a frame that does not offer all of it, and the
    dynamic compiler derives a doubly stochastic ``W_t`` from the frame."""
    frame = frame_of(TIME_VARYING, 0)          # peer 3 has degree 3
    with pytest.raises(CompilationError, match="disjoint edges"):
        compile_graph_round(frame, (1, 2, 3, 4, 5, 6), 0.5)

    matrix, _edges, _digest, _source = resolve_fixed_q("ring", (1, 2, 3, 4, 5, 6), 0.5)
    fixed = compile_fixed_graph_round(frame, (1, 2, 3, 4, 5, 6), matrix)
    assert fixed.matrix == matrix, "the fixed mode is time-invariant by construction"
    assert fixed.unavailable_edges, "and this frame does not even offer every ring edge"

    dynamic = compile_dynamic_graph_round(frame, (1, 2, 3, 4, 5, 6))
    assert dynamic.matrix != matrix
    assert_doubly_stochastic(dynamic.matrix)


# -- the rule -----------------------------------------------------------------------------

def test_w_t_is_symmetric_doubly_stochastic_on_every_frame_of_a_time_varying_trace():
    trace = load_trace(TIME_VARYING)
    seen = set()
    for frame in trace.frames:
        event = compile_dynamic_graph_round(frame, trace.peer_ids, event_id=frame.index)
        assert_doubly_stochastic(event.matrix)
        assert all(w >= 0.0 for row in event.matrix for w in row)
        assert event.peer_order == trace.peer_ids
        assert event.unavailable_edges == ()
        seen.add(event.matrix)
    assert len(seen) == len(trace.frames), "a time-varying trace must give a varying W_t"


def test_w_t_follows_the_metropolis_hastings_rule():
    """w_ij = 1 / (1 + max(d_i, d_j)) on an available edge, w_ii = 1 - sum_j w_ij."""
    event = compile_dynamic_graph_round(frame_of(TIME_VARYING, 0), (1, 2, 3, 4, 5, 6))
    degree = {1: 2, 2: 2, 3: 3, 4: 1, 5: 0, 6: 0}
    for a, b in ((1, 2), (1, 3), (2, 3), (3, 4)):
        assert event.weight(a, b) == pytest.approx(1.0 / (1 + max(degree[a], degree[b])))
        assert event.weight(b, a) == event.weight(a, b)
    for peer in (1, 2, 3, 4, 5, 6):
        off = sum(event.weight(peer, other) for other in (1, 2, 3, 4, 5, 6) if other != peer)
        assert event.weight(peer, peer) == pytest.approx(1.0 - off)
    assert event.weight(5, 5) == 1.0 and event.weight(6, 6) == 1.0
    assert "1/(1+max(d_i,d_j))" in METROPOLIS_RULE.replace(" ", "")
    assert [tuple(e.endpoints) for e in event.edges] == [(1, 2), (1, 3), (2, 3), (3, 4)]


def test_an_unavailable_edge_gets_weight_zero_and_the_diagonal_absorbs_it():
    frame = frame_of(document([[(1, 2), (2, 3, False), (3, 4)]]))
    event = compile_dynamic_graph_round(frame, PEERS)
    assert event.weight(2, 3) == 0.0 and event.weight(3, 2) == 0.0
    # peers 2 and 3 have degree one each once the unavailable edge is out
    assert event.weight(1, 2) == pytest.approx(0.5)
    assert event.weight(2, 2) == pytest.approx(0.5)
    assert event.weight(3, 3) == pytest.approx(0.5)
    assert [tuple(e.endpoints) for e in event.edges] == [(1, 2), (3, 4)]
    assert event.unavailable_edges == (), "nothing is left for the executor to resolve"
    assert_doubly_stochastic(event.matrix)


def test_at_degree_one_the_dynamic_w_t_is_the_pair_block_matrix_at_gamma_one_half():
    frame = frame_of(document([[(1, 2), (3, 4)]]))
    dynamic = compile_dynamic_graph_round(frame, PEERS)
    pair_block = compile_graph_round(frame, PEERS, 0.5)
    assert dynamic.matrix == pair_block.matrix
    assert dynamic.edges == pair_block.edges


def test_an_edge_to_an_unknown_peer_is_refused():
    with pytest.raises(CompilationError, match="unknown peer"):
        compile_dynamic_graph_round(frame_of(document([[(1, 2)]])), (1, 3))


def test_the_matrix_builder_is_a_pure_function_of_the_edge_set():
    order = (1, 2, 3, 4)
    edges = [(1, 2), (1, 3), (2, 3), (3, 4)]
    assert metropolis_hastings_matrix(order, edges) == \
        metropolis_hastings_matrix(order, list(reversed(edges)))
    assert metropolis_hastings_matrix(order, []) == tuple(
        tuple(1.0 if i == j else 0.0 for j in range(4)) for i in range(4))


# -- the executor: a matrix-vector product from one pre-round barrier ------------------------

def test_the_executor_applies_the_matrix_vector_product_on_a_toy_state():
    """Hand-computed. Edges (1,2),(1,3),(2,3),(3,4): d = (2,2,3,1), so
    w12 = 1/3, w13 = w23 = w34 = 1/4, w11 = w22 = 5/12, w33 = 1/4, w44 = 3/4.
    x = (10,20,30,40) -> (55/3, 115/6, 25, 75/2)."""
    frame = frame_of(document([[(1, 2), (1, 3), (2, 3), (3, 4)]], goodput=1_000_000))
    event = compile_dynamic_graph_round(frame, PEERS)
    algorithms = {1: ToyAlgorithm(1, 10.0), 2: ToyAlgorithm(2, 20.0),
                  3: ToyAlgorithm(3, 30.0), 4: ToyAlgorithm(4, 40.0)}

    report = execute_graph_round(event, algorithms, ByteLedger())

    expected = {1: Fraction(55, 3), 2: Fraction(115, 6), 3: Fraction(25), 4: Fraction(75, 2)}
    for peer, value in expected.items():
        assert algorithms[peer].value == pytest.approx(float(value)), peer
    assert report["status"] == "committed" and report["skipped_edges"] == []
    assert report["mixed_peers"] == [1, 2, 3, 4]
    assert report["realized_matrix"] == [list(row) for row in event.matrix]


def test_mixing_at_degree_above_one_is_mean_preserving_and_contracts_disagreement():
    trace = load_trace(document([[(1, 2), (1, 3), (2, 3), (3, 4)], [(1, 4), (2, 3), (1, 2)]],
                                goodput=1_000_000))
    algorithms = {p: ToyAlgorithm(p, float(p) * 10) for p in PEERS}
    total = sum(a.value for a in algorithms.values())
    spread = max(a.value for a in algorithms.values()) - min(a.value for a in algorithms.values())
    for frame in trace.frames:
        execute_graph_round(compile_dynamic_graph_round(frame, PEERS, frame.index),
                            algorithms, ByteLedger())
    assert sum(a.value for a in algorithms.values()) == pytest.approx(total)
    after = max(a.value for a in algorithms.values()) - min(a.value for a in algorithms.values())
    assert after < spread


# -- configuration ------------------------------------------------------------------------

def test_the_mode_names_agree_between_the_config_schema_and_the_compiler():
    from federatedscope.contrib.configs.cfg_decentralized import GRAPH_MIXING_MODES as declared

    assert tuple(declared) == tuple(GRAPH_MIXING_MODES) == (GRAPH_MIXING_PAIR_BLOCK,
                                                             GRAPH_MIXING_DYNAMIC)


def _tad(tmp_path, doc, **overrides):
    from federatedscope.contrib.decentralized.core.runner import DecentralizedRunner
    from federatedscope.contrib.decentralized.tests.methods.test_tad_lora import (
        ToyTADFactory, tad_cfg)

    path = tmp_path / "trace.json"
    path.write_text(json.dumps(doc))
    settings = {"decentralized.opportunities.source": "file",
                "decentralized.opportunities.path": str(path),
                "decentralized.comparison.profile": "paper_native",
                "decentralized.comparison.dataset_free": True}
    settings.update(overrides)
    cfg = tad_cfg(client_num=len(doc["peers"]), frames=len(doc["frames"]), **settings)
    return DecentralizedRunner(data={}, config=cfg, factory=ToyTADFactory(cfg))


def test_the_dynamic_mode_is_selected_by_configuration_and_refused_off_the_graph_form():
    from federatedscope.contrib.decentralized.tests.core.test_runner import toy_cfg

    cfg = toy_cfg(**{"decentralized.execution.graph_mixing": GRAPH_MIXING_DYNAMIC,
                     "decentralized.execution.interaction": "graph_round"})
    assert cfg.decentralized.execution.graph_mixing == GRAPH_MIXING_DYNAMIC
    with pytest.raises(ValueError, match="graph_mixing"):
        toy_cfg(**{"decentralized.execution.graph_mixing": "improvise"})
    with pytest.raises(ValueError, match="dynamic_metropolis.*graph_round"):
        toy_cfg(**{"decentralized.execution.graph_mixing": GRAPH_MIXING_DYNAMIC,
                   "decentralized.execution.interaction": "pairwise_session"})


def test_the_dynamic_mode_and_a_fixed_matrix_are_two_experiments_and_refused_together(tmp_path):
    with pytest.raises(ValueError, match="fixed mixing matrix.*dynamic"):
        _tad(tmp_path, TIME_VARYING,
             **{"decentralized.execution.graph_mixing": GRAPH_MIXING_DYNAMIC,
                "decentralized.methods.tad_lora.profile": "paper_wt",
                "decentralized.methods.tad_lora.fixed_wt": "ring"})


def test_the_simulator_runs_tad_lora_over_a_time_varying_trace_to_completion(tmp_path):
    """Six frames, degrees up to five. Local episodes train nothing on the toy
    fixture, so every peer's adapter mean must follow the product of the per-frame
    matrices exactly - the mean is linear in the state the round mixes."""
    runner = _tad(tmp_path, TIME_VARYING,
                  **{"decentralized.execution.graph_mixing": GRAPH_MIXING_DYNAMIC})
    runner._build_peers()
    before = {pid: peer.algorithm.consensus_fingerprint()["adapter_mean"]
              for pid, peer in runner.peers.items()}
    order = tuple(sorted(before))
    means = [before[p] for p in order]
    for frame in runner.trace.frames:
        matrix = compile_dynamic_graph_round(frame, order).matrix
        means = [sum(matrix[i][j] * means[j] for j in range(len(order)))
                 for i in range(len(order))]
    expected = dict(zip(order, means))

    results = runner.run()

    rounds = [e for e in runner.sink.events if e.get("event") == "graph_round"]
    assert len(rounds) == len(TIME_VARYING["frames"])
    for record in rounds:
        assert record["status"] == "committed"
        assert_doubly_stochastic(record["offered_matrix"])
        assert record["realized_matrix"] == record["offered_matrix"], record["skipped_edges"]
    assert sorted(rounds[0]["mixed_peers"]) == [1, 2, 3, 4]
    assert rounds[4]["mixed_peers"] == [1, 2, 3, 4, 5, 6]
    for pid in order:
        assert results["per_peer"][pid]["adapter_mean"] == pytest.approx(expected[pid], abs=1e-6)
    assert results["per_peer"][1]["adapter_mean"] != pytest.approx(before[1], abs=1e-6)


# -- the manifest: recorded only when the mode is on, and then a comparability axis ------------

def test_the_manifest_carries_the_mode_only_when_it_is_on(tmp_path):
    matching = document([[(1, 2), (3, 4)], [(1, 3), (2, 4)]])
    off = build_manifest(_tad(tmp_path, matching))
    assert "graph_mixing" not in off, "a run without the dynamic mode must not record it"
    on = build_manifest(_tad(tmp_path, matching,
                             **{"decentralized.execution.graph_mixing": GRAPH_MIXING_DYNAMIC}))
    assert on["graph_mixing"] == {"mode": GRAPH_MIXING_DYNAMIC, "rule": METROPOLIS_RULE}
    assert set(on) - set(off) == {"graph_mixing"}
    with pytest.raises(ValueError, match="graph_mixing"):
        assert_comparable(off, on)


# -- run labels --------------------------------------------------------------------------------

def _manifest(**fields):
    base = {"method": "tad_lora", "interaction_form": "graph_round"}
    base.update(fields)
    return base


def test_run_labels_follow_the_naming_rule(tmp_path):
    dynamic = _manifest(graph_mixing={"mode": GRAPH_MIXING_DYNAMIC, "rule": METROPOLIS_RULE})
    assert run_label(dynamic) == "tad-native-dynamic-graph"
    assert run_label(dict(dynamic, trace_generator={"name": "er_gnp"})) == \
        "tad-native-dynamic-er"
    # a fixed-matrix run of TAD-LoRA is static, never native
    assert run_label(_manifest(fixed_mixing_source="topology:ring:gamma=0.5")) == \
        "tad-ring-static"
    assert run_label(_manifest(fixed_mixing_source="explicit")) == "tad-fixed-q-static"
    # degree-one execution of a graph method is pairwise-adapted, never native
    assert run_label(_manifest()) == "tad-pairwise-adapted"
    assert run_label(_manifest(method="dec_lora", interaction_form="pairwise_session")) == \
        "dec-pairwise-adapted"
    assert run_label(_manifest(method="pulse_lora", interaction_form="pairwise_session"),
                     native_form="pairwise_session") == "pulse-native"
    for label in (run_label(_manifest()), run_label(_manifest(fixed_mixing_source="explicit"))):
        assert "native" not in label

    # and from a runner's manifest, with the dynamic mode off and on
    matching = document([[(1, 2), (3, 4)]])
    assert run_label(build_manifest(_tad(tmp_path, matching))) == "tad-pairwise-adapted"
    assert run_label(build_manifest(_tad(
        tmp_path, matching, **{"decentralized.execution.graph_mixing": GRAPH_MIXING_DYNAMIC}))) \
        == "tad-native-dynamic-graph"
