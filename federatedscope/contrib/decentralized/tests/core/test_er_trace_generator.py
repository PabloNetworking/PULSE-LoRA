"""The seeded Erdos-Renyi contact-trace generator.

``er_trace_document`` writes a trace document (schema version 1, core/traces.py)
whose ``generator`` block records the generator's name and version, the seed,
``p``, ``n``, the number of frames and the trace digest: the digest the runner and
the manifest compute for the trace, checked on load.
"""
import json

import pytest

from federatedscope.contrib.decentralized.core import traces as T
from federatedscope.contrib.decentralized.core.manifest import build_manifest, run_label
from federatedscope.contrib.decentralized.core.traces import (
    ER_GENERATOR_NAME,
    ER_GENERATOR_VERSION,
    TraceValidationError,
    er_trace_document,
    generate_er_trace,
    load_trace,
)

PEERS = (1, 2, 3, 4, 5, 6)


# -- the document -----------------------------------------------------------------------------


def test_the_document_records_seed_p_n_generator_and_digest():
    document = er_trace_document(PEERS, 5, 0.25, seed=3)
    block = document["generator"]
    assert block == {"name": ER_GENERATOR_NAME, "version": ER_GENERATOR_VERSION,
                     "seed": 3, "p": 0.25, "n": 6, "num_frames": 5,
                     "trace_id": block["trace_id"]}
    assert len(block["trace_id"]) == 64
    assert document["trace_schema_version"] == T.TRACE_SCHEMA_VERSION
    assert document["peers"] == list(PEERS)
    assert len(document["frames"]) == 5


def test_the_recorded_digest_is_the_one_the_runner_and_the_manifest_compute(tmp_path):
    from federatedscope.contrib.decentralized.tests.core.test_dynamic_graph_round import _tad

    document = er_trace_document(PEERS, 4, 0.5, seed=11)
    recorded = document["generator"]["trace_id"]
    trace = load_trace(document)
    assert trace.trace_id == recorded
    assert T.compute_trace_digest(trace.normalised) == recorded
    assert trace.provenance == document["generator"]

    runner = _tad(tmp_path, document)
    assert runner.trace.trace_id == recorded
    manifest = build_manifest(runner)
    assert manifest["trace_id"] == recorded
    assert manifest["trace_generator"] == document["generator"]


def test_the_generator_block_is_written_only_by_the_er_generator(tmp_path):
    from federatedscope.contrib.decentralized.tests.core.test_dynamic_graph_round import (
        _tad, document)

    hand_written = document([[(1, 2), (3, 4)]])
    assert load_trace(hand_written).provenance is None
    manifest = build_manifest(_tad(tmp_path, hand_written))
    assert "trace_generator" not in manifest, "a hand-written trace records no generator"
    matching = T.generate_matching_trace(PEERS, 3, __import__("random").Random(1))
    assert matching.provenance is None


def test_every_edge_has_both_endpoints_in_peers_and_no_edge_repeats():
    trace = generate_er_trace(PEERS, 50, 0.5, seed=5)
    peers = set(trace.peer_ids)
    for frame in trace.frames:
        seen = set()
        for opportunity in frame.opportunities:
            a, b = opportunity.endpoints
            assert a in peers and b in peers and a < b
            assert (a, b) not in seen
            seen.add((a, b))
            assert opportunity.available


def test_the_mean_degree_is_within_tolerance_of_p_times_n_minus_one():
    n, p, frames = 8, 0.3, 400
    trace = generate_er_trace(tuple(range(1, n + 1)), frames, p, seed=2)
    edges = sum(len(frame.opportunities) for frame in trace.frames)
    mean_degree = 2.0 * edges / (n * frames)
    # edges ~ Binomial(28 * 400, 0.3): sd of the mean degree is 0.03, so 0.15 is 5 sd
    assert abs(mean_degree - p * (n - 1)) < 0.15, mean_degree
    assert any(len(frame.opportunities) != len(trace.frames[0].opportunities)
               for frame in trace.frames), "a G(n, p) trace varies from frame to frame"


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_the_degenerate_probabilities_give_the_empty_and_the_complete_graph(p):
    trace = generate_er_trace(PEERS, 3, p, seed=9)
    for frame in trace.frames:
        assert len(frame.opportunities) == (0 if p == 0.0 else 15)


@pytest.mark.parametrize("bad", [-0.1, 1.5, "half"])
def test_a_probability_outside_the_unit_interval_is_refused(bad):
    with pytest.raises((TraceValidationError, ValueError, TypeError)):
        er_trace_document(PEERS, 3, bad, seed=1)


def test_fewer_than_two_peers_or_no_frames_are_refused():
    with pytest.raises(TraceValidationError):
        er_trace_document((1,), 3, 0.5, seed=1)
    with pytest.raises(TraceValidationError):
        er_trace_document(PEERS, 0, 0.5, seed=1)


# -- the digest inside the document is checked, never trusted --------------------------------

def test_the_recorded_digest_is_checked_on_load():
    document = er_trace_document(PEERS, 3, 0.5, seed=4)
    tampered = json.loads(json.dumps(document))
    tampered["generator"]["trace_id"] = "0" * 64
    with pytest.raises(TraceValidationError, match="records trace_id"):
        load_trace(tampered)
    edited = json.loads(json.dumps(document))
    edited["frames"][0]["edges"] = edited["frames"][0]["edges"][:-1]
    with pytest.raises(TraceValidationError, match="records trace_id"):
        load_trace(edited)


def test_a_generator_block_that_is_not_a_plain_mapping_is_refused():
    document = er_trace_document(PEERS, 3, 0.5, seed=4)
    document["generator"] = ["er_gnp"]
    with pytest.raises(TraceValidationError, match="generator"):
        load_trace(document)


# -- the label: only a dynamic run over an ER document is -native-dynamic-er ------------------

def test_a_dynamic_run_over_an_er_document_carries_the_er_label(tmp_path):
    from federatedscope.contrib.decentralized.core.interactions import GRAPH_MIXING_DYNAMIC
    from federatedscope.contrib.decentralized.tests.core.test_dynamic_graph_round import _tad

    document = er_trace_document(PEERS, 3, 0.5, seed=4)
    dynamic = build_manifest(_tad(tmp_path, document,
                                  **{"decentralized.execution.graph_mixing": GRAPH_MIXING_DYNAMIC}))
    assert run_label(dynamic) == "tad-native-dynamic-er"
    # the same document under the fixed ring stays static, whatever the trace was
    static = build_manifest(_tad(tmp_path, document,
                                 **{"decentralized.methods.tad_lora.profile": "paper_wt",
                                    "decentralized.methods.tad_lora.fixed_wt": "ring"}))
    assert run_label(static) == "tad-ring-static"
