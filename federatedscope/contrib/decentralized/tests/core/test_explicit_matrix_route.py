"""Explicit mixing matrices from configuration.

``fixed_q`` / ``fixed_wt`` are ``str`` options in cfg_decentralized.py, and yacs
refuses a list for a ``str`` key, so an explicit doubly stochastic matrix (a fixed
ER graph, for instance) is given in the option as a ``json:``-marked JSON literal
or as the path of a JSON file holding one. It is parsed and validated: square,
symmetric, doubly stochastic, connected, peers matched. Named topologies resolve
as ``build_named_q`` builds them.
"""
import json

import pytest

from federatedscope.contrib.decentralized.core.mixing_matrix import (
    JSON_PREFIX,
    MixingMatrixError,
    build_named_q,
    parse_explicit_q,
    q_digest,
    resolve_fixed_q,
)
from federatedscope.core.configs.config import global_cfg

PEERS = (1, 2, 3, 4)
#: A connected symmetric doubly stochastic matrix that is no named topology: the
#: 4-cycle 1-2-4-3-1 with unequal weights, i.e. what a fixed ER draw looks like.
ER_LIKE = [[0.5, 0.3, 0.2, 0.0],
           [0.3, 0.5, 0.0, 0.2],
           [0.2, 0.0, 0.5, 0.3],
           [0.0, 0.2, 0.3, 0.5]]
ER_JSON = JSON_PREFIX + json.dumps(ER_LIKE)


def dec_lora_cfg():
    cfg = global_cfg.clone()
    cfg.merge_from_list(["federate.method", "dec_lora", "decentralized.enabled", True])
    return cfg


# -- yacs and the str option ----------------------------------------------------------------

def test_yacs_refuses_a_list_in_the_str_option():
    """The option is a str, and yacs refuses a list at merge time."""
    cfg = dec_lora_cfg()
    with pytest.raises(ValueError, match="Type mismatch"):
        cfg.merge_from_list(["decentralized.methods.dec_lora.fixed_q", ER_LIKE])


def test_yacs_literal_evaluates_a_bare_json_literal_into_a_list_and_refuses_it_too():
    """Why the literal is marked: the loader literal-evaluates every string, so a bare
    ``[[...]]`` never arrives as a str. ``json:[[...]]`` is no Python literal and does."""
    cfg = dec_lora_cfg()
    with pytest.raises(ValueError, match="Type mismatch"):
        cfg.merge_from_list(["decentralized.methods.dec_lora.fixed_q", json.dumps(ER_LIKE)])
    cfg = dec_lora_cfg()
    cfg.merge_from_list(["decentralized.methods.dec_lora.fixed_q", ER_JSON])
    assert cfg.decentralized.methods.dec_lora.fixed_q == ER_JSON


# -- the supported route: a marked JSON literal, or a file, in the str option ----------------

def test_a_json_matrix_literal_resolves_validated_and_equal_to_the_list_route():
    matrix, edges, digest, source = resolve_fixed_q(ER_JSON, PEERS, 0.5)
    assert source == "explicit:json"
    assert [list(row) for row in matrix] == ER_LIKE
    assert edges == ((1, 2), (1, 3), (2, 4), (3, 4))
    listed = resolve_fixed_q(ER_LIKE, PEERS, 0.5)
    assert (listed[0], listed[1], listed[2]) == (matrix, edges, digest)
    assert digest == q_digest(ER_LIKE, PEERS)
    # the dict form carries peers, which must be the run's own
    declared = JSON_PREFIX + json.dumps({"matrix": ER_LIKE, "peers": [1, 2, 3, 4]})
    assert resolve_fixed_q(declared, PEERS, 0.5)[2] == digest
    with pytest.raises(MixingMatrixError, match="peers"):
        resolve_fixed_q(JSON_PREFIX + json.dumps({"matrix": ER_LIKE, "peers": [2, 3, 4, 5]}),
                        PEERS, 0.5)


def test_a_matrix_file_resolves_and_its_declared_peers_must_match(tmp_path):
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(ER_LIKE))
    matrix, _edges, digest, source = resolve_fixed_q(str(bare), PEERS, 0.5)
    assert source == "explicit:file:bare.json"
    assert digest == q_digest(ER_LIKE, PEERS)

    declared = tmp_path / "declared.json"
    declared.write_text(json.dumps({"matrix": ER_LIKE, "peers": [1, 2, 3, 4]}))
    assert resolve_fixed_q(str(declared), PEERS, 0.5)[2] == digest

    other = tmp_path / "other.json"
    other.write_text(json.dumps({"matrix": ER_LIKE, "peers": [1, 2, 3, 5]}))
    with pytest.raises(MixingMatrixError, match="peers"):
        resolve_fixed_q(str(other), PEERS, 0.5)


@pytest.mark.parametrize("bad,expected", [
    (json.dumps([[0.5, 0.6, 0.0, 0.0], [0.4, 0.5, 0.1, 0.0],
                 [0.0, 0.1, 0.5, 0.4], [0.0, 0.0, 0.4, 0.6]]), "not symmetric"),
    (json.dumps([[0.7, 0.3, 0.0, 0.0], [0.3, 0.5, 0.3, 0.0],
                 [0.0, 0.3, 0.5, 0.2], [0.0, 0.0, 0.2, 0.8]]), "sums to"),
    (json.dumps([[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0],
                 [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5]]), "not connected"),
    (json.dumps([[0.5, 0.5], [0.5, 0.5]]), "rows"),
    (json.dumps([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
     "must be square"),
    ("[[0.5, 0.5], [0.5,", "not valid JSON"),
    (json.dumps([[0.5, "half"], ["half", 0.5]]), "number"),
    (json.dumps({"rows": ER_LIKE}), "list of rows"),
    (json.dumps({"matrix": ER_LIKE, "peers": "all"}), "list of integers"),
])
def test_an_invalid_json_matrix_from_configuration_is_refused(bad, expected):
    with pytest.raises(MixingMatrixError, match=expected):
        resolve_fixed_q(JSON_PREFIX + bad, PEERS, 0.5)


def test_a_string_that_is_neither_a_topology_a_literal_nor_a_file_is_refused_by_name():
    with pytest.raises(MixingMatrixError, match="neither a named topology"):
        resolve_fixed_q("hexagon", PEERS, 0.5)
    with pytest.raises(MixingMatrixError, match="neither a named topology"):
        resolve_fixed_q("/no/such/matrix.json", PEERS, 0.5)


def test_parse_explicit_q_names_its_route():
    rows, declared, source = parse_explicit_q(ER_JSON)
    assert rows == ER_LIKE and declared is None and source == "explicit:json"
    # a bare literal is accepted for callers that never pass through the loader
    assert parse_explicit_q(json.dumps(ER_LIKE)) == (ER_LIKE, None, "explicit:json")


@pytest.mark.parametrize("topology", ["ring", "complete", "star"])
def test_named_topologies_resolve_to_the_named_matrices(topology):
    matrix, edges, digest, source = resolve_fixed_q(topology, PEERS, 0.5)
    named, named_edges = build_named_q(topology, PEERS, 0.5)
    assert (matrix, edges) == (named, named_edges)
    assert source == f"topology:{topology}:gamma=0.5"
    assert digest == q_digest(named, PEERS)


# -- from a scenario's cfg_overrides --------------------------------------------------------

def _fixed_er_trace(frames=3, peers=PEERS, goodput=10_000_000):
    edges = [(1, 2), (1, 3), (2, 4), (3, 4)]
    return {"trace_schema_version": 1, "time_unit": "ms", "duration_unit": "ms",
            "goodput_unit": "bps", "peers": list(peers),
            "frames": [{"index": index, "time": index * 1000,
                        "edges": [{"endpoints": list(edge), "duration": 1000,
                                   "goodput": goodput} for edge in edges]}
                       for index in range(frames)]}


def test_an_invalid_matrix_in_cfg_overrides_fails_at_config_time_not_mid_run():
    from federatedscope.contrib.decentralized.simulation.scenario import Scenario, build_cfg

    overrides = {"decentralized.execution.interaction": "graph_round",
                 "decentralized.methods.dec_lora.profile": "paper_fixed_q",
                 "decentralized.methods.dec_lora.fixed_q": JSON_PREFIX + json.dumps(
                     [[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0],
                      [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5]]),
                 "decentralized.comparison.profile": "paper_native"}
    scenario = Scenario(method="dec_lora", peers=PEERS, trace=_fixed_er_trace(),
                        adapter="federatedscope.contrib.decentralized.tests.fixtures.build_toy_lora_model",
                        adapter_rank=4, cfg_overrides=overrides,
                        construction={"mode": "common_canonical_v1", "model_init_seed": 1})
    with pytest.raises(MixingMatrixError, match="not connected"):
        build_cfg(scenario)


def test_the_route_reaches_the_runner_through_the_tad_lora_paper_wt_profile():
    from federatedscope.contrib.decentralized.core.interactions import assert_doubly_stochastic
    from federatedscope.contrib.decentralized.tests.methods.test_tad_lora import (
        build_runner, tad_cfg)

    cfg = tad_cfg(profile="paper_wt", frames=2,
                  **{"decentralized.methods.tad_lora.fixed_wt": ER_JSON,
                     "decentralized.comparison.dataset_free": True})
    runner = build_runner(cfg)
    assert runner.fixed_mixing[3] == "explicit:json"
    assert [list(row) for row in runner.fixed_mixing[0]] == ER_LIKE
    results = runner.run()
    rounds = [e for e in runner.sink.events if e.get("event") == "graph_round"]
    assert len(rounds) == 2 and all(r["status"] == "committed" for r in rounds)
    assert rounds[0]["offered_matrix"] == ER_LIKE
    assert_doubly_stochastic(rounds[0]["realized_matrix"])
    assert set(results["per_peer"]) == set(PEERS)
