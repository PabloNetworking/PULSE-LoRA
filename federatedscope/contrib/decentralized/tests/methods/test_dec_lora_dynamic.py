"""Dec-LoRA's dynamic-network adaptation (profile ``dynamic_graph_adapted``).

``paper_fixed_q`` mixes over one fixed Q and ``common_trace_pairwise`` is pairwise gossip. This
profile runs Dec-LoRA's graph round on a time-varying graph: it declares no fixed matrix, so the
runner builds a per-frame Metropolis W_t (``execution.graph_mixing: dynamic_metropolis``), and
the run label marks a fixed-graph method run this way as an adaptation. CPU-only, on the toy
adapter.
"""
import json

import pytest

import federatedscope.core.configs.config  # noqa: F401  (global cfg with the extension)
from federatedscope.contrib.decentralized.core.interactions import (
    GRAPH_MIXING_DYNAMIC,
    assert_doubly_stochastic,
    compile_dynamic_graph_round,
)
from federatedscope.contrib.decentralized.core.manifest import build_manifest, run_label
from federatedscope.contrib.decentralized.core.runner import DecentralizedRunner
from federatedscope.contrib.decentralized.methods.dec_lora.config import (
    PROFILE_DYNAMIC_GRAPH_ADAPTED,
)
from federatedscope.contrib.decentralized.tests.core.test_dynamic_graph_round import (
    TIME_VARYING,
    document,
)
from federatedscope.contrib.decentralized.tests.methods.test_dec_lora import ToyLoRAFactory, dec_cfg

DYNAMIC = {"decentralized.execution.graph_mixing": GRAPH_MIXING_DYNAMIC,
           "decentralized.comparison.profile": "common_study"}


def _runner(tmp_path, doc, **overrides):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(doc))
    settings = dict(DYNAMIC, **{"decentralized.opportunities.source": "file",
                                "decentralized.opportunities.path": str(path)})
    settings.update(overrides)
    cfg = dec_cfg(profile=PROFILE_DYNAMIC_GRAPH_ADAPTED, client_num=len(doc["peers"]),
                  frames=len(doc["frames"]), auxiliary="local", **settings)
    return DecentralizedRunner(data={}, config=cfg, factory=ToyLoRAFactory(cfg))


def test_the_adaptation_uses_graph_rounds_and_declares_no_fixed_matrix():
    cfg = dec_cfg(profile=PROFILE_DYNAMIC_GRAPH_ADAPTED, auxiliary="local", **DYNAMIC)
    assert cfg.decentralized.execution.interaction == "graph_round"
    assert ToyLoRAFactory(cfg).fixed_mixing_matrix([1, 2, 3, 4]) is None


def test_a_fixed_q_under_the_adaptation_is_refused():
    with pytest.raises(ValueError, match="fixed_q is the paper_fixed_q experiment"):
        dec_cfg(profile=PROFILE_DYNAMIC_GRAPH_ADAPTED, auxiliary="local",
                **dict(DYNAMIC, **{"decentralized.methods.dec_lora.fixed_q": "ring"}))


def test_the_adaptation_without_the_dynamic_w_t_is_refused():
    with pytest.raises(ValueError, match="needs decentralized.execution.graph_mixing"):
        dec_cfg(profile=PROFILE_DYNAMIC_GRAPH_ADAPTED, auxiliary="local",
                **{"decentralized.comparison.profile": "common_study"})


def test_the_adaptation_is_never_reported_as_paper_native():
    with pytest.raises(ValueError, match="not the paper's fixed-Q method"):
        dec_cfg(profile=PROFILE_DYNAMIC_GRAPH_ADAPTED, auxiliary="local",
                **dict(DYNAMIC, **{"decentralized.comparison.profile": "paper_native"}))


def test_the_simulator_mixes_every_frame_with_that_frames_metropolis_w_t(tmp_path):
    """The toy episode trains nothing, so every peer's adapter mean must follow the product of
    the per-frame matrices exactly - the mean is linear in the state the round mixes."""
    runner = _runner(tmp_path, TIME_VARYING)
    runner._build_peers()
    before = {pid: peer.algorithm.consensus_fingerprint()["adapter_mean"]
              for pid, peer in runner.peers.items()}
    order = tuple(sorted(before))
    means = [before[p] for p in order]
    for frame in runner.trace.frames:
        matrix = compile_dynamic_graph_round(frame, order).matrix
        means = [sum(matrix[i][j] * means[j] for j in range(len(order))) for i in range(len(order))]
    results = runner.run()
    rounds = [e for e in runner.sink.events if e.get("event") == "graph_round"]
    assert len(rounds) == len(TIME_VARYING["frames"])
    for record in rounds:
        assert record["status"] == "committed"
        assert_doubly_stochastic(record["offered_matrix"])
        assert record["realized_matrix"] == record["offered_matrix"], record["skipped_edges"]
    assert rounds[4]["mixed_peers"] == [1, 2, 3, 4, 5, 6]
    for position, pid in enumerate(order):
        assert results["per_peer"][pid]["adapter_mean"] == pytest.approx(means[position], abs=1e-6)


def test_the_run_is_labelled_a_dynamic_network_adaptation(tmp_path):
    manifest = build_manifest(_runner(tmp_path, document([[(1, 2), (1, 3), (3, 4)]])))
    assert manifest["graph_mixing"]["mode"] == GRAPH_MIXING_DYNAMIC
    label = run_label(manifest, native_graph="fixed")
    assert label == "dec-dynamic-graph-adapted"
    assert "native" not in label
