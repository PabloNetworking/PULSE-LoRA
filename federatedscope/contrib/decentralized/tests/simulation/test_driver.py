"""The driver end to end on the tiny trainable fixture (simulation/driver.py)."""
import dataclasses
import json
import os

import numpy as np
import pytest

from federatedscope.contrib.decentralized.simulation.driver import (
    probability_ensemble,
    run_scenario,
    softmax,
)
from federatedscope.contrib.decentralized.simulation.scenario import ScenarioError
from federatedscope.contrib.decentralized.tests.simulation.training_fixtures import (
    VARIED,
    training_scenario,
)

SCHEDULE = {"every_frames": 2, "initial": True, "holdout": {"source": "train", "count": 12, "seed": 5}}
METHODS = ("pulse_lora", "dec_lora", "tad_lora")


def scenario(method, schedule=SCHEDULE, **kwargs):
    return training_scenario(method, stochastic=True, readout_schedule=schedule, **kwargs)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finals(out):
    return {p: json.load(open(os.path.join(out, "peers", f"peer{p}.json")))["final_learned_tensor_digest"]
            for p in VARIED["peers"]}


@pytest.mark.parametrize("method", METHODS)
def test_a_run_writes_its_records(method, tmp_path):
    out = str(tmp_path / "run")
    summary = run_scenario(scenario(method), out, log=lambda message: None)
    frames = len(VARIED["frames"])
    assert summary["frames"] == frames and summary["method"] == method
    captures = read_jsonl(os.path.join(out, "readouts.jsonl"))
    assert [c["frames_completed"] for c in captures] == [0, 2, 4, 5]
    for capture in captures:
        assert 0.0 <= capture["accuracy"] <= 1.0
        assert set(capture["per_peer_accuracy"]) == {str(p) for p in VARIED["peers"]}
    rows = read_jsonl(os.path.join(out, "sessions.jsonl"))
    assert summary["exchanges"] == len(rows) > 0
    assert summary["upload_bytes_total"] == sum(
        d["priced"] for row in rows for d in row["directions"].values())
    labels = np.load(os.path.join(out, "readout", "labels.npy"))
    logits = np.load(os.path.join(out, "readout", "peer1", "frame0005.npy"))
    assert logits.shape == (len(labels), 2)
    for p in VARIED["peers"]:
        record = json.load(open(os.path.join(out, "peers", f"peer{p}.json")))
        assert len(record["episodes"]) == frames
        assert record["method_metrics"]
    assert summary["accuracy"]["final"] == captures[-1]["accuracy"]


@pytest.mark.parametrize("method", METHODS)
def test_readouts_leave_training_unchanged(method, tmp_path):
    """Under the per-peer episode rule, a run with readouts trains exactly as without them."""
    with_readouts = str(tmp_path / "with")
    without = str(tmp_path / "without")
    run_scenario(scenario(method), with_readouts, log=lambda message: None)
    run_scenario(scenario(method, schedule=None), without, log=lambda message: None)
    assert finals(with_readouts) == finals(without)
    assert read_jsonl(os.path.join(with_readouts, "sessions.jsonl")) == \
        read_jsonl(os.path.join(without, "sessions.jsonl"))


def test_a_run_is_reproducible(tmp_path):
    first, second = str(tmp_path / "a"), str(tmp_path / "b")
    run_scenario(scenario("pulse_lora"), first, log=lambda message: None)
    run_scenario(scenario("pulse_lora"), second, log=lambda message: None)
    assert finals(first) == finals(second)
    for p in VARIED["peers"]:
        for name in os.listdir(os.path.join(first, "readout", f"peer{p}")):
            assert np.array_equal(np.load(os.path.join(first, "readout", f"peer{p}", name)),
                                  np.load(os.path.join(second, "readout", f"peer{p}", name)))


def test_readouts_without_the_per_peer_episode_rule_are_refused(tmp_path):
    with pytest.raises(ScenarioError, match="per_peer_episode"):
        run_scenario(scenario("dec_lora", training_rng=None), str(tmp_path / "run"))


def test_readouts_without_data_are_refused_before_anything_is_written(tmp_path):
    out = tmp_path / "run"
    with pytest.raises(ScenarioError, match="data"):
        run_scenario(dataclasses.replace(scenario("dec_lora"), data=None), str(out))
    assert not out.exists()


def test_an_existing_output_directory_is_refused(tmp_path):
    with pytest.raises(FileExistsError):
        run_scenario(scenario("dec_lora"), str(tmp_path))


def test_the_saved_state_holds_no_optimizer(tmp_path):
    import torch

    out = str(tmp_path / "run")
    run_scenario(scenario("pulse_lora"), out, save_state=True, log=lambda message: None)
    state = torch.load(os.path.join(out, "final", "peer1.pt"), map_location="cpu")
    assert "adapter" in state and "trainer" not in state


def test_the_adas_c_controller_runs_end_to_end(tmp_path):
    base = scenario("pulse_lora")
    overrides = dict(base.cfg_overrides)
    overrides.update({"decentralized.methods.pulse_lora.controller": "adas_c",
                      "decentralized.methods.pulse_lora.prune_gamma": 0.5,
                      "decentralized.methods.pulse_lora.rank_min": 1})
    out = str(tmp_path / "run")
    run_scenario(dataclasses.replace(base, cfg_overrides=overrides), out, log=lambda message: None)
    for p in VARIED["peers"]:
        record = json.load(open(os.path.join(out, "peers", f"peer{p}.json")))
        ranks = [e["active_rank"] for e in record["episodes"]]
        assert ranks == sorted(ranks, reverse=True), "pruning is monotone"
        assert record["method_metrics"]["pulse_controller"] == "adas_c"


def test_the_ensemble_averages_probabilities():
    logits = {1: np.array([[2.0, 0.0]]), 2: np.array([[0.0, 1.0]])}
    expected = (softmax(logits[1]) + softmax(logits[2])) / 2
    assert np.allclose(probability_ensemble(logits), expected)
