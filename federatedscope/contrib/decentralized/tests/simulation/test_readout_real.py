"""Readouts on real GLUE data (RTE) and a real backbone (DeBERTa).

The readout holdout is one set of training examples, drawn over the full training split and
removed from every peer after the IID, Dirichlet or label-stratified split: identical across
peers and methods, disjoint from what any peer trains on, and it leaves the validation split as
the split made it. The data is built as the driver builds it (``glue_data.central_glue_data``) and agrees
with the per-peer builder. Under the common construction every peer of every method starts
from the same model, so their initial readouts agree.

Slow and opt-in: the scenarios are the reference ones of ``scripts/make_scenario.py`` on RTE,
reduced to two peers, and they need the GLUE cache and a local model directory.

    PULSE_GLUE_DATA_ROOT=data/ PULSE_MODEL_DIR=models/deberta-base HF_DATASETS_OFFLINE=1 \\
        python -m pytest federatedscope/contrib/decentralized/tests/simulation/test_readout_real.py
"""
import importlib.util
import os
from dataclasses import replace

import numpy as np
import pytest

DATA_ROOT = os.environ.get("PULSE_GLUE_DATA_ROOT")
MODEL_DIR = os.environ.get("PULSE_MODEL_DIR")

pytestmark = pytest.mark.skipif(
    not (DATA_ROOT and MODEL_DIR),
    reason="set PULSE_GLUE_DATA_ROOT (the GLUE cache) and PULSE_MODEL_DIR (a local DeBERTa "
           "directory) to run the tests on real GLUE data",
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 5))
HOLDOUT = {"source": "train", "count": 64, "seed": 3}
LDA = {"splitter": "lda", "alpha": 0.3, "seed": 11}
VARIANTS = ("iid", "dirichlet", "stratified")
METHODS = ("dec_lora", "pulse_lora", "tad_lora")


@pytest.fixture(scope="module")
def offline():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HF_DATASETS_OFFLINE", "1")
        patch.setenv("CUDA_VISIBLE_DEVICES", "")
        yield


def _make_scenario():
    spec = importlib.util.spec_from_file_location(
        "make_scenario", os.path.join(REPO, "scripts", "make_scenario.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def real_scenario(method, variant):
    """The reference scenario of ``method`` on RTE with two peers at rank 8 and a 64-example
    holdout; ``variant`` chooses the split. The reference split is the label-stratified one."""
    script = _make_scenario()
    args = script.parse_args([
        method, "--out", os.devnull, "--task", "rte", "--model", MODEL_DIR,
        "--data-root", DATA_ROOT, "--peers", "2", "--frames", "1", "--local-steps", "2",
        "--batch-size", "4", "--max-length", "32", "--ranks", "8,8", "--r-max", "8",
        "--rank", "8", "--holdout", str(HOLDOUT["count"]), "--holdout-seed", str(HOLDOUT["seed"])])
    scenario = script.build(args)
    if variant == "stratified":
        return scenario
    schedule = dict(scenario.readout_schedule, holdout=dict(HOLDOUT))
    return replace(scenario, data_partition=LDA if variant == "dirichlet" else None,
                   readout_schedule=schedule)


def _central(scenario):
    from federatedscope.contrib.decentralized.simulation.glue_data import central_glue_data
    from federatedscope.contrib.decentralized.simulation.scenario import build_cfg

    return central_glue_data(scenario, build_cfg(scenario))


def _indices(partition, split):
    return list(partition[split].dataset.indices)


@pytest.mark.parametrize("variant", VARIANTS)
def test_slow_the_glue_holdout_is_shared_disjoint_and_removed_after_the_split(offline, variant):
    from federatedscope.contrib.decentralized.simulation.glue_data import glue_partition
    from federatedscope.contrib.decentralized.simulation.readout import holdout_positions_for
    from federatedscope.contrib.decentralized.simulation.scenario import build_cfg

    scheduled = real_scenario("dec_lora", variant)
    spec = scheduled.readout_schedule["holdout"]
    after = _central(scheduled)
    peers = sorted(after)
    full = after[peers[0]]["train"].dataset.dataset
    held = list(holdout_positions_for(full, spec))
    assert len(held) == spec["count"] == len(set(held))
    trains = {peer: set(_indices(after[peer], "train")) for peer in peers}
    for peer in peers:
        assert list(after[peer].readout_holdout.indices) == held
        assert not trains[peer] & set(held), "a peer trains on a holdout example"
        assert getattr(after[peer], "applied_partition", None) == scheduled.data_partition
    assert len({after[peer].applied_holdout["digest"] for peer in peers}) == 1
    assert sum(len(train) for train in trains.values()) + len(held) == len(full)
    assert set().union(*trains.values()) | set(held) == set(range(len(full)))

    if variant == "stratified":
        # The stratified split deals training examples only: validation stays as
        # FederatedScope's IID split shards it.
        reference = _central(replace(scheduled, data_partition=None, readout_schedule=None))
        sizes = [len(train) for train in trains.values()]
        assert max(sizes) - min(sizes) <= 1
    else:
        reference = _central(replace(scheduled, readout_schedule=None))
        for peer in peers:
            assert trains[peer] == set(_indices(reference[peer], "train")) - set(held)
    for peer in peers:
        assert _indices(after[peer], "val") == _indices(reference[peer], "val")

    alone = glue_partition(scheduled, peers[-1], build_cfg(scheduled, peers[-1]))
    assert _indices(alone, "train") == _indices(after[peers[-1]], "train")
    assert alone.applied_holdout == after[peers[-1]].applied_holdout
    other = _central(real_scenario("pulse_lora", variant))
    for peer in peers:
        assert _indices(other[peer], "train") == _indices(after[peer], "train")
        assert other[peer].applied_holdout == after[peer].applied_holdout


def test_slow_every_peer_of_every_method_starts_from_one_model(offline):
    """Every LoRA factor starts at its canonical value with B zero, and the classifier and the
    pooler are copied from one donor model, so before any training every peer of every method
    reads out the same logits on the holdout."""
    from federatedscope.contrib.decentralized.simulation.readout import (
        forward_logits,
        holdout_labels,
        holdout_loader,
    )
    from federatedscope.contrib.decentralized.simulation.scenario import build_peer

    logits = {}
    for method in METHODS:
        scenario = real_scenario(method, "stratified")
        data = _central(scenario)
        for peer in sorted(data):
            algorithm = build_peer(scenario, peer, data=data[peer])
            logits[(method, peer)] = forward_logits(algorithm, holdout_loader(data[peer]))
            assert len(holdout_labels(data[peer])) == HOLDOUT["count"]
    first = logits[("dec_lora", 1)]
    assert first.shape == (HOLDOUT["count"], 2) and first.dtype == np.float32
    for key, value in logits.items():
        assert np.array_equal(value, first), key
