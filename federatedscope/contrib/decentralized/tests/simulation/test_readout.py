"""Readouts (simulation/readout.py): the schedule, the holdout and the forward pass, and the
readout rows the driver writes from them.

The holdout is a fixed set of training examples, identical for every peer and method and
removed from every peer's training data; the official development split is not used.
"""
import json
import os
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from federatedscope.contrib.decentralized.simulation import driver
from federatedscope.contrib.decentralized.simulation.driver import run_scenario
from federatedscope.contrib.decentralized.simulation.readout import (
    HOLDOUT_EMPTIES_A_PEER,
    HOLDOUT_NOT_APPLIED,
    HOLDOUT_TOO_LARGE,
    HOLDOUT_UNSUPPORTED,
    READOUT_SCHEDULE_INVALID,
    ReadoutError,
    apply_holdout,
    capture_due,
    capture_points,
    declare_holdout,
    example_ids_and_labels,
    forward_logits,
    holdout_digest,
    holdout_labels,
    holdout_loader,
    holdout_positions,
    holdout_positions_for,
    logits_sha256,
    validate_readout_schedule,
)
from federatedscope.contrib.decentralized.simulation.scenario import (
    ScenarioError,
    build_cfg,
    build_peer,
    exact_state_digest,
    scenario_from_dict,
)
from federatedscope.contrib.decentralized.simulation.training import build_data
from federatedscope.contrib.decentralized.tests.simulation.training_fixtures import (
    LABELS,
    LOCAL_STEPS,
    VARIED,
    TinyDataset,
    tiny_partition,
    training_scenario,
)

MODULE = "federatedscope.contrib.decentralized.tests.simulation.test_readout"
SCHEDULE = {"every_frames": 2, "initial": True,
            "holdout": {"source": "train", "count": 12, "seed": 5}}
METHODS = ("pulse_lora", "dec_lora", "tad_lora")


def with_schedule(scenario, schedule=SCHEDULE):
    return replace(scenario, readout_schedule=schedule)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# -- the schedule ----------------------------------------------------------------------------

def test_a_schedule_is_canonicalised_carried_by_the_document_and_moves_the_digest():
    plain = training_scenario("pulse_lora", stochastic=True)
    scheduled = with_schedule(plain)
    assert plain.to_dict()["readout_schedule"] is None
    assert scenario_from_dict(plain.to_dict()).readout_schedule is None
    assert scheduled.to_dict()["readout_schedule"] == SCHEDULE
    assert scenario_from_dict(scheduled.to_dict()) == scheduled
    assert scenario_from_dict(scheduled.to_dict()).digest == scheduled.digest
    assert scheduled.digest != plain.digest
    assert with_schedule(plain, dict(SCHEDULE, every_frames=3)).digest != scheduled.digest
    assert with_schedule(plain, dict(SCHEDULE, initial=False)).digest != scheduled.digest
    holdout = dict(SCHEDULE["holdout"], count=13)
    assert with_schedule(plain, dict(SCHEDULE, holdout=holdout)).digest != scheduled.digest


@pytest.mark.parametrize("change", [
    {"extra": 1}, {"every_frames": 0}, {"every_frames": True}, {"every_frames": 2.0},
    {"initial": 1}, {"holdout": [1, 2]},
    {"holdout": {"source": "val", "count": 12, "seed": 5}},
    {"holdout": {"source": "train", "count": 0, "seed": 5}},
    {"holdout": {"source": "train", "count": True, "seed": 5}},
    {"holdout": {"source": "train", "count": 12, "seed": -1}},
    {"holdout": {"source": "train", "count": 12, "seed": 2 ** 32}},
    {"holdout": {"source": "train", "count": 12}},
    {"holdout": {"source": "train", "count": 12, "seed": 5, "fraction": 0.05}},
])
def test_a_malformed_schedule_is_refused_by_name(change):
    """An unknown key is refused rather than ignored: an ignored ``"count "`` would be a run
    that states a holdout size nothing applied."""
    schedule = dict(SCHEDULE, **change)
    with pytest.raises(ReadoutError, match=READOUT_SCHEDULE_INVALID):
        validate_readout_schedule(schedule)
    with pytest.raises(ScenarioError, match=READOUT_SCHEDULE_INVALID):
        with_schedule(training_scenario("pulse_lora", stochastic=True), schedule)


def test_a_missing_key_is_refused_rather_than_defaulted():
    for key in ("every_frames", "initial", "holdout"):
        schedule = {k: v for k, v in SCHEDULE.items() if k != key}
        with pytest.raises(ReadoutError, match=READOUT_SCHEDULE_INVALID):
            validate_readout_schedule(schedule)


def test_capture_points_are_the_initial_state_every_kth_frame_and_the_last():
    assert capture_points(SCHEDULE, 5) == [0, 2, 4, 5]
    assert capture_points(dict(SCHEDULE, initial=False), 4) == [2, 4]
    assert capture_points(dict(SCHEDULE, every_frames=10), 120)[:3] == [0, 10, 20]
    assert capture_points(dict(SCHEDULE, every_frames=7), 20)[-2:] == [14, 20]
    assert capture_due(SCHEDULE, 0, 5) and not capture_due(dict(SCHEDULE, initial=False), 0, 5)
    assert not capture_due(SCHEDULE, 3, 5) and capture_due(SCHEDULE, 5, 5)


def test_a_schedule_on_a_dataset_free_scenario_is_refused_before_any_peer_is_built(tmp_path):
    scenario = replace(training_scenario("pulse_lora", stochastic=True, readout_schedule=SCHEDULE),
                       data=None)
    with pytest.raises(ScenarioError, match="needs data with a holdout"):
        run_scenario(scenario, str(tmp_path / "run"), log=lambda message: None)
    assert not os.path.exists(tmp_path / "run" / "peers")


# -- the holdout -------------------------------------------------------------------------------

def ignoring_builder(scenario, peer_id, cfg):
    """A builder that trains on everything and says nothing about the stated holdout."""
    return tiny_partition(replace(scenario, readout_schedule=None), peer_id, cfg)


def other_holdout_builder(scenario, peer_id, cfg):
    """A builder that applied a different holdout from the one the scenario states."""
    partition = tiny_partition(replace(scenario, readout_schedule=None), peer_id, cfg)
    spec = dict(scenario.readout_schedule["holdout"], count=7)
    return declare_holdout(partition, spec, TinyDataset(torch.zeros(7, 8),
                                                       torch.zeros(7, dtype=torch.long)))


@pytest.mark.parametrize("builder", ["ignoring_builder", "other_holdout_builder"])
def test_a_builder_that_did_not_apply_the_stated_holdout_is_refused_by_name(builder):
    scenario = replace(with_schedule(training_scenario("pulse_lora", stochastic=True)),
                       data=f"{MODULE}.{builder}")
    with pytest.raises(ReadoutError, match=HOLDOUT_NOT_APPLIED):
        build_data(scenario, 1, build_cfg(scenario, 1))


def test_holdout_positions_are_a_pure_function_and_leave_every_global_generator_alone():
    np.random.seed(123)
    before = np.random.get_state()
    torch_before = torch.get_rng_state()
    first = holdout_positions(1000, 50, 7)
    assert first == holdout_positions(1000, 50, 7)
    assert first != holdout_positions(1000, 50, 8)
    assert len(first) == len(set(first)) == 50 and list(first) == sorted(first)
    assert all(0 <= i < 1000 for i in first)
    after = np.random.get_state()
    assert after[0] == before[0] and np.array_equal(after[1], before[1]) and after[2:] == before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    with pytest.raises(ReadoutError, match=HOLDOUT_TOO_LARGE):
        holdout_positions(50, 50, 7)


def _split_partitions(peers=4, size=100):
    """Four peers' ClientData over one full dataset, as ``split_to_client`` hands them out."""
    from torch.utils.data import Subset

    from federatedscope.core.data.base_data import ClientData

    scenario = training_scenario("pulse_lora", stochastic=False)
    cfg = build_cfg(scenario, 1)
    generator = torch.Generator().manual_seed(3)
    full = TinyDataset(torch.randn(size, 8, generator=generator),
                       torch.randint(0, 2, (size,), generator=generator))
    val = TinyDataset(torch.randn(40, 8, generator=generator),
                      torch.randint(0, 2, (40,), generator=generator))
    order = torch.randperm(size, generator=generator).tolist()
    shares = [sorted(order[p::peers]) for p in range(peers)]
    return full, [ClientData(cfg, train=Subset(full, share),
                             val=Subset(val, list(range(p * 10, p * 10 + 10))))
                  for p, share in enumerate(shares)]


def test_the_holdout_is_one_shared_set_subtracted_from_every_peer_after_the_split():
    full, partitions = _split_partitions()
    spec = {"source": "train", "count": 17, "seed": 9}
    held = apply_holdout(partitions[0], spec).readout_holdout.indices
    rebuilt = []
    for partition in partitions:
        partition.applied_partition = {"splitter": "lda", "alpha": 0.3, "seed": 1}
        after = apply_holdout(partition, spec)
        rebuilt.append(after)
        train = set(after["train"].dataset.indices)
        assert after.readout_holdout.indices == held
        assert not train & set(held), "a peer trains on a holdout example"
        assert train == set(partition["train"].dataset.indices) - set(held)
        assert after["val"].dataset.indices == partition["val"].dataset.indices
        assert after.applied_partition == partition.applied_partition
        assert after.applied_holdout["spec"] == spec and after.applied_holdout["size"] == 17
    assert len({after.applied_holdout["digest"] for after in rebuilt}) == 1
    union = set().union(*(set(after["train"].dataset.indices) for after in rebuilt))
    assert union | set(held) == set(range(len(full))) and not union & set(held)


def test_a_holdout_that_would_empty_a_peer_is_refused():
    from torch.utils.data import Subset

    from federatedscope.core.data.base_data import ClientData

    full, partitions = _split_partitions()
    spec = {"source": "train", "count": 17, "seed": 9}
    held = holdout_positions(len(full), 17, 9)
    only_held = ClientData(partitions[0].client_cfg, train=Subset(full, list(held[:3])))
    with pytest.raises(ReadoutError, match=HOLDOUT_EMPTIES_A_PEER):
        apply_holdout(only_held, spec)


def test_the_holdout_digest_covers_positions_and_contents():
    from torch.utils.data import Subset

    full, _ = _split_partitions()
    a = holdout_digest(Subset(full, [1, 2, 3]))
    assert a == holdout_digest(Subset(full, [1, 2, 3]))
    assert a != holdout_digest(Subset(full, [1, 2, 4]))
    changed = TinyDataset(full.features.clone(), full.labels.clone())
    changed.features[2, 0] += 1.0
    assert a != holdout_digest(Subset(changed, [1, 2, 3]))


def test_a_holdout_that_cannot_be_drawn_or_identified_is_refused_by_name():
    from torch.utils.data import Subset

    full, _ = _split_partitions()
    spec = {"source": "train", "count": 5, "seed": 1}
    scenario = training_scenario("dec_lora", stochastic=False)
    direct = tiny_partition(scenario, 1, build_cfg(scenario, 1))
    with pytest.raises(ReadoutError, match=HOLDOUT_UNSUPPORTED):
        apply_holdout(direct, spec)          # its training split is not a Subset of a full split
    with pytest.raises(ReadoutError, match=HOLDOUT_UNSUPPORTED):
        example_ids_and_labels(full)          # no raw examples, so no original ids
    repeated = TinyDataset(full.features, full.labels)
    repeated.examples = [{"idx": 1, "label": 0}, {"idx": 1, "label": 1}]
    with pytest.raises(ReadoutError, match=HOLDOUT_UNSUPPORTED):
        example_ids_and_labels(repeated)
    with pytest.raises(ReadoutError, match=HOLDOUT_UNSUPPORTED):
        holdout_positions_for(full, dict(spec, rule="random_v0"))
    with pytest.raises(ReadoutError, match=HOLDOUT_UNSUPPORTED):
        holdout_digest(Subset([{"input_ids": object()}], [0]))
    with pytest.raises(ReadoutError, match=HOLDOUT_NOT_APPLIED):
        holdout_loader(direct)


@pytest.mark.parametrize("method", METHODS)
def test_the_tiny_holdout_is_shared_by_every_peer_and_method_and_leaves_training_alone(method):
    plain = training_scenario(method, stochastic=True)
    scheduled = with_schedule(plain)
    digests = set()
    for peer in plain.peers:
        before = build_data(plain, peer, build_cfg(plain, peer))
        after = build_data(scheduled, peer, build_cfg(scheduled, peer))
        for split in ("train", "val"):
            assert torch.equal(before[split].dataset.features, after[split].dataset.features)
        digests.add(after.applied_holdout["digest"])
    assert len(digests) == 1
    reference = with_schedule(training_scenario("pulse_lora", stochastic=True))
    assert digests == {build_data(reference, 1, build_cfg(reference, 1)).applied_holdout["digest"]}


# -- the forward pass ------------------------------------------------------------------------

def trained_peer(method="pulse_lora"):
    scenario = training_scenario(method, stochastic=True, readout_schedule=SCHEDULE)
    cfg = build_cfg(scenario, 1)
    data = build_data(scenario, 1, cfg)
    peer = build_peer(scenario, 1, cfg, data=data)
    peer.local_episode(LOCAL_STEPS)
    return peer, data


def test_forward_logits_reads_the_model_without_changing_it():
    """Logits in loader order, float32, the model's own evaluation forward; the model's mode
    and state are left as they were. Numpy's and Python's generators are not drawn from
    (iterating a data loader does draw from torch's, which is why readouts inside a run need
    the per-peer episode rule)."""
    peer, data = trained_peer()
    model = peer.trainer.ctx.model
    count = SCHEDULE["holdout"]["count"]
    seen = []
    for mode in (True, False):
        model.train(mode)
        exact = exact_state_digest(peer)
        numpy_state, python_state = np.random.get_state(), random.getstate()
        logits = forward_logits(peer, holdout_loader(data))
        assert model.training is mode
        assert exact_state_digest(peer) == exact
        assert np.array_equal(np.random.get_state()[1], numpy_state[1])
        assert random.getstate() == python_state
        assert logits.shape == (count, LABELS) and logits.dtype == np.float32
        seen.append(logits)
    assert logits_sha256(seen[0]) == logits_sha256(seen[1])

    model.eval()
    with torch.no_grad():
        expected = torch.cat([model(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"]).logits
                              for batch in holdout_loader(data)])
    assert np.array_equal(seen[0], expected.numpy())
    assert len(holdout_labels(data)) == count


def test_the_holdout_loader_keeps_its_order_and_takes_its_own_batch_size():
    peer, data = trained_peer("dec_lora")
    configured = int(data.client_cfg.dataloader.batch_size)
    small = holdout_loader(data, batch_size=5)
    assert [len(batch["labels"]) for batch in small] == [5, 5, 2]
    assert int(data.client_cfg.dataloader.batch_size) == configured
    labels = torch.cat([batch["labels"] for batch in small]).numpy()
    assert np.array_equal(labels, holdout_labels(data))
    assert np.allclose(forward_logits(peer, small), forward_logits(peer, holdout_loader(data)),
                       atol=1e-6)


def test_the_logits_digest_covers_dtype_shape_and_values():
    logits = np.arange(6, dtype=np.float32).reshape(3, 2)
    assert logits_sha256(logits) == logits_sha256(logits.copy())
    assert logits_sha256(logits) != logits_sha256(logits.astype(np.float64))
    assert logits_sha256(logits) != logits_sha256(logits.reshape(2, 3))
    changed = logits.copy()
    changed[2, 1] += 1.0
    assert logits_sha256(logits) != logits_sha256(changed)


# -- the readout rows of a run -----------------------------------------------------------------

def test_readout_counters_start_at_zero_accumulate_and_end_at_the_run_totals(tmp_path):
    out = str(tmp_path / "run")
    schedule = dict(SCHEDULE, every_frames=1)
    summary = run_scenario(training_scenario("dec_lora", stochastic=True, readout_schedule=schedule),
                           out, log=lambda message: None)
    rows = read_jsonl(os.path.join(out, "readouts.jsonl"))
    frames = len(VARIED["frames"])
    assert [row["frames_completed"] for row in rows] == list(range(frames + 1))
    peers = [str(p) for p in VARIED["peers"]]
    for counter in rows[0]["per_peer"].values():
        assert set(counter.values()) == {0}
    for earlier, later in zip(rows, rows[1:]):
        assert later["upload_bytes_total"] >= earlier["upload_bytes_total"]
        for peer in peers:
            a, b = earlier["per_peer"][peer], later["per_peer"][peer]
            assert b["local_episodes"] == later["frames_completed"]
            assert all(b[key] >= a[key] for key in a), (peer, a, b)
    assert rows[-1]["upload_bytes_total"] == summary["upload_bytes_total"] > 0
    for peer in peers:
        record = json.load(open(os.path.join(out, "peers", f"peer{peer}.json")))
        last = rows[-1]["per_peer"][peer]
        assert last["optimizer_steps"] == sum(e["optimizer_steps"] for e in record["episodes"])
        assert last["upload_bytes"] == record["traffic"]["upload_bytes"]
        assert last["download_bytes"] == record["traffic"]["download_bytes"]


def test_elapsed_time_excludes_the_readouts_own_time(tmp_path, monkeypatch):
    """A clock that advances only while a peer is read out: every row's elapsed time is then
    zero, and the readout time is all of the run's time."""
    class Clock:
        now = 0.0

        def perf_counter(self):
            return self.now

    clock = Clock()
    read = driver.forward_logits

    def slow(*args, **kwargs):
        clock.now += 100.0
        return read(*args, **kwargs)

    monkeypatch.setattr(driver, "time", clock)
    monkeypatch.setattr(driver, "forward_logits", slow)
    out = str(tmp_path / "run")
    summary = run_scenario(training_scenario("pulse_lora", stochastic=True, readout_schedule=SCHEDULE),
                           out, log=lambda message: None)
    rows = read_jsonl(os.path.join(out, "readouts.jsonl"))
    assert [row["elapsed_s"] for row in rows] == [0.0] * len(rows)
    assert summary["readout_s"] == summary["wall_s"] == 100.0 * len(rows) * len(VARIED["peers"])
