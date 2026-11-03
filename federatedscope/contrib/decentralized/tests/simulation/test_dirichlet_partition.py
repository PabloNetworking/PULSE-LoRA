"""Dirichlet (LDA) partitions (simulation/partition.py): the specification a scenario states, the
GLUE builder applying it, and the index sets, drawn by FederatedScope's ``LDASplitter`` under a
seeded numpy state.

The builder runs on the stand-in for ``get_data`` (``training_fixtures.stub_glue_data``); the
splitter is FederatedScope's own in every test.
"""
import dataclasses
import json

import numpy
import pytest
import torch

from federatedscope.contrib.decentralized.simulation.glue_data import GLUE_BUILDER, glue_partition
from federatedscope.contrib.decentralized.simulation.partition import (
    PartitionError,
    apply_partition_spec,
    labels_of,
    lda_index_sets,
    lda_index_sets_for_splits,
    seeded_numpy,
)
from federatedscope.contrib.decentralized.simulation.scenario import (
    ScenarioError,
    build_cfg,
    build_peer,
    scenario_from_dict,
)
from federatedscope.contrib.decentralized.simulation.training import build_data
from federatedscope.contrib.decentralized.tests.simulation.training_fixtures import (
    PARTITION,
    STUB_SPLITS,
    install_stub_loader,
    stub_glue_data,
    tiny_partition,
    training_scenario,
)

MODULE = "federatedscope.contrib.decentralized.tests.simulation.test_dirichlet_partition"
#: A partition is a function of (labels, alpha, seed, peers), never of the run seed or the method.
SPEC = {"splitter": "lda", "alpha": 0.3, "seed": 11}
METHOD_NAMES = ("dec_lora", "pulse_lora", "tad_lora")


@pytest.fixture
def stub_loader(monkeypatch):
    install_stub_loader(monkeypatch, lambda config=None: (stub_glue_data(config), config))


def _iid_slice(peer_id, split):
    """The contiguous slice the stand-in's IID ``get_data`` hands peer ``peer_id``."""
    size = STUB_SPLITS[split]
    index = int(peer_id) - 1
    return list(range(index * size, (index + 1) * size))


def spec_scenario(method="dec_lora", spec=SPEC, builder=GLUE_BUILDER):
    """The tiny fixture's trainable scenario for ``method``, built by ``builder`` under ``spec``."""
    return dataclasses.replace(training_scenario(method, stochastic=True),
                               data=builder, data_partition=dict(spec) if spec else None)


def _indices(partition, split):
    return list(partition[split].dataset.indices)


# -- the specification in the scenario -----------------------------------------------------

def test_a_dirichlet_partition_spec_is_carried_by_the_scenario_and_moves_its_digest():
    document = spec_scenario(spec=None).to_dict()
    assert document["data_partition"] is None
    without = scenario_from_dict(json.loads(json.dumps(document)))
    assert without.data_partition is None
    document["data_partition"] = dict(SPEC)
    loaded = scenario_from_dict(json.loads(json.dumps(document)))

    assert loaded.data_partition == SPEC
    assert loaded.to_dict()["data_partition"] == SPEC
    assert scenario_from_dict(json.loads(json.dumps(loaded.to_dict()))) == loaded
    assert loaded.digest != without.digest


@pytest.mark.parametrize("broken, match", [
    ({"splitter": "iid", "alpha": 0.3, "seed": 11}, "IID is the absence"),
    ({"splitter": "dirichlet", "alpha": 0.3, "seed": 11}, "not one of"),
    ({"splitter": "lda", "alpha": 0, "seed": 11}, "alpha must be"),
    ({"splitter": "lda", "alpha": -0.3, "seed": 11}, "alpha must be"),
    ({"splitter": "lda", "alpha": True, "seed": 11}, "alpha must be"),
    ({"splitter": "lda", "alpha": "0.3", "seed": 11}, "alpha must be"),
    ({"splitter": "lda", "alpha": float("inf"), "seed": 11}, "alpha must be"),
    ({"splitter": "lda", "alpha": 0.3, "seed": -1}, "seed must be"),
    ({"splitter": "lda", "alpha": 0.3, "seed": 2 ** 32}, "seed must be"),
    ({"splitter": "lda", "alpha": 0.3, "seed": 1.5}, "seed must be"),
    ({"splitter": "lda", "alpha": 0.3, "seed": True}, "seed must be"),
    ({"splitter": "lda", "alpha": 0.3, "seed": 11, "alpha ": 0.1}, "unknown"),
    ({"splitter": "lda", "alpha": 0.3}, "missing"),
    ("lda", "plain mapping"),
    ([0.3, 11], "plain mapping"),
])
def test_a_malformed_partition_spec_is_refused_by_name(broken, match):
    """An unknown key is refused rather than ignored: an ignored ``"alpha "`` would be an IID
    run whose document states an alpha."""
    document = spec_scenario(spec=None).to_dict()
    document["data_partition"] = broken
    with pytest.raises(ScenarioError, match=match):
        scenario_from_dict(json.loads(json.dumps(document)))
    with pytest.raises(ScenarioError, match=match):
        dataclasses.replace(spec_scenario(spec=None), data_partition=broken)


def test_the_spec_is_canonicalised_and_round_trips():
    """``alpha: 1`` and ``alpha: 1.0`` are one document; the digest is over the canonical form."""
    as_int = spec_scenario(spec={"splitter": "lda", "alpha": 1, "seed": 3})
    as_float = spec_scenario(spec={"splitter": "lda", "alpha": 1.0, "seed": 3})
    assert as_int.data_partition == {"splitter": "lda", "alpha": 1.0, "seed": 3}
    assert type(as_int.data_partition["alpha"]) is float
    assert as_int.digest == as_float.digest
    assert scenario_from_dict(json.loads(json.dumps(as_int.to_dict()))) == as_int


def test_a_spec_moves_the_scenario_digest_by_alpha_and_by_seed():
    base = spec_scenario()
    assert base.digest != spec_scenario(spec={**SPEC, "alpha": 0.1}).digest
    assert base.digest != spec_scenario(spec={**SPEC, "seed": 12}).digest
    assert base.digest != spec_scenario(spec=None).digest
    assert base.digest == spec_scenario().digest


# -- the builder ---------------------------------------------------------------------------------

def test_the_builder_applies_a_dirichlet_spec_to_this_peers_splits(stub_loader):
    scenario = spec_scenario()
    partition = glue_partition(scenario, scenario.peers[0], build_cfg(scenario))
    assert _indices(partition, "train") != _iid_slice(scenario.peers[0], "train"), \
        "the builder returned get_data's IID slice"
    assert getattr(partition, "applied_partition", None) == SPEC


def test_a_builder_that_ignores_a_stated_spec_is_refused_by_name():
    """The tiny fixture's builder draws its partitions from private generators and knows
    nothing of a specification. A scenario that states one over it is refused wherever its
    data is built, rather than run as a Dirichlet run that trained IID."""
    scenario = spec_scenario(builder=PARTITION)
    cfg = build_cfg(scenario)
    with pytest.raises(ValueError, match="declares no partition") as refused:
        build_data(scenario, scenario.peers[0], cfg)
    assert "tiny_partition" in str(refused.value) and "apply_partition_spec" in str(refused.value)
    with pytest.raises(ValueError, match="declares no partition"):
        build_peer(scenario, scenario.peers[0], cfg)


def lying_builder(scenario, peer_id, cfg):
    """Applies the stated spec, then declares another."""
    partition = glue_partition(scenario, peer_id, cfg)
    partition.applied_partition = {**scenario.data_partition, "alpha": 1.0}
    return partition


def test_a_builder_that_declares_a_different_spec_is_refused_by_name(stub_loader):
    """The declaration is compared with the stated spec, not merely required."""
    scenario = spec_scenario(builder=f"{MODULE}.lying_builder")
    with pytest.raises(ValueError, match="'alpha': 1.0") as refused:
        build_data(scenario, scenario.peers[0], build_cfg(scenario))
    assert "declares" in str(refused.value)


def test_every_split_is_re_drawn_over_the_loaders_own_full_dataset(stub_loader):
    """Train, val and test are all re-partitioned, as FederatedScope's translator splits every
    split; each new ``Subset`` indexes the very dataset the IID ``Subset`` indexed."""
    scenario = spec_scenario()
    cfg = build_cfg(scenario)
    iid = glue_partition(dataclasses.replace(scenario, data_partition=None),
                         scenario.peers[0], cfg)
    first = glue_partition(scenario, scenario.peers[0], cfg)
    second = glue_partition(scenario, scenario.peers[1], cfg)

    for split in STUB_SPLITS:
        assert _indices(first, split) != _iid_slice(scenario.peers[0], split), split
        assert _indices(second, split) != _iid_slice(scenario.peers[1], split), split
        assert len(first[split].dataset.dataset) == STUB_SPLITS[split] * len(scenario.peers)
        assert type(first[split].dataset.dataset) is type(iid[split].dataset.dataset)
    assert first.applied_partition == second.applied_partition == SPEC


def test_the_peers_of_one_spec_cover_every_item_exactly_once(stub_loader):
    """Across the five peers, each built from its own ``get_data`` call, every split's index
    sets are pairwise disjoint and their union is the whole dataset."""
    scenario = spec_scenario()
    cfg = build_cfg(scenario)
    partitions = {peer: glue_partition(scenario, peer, cfg) for peer in scenario.peers}
    for split, size in STUB_SPLITS.items():
        sets = [set(_indices(partitions[peer], split)) for peer in scenario.peers]
        assert all(sets), f"{split}: an empty peer"
        assert sum(len(s) for s in sets) == size * len(scenario.peers), split
        assert set().union(*sets) == set(range(size * len(scenario.peers))), split


def test_the_same_spec_gives_every_method_identical_partitions(stub_loader):
    scenarios = [spec_scenario(method) for method in METHOD_NAMES]
    assert len({scenario.digest for scenario in scenarios}) == len(METHOD_NAMES)
    partitions = [{peer: {split: _indices(glue_partition(scenario, peer, build_cfg(scenario)), split)
                          for split in STUB_SPLITS}
                   for peer in scenario.peers}
                  for scenario in scenarios]
    assert all(partition == partitions[0] for partition in partitions[1:])


def test_a_partition_that_cannot_be_re_split_is_refused_by_name(stub_loader):
    scenario = spec_scenario()
    cfg = build_cfg(scenario)
    direct = tiny_partition(scenario, 1, cfg)            # its splits are not Subsets
    with pytest.raises(PartitionError, match="not a Subset"):
        apply_partition_spec(direct, SPEC, 1, scenario.peers)
    iid = stub_glue_data(cfg)[1]
    with pytest.raises(PartitionError, match="not one of the peers"):
        apply_partition_spec(iid, SPEC, 9, scenario.peers)


# -- the index sets: a pure function, through FederatedScope's splitter --------------------------

def _labels(items, classes=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, classes, (items,), generator=generator).tolist()


def test_the_index_sets_are_a_function_of_labels_alpha_seed_and_peers():
    labels = _labels(200)
    base = lda_index_sets(labels, 4, 0.3, 11)
    assert lda_index_sets(list(labels), 4, 0.3, 11) == base
    assert lda_index_sets(labels, 4, 0.3, 12) != base, "the seed is ignored"
    assert lda_index_sets(labels, 4, 1.0, 11) != base, "alpha is ignored"
    assert len(lda_index_sets(labels, 5, 0.3, 11)) == 5 and len(base) == 4


def _skew(labels, peers, alpha, seed):
    """Mean over peers of |p_peer - p|: the absolute deviation of each peer's class-1
    proportion from the global class-1 proportion."""
    labels = numpy.asarray(labels)
    global_p = float(labels.mean())
    sets = lda_index_sets(labels.tolist(), peers, alpha, seed)
    return float(numpy.mean([abs(float(labels[list(s)].mean()) - global_p) for s in sets]))


def test_alpha_controls_heterogeneity():
    """2 classes, 200 items, 4 peers, five seeds. Statistic: the mean absolute deviation of the
    peers' class-1 proportions from the global proportion. alpha=0.1 skews more than
    alpha=1.0, on average and for at least four of the five seeds; alpha=0.3 sits between."""
    labels = _labels(200)
    seeds = (0, 1, 2, 3, 4)
    low = [_skew(labels, 4, 0.1, seed) for seed in seeds]
    high = [_skew(labels, 4, 1.0, seed) for seed in seeds]
    assert numpy.mean(low) > numpy.mean(high), (low, high)
    assert sum(a > b for a, b in zip(low, high)) >= 4, (low, high)
    middle = [_skew(labels, 4, 0.3, seed) for seed in seeds]
    assert numpy.mean(low) > numpy.mean(middle) > numpy.mean(high), (low, middle, high)


@pytest.mark.parametrize("peers, alpha, seed, items, classes", [
    (4, 0.1, 0, 200, 2), (4, 1.0, 3, 200, 2), (5, 0.3, 11, 80, 2), (3, 0.1, 7, 60, 3),
    (2, 0.05, 1, 12, 2),
])
def test_every_item_is_assigned_exactly_once_and_every_peer_is_non_empty(
        peers, alpha, seed, items, classes):
    labels = _labels(items, classes, seed)
    sets = lda_index_sets(labels, peers, alpha, seed)
    assert len(sets) == peers
    assert all(len(s) > 0 for s in sets), "an empty peer"
    assert sorted(i for s in sets for i in s) == list(range(items))


def test_labels_with_a_gap_or_too_few_items_are_refused_before_any_draw():
    """The splitter iterates ``range(C)`` over ``label == k``: with labels {0, 2} every item of
    class 2 would be dropped without a word. Refused by name instead."""
    with pytest.raises(PartitionError, match="0..C-1"):
        lda_index_sets([0, 2, 0, 2, 0, 2, 0, 2], 2, 0.3, 1)
    with pytest.raises(PartitionError, match="non-empty peers"):
        lda_index_sets([0, 1, 0, 1], 4, 0.3, 1)
    with pytest.raises(PartitionError, match="unknown splits"):
        lda_index_sets_for_splits({"train": [0, 1, 0, 1, 0], "dev": [0, 1, 0, 1, 0]}, 2, 0.3, 1)


def test_the_labels_and_the_index_sets_are_the_splitters_own():
    """``labels_of`` reads labels as ``LDASplitter`` reads them, and the index sets are the
    splitter's: called directly on the dataset under the same seed it returns the same
    ``Subset`` indices, for GLUE's dict-with-``labels`` items and for ``(x, y)`` tuples."""
    from torch.utils.data import Dataset

    from federatedscope.core.splitters.generic.lda_splitter import LDASplitter

    labels = _labels(120, 3, seed=5)

    class Dicts(Dataset):
        def __len__(self):
            return len(labels)

        def __getitem__(self, index):
            return {"input_ids": torch.zeros(2), "labels": torch.tensor(labels[index])}

    class Tuples(Dataset):
        def __len__(self):
            return len(labels)

        def __getitem__(self, index):
            return (torch.zeros(2), labels[index])

    for dataset in (Dicts(), Tuples()):
        assert labels_of(dataset).tolist() == labels
        with seeded_numpy(9):
            direct = LDASplitter(4, alpha=0.3)(dataset)
        assert tuple(tuple(int(i) for i in s.indices) for s in direct) == \
            lda_index_sets(labels_of(dataset), 4, 0.3, 9)


def test_splits_are_drawn_from_one_stream_in_the_translators_order():
    """FederatedScope's translator seeds once and splits train, then val, then test. So a
    partition's train split is the single-split answer whatever else it holds, val is the
    second draw of the stream (not a fresh one), and dict order does not matter."""
    train, val = _labels(200, seed=1), _labels(80, seed=2)
    both = lda_index_sets_for_splits({"val": val, "train": train}, 4, 0.3, 11)
    assert list(both) == ["train", "val"]
    assert both["train"] == lda_index_sets(train, 4, 0.3, 11)
    assert both["val"] != lda_index_sets(val, 4, 0.3, 11)
    assert both == lda_index_sets_for_splits({"train": train, "val": val}, 4, 0.3, 11)


def test_the_split_leaves_the_numpy_generator_where_it_found_it():
    """The splitter draws on numpy's global generator; the caller's state is restored, after a
    refusal too."""
    numpy.random.seed(5)
    expected = numpy.random.rand()
    numpy.random.seed(5)
    lda_index_sets(_labels(50), 3, 0.3, 11)
    assert numpy.random.rand() == expected
    numpy.random.seed(5)
    with pytest.raises(PartitionError):
        lda_index_sets([0, 2, 0, 2], 2, 0.3, 11)
    assert numpy.random.rand() == expected
