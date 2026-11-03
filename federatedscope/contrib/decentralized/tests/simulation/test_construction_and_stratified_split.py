"""How ``construct_peer`` hands a scenario's common construction to the method, and the
label-stratified readout holdout and training split.

  - the construction (``{"mode": "common_canonical_v1", "model_init_seed": s}``) is in the
    method's context before it builds the peer, and is written into the built model after;
  - the label-stratified holdout (``readout_schedule.holdout.rule``: ``label_stratified_hash_v1``);
  - the balanced label-stratified round-robin split (``data_partition`` ``label_stratified_
    round_robin_v1``): exact coverage, balance, disjointness and no overlap with the holdout.
"""
import hashlib
import types

import pytest
import torch

import federatedscope.contrib.decentralized.lora.canonical_init as canonical_init
from federatedscope.contrib.decentralized.core.interfaces import PeerContext
from federatedscope.contrib.decentralized.simulation import partition as P
from federatedscope.contrib.decentralized.simulation import readout as R
from federatedscope.contrib.decentralized.simulation.scenario import (
    ScenarioError,
    build_cfg,
    build_factory,
    construct_peer,
    scenario_from_dict,
)
from federatedscope.contrib.decentralized.simulation.training import build_data
from federatedscope.contrib.decentralized.tests.simulation.training_fixtures import (
    training_scenario,
)

STRATIFIED_HOLDOUT = {"source": "train", "count": 2000, "seed": 39031,
                      "rule": "label_stratified_hash_v1"}
STRATIFIED_SPLIT = {"splitter": "label_stratified_round_robin_v1", "seed": 31}
MNLI_COUNTS = {0: 130899, 1: 130900, 2: 130903}
METHODS = ("pulse_lora", "dec_lora", "tad_lora")


class IdDataset(torch.utils.data.Dataset):
    """GLUE-shaped: raw ``examples`` with the original ``idx`` and ``label``; tensor items."""

    def __init__(self, labels, ids):
        self.examples = [{"idx": int(i), "label": int(c)} for i, c in zip(ids, labels)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, position):
        example = self.examples[position]
        return {"input_ids": torch.tensor([example["idx"]]),
                "labels": torch.tensor(example["label"])}


def synthetic(n=3001, classes=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, classes, (n,), generator=generator).tolist()
    ids = (torch.randperm(n, generator=generator) * 7 + 11).tolist()       # not positions
    return IdDataset(labels, ids)


# -- construct_peer: the construction reaches the method -------------------------------------

def _context(scenario, peer_id):
    cfg = build_cfg(scenario, peer_id)
    data = build_data(scenario, peer_id, cfg)
    return PeerContext(peer_id=peer_id, config=cfg, data=data, device="cpu", extras={})


@pytest.mark.parametrize("method", METHODS)
def test_construct_peer_supplies_the_construction_before_create_peer_and_applies_it_after(
        method, monkeypatch):
    """The construction is in the context when the method builds the peer (PULSE-LoRA binds
    its lazy initial values to it) and is applied to the built model before any optimizer
    exists: every factor of the built model is then the canonical value."""
    calls = []
    apply = canonical_init.CommonConstruction.apply

    def spy(self, algorithm, context, *, who, donor_builder=None):
        calls.append(("apply", who, context.extras.get(canonical_init.EXTRAS_KEY) is self))
        return apply(self, algorithm, context, who=who, donor_builder=donor_builder)

    monkeypatch.setattr(canonical_init.CommonConstruction, "apply", spy)
    scenario = training_scenario(method, stochastic=True)
    context = _context(scenario, 2)
    create_peer = build_factory(scenario, build_cfg(scenario)).create_peer

    def create(ctx):
        construction = ctx.extras.get(canonical_init.EXTRAS_KEY)
        calls.append(("create", isinstance(construction, canonical_init.CommonConstruction)
                      and construction.spec == scenario.construction))
        return create_peer(ctx)

    algorithm = construct_peer(scenario, 2, context, create)
    assert calls == [("create", True), ("apply", "peer 2", True)]

    construction = context.extras[canonical_init.EXTRAS_KEY]
    assert construction.backbone_digest == hashlib.sha256(
        f"adapter:{scenario.adapter}".encode("utf-8")).hexdigest(), \
        "a scenario's adapter is identified by its dotted path"
    bridge, ids = algorithm.common_construction_targets()
    init = construction.bind(bridge.schema)
    for spec in bridge.schema.modules:
        tensors = bridge.read([spec.a_key, spec.b_key])
        expected = torch.stack([init.a_row(spec.module, c) for c in ids])
        assert torch.equal(tensors[spec.a_key], expected), spec.module
        assert not torch.any(tensors[spec.b_key]), spec.module


def test_construct_peer_refuses_a_method_that_cannot_apply_the_construction():
    scenario = training_scenario("dec_lora", stochastic=True)
    context = _context(scenario, 2)

    class Unconstructible:
        trainer = None

    with pytest.raises(ScenarioError, match="does not implement common_canonical_v1"):
        construct_peer(scenario, 2, context, lambda ctx: Unconstructible())


def test_construct_peer_refuses_a_model_built_on_a_card():
    """Models are built on the host; the driver moves them to the compute device to train."""
    scenario = training_scenario("dec_lora", stochastic=True)
    context = PeerContext(peer_id=2, config=types.SimpleNamespace(device=0), extras={})
    with pytest.raises(ScenarioError, match="built on the host"):
        construct_peer(scenario, 2, context, lambda ctx: pytest.fail("create_peer was called"))


# -- the stratified split needs the stratified holdout ---------------------------------------

def test_the_stratified_split_needs_the_stratified_holdout():
    document = dict(training_scenario("pulse_lora", stochastic=True).to_dict(),
                    data_partition=dict(STRATIFIED_SPLIT))
    with pytest.raises(ScenarioError, match="label_stratified_hash_v1"):
        scenario_from_dict(document)
    seeded = {k: v for k, v in STRATIFIED_HOLDOUT.items() if k != "rule"}
    document["readout_schedule"] = {"every_frames": 5, "initial": True, "holdout": seeded}
    with pytest.raises(ScenarioError, match="label_stratified_hash_v1"):
        scenario_from_dict(document)
    document["readout_schedule"]["holdout"] = dict(STRATIFIED_HOLDOUT)
    scenario = scenario_from_dict(document)
    assert scenario.data_partition == STRATIFIED_SPLIT
    assert scenario.to_dict()["readout_schedule"]["holdout"] == STRATIFIED_HOLDOUT


# -- the readout holdout rule --------------------------------------------------------------------

def test_the_holdout_rule_is_optional_and_its_absence_is_the_seeded_permutation():
    seeded = {"every_frames": 5, "initial": True,
              "holdout": {"source": "train", "count": 2000, "seed": 21}}
    assert R.validate_readout_schedule(seeded) == seeded
    stated = dict(seeded, holdout=STRATIFIED_HOLDOUT)
    assert R.validate_readout_schedule(stated)["holdout"] == STRATIFIED_HOLDOUT
    for bad in (dict(STRATIFIED_HOLDOUT, rule="random"), dict(STRATIFIED_HOLDOUT, extra=1)):
        with pytest.raises(R.ReadoutError, match=R.READOUT_SCHEDULE_INVALID):
            R.validate_readout_schedule(dict(seeded, holdout=bad))
    full = synthetic(n=500)
    plain = seeded["holdout"]
    assert R.holdout_positions_for(full, dict(plain, count=40)) == R.holdout_positions(500, 40, 21)


def test_mnli_quotas_are_the_largest_remainders():
    assert R.stratified_quotas(MNLI_COUNTS, 2000) == {0: 666, 1: 667, 2: 667}
    assert sum(MNLI_COUNTS.values()) == 392702
    # equal remainders: the lower label id takes the extra slot first
    assert R.stratified_quotas({0: 5, 1: 5, 2: 5}, 4) == {0: 2, 1: 1, 2: 1}
    with pytest.raises(R.ReadoutError, match=R.HOLDOUT_TOO_LARGE):
        R.stratified_quotas({0: 5, 1: 5}, 10)


def test_the_stratified_holdout_is_the_specified_recipe():
    full = synthetic()
    spec = dict(STRATIFIED_HOLDOUT, count=300)
    positions = R.stratified_holdout_positions(full, spec)
    ids = [e["idx"] for e in full.examples]
    labels = [e["label"] for e in full.examples]
    counts = {c: labels.count(c) for c in sorted(set(labels))}
    quotas = R.stratified_quotas(counts, 300)
    expected = []
    for label in sorted(counts):
        members = [p for p in range(len(full)) if labels[p] == label]
        members.sort(key=lambda p: (hashlib.sha256(
            f"pulse-h4-holdout-v1|39031|example={ids[p]}".encode("utf-8")).digest(), ids[p]))
        expected.extend(members[:quotas[label]])
    expected.sort(key=lambda p: ids[p])
    assert list(positions) == expected
    assert [ids[p] for p in positions] == sorted(ids[p] for p in positions), "ascending original id"
    assert R.stratified_holdout_positions(full, dict(spec, seed=1)) != positions
    assert R.holdout_positions_for(full, spec) == positions


# -- the balanced split -----------------------------------------------------------------------------

def test_the_split_is_the_specified_deal_complete_balanced_disjoint_and_held_out():
    full = synthetic()
    holdout = dict(STRATIFIED_HOLDOUT, count=300)
    sets = P.stratified_index_sets(full, 10, 31, holdout)
    held = set(R.stratified_holdout_positions(full, holdout))
    every = [p for chosen in sets for p in chosen]
    assert len(every) == len(set(every)), "pairwise disjoint"
    assert set(every) | held == set(range(len(full))) and not set(every) & held
    sizes = [len(chosen) for chosen in sets]
    assert max(sizes) - min(sizes) <= 1
    labels = [e["label"] for e in full.examples]
    for label in set(labels):
        per = [sum(1 for p in chosen if labels[p] == label) for chosen in sets]
        assert max(per) - min(per) <= 1, f"label {label}"
    # the recipe, recomputed: labels in order, sha256 order within a label, one carried offset
    ids = [e["idx"] for e in full.examples]
    sequence = []
    for label in sorted(set(labels)):
        members = [p for p in range(len(full)) if labels[p] == label and p not in held]
        members.sort(key=lambda p: (hashlib.sha256(
            f"pulse-h4-iid-v1|seed=31|label={label}|example={ids[p]}".encode("utf-8")).digest(),
            ids[p]))
        sequence.extend(members)
    assert [tuple(sorted(sequence[k::10])) for k in range(10)] == list(sets)
    assert P.stratified_index_sets(full, 10, 32, holdout) != sets


def test_on_mnli_sizes_the_deal_gives_eight_peers_39070_and_two_39071():
    total = 392702 - 2000
    sizes = [len(range(k, total, 10)) for k in range(10)]
    assert sorted(sizes) == [39070] * 8 + [39071] * 2


def _client_data(full, val=None):
    from federatedscope.core.configs.config import global_cfg
    from federatedscope.core.data.base_data import ClientData

    cfg = global_cfg.clone()
    cfg.merge_from_list(["dataloader.batch_size", 8])
    return ClientData(cfg, train=torch.utils.data.Subset(full, list(range(len(full)))),
                      val=val, test=None)


def test_applying_the_split_deals_training_only_and_requires_the_stated_holdout():
    full = synthetic(n=600)
    holdout = dict(STRATIFIED_HOLDOUT, count=60)
    val = IdDataset([0, 1, 2], [1, 2, 3])
    spec = dict(STRATIFIED_SPLIT)
    with pytest.raises(P.PartitionError, match="label_stratified_hash_v1"):
        P.apply_partition_spec(_client_data(full, val), spec, 1, range(1, 11))
    got = P.apply_partition_spec(_client_data(full, val), spec, 3, range(1, 11), holdout=holdout)
    assert got.applied_partition == spec
    assert list(got["train"].dataset.indices) == list(P.stratified_index_sets(full, 10, 31, holdout)[2])
    assert got["val"].dataset is val, "validation passes through untouched"
    applied = R.apply_holdout(got, holdout)
    assert len(applied["train"].dataset) == len(got["train"].dataset), "nothing more to remove"
    assert applied.applied_holdout["spec"] == holdout
    assert list(applied.readout_holdout.indices) == list(R.stratified_holdout_positions(full, holdout))


def test_each_splitter_takes_its_own_keys():
    assert P.validate_partition_spec({"splitter": "lda", "alpha": 0.3, "seed": 21}) == \
        {"splitter": "lda", "alpha": 0.3, "seed": 21}
    assert P.validate_partition_spec(dict(STRATIFIED_SPLIT)) == STRATIFIED_SPLIT
    with pytest.raises(P.PartitionError, match="unknown"):
        P.validate_partition_spec(dict(STRATIFIED_SPLIT, alpha=0.3))
    with pytest.raises(P.PartitionError, match="missing"):
        P.validate_partition_spec({"splitter": "label_stratified_round_robin_v1"})
