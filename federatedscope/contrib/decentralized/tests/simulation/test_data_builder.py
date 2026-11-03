"""The GLUE data builder (simulation/glue_data.py) on a stand-in for FederatedScope's
``get_data``.

``training_fixtures.stub_glue_data`` returns what ``get_data`` returns for a GLUE task, in shape:
one dataset per split, every peer a ``Subset`` of each. The builder, the partition and the
holdout code are the release's own.
"""
import dataclasses
import os
import sys

import pytest

from federatedscope.contrib.decentralized.simulation.glue_data import (
    GLUE_BUILDER,
    TASK_KEYS,
    central_glue_data,
    disagreeing_task_keys,
    glue_partition,
)
from federatedscope.contrib.decentralized.simulation.partition import STRATIFIED
from federatedscope.contrib.decentralized.simulation.readout import STRATIFIED_HOLDOUT
from federatedscope.contrib.decentralized.simulation.scenario import build_cfg
from federatedscope.contrib.decentralized.simulation.training import resolve
from federatedscope.contrib.decentralized.tests.simulation.training_fixtures import (
    LOADER,
    STUB_SPLITS,
    install_stub_loader,
    stub_glue_data,
    training_scenario,
)

LDA = {"splitter": "lda", "alpha": 0.3, "seed": 11}
HOLDOUT = {"source": "train", "count": 9, "seed": 5}


@pytest.fixture
def stub_loader(monkeypatch):
    """The stand-in ``get_data`` for one test. It records the configuration of every call and
    hands it back unchanged, so it never disagrees with itself about a task key."""
    calls = []

    def get_data(config=None):
        calls.append({"cfg": config, "outdir": str(config.outdir),
                      "existed": os.path.isdir(str(config.outdir))})
        return stub_glue_data(config), config

    install_stub_loader(monkeypatch, get_data)
    return calls


@pytest.fixture
def loader_that_sets_a_task_key(monkeypatch):
    """A ``get_data`` that writes a task default onto the configuration it is given and returns
    the changed one, as ``_apply_glue_task_defaults`` does."""
    def get_data(config=None):
        changed = config.clone()
        changed.defrost()
        changed.merge_from_list(["model.task", "CausalLM"])
        return stub_glue_data(config), changed

    install_stub_loader(monkeypatch, get_data)


def glue_scenario(**kwargs):
    """The tiny fixture's five-peer scenario, its data built by the GLUE builder."""
    return dataclasses.replace(training_scenario("dec_lora", stochastic=True), data=GLUE_BUILDER,
                               **kwargs)


def schedule(holdout):
    return {"every_frames": 2, "initial": True, "holdout": dict(holdout)}


def indices(partition):
    """Every split's positions in its full dataset, and the holdout's, if any."""
    out = {split: list(partition[split].dataset.indices) for split in STUB_SPLITS
           if partition.get(split) is not None}
    if getattr(partition, "readout_holdout", None) is not None:
        out["holdout"] = list(partition.readout_holdout.indices)
    return out


def test_the_builder_is_named_by_its_dotted_path():
    assert resolve(GLUE_BUILDER) is glue_partition


def test_without_a_partition_or_a_holdout_a_peer_gets_its_iid_slice(stub_loader):
    scenario = glue_scenario()
    cfg = build_cfg(scenario)
    for peer in scenario.peers:
        got = indices(glue_partition(scenario, peer, cfg))
        assert got == {split: list(range((peer - 1) * size, peer * size))
                       for split, size in STUB_SPLITS.items()}
    assert len(stub_loader) == len(scenario.peers)


@pytest.mark.parametrize("partition, holdout", [
    (None, None), (LDA, None), (None, HOLDOUT), (LDA, HOLDOUT),
    ({"splitter": STRATIFIED, "seed": 31}, dict(HOLDOUT, rule=STRATIFIED_HOLDOUT)),
], ids=["iid", "lda", "iid-holdout", "lda-holdout", "stratified"])
def test_the_central_builder_gives_every_peer_what_the_per_peer_builder_gives(
        stub_loader, partition, holdout):
    scenario = glue_scenario(data_partition=partition,
                             readout_schedule=None if holdout is None else schedule(holdout))
    cfg = build_cfg(scenario)
    central = central_glue_data(scenario, cfg)
    assert len(stub_loader) == 1, "every peer's data comes from one get_data call"
    assert sorted(central) == sorted(scenario.peers)
    for peer in scenario.peers:
        alone = glue_partition(scenario, peer, cfg)
        assert indices(central[peer]) == indices(alone), peer
        assert getattr(central[peer], "applied_partition", None) == scenario.data_partition
        assert getattr(central[peer], "applied_holdout", None) == \
            getattr(alone, "applied_holdout", None)


def test_the_central_builder_refuses_peers_whose_splits_index_different_datasets(monkeypatch):
    def get_data(config=None):
        data = stub_glue_data(config)
        data[2] = stub_glue_data(config)[2]          # another call's datasets
        return data, config

    install_stub_loader(monkeypatch, get_data)
    scenario = glue_scenario()
    with pytest.raises(ValueError, match="index different full splits"):
        central_glue_data(scenario, build_cfg(scenario))


def test_the_builders_refuse_a_client_num_the_scenario_does_not_state(monkeypatch):
    """A configuration stating another ``client_num`` would hand this peer a different split.
    It is refused before ``get_data`` is imported: with the loader blocked, the bad
    configuration still raises ``ValueError``, while a good one gets as far as the import."""
    scenario = glue_scenario()
    cfg = build_cfg(scenario)
    cfg.merge_from_list(["federate.client_num", len(scenario.peers) + 1])
    monkeypatch.setitem(sys.modules, LOADER, None)      # makes the import raise
    with pytest.raises(ValueError, match="client_num"):
        glue_partition(scenario, scenario.peers[0], cfg)
    with pytest.raises(ValueError, match="client_num"):
        central_glue_data(scenario, cfg)
    with pytest.raises(ImportError):
        glue_partition(scenario, scenario.peers[0], build_cfg(scenario))


def test_the_builder_refuses_task_keys_get_data_sets_that_the_scenario_did_not_state(
        loader_that_sets_a_task_key):
    """``get_data`` writes task defaults onto the configuration it is given, a copy, while the
    peers train with the scenario's own. A key it sets that the scenario does not state would
    be one the trainer never sees. Both values are named, so the refusal says which side to
    fix."""
    scenario = glue_scenario()
    cfg = build_cfg(scenario)
    with pytest.raises(ValueError, match="model.task") as refused:
        glue_partition(scenario, scenario.peers[0], cfg)
    assert f"({cfg.model.task!r}, 'CausalLM')" in str(refused.value)
    with pytest.raises(ValueError, match="model.task"):
        central_glue_data(scenario, cfg)


def test_the_task_key_comparison_names_every_key_get_data_sets():
    scenario = glue_scenario()
    stated, modified = build_cfg(scenario), build_cfg(scenario)
    assert disagreeing_task_keys(stated, modified) == {}

    changed = {"glue.task_name": "mrpc", "model.out_channels": 3,
               "model.task": "CausalLM", "criterion.type": "MSELoss"}
    flat = []
    for key, value in changed.items():
        flat.extend([key, value])
    modified.merge_from_list(flat)
    disagree = disagreeing_task_keys(stated, modified)
    assert set(disagree) == set(TASK_KEYS) == set(changed)
    for key, value in changed.items():
        section, name = key.split(".")
        assert disagree[key] == (getattr(getattr(stated, section), name), value)


def test_get_data_is_given_an_outdir_of_its_own_and_the_peers_cfg_is_untouched(stub_loader):
    """``get_data`` freezes the configuration it is given, and freezing writes ``config.yaml``
    into ``cfg.outdir``. ``build_cfg`` sets ``outdir`` to "" - the working directory - so the
    builder hands ``get_data`` a copy with a temporary output directory."""
    scenario = glue_scenario()
    cfg = build_cfg(scenario)
    assert cfg.outdir == ""
    glue_partition(scenario, scenario.peers[0], cfg)
    [call] = stub_loader
    assert call["cfg"] is not cfg                       # a copy, never the peer's configuration
    assert call["outdir"] not in ("", cfg.outdir)       # with an output directory of its own,
    assert call["existed"] is True                      # which existed while it ran
    assert not os.path.isdir(call["outdir"])            # and is gone afterwards
    assert cfg.outdir == ""
