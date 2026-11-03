"""A model, data and configuration that train through the GLUE decentralized trainer in seconds.

The smallest model the trainer accepts unchanged: a dict batch in, ``.loss`` and ``.logits`` out,
real peft LoRA factors and a trainable head. Everything around it is the release code: the
``GLUEDecentralizedTrainer`` with its persistent AdamW and scheduler, ``ClientData`` and its data
loaders, the three methods, scenarios and the driver.

  - LoRA targets three 8x8 linears at rank 4 with ``lora_alpha`` 8, so the live scale is 2.0, as
    the common construction requires.
  - The head is peft ``modules_to_save``. All three methods can exchange it
    (``pairwise_full`` / ``graph_full``), which is the default; ``payload_head=False`` keeps it
    ``local``.
  - The scheduler is ``LinearLR`` with its defaults, whose learning rate moves over the first
    five steps.
  - TAD-LoRA alternates every episode (``topology_aware`` off), so both factor groups train
    within a five-frame run.

Partitions come from private torch generators seeded from (scenario seed, peer), never from the
global generator.

:func:`stub_glue_data` stands in for FederatedScope's ``get_data`` on a GLUE task, so the GLUE
builder (``simulation/glue_data.py``) runs without a tokenizer, a dataset download or a backbone.
"""
import hashlib
import sys
import types

import torch
import torch.nn as nn

from federatedscope.contrib.decentralized.core.interfaces import (
    INTERACTION_GRAPH as GRAPH,
    INTERACTION_PAIRWISE as PAIRWISE,
)
from federatedscope.contrib.decentralized.simulation.readout import declare_holdout
from federatedscope.contrib.decentralized.simulation.scenario import Scenario
from federatedscope.contrib.decentralized.simulation.training import PER_PEER_EPISODE

FEATURES = 8
LABELS = 2
RANK = 4
ALPHA = 8
LOCAL_STEPS = 2
TARGETS = ("lin_a", "lin_b", "lin_c")

MODULE = "federatedscope.contrib.decentralized.tests.simulation.training_fixtures"
STOCHASTIC = f"{MODULE}.stochastic_classifier"
DETERMINISTIC = f"{MODULE}.deterministic_classifier"
PARTITION = f"{MODULE}.tiny_partition"
CONSTRUCTION = {"mode": "common_canonical_v1", "model_init_seed": 1}

#: Per-method settings of the tiny runs.
METHODS = {
    "dec_lora": {"decentralized.execution.interaction": PAIRWISE,
                 "decentralized.auxiliary_state.policy": "local",
                 "decentralized.methods.dec_lora.rank": RANK},
    "pulse_lora": {"decentralized.execution.interaction": PAIRWISE,
                   "decentralized.auxiliary_state.policy": "local",
                   "decentralized.methods.pulse_lora.r_max": RANK,
                   "decentralized.methods.pulse_lora.initial_rank": RANK},
    "tad_lora": {"decentralized.execution.interaction": GRAPH,
                 "decentralized.auxiliary_state.policy": "local",
                 "decentralized.methods.tad_lora.rank": RANK},
}

#: Five frames over five peers: a frame without contacts, an unavailable contact, and pairs
#: that recur.
VARIED = {
    "trace_schema_version": 1, "time_unit": "ms", "duration_unit": "ms",
    "goodput_unit": "bps", "peers": [1, 2, 3, 4, 5],
    "frames": [
        {"index": 0, "time": 0, "edges": [
            {"endpoints": [3, 4], "duration": 1000, "goodput": 10_000_000},
            {"endpoints": [2, 1], "duration": 1000, "goodput": 10_000_000}]},
        {"index": 1, "time": 1000, "edges": [
            {"endpoints": [1, 3], "duration": 1000, "goodput": 10_000_000},
            {"endpoints": [2, 5], "duration": 1000, "goodput": 10_000_000,
             "available": False}]},
        {"index": 2, "time": 2000, "edges": []},
        {"index": 3, "time": 3000, "edges": [
            {"endpoints": [2, 5], "duration": 1000, "goodput": 10_000_000},
            {"endpoints": [1, 4], "duration": 1000, "goodput": 10_000_000}]},
        {"index": 4, "time": 4000, "edges": [
            {"endpoints": [1, 5], "duration": 1000, "goodput": 10_000_000},
            {"endpoints": [2, 3], "duration": 1000, "goodput": 10_000_000}]},
    ],
}


class TinyClassifier(nn.Module):
    """HF-shaped: ``forward(input_ids, attention_mask, token_type_ids, labels)`` returns
    a ``SequenceClassifierOutput``. ``input_ids`` are float features, which the
    trainer moves to the device without looking at them."""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.lin_a = nn.Linear(FEATURES, FEATURES, bias=False)
        self.lin_b = nn.Linear(FEATURES, FEATURES, bias=False)
        self.lin_c = nn.Linear(FEATURES, FEATURES, bias=False)
        self.head = nn.Linear(FEATURES, LABELS)
        # Identity, not Dropout(0): a deterministic model should not even call a
        # dropout kernel.
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        from transformers.modeling_outputs import SequenceClassifierOutput

        hidden = self.drop(torch.relu(self.lin_a(input_ids.float())))
        hidden = self.drop(torch.relu(self.lin_b(hidden)))
        logits = self.head(self.lin_c(hidden))
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)


def _classifier(rank: int, seed: int, lora_dropout: float, dropout: float):
    """A peft-wrapped TinyClassifier, built from ``seed`` alone (it seeds torch itself, as
    ``build_toy_lora_model`` does)."""
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    return get_peft_model(TinyClassifier(dropout), LoraConfig(
        r=rank, lora_alpha=ALPHA, lora_dropout=lora_dropout, target_modules=list(TARGETS),
        modules_to_save=["head"], bias="none"))


def stochastic_classifier(rank=RANK, seed=0):
    """LoRA dropout 0.05 and backbone dropout 0.1: training draws on the global RNG."""
    return _classifier(rank, seed, lora_dropout=0.05, dropout=0.1)


def deterministic_classifier(rank=RANK, seed=0):
    """No dropout anywhere: with an unshuffled loader, nothing training does depends
    on the global RNG."""
    return _classifier(rank, seed, lora_dropout=0.0, dropout=0.0)


def _derived(seed, label) -> int:
    blob = f"tests/tiny/{seed}/{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") >> 1


class TinyDataset(torch.utils.data.Dataset):
    """GLUE-shaped items: ``input_ids`` (float features), ``attention_mask``, ``labels``."""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"input_ids": self.features[index],
                "attention_mask": torch.ones(FEATURES, dtype=torch.long),
                "labels": self.labels[index]}


def tiny_partition(scenario, peer_id, cfg):
    """Peer ``peer_id``'s data: fresh objects, a pure function of (scenario seed, peer).

    Non-IID by peer: one labelling rule for the whole run, a covariate shift along
    it that depends on ``peer_id % 3`` (so label balance varies across peers), and
    sizes by the same rule (40, 44 or 48 training examples; 16 validation). Every
    peer's examples are its own draws."""
    from federatedscope.core.data.base_data import ClientData

    concept = torch.randn(FEATURES, generator=torch.Generator().manual_seed(
        _derived(scenario.seed, "concept")))
    local = torch.Generator().manual_seed(_derived(scenario.seed, f"peer{int(peer_id)}"))
    shift = (int(peer_id) % 3 - 1) * 0.75 * concept / concept.norm()

    def split(size):
        features = torch.randn(size, FEATURES, generator=local) + shift
        return TinyDataset(features, (features @ concept > 0).long())

    partition = ClientData(cfg, train=split(40 + 4 * (int(peer_id) % 3)), val=split(16),
                           test=None)
    schedule = getattr(scenario, "readout_schedule", None)
    if schedule is not None:
        partition = declare_holdout(partition, schedule["holdout"],
                                    tiny_holdout(scenario, schedule["holdout"]))
    return partition


def tiny_holdout(scenario, spec):
    """The readout holdout of the tiny fixture: ``count`` examples of the run's one labelling
    rule, without any peer's covariate shift, from a generator of their own - identical for
    every peer and method, and never an example any peer trains on (each peer draws its own
    examples from its own generator). A peer's training and validation draws are exactly
    those of a run without a schedule."""
    concept = torch.randn(FEATURES, generator=torch.Generator().manual_seed(
        _derived(scenario.seed, "concept")))
    held = torch.Generator().manual_seed(_derived(scenario.seed, f"holdout/{int(spec['seed'])}"))
    features = torch.randn(int(spec["count"]), FEATURES, generator=held)
    return TinyDataset(features, (features @ concept > 0).long())


def training_overrides(method, *, shuffle, payload_head=True):
    """cfg overrides that make GLUEDecentralizedTrainer train ``method`` on CPU."""
    settings = dict(METHODS[method])
    interaction = settings["decentralized.execution.interaction"]
    settings.update({
        "trainer.type": "gluedecentralizedtrainer",
        "train.local_update_steps": LOCAL_STEPS,
        "decentralized.execution.local_steps": LOCAL_STEPS,
        "train.batch_or_epoch": "batch",
        "train.optimizer.type": "AdamW",
        "train.optimizer.lr": 0.01,
        "train.scheduler.type": "LinearLR",
        "dataloader.batch_size": 8,
        "dataloader.shuffle": bool(shuffle),
        "criterion.type": "CrossEntropyLoss",
        "eval.count_flops": False,
        "model.out_channels": LABELS,
        "decentralized.auxiliary_state.policy": (
            ("graph_full" if interaction == GRAPH else "pairwise_full") if payload_head
            else "local"),
    })
    if method == "pulse_lora":
        # adas_c at gamma 0.99 keeps floor(0.99 r) = r - 1 and puts one component in
        # the low set; with a non-zero B a prune happens within the first episodes.
        settings.update({"decentralized.methods.pulse_lora.controller": "adas_c",
                         "decentralized.methods.pulse_lora.prune_gamma": 0.99,
                         "decentralized.methods.pulse_lora.rank_min": 1})
    if method == "tad_lora":
        settings.update({"decentralized.methods.tad_lora.topology_aware": False,
                         "decentralized.methods.tad_lora.alternation_interval": 1})
    return settings


def training_scenario(method, *, stochastic, trace=None, training_rng="default",
                      payload_head=True, readout_schedule=None):
    """A training run over ``trace`` (default :data:`VARIED`).

    ``stochastic`` selects dropout and a shuffled loader, and by default the per-peer episode
    rule; otherwise no dropout, no shuffle and no rule. ``training_rng`` overrides the rule."""
    trace = VARIED if trace is None else trace
    if training_rng == "default":
        training_rng = PER_PEER_EPISODE if stochastic else None
    return Scenario(method=method, peers=tuple(trace["peers"]), trace=trace,
                    adapter=STOCHASTIC if stochastic else DETERMINISTIC, adapter_rank=RANK,
                    cfg_overrides=training_overrides(method, shuffle=stochastic,
                                                     payload_head=payload_head),
                    data=PARTITION, training_rng=training_rng,
                    readout_schedule=readout_schedule, construction=dict(CONSTRUCTION))


# -- a stand-in for get_data on a GLUE task ---------------------------------------------------

#: The module the GLUE builder imports ``get_data`` from when it is called.
LOADER = "federatedscope.core.auxiliaries.data_builder"
#: A stand-in partition's splits, and how many of a split's examples each peer gets.
STUB_SPLITS = {"train": 16, "val": 8, "test": 8}


class GlueShapedDataset(TinyDataset):
    """TinyDataset with the raw ``examples`` a GLUE dataset keeps (original ``idx`` and
    ``label``). The ids are not the positions, as in a real split."""

    def __init__(self, features, labels, first_id=0):
        super().__init__(features, labels)
        self.examples = [{"idx": first_id + 3 * position, "label": int(label)}
                         for position, label in enumerate(labels.tolist())]


def stub_glue_data(config):
    """What ``get_data`` returns for a GLUE task, in shape: one dataset per split, and every
    peer a ``Subset`` of each - its contiguous IID slice - in a ``ClientData`` built with
    ``config``. A fixed generator: every call returns the same examples."""
    from torch.utils.data import Subset

    from federatedscope.core.data.base_data import ClientData

    peers = int(config.federate.client_num)
    generator = torch.Generator().manual_seed(4242)
    concept = torch.randn(FEATURES, generator=generator)
    datasets = {}
    for offset, (split, size) in enumerate(STUB_SPLITS.items()):
        features = torch.randn(size * peers, FEATURES, generator=generator)
        datasets[split] = GlueShapedDataset(features, (features @ concept > 0).long(),
                                            first_id=1000 * offset + 7)
    return {peer: ClientData(config, **{
        split: Subset(datasets[split], list(range((peer - 1) * size, peer * size)))
        for split, size in STUB_SPLITS.items()})
        for peer in range(1, peers + 1)}


def install_stub_loader(monkeypatch, get_data):
    """Make ``get_data`` the one the GLUE builder imports, for one test. Reading anything else
    from the loader module while it is installed is an error."""
    def missing(name):
        raise AttributeError(f"the stand-in loader has no {name!r}")

    module = types.ModuleType(LOADER)
    module.get_data = get_data
    module.__getattr__ = missing
    monkeypatch.setitem(sys.modules, LOADER, module)
