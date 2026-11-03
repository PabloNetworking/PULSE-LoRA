"""``common_canonical_v1`` (lora.canonical_init): the canonical initial values and their writer,
on toy adapters."""
import hashlib
import json
import math
import random

import numpy as np
import pytest
import torch

from federatedscope.contrib.decentralized.lora import canonical_init as ci
from federatedscope.contrib.decentralized.lora.adapter_bridge import AdapterBridge
from federatedscope.contrib.decentralized.tests.fixtures import build_toy_lora_model


def construction(seed=7, backbone="toy-backbone"):
    return ci.CommonConstruction(model_init_seed=seed, backbone_digest=backbone,
                                 backbone_path="/toy")


# -- the specification and the seed --------------------------------------------------------------

def test_the_specification_is_exactly_mode_and_seed():
    assert ci.validate_spec({"mode": "common_canonical_v1", "model_init_seed": 31001}) == \
        {"mode": "common_canonical_v1", "model_init_seed": 31001}
    for bad in ({"mode": "other", "model_init_seed": 1}, {"mode": "common_canonical_v1"},
                {"mode": "common_canonical_v1", "model_init_seed": 1, "extra": 0},
                {"mode": "common_canonical_v1", "model_init_seed": True},
                {"mode": "common_canonical_v1", "model_init_seed": -1}, ["mode"]):
        with pytest.raises(ci.ConstructionError):
            ci.validate_spec(bad)


def test_the_component_seed_is_the_specified_encoding():
    """sha256 of UTF-8 canonical JSON (sorted keys, compact separators), first 8 bytes unsigned
    big-endian, modulo 2**63 - recomputed here independently of the module."""
    material = json.dumps({"namespace": "pulse_common_component_v1", "schema_id": "abc",
                           "model_init_seed": 1, "module": "deberta.encoder.layer.0.output.dense",
                           "component_id": 63}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2 ** 63)
    assert ci.component_seed("abc", 1, "deberta.encoder.layer.0.output.dense", 63) == expected
    assert ci.component_seed("abc", 2, "deberta.encoder.layer.0.output.dense", 63) != expected


def test_canonical_module_names_drop_only_the_peft_wrapper():
    assert ci.canonical_module("model.base_model.model.deberta.encoder.layer.3.output.dense") == \
        "deberta.encoder.layer.3.output.dense"
    assert ci.canonical_module("lin_a") == "lin_a"


# -- the schema id ----------------------------------------------------------------------------------

def test_the_schema_id_ignores_rank_and_seed_and_sees_backbone_dimensions_and_version():
    small = AdapterBridge(build_toy_lora_model(rank=2, seed=0)).schema
    large = AdapterBridge(build_toy_lora_model(rank=8, seed=5)).schema
    a = construction(seed=1).bind(small)
    b = construction(seed=99).bind(large)
    assert a.schema_id == b.schema_id, "rank and model-init seed are not schema identity"
    assert construction(backbone="other").bind(small).schema_id != a.schema_id
    document = construction().schema_document(small)
    assert set(document) == {"init_version", "backbone", "modules", "dtype", "effective_scale",
                             "rank_universe"}
    assert document["rank_universe"] == 200 and document["effective_scale"] == 2.0


# -- the values -----------------------------------------------------------------------------------

def test_a_rows_are_uniform_within_the_kaiming_bound_and_b_columns_are_zero():
    init = construction().bind(AdapterBridge(build_toy_lora_model(rank=4)).schema)
    module, d_in, d_out = init.dims[0]
    rows = torch.stack([init.a_row(module, c) for c in range(200)])
    bound = 1.0 / math.sqrt(d_in)
    assert rows.dtype == torch.float32 and rows.shape == (200, d_in)
    assert float(rows.abs().max()) <= bound
    assert float(rows.abs().max()) > 0.5 * bound, "the values spread over the interval"
    assert torch.equal(init.b_column(module, 5), torch.zeros(d_out))


def test_the_same_id_gives_the_same_row_whatever_the_rank_and_a_copy_every_time():
    init_small = construction().bind(AdapterBridge(build_toy_lora_model(rank=2)).schema)
    init_large = construction().bind(AdapterBridge(build_toy_lora_model(rank=8)).schema)
    module = init_small.dims[0][0]
    first = init_small.a_row(module, 150)
    assert torch.equal(first, init_large.a_row(module, 150))
    first.add_(1.0)                                    # a caller mutating its copy...
    assert not torch.equal(first, init_small.a_row(module, 150)), "...never reaches the cache"
    assert not torch.equal(init_small.a_row(module, 150), init_small.a_row(module, 151))
    other_seed = construction(seed=8).bind(AdapterBridge(build_toy_lora_model(rank=2)).schema)
    assert not torch.equal(other_seed.a_row(module, 150), init_small.a_row(module, 150))


def test_a_component_outside_the_universe_is_refused():
    init = construction().bind(AdapterBridge(build_toy_lora_model(rank=2)).schema)
    with pytest.raises(ci.ConstructionError):
        init.a_row(init.dims[0][0], 200)


# -- the writer --------------------------------------------------------------------------------------

class _Peer:
    """A method-shaped object: a model, a bridge and the targets hook."""

    def __init__(self, rank, seed, ids=None):
        self.model = build_toy_lora_model(rank=rank, seed=seed, lora_alpha=2 * rank)
        self.bridge = AdapterBridge(self.model)
        self.ids = tuple(range(rank)) if ids is None else tuple(ids)
        self.trainer = None

    def common_construction_targets(self):
        return self.bridge, self.ids


class _Context:
    def __init__(self):
        self.config = None


class _Auxiliary(torch.nn.Module):
    """A donor-shaped module: a classifier and a pooler, drawn from the global generator."""

    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Linear(4, 3)
        self.pooler = torch.nn.Module()
        self.pooler.dense = torch.nn.Linear(4, 4)


def _donor(seed_box):
    def builder(_cfg):
        seed_box.append(torch.initial_seed())
        random.random()                  # a constructor may draw from every process-global stream;
        np.random.rand()                 # the donor's scope must give each of them back unchanged
        return _Auxiliary()
    return builder


def test_the_writer_gives_every_rank_and_every_peer_the_same_rows(monkeypatch):
    monkeypatch.setattr(ci, "donor_auxiliary", lambda *a, **k: {})
    peers = [_Peer(4, seed=1), _Peer(8, seed=2), _Peer(8, seed=3, ids=(5, 0, 3, 7, 1, 2, 4, 6))]
    with torch.no_grad():                # peft starts B at zero: give it values the writer must clear
        for i, peer in enumerate(peers):
            for spec in peer.bridge.schema.modules:
                peer.bridge._params[spec.b_key].fill_(0.25 * (i + 1))
    for i, peer in enumerate(peers):
        construction().apply(peer, _Context(), who=f"p{i}")
    assert len({construction().bind(p.bridge.schema).schema_id for p in peers}) == 1

    def row(peer, module, component):
        return peer.bridge._params[module.a_key][peer.ids.index(component)]

    for component in range(4):
        for position, module in enumerate(peers[0].bridge.schema.modules):
            rows = [row(peer, peer.bridge.schema.modules[position], component) for peer in peers]
            assert all(torch.equal(rows[0], other) for other in rows[1:]), \
                f"component {component} differs across peers or ranks"
    # a non-prefix slot order still maps every id to its own canonical row
    module = peers[2].bridge.schema.modules[0]
    slot = peers[2].ids.index(5)
    init = construction().bind(peers[2].bridge.schema)
    assert torch.equal(peers[2].bridge._params[module.a_key][slot], init.a_row(module.module, 5))
    assert all(float(peer.bridge._params[spec.b_key].abs().sum()) == 0.0
               for peer in peers for spec in peer.bridge.schema.modules)


def test_the_writer_refuses_a_method_without_the_hook_an_existing_optimizer_and_a_wrong_scale(monkeypatch):
    monkeypatch.setattr(ci, "donor_auxiliary", lambda *a, **k: {})

    class NoHook:
        model = None

    with pytest.raises(ci.ConstructionError, match="does not implement"):
        construction().apply(NoHook(), _Context(), who="x")
    peer = _Peer(4, seed=1)
    peer.trainer = type("T", (), {"optimizer": object()})()
    with pytest.raises(ci.ConstructionError, match="optimizer exists"):
        construction().apply(peer, _Context(), who="x")
    wrong = _Peer(4, seed=1)
    wrong.model = build_toy_lora_model(rank=4, seed=1, lora_alpha=4)       # scale 1.0
    wrong.bridge = AdapterBridge(wrong.model)
    with pytest.raises(ci.ConstructionError, match="scale"):
        construction().apply(wrong, _Context(), who="x")


def test_the_donor_is_built_once_under_the_seed_and_restores_every_rng(monkeypatch):
    ci._DONOR_CACHE.clear()
    seen = []

    class Cfg:
        def __init__(self):
            self.glue = type("G", (), {"adapter": type("A", (), {"args": [{"r": 64, "lora_alpha": 128,
                                                                          "target_modules": ["x"],
                                                                          "lora_dropout": 0.05}]})()})()
            self.model = type("M", (), {"type": "toy@huggingface_llm", "out_channels": 2})()
            self.device = -1

    monkeypatch.setattr(ci, "_donor_config", lambda cfg: cfg)
    torch.manual_seed(123)
    random.seed(5)
    np.random.seed(6)
    before = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state()[1].copy())
    first = ci.donor_auxiliary(Cfg(), 31001, "bb", builder=_donor(seen))
    second = ci.donor_auxiliary(Cfg(), 31001, "bb", builder=_donor(seen))
    assert torch.equal(torch.get_rng_state(), before[0]), "the torch generator was restored"
    assert random.getstate() == before[1] and np.array_equal(np.random.get_state()[1], before[2])
    assert seen == [31001], "built once per key, under the model-init seed"
    assert sorted(first) == ["classifier.bias", "classifier.weight", "pooler.dense.bias",
                             "pooler.dense.weight"]
    torch.manual_seed(31001)
    expected = _Auxiliary()
    assert torch.equal(first["classifier.weight"], expected.classifier.weight), \
        "the donor's values are the seeded constructor's"
    for name in first:
        assert torch.equal(first[name], second[name])
    first["classifier.weight"].add_(1.0)                 # a peer mutating its copy...
    assert torch.equal(ci.donor_auxiliary(Cfg(), 31001, "bb", builder=_donor(seen))
                       ["classifier.weight"], second["classifier.weight"]), "...never the cache"
    ci.donor_auxiliary(Cfg(), 31002, "bb", builder=_donor(seen))
    assert seen == [31001, 31002], "another seed is another key"
    ci._DONOR_CACHE.clear()


def test_the_backbone_identity_is_the_config_and_weight_files_by_sha256(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "pytorch_model.bin").write_bytes(b"weights")
    (tmp_path / "vocab.json").write_text("{}")
    identity = ci.backbone_identity(str(tmp_path))
    assert [f["file"] for f in identity["files"]] == ["config.json", "pytorch_model.bin"]
    assert identity["files"][1]["sha256"] == hashlib.sha256(b"weights").hexdigest()
    (tmp_path / "pytorch_model.bin").write_bytes(b"other weights")
    assert ci.backbone_identity(str(tmp_path))["digest"] != identity["digest"]
    with pytest.raises(ci.ConstructionError):
        ci.backbone_identity(str(tmp_path / "missing"))
