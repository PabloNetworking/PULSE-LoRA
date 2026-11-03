"""The shared schema is stable, ordered canonically, and rejects incompatible peers."""
import pytest
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from federatedscope.contrib.decentralized.lora.schema import SchemaError, build_schema
from federatedscope.contrib.decentralized.tests.fixtures import (
    DEFAULT_ALPHA,
    TOY_TARGET_MODULES,
    ToyBackbone,
    build_toy_lora_model,
)


def test_schema_records_rank_alpha_and_scale_explicitly():
    schema = build_schema(build_toy_lora_model(rank=4, lora_alpha=16))
    assert schema.rank == 4
    assert schema.scaling == pytest.approx(4.0)
    assert schema.lora_alpha == pytest.approx(16.0)


def test_modules_are_ordered_by_canonical_name_not_insertion():
    schema = build_schema(build_toy_lora_model())
    names = [spec.module for spec in schema.modules]
    assert names == sorted(names)
    assert len(names) == len(TOY_TARGET_MODULES)


def test_hash_is_stable_across_rebuilds_and_weights():
    a = build_schema(build_toy_lora_model(seed=1))
    b = build_schema(build_toy_lora_model(seed=2))
    assert a.compatibility_hash == b.compatibility_hash
    a.require_compatible(b)


@pytest.mark.parametrize("kwargs", [
    {"rank": 8},
    {"lora_alpha": 32},
    {"target_modules": ("lin_a", "lin_b")},
])
def test_hash_changes_when_the_contract_changes(kwargs):
    base = build_schema(build_toy_lora_model())
    other = build_schema(build_toy_lora_model(**kwargs))
    assert base.compatibility_hash != other.compatibility_hash
    with pytest.raises(SchemaError, match="incompatible LoRA schemas"):
        base.require_compatible(other)


def test_ab_keys_and_shapes_follow_the_peft_layout():
    schema = build_schema(build_toy_lora_model(rank=4))
    for spec in schema.modules:
        assert spec.a_key.endswith(".lora_A.default.weight")
        assert spec.b_key.endswith(".lora_B.default.weight")
        assert schema.shape_of(spec.a_key) == (4, spec.in_features)
        assert schema.shape_of(spec.b_key) == (spec.out_features, 4)


def test_payload_keys_respect_the_auxiliary_flag():
    schema = build_schema(build_toy_lora_model())
    assert schema.payload_keys(False) == frozenset(schema.lora_keys())
    assert schema.payload_keys(True) >= schema.payload_keys(False)


def test_a_model_without_lora_is_rejected():
    with pytest.raises(SchemaError, match="no trainable LoRA factors"):
        build_schema(ToyBackbone())


def test_asymmetric_ab_rank_is_rejected():
    model = build_toy_lora_model(rank=4)
    module = model.base_model.model.lin_a
    # Corrupt one side so A and B disagree on rank.
    module.lora_B["default"] = nn.Linear(2, 8, bias=False)
    with pytest.raises(SchemaError, match="asymmetric A/B rank"):
        build_schema(model)


def test_mixed_ranks_across_modules_are_rejected():
    model = build_toy_lora_model(rank=4)
    model.base_model.model.lin_a.update_layer("default", 2, DEFAULT_ALPHA, 0.0, True)
    with pytest.raises(SchemaError, match="mixed LoRA ranks"):
        build_schema(model)


def test_auxiliary_keys_are_captured_when_the_model_has_a_head():
    """A modules_to_save head must appear in the schema, not vanish silently."""

    class WithHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin_a = nn.Linear(8, 8, bias=False)
            self.classifier = nn.Linear(8, 2)

        def forward(self, x):
            return self.classifier(self.lin_a(x))

    peft_model = get_peft_model(WithHead(), LoraConfig(
        r=4, lora_alpha=16, lora_dropout=0.0,
        target_modules=["lin_a"], modules_to_save=["classifier"], bias="none"))
    schema = build_schema(peft_model)
    assert schema.auxiliary_keys, "classifier head must be captured as auxiliary state"
    assert all("classifier" in key for key in schema.auxiliary_keys)


def test_peft_original_module_shadow_is_excluded_from_the_payload():
    """peft 0.4.0 leaves original_module trainable, but it never affects forward.

    Communicating it would double head traffic and mix a tensor that changes
    nothing. It is recorded in shadow_keys so the exclusion is visible rather
    than silent; the auxiliary keys are the ``modules_to_save.`` copies.
    """
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model

    class WithHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin_a = nn.Linear(8, 8, bias=False)
            self.classifier = nn.Linear(8, 2)

        def forward(self, x):
            return self.classifier(self.lin_a(x))

    model = get_peft_model(WithHead(), LoraConfig(
        r=4, lora_alpha=16, lora_dropout=0.0, target_modules=["lin_a"],
        modules_to_save=["classifier"], bias="none"))
    schema = build_schema(model)

    assert schema.shadow_keys, "expected peft to expose an original_module shadow"
    assert all("original_module" in key for key in schema.shadow_keys)
    assert all("original_module" not in key for key in schema.auxiliary_keys)
    assert all("modules_to_save" in key for key in schema.auxiliary_keys)
    assert not (set(schema.payload_keys(True)) & set(schema.shadow_keys))
