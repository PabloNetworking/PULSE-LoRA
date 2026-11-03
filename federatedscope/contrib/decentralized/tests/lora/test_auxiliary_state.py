"""Auxiliary state is one atomic item under 'full', and zero bytes otherwise."""
import pytest

from federatedscope.contrib.decentralized.lora.auxiliary_state import (
    INTERACTION_GRAPH,
    INTERACTION_PAIRWISE,
    POLICY_FROZEN,
    POLICY_GRAPH_FULL,
    POLICY_LOCAL,
    POLICY_PAIRWISE_FULL,
    AuxiliaryPolicyError,
    apply_policy,
    communicates_auxiliary,
)
from federatedscope.contrib.decentralized.lora.schema import build_schema
from federatedscope.contrib.decentralized.tests.fixtures import build_toy_lora_model


@pytest.mark.parametrize("policy,interaction,expected", [
    (POLICY_PAIRWISE_FULL, INTERACTION_PAIRWISE, True),
    (POLICY_GRAPH_FULL, INTERACTION_GRAPH, True),
    (POLICY_LOCAL, INTERACTION_PAIRWISE, False),
    (POLICY_LOCAL, INTERACTION_GRAPH, False),
    (POLICY_FROZEN, INTERACTION_PAIRWISE, False),
    (POLICY_FROZEN, INTERACTION_GRAPH, False),
])
def test_policy_decides_whether_auxiliary_goes_on_the_wire(policy, interaction, expected):
    assert communicates_auxiliary(policy, interaction) is expected


@pytest.mark.parametrize("policy,interaction", [
    (POLICY_PAIRWISE_FULL, INTERACTION_GRAPH),
    (POLICY_GRAPH_FULL, INTERACTION_PAIRWISE),
])
def test_a_full_policy_must_match_the_interaction_form(policy, interaction):
    with pytest.raises(AuxiliaryPolicyError, match="does not match interaction"):
        communicates_auxiliary(policy, interaction)


def test_unknown_policy_is_rejected():
    with pytest.raises(AuxiliaryPolicyError, match="unknown auxiliary policy"):
        communicates_auxiliary("sometimes", INTERACTION_PAIRWISE)


def test_local_and_frozen_contribute_exactly_zero_bytes():
    """Not 'free' - absent from the frame entirely."""
    model = build_toy_lora_model()
    schema = build_schema(model)
    for policy in (POLICY_LOCAL, POLICY_FROZEN):
        include = communicates_auxiliary(policy, INTERACTION_PAIRWISE)
        keys = schema.payload_keys(include)
        assert not (set(keys) & set(schema.auxiliary_keys))


def test_frozen_actually_freezes_the_head():
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

    frozen = apply_policy(model, schema.auxiliary_keys, POLICY_FROZEN)
    assert frozen == len(schema.auxiliary_keys) > 0
    for name, param in model.named_parameters():
        if name in schema.auxiliary_keys:
            assert not param.requires_grad


def test_non_frozen_policies_freeze_nothing():
    model = build_toy_lora_model()
    schema = build_schema(model)
    for policy in (POLICY_PAIRWISE_FULL, POLICY_GRAPH_FULL, POLICY_LOCAL):
        assert apply_policy(model, schema.auxiliary_keys, policy) == 0
