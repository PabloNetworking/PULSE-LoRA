"""Pin the peft key/shape/scale contract the decentralized substrate is built on.

The shared LoRA code assumes the key format and tensor layout asserted here,
and peft 0.4.0 makes the scaling depend on the rank (scaling = alpha / r).
"""
import torch

from federatedscope.contrib.decentralized.tests.fixtures import (
    DEFAULT_ALPHA,
    DEFAULT_IN_FEATURES,
    DEFAULT_RANK,
    TOY_TARGET_MODULES,
    build_toy_lora_model,
    lora_scalings,
    trainable_lora_state,
)


def test_key_format_and_shapes_match_the_documented_contract():
    state = trainable_lora_state(build_toy_lora_model())

    expected = set()
    for module in TOY_TARGET_MODULES:
        for side in ("lora_A", "lora_B"):
            expected.add(f"base_model.model.{module}.{side}.default.weight")
    assert set(state) == expected, "peft key format drifted from the pinned contract"

    for name, param in state.items():
        if "lora_A" in name:
            assert tuple(param.shape) == (DEFAULT_RANK, DEFAULT_IN_FEATURES)
        else:
            assert tuple(param.shape) == (DEFAULT_IN_FEATURES, DEFAULT_RANK)


def test_flat_component_spans_every_injected_layer():
    """Component p is row p of every A and column p of every B, in all layers."""
    state = trainable_lora_state(build_toy_lora_model())
    for p in range(DEFAULT_RANK):
        slices = []
        for module in TOY_TARGET_MODULES:
            a = state[f"base_model.model.{module}.lora_A.default.weight"]
            b = state[f"base_model.model.{module}.lora_B.default.weight"]
            slices.append(a[p, :])
            slices.append(b[:, p])
        # one slice per side per layer, and every slice is a real vector
        assert len(slices) == 2 * len(TOY_TARGET_MODULES)
        assert all(s.numel() == DEFAULT_IN_FEATURES for s in slices)


def test_same_seed_is_bitwise_identical_and_different_seed_is_not():
    left = trainable_lora_state(build_toy_lora_model(seed=1234))
    right = trainable_lora_state(build_toy_lora_model(seed=1234))
    other = trainable_lora_state(build_toy_lora_model(seed=4321))

    assert set(left) == set(right) == set(other)
    for key in left:
        assert torch.equal(left[key], right[key]), f"{key} not reproducible at equal seed"

    assert any(
        not torch.equal(left[key], other[key]) for key in left
    ), "different seeds produced identical adapters"


def test_scaling_is_alpha_over_rank():
    scalings = lora_scalings(build_toy_lora_model(rank=DEFAULT_RANK, lora_alpha=DEFAULT_ALPHA))
    assert set(scalings) == set(TOY_TARGET_MODULES)
    for value in scalings.values():
        assert value == DEFAULT_ALPHA / DEFAULT_RANK


def test_a_rank_change_rescales_the_adapter():
    """peft 0.4.0 recomputes scaling = alpha / r, so a rank change rescales.

    Dec-LoRA holds one common rank and is unaffected. A method whose ranks
    differ or change must pin a rank-independent scale
    (``AdapterBridge.enforce_fixed_scale``).
    """
    model = build_toy_lora_model(rank=8, lora_alpha=DEFAULT_ALPHA)
    assert set(lora_scalings(model).values()) == {DEFAULT_ALPHA / 8}

    module = model.base_model.model.lin_a
    module.update_layer("default", 2, DEFAULT_ALPHA, 0.0, True)

    assert dict(module.scaling)["default"] == DEFAULT_ALPHA / 2
    assert tuple(module.lora_A["default"].weight.shape) == (2, DEFAULT_IN_FEATURES)
    assert tuple(module.lora_B["default"].weight.shape) == (DEFAULT_IN_FEATURES, 2)
