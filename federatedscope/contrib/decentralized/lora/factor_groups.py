"""Factor groups: address the A factors and the B factors of an adapter separately.

A LoRA adapter's trainable factors split into two groups, ``A`` and ``B``.
Methods that update one group at a time treat ``B @ A`` as a bilinear form:
non-convex jointly, convex in each factor with the other held fixed. This module
only addresses the groups (which parameters belong to each, and making exactly
one of them trainable); which group trains, when and for how long is the
method's policy.

Groups are switched through ``requires_grad`` rather than a gradient mask: the
frozen group's gradients are never computed, and the optimizer skips its
parameters without any change to optimizer state.
"""
from typing import Dict, Mapping, Tuple


from federatedscope.contrib.decentralized.lora.schema import LoRASchema

#: The two groups an adapter's trainable factors split into, in a fixed order so
#: an alternating schedule has a deterministic successor.
FACTOR_A = "A"
FACTOR_B = "B"
FACTOR_GROUPS = (FACTOR_A, FACTOR_B)


class FactorGroupError(ValueError):
    """A factor group was named that this adapter does not have."""


def other_group(group: str) -> str:
    """The successor in the alternation. Total over FACTOR_GROUPS."""
    if group not in FACTOR_GROUPS:
        raise FactorGroupError(
            f"unknown factor group {group!r}; expected one of {list(FACTOR_GROUPS)}")
    return FACTOR_B if group == FACTOR_A else FACTOR_A


def group_keys(schema: LoRASchema, group: str) -> Tuple[str, ...]:
    """Every parameter key in one group, in the schema's own module order.

    Taken from the schema rather than by re-scanning the model, because the
    payload and the compatibility hash are defined over the schema.
    """
    if group not in FACTOR_GROUPS:
        raise FactorGroupError(
            f"unknown factor group {group!r}; expected one of {list(FACTOR_GROUPS)}")
    keys = []
    for spec in schema.modules:
        keys.append(spec.a_key if group == FACTOR_A else spec.b_key)
    return tuple(keys)


def group_of(schema: LoRASchema, key: str) -> str:
    """Which group a key belongs to, or FactorGroupError if neither."""
    for spec in schema.modules:
        if key == spec.a_key:
            return FACTOR_A
        if key == spec.b_key:
            return FACTOR_B
    raise FactorGroupError(f"{key!r} is not a LoRA factor key in this schema")


def split_by_group(schema: LoRASchema,
                   state: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    """Partition a payload's LoRA entries into the two groups.

    Auxiliary and any other non-factor keys are returned under neither group.
    """
    split = {FACTOR_A: {}, FACTOR_B: {}}
    a_keys = set(group_keys(schema, FACTOR_A))
    b_keys = set(group_keys(schema, FACTOR_B))
    for key, value in state.items():
        if key in a_keys:
            split[FACTOR_A][key] = value
        elif key in b_keys:
            split[FACTOR_B][key] = value
    return split


def set_trainable_group(model, schema: LoRASchema, group: str) -> Dict[str, int]:
    """Make exactly ``group`` trainable among the LoRA factors.

    Returns, per group, how many parameters changed state, so a caller can
    check that the switch happened. Parameters outside the two factor groups,
    such as an auxiliary head, are not touched: whether they train is the
    auxiliary-state policy's decision.
    """
    if group not in FACTOR_GROUPS:
        raise FactorGroupError(
            f"unknown factor group {group!r}; expected one of {list(FACTOR_GROUPS)}")

    wanted = {name: (name in set(group_keys(schema, group)))
              for name in schema.lora_keys()}
    changed = {FACTOR_A: 0, FACTOR_B: 0}
    seen = set()
    for name, param in model.named_parameters():
        if name not in wanted:
            continue
        seen.add(name)
        if bool(param.requires_grad) != wanted[name]:
            param.requires_grad_(wanted[name])
            changed[group_of(schema, name)] += 1

    missing = set(wanted) - seen
    if missing:
        raise FactorGroupError(
            f"the schema names {len(missing)} factor parameter(s) the model does "
            f"not expose, so a factor group cannot be set exactly: "
            f"{sorted(missing)[:3]}")
    return changed


def trainable_groups(model, schema: LoRASchema) -> Dict[str, bool]:
    """Which groups are currently trainable, read back from the model rather
    than from a method's own bookkeeping."""
    state = {}
    for group in FACTOR_GROUPS:
        keys = set(group_keys(schema, group))
        flags = [bool(param.requires_grad)
                 for name, param in model.named_parameters() if name in keys]
        state[group] = bool(flags) and all(flags)
    return state
