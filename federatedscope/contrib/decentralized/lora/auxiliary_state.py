"""Policy for auxiliary (non-LoRA) trainable state.

For sequence classification peft 0.4.0 adds ``{"classifier", "score"}`` to
``modules_to_save``, which the GLUE model builder does not set itself, so a GLUE
adapter always carries trainable head parameters. A policy decides whether they
are communicated, and so whether their bytes are charged.

The four policies:

``pairwise_full`` / ``graph_full``
    Auxiliary state is part of the method's single atomic payload, mixed with
    the same coefficient and charged its exact encoded bytes. Which of the two
    names is valid depends on the interaction form.
``local``
    Personalised and never communicated, so it adds no bytes.
``frozen``
    Not trained and not communicated. Tests use it where head drift would
    confound a consensus measurement.
"""
from typing import Iterable


POLICY_PAIRWISE_FULL = "pairwise_full"
POLICY_GRAPH_FULL = "graph_full"
POLICY_LOCAL = "local"
POLICY_FROZEN = "frozen"

POLICIES = (POLICY_PAIRWISE_FULL, POLICY_GRAPH_FULL, POLICY_LOCAL, POLICY_FROZEN)

INTERACTION_PAIRWISE = "pairwise_session"
INTERACTION_GRAPH = "graph_round"

_FULL_FOR_INTERACTION = {
    INTERACTION_PAIRWISE: POLICY_PAIRWISE_FULL,
    INTERACTION_GRAPH: POLICY_GRAPH_FULL,
}


class AuxiliaryPolicyError(ValueError):
    pass


def communicates_auxiliary(policy: str, interaction: str) -> bool:
    """Does this policy put auxiliary state on the wire under this interaction?"""
    if policy not in POLICIES:
        raise AuxiliaryPolicyError(f"unknown auxiliary policy {policy!r}; expected {list(POLICIES)}")
    if interaction not in _FULL_FOR_INTERACTION:
        raise AuxiliaryPolicyError(f"unknown interaction form {interaction!r}")

    if policy in (POLICY_LOCAL, POLICY_FROZEN):
        return False

    expected = _FULL_FOR_INTERACTION[interaction]
    if policy != expected:
        raise AuxiliaryPolicyError(
            f"auxiliary policy {policy!r} does not match interaction {interaction!r}; "
            f"use {expected!r}, or one of {POLICY_LOCAL!r}/{POLICY_FROZEN!r}"
        )
    return True


def apply_policy(model, auxiliary_keys: Iterable[str], policy: str) -> int:
    """Freeze auxiliary parameters under ``frozen``. Returns how many were frozen."""
    if policy != POLICY_FROZEN:
        return 0
    keys = set(auxiliary_keys)
    frozen = 0
    for name, param in model.named_parameters():
        if name in keys:
            param.requires_grad_(False)
            frozen += 1
    return frozen
