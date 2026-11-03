"""PULSE-LoRA configuration and its validation."""
from federatedscope.contrib.decentralized.core.interfaces import INTERACTION_PAIRWISE
from federatedscope.contrib.decentralized.methods.pulse_lora.state import RELAY_CONTENT
from federatedscope.contrib.decentralized.methods.pulse_lora.pruning import CONTROLLERS
from federatedscope.contrib.decentralized.methods.pulse_lora.selection import SELECTORS

#: PULSE interacts through bilateral endpoint sessions.
INTERACTION = INTERACTION_PAIRWISE


def validate(cfg) -> None:
    """Validate the pulse_lora subtree plus the shared capabilities it declares."""
    settings = cfg.decentralized.methods.pulse_lora

    if int(settings.r_max) <= 0:
        raise ValueError(
            f"decentralized.methods.pulse_lora.r_max must be positive, "
            f"got {settings.r_max}")

    if settings.controller not in CONTROLLERS:
        raise ValueError(
            f"decentralized.methods.pulse_lora.controller is {settings.controller!r}; "
            f"expected one of {list(CONTROLLERS)}")

    # Named prune_gamma, not gamma: gamma is the mixing coefficient, while this is
    # AdaS-C's keep fraction, which sets the size of the low set.
    if not 0.0 < float(settings.prune_gamma) <= 1.0:
        raise ValueError(
            f"decentralized.methods.pulse_lora.prune_gamma must satisfy "
            f"0 < prune_gamma <= 1, got {settings.prune_gamma}")

    if int(settings.rank_min) < 1:
        raise ValueError(
            f"decentralized.methods.pulse_lora.rank_min must be at least 1, "
            f"got {settings.rank_min}")

    if int(settings.initial_rank) > int(settings.r_max):
        raise ValueError(
            f"initial_rank={settings.initial_rank} exceeds r_max={settings.r_max}")

    if int(settings.rank_min) > int(settings.initial_rank):
        raise ValueError(
            f"rank_min={settings.rank_min} exceeds initial_rank={settings.initial_rank}")

    support = [int(c) for c in settings.fixed_support]
    if support:
        if len(set(support)) != len(support):
            raise ValueError(
                f"decentralized.methods.pulse_lora.fixed_support has duplicates: {support}")
        outside = [c for c in support if not 0 <= c < int(settings.r_max)]
        if outside:
            raise ValueError(
                f"fixed_support components {outside} are outside "
                f"P = 0..{int(settings.r_max) - 1}")
        if len(support) != int(settings.initial_rank):
            raise ValueError(
                f"fixed_support has {len(support)} components but initial_rank is "
                f"{settings.initial_rank}; they must agree")
        if len(support) < int(settings.rank_min):
            raise ValueError(
                f"fixed_support has {len(support)} components, below "
                f"rank_min={settings.rank_min}")

    if settings.selector not in SELECTORS:
        raise ValueError(
            f"decentralized.methods.pulse_lora.selector is {settings.selector!r}; "
            f"expected one of {list(SELECTORS)}")

    fraction = float(settings.exploration_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            f"decentralized.methods.pulse_lora.exploration_fraction must satisfy "
            f"0 <= f <= 1, got {settings.exploration_fraction}")

    if int(settings.sketch_dim) <= 0:
        raise ValueError(
            f"decentralized.methods.pulse_lora.sketch_dim must be positive, "
            f"got {settings.sketch_dim}")

    configured = cfg.decentralized.execution.interaction
    if configured != INTERACTION:
        raise ValueError(
            f"pulse_lora needs interaction {INTERACTION!r} but "
            f"decentralized.execution.interaction is {configured!r}")

    from federatedscope.contrib.decentralized.lora.auxiliary_state import (
        communicates_auxiliary,
    )

    communicates_auxiliary(cfg.decentralized.auxiliary_state.policy, INTERACTION)

    relay_content = str(settings.relay_content)
    if relay_content not in RELAY_CONTENT:
        raise ValueError(
            f"decentralized.methods.pulse_lora.relay_content must be one of "
            f"{list(RELAY_CONTENT)}, got {relay_content!r}")
