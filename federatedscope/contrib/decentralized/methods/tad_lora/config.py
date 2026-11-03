"""TAD-LoRA configuration: profiles, alternation settings, and their validation.

Two profiles, labelled separately because they are different settings:

``paper_wt``
    The published TAD-LoRA formulation: graph-round mixing of both factors under
    a fixed doubly-stochastic ``W_t`` over a connected topology (``fixed_wt``),
    reported under ``comparison.profile: paper_native``.
``common_matching_trace``
    The runner compiles ``W_t`` from each frame of the run's contact trace:
    pair blocks at ``comparison.gamma`` (``execution.graph_mixing: pair_block``)
    or Metropolis weights over the frame's available contacts
    (``dynamic_metropolis``). The contacts are those the other methods see on the
    same trace; TAD-LoRA differs in its payload and local-update policy.

Both profiles are graph-round. TAD-LoRA mixes both LoRA factors jointly from one
pre-round snapshot barrier; expressing that as a sequence of pair sessions would
be a different update, not a different encoding of the same one.
"""
from federatedscope.contrib.decentralized.core.interfaces import INTERACTION_GRAPH
from federatedscope.contrib.decentralized.lora.factor_groups import FACTOR_GROUPS

PROFILE_PAPER_WT = "paper_wt"
PROFILE_COMMON_MATCHING_TRACE = "common_matching_trace"
PROFILES = (PROFILE_PAPER_WT, PROFILE_COMMON_MATCHING_TRACE)

#: Both profiles need the graph form; declared per profile, as for Dec-LoRA.
PROFILE_INTERACTION = {
    PROFILE_PAPER_WT: INTERACTION_GRAPH,
    PROFILE_COMMON_MATCHING_TRACE: INTERACTION_GRAPH,
}


def validate(cfg) -> None:
    """Validate the tad_lora subtree plus the shared capabilities it declares."""
    settings = cfg.decentralized.methods.tad_lora

    if settings.profile not in PROFILES:
        raise ValueError(
            f"decentralized.methods.tad_lora.profile is {settings.profile!r}; "
            f"expected one of {list(PROFILES)}")

    if int(settings.rank) <= 0:
        raise ValueError(
            f"decentralized.methods.tad_lora.rank must be positive, got "
            f"{settings.rank}")

    interval = int(settings.alternation_interval)
    if interval < 1:
        raise ValueError(
            f"decentralized.methods.tad_lora.alternation_interval must be at "
            f"least one local episode, got {interval}")
    ceiling = int(settings.max_alternation_interval)
    if ceiling < interval:
        raise ValueError(
            f"decentralized.methods.tad_lora.max_alternation_interval "
            f"({ceiling}) is below alternation_interval ({interval}), so the "
            f"topology-aware ceiling would overrule the configured floor")

    if settings.start_group not in FACTOR_GROUPS:
        raise ValueError(
            f"decentralized.methods.tad_lora.start_group is "
            f"{settings.start_group!r}; expected one of {list(FACTOR_GROUPS)}")

    gamma = float(settings.gamma)
    if not 0.0 < gamma <= 0.5:
        raise ValueError(
            f"decentralized.methods.tad_lora.gamma must satisfy 0 < gamma <= 0.5, "
            f"got {gamma}")
    shared = float(cfg.decentralized.comparison.gamma)
    if abs(gamma - shared) > 1e-12:
        # The same rule as for Dec-LoRA: comparison.gamma is authoritative and a
        # method-level gamma must agree with it.
        raise ValueError(
            f"decentralized.methods.tad_lora.gamma is {gamma} but "
            f"decentralized.comparison.gamma is {shared}; comparison.gamma is "
            f"authoritative for the mixing strength")

    if settings.profile == PROFILE_PAPER_WT:
        if not settings.fixed_wt:
            raise ValueError(
                "tad_lora profile 'paper_wt' requires "
                "decentralized.methods.tad_lora.fixed_wt: a topology name "
                "('ring'/'complete'/'star') or an explicit doubly-stochastic "
                "matrix. Compiling a per-frame matrix from a matching trace is "
                "a different experiment and is not paper-native.")
        from federatedscope.contrib.decentralized.core.mixing_matrix import resolve_fixed_q

        # Resolve now, so a malformed W_t fails at config time and not mid-run.
        resolve_fixed_q(settings.fixed_wt,
                        list(range(1, int(cfg.federate.client_num) + 1)), gamma)

        if cfg.decentralized.comparison.profile != "paper_native":
            raise ValueError(
                f"tad_lora profile 'paper_wt' mixes over a fixed connected "
                f"graph, in which peers have degree > 1, so it is not a matching "
                f"and cannot be reported under comparison.profile="
                f"{cfg.decentralized.comparison.profile!r}. Set "
                f"decentralized.comparison.profile: paper_native.")

    required = PROFILE_INTERACTION[settings.profile]
    configured = cfg.decentralized.execution.interaction
    if configured != required:
        raise ValueError(
            f"tad_lora profile {settings.profile!r} needs interaction "
            f"{required!r} but decentralized.execution.interaction is "
            f"{configured!r}")

    from federatedscope.contrib.decentralized.lora.auxiliary_state import (
        communicates_auxiliary,
    )

    communicates_auxiliary(cfg.decentralized.auxiliary_state.policy, required)
