"""Dec-LoRA configuration: profiles and their validation.

Three profiles, labelled separately because they are different settings:

``paper_fixed_q``
    The published Dec-LoRA formulation: snapshot mixing over a fixed connected
    graph with a symmetric doubly stochastic Q (``fixed_q``), reported under
    ``comparison.profile: paper_native``.
``common_trace_pairwise``
    Symmetric mean-preserving full-state gossip in pair sessions over the contact
    trace. Fixed-Q theory does not cover these time-varying contacts.
``dynamic_graph_adapted``
    Dec-LoRA's mixing step on a time-varying graph: one graph round per frame over
    the frame's available contacts, with the runtime's per-frame Metropolis
    W_t (``execution.graph_mixing: dynamic_metropolis``). An adaptation of a
    fixed-graph method (run label ``dec-dynamic-<graph>-adapted``), never the
    published fixed-Q method and never ``paper_native``.
"""
from federatedscope.contrib.decentralized.core.interfaces import (
    INTERACTION_GRAPH,
    INTERACTION_PAIRWISE,
)

PROFILE_PAPER_FIXED_Q = "paper_fixed_q"
PROFILE_COMMON_TRACE_PAIRWISE = "common_trace_pairwise"
PROFILE_DYNAMIC_GRAPH_ADAPTED = "dynamic_graph_adapted"
PROFILES = (PROFILE_PAPER_FIXED_Q, PROFILE_COMMON_TRACE_PAIRWISE, PROFILE_DYNAMIC_GRAPH_ADAPTED)

#: Each profile declares the interaction form it needs.
PROFILE_INTERACTION = {
    PROFILE_PAPER_FIXED_Q: INTERACTION_GRAPH,
    PROFILE_COMMON_TRACE_PAIRWISE: INTERACTION_PAIRWISE,
    PROFILE_DYNAMIC_GRAPH_ADAPTED: INTERACTION_GRAPH,
}


def validate(cfg) -> None:
    """Validate the dec_lora subtree plus the shared capabilities it declares."""
    settings = cfg.decentralized.methods.dec_lora

    if settings.profile not in PROFILES:
        raise ValueError(
            f"decentralized.methods.dec_lora.profile is {settings.profile!r}; "
            f"expected one of {list(PROFILES)}"
        )

    if int(settings.rank) <= 0:
        raise ValueError(
            f"decentralized.methods.dec_lora.rank must be positive, got {settings.rank}"
        )

    gamma = float(settings.gamma)
    if not 0.0 < gamma <= 0.5:
        raise ValueError(
            f"decentralized.methods.dec_lora.gamma must satisfy 0 < gamma <= 0.5, "
            f"got {gamma}"
        )

    if settings.profile == PROFILE_PAPER_FIXED_Q:
        if not settings.fixed_q:
            raise ValueError(
                "dec_lora profile 'paper_fixed_q' requires "
                "decentralized.methods.dec_lora.fixed_q: a topology name "
                "('ring'/'complete'/'star') or an explicit doubly-stochastic matrix. "
                "Compiling a per-frame matrix from a matching trace is a different "
                "experiment and is not paper-native."
            )
        from federatedscope.contrib.decentralized.core.mixing_matrix import resolve_fixed_q

        # Resolve and validate now, so a malformed Q fails at config time.
        resolve_fixed_q(settings.fixed_q, list(range(1, int(cfg.federate.client_num) + 1)),
                        float(settings.gamma))

        if cfg.decentralized.comparison.profile != "paper_native":
            raise ValueError(
                f"dec_lora profile 'paper_fixed_q' offers a fixed connected graph, in "
                f"which peers have degree > 1, so it is not a matching and cannot be "
                f"reported under comparison.profile="
                f"{cfg.decentralized.comparison.profile!r}. Set "
                f"decentralized.comparison.profile: paper_native."
            )

    if settings.profile == PROFILE_DYNAMIC_GRAPH_ADAPTED:
        if settings.fixed_q:
            raise ValueError(
                "dec_lora profile 'dynamic_graph_adapted' derives W_t per frame from the "
                "trace; a fixed_q is the paper_fixed_q experiment, not this one")
        mixing = str(getattr(cfg.decentralized.execution, "graph_mixing", ""))
        if mixing != "dynamic_metropolis":
            raise ValueError(
                f"dec_lora profile 'dynamic_graph_adapted' needs "
                f"decentralized.execution.graph_mixing: dynamic_metropolis, got {mixing!r}")
        if cfg.decentralized.comparison.profile == "paper_native":
            raise ValueError(
                "dec_lora profile 'dynamic_graph_adapted' is a dynamic-network adaptation of "
                "Dec-LoRA, not the paper's fixed-Q method; it cannot be reported under "
                "comparison.profile=paper_native")

    required = PROFILE_INTERACTION[settings.profile]
    configured = cfg.decentralized.execution.interaction
    if configured != required:
        raise ValueError(
            f"dec_lora profile {settings.profile!r} needs interaction {required!r} "
            f"but decentralized.execution.interaction is {configured!r}"
        )

    # Auxiliary policy is a shared capability, but the method declares which
    # variants make sense for the interaction form it is running under.
    from federatedscope.contrib.decentralized.lora.auxiliary_state import (
        communicates_auxiliary,
    )

    communicates_auxiliary(cfg.decentralized.auxiliary_state.policy, required)
