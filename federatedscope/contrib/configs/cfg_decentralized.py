"""Configuration of the decentralized runtime (``cfg.decentralized``).

``core/configs/config.py`` imports every module of ``contrib/configs/``, so the
node is registered without changes to a core file.

Two constraints on the schema:

* The gate is ``enabled``, not ``use``: ``CN.clean_unused_sub_cfgs`` deletes
  every key of a second-level node whose ``use`` is ``False``, which would strip
  this subtree at ``freeze()``.
* Method options live under ``decentralized.methods.<name>``. yacs rejects
  unknown keys at merge time, so the schema itself stops an option from being
  set in the wrong place: the runtime has no ``r_max``, and ``dec_lora`` has no
  selector.
"""
import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

logger = logging.getLogger(__name__)

INTERACTION_FORMS = ("pairwise_session", "graph_round")
SCHEDULING_MODES = ("synchronized", "event_driven")
SKIPPED_EDGE_POLICIES = ("self_weight", "identity_round")
#: A copy of core/interactions.GRAPH_MIXING_MODES (a test keeps the two equal), so
#: that importing this module imports nothing from the runtime.
GRAPH_MIXING_MODES = ("pair_block", "dynamic_metropolis")
OPPORTUNITY_SOURCES = ("generated", "file")
#: ``common_matching_trace``: every frame of the trace must be a matching.
#: ``common_study``: any realized contact trace (a G(n, p) sequence, for example), in
#: which a peer may meet several partners in one frame; a pairwise method takes those
#: sessions in event order, as under ``paper_native``. ``paper_native``: a method in its
#: published form (over a fixed mixing matrix, for example). The profile is recorded in
#: the manifest, and runs compare only with runs of the same profile.
COMPARISON_PROFILES = ("common_matching_trace", "common_study", "paper_native")
AUXILIARY_POLICIES = ("pairwise_full", "graph_full", "local", "frozen")


def extend_decentralized_cfg(cfg):
    cfg.decentralized = CN()

    # The gate; not named `use` (see the module docstring).
    cfg.decentralized.enabled = False
    cfg.decentralized.plugin_version = ""
    cfg.decentralized.seed = 0

    # --- execution -------------------------------------------------------
    cfg.decentralized.execution = CN()
    cfg.decentralized.execution.interaction = "pairwise_session"
    cfg.decentralized.execution.scheduling = "synchronized"
    cfg.decentralized.execution.local_steps = 1
    cfg.decentralized.execution.total_frames = 10
    cfg.decentralized.execution.eval_every_frames = 5
    # Evaluate every peer on its validation split after the last frame.
    cfg.decentralized.execution.final_evaluation = True
    cfg.decentralized.execution.skipped_edge_policy = "self_weight"
    # With a fixed mixing matrix: reject a frame that does not offer every edge of
    # the fixed graph, instead of handling the missing edges by skipped_edge_policy.
    cfg.decentralized.execution.require_complete_graph_frames = False
    # How a graph round obtains W_t when the method declares no fixed matrix.
    #   pair_block         - the matching-derived pair-block matrix at comparison.gamma;
    #                        refuses a peer on more than one edge (degree one only).
    #   dynamic_metropolis - a W_t renormalised every frame over the frame's available
    #                        edges at any degree, by Metropolis-Hastings weights
    #                        (core/interactions.METROPOLIS_RULE): the time-varying
    #                        mixing of the TAD-LoRA and D-PSGD papers. Needs interaction
    #                        'graph_round'; refused together with a fixed matrix.
    #                        Recorded in the manifest only when it is on.
    cfg.decentralized.execution.graph_mixing = "pair_block"
    # Local-compute duration in microseconds, used only by event_driven scheduling.
    # It comes from this explicit model, not from the simulation clock: an offset
    # derived from absolute time is not a duration, and would change the run when
    # every timestamp of the trace is shifted by a constant.
    cfg.decentralized.execution.compute_latency_us = 0
    cfg.decentralized.execution.compute_latency_jitter_us = 0

    # --- topology --------------------------------------------------------
    cfg.decentralized.topology = CN()
    cfg.decentralized.topology.history_window = 16

    # --- offered communication ------------------------------------------
    cfg.decentralized.opportunities = CN()
    cfg.decentralized.opportunities.source = "generated"
    cfg.decentralized.opportunities.path = ""
    cfg.decentralized.opportunities.schema_version = 1
    cfg.decentralized.opportunities.duration_ms = 1000
    cfg.decentralized.opportunities.goodput_kbps = 1000
    cfg.decentralized.opportunities.goodput_kbps_reverse = -1  # -1 => symmetric
    cfg.decentralized.opportunities.frame_interval_ms = 1000
    cfg.decentralized.opportunities.transport = "direct"

    # --- cross-method comparison ----------------------------------------
    cfg.decentralized.comparison = CN()
    cfg.decentralized.comparison.profile = "common_matching_trace"
    cfg.decentralized.comparison.expected_trace_digest = ""
    cfg.decentralized.comparison.gamma = 0.5
    # Declares a run without any dataset (the toy test fixtures). Without it, a run
    # with an empty data map records every peer's partition in the manifest as
    # unproven: an empty data map does not show that two runs used the same data.
    cfg.decentralized.comparison.dataset_free = False

    # --- transport / wire -------------------------------------------------
    cfg.decentralized.transport = CN()
    cfg.decentralized.transport.frame_version = 1
    cfg.decentralized.transport.checksum = True
    cfg.decentralized.transport.acknowledge = True
    # The byte denomination of session admission. "logical" prices a session in the
    # ledger's own bytes (the encoded frames). Any other name resolves a registered
    # wire-cost model: "socket_v1" (simulation/wire_quote.py; scenarios default to it)
    # plans k_e in the serialized bytes of the reference peer-to-peer protocol.
    # Recorded in the manifest; two runs under different models are not comparable.
    cfg.decentralized.transport.wire_cost = "logical"
    # The envelope: the largest single message body of a pair session or graph-round
    # edge; socket_v1 prices a larger message so that no credit admits it. 64 MiB by
    # default and at most 512 MiB; a non-default value needs a wire-cost model other
    # than "logical". Recorded in the manifest (and in every wire context digest) only
    # when it is not the default.
    cfg.decentralized.transport.envelope_bytes = 64 * 1024 * 1024

    # --- transactions -----------------------------------------------------
    cfg.decentralized.transaction = CN()
    cfg.decentralized.transaction.failure_injection_stage = ""
    cfg.decentralized.transaction.failure_injection_every = 0

    # --- auxiliary (non-LoRA) trainable state: the classification head ---
    cfg.decentralized.auxiliary_state = CN()
    cfg.decentralized.auxiliary_state.policy = "pairwise_full"

    # --- observability -----------------------------------------------------
    cfg.decentralized.metrics = CN()
    cfg.decentralized.metrics.event_log = True
    cfg.decentralized.metrics.peer_log = True

    cfg.decentralized.checkpoint = CN()
    cfg.decentralized.checkpoint.enabled = False
    cfg.decentralized.checkpoint.every_frames = 0
    cfg.decentralized.checkpoint.dir = ""

    # --- method subtrees ---------------------------------------------------
    cfg.decentralized.methods = CN()
    cfg.decentralized.methods.dec_lora = CN()
    cfg.decentralized.methods.dec_lora.profile = "common_trace_pairwise"
    cfg.decentralized.methods.dec_lora.rank = 8
    cfg.decentralized.methods.dec_lora.gamma = 0.5
    # The fixed Q of the paper_fixed_q profile: a topology name ('ring'/'complete'/
    # 'star'), a JSON matrix literal marked 'json:' ("json:[[0.5, 0.5], [0.5, 0.5]]"),
    # or the path of a JSON file holding one (a bare list of rows, or
    # {"matrix": [...], "peers": [...]}). The option is a str: the loader refuses a
    # list here and literal-evaluates a bare "[[...]]" into one, so an explicit matrix
    # is written behind the 'json:' mark (core/mixing_matrix.parse_explicit_q). It is
    # validated at config time: square, symmetric, doubly stochastic, connected,
    # peers matched.
    cfg.decentralized.methods.dec_lora.fixed_q = ""

    cfg.decentralized.methods.tad_lora = CN()
    cfg.decentralized.methods.tad_lora.profile = "common_matching_trace"
    cfg.decentralized.methods.tad_lora.rank = 8
    cfg.decentralized.methods.tad_lora.gamma = 0.5
    #: Local episodes one factor group trains for before the alternation
    #: switches. With topology_aware, poor recent connectivity stretches it, up to
    #: max_alternation_interval; it is never shortened.
    cfg.decentralized.methods.tad_lora.alternation_interval = 1
    cfg.decentralized.methods.tad_lora.max_alternation_interval = 8
    cfg.decentralized.methods.tad_lora.topology_aware = True
    cfg.decentralized.methods.tad_lora.start_group = "A"
    #: The fixed W_t of the paper_wt profile: a topology name, a 'json:'-marked JSON
    #: matrix literal, or the path of a JSON matrix file, as for dec_lora.fixed_q above.
    cfg.decentralized.methods.tad_lora.fixed_wt = ""

    cfg.decentralized.methods.pulse_lora = CN()
    cfg.decentralized.methods.pulse_lora.r_max = 8
    cfg.decentralized.methods.pulse_lora.initial_rank = 8
    cfg.decentralized.methods.pulse_lora.rank_min = 1
    cfg.decentralized.methods.pulse_lora.controller = "fixed"
    cfg.decentralized.methods.pulse_lora.prune_gamma = 0.99
    cfg.decentralized.methods.pulse_lora.component_seed = 1234
    # 0.0 means derive the fixed scale as lora_alpha / r_max, which is the scale
    # the adapter would apply at full rank. It does not follow the current rank.
    cfg.decentralized.methods.pulse_lora.scale = 0.0
    # An explicit initial support (component ids), usually set per peer through a
    # client_<id> overlay. Empty means the first initial_rank components. Need not
    # be a prefix.
    cfg.decentralized.methods.pulse_lora.fixed_support = []

    # --- session policy and byte-budgeted selection ----------------------------
    # Which components a pair session exchanges. Run-level: the two endpoints
    # must rank candidates the same way or they would select different sets,
    # and there is no negotiation round to discover the disagreement.
    cfg.decentralized.methods.pulse_lora.selector = "local_update"
    # Fraction of the affordable slots reserved for the oldest eligible
    # components. 0.0 is the no-exploration ablation; any positive value
    # reserves at least one slot when any slot exists.
    cfg.decentralized.methods.pulse_lora.exploration_fraction = 0.5
    # Width of the disagreement sketch of the sketch_disagreement selector. The
    # sketch travels in the session manifest and is charged as control bytes.
    cfg.decentralized.methods.pulse_lora.sketch_dim = 16
    # "carried": the relay plane keeps the values it received. "scrubbed" (ablation):
    # a component received into the relay plane is stored as its deterministic
    # initial value instead. Bytes, candidates, sessions, supports and budgets are
    # identical either way; only the values relayed components carry differ.
    cfg.decentralized.methods.pulse_lora.relay_content = "carried"

    cfg.register_cfg_check_fun(assert_decentralized_cfg)


def _one_of(value, allowed, key):
    if value not in allowed:
        raise ValueError(
            f"cfg.{key} is {value!r}; expected one of {list(allowed)}"
        )


def assert_decentralized_cfg(cfg):
    """Validate the subtree, and keep ``federate.method`` and the gate consistent.

    Check functions run inside ``assert_cfg()``, at every ``merge_from_*`` and at
    ``freeze()``, possibly before any method plugin has been imported, so this
    function loads the method registry itself. The imports are inside the
    function because this module is imported while ``core/configs/config.py`` is
    still executing, and a module-level import would pull the LoRA code, torch
    and peft into config import time.
    """
    # `is_reserved_name` is a static name check that imports no plugin, so a
    # configuration that is not decentralized imports nothing here.
    from federatedscope.contrib.decentralized.registry import is_reserved_name

    dec = cfg.decentralized
    enabled = bool(dec.enabled)
    method = getattr(getattr(cfg, "federate", None), "method", "")
    reserved = is_reserved_name(method)

    if enabled and not reserved:
        raise ValueError(
            f"cfg.decentralized.enabled is True but federate.method is {method!r}, "
            f"which is not a decentralized method. Set a decentralized method, or "
            f"disable the decentralized runtime."
        )
    if reserved and not enabled:
        raise ValueError(
            f"federate.method is {method!r}, a decentralized method, but "
            f"cfg.decentralized.enabled is False. Set decentralized.enabled: true."
        )
    if not enabled:
        return

    # From here on the run is decentralized, so the plugins are imported; resolving
    # the factory now makes a missing or broken plugin fail at config time rather
    # than mid-run.
    from federatedscope.contrib.decentralized import bootstrap
    from federatedscope.contrib.decentralized.registry import (
        get_factory,
        is_decentralized_method,
        known_methods,
    )

    bootstrap.load_methods()
    if not is_decentralized_method(method):
        raise ValueError(
            f"federate.method {method!r} is reserved by the decentralized package "
            f"but no plugin registered it; registered: {list(known_methods())}"
        )

    _one_of(dec.execution.interaction, INTERACTION_FORMS, "decentralized.execution.interaction")
    _one_of(dec.execution.scheduling, SCHEDULING_MODES, "decentralized.execution.scheduling")
    _one_of(dec.opportunities.source, OPPORTUNITY_SOURCES, "decentralized.opportunities.source")
    _one_of(dec.comparison.profile, COMPARISON_PROFILES, "decentralized.comparison.profile")
    _one_of(dec.auxiliary_state.policy, AUXILIARY_POLICIES, "decentralized.auxiliary_state.policy")
    _one_of(dec.execution.skipped_edge_policy, SKIPPED_EDGE_POLICIES,
            "decentralized.execution.skipped_edge_policy")
    _one_of(dec.execution.graph_mixing, GRAPH_MIXING_MODES,
            "decentralized.execution.graph_mixing")
    from federatedscope.contrib.decentralized.core.wire_cost import validate_envelope_bytes
    validate_envelope_bytes(dec.transport.envelope_bytes, dec.transport.wire_cost)
    if (dec.execution.graph_mixing == "dynamic_metropolis"
            and dec.execution.interaction != "graph_round"):
        raise ValueError(
            f"decentralized.execution.graph_mixing 'dynamic_metropolis' derives a per-frame "
            f"W_t for a graph_round, but decentralized.execution.interaction is "
            f"{dec.execution.interaction!r}; a pairwise session has no W_t to renormalise")

    if dec.opportunities.source == "file" and not dec.opportunities.path:
        raise ValueError(
            "decentralized.opportunities.source is 'file' but no path was given; "
            "the loader never falls back to generated bandwidth"
        )

    # decentralized.comparison.gamma is the one mixing coefficient: the graph compiler
    # and the manifest read it, and assert_comparable compares it. A method may restate
    # it but may not disagree, or the mixing strength a run reports would differ from
    # the one it applied.
    gamma = float(dec.comparison.gamma)
    if not 0.0 < gamma <= 0.5:
        raise ValueError(
            f"decentralized.comparison.gamma must satisfy 0 < gamma <= 0.5, got {gamma}"
        )
    # Only the active method's gamma is checked: the other methods' subtrees keep
    # their defaults, which this run never applies.
    active = getattr(getattr(cfg, "federate", None), "method", None)
    if active in dec.methods:
        method_gamma = getattr(dec.methods[active], "gamma", None)
        if method_gamma is not None and float(method_gamma) != gamma:
            raise ValueError(
                f"decentralized.methods.{active}.gamma={float(method_gamma)} "
                f"disagrees with the authoritative decentralized.comparison.gamma="
                f"{gamma}. Set them equal, or change only comparison.gamma."
            )

    # decentralized.execution.local_steps sets the local work, and the trainer reads
    # train.local_update_steps, so the two must agree; otherwise the local-work
    # schedule recorded in the manifest would describe something other than what ran.
    configured_steps = int(dec.execution.local_steps)
    trainer_steps = int(getattr(cfg.train, "local_update_steps", configured_steps))
    if trainer_steps != configured_steps:
        raise ValueError(
            f"contradictory local-step configuration: "
            f"decentralized.execution.local_steps={configured_steps} but "
            f"train.local_update_steps={trainer_steps}. The decentralized setting is "
            f"authoritative; set train.local_update_steps to the same value."
        )
    batch_or_epoch = getattr(cfg.train, "batch_or_epoch", "batch")
    if batch_or_epoch != "batch":
        raise ValueError(
            f"decentralized runs need train.batch_or_epoch='batch' so a local episode "
            f"is a countable number of optimizer steps, got {batch_or_epoch!r}"
        )

    if int(dec.execution.compute_latency_us) < 0:
        raise ValueError("decentralized.execution.compute_latency_us must be >= 0")
    if int(dec.execution.compute_latency_jitter_us) < 0:
        raise ValueError("decentralized.execution.compute_latency_jitter_us must be >= 0")

    if int(dec.execution.total_frames) <= 0:
        raise ValueError("decentralized.execution.total_frames must be positive")
    if int(dec.execution.local_steps) <= 0:
        raise ValueError("decentralized.execution.local_steps must be positive")

    # Each method plugin validates its own subtree and the shared settings it uses.
    factory_cls = get_factory(method)
    validate = getattr(factory_cls, "validate_config", None)
    if callable(validate):
        validate(cfg)


register_config("decentralized", extend_decentralized_cfg)
