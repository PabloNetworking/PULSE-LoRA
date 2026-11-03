"""Two methods are only directly comparable when the shared axes agree.

The manifest records those axes so that they can be checked: the offered trace,
the local-work schedule, gamma, the auxiliary-state policy and the wire version.
Payload and local-update policy are free to differ - those are the things under
comparison.
"""
import pytest

from federatedscope.contrib.decentralized.core.manifest import (
    assert_comparable,
    build_manifest,
)
from federatedscope.contrib.decentralized.core.runner import DecentralizedRunner
from federatedscope.contrib.decentralized.tests.core.toy_method import ToyFactory
from federatedscope.contrib.decentralized.tests.core.test_runner import toy_cfg
from federatedscope.contrib.decentralized.tests.methods.test_dec_lora import (
    ToyLoRAFactory,
    dec_cfg,
)
from federatedscope.contrib.decentralized.methods.dec_lora.config import (
    PROFILE_PAPER_FIXED_Q,
)

# Settings shared by both sides of every comparison below. The seed and the
# auxiliary-state policy are comparability axes too, so they are pinned here.
COMMON = {"client_num": 4, "frames": 6, "decentralized.seed": 42}
AUXILIARY = "pairwise_full"


def dec_lora_manifest(**overrides):
    overrides.setdefault("auxiliary", AUXILIARY)
    cfg = dec_cfg(**dict(COMMON, **overrides))
    return build_manifest(DecentralizedRunner(data={}, config=cfg,
                                              factory=ToyLoRAFactory(cfg)))


def toy_manifest(**overrides):
    overrides.setdefault("decentralized.auxiliary_state.policy", AUXILIARY)
    cfg = toy_cfg(**dict(COMMON, **overrides))
    return build_manifest(DecentralizedRunner(data={}, config=cfg,
                                              factory=ToyFactory(cfg)))


def test_two_methods_on_the_same_trace_share_its_digest():
    left, right = dec_lora_manifest(), toy_manifest()
    assert left["method"] != right["method"], "these must be different methods"
    assert left["trace_id"] == right["trace_id"]
    assert left["peer_ids"] == right["peer_ids"]


def test_matching_manifests_are_declared_comparable():
    assert_comparable(dec_lora_manifest(), toy_manifest())


@pytest.mark.parametrize("override,axis", [
    ({"decentralized.comparison.gamma": 0.25}, "gamma"),
    # both must move together: config validation requires them to be equal
    ({"decentralized.execution.local_steps": 3, "train.local_update_steps": 3},
     "local_steps_per_episode"),
    ({"decentralized.seed": 77}, "trace_id"),
    ({"client_num": 6}, "peer_ids"),
    ({"frames": 8}, "total_frames"),
])
def test_a_differing_axis_blocks_comparison(override, axis):
    with pytest.raises(ValueError, match=axis):
        assert_comparable(dec_lora_manifest(), toy_manifest(**override))


def test_paper_native_is_not_comparable_with_a_common_trace_run():
    """A paper-native result is not comparable with a common-trace result.

    These are two separate settings: the *method* profile (paper_fixed_q)
    selects the interaction form, while the *comparison* profile (paper_native)
    declares how the result may be reported. Both are set here.
    """
    common = dec_lora_manifest()
    native = dec_lora_manifest(
        profile=PROFILE_PAPER_FIXED_Q, auxiliary="graph_full",
        **{"decentralized.comparison.profile": "paper_native"})
    assert native["comparison_profile"] == "paper_native"
    with pytest.raises(ValueError, match="comparison_profile"):
        assert_comparable(common, native)


def test_manifest_records_the_axes_a_reader_needs():
    manifest = dec_lora_manifest()
    for field in ("trace_id", "gamma", "auxiliary_state_policy", "peer_ids",
                  "local_steps_per_episode", "wire_version", "interaction_form",
                  "comparison_profile", "seed", "environment"):
        assert field in manifest, f"manifest is missing {field}"


def test_payload_policy_may_differ_without_blocking_comparison():
    """Different methods exchange different payloads and stay comparable."""
    left, right = dec_lora_manifest(), toy_manifest()
    assert left["interaction_form"] == right["interaction_form"]
    assert_comparable(left, right)  # despite entirely different payloads


# --- the scheduling axis -----------------------------------------------------------

EVENT_DRIVEN = {
    "decentralized.execution.scheduling": "event_driven",
    "decentralized.execution.compute_latency_us": 250_000,
    "decentralized.execution.compute_latency_jitter_us": 100_000,
}


def test_synchronized_and_event_driven_runs_are_not_comparable():
    """The scheduling mode is part of the local-work schedule."""
    with pytest.raises(ValueError, match="scheduling"):
        assert_comparable(toy_manifest(), toy_manifest(**EVENT_DRIVEN))


def test_event_driven_runs_with_different_base_latency_are_not_comparable():
    faster = dict(EVENT_DRIVEN, **{"decentralized.execution.compute_latency_us": 500_000})
    with pytest.raises(ValueError, match="compute_latency_us"):
        assert_comparable(toy_manifest(**EVENT_DRIVEN), toy_manifest(**faster))


def test_event_driven_runs_with_different_jitter_are_not_comparable():
    jumpier = dict(EVENT_DRIVEN,
                   **{"decentralized.execution.compute_latency_jitter_us": 400_000})
    with pytest.raises(ValueError, match="compute_latency_jitter_us"):
        assert_comparable(toy_manifest(**EVENT_DRIVEN), toy_manifest(**jumpier))


def test_identical_event_driven_settings_stay_comparable():
    assert_comparable(toy_manifest(**EVENT_DRIVEN), toy_manifest(**EVENT_DRIVEN))
    assert_comparable(dec_lora_manifest(**EVENT_DRIVEN), toy_manifest(**EVENT_DRIVEN))


def test_latency_ignored_under_synchronized_scheduling_is_canonicalised_to_zero():
    """Compute latency only applies to event-driven scheduling, so a configured
    but unused latency is recorded as zero and causes no mismatch."""
    configured_but_unused = toy_manifest(**{
        "decentralized.execution.compute_latency_us": 999_999,
        "decentralized.execution.compute_latency_jitter_us": 7,
    })
    assert configured_but_unused["scheduling"] == "synchronized"
    assert configured_but_unused["compute_latency_us"] == 0
    assert configured_but_unused["compute_latency_jitter_us"] == 0
    assert_comparable(toy_manifest(), configured_but_unused)


def test_event_driven_records_the_values_it_actually_applies():
    manifest = toy_manifest(**EVENT_DRIVEN)
    assert manifest["scheduling"] == "event_driven"
    assert manifest["compute_latency_us"] == 250_000
    assert manifest["compute_latency_jitter_us"] == 100_000


# -- Dec-LoRA, PULSE-LoRA and TAD-LoRA on one offered communication -------------------
#
# auxiliary_state.policy is 'local' because it is a checked axis and must hold
# one value across the three. 'pairwise_full' and 'graph_full' are
# interaction-matched forms of one policy but are different values, and TAD-LoRA
# needs the graph form while Dec-LoRA and PULSE-LoRA need the pairwise one - so no
# 'full' value can be shared.

from federatedscope.contrib.decentralized.tests.methods.test_tad_lora import (  # noqa: E402
    ToyTADFactory,
    tad_cfg,
)
from federatedscope.contrib.decentralized.tests.methods.test_pulse_peer import (  # noqa: E402
    ToyPulseFactory,
    pulse_cfg,
)

THREE_WAY = dict(COMMON)
THREE_WAY_AUXILIARY = "local"
THREE_METHODS = {"dec_lora", "pulse_lora", "tad_lora"}


def _dec_lora_three_way(**overrides):
    cfg = dec_cfg(**dict(THREE_WAY, auxiliary=THREE_WAY_AUXILIARY, **overrides))
    return build_manifest(DecentralizedRunner(data={}, config=cfg,
                                              factory=ToyLoRAFactory(cfg)))


def _tad_lora_three_way(**overrides):
    cfg = tad_cfg(**dict(THREE_WAY, auxiliary=THREE_WAY_AUXILIARY, **overrides))
    return build_manifest(DecentralizedRunner(data={}, config=cfg,
                                              factory=ToyTADFactory(cfg)))


def _pulse_lora_three_way(**overrides):
    settings = dict(THREE_WAY)
    frames = settings.pop("frames", None)
    client_num = settings.pop("client_num", None)
    settings["decentralized.auxiliary_state.policy"] = THREE_WAY_AUXILIARY
    # ToyPulseFactory builds a rank-4 adapter; start at that rank in a 4-component space.
    settings.setdefault("decentralized.methods.pulse_lora.initial_rank", 4)
    settings.setdefault("decentralized.methods.pulse_lora.r_max", 4)
    settings.update(overrides)
    cfg = pulse_cfg(client_num=client_num, frames=frames, **settings)
    return build_manifest(DecentralizedRunner(data={}, config=cfg,
                                              factory=ToyPulseFactory(cfg)))


def _three():
    return [_dec_lora_three_way(), _pulse_lora_three_way(), _tad_lora_three_way()]


def test_the_three_are_exactly_the_three_research_methods():
    """The three manifests are Dec-LoRA, PULSE-LoRA and TAD-LoRA, not a test double."""
    names = {m["method"] for m in _three()}
    assert names == THREE_METHODS, names


def test_three_methods_share_one_offered_communication():
    """The offered trace and its structural digest are identical for the three.

    TAD-LoRA compiles each frame into a graph round while Dec-LoRA and PULSE-LoRA
    compile it into pair sessions - different interaction forms of the same
    offered contacts - so what was offered must still be identical.
    """
    manifests = _three()
    for axis in ("trace_id", "offered_communication_digest", "peer_ids",
                 "total_frames", "gamma"):
        values = {repr(m[axis]) for m in manifests}
        assert len(values) == 1, f"{axis} differs across the three methods: {values}"


def test_three_methods_share_one_local_work_schedule():
    manifests = _three()
    for axis in ("local_steps_per_episode", "scheduling", "compute_latency_us",
                 "seed", "value_precision", "auxiliary_state_policy"):
        values = {repr(m[axis]) for m in manifests}
        assert len(values) == 1, f"{axis} differs across the three methods: {values}"


def test_all_three_pairs_are_declared_comparable():
    dec, pulse, tad = _three()
    assert_comparable(dec, pulse)
    assert_comparable(pulse, tad)
    assert_comparable(dec, tad)


@pytest.mark.parametrize("override,axis", [
    ({"decentralized.comparison.gamma": 0.25,
      "decentralized.methods.tad_lora.gamma": 0.25}, "gamma"),
    ({"decentralized.seed": 77}, "trace_id"),
    ({"frames": 8}, "total_frames"),
])
def test_a_differing_axis_blocks_the_three_way_comparison(override, axis):
    """A shared axis changed on one side blocks the comparison."""
    with pytest.raises(ValueError, match=axis):
        assert_comparable(_dec_lora_three_way(), _tad_lora_three_way(**override))
