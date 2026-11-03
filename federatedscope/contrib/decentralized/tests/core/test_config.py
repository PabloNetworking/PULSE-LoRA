"""Config node shape, the enabled/method pair, value validation and registry loading.

Config check functions run inside ``assert_cfg()``, which fires at
``merge_from_file``, ``merge_from_list`` and ``freeze()``, before any runner is
built, so ``assert_decentralized_cfg`` loads the method registry itself.
"""

import pytest

from federatedscope.core.configs.config import global_cfg


def fresh():
    return global_cfg.clone()


# --- node shape ------------------------------------------------------------

def test_node_is_registered_without_editing_any_core_file():
    cfg = fresh()
    assert hasattr(cfg, "decentralized")
    assert hasattr(cfg.decentralized, "enabled")


def test_gate_is_not_named_use():
    """`clean_unused_sub_cfgs` strips a second-level node whose `.use` is False."""
    cfg = fresh()
    assert not hasattr(cfg.decentralized, "use"), (
        "naming the gate `use` would make freeze() delete the whole subtree"
    )


def test_subtree_survives_a_freeze_round_trip():
    cfg = fresh()
    expected = set(cfg.decentralized.keys())
    assert len(expected) > 10

    cfg.freeze(save=False)

    assert set(cfg.decentralized.keys()) == expected
    assert cfg.decentralized.execution.interaction == "pairwise_session"
    assert cfg.decentralized.methods.dec_lora.rank == 8


# --- the enabled/method pair ------------------------------------------------

def test_centralized_default_is_valid():
    cfg = fresh()
    assert cfg.decentralized.enabled is False
    cfg.merge_from_list(["federate.method", "FedAvg"])
    cfg.freeze(save=False)  # must not raise


def test_enabled_with_a_centralized_method_is_rejected():
    cfg = fresh()
    with pytest.raises(ValueError, match="not a decentralized method"):
        cfg.merge_from_list(["federate.method", "FedAvg", "decentralized.enabled", "True"])


def test_decentralized_method_without_the_gate_is_rejected():
    cfg = fresh()
    with pytest.raises(ValueError, match="decentralized.enabled is False"):
        cfg.merge_from_list(["federate.method", "dec_lora"])


def test_reserved_name_with_no_plugin_fails_at_config_time_not_mid_run():
    """A reserved name must resolve to a registered factory before the run starts."""
    from federatedscope.contrib.decentralized import bootstrap, registry

    bootstrap.reset_for_tests()
    had = registry.is_decentralized_method("dec_lora")
    cfg = fresh()
    if had:
        cfg.merge_from_list(["federate.method", "dec_lora", "decentralized.enabled", "True"])
        cfg.freeze(save=False)
    else:
        with pytest.raises(ValueError, match="no plugin registered"):
            cfg.merge_from_list(
                ["federate.method", "dec_lora", "decentralized.enabled", "True"])


def test_validation_loads_the_registry_itself():
    """Config validation imports the method plugins itself."""
    from federatedscope.contrib.decentralized import bootstrap

    bootstrap.reset_for_tests()
    assert bootstrap._loaded is False
    cfg = fresh()
    try:
        cfg.merge_from_list(["federate.method", "dec_lora", "decentralized.enabled", "True"])
    except ValueError:
        pass  # either outcome is fine; what matters is that loading was attempted
    assert bootstrap._loaded is True, "assert_decentralized_cfg did not load the registry"


# --- value validation --------------------------------------------------------

def _enabled_cfg():
    """A cfg whose gate is on, with ``dec_lora`` registered as a bare factory.

    The factory has no ``validate_config``, so only the runtime's own checks run.
    """
    from federatedscope.contrib.decentralized import registry

    registry.register_decentralized_method("dec_lora", lambda cfg: object(), override=True)
    cfg = fresh()
    cfg.merge_from_list(["federate.method", "dec_lora", "decentralized.enabled", "True"])
    return cfg


@pytest.mark.parametrize("override,expected", [
    (["decentralized.execution.interaction", "telepathy"], "execution.interaction"),
    (["decentralized.execution.scheduling", "whenever"], "execution.scheduling"),
    (["decentralized.opportunities.source", "vibes"], "opportunities.source"),
    (["decentralized.comparison.profile", "handwave"], "comparison.profile"),
    (["decentralized.auxiliary_state.policy", "maybe"], "auxiliary_state.policy"),
    (["decentralized.comparison.gamma", "0.9"], "gamma"),
    (["decentralized.comparison.gamma", "0.0"], "gamma"),
    (["decentralized.execution.total_frames", "0"], "total_frames"),
    (["decentralized.execution.local_steps", "0"], "local_steps"),
])
def test_out_of_range_values_are_rejected(override, expected):
    cfg = _enabled_cfg()
    with pytest.raises(ValueError, match=expected):
        cfg.merge_from_list(override)


def test_contradictory_local_step_settings_are_rejected():
    """decentralized.execution.local_steps is authoritative; train must agree."""
    cfg = _enabled_cfg()
    with pytest.raises(ValueError, match="contradictory local-step configuration"):
        cfg.merge_from_list(["decentralized.execution.local_steps", 3,
                             "train.local_update_steps", 7])


def test_matching_local_step_settings_are_accepted():
    cfg = _enabled_cfg()
    cfg.merge_from_list(["decentralized.execution.local_steps", 4,
                         "train.local_update_steps", 4])
    assert cfg.decentralized.execution.local_steps == 4


def test_epoch_based_training_is_rejected():
    """A local episode must be a countable number of optimizer steps."""
    cfg = _enabled_cfg()
    with pytest.raises(ValueError, match="batch_or_epoch"):
        cfg.merge_from_list(["train.batch_or_epoch", "epoch"])


def test_file_source_requires_a_path_and_never_falls_back():
    cfg = _enabled_cfg()
    with pytest.raises(ValueError, match="never falls back"):
        cfg.merge_from_list(["decentralized.opportunities.source", "file"])


def test_unknown_key_in_a_method_subtree_is_rejected_by_the_schema():
    """Layer discipline is enforced by yacs itself: no selector under dec_lora."""
    cfg = _enabled_cfg()
    with pytest.raises(Exception):
        cfg.merge_from_list(["decentralized.methods.dec_lora.selector", "exact_oracle"])


def test_runtime_has_no_method_specific_keys():
    cfg = fresh()
    runtime_nodes = ["execution", "topology", "opportunities", "comparison",
                     "transport", "transaction", "metrics", "checkpoint"]
    for node in runtime_nodes:
        keys = set(getattr(cfg.decentralized, node).keys())
        for forbidden in ("r_max", "relay", "selector", "components"):
            assert forbidden not in keys, f"{node} leaked a method concept: {forbidden}"
