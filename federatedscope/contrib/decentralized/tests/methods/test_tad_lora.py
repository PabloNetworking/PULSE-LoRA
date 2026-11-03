"""TAD-LoRA, a second graph method built only from shared capabilities.

Checks that it registers as a plugin, imports no other method package and needs no
branch in core/, runs end to end in both profiles, alternates its trainable factor
group on a topology-aware interval, checkpoints that phase, and mixes through the
shared compiler (or a fixed W_t) at comparison.gamma.

Runs on the toy adapter, so the whole file is CPU-only and needs no download.

Run::

    python -m pytest federatedscope/contrib/decentralized/tests/methods/test_tad_lora.py -q
"""
import pytest
import torch

from federatedscope.contrib.decentralized import registry
from federatedscope.contrib.decentralized.core.interfaces import (
    INTERACTION_GRAPH,
    PeerContext,
)
from federatedscope.contrib.decentralized.core.runner import DecentralizedRunner
from federatedscope.contrib.decentralized.lora.factor_groups import (
    FACTOR_A,
    FACTOR_B,
    group_keys,
    trainable_groups,
)
from federatedscope.contrib.decentralized.methods.tad_lora import TADLoRAFactory
from federatedscope.contrib.decentralized.methods.tad_lora.config import (
    PROFILE_COMMON_MATCHING_TRACE,
    PROFILE_PAPER_WT,
)
from federatedscope.contrib.decentralized.methods.tad_lora.phases import (
    AlternationError,
    AlternationSchedule,
    effective_interval,
)
from federatedscope.contrib.decentralized.tests.fixtures import build_toy_lora_model
from federatedscope.core.configs.config import global_cfg


class ToyTADFactory(TADLoRAFactory):
    """Injects a toy adapter so the tests need no model download."""

    def create_peer(self, context):
        context.extras["model"] = build_toy_lora_model(seed=context.peer_id)
        return super().create_peer(context)


def tad_cfg(profile=PROFILE_COMMON_MATCHING_TRACE, client_num=4, frames=4,
            auxiliary="local", **overrides):
    registry.register_decentralized_method("tad_lora", ToyTADFactory, override=True)
    cfg = global_cfg.clone()
    settings = {
        "federate.method": "tad_lora",
        "federate.client_num": client_num,
        "decentralized.enabled": True,
        "decentralized.seed": 11,
        "decentralized.execution.interaction": INTERACTION_GRAPH,
        "decentralized.execution.total_frames": frames,
        "decentralized.execution.local_steps": 1,
        "decentralized.execution.eval_every_frames": 0,
        "decentralized.auxiliary_state.policy": auxiliary,
        "decentralized.methods.tad_lora.profile": profile,
        "decentralized.methods.tad_lora.rank": 4,   # the toy adapter's rank
        "outdir": "",
    }
    settings.update(overrides)
    if settings["decentralized.methods.tad_lora.profile"] == PROFILE_PAPER_WT:
        settings.setdefault("decentralized.methods.tad_lora.fixed_wt", "ring")
        settings.setdefault("decentralized.comparison.profile", "paper_native")
    flat = []
    for key, value in settings.items():
        flat.extend([key, value])
    # No freeze(): config validation already runs in merge_from_list, and freeze()
    # writes config.yaml into outdir, which is the working directory when outdir is
    # empty. A unit test must not write run artefacts into the repository;
    # test_dec_lora.py does the same.
    cfg.merge_from_list(flat)
    return cfg


def build_runner(cfg):
    return DecentralizedRunner(data={}, config=cfg, factory=ToyTADFactory(cfg))


def toy_peer(cfg, peer_id=1):
    """One peer, built directly rather than through the runner, so a test of the
    peer's local policy does not also run frames, sessions or mixing."""
    context = PeerContext(peer_id=peer_id, config=cfg)
    return ToyTADFactory(cfg).create_peer(context)


# -- registration, independence and end-to-end runs ---------------------------------

def test_tad_registers_as_a_plugin():
    from federatedscope.contrib.decentralized import bootstrap

    bootstrap.load_methods()
    assert registry.is_decentralized_method("tad_lora")


def test_tad_imports_no_other_method_package():
    """No module of the tad_lora package imports another method package (read off the
    syntax tree)."""
    import ast
    import pathlib

    import federatedscope.contrib.decentralized.methods.tad_lora as package

    offenders = []
    for path in sorted(pathlib.Path(package.__file__).parent.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                if "methods." in name and "methods.tad_lora" not in name:
                    offenders.append(f"{path.name}:{node.lineno}: {name}")
    assert not offenders, f"tad_lora imports another method package: {offenders}"


def test_the_core_runner_contains_no_tad_branch():
    """No line of core/ outside comments names TAD: the runtime treats every method alike."""
    import pathlib
    import re

    core = pathlib.Path(__file__).resolve().parents[2].joinpath("core")
    offenders = []
    for path in sorted(core.glob("*.py")):
        text = path.read_text()
        for number, line in enumerate(text.splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            # Word boundary, so words such as "metadata" do not match.
            if re.search(r"\btad\b|tad_lora|TADLoRA", line, re.IGNORECASE):
                offenders.append(f"{path.name}:{number}: {line.strip()[:70]}")
    assert not offenders, f"core/ names TAD: {offenders}"


@pytest.mark.parametrize("profile", [PROFILE_COMMON_MATCHING_TRACE, PROFILE_PAPER_WT])
def test_tad_runs_end_to_end_in_both_profiles(profile):
    runner = build_runner(tad_cfg(profile=profile))
    runner.run()

    assert runner.peers
    for peer in runner.peers.values():
        metrics = peer.algorithm.method_metrics()
        assert metrics["tad_lora_profile"] == profile
        assert metrics["tad_lora_rank"] == 4
        assert metrics["tad_lora_mixing_rounds"] > 0, (
            f"{profile}: no mixing round committed, so nothing was exchanged")


# -- the alternating local update ----------------------------------------------------

def test_the_alternation_actually_switches_the_trainable_group():
    peer = toy_peer(tad_cfg(frames=1))

    # Record the group each episode trained. The model's trainable group is set
    # at the start of an episode, so reading it after the episode returns shows
    # the group that episode trained; reading it before would show the previous
    # episode's group, one step behind the schedule.
    trained, live_during = [], []
    for _ in range(4):
        outcome = peer.local_episode(steps=1)
        trained.append(outcome["factor_group"])
        live = trainable_groups(peer.model, peer.schema)
        live_during.append("".join(g for g in (FACTOR_A, FACTOR_B) if live[g]))

    assert trained == [FACTOR_A, FACTOR_B, FACTOR_A, FACTOR_B], trained
    # and the model was in that group while the episode ran
    assert live_during == trained, (live_during, trained)


def test_exactly_one_group_is_trainable_at_a_time():
    peer = toy_peer(tad_cfg(frames=1))
    for _ in range(3):
        live = trainable_groups(peer.model, peer.schema)
        assert sum(bool(v) for v in live.values()) == 1, live
        peer.local_episode(steps=1)


def test_the_snapshot_carries_both_factors_whatever_the_phase():
    """Mixing is joint. Sending only the active group would make the exchanged
    quantity depend on the sender's private phase."""
    peer = toy_peer(tad_cfg(frames=1))
    from federatedscope.contrib.decentralized.lora.payloads import decode_state

    for _ in range(2):
        state = decode_state(peer.snapshot())
        for group in (FACTOR_A, FACTOR_B):
            for key in group_keys(peer.schema, group):
                assert key in state, f"{group} missing from the snapshot in phase {peer.schedule.group}"
        peer.local_episode(steps=1)


# -- the topology-aware interval -----------------------------------------------------

def test_poor_connectivity_stretches_the_interval_and_never_shortens_it():
    base, window, ceiling = 2, 4, 8
    stretched = [effective_interval(base, topology_aware=True, recent_degree=d,
                                    window=window, max_interval=ceiling)
                 for d in (4, 3, 2, 1, 0)]
    assert stretched == sorted(stretched), stretched
    assert min(stretched) >= base, "connectivity must never shorten below the base"
    assert max(stretched) <= ceiling
    assert effective_interval(base, topology_aware=True, recent_degree=None,
                              window=window, max_interval=ceiling) == base
    assert effective_interval(base, topology_aware=False, recent_degree=0,
                              window=window, max_interval=ceiling) == base


def test_a_ceiling_below_the_base_is_refused():
    with pytest.raises(AlternationError, match="below the base"):
        AlternationSchedule(base_interval=4, max_interval=2)


# -- the checkpoint carries the phase ------------------------------------------------

def test_the_phase_survives_a_checkpoint_round_trip():
    peer = toy_peer(tad_cfg(frames=1))
    peer.local_episode(steps=1)
    peer.local_episode(steps=1)

    saved = peer.state_dict()
    group_at_save = peer.schedule.group

    resumed = toy_peer(tad_cfg(frames=1))
    resumed.load_state_dict(saved)

    assert resumed.schedule.state_dict() == peer.schedule.state_dict()
    # and the model's trainable group matches it, not just the counter
    live = trainable_groups(resumed.model, resumed.schema)
    assert "".join(g for g in (FACTOR_A, FACTOR_B) if live[g]) == group_at_save


def test_a_checkpoint_naming_an_unknown_group_is_refused():
    peer = toy_peer(tad_cfg(frames=1))
    saved = dict(peer.state_dict())
    saved["alternation"] = dict(saved["alternation"], group="Z")
    with pytest.raises(AlternationError, match="does not exist"):
        peer.load_state_dict(saved)


# -- the common profile uses the shared compiler and gamma ---------------------------

def test_the_common_profile_declares_no_fixed_matrix():
    """Returning None is what puts TAD on the shared compiler's path."""
    cfg = tad_cfg(profile=PROFILE_COMMON_MATCHING_TRACE)
    assert TADLoRAFactory(cfg).fixed_mixing_matrix([1, 2, 3, 4]) is None


def test_the_paper_profile_supplies_a_validated_fixed_matrix():
    cfg = tad_cfg(profile=PROFILE_PAPER_WT)
    resolved = TADLoRAFactory(cfg).fixed_mixing_matrix([1, 2, 3, 4])
    assert resolved is not None
    matrix, edges, digest, source = resolved
    assert edges and digest and source


def test_a_method_gamma_disagreeing_with_the_comparison_gamma_is_refused():
    with pytest.raises(ValueError, match="authoritative"):
        tad_cfg(**{"decentralized.methods.tad_lora.gamma": 0.25,
                   "decentralized.comparison.gamma": 0.5})


def test_a_paper_profile_cannot_be_reported_as_a_matching_comparison():
    with pytest.raises(ValueError, match="not a matching"):
        tad_cfg(profile=PROFILE_PAPER_WT,
                **{"decentralized.comparison.profile": "common_matching_trace"})


def test_a_rank_mismatch_between_config_and_adapter_is_refused():
    with pytest.raises(ValueError, match="common-rank"):
        toy_peer(tad_cfg(**{"decentralized.methods.tad_lora.rank": 7}))
