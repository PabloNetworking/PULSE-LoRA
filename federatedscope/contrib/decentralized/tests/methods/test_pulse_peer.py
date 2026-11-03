"""PULSE-LoRA peer: configuration, local episodes and pruning through the generic
peer shell, checkpoints, pair sessions, and method metrics.

Also defines the toy factory and config helpers that other PULSE tests import.
"""
import pytest
import torch

from federatedscope.contrib.decentralized import registry
from federatedscope.contrib.decentralized.core.accounting import ByteLedger
from federatedscope.contrib.decentralized.core.interactions import compile_pair_sessions
from federatedscope.contrib.decentralized.core.interfaces import PeerContext
from federatedscope.contrib.decentralized.core.pair_session import execute_pair_session
from federatedscope.contrib.decentralized.core.peer import Peer
from federatedscope.contrib.decentralized.core.runner import DecentralizedRunner
from federatedscope.contrib.decentralized.core.traces import load_trace
from federatedscope.contrib.decentralized.core.transaction import TransactionEngine
from federatedscope.contrib.decentralized.methods.pulse_lora import PulseLoRAFactory
from federatedscope.contrib.decentralized.methods.pulse_lora.units import unit_cost_bytes
from federatedscope.contrib.decentralized.tests.fixtures import (
    CONSTRUCTION_KEY,
    build_toy_lora_model,
    toy_construction,
    toy_extras,
)
from federatedscope.core.configs.config import global_cfg


class ToyPulseFactory(PulseLoRAFactory):
    """Injects a toy adapter and the toy common construction so these stay CPU-only."""

    def create_peer(self, context):
        context.extras["model"] = build_toy_lora_model(rank=4, seed=context.peer_id)
        context.extras.setdefault(CONSTRUCTION_KEY, toy_construction())
        return super().create_peer(context)


def pulse_cfg(client_num=2, frames=2, controller="fixed", **overrides):
    registry.register_decentralized_method("pulse_lora", ToyPulseFactory, override=True)
    cfg = global_cfg.clone()
    settings = {
        "federate.method": "pulse_lora",
        "federate.client_num": client_num,
        "decentralized.enabled": True,
        "decentralized.seed": 7,
        "decentralized.execution.interaction": "pairwise_session",
        "decentralized.execution.total_frames": frames,
        "decentralized.execution.local_steps": 1,
        "decentralized.execution.eval_every_frames": 0,
        "decentralized.auxiliary_state.policy": "local",
        "decentralized.methods.pulse_lora.r_max": 8,
        "decentralized.methods.pulse_lora.initial_rank": 4,
        "decentralized.methods.pulse_lora.rank_min": 2,
        "decentralized.methods.pulse_lora.controller": controller,
        "outdir": "",
    }
    settings.update(overrides)
    flat = []
    for key, value in settings.items():
        flat.extend([key, value])
    cfg.merge_from_list(flat)
    return cfg


def make_peer(peer_id=1, cfg=None):
    cfg = cfg or pulse_cfg()
    context = PeerContext(peer_id=peer_id, config=cfg,
                          extras=toy_extras(build_toy_lora_model(rank=4, seed=peer_id)))
    return ToyPulseFactory(cfg).create_peer(context)


# --- configuration ---------------------------------------------------------------

def test_pulse_registers_and_declares_a_pairwise_interaction():
    cfg = pulse_cfg()
    factory = registry.get_factory("pulse_lora")(cfg)
    assert factory.name == "pulse_lora"
    assert factory.interaction_form == "pairwise_session"


@pytest.mark.parametrize("override,expected", [
    ({"decentralized.methods.pulse_lora.r_max": 0}, "r_max must be positive"),
    ({"decentralized.methods.pulse_lora.controller": "vibes"}, "expected one of"),
    ({"decentralized.methods.pulse_lora.prune_gamma": 1.5}, "prune_gamma"),
    ({"decentralized.methods.pulse_lora.rank_min": 0}, "rank_min must be at least 1"),
    ({"decentralized.methods.pulse_lora.initial_rank": 99}, "exceeds r_max"),
])
def test_invalid_pulse_settings_are_rejected(override, expected):
    with pytest.raises(ValueError, match=expected):
        pulse_cfg(**override)


def test_the_pruning_fraction_is_not_called_gamma():
    """gamma is the mixing coefficient in this package: the config check compares
    methods.<method>.gamma with comparison.gamma, so a pruning fraction stored under
    that name would be checked against the mixing coefficient."""
    cfg = pulse_cfg()
    assert hasattr(cfg.decentralized.methods.pulse_lora, "prune_gamma")
    assert not hasattr(cfg.decentralized.methods.pulse_lora, "gamma")


# --- local episodes, pruning and checkpoints -----------------------------------------

def test_a_pulse_peer_runs_local_episodes_through_the_generic_peer_shell():
    algorithm = make_peer()
    shell = Peer(1, algorithm, algorithm.context)

    for _ in range(3):
        metrics = shell.local_episode(1)

    assert shell.local_episodes == 3
    assert metrics["active_rank"] == len(algorithm.state.support)
    algorithm.state.check_invariants()


def test_the_adas_c_controller_prunes_a_shrinking_low_set_component():
    algorithm = make_peer(cfg=pulse_cfg(controller="adas_c",
                                        **{"decentralized.methods.pulse_lora.prune_gamma": 0.5}))
    # give the components a nonzero B so scores are meaningful, then shrink one
    # low-set component between before_episode and select_for_pruning
    generator = torch.Generator().manual_seed(0)
    for component in algorithm.state.support:
        block = algorithm.state.gather(component)
        algorithm.state.scatter(component, {
            k: torch.rand(v.shape, generator=generator) + 0.5 for k, v in block.items()})

    shell = Peer(1, algorithm, algorithm.context)
    algorithm.controller.before_episode(algorithm.state)
    doomed = algorithm.controller._low_set[0]
    block = algorithm.state.gather(doomed)
    algorithm.state.scatter(doomed, {k: v * 0.01 for k, v in block.items()})

    pruned = algorithm.controller.select_for_pruning(algorithm.state)
    assert doomed in pruned

    algorithm.state.prune(pruned)
    assert doomed not in algorithm.state.support
    assert doomed in algorithm.state.materialised, "pruning must not drop relay eligibility"
    algorithm.state.check_invariants()


def test_the_fixed_controller_keeps_the_support_across_a_whole_run():
    cfg = pulse_cfg(frames=4, controller="fixed")
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyPulseFactory(cfg))
    runner.run()
    for peer in runner.peers.values():
        assert peer.algorithm.state.support == (0, 1, 2, 3)
        peer.algorithm.state.check_invariants()


def test_pulse_runs_end_to_end_through_the_shared_runner():
    cfg = pulse_cfg(frames=3)
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyPulseFactory(cfg))
    results = runner.run()

    assert results["method"] == "pulse_lora"
    assert set(results["per_peer"]) == {1, 2}, "every peer must be evaluated"
    assert "adapter_norm" in results["distribution"]


def test_pulse_state_round_trips_through_a_peer_checkpoint():
    algorithm = make_peer()
    algorithm.state.prune([1])
    payload = algorithm.state_dict()

    restored = make_peer()
    restored.state.prune([1])
    restored.load_state_dict(payload)

    assert restored.state.support == algorithm.state.support
    assert restored.state.relay_components == algorithm.state.relay_components


# --- pair sessions -------------------------------------------------------------------

def test_a_pulse_session_negotiates_plans_and_commits_a_mix():
    """A session with enough credit commits: the payload is charged, the mix changes
    component 0 on at least one peer, and the support stays the same."""
    algorithms = {1: make_peer(1), 2: make_peer(2)}
    frame = load_trace({
        "trace_schema_version": 1, "time_unit": "ms", "duration_unit": "ms",
        "goodput_unit": "bps", "peers": [1, 2],
        "frames": [{"index": 0, "time": 0, "edges": [
            {"endpoints": [1, 2], "duration": 1000, "goodput": 1_000_000}]}],
    }).frame(0)
    before = {pid: a.state.gather(0) for pid, a in algorithms.items()}

    ledger = ByteLedger()
    record = execute_pair_session(compile_pair_sessions(frame)[0], algorithms,
                                  TransactionEngine(), ledger)

    assert record["status"] == "committed"
    assert record["payload_bytes"] > 0, "the plan was framed, sent and debited"
    for pid, algorithm in algorithms.items():
        assert algorithm.state.support == (0, 1, 2, 3), "support must not change"
        algorithm.state.check_invariants()
    moved = [pid for pid, algorithm in algorithms.items()
             if not all(torch.equal(algorithm.state.gather(0)[k], before[pid][k])
                        for k in before[pid])]
    assert moved, "a committed mix changed nothing"


def test_a_pulse_session_is_control_only_when_nothing_can_fit():
    """A session whose payload credit cannot fit one component ends control-only."""
    algorithms = {1: make_peer(1), 2: make_peer(2)}
    frame = load_trace({
        "trace_schema_version": 1, "time_unit": "ms", "duration_unit": "ms",
        "goodput_unit": "bps", "peers": [1, 2],
        "frames": [{"index": 0, "time": 0, "edges": [
            {"endpoints": [1, 2], "duration": 5, "goodput": 1_000_000}]}],
    }).frame(0)

    record = execute_pair_session(compile_pair_sessions(frame)[0], algorithms,
                                  TransactionEngine(), ByteLedger())

    # 372 bytes of payload credit against a 402-byte encoded component: the handshake
    # itself fits, so this is a capacity refusal rather than an unusable link.
    assert record["status"] == "control_only"
    assert record["offered_payload_credit"] < unit_cost_bytes(algorithms[1].components)
    assert record["payload_bytes"] == 0


def test_the_session_modules_are_present_and_core_defines_no_selection():
    """Selection is method policy: the pulse_lora session modules exist, and no name
    defined in core/ refers to selection."""
    import pathlib

    package = pathlib.Path(__file__).resolve().parents[2].joinpath("methods/pulse_lora").resolve()
    present = {p.name for p in package.glob("*.py")}
    assert {"selection.py", "scoring.py", "session.py", "units.py",
            "ages.py"} <= present

    # AST, not a substring scan: core's docstrings describe the seam a method plugs
    # into, so only defined names (functions, classes, assigned names and
    # attributes) are checked.
    import ast

    banned = ("selector", "candidate", "disagreement", "eligible_age",
              "exploration", "sketch")
    core = pathlib.Path(__file__).resolve().parents[2].joinpath("core").resolve()
    offences = []
    for module in sorted(core.glob("*.py")):
        tree = ast.parse(module.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names = [node.id]
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                names = [node.attr]
            for name in names:
                lowered = name.lower()
                if any(word in lowered for word in banned):
                    offences.append(f"core/{module.name}: {name}")
    assert not offences, f"selection is method policy, but core defines: {offences}"


def test_method_metrics_expose_the_pulse_state():
    algorithm = make_peer()
    metrics = algorithm.method_metrics()
    for key in ("pulse_active_rank", "pulse_materialised", "pulse_relay_components",
                "pulse_relay_memory_bytes", "pulse_r_max", "pulse_scale",
                "pulse_component_cost_bytes", "pulse_identity", "pulse_controller"):
        assert key in metrics, key
