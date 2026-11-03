"""The PULSE-LoRA selector is a method policy, not part of the shared runtime.

* **Swapping the selector changes only what PULSE-LoRA chooses to send.** It
  does not touch core, does not perturb the opportunity trace or the local-work
  schedule, and does not make two runs incomparable; otherwise a comparison
  between selectors would be confounded.

* **Dec-LoRA has no component selection.** It has no component selector and
  exchanges one indivisible full-state unit.
"""
import ast
import pathlib

import pytest

from federatedscope.contrib.decentralized.core.manifest import assert_comparable, build_manifest
from federatedscope.contrib.decentralized.core.runner import DecentralizedRunner
from federatedscope.contrib.decentralized.core.traces import compute_trace_digest
from federatedscope.contrib.decentralized.methods.pulse_lora import selection
from federatedscope.contrib.decentralized.tests.methods.test_pulse_peer import (
    ToyPulseFactory, pulse_cfg,
)

DEPLOYABLE = [s for s in selection.SELECTORS if s not in selection.ORACLE_SELECTORS]


def run(selector, **overrides):
    settings = {"decentralized.methods.pulse_lora.selector": selector}
    settings.update(overrides)
    cfg = pulse_cfg(client_num=4, frames=4, **settings)
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyPulseFactory(cfg))
    results = runner.run()
    return runner, results


# -- a selector swap perturbs nothing exogenous --------------------------------------

@pytest.mark.parametrize("selector", DEPLOYABLE)
def test_a_selector_swap_leaves_the_opportunity_trace_identical(selector):
    baseline, _ = run("age_only")
    other, _ = run(selector)
    assert other.trace.trace_id == baseline.trace.trace_id
    # trace_id is the digest of the normalised trace, so compare the content
    # too: a bug changing both together would otherwise slip through.
    assert other.trace.normalised == baseline.trace.normalised
    assert compute_trace_digest(other.trace.normalised) == other.trace.trace_id


@pytest.mark.parametrize("selector", DEPLOYABLE)
def test_a_selector_swap_leaves_the_offered_communication_identical(selector):
    baseline, _ = run("age_only")
    other, _ = run(selector)
    left, right = build_manifest(baseline), build_manifest(other)
    for axis in ("trace_id", "expected_trace_digest", "peer_ids", "total_frames",
                 "local_steps_per_episode", "gamma", "auxiliary_state_policy",
                 "wire_version", "seed", "checksum", "comparison_profile",
                 "interaction_form", "trace_frames", "scheduling"):
        assert left[axis] == right[axis], f"selector swap moved the shared axis {axis}"

    # ...and the selector *is* recorded, in the one place that should differ.
    # applied_method_settings is descriptive, not a comparability axis, so two
    # selectors stay comparable while the manifest still names each one.
    assert left["applied_method_settings"]["pulse_selector"] == "age_only"
    assert right["applied_method_settings"]["pulse_selector"] == selector


@pytest.mark.parametrize("selector", DEPLOYABLE)
def test_two_selectors_remain_directly_comparable(selector):
    baseline, _ = run("age_only")
    other, _ = run(selector)
    assert_comparable(build_manifest(baseline), build_manifest(other))


@pytest.mark.parametrize("selector", DEPLOYABLE)
def test_a_selector_swap_leaves_the_local_work_identical(selector):
    """Selection must not change how many local episodes a peer trains."""
    baseline, _ = run("age_only")
    other, _ = run(selector)
    assert ({p: peer.algorithm.episodes for p, peer in other.peers.items()}
            == {p: peer.algorithm.episodes for p, peer in baseline.peers.items()})


def test_the_selectors_select_components():
    """Components are selected in these runs; otherwise the tests above would
    compare runs in which nothing was selected."""
    chosen = {}
    for selector in DEPLOYABLE:
        runner, _ = run(selector)
        chosen[selector] = tuple(sorted(
            (p, peer.algorithm.policy.metrics["units_selected"])
            for p, peer in runner.peers.items()))
    assert len({tuple(v) for v in chosen.values()}) > 1 or any(
        v[0][1] > 0 for v in chosen.values()), (
        "no selector selected anything, so nothing above was actually exercised")


def test_the_selector_is_recorded_in_the_run_metrics():
    runner, _ = run("sketch_disagreement")
    for peer in runner.peers.values():
        metrics = peer.algorithm.method_metrics()
        assert metrics["pulse_selector"] == "sketch_disagreement"
        assert "pulse_max_eligible_age" in metrics
        assert "pulse_units_selected" in metrics


# -- the shared runtime has no selection code ----------------------------------------------

def test_no_selector_name_appears_in_core():
    """Only the distinctive names.

    'random' is also the name of Python's own module, which core/rng.py imports,
    so a substring scan for it would match that import. Every other selector
    name could only come from the PULSE-LoRA method.
    """
    distinctive = tuple(n for n in selection.SELECTORS if n != "random")
    core = pathlib.Path(__file__).resolve().parents[2].joinpath("core").resolve()
    for module in sorted(core.glob("*.py")):
        text = module.read_text()
        for name in distinctive:
            assert name not in text, f"core/{module.name} names selector {name!r}"
        assert "pulse_lora" not in text, f"core/{module.name} names pulse_lora"


def test_the_transaction_engine_has_no_selection_code():
    """core/transaction.py imports nothing from the methods or PULSE-LoRA, and defines
    no function or class named after pulse or selection."""
    engine = pathlib.Path(__file__).resolve().parents[2].joinpath("core/transaction.py").resolve()
    tree = ast.parse(engine.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", "") or ""
            assert "methods" not in module and "pulse" not in module
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            lowered = node.name.lower()
            assert "pulse" not in lowered and "select" not in lowered


# -- Dec-LoRA has no component selection ---------------------------------------------------

def test_dec_lora_uses_no_component_selector():
    from federatedscope.contrib.decentralized.methods import dec_lora

    package = pathlib.Path(dec_lora.__file__).resolve().parent
    for module in sorted(package.glob("*.py")):
        text = module.read_text()
        assert "pulse" not in text.lower(), f"dec_lora/{module.name} references pulse"
        for name in ("selector", "candidate_components", "eligible_age", "sketch"):
            assert name not in text, f"dec_lora/{module.name} names {name!r}"


def test_dec_lora_runs_sessions_without_a_session_policy():
    from federatedscope.contrib.decentralized.tests.methods.test_dec_lora import (
        ToyLoRAFactory, dec_cfg,
    )

    cfg = dec_cfg(client_num=4, frames=4)
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyLoRAFactory(cfg))
    runner.run()
    for peer in runner.peers.values():
        assert not hasattr(peer.algorithm, "policy"), (
            "Dec-LoRA must not have a session policy")
        assert peer.algorithm.committed_sessions + peer.algorithm.unsuccessful_sessions > 0


# -- checkpoint / resume determinism with a selector in play --------------------------

@pytest.mark.parametrize("selector", ["local_update", "age_only", "random",
                                      "sketch_disagreement"])
def test_resume_reproduces_an_uninterrupted_selector_run_exactly(selector):
    """Byte-identical events, traffic and selector state across a mid-run resume."""
    import json
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cfg = pulse_cfg(client_num=4, frames=8, **{
            "decentralized.methods.pulse_lora.selector": selector,
            "decentralized.checkpoint.enabled": True,
            "decentralized.checkpoint.every_frames": 4,
            "decentralized.checkpoint.dir": tmp,
        })
        reference = DecentralizedRunner(data={}, config=cfg, factory=ToyPulseFactory(cfg))
        reference.run()
        reference_events = json.dumps(reference.sink.events, sort_keys=True)
        reference_ages = {p: peer.algorithm.policy.memory.state_dict()
                          for p, peer in reference.peers.items()}
        reference_metrics = {p: dict(peer.algorithm.policy.metrics)
                             for p, peer in reference.peers.items()}

        midpoint = os.path.join(tmp, "decentralized_frame3.ckpt")
        assert os.path.isfile(midpoint)

        plain = pulse_cfg(client_num=4, frames=8, **{
            "decentralized.methods.pulse_lora.selector": selector, "outdir": ""})
        resumed = DecentralizedRunner(data={}, config=plain,
                                      factory=ToyPulseFactory(plain))
        resumed.run(resume_from=midpoint)

        assert json.dumps(resumed.sink.events, sort_keys=True) == reference_events, \
            "resumed event log differs"
        assert resumed.ledger.total() == reference.ledger.total(), \
            "resumed traffic accounting differs"
        assert {p: peer.algorithm.policy.memory.state_dict()
                for p, peer in resumed.peers.items()} == reference_ages, \
            "resumed per-edge selector state differs"
        assert {p: dict(peer.algorithm.policy.metrics)
                for p, peer in resumed.peers.items()} == reference_metrics, \
            "resumed selector metrics differ"


def test_the_selector_state_is_actually_in_the_checkpoint():
    """Otherwise the resume test above would pass by both sides being empty."""
    import os
    import tempfile

    from federatedscope.contrib.decentralized.core.checkpoint import load_checkpoint

    with tempfile.TemporaryDirectory() as tmp:
        cfg = pulse_cfg(client_num=4, frames=4, **{
            "decentralized.methods.pulse_lora.selector": "local_update",
            "decentralized.checkpoint.enabled": True,
            "decentralized.checkpoint.every_frames": 2,
            "decentralized.checkpoint.dir": tmp,
        })
        DecentralizedRunner(data={}, config=cfg, factory=ToyPulseFactory(cfg)).run()
        payload = load_checkpoint(os.path.join(tmp, "decentralized_frame3.ckpt"))

    for peer_state in payload["peers"].values():
        policy = peer_state["algorithm"]["session_policy"]
        assert policy["memory"]["edges"], (
            "the checkpoint records no per-edge state, so the resume test compares nothing")
        assert any(edge["ages"] for edge in policy["memory"]["edges"].values()), (
            "no component was ever aged, so the resume test compares empty ages")
