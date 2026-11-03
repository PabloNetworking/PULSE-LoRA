"""PULSE-LoRA relay propagation along a three-peer chain, and resume after committed
and failed sessions.

The relay plane is what lets a narrow peer carry a component it does not train.
A three-peer chain propagates a component through the middle peer *without
changing that peer's active support*: pruning is monotone, so relaying must never
activate a component.
"""
import json
import os
import tempfile

import pytest
import torch

from federatedscope.contrib.decentralized.core.accounting import ByteLedger
from federatedscope.contrib.decentralized.core.pair_session import execute_pair_session
from federatedscope.contrib.decentralized.core.runner import DecentralizedRunner
from federatedscope.contrib.decentralized.core.transaction import (
    TransactionEngine,
)
from federatedscope.contrib.decentralized.tests.methods.test_pulse_peer import (
    ToyPulseFactory, make_peer, pulse_cfg,
)
from federatedscope.contrib.decentralized.tests.methods.test_pulse_session import (
    event, trained,
)


def chain_peers(selector="full_exchange"):
    cfg = pulse_cfg(**{"decentralized.methods.pulse_lora.selector": selector})
    return (make_peer(1, cfg=cfg), make_peer(2, cfg=cfg), make_peer(3, cfg=cfg))


# -- three-peer relay propagation ------------------------------------------------------

def test_a_component_propagates_through_a_narrow_middle_peer():
    left, middle, right = chain_peers()
    trained(left, seed=1)
    middle.state.prune([2, 3])                  # the middle peer trains only 0 and 1
    support_before = middle.state.support
    assert support_before == (0, 1)

    first = execute_pair_session(event(1, 2), {1: left, 2: middle},
                                 TransactionEngine(), ByteLedger())
    second = execute_pair_session(event(2, 3), {2: middle, 3: right},
                                  TransactionEngine(), ByteLedger())

    assert first["status"] == "committed" and second["status"] == "committed"
    assert middle.state.support == support_before, (
        "relaying activated a component; PULSE never grows a support")
    assert set(middle.state.relay_components) >= {2, 3}
    middle.state.check_invariants()


def test_the_middle_peer_forwards_a_value_it_never_trained():
    left, middle, right = chain_peers()
    trained(left, seed=7)
    middle.state.prune([2, 3])
    carried = left.state.gather(2)

    execute_pair_session(event(1, 2), {1: left, 2: middle},
                         TransactionEngine(), ByteLedger())
    relayed = middle.state.gather(2)
    assert any(not torch.equal(relayed[k], v) for k, v in carried.items()), (
        "the middle peer did not take up the component at all")

    before_right = right.state.gather(2)
    execute_pair_session(event(2, 3), {2: middle, 3: right},
                         TransactionEngine(), ByteLedger())
    after_right = right.state.gather(2)

    assert any(not torch.equal(after_right[k], before_right[k]) for k in after_right), (
        "the far peer never saw the component the middle peer relayed")
    assert middle.state.support == (0, 1)


def test_a_relayed_component_receives_no_gradient_on_the_middle_peer():
    left, middle, _ = chain_peers()
    trained(left, seed=1)
    middle.state.prune([2, 3])
    execute_pair_session(event(1, 2), {1: left, 2: middle},
                         TransactionEngine(), ByteLedger())

    # A relay component lives on CPU, detached, outside the compact tensors.
    for component in (2, 3):
        block = middle.state.gather(component)
        for tensor in block.values():
            assert not tensor.requires_grad
        assert not middle.state.is_active(component)
    assert middle.bridge.schema.rank == 2, "the compact model still holds only S_i"


# -- resume with committed and failed sessions -------------------------------------------

def run_with_checkpoints(directory, frames=8, every=4, injector=None, extra=None):
    settings = {
        "decentralized.methods.pulse_lora.selector": "full_exchange",
        "decentralized.checkpoint.enabled": True,
        "decentralized.checkpoint.every_frames": every,
        "decentralized.checkpoint.dir": directory,
    }
    if injector:
        settings["decentralized.transaction.failure_injection_stage"] = injector[0]
        settings["decentralized.transaction.failure_injection_every"] = injector[1]
    settings.update(extra or {})
    cfg = pulse_cfg(client_num=4, frames=frames, **settings)
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyPulseFactory(cfg))
    runner.run()
    return runner, cfg


def resume_from(checkpoint, frames=8, injector=None, extra=None):
    settings = {"decentralized.methods.pulse_lora.selector": "full_exchange",
                "outdir": ""}
    if injector:
        settings["decentralized.transaction.failure_injection_stage"] = injector[0]
        settings["decentralized.transaction.failure_injection_every"] = injector[1]
    settings.update(extra or {})
    cfg = pulse_cfg(client_num=4, frames=frames, **settings)
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyPulseFactory(cfg))
    runner.run(resume_from=checkpoint)
    return runner


def peer_state(runner):
    return {pid: {c: {k: v.clone() for k, v in peer.algorithm.state.gather(c).items()}
                  for c in range(peer.algorithm.components.r_max)}
            for pid, peer in runner.peers.items()}


@pytest.mark.parametrize("injector", [None, ("commit", 2), ("stage", 2)])
def test_resume_reproduces_a_mixing_run_exactly(injector):
    """Resuming from the frame-3 checkpoint reproduces the uninterrupted run.

    Covered with no injected failure and with failures injected at commit or at
    stage, so the resumed run replays committed and rolled-back sessions.
    """
    with tempfile.TemporaryDirectory() as tmp:
        reference, _ = run_with_checkpoints(tmp, injector=injector)
        reference_events = json.dumps(reference.sink.events, sort_keys=True)
        reference_state = peer_state(reference)
        reference_versions = {p: peer.algorithm.state_version()
                              for p, peer in reference.peers.items()}

        midpoint = os.path.join(tmp, "decentralized_frame3.ckpt")
        assert os.path.isfile(midpoint)
        resumed = resume_from(midpoint, injector=injector)

    assert json.dumps(resumed.sink.events, sort_keys=True) == reference_events, \
        "resumed event log differs"
    assert resumed.ledger.total() == reference.ledger.total(), "traffic differs"
    assert {p: peer.algorithm.state_version()
            for p, peer in resumed.peers.items()} == reference_versions
    resumed_state = peer_state(resumed)
    for pid, blocks in reference_state.items():
        for component, block in blocks.items():
            for key, tensor in block.items():
                assert torch.equal(resumed_state[pid][component][key], tensor), \
                    f"peer {pid} component {component} differs after resume"


def test_a_run_with_injected_failures_really_did_fail_some_sessions():
    """Otherwise the rolled-back resume case above would test only committed sessions."""
    with tempfile.TemporaryDirectory() as tmp:
        runner, _ = run_with_checkpoints(tmp, injector=("commit", 2))
    outcomes = [e["status"] for e in runner.sink.events if "status" in e]
    assert "failed" in outcomes, f"no session failed; outcomes were {set(outcomes)}"


def test_a_run_without_injection_really_did_commit_some_sessions():
    with tempfile.TemporaryDirectory() as tmp:
        runner, _ = run_with_checkpoints(tmp)
    outcomes = [e["status"] for e in runner.sink.events if "status" in e]
    assert "committed" in outcomes, f"nothing committed; outcomes were {set(outcomes)}"
