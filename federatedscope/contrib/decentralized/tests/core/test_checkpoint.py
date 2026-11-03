"""Checkpoint and resume must reproduce a run exactly."""
import json
import os
import tempfile

import pytest

from federatedscope.contrib.decentralized.core.checkpoint import load_checkpoint
from federatedscope.contrib.decentralized.core.runner import DecentralizedRunner
from federatedscope.contrib.decentralized.tests.core.test_runner import build, toy_cfg
from federatedscope.contrib.decentralized.tests.core.toy_method import ToyFactory


def run_with_checkpoints(directory, frames=8, every=4):
    cfg = toy_cfg(frames=frames, **{
        "decentralized.checkpoint.enabled": True,
        "decentralized.checkpoint.every_frames": every,
        "decentralized.checkpoint.dir": directory,
    })
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyFactory(cfg))
    runner.run()
    return runner


def test_resume_reproduces_an_uninterrupted_run_exactly():
    with tempfile.TemporaryDirectory() as tmp:
        reference = run_with_checkpoints(tmp, frames=8, every=4)
        reference_values = {pid: peer.algorithm.value
                            for pid, peer in reference.peers.items()}
        reference_events = json.dumps(reference.sink.events, sort_keys=True)

        midpoint = os.path.join(tmp, "decentralized_frame3.ckpt")
        assert os.path.isfile(midpoint), "expected a checkpoint at frame 3"

        cfg = toy_cfg(frames=8, **{"outdir": ""})
        resumed = DecentralizedRunner(data={}, config=cfg, factory=ToyFactory(cfg))
        resumed.run(resume_from=midpoint)

        assert {pid: peer.algorithm.value for pid, peer in resumed.peers.items()} == \
               reference_values, "resumed peer values differ"
        assert json.dumps(resumed.sink.events, sort_keys=True) == reference_events, \
            "resumed event log differs"
        assert resumed.ledger.total() == reference.ledger.total(), \
            "resumed traffic accounting differs"


def test_checkpoint_carries_rng_traffic_and_topology():
    with tempfile.TemporaryDirectory() as tmp:
        runner = run_with_checkpoints(tmp, frames=4, every=2)
        payload = load_checkpoint(os.path.join(tmp, "decentralized_frame1.ckpt"))

        assert set(payload) == {"coordinator", "queue", "ledger", "engine",
                                "injector", "peers", "framework_rng"}
        # training draws on the framework's global generators, so the
        # checkpoint has to capture them too
        assert set(payload["framework_rng"]) >= {"torch", "python", "numpy"}
        assert payload["coordinator"]["trace_id"] == runner.trace.trace_id
        assert payload["coordinator"]["rng"]["master_seed"] == runner.rng.master_seed
        assert payload["coordinator"]["topology"]["frames"]
        assert payload["ledger"]["records"]
        assert payload["engine"]["next_id"] > 1, "transaction numbering must survive"


def test_resume_refuses_a_checkpoint_from_a_different_trace():
    with tempfile.TemporaryDirectory() as tmp:
        run_with_checkpoints(tmp, frames=4, every=2)
        checkpoint = os.path.join(tmp, "decentralized_frame1.ckpt")

        # a different seed produces a different offered trace
        cfg = toy_cfg(frames=4, **{"decentralized.seed": 99, "outdir": ""})
        other = DecentralizedRunner(data={}, config=cfg, factory=ToyFactory(cfg))
        with pytest.raises(ValueError, match="refusing to resume"):
            other.run(resume_from=checkpoint)


def test_checkpointing_is_off_by_default():
    runner = build()
    runner.run()
    assert runner._maybe_checkpoint(0) is None


# -- a consumer that captured its stream before the restore must still replay ----------

def test_a_peer_that_captured_its_rng_before_loading_resumes_byte_identically():
    """The runner builds peers, and hands them their streams, before it loads the
    checkpoint. A method that draws from ``context.rng`` therefore reproduces the
    uninterrupted run only if the restore keeps the generator objects it holds.
    """
    import json
    import os
    import tempfile

    from federatedscope.contrib.decentralized.tests.core.toy_method import ToyFactory

    class DrawingFactory(ToyFactory):
        """A toy method that draws from its named stream each episode."""

        def create_peer(self, context):
            peer = super().create_peer(context)
            # Captured by reference when the peer is built, as a method does; the
            # runner builds peers before it loads a checkpoint.
            stream = context.rng
            original = peer.local_episode

            def local_episode(steps, _peer=peer, _original=original, _rng=stream):
                result = dict(_original(steps))
                _peer.value = _peer.value + _rng.random()
                return result

            peer.local_episode = local_episode
            return peer

    with tempfile.TemporaryDirectory() as tmp:
        cfg = toy_cfg(frames=8, **{
            "decentralized.checkpoint.enabled": True,
            "decentralized.checkpoint.every_frames": 4,
            "decentralized.checkpoint.dir": tmp,
        })
        reference = DecentralizedRunner(data={}, config=cfg, factory=DrawingFactory(cfg))
        reference.run()
        reference_values = {p: peer.algorithm.value for p, peer in reference.peers.items()}
        reference_events = json.dumps(reference.sink.events, sort_keys=True)

        plain = toy_cfg(frames=8, **{"outdir": ""})
        resumed = DecentralizedRunner(data={}, config=plain, factory=DrawingFactory(plain))
        resumed.run(resume_from=os.path.join(tmp, "decentralized_frame3.ckpt"))

    assert {p: peer.algorithm.value for p, peer in resumed.peers.items()} == \
        reference_values, "a peer drawing from its captured stream did not replay"
    assert json.dumps(resumed.sink.events, sort_keys=True) == reference_events
