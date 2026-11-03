"""Exactly the requested number of optimizer steps, counted rather than assumed."""
import types

import pytest
import torch

from federatedscope.contrib.decentralized.tests.fixtures import build_toy_lora_model
from federatedscope.contrib.trainer.glue_decentralized_trainer import GLUEDecentralizedTrainer
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.trainers.enums import MODE


def stub_trainer():
    trainer = object.__new__(GLUEDecentralizedTrainer)
    trainer._persistent_optimizer = None
    trainer._persistent_scheduler = None
    trainer.local_episodes = 0
    trainer.optimizer_steps = 0
    trainer.skipped_batches = 0
    ctx = types.SimpleNamespace(model=build_toy_lora_model(), device="cpu",
                                cur_mode=MODE.TRAIN, cfg=global_cfg.clone())
    trainer._hook_on_fit_start_init(ctx)
    return trainer, ctx


def test_optimizer_steps_are_counted_not_inferred():
    trainer, ctx = stub_trainer()
    for param in ctx.model.parameters():
        if param.requires_grad:
            param.grad = torch.ones_like(param)

    assert trainer.optimizer_steps == 0
    for expected in range(1, 4):
        ctx.optimizer.step()
        assert trainer.optimizer_steps == expected


def test_the_counter_survives_the_persistent_optimizer_across_episodes():
    trainer, ctx = stub_trainer()
    for param in ctx.model.parameters():
        if param.requires_grad:
            param.grad = torch.ones_like(param)
    ctx.optimizer.step()

    trainer._hook_on_fit_start_init(ctx)   # next local episode
    ctx.optimizer.step()

    assert trainer.optimizer_steps == 2, "the step counter must survive optimizer reuse"


def test_skipped_batches_are_counted_separately():
    """A NaN-loss batch consumes a batch but performs no optimizer step."""
    trainer, ctx = stub_trainer()
    ctx.skip_this_batch = True
    trainer._hook_on_batch_backward(ctx)
    assert trainer.skipped_batches == 1
    assert trainer.optimizer_steps == 0, "a skipped batch must not count as a step"


# --- the plugin reconciles the two counts ------------------------------------------

class FakeTrainer:
    def __init__(self, steps, skips=0):
        self.optimizer_steps = 0
        self.skipped_batches = 0
        self._steps = steps
        self._skips = skips

    def train(self):
        self.optimizer_steps += self._steps
        self.skipped_batches += self._skips
        return 0, None, {"train_loss": 1.0}


def peer_with(trainer):
    from federatedscope.contrib.decentralized.tests.methods.test_dec_lora import make_peer

    peer = make_peer(1)
    peer.trainer = trainer
    return peer


def test_requested_steps_are_reported_as_performed():
    peer = peer_with(FakeTrainer(steps=5))
    metrics = peer.local_episode(5)
    assert metrics["optimizer_steps"] == 5
    assert metrics["requested_steps"] == 5
    assert metrics["skipped_batches"] == 0


def test_a_skipped_batch_reconciles_against_the_request():
    peer = peer_with(FakeTrainer(steps=4, skips=1))
    metrics = peer.local_episode(5)
    assert metrics["optimizer_steps"] == 4
    assert metrics["skipped_batches"] == 1


def test_an_unaccounted_step_count_is_a_hard_error():
    """Performed steps plus skipped batches must equal the requested steps."""
    peer = peer_with(FakeTrainer(steps=30))
    with pytest.raises(RuntimeError, match="does not account for the 5 requested"):
        peer.local_episode(5)


def test_peer_records_the_actual_count_not_the_requested_one():
    from federatedscope.contrib.decentralized.core.interfaces import PeerContext
    from federatedscope.contrib.decentralized.core.peer import Peer

    algorithm = peer_with(FakeTrainer(steps=4, skips=1))
    shell = Peer(1, algorithm, PeerContext(peer_id=1, config=None))
    shell.local_episode(5)

    assert shell.local_steps == 4, "recording the requested value would restate the config"
    assert shell.skipped_batches == 1
