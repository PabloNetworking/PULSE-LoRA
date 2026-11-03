"""The optimizer survives local episodes, and moment resets are precise."""
import types

import pytest
import torch

from federatedscope.contrib.decentralized.lora.optimizer_state import (
    clear_moments,
    has_state,
    moment_norms,
)
from federatedscope.contrib.decentralized.tests.fixtures import build_toy_lora_model
from federatedscope.contrib.trainer.glue_decentralized_trainer import GLUEDecentralizedTrainer
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.trainers.enums import MODE


def stub_trainer_and_ctx():
    """Exercise the real hook without building a full FederatedScope Context."""
    trainer = object.__new__(GLUEDecentralizedTrainer)
    trainer._persistent_optimizer = None
    trainer._persistent_scheduler = None
    trainer.local_episodes = 0
    trainer.optimizer_steps = 0
    trainer.skipped_batches = 0

    cfg = global_cfg.clone()
    ctx = types.SimpleNamespace(
        model=build_toy_lora_model(),
        device="cpu",
        cur_mode=MODE.TRAIN,
        cfg=cfg,
    )
    return trainer, ctx


def test_optimizer_persists_across_local_episodes():
    trainer, ctx = stub_trainer_and_ctx()

    trainer._hook_on_fit_start_init(ctx)
    first = ctx.optimizer
    first_scheduler = ctx.scheduler

    trainer._hook_on_fit_start_init(ctx)
    assert ctx.optimizer is first, "a fresh optimizer would discard Adam moments"
    assert ctx.scheduler is first_scheduler
    assert trainer.local_episodes == 2


def test_moment_state_survives_a_second_episode():
    trainer, ctx = stub_trainer_and_ctx()
    trainer._hook_on_fit_start_init(ctx)

    params = [p for p in ctx.model.parameters() if p.requires_grad]
    for param in params:
        param.grad = torch.ones_like(param)
    ctx.optimizer.step()
    assert has_state(ctx.optimizer)
    norms_before = moment_norms(ctx.optimizer)

    trainer._hook_on_fit_start_init(ctx)
    assert moment_norms(ctx.optimizer) == norms_before, "moments were discarded"


def test_reset_optimizer_forces_a_rebuild():
    trainer, ctx = stub_trainer_and_ctx()
    trainer._hook_on_fit_start_init(ctx)
    first = ctx.optimizer

    trainer.reset_optimizer()
    trainer._hook_on_fit_start_init(ctx)
    assert ctx.optimizer is not first


def test_eval_mode_does_not_create_an_optimizer():
    trainer, ctx = stub_trainer_and_ctx()
    ctx.cur_mode = MODE.TEST
    trainer._hook_on_fit_start_init(ctx)
    assert trainer._persistent_optimizer is None
    assert trainer.local_episodes == 0


# --- optimizer_state primitives ------------------------------------------------

def _optimizer_with_state():
    model = build_toy_lora_model()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    for param in params:
        param.grad = torch.ones_like(param)
    optimizer.step()
    return model, optimizer, params


def test_clear_moments_touches_exactly_the_named_parameters():
    _, optimizer, params = _optimizer_with_state()
    target = params[:2]

    touched = clear_moments(optimizer, target)

    assert touched == len(target)
    for param in target:
        assert float(optimizer.state[param]["exp_avg"].abs().sum()) == 0.0
    for param in params[2:]:
        assert float(optimizer.state[param]["exp_avg"].abs().sum()) > 0.0


def test_clear_moments_on_an_untracked_parameter_is_a_no_op():
    _, optimizer, _ = _optimizer_with_state()
    assert clear_moments(optimizer, [torch.nn.Parameter(torch.zeros(2))]) == 0


def test_has_state_is_false_before_the_first_step():
    model = build_toy_lora_model()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3)
    assert not has_state(optimizer)
