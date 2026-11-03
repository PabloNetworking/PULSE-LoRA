"""GLUE trainer whose optimizer persists across local episodes.

``GLUEBaseTrainer._hook_on_fit_start_init`` builds a new optimizer and scheduler
on every ``train()`` call, which would discard the AdamW moments at every local
episode. A decentralized peer runs many local episodes, between and independently
of its peer interactions, so this trainer builds the optimizer and scheduler once
per peer and reuses them. This works because the base assigns ``ctx.optimizer``
as a plain attribute rather than a routine-scoped ``CtxVar``, so
``ctx.clear("routine")`` does not delete it.

Changes to the optimizer state that a method needs (moment resets, migration when
a support shrinks, factor alternation) are made by the method, not by this trainer.
"""
import logging

from federatedscope.contrib.trainer.glue_base_trainer import GLUEBaseTrainer
from federatedscope.core.trainers.enums import MODE
from federatedscope.register import register_trainer

logger = logging.getLogger(__name__)


class GLUEDecentralizedTrainer(GLUEBaseTrainer):
    """Task-generic GLUE trainer whose optimizer survives local episodes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._persistent_optimizer = None
        self._persistent_scheduler = None
        self.local_episodes = 0
        #: optimizer.step() calls, counted by a hook rather than inferred, so a
        #: caller can check that the configured number of local steps ran.
        self.optimizer_steps = 0
        #: Batches the base trainer skipped because their loss was NaN. A skipped
        #: batch consumes data but makes no optimizer step; this count reconciles
        #: the batches drawn with optimizer_steps.
        self.skipped_batches = 0

    def _hook_on_fit_start_init(self, ctx):
        # Run the base hook, which sets up the routine-scoped CtxVars, then swap in
        # the persistent optimizer. The optimizer the base builds on later calls is
        # discarded unused, which is cheap: AdamW allocates its state at the first step.
        super()._hook_on_fit_start_init(ctx)

        if ctx.cur_mode not in (MODE.TRAIN, MODE.FINETUNE):
            return

        if self._persistent_optimizer is None:
            # The first episode keeps the optimizer and scheduler the base built,
            # with a post-step hook that counts optimizer steps (see _counted).
            self._persistent_optimizer = self._counted(ctx.optimizer)
            self._persistent_scheduler = ctx.scheduler
            ctx.optimizer = self._persistent_optimizer
            logger.debug("decentralized trainer: created the persistent optimizer")
        else:
            ctx.optimizer = self._persistent_optimizer
            ctx.scheduler = self._persistent_scheduler

        self.local_episodes += 1

    def _hook_on_batch_backward(self, ctx):
        if getattr(ctx, "skip_this_batch", False):
            self.skipped_batches += 1
        super()._hook_on_batch_backward(ctx)

    def _counted(self, optimizer):
        """Count optimizer updates without replacing ``step``.

        The scheduler is already bound to the optimizer here, and replacing the
        bound ``step`` would make torch warn that step() was overridden after the
        LR scheduler was created. A post-step hook counts the same events and
        leaves ``step`` untouched.
        """
        trainer = self

        def _observe(_optimizer, _args, _kwargs):
            trainer.optimizer_steps += 1

        register = getattr(optimizer, "register_step_post_hook", None)
        if callable(register):
            optimizer.step  # noqa: B018 - attribute access only; step is not replaced
            register(_observe)
            return optimizer

        # Fallback for a torch without step hooks: wrap step (torch then warns).
        original_step = optimizer.step

        def step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            trainer.optimizer_steps += 1
            return result

        optimizer.step = step
        return optimizer

    # -- method-facing hooks -------------------------------------------------

    @property
    def optimizer(self):
        return self._persistent_optimizer

    def trainable_parameters(self):
        return [p for p in self.ctx.model.parameters() if p.requires_grad]

    def ensure_optimizer(self):
        """Create the persistent optimizer now if it does not exist yet.

        It is otherwise created at the first fit start, which has not taken place
        when a peer restores a checkpoint, so there would be nothing to load the
        optimizer and scheduler state into. ``load_trainer_state`` calls it.
        """
        if self._persistent_optimizer is not None:
            return self._persistent_optimizer

        from federatedscope.core.trainers.enums import MODE

        context = self.ctx
        previous_mode = getattr(context, "cur_mode", None)
        context.cur_mode = MODE.TRAIN
        try:
            self._hook_on_fit_start_init(context)
        finally:
            if previous_mode is not None:
                context.cur_mode = previous_mode
        # the hook counts a local episode; restoring is not one
        self.local_episodes = max(0, self.local_episodes - 1)
        return self._persistent_optimizer

    def load_trainer_state(self, record) -> None:
        """Restore optimizer, scheduler and counters with exact-resume semantics."""
        if not record:
            return
        self.ensure_optimizer()
        if record.get("optimizer") is not None:
            self._persistent_optimizer.load_state_dict(record["optimizer"])
        if record.get("scheduler") is not None and self._persistent_scheduler is not None:
            self._persistent_scheduler.load_state_dict(record["scheduler"])
        self.optimizer_steps = int(record["optimizer_steps"])
        self.skipped_batches = int(record["skipped_batches"])
        self.local_episodes = int(record["local_episodes"])

    def trainer_state(self) -> dict:
        optimizer = self._persistent_optimizer
        scheduler = self._persistent_scheduler
        return {
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "optimizer_steps": self.optimizer_steps,
            "skipped_batches": self.skipped_batches,
            "local_episodes": self.local_episodes,
        }

    def reset_optimizer(self):
        """Drop the persistent optimizer and scheduler so the next episode rebuilds them.

        For a method whose trainable parameters change shape. None of the methods
        here calls it: PULSE-LoRA migrates the optimizer state in place when it
        prunes.
        """
        self._persistent_optimizer = None
        self._persistent_scheduler = None


def call_glue_decentralized_trainer(trainer_type):
    if trainer_type == "gluedecentralizedtrainer":
        return GLUEDecentralizedTrainer
    return None


register_trainer("gluedecentralizedtrainer", call_glue_decentralized_trainer)
