"""Base GLUE trainer.

This file contains the task-generic GLUE training and evaluation hooks for
sequence classification and regression.
"""

import logging

import torch
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import LIFECYCLE, MODE

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _safe_tensor_to_numpy(tensor):
    """Convert a tensor to NumPy safely for metric computation.

    NumPy does not support direct conversion from torch.bfloat16, so floating
    tensors are promoted to float32 before calling ``.numpy()``. Integer and
    boolean tensors preserve their dtype.
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    tensor = tensor.detach().cpu()
    if tensor.is_floating_point():
        tensor = tensor.to(torch.float32)
    return tensor.numpy()



class GLUEBaseTrainer(GeneralTorchTrainer):
    """Task-generic trainer for GLUE classification and regression."""

    def _hook_on_fit_start_numerical_precision(self, ctx):
        if self.cfg.train.is_enable_half:
            if not (
                hasattr(ctx.cfg, 'llm') and
                hasattr(ctx.cfg.llm, 'deepspeed') and
                ctx.cfg.llm.deepspeed.use
            ):
                ctx.model = ctx.model.half()

    def _hook_on_fit_start_init(self, ctx):
        ctx.model.to(ctx.device)

        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            ctx.optimizer = get_optimizer(
                ctx.model, **ctx.cfg[ctx.cur_mode].optimizer
            )
            ctx.scheduler = get_scheduler(
                ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler
            )

        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)

        self.is_regression = False
        if hasattr(ctx.cfg, 'model'):
            num_labels = getattr(ctx.cfg.model, 'out_channels', 2)
            self.is_regression = (num_labels == 1)

    def _hook_on_batch_forward(self, ctx):
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
        labels = ctx.data_batch['labels'].to(ctx.device)
        # CrossEntropyLoss requires Long targets, so float class labels are cast.
        # Regression tasks (STS-B, out_channels=1) keep float labels.
        if labels.is_floating_point() and getattr(ctx.cfg.model, 'out_channels', 2) > 1:
            labels = labels.long()

        token_type_ids = ctx.data_batch.get('token_type_ids', None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(ctx.device)

        outputs = ctx.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

        loss = outputs.loss
        logits = outputs.logits

        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning(
                'Skip batch due to NaN loss. This may be caused by '
                'precision issues or invalid labels.'
            )
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        if self.is_regression:
            ctx.y_prob = CtxVar(logits.squeeze(-1), LIFECYCLE.BATCH)
        else:
            ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_batch_backward(self, ctx):
        if ctx.skip_this_batch:
            return

        ctx.optimizer.zero_grad()
        ctx.loss_task.backward()

        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                ctx.model.parameters(), ctx.grad_clip
            )

        ctx.optimizer.step()

        if ctx.scheduler is not None:
            ctx.scheduler.step()

    def _hook_on_batch_end(self, ctx):
        if ctx.skip_this_batch:
            if hasattr(ctx.cfg, 'glue') and \
               getattr(ctx.cfg.glue, 'retry_on_nan_loss', False):
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return

        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get('loss_regular', 0.))
        ctx.ys_true.append(ctx.y_true.detach().cpu())
        ctx.ys_prob.append(ctx.y_prob.detach().cpu())

    def _hook_on_fit_end(self, ctx):
        avg_loss = 0 if float(ctx.num_samples) == 0 else \
            ctx.loss_batch_total / float(ctx.num_samples)

        eval_results = {
            f'{ctx.cur_split}_loss': ctx.loss_batch_total,
            f'{ctx.cur_split}_total': ctx.num_samples,
            f'{ctx.cur_split}_avg_loss': avg_loss,
        }

        if ctx.ys_true and ctx.ys_prob:
            try:
                all_true = torch.cat(ctx.ys_true, dim=0)
                all_prob = torch.cat(ctx.ys_prob, dim=0)

                if self.is_regression:
                    from scipy.stats import pearsonr, spearmanr
                    all_true_np = _safe_tensor_to_numpy(all_true).reshape(-1)
                    all_prob_np = _safe_tensor_to_numpy(all_prob).reshape(-1)
                    pearson = pearsonr(all_true_np, all_prob_np)[0]
                    spearman = spearmanr(all_true_np, all_prob_np)[0]
                    eval_results[f'{ctx.cur_split}_pearson'] = pearson
                    eval_results[f'{ctx.cur_split}_spearman'] = spearman
                else:
                    predictions = torch.argmax(all_prob, dim=-1)
                    correct = (predictions == all_true).sum().item()
                    total = len(all_true)
                    accuracy = correct / total if total > 0 else 0.0
                    eval_results[f'{ctx.cur_split}_acc'] = accuracy
                    eval_results[f'{ctx.cur_split}_correct'] = correct

                    if all_prob.shape[-1] == 2:
                        true_np = _safe_tensor_to_numpy(all_true).reshape(-1)
                        pred_np = _safe_tensor_to_numpy(predictions).reshape(-1)
                        from sklearn.metrics import f1_score, matthews_corrcoef
                        eval_results[f'{ctx.cur_split}_f1'] = f1_score(
                            true_np, pred_np, average='binary'
                        )
                        eval_results[f'{ctx.cur_split}_mcc'] = matthews_corrcoef(
                            true_np, pred_np
                        )
            except Exception as e:
                logger.warning(f"Error computing metrics: {e}")

        setattr(ctx, 'eval_metrics', eval_results)

        adapter_cfg = None
        if hasattr(ctx.cfg, 'glue') and hasattr(ctx.cfg.glue, 'adapter'):
            adapter_cfg = ctx.cfg.glue.adapter
        elif hasattr(ctx.cfg, 'llm') and hasattr(ctx.cfg.llm, 'adapter'):
            adapter_cfg = ctx.cfg.llm.adapter

        if adapter_cfg and getattr(adapter_cfg, 'mv_to_cpu', False):
            for p in ctx.model.parameters():
                if p.requires_grad:
                    p.data = p.to('cpu')
                    if p.grad is not None:
                        p.grad.data = p.grad.to('cpu')
