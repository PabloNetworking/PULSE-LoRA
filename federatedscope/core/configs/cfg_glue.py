"""
Configuration for GLUE benchmark tasks.

This module extends the configuration with GLUE-specific options for
sequence classification and regression tasks (SST-2, QNLI, MRPC, STS-B, RTE, etc.)
with DeBERTa and other transformer models, and with their LoRA adapter.
"""
from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_glue_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # GLUE related options
    # ---------------------------------------------------------------------- #
    cfg.glue = CN()
    cfg.glue.max_length = 128  # Maximum sequence length for tokenization
    # A batch whose loss is NaN is skipped; with this option one more batch
    # is trained in its place.
    cfg.glue.retry_on_nan_loss = True
    cfg.glue.task_name = ''  # Populated automatically from data.type during dataset loading
    cfg.glue.mnli_eval_split = 'matched'  # 'matched' or 'mismatched'

    # HuggingFace cache directory for the model weights
    cfg.glue.cache_dir = ''

    # ---------------------------------------------------------------------- #
    # Adapter for GLUE models (same structure as llm.adapter)
    # ---------------------------------------------------------------------- #
    cfg.glue.adapter = CN()
    cfg.glue.adapter.use = False
    # Adapter arguments, e.g. [{'adapter_package': 'peft',
    # 'adapter_method': 'lora', 'r': 8, 'lora_alpha': 16, ...}]
    cfg.glue.adapter.args = [{}]
    # Move the trainable parameters to `cpu` after each routine
    cfg.glue.adapter.mv_to_cpu = False

    # LoRA rank of the adapter; it replaces `r` in `args`
    cfg.glue.adapter.max_rank = 64

    # Rank assignment. Only 'homo' exists: every peer builds its adapter from
    # `args` and `max_rank`. No code reads this value; PULSE-LoRA sets per-peer
    # ranks through decentralized.methods.pulse_lora.initial_rank.
    cfg.glue.adapter.hetero_strategy = 'homo'


register_config('glue', extend_glue_cfg)
