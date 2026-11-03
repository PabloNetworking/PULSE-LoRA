"""
GLUE support: dataset loading and LoRA sequence classification models.

This module provides support for GLUE benchmark datasets (CoLA, SST-2, MRPC,
STS-B, QQP, MNLI, QNLI, RTE).
WNLI is not supported.
"""

from federatedscope.glue.dataloader import load_glue_dataset, GLUEDataCollator
from federatedscope.glue.model import get_glue_model

__all__ = [
    'load_glue_dataset',
    'GLUEDataCollator',
    'get_glue_model',
]
