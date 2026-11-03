"""Toy multi-layer LoRA fixtures that require no model download.

These exist so schema/bridge/mixing tests exercise the *real* peft key format and
tensor layout rather than a hand-rolled mock, while still running on CPU in
milliseconds.

The canonical contract they pin down (peft 0.4.0)::

    base_model.model.<module>.lora_A.default.weight   (r, in_features)
    base_model.model.<module>.lora_B.default.weight   (out_features, r)

so a flat canonical component ``p`` is row ``p`` of every ``A`` and column ``p``
of every ``B``.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from federatedscope.contrib.decentralized.lora.canonical_init import (
    EXTRAS_KEY as CONSTRUCTION_KEY,
    CommonConstruction,
)

#: Modules the toy backbone exposes for LoRA injection, in canonical order.
TOY_TARGET_MODULES = ("lin_a", "lin_b", "lin_c")

DEFAULT_RANK = 4
DEFAULT_ALPHA = 16
DEFAULT_IN_FEATURES = 8


class ToyBackbone(nn.Module):
    """Three stacked linear layers - enough to make 'flat across layers' meaningful."""

    def __init__(self, in_features=DEFAULT_IN_FEATURES, hidden=DEFAULT_IN_FEATURES):
        super().__init__()
        self.lin_a = nn.Linear(in_features, hidden, bias=False)
        self.lin_b = nn.Linear(hidden, hidden, bias=False)
        self.lin_c = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.lin_c(self.lin_b(self.lin_a(x)))


def build_toy_lora_model(rank=DEFAULT_RANK, lora_alpha=DEFAULT_ALPHA, seed=0,
                         target_modules=TOY_TARGET_MODULES, lora_dropout=0.0):
    """Build a peft-wrapped toy adapter deterministically.

    Two calls with the same ``seed`` produce bitwise-identical parameters, which is
    what peer-consensus tests need as a starting condition.
    """
    torch.manual_seed(seed)
    backbone = ToyBackbone()
    config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules),
        bias="none",
    )
    return get_peft_model(backbone, config)


def trainable_lora_state(model):
    """Trainable parameters in canonical (sorted-key) order."""
    return OrderedDict(
        (name, param)
        for name, param in sorted(model.named_parameters())
        if param.requires_grad
    )


def lora_scalings(model):
    """``{module_name: scaling}`` for every injected module."""
    return {
        name.split(".")[-1]: dict(module.scaling)["default"]
        for name, module in model.named_modules()
        if hasattr(module, "scaling") and "default" in getattr(module, "scaling", {})
    }


def toy_construction(seed=0):
    """The common construction for a toy adapter, which has no backbone directory: the
    canonical values of the adapter's own modules under ``seed``."""
    return CommonConstruction(model_init_seed=int(seed), backbone_digest="toy-backbone",
                              backbone_path="")


def toy_extras(model, seed=0, **extras):
    """``PeerContext.extras`` for a PULSE-LoRA peer on a toy adapter."""
    return dict(extras, model=model, **{CONSTRUCTION_KEY: toy_construction(seed)})


def toy_component_schema(lora_schema, r_max, seed, scale=None):
    """A PULSE-LoRA component space over a toy adapter, initialised by :func:`toy_construction`."""
    from federatedscope.contrib.decentralized.methods.pulse_lora.components import (
        build_component_schema,
    )

    return build_component_schema(lora_schema, r_max=r_max, seed=seed, scale=scale,
                                  init_rule=toy_construction().bind(lora_schema))
