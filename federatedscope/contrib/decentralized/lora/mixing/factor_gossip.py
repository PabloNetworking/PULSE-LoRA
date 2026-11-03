"""Symmetric bilateral factor gossip.

``x_i' = (1-gamma) x_i + gamma x_j`` on both sides, which is mean-preserving for
each mixed block and contracts disagreement for ``0 < gamma <= 0.5``.
"""
from collections import OrderedDict
from typing import Mapping

import torch


class GossipError(ValueError):
    pass


def bilateral_gossip(local: Mapping[str, torch.Tensor],
                     remote: Mapping[str, torch.Tensor],
                     gamma: float) -> "OrderedDict[str, torch.Tensor]":
    if not 0.0 < gamma <= 0.5:
        raise GossipError(f"gamma must satisfy 0 < gamma <= 0.5, got {gamma}")
    if frozenset(local) != frozenset(remote):
        raise GossipError(
            f"gossip needs identical key sets; local-only="
            f"{sorted(frozenset(local) - frozenset(remote))}, remote-only="
            f"{sorted(frozenset(remote) - frozenset(local))}"
        )

    out = OrderedDict()
    for key in sorted(local):
        mine, theirs = local[key], remote[key]
        if tuple(mine.shape) != tuple(theirs.shape):
            raise GossipError(f"{key}: shape mismatch {tuple(mine.shape)} vs {tuple(theirs.shape)}")
        out[key] = (1.0 - gamma) * mine + gamma * theirs.to(mine.dtype)
    return out
