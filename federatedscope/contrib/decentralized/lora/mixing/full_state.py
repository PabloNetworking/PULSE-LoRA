"""Complete-state mixing from pre-round snapshots under a mixing matrix row."""
from collections import OrderedDict
from typing import Mapping

import torch


class MixError(ValueError):
    pass


def weighted_mix(weights: Mapping[int, float],
                 snapshots: Mapping[int, Mapping[str, torch.Tensor]]
                 ) -> "OrderedDict[str, torch.Tensor]":
    """``sum_j W[i][j] * x_j`` over the peers with a nonzero weight."""
    if not weights:
        raise MixError("no weights supplied")
    missing = [peer for peer in weights if peer not in snapshots]
    if missing:
        raise MixError(f"missing pre-round snapshots for peers {missing}")

    reference = snapshots[next(iter(weights))]
    keys = frozenset(reference)
    for peer, snapshot in snapshots.items():
        if peer in weights and frozenset(snapshot) != keys:
            raise MixError(f"peer {peer} snapshot has a different key set")

    out = OrderedDict()
    for key in sorted(keys):
        total = None
        for peer, weight in sorted(weights.items()):
            if weight == 0.0:
                continue
            term = snapshots[peer][key] * weight
            total = term if total is None else total + term
        out[key] = total
    return out
