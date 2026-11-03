"""Local-support controllers: which components a peer keeps active.

Two controllers, both local to the peer:

``fixed``   the configured support (``fixed_support``, or the first
            ``initial_rank`` components) never changes. Holding supports constant
            separates the effect of the exchange from that of pruning.
``adas_c``  AdaS-LoRA-C's conservative indexed rank-1 pruning, applied to the
            flat canonical components, with scores summed over injected layers.

A component is pruned only if it was in the low set and its score decreased
over the episode; pruning on the low set alone would discard components that
are small but growing. AdaS-LoRA-C's norm-product regularizer is not
implemented: ``adas_c`` is the pruning rule only.
"""
from typing import Dict, Mapping, Tuple

import torch

CONTROLLER_FIXED = "fixed"
CONTROLLER_ADAS_C = "adas_c"
CONTROLLERS = (CONTROLLER_FIXED, CONTROLLER_ADAS_C)


class PruningError(ValueError):
    pass


def component_scores(state) -> Dict[int, float]:
    """``score[p] = sum over layers of ||A_row_p||_2 * ||B_col_p||_2``.

    Summed over injected layers because a canonical component spans all of them:
    scoring per layer would be scoring a different object.
    """
    scores: Dict[int, float] = {}
    bridge = state.bridge
    for component in state.support:
        position = state.compact_index(component)
        total = 0.0
        for spec in bridge.schema.modules:
            a = bridge._params[spec.a_key]
            b = bridge._params[spec.b_key]
            total += float(torch.norm(a[position, :], p=2)) * \
                float(torch.norm(b[:, position], p=2))
        scores[component] = total
    return scores


def low_set(scores: Mapping[int, float], gamma: float, rank_min: int) -> Tuple[int, ...]:
    """The ``m`` lowest-scoring components, ``m = r - max(rank_min, floor(r*gamma))``.

    Returns them in ascending score order, ties broken by component id so the
    choice is deterministic.
    """
    if not 0.0 < gamma <= 1.0:
        raise PruningError(f"gamma must satisfy 0 < gamma <= 1, got {gamma}")
    rank = len(scores)
    if rank == 0:
        return ()
    keep_target = max(int(rank_min), int(rank * gamma))
    candidates = rank - keep_target
    if candidates <= 0:
        return ()
    ordered = sorted(scores, key=lambda c: (scores[c], c))
    return tuple(ordered[:candidates])


class FixedSupportController:
    """Never prunes. The support is whatever it was assigned."""

    name = CONTROLLER_FIXED

    def __init__(self, **_kwargs):
        pass

    def before_episode(self, state) -> None:
        return None

    def select_for_pruning(self, state) -> Tuple[int, ...]:
        return ()

    def metrics(self) -> dict:
        return {"pulse_controller": self.name}


class AdaSCController:
    """Conservative indexed rank-1 pruning over flat canonical components."""

    name = CONTROLLER_ADAS_C

    def __init__(self, gamma: float = 0.99, rank_min: int = 1, **_kwargs):
        if rank_min < 1:
            raise PruningError(f"rank_min must be at least 1, got {rank_min}")
        self.gamma = float(gamma)
        self.rank_min = int(rank_min)
        self._scores_before: Dict[int, float] = {}
        self._low_set: Tuple[int, ...] = ()
        self.last_decision: dict = {}

    def before_episode(self, state) -> None:
        """Record the low set and its scores *before* local training."""
        scores = component_scores(state)
        self._low_set = low_set(scores, self.gamma, self.rank_min)
        self._scores_before = {c: scores[c] for c in self._low_set}

    def select_for_pruning(self, state) -> Tuple[int, ...]:
        """Prune only low-set components whose score decreased during the episode."""
        if not self._low_set:
            self.last_decision = {"candidates": 0, "decreased": 0, "pruned": []}
            return ()

        after = component_scores(state)
        decreased = [c for c in self._low_set
                     if c in after and after[c] < self._scores_before.get(c, 0.0)]

        # Never go below rank_min (at least 1), so the support is never emptied.
        headroom = max(0, len(state.support) - self.rank_min)
        pruned = tuple(sorted(decreased)[:headroom])

        self.last_decision = {
            "candidates": len(self._low_set),
            "decreased": len(decreased),
            "pruned": list(pruned),
            "truncated_by_rank_min": len(decreased) > len(pruned),
        }
        return pruned

    def metrics(self) -> dict:
        return {"pulse_controller": self.name,
                "pulse_gamma": self.gamma,
                "pulse_rank_min": self.rank_min,
                "pulse_last_pruned": len(self.last_decision.get("pruned", []))}


def build_controller(name: str, **kwargs):
    if name == CONTROLLER_FIXED:
        return FixedSupportController(**kwargs)
    if name == CONTROLLER_ADAS_C:
        return AdaSCController(**kwargs)
    raise PruningError(f"unknown controller {name!r}; expected one of {list(CONTROLLERS)}")
