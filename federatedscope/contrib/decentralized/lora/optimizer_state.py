"""Optimizer-state helpers.

Clear, capture and restore Adam moment buffers, for whole parameters or for
single rows and columns. When to call them is the method's policy.
"""
from typing import Iterable, List, Tuple

import torch

_MOMENT_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")


def clear_moments(optimizer: torch.optim.Optimizer,
                  parameters: Iterable[torch.nn.Parameter]) -> int:
    """Zero the Adam moment buffers for exactly these parameters.

    Returns how many parameters had a buffer zeroed.
    """
    touched = 0
    targets = {id(p) for p in parameters}
    for param, state in optimizer.state.items():
        if id(param) not in targets:
            continue
        cleared = False
        for key in _MOMENT_KEYS:
            if key in state and torch.is_tensor(state[key]):
                state[key].zero_()
                cleared = True
        if cleared:
            touched += 1
    return touched


def clear_moment_slices(optimizer: torch.optim.Optimizer,
                        slices: Iterable[Tuple[torch.nn.Parameter, int, int]]) -> int:
    """Zero one index of one dimension of a moment buffer, leaving the rest intact.

    ``slices`` yields ``(parameter, dim, index)``: for example a LoRA A row and
    its paired B column. Clearing the whole buffer would also reset the other
    r - 1 components of a rank-r adapter that share the tensor. Returns how many
    buffer slices were zeroed.
    """
    touched = 0
    for param, dim, index in slices:
        state = optimizer.state.get(param)
        if not state:
            continue
        for key in _MOMENT_KEYS:
            buffer = state.get(key)
            if torch.is_tensor(buffer) and buffer.dim() > dim:
                buffer.select(dim, index).zero_()
                touched += 1
    return touched


def read_moment_slices(optimizer: torch.optim.Optimizer,
                       slices: Iterable[Tuple[torch.nn.Parameter, int, int]]) -> List[tuple]:
    """Capture exactly the slices ``clear_moment_slices`` would zero.

    A method that resets moments inside a transaction restores them from these
    records on rollback; restoring a whole buffer would overwrite the other
    components that share the tensor.

    Returns ``(parameter, dim, index, {buffer name: values})`` records; the
    parameter is held by reference.
    """
    captured = []
    for param, dim, index in slices:
        state = optimizer.state.get(param)
        if not state:
            continue
        values = {}
        for key in _MOMENT_KEYS:
            buffer = state.get(key)
            if torch.is_tensor(buffer) and buffer.dim() > dim:
                values[key] = buffer.select(dim, index).detach().clone()
        if values:
            captured.append((param, dim, index, values))
    return captured


def write_moment_slices(optimizer: torch.optim.Optimizer,
                        captured: Iterable[tuple]) -> int:
    """Put back what :func:`read_moment_slices` captured, slice by slice."""
    touched = 0
    for param, dim, index, values in captured:
        state = optimizer.state.get(param)
        if not state:
            continue
        for key, value in values.items():
            buffer = state.get(key)
            if torch.is_tensor(buffer) and buffer.dim() > dim:
                buffer.select(dim, index).copy_(value)
                touched += 1
    return touched


def moment_norms(optimizer: torch.optim.Optimizer) -> List[float]:
    """L2 norms of every first-moment buffer - a cheap fingerprint for tests."""
    norms = []
    for _, state in optimizer.state.items():
        if "exp_avg" in state and torch.is_tensor(state["exp_avg"]):
            norms.append(float(state["exp_avg"].norm()))
    return norms


def has_state(optimizer: torch.optim.Optimizer) -> bool:
    return any(bool(state) for state in optimizer.state.values())
