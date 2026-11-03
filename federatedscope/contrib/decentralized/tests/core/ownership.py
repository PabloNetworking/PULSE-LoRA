"""Recursive reachability scan used by the ownership-boundary tests.

Walks containers, ``__dict__``, ``__slots__``, the ``__self__`` of bound methods
and the ``__closure__`` cells of stored callables, because a callback closing
over a peer reaches tensors as effectively as an attribute does.
"""
import inspect


def reachable(root, max_nodes=20000):
    """Yield the objects reachable from ``root``, up to ``max_nodes``.

    Modules and classes are yielded but not entered.
    """
    seen = set()
    stack = [root]
    yielded = 0
    while stack and yielded < max_nodes:
        obj = stack.pop()
        marker = id(obj)
        if marker in seen:
            continue
        seen.add(marker)
        yielded += 1
        yield obj

        if inspect.ismodule(obj) or inspect.isclass(obj):
            continue
        if isinstance(obj, (str, bytes, bytearray, int, float, bool, type(None))):
            continue

        if isinstance(obj, dict):
            stack.extend(obj.keys())
            stack.extend(obj.values())
            continue
        if isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
            continue

        # closures of any stored callable
        closure = getattr(obj, "__closure__", None)
        if closure:
            for cell in closure:
                try:
                    stack.append(cell.cell_contents)
                except ValueError:
                    pass
        bound = getattr(obj, "__self__", None)
        if bound is not None:
            stack.append(bound)

        state = getattr(obj, "__dict__", None)
        if isinstance(state, dict):
            stack.extend(state.values())
        slots = getattr(type(obj), "__slots__", ())
        for slot in slots or ():
            if hasattr(obj, slot):
                stack.append(getattr(obj, slot))


def find_learning_state(root):
    """Names of forbidden object types reachable from ``root``."""
    import torch

    from federatedscope.contrib.decentralized.core.peer import Peer

    offenders = []
    for obj in reachable(root):
        if isinstance(obj, torch.Tensor):
            offenders.append(f"torch.Tensor{tuple(obj.shape)}")
        elif isinstance(obj, torch.nn.Module):
            offenders.append(f"nn.Module:{type(obj).__name__}")
        elif isinstance(obj, torch.optim.Optimizer):
            offenders.append(f"Optimizer:{type(obj).__name__}")
        elif isinstance(obj, Peer):
            offenders.append(f"Peer:{obj.peer_id}")
    return offenders
