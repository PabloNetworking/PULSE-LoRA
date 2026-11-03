"""Checkpoint orchestration.

The runner persists its own scheduling state and delegates method state through
the plugin interface: it never reaches into a peer's tensors itself.
"""
import os
from typing import Any, Dict


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    import torch

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    import torch

    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)
