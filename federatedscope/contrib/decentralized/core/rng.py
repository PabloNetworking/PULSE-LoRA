"""Isolated, named, checkpointable random streams.

Every stream is seeded from the run's master seed and the stream's name. The
process-global ``random`` / ``numpy.random`` generators are never used or reseeded
here.

Streams are namespaced so that changing one decision cannot perturb another - a
selector change must not alter data order or the opportunity trace.
"""
import hashlib
import random
from typing import Dict, Iterable

import numpy as np

#: Names of the runtime's own streams. The runner hands each peer's method the stream
#: ``method:<method>:peer<id>``, so adding a method or a peer cannot perturb another
#: stream's replay.
STREAM_TOPOLOGY = "topology"
STREAM_OPPORTUNITIES = "opportunities"
STREAM_PARTITION = "partition"
STREAM_LOCAL_COMPUTE = "local_compute"
STREAM_FAILURE_INJECTION = "failure_injection"
STREAM_EVALUATION = "evaluation"


def derive_seed(master_seed: int, name: str) -> int:
    """Deterministically derive a 64-bit stream seed from a master seed + name."""
    digest = hashlib.sha256(f"{int(master_seed)}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class RngService:
    """Owns every named stream for one run."""

    def __init__(self, master_seed: int):
        self.master_seed = int(master_seed)
        self._python: Dict[str, random.Random] = {}
        self._numpy: Dict[str, np.random.Generator] = {}

    def stream(self, name: str) -> random.Random:
        if name not in self._python:
            self._python[name] = random.Random(derive_seed(self.master_seed, name))
        return self._python[name]

    def numpy_stream(self, name: str) -> np.random.Generator:
        if name not in self._numpy:
            self._numpy[name] = np.random.default_rng(
                derive_seed(self.master_seed, f"numpy:{name}")
            )
        return self._numpy[name]

    def names(self) -> Iterable[str]:
        return sorted(set(self._python) | set(self._numpy))

    # -- checkpointing ----------------------------------------------------
    # State objects are pickle-shaped (nested tuples / dicts), so persist them
    # with torch.save or pickle, not JSON.

    def state_dict(self) -> dict:
        return {
            "master_seed": self.master_seed,
            "python": {k: v.getstate() for k, v in self._python.items()},
            "numpy": {k: v.bit_generator.state for k, v in self._numpy.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore every stream in place, keeping the generator objects.

        Consumers hold their stream by reference: ``DecentralizedRunner`` hands a
        peer ``rng.stream(f"method:{method}:peer{id}")`` when it builds the peer,
        and it builds peers before it loads a checkpoint. So the generators are
        mutated, never replaced:

        * a stream in the checkpoint that this service has not created yet is
          created and restored;
        * a stream this service holds that the checkpoint does not mention is
          reset to its initial state from the master seed, so the resumed run does
          not depend on draws made before the checkpoint was loaded.
        """
        master_seed = int(state["master_seed"])
        python_state = dict(state.get("python", {}) or {})
        numpy_state = dict(state.get("numpy", {}) or {})

        # Validate everything before mutating anything: a half-restored RNG is
        # worse than a refused one, because the run continues and diverges.
        probe = random.Random()
        for name, py_state in python_state.items():
            probe.setstate(py_state)
        probe_np = np.random.default_rng()
        for name, np_state in numpy_state.items():
            probe_np.bit_generator.state = np_state

        self.master_seed = master_seed

        for name, py_state in python_state.items():
            self.stream(name).setstate(py_state)          # creates then restores
        for name in set(self._python) - set(python_state):
            self._python[name].seed(derive_seed(self.master_seed, name))

        for name, np_state in numpy_state.items():
            self.numpy_stream(name).bit_generator.state = np_state
        for name in set(self._numpy) - set(numpy_state):
            fresh = np.random.default_rng(
                derive_seed(self.master_seed, f"numpy:{name}"))
            self._numpy[name].bit_generator.state = fresh.bit_generator.state
