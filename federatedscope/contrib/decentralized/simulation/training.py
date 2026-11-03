"""Per-episode training seeds, and a peer's data from the scenario's data builder.

Under ``training_rng = "per_peer_episode"``, torch is reseeded immediately before every local
episode with :func:`episode_seed` of (scenario seed, peer id, episode index), where the index is
the peer trainer's own ``local_episodes`` counter. Batch order and dropout masks then depend on
these three values alone: not on what another peer drew, and not on anything evaluated between
episodes. The driver relies on this to compute readouts inside the run without changing training.

Only torch is reseeded. Training and evaluation draw nothing from Python's ``random`` or numpy's
global generator.
"""
import hashlib
import importlib
from typing import Any, Optional

#: The per-episode reseeding rule. ``None`` instead leaves torch's process-global generator
#: as the runner uses it.
PER_PEER_EPISODE = "per_peer_episode"


TRAINING_RNG_POLICIES = (None, PER_PEER_EPISODE)


def episode_seed(seed: int, peer_id: int, episode: int) -> int:
    """The torch seed of one peer's one episode: a labelled hash, 63 bits.

    The label keeps it apart from every other seed derived from the scenario seed, and the
    hash keeps neighbouring peers and episodes from getting neighbouring seeds. The label is a
    fixed namespace string: changing it changes the training of every run.
    """
    label = f"fedlora/phase9/training/{int(seed)}/peer{int(peer_id)}/episode{int(episode)}"
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big") >> 1


def install_training_rng(algorithm, scenario, peer_id: int, policy: Any = "scenario") -> Optional[str]:
    """Apply a training RNG rule to one peer's algorithm, on the instance.

    ``policy`` defaults to the scenario's own; a test may name another. Returns the policy
    applied.
    """
    policy = scenario.training_rng if policy == "scenario" else policy
    if policy not in TRAINING_RNG_POLICIES:
        raise ValueError(f"unknown training RNG policy {policy!r}; "
                         f"expected one of {list(TRAINING_RNG_POLICIES)}")
    if policy is None:
        return None
    if getattr(algorithm, "trainer", None) is None:
        raise ValueError(f"peer {peer_id}: the {policy!r} rule keys episodes by the "
                         f"trainer's own counter, and this peer has no trainer")
    seed, peer = int(scenario.seed), int(peer_id)
    original = algorithm.local_episode

    def local_episode(steps):
        import torch

        episode = int(original.__self__.trainer.local_episodes)
        torch.manual_seed(episode_seed(seed, peer, episode))
        return dict(original(steps) or {})

    algorithm.local_episode = local_episode
    return policy


def plain(value):
    """``value`` as plain JSON data, exactly.

    Torch and numpy scalars become Python numbers, tuples lists, mapping keys strings; a
    float stays a float, whose ``repr`` JSON round-trips exactly. Anything else is refused
    rather than turned into a string.
    """
    import numpy
    import torch

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, (numpy.generic, torch.Tensor)):
        if isinstance(value, torch.Tensor) and value.numel() != 1:
            raise TypeError(f"a tensor of {value.numel()} elements is not a scalar metric")
        return plain(value.item())
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    raise TypeError(f"{type(value).__name__} has no exact plain form")


def resolve(dotted: str):
    """The attribute a dotted ``module.attribute`` path names."""
    module_name, _, attribute = str(dotted).rpartition(".")
    return getattr(importlib.import_module(module_name), attribute)


def build_data(scenario, peer_id: int, cfg) -> Any:
    """This peer's data, from the scenario's builder; None for a dataset-free run.

    The builder is called as ``builder(scenario, peer_id, cfg)``. It must return fresh objects
    on every call and must not depend on the global RNG state.
    """
    if not scenario.data:
        return None
    data = _built_data(scenario, peer_id, cfg)
    # A stated readout holdout the builder did not subtract is refused by name (readout.py):
    # otherwise the run would train on the examples its readouts are scored on.
    schedule = getattr(scenario, "readout_schedule", None)
    if schedule is not None:
        from federatedscope.contrib.decentralized.simulation.readout import require_applied_holdout

        require_applied_holdout(data, schedule, int(peer_id), scenario.data)
    return data


def _built_data(scenario, peer_id: int, cfg) -> Any:
    if getattr(scenario, "data_partition", None) is not None:
        return _partitioned_data(scenario, peer_id, cfg)
    return resolve(scenario.data)(scenario, int(peer_id), cfg)


def _partitioned_data(scenario, peer_id: int, cfg) -> Any:
    """This peer's data under the scenario's ``data_partition``, from the builder.

    The builder must apply the stated partition and declare it (``applied_partition``); a
    builder that applied none, or a different one, is refused.
    """
    spec = scenario.data_partition
    data = resolve(scenario.data)(scenario, int(peer_id), cfg)
    applied = getattr(data, "applied_partition", None)
    if applied != spec:
        raise ValueError(
            f"peer {peer_id}: the scenario states data_partition {spec!r} and the builder "
            f"{scenario.data!r} returned a partition that declares "
            f"{'no partition' if applied is None else repr(applied)}; a builder must apply "
            f"the stated specification (federatedscope.contrib.decentralized.simulation.partition."
            f"apply_partition_spec) and say so, or the scenario must not state one")
    return data
