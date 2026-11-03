"""GLUE data for a scenario: FederatedScope's ``get_data``, the scenario's partition and the
readout holdout.

``glue_partition`` is the scenario data builder for a GLUE task (``Scenario.data`` names it).
It builds one peer's partition. ``central_glue_data`` builds every peer's partition from a
single ``get_data`` call and gives the same index sets; the driver uses it.

The order is fixed: FederatedScope's IID split over ``federate.client_num`` peers, then the
scenario's ``data_partition`` if it states one, then the readout holdout, which is removed
from every peer's training split.

``get_data`` freezes the configuration it is given, and freezing writes ``config.yaml`` into
``cfg.outdir``, so it is called on a copy with a temporary output directory. It also writes
HuggingFace dataset lock files under ``cfg.data.root``.
"""
import tempfile
from typing import Any, Dict

from federatedscope.contrib.decentralized.simulation.partition import apply_partition_spec
from federatedscope.contrib.decentralized.simulation.readout import (
    apply_holdout,
    require_applied_holdout,
)

#: The dotted path a scenario names this builder by.
GLUE_BUILDER = "federatedscope.contrib.decentralized.simulation.glue_data.glue_partition"

#: What ``get_data`` sets on the configuration it is given. The scenario must already state each
#: of them, identically, because the peers train with the scenario's configuration.
TASK_KEYS = ("glue.task_name", "model.out_channels", "model.task", "criterion.type")


def _lookup(cfg, dotted):
    for part in dotted.split("."):
        cfg = getattr(cfg, part)
    return cfg


def disagreeing_task_keys(stated, modified):
    """The ``TASK_KEYS`` on which two configurations differ: ``{key: (stated, modified)}``."""
    return {key: (_lookup(stated, key), _lookup(modified, key)) for key in TASK_KEYS
            if _lookup(stated, key) != _lookup(modified, key)}


def _require_client_num(scenario, cfg):
    if int(cfg.federate.client_num) != len(scenario.peers):
        raise ValueError(f"federate.client_num is {cfg.federate.client_num} but the scenario "
                         f"has {len(scenario.peers)} peers: the split would be a different one")


def get_glue_data(cfg):
    """FederatedScope's ``get_data`` on a copy of ``cfg`` with a temporary output directory.

    Refused if ``get_data`` changes any of the ``TASK_KEYS``."""
    from federatedscope.core.auxiliaries.data_builder import get_data

    with tempfile.TemporaryDirectory(prefix="glue-data-") as outdir:
        given = cfg.clone()
        given.merge_from_list(["outdir", outdir])
        data, modified = get_data(config=given)
    disagree = disagreeing_task_keys(cfg, modified)
    if disagree:
        raise ValueError(f"get_data sets task keys the scenario does not state "
                         f"(scenario, get_data): {disagree}")
    return data


def _holdout_spec(scenario):
    schedule = getattr(scenario, "readout_schedule", None)
    return None if schedule is None else schedule["holdout"]


def glue_partition(scenario, peer_id, cfg):
    """Peer ``peer_id``'s GLUE partition: the IID split, the stated partition, the holdout."""
    _require_client_num(scenario, cfg)
    partition = get_glue_data(cfg)[int(peer_id)]
    if scenario.data_partition is not None:
        partition = apply_partition_spec(partition, scenario.data_partition, peer_id,
                                         scenario.peers, holdout=_holdout_spec(scenario))
    if getattr(scenario, "readout_schedule", None) is not None:
        partition = apply_holdout(partition, scenario.readout_schedule["holdout"])
    return partition


def _full_training_split(partition):
    """The full training dataset a partition's training ``Subset`` indexes into."""
    from torch.utils.data import Subset

    loader = partition.get("train") if hasattr(partition, "get") else None
    subset = getattr(loader, "dataset", None)
    if not isinstance(subset, Subset) or getattr(subset, "dataset", None) is None:
        raise ValueError(f"the training split is not a Subset of a dataset "
                         f"({type(subset).__name__})")
    return subset.dataset


def central_glue_data(scenario, cfg) -> Dict[int, Any]:
    """Every peer's ``glue_partition`` from one ``get_data`` call."""
    _require_client_num(scenario, cfg)
    peers = tuple(sorted(int(p) for p in scenario.peers))
    schedule = getattr(scenario, "readout_schedule", None)
    raw = get_glue_data(cfg)
    if len({id(_full_training_split(raw[peer])) for peer in peers}) != 1:
        raise ValueError("the peers' training Subsets index different full splits")
    data: Dict[int, Any] = {}
    for peer in peers:
        partition = raw[peer]
        if scenario.data_partition is not None:
            partition = apply_partition_spec(partition, scenario.data_partition, peer, peers,
                                             holdout=_holdout_spec(scenario))
        if schedule is not None:
            partition = apply_holdout(partition, schedule["holdout"])
            require_applied_holdout(partition, schedule, peer, scenario.data)
        if scenario.data_partition is not None and \
                getattr(partition, "applied_partition", None) != scenario.data_partition:
            raise ValueError(f"peer {peer}: the stated data_partition was not applied")
        data[peer] = partition
    return data
