"""How a scenario's dataset is split between peers (``Scenario.data_partition``).

Without a ``data_partition`` the peers get FederatedScope's IID split. Two specifications
re-split the training data:

  - ``{"splitter": "lda", "alpha": a, "seed": s}``: a Dirichlet (LDA) label partition drawn by
    FederatedScope's ``LDASplitter``. Its slice draws on numpy's global generator; here that
    generator is seeded with ``s`` for the split and restored afterwards, and every split the
    partition holds is drawn from that one stream in the order train, val, test, as
    ``BaseDataTranslator.split_to_client`` does. No label prior is passed between splits.
    Labels must be exactly 0..C-1: the slice would silently drop every example of a class
    beyond a gap.
  - ``{"splitter": "label_stratified_round_robin_v1", "seed": s}``: a balanced deal of the
    training examples left after the scenario's label-stratified readout holdout. Each label in
    numeric order, its examples sorted by the sha256 of
    ``{STRATIFIED_KEY}|seed={s}|label={c}|example={idx}`` (ties by idx), dealt round-robin to
    the peers in ascending id with the offset carried across labels. Validation and test splits
    are left as FederatedScope shards them.

A partition built under a specification records it as the ``applied_partition`` attribute of
its ``ClientData``. The index sets are a function of the labels, the specification and the
peers only, so every method of a comparison trains on identical partitions.
"""
import contextlib
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

LDA = "lda"
STRATIFIED = "label_stratified_round_robin_v1"
STRATIFIED_KEY = "pulse-h4-iid-v1"
SPLITTERS = (LDA, STRATIFIED)
SPEC_KEYS = ("splitter", "alpha", "seed")
STRATIFIED_SPEC_KEYS = ("splitter", "seed")
#: The order ``split_to_client`` draws the splits in, from one seeded stream.
SPLIT_ORDER = ("train", "val", "test")


class PartitionError(ValueError):
    """A partition specification or materialisation that cannot be honoured."""


def validate_partition_spec(spec: Any) -> Dict[str, Any]:
    """The specification in canonical form, or a refusal naming what is wrong.

    ``{"splitter": "lda", "alpha": <float > 0>, "seed": <int in [0, 2**32)>}`` or
    ``{"splitter": "label_stratified_round_robin_v1", "seed": <int in [0, 2**32)>}``, and
    nothing else: an unknown key is refused rather than ignored. The seed is stated, never
    inherited from the run seed, so runs with different seeds can share a partition.
    """
    if type(spec) is not dict:
        raise PartitionError(
            f"data_partition must be a plain mapping with keys {list(SPEC_KEYS)}, "
            f"not {type(spec).__name__}")
    if spec.get("splitter") == STRATIFIED:
        return _validate_stratified_spec(spec)
    unknown = sorted(set(spec) - set(SPEC_KEYS))
    missing = [key for key in SPEC_KEYS if key not in spec]
    if unknown or missing:
        raise PartitionError(
            f"data_partition must have exactly the keys {list(SPEC_KEYS)}; "
            f"unknown {unknown}, missing {missing}")
    splitter = spec["splitter"]
    if splitter not in SPLITTERS:
        raise PartitionError(
            f"data_partition.splitter {splitter!r} is not one of {list(SPLITTERS)}; IID is "
            f"the absence of a data_partition, not a value of it")
    alpha = spec["alpha"]
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise PartitionError(
            f"data_partition.alpha must be a finite number > 0, not {alpha!r}")
    seed = spec["seed"]
    if type(seed) is not int or seed < 0 or seed >= 2 ** 32:
        raise PartitionError(
            f"data_partition.seed must be an integer in [0, 2**32), not {seed!r}")
    return {"splitter": str(splitter), "alpha": float(alpha), "seed": int(seed)}


def _validate_stratified_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    unknown = sorted(set(spec) - set(STRATIFIED_SPEC_KEYS))
    missing = [key for key in STRATIFIED_SPEC_KEYS if key not in spec]
    if unknown or missing:
        raise PartitionError(
            f"a {STRATIFIED} data_partition must have exactly the keys "
            f"{list(STRATIFIED_SPEC_KEYS)}; unknown {unknown}, missing {missing}")
    seed = spec["seed"]
    if type(seed) is not int or seed < 0 or seed >= 2 ** 32:
        raise PartitionError(
            f"data_partition.seed must be an integer in [0, 2**32), not {seed!r}")
    return {"splitter": STRATIFIED, "seed": int(seed)}


@contextlib.contextmanager
def seeded_numpy(seed: int):
    """The global numpy generator seeded for the block, and restored after it.

    FederatedScope's Dirichlet slice draws on ``np.random`` and takes no generator, so it is
    seeded around the call. The caller's numpy state is left as it was.
    """
    import numpy as np

    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(state)


def labels_of(dataset) -> "Any":
    """Every item's class label, in dataset order, read as ``LDASplitter`` reads it.

    Tuples ``(x, y)``; dicts by ``categories``, then ``label``, then ``labels`` (GLUE's
    scalar tensor). Mirrors ``LDASplitter.__call__`` so that labels with a gap can be
    refused before the splitter is called; the split itself is the splitter's.
    """
    import numpy as np

    items = [item for item in dataset]
    if not items:
        raise PartitionError("the dataset is empty; there is nothing to partition")
    first = items[0]
    if isinstance(first, tuple):
        raw = [y for _, y in items]
    elif isinstance(first, dict):
        key = next((k for k in ("categories", "label", "labels") if k in first), None)
        if key is None:
            raise PartitionError("cannot find 'label', 'labels' or 'categories' in the "
                                 "dataset's items; the Dirichlet split is over labels")
        raw = [item[key] for item in items]
    else:
        raise PartitionError(f"unsupported item type {type(first).__name__}; expected a "
                             f"(x, y) tuple or a mapping carrying a label")
    return np.array([int(y) for y in raw])


def _checked_labels(labels, peers: int):
    import numpy as np

    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise PartitionError("only single-label tasks can be Dirichlet-split")
    classes = np.unique(labels)
    if len(classes) == 0 or not np.array_equal(classes, np.arange(len(classes))):
        raise PartitionError(
            f"labels must be exactly 0..C-1 for the Dirichlet split; found "
            f"{classes.tolist()}. The Dirichlet slice iterates range(C) and would silently "
            f"drop every item of a class beyond a gap, so this is refused before any draw")
    if len(labels) <= int(peers):
        raise PartitionError(
            f"{len(labels)} items cannot be split into {int(peers)} non-empty peers; the "
            f"Dirichlet slice requires more items than peers")
    return labels


class _Labelled:
    """A dataset of ``{"labels": y}`` items over a label vector: the shape in which the
    splitter reads a GLUE dataset, so a label vector takes the same code path. The splitter
    returns ``Subset``s, whose ``indices`` are the index sets."""

    def __init__(self, labels):
        self._labels = [int(y) for y in labels]

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        return {"labels": self._labels[index]}


def _lda_split(labels, peers: int, alpha: float) -> Tuple[Tuple[int, ...], ...]:
    """One call of FederatedScope's ``LDASplitter`` over a label vector, under the live
    numpy state (the caller seeds it). Returns each peer's index set, peer by peer."""
    from torch.utils.data import Dataset

    from federatedscope.core.splitters.generic.lda_splitter import LDASplitter

    class Labelled(_Labelled, Dataset):
        pass

    try:
        subsets = LDASplitter(int(peers), alpha=float(alpha))(Labelled(labels))
    except AssertionError as exc:
        # The slice asserts rather than raising a named error.
        raise PartitionError(f"the Dirichlet slice refused the split: {exc}") from exc
    return tuple(tuple(int(i) for i in subset.indices) for subset in subsets)


def lda_index_sets_for_splits(labels_by_split: Mapping[str, Sequence[int]], peers: int,
                              alpha: float, seed: int) -> Dict[str, Tuple[Tuple[int, ...], ...]]:
    """Per-peer index sets for every split, drawn from one seeded stream in the order
    train, val, test. ``{split: (peer_1_indices, peer_2_indices, ...)}``."""
    unknown = sorted(set(labels_by_split) - set(SPLIT_ORDER))
    if unknown:
        raise PartitionError(f"unknown splits {unknown}; expected some of {list(SPLIT_ORDER)}")
    if not labels_by_split:
        raise PartitionError("no split to partition")
    checked = {name: _checked_labels(labels_by_split[name], peers)
               for name in SPLIT_ORDER if name in labels_by_split}
    out: Dict[str, Tuple[Tuple[int, ...], ...]] = {}
    with seeded_numpy(seed):
        for name, labels in checked.items():
            out[name] = _lda_split(labels, peers, alpha)
    return out


def lda_index_sets(labels: Sequence[int], peers: int, alpha: float,
                   seed: int) -> Tuple[Tuple[int, ...], ...]:
    """One split's per-peer index sets: a pure function of (labels, alpha, seed, peers)."""
    return lda_index_sets_for_splits({"train": labels}, peers, alpha, seed)["train"]


_STRATIFIED_CACHE: Dict[Tuple[str, int, int, int, str], Tuple[Tuple[int, ...], ...]] = {}


def stratified_index_sets(full, peers: int, seed: int, holdout: Mapping[str, Any]
                          ) -> Tuple[Tuple[int, ...], ...]:
    """Per peer position, the ascending positions of its training examples (see STRATIFIED).

    The holdout's examples - the scenario's label-stratified readout holdout - are removed
    first; every other training example goes to exactly one peer."""
    import hashlib
    import json

    from federatedscope.contrib.decentralized.simulation.readout import (
        STRATIFIED_HOLDOUT,
        example_ids_and_labels,
        holdout_positions_for,
    )

    if holdout is None or holdout.get("rule") != STRATIFIED_HOLDOUT:
        raise PartitionError(
            f"a {STRATIFIED} split removes the scenario's {STRATIFIED_HOLDOUT} readout holdout "
            f"before it deals; this scenario states {holdout!r}")
    ids, labels = example_ids_and_labels(full)
    fingerprint = hashlib.sha256(json.dumps([ids, labels]).encode("utf-8")).hexdigest()
    key = (fingerprint, len(full), int(peers), int(seed),
           json.dumps(dict(holdout), sort_keys=True))
    if key not in _STRATIFIED_CACHE:
        held = set(holdout_positions_for(full, holdout))
        sequence = []
        for label in sorted(set(labels)):
            members = [position for position, value in enumerate(labels)
                       if value == label and position not in held]

            def order(position, _label=label):
                material = f"{STRATIFIED_KEY}|seed={int(seed)}|label={_label}|example={ids[position]}"
                return (hashlib.sha256(material.encode("utf-8")).digest(), ids[position])
            sequence.extend(sorted(members, key=order))
        sets = [[] for _ in range(int(peers))]
        for index, position in enumerate(sequence):
            sets[index % int(peers)].append(position)
        _STRATIFIED_CACHE[key] = tuple(tuple(sorted(chosen)) for chosen in sets)
    return _STRATIFIED_CACHE[key]


def _apply_stratified(partition, spec: Mapping[str, Any], peer_id: int, peers: Sequence[int],
                      holdout: Optional[Mapping[str, Any]]):
    from torch.utils.data import Subset

    from federatedscope.core.data.base_data import ClientData

    order = tuple(sorted(int(p) for p in peers))
    if int(peer_id) not in order:
        raise PartitionError(f"peer {peer_id} is not one of the peers {list(order)}")
    loader = partition.get("train") if hasattr(partition, "get") else None
    subset = getattr(loader, "dataset", None)
    if not isinstance(subset, Subset) or getattr(subset, "dataset", None) is None:
        raise PartitionError(
            f"the training split is not a Subset of a dataset ({type(subset).__name__}); the "
            f"full split cannot be recovered, so it cannot be dealt")
    full = subset.dataset
    sets = stratified_index_sets(full, len(order), spec["seed"], holdout)
    splits = {"train": Subset(full, list(sets[order.index(int(peer_id))]))}
    for name in ("val", "test"):
        other = partition.get(name)
        if other is not None:
            splits[name] = getattr(other, "dataset", other)
    rebuilt = ClientData(partition.client_cfg, **splits)
    rebuilt.applied_partition = dict(spec)
    return rebuilt


def apply_partition_spec(partition, spec: Mapping[str, Any], peer_id: int,
                         peers: Sequence[int], holdout: Optional[Mapping[str, Any]] = None):
    """This peer's ``ClientData`` re-split under ``spec``.

    FederatedScope gives every peer, per split, a ``Subset`` of the full dataset of that
    split. The full dataset is recovered from it, the split is redrawn under ``spec``, and a
    new ``ClientData`` is built over this peer's new ``Subset``s with the configuration the
    loader used, so its data loaders are built the same way as FederatedScope's own.
    """
    from torch.utils.data import Subset

    from federatedscope.core.data.base_data import ClientData

    spec = validate_partition_spec(spec)
    if spec["splitter"] == STRATIFIED:
        return _apply_stratified(partition, spec, peer_id, peers, holdout)
    order = tuple(sorted(int(p) for p in peers))
    if int(peer_id) not in order:
        raise PartitionError(f"peer {peer_id} is not one of the peers {list(order)}")
    position = order.index(int(peer_id))
    full = {}
    for name in SPLIT_ORDER:
        loader = partition.get(name) if hasattr(partition, "get") else None
        if loader is None:
            continue
        subset = getattr(loader, "dataset", None)
        if not isinstance(subset, Subset) or getattr(subset, "dataset", None) is None:
            raise PartitionError(
                f"split {name!r} is not a Subset of a dataset ({type(subset).__name__}); the "
                f"full dataset cannot be recovered from it, so it cannot be re-partitioned")
        full[name] = subset.dataset
    if not full:
        raise PartitionError("the partition holds no split to re-partition")
    index_sets = lda_index_sets_for_splits(
        {name: labels_of(dataset) for name, dataset in full.items()},
        len(order), spec["alpha"], spec["seed"])
    splits = {name: Subset(full[name], list(index_sets[name][position])) for name in full}
    rebuilt = ClientData(partition.client_cfg, **splits)
    rebuilt.applied_partition = dict(spec)
    return rebuilt
