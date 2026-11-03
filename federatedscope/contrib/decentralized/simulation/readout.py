"""Readout schedule, the readout holdout, and the forward pass that produces readout logits.

A scenario's ``readout_schedule`` is ``{"every_frames": k, "initial": bool, "holdout":
{"source": "train", "count": N, "seed": s[, "rule": "label_stratified_hash_v1"]}}``. Every
peer is read out before the first frame (``initial``), after every k-th completed frame and
after the last frame. A readout evaluates the peer's current model on the holdout and keeps its
logits.

The holdout is a fixed number of training examples, drawn from the full training split and
removed from every peer's training data after the split between peers. It is identical for
every peer and method and disjoint from all training data. The GLUE validation split is not
used.
"""
import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

SCHEDULE_KEYS = ("every_frames", "holdout", "initial")
HOLDOUT_KEYS = ("count", "seed", "source")
#: Optional holdout key naming the selection rule. Without it the holdout is a seeded
#: permutation of the training split.
HOLDOUT_RULE = "rule"
#: The label-stratified rule: per-label quotas by largest remainder over the full split's label
#: counts (ties to the lower label), each label's examples ordered by the sha256 of
#: ``{STRATIFIED_HOLDOUT_KEY}|{seed}|example={idx}`` (ties by idx), the first quota taken, the
#: holdout kept in ascending original id.
STRATIFIED_HOLDOUT = "label_stratified_hash_v1"
HOLDOUT_RULES = (STRATIFIED_HOLDOUT,)
STRATIFIED_HOLDOUT_KEY = "pulse-h4-holdout-v1"
#: The holdout is always drawn from the training split.
TRAIN = "train"
HOLDOUT_SOURCES = (TRAIN,)

#: Refusals, named so a test can say which one fired.
READOUT_SCHEDULE_INVALID = "READOUT_SCHEDULE_INVALID"
HOLDOUT_NOT_APPLIED = "HOLDOUT_NOT_APPLIED"
HOLDOUT_TOO_LARGE = "HOLDOUT_TOO_LARGE"
HOLDOUT_UNSUPPORTED = "HOLDOUT_UNSUPPORTED"
HOLDOUT_EMPTIES_A_PEER = "HOLDOUT_EMPTIES_A_PEER"


class ReadoutError(ValueError):
    """A readout schedule or holdout that cannot be honoured, with a code naming the refusal."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _plain_int(value: Any) -> bool:
    return type(value) is int


def validate_readout_schedule(spec: Any) -> Dict[str, Any]:
    """The schedule in canonical form, or a refusal naming what is wrong.

    Exactly the keys ``every_frames`` (int >= 1), ``initial`` (bool) and ``holdout``, whose
    keys are ``source`` ("train"), ``count`` (int >= 1), ``seed`` (int in [0, 2**32)) and
    optionally ``rule`` (``HOLDOUT_RULES``). An unknown key is refused rather than ignored:
    an ignored ``"count "`` would be a run carrying a holdout size nothing applied. ``True``
    is not an integer here.
    """
    if type(spec) is not dict:
        raise ReadoutError(READOUT_SCHEDULE_INVALID,
                           f"readout_schedule must be a plain mapping with keys "
                           f"{list(SCHEDULE_KEYS)}, not {type(spec).__name__}")
    unknown = sorted(set(spec) - set(SCHEDULE_KEYS))
    missing = [key for key in SCHEDULE_KEYS if key not in spec]
    if unknown or missing:
        raise ReadoutError(READOUT_SCHEDULE_INVALID,
                           f"readout_schedule must have exactly the keys {list(SCHEDULE_KEYS)}; "
                           f"unknown {unknown}, missing {missing}")
    every = spec["every_frames"]
    if not _plain_int(every) or every < 1:
        raise ReadoutError(READOUT_SCHEDULE_INVALID,
                           f"readout_schedule.every_frames must be an integer >= 1, not {every!r}")
    initial = spec["initial"]
    if type(initial) is not bool:
        raise ReadoutError(READOUT_SCHEDULE_INVALID,
                           f"readout_schedule.initial must be true or false, not {initial!r}")
    holdout = spec["holdout"]
    if type(holdout) is not dict:
        raise ReadoutError(READOUT_SCHEDULE_INVALID,
                           f"readout_schedule.holdout must be a plain mapping with keys "
                           f"{list(HOLDOUT_KEYS)}")
    allowed = HOLDOUT_KEYS + ((HOLDOUT_RULE,) if HOLDOUT_RULE in holdout else ())
    unknown = sorted(set(holdout) - set(allowed))
    missing = [key for key in HOLDOUT_KEYS if key not in holdout]
    if unknown or missing:
        raise ReadoutError(READOUT_SCHEDULE_INVALID,
                           f"readout_schedule.holdout must have exactly the keys "
                           f"{list(HOLDOUT_KEYS)} (and optionally {HOLDOUT_RULE!r}); unknown "
                           f"{unknown}, missing {missing}")
    if HOLDOUT_RULE in holdout and holdout[HOLDOUT_RULE] not in HOLDOUT_RULES:
        raise ReadoutError(READOUT_SCHEDULE_INVALID,
                           f"readout_schedule.holdout.rule {holdout[HOLDOUT_RULE]!r} is not one "
                           f"of {list(HOLDOUT_RULES)}; the seeded permutation is the absence "
                           f"of a rule")
    if holdout["source"] not in HOLDOUT_SOURCES:
        raise ReadoutError(READOUT_SCHEDULE_INVALID,
                           f"readout_schedule.holdout.source {holdout['source']!r} is not one "
                           f"of {list(HOLDOUT_SOURCES)}; the official development split is "
                           f"reserved for final evaluation")
    count = holdout["count"]
    if not _plain_int(count) or count < 1:
        raise ReadoutError(READOUT_SCHEDULE_INVALID,
                           f"readout_schedule.holdout.count must be an integer >= 1, not "
                           f"{count!r}")
    seed = holdout["seed"]
    if not _plain_int(seed) or seed < 0 or seed >= 2 ** 32:
        raise ReadoutError(READOUT_SCHEDULE_INVALID,
                           f"readout_schedule.holdout.seed must be an integer in [0, 2**32), "
                           f"not {seed!r}")
    canonical = {"source": str(holdout["source"]), "count": int(count), "seed": int(seed)}
    if HOLDOUT_RULE in holdout:
        canonical[HOLDOUT_RULE] = str(holdout[HOLDOUT_RULE])
    return {"every_frames": int(every), "initial": bool(initial), "holdout": canonical}


def capture_due(schedule: Mapping[str, Any], frames_completed: int, total_frames: int) -> bool:
    """Whether a capture follows the frame that brought the run to ``frames_completed``.

    Every ``every_frames``-th completed frame, and always the last one, so the final state
    is read whatever the cadence. ``frames_completed`` 0 is the initial capture."""
    if frames_completed == 0:
        return bool(schedule["initial"])
    return (frames_completed % int(schedule["every_frames"]) == 0
            or frames_completed == int(total_frames))


def capture_points(schedule: Mapping[str, Any], total_frames: int) -> List[int]:
    """Every ``frames_completed`` value a run of ``total_frames`` frames captures after."""
    return [n for n in range(0, int(total_frames) + 1) if capture_due(schedule, n, total_frames)]


def holdout_positions(total: int, count: int, seed: int) -> Tuple[int, ...]:
    """``count`` positions of a ``total``-example split, by a seeded permutation, sorted.

    A private ``numpy.random.RandomState`` (numpy's legacy generator, whose stream numpy keeps
    stable across versions), so the global numpy state is never read or moved."""
    import numpy as np

    total, count = int(total), int(count)
    if count >= total:
        raise ReadoutError(HOLDOUT_TOO_LARGE,
                           f"a holdout of {count} from a split of {total} examples leaves "
                           f"nothing to train on")
    permutation = np.random.RandomState(int(seed)).permutation(total)
    return tuple(sorted(int(i) for i in permutation[:count]))


def example_ids_and_labels(dataset: Any) -> Tuple[List[int], List[int]]:
    """Every example's original id (the dataset's ``idx`` column) and label, in dataset order.

    A GLUE dataset keeps its raw examples (``examples``: ``idx``, ``label``); the label-
    stratified rules identify examples by that id, never by a position, so a dataset without
    one is refused rather than given positions as ids."""
    examples = getattr(dataset, "examples", None)
    if examples is None:
        raise ReadoutError(HOLDOUT_UNSUPPORTED,
                           f"{type(dataset).__name__} keeps no raw examples; the label-stratified "
                           f"rules identify examples by their original id")
    ids = [int(example["idx"]) for example in examples]
    labels = [int(example["label"]) for example in examples]
    if len(set(ids)) != len(ids):
        raise ReadoutError(HOLDOUT_UNSUPPORTED, "the dataset repeats an original id")
    return ids, labels


def stratified_quotas(label_counts: Mapping[int, int], count: int) -> Dict[int, int]:
    """``count`` slots over the labels in proportion to their counts: floors, then one more slot
    to each of the largest exact remainders, ties to the lower label id. Integers only."""
    total = sum(int(n) for n in label_counts.values())
    if not 0 < int(count) < total:
        raise ReadoutError(HOLDOUT_TOO_LARGE, f"a holdout of {count} from {total} examples")
    floors = {int(c): int(count) * int(n) // total for c, n in label_counts.items()}
    remainders = {int(c): int(count) * int(n) % total for c, n in label_counts.items()}
    extra = int(count) - sum(floors.values())
    for label in sorted(remainders, key=lambda c: (-remainders[c], c))[:extra]:
        floors[label] += 1
    return dict(sorted(floors.items()))


def stratified_holdout_positions(dataset: Any, spec: Mapping[str, Any]) -> Tuple[int, ...]:
    """The label-stratified holdout's positions in ``dataset``, in ascending original id."""
    ids, labels = example_ids_and_labels(dataset)
    by_label: Dict[int, List[int]] = {}
    for position, label in enumerate(labels):
        by_label.setdefault(int(label), []).append(position)
    quotas = stratified_quotas({c: len(v) for c, v in by_label.items()}, int(spec["count"]))
    chosen: List[int] = []
    for label in sorted(by_label):
        def key(position, _seed=int(spec["seed"])):
            material = f"{STRATIFIED_HOLDOUT_KEY}|{_seed}|example={ids[position]}"
            return (hashlib.sha256(material.encode("utf-8")).digest(), ids[position])
        chosen.extend(sorted(by_label[label], key=key)[:quotas[label]])
    return tuple(sorted(chosen, key=lambda position: ids[position]))


def holdout_positions_for(dataset: Any, spec: Mapping[str, Any]) -> Tuple[int, ...]:
    """The holdout's positions in the full split ``dataset`` under ``spec``'s rule."""
    if spec.get(HOLDOUT_RULE) is None:
        return holdout_positions(len(dataset), spec["count"], spec["seed"])
    if spec[HOLDOUT_RULE] == STRATIFIED_HOLDOUT:
        return stratified_holdout_positions(dataset, spec)
    raise ReadoutError(HOLDOUT_UNSUPPORTED, f"unknown holdout rule {spec[HOLDOUT_RULE]!r}")


def _item_bytes(item: Any) -> bytes:
    """A canonical encoding of one example: every tensor's dtype, shape and bytes, every
    mapping in sorted key order. Stable across processes."""
    import torch

    if isinstance(item, torch.Tensor):
        tensor = item.detach().cpu().contiguous()
        return (f"t:{tensor.dtype}:{tuple(tensor.shape)}:".encode("utf-8")
                + tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    if isinstance(item, Mapping):
        parts = [f"m{len(item)}:".encode("utf-8")]
        for key in sorted(item, key=str):
            parts.append(f"k:{key}:".encode("utf-8"))
            parts.append(_item_bytes(item[key]))
        return b"".join(parts)
    if isinstance(item, (list, tuple)):
        return (f"s{len(item)}:".encode("utf-8")
                + b"".join(_item_bytes(value) for value in item))
    if isinstance(item, (bool, int, float, str, type(None))):
        return f"v:{type(item).__name__}:{item!r}".encode("utf-8")
    raise ReadoutError(HOLDOUT_UNSUPPORTED,
                       f"a holdout example holds a {type(item).__name__}, which has no "
                       f"canonical encoding here")


def holdout_digest(dataset: Any) -> str:
    """The holdout's identity: its positions in the full split (when it is a ``Subset``)
    and every example's contents, in order."""
    digest = hashlib.sha256(b"pulse/readout/holdout/v1\n")
    indices = getattr(dataset, "indices", None)
    if indices is not None:
        base = getattr(dataset, "dataset", None)
        digest.update(f"subset:{len(base) if base is not None else '?'}:".encode("utf-8"))
        digest.update(",".join(str(int(i)) for i in indices).encode("utf-8"))
    digest.update(f"|n={len(dataset)}|".encode("utf-8"))
    for position in range(len(dataset)):
        digest.update(_item_bytes(dataset[position]))
        digest.update(b"\x1e")
    return digest.hexdigest()


def declare_holdout(partition: Any, spec: Mapping[str, Any], dataset: Any) -> Any:
    """Mark ``partition`` as carrying ``spec``'s holdout ``dataset``.

    Attributes, never items: the manifest's partition digest iterates a ``ClientData``'s
    items as splits, and the holdout is not a split any peer trains or is evaluated on."""
    partition.readout_holdout = dataset
    partition.applied_holdout = {"spec": dict(spec), "size": int(len(dataset)),
                                 "digest": holdout_digest(dataset)}
    return partition


def apply_holdout(partition: Any, spec: Mapping[str, Any]) -> Any:
    """``partition`` with ``spec``'s holdout subtracted from its training split.

    The training loader must hold a ``Subset`` of the full training split - what
    ``split_to_client`` and ``apply_partition_spec`` both hand every peer - so the holdout
    is drawn over the full split, identical for every peer, and removed from this peer's
    indices. A fresh ``ClientData`` is built over the remaining indices with the cfg the
    loader built the old one with; validation and test splits are passed through unchanged,
    and a partition declaration (``applied_partition``) is carried over."""
    from torch.utils.data import Subset

    from federatedscope.core.data.base_data import ClientData

    loader = partition.get("train") if hasattr(partition, "get") else None
    subset = getattr(loader, "dataset", None)
    if not isinstance(subset, Subset) or getattr(subset, "dataset", None) is None:
        raise ReadoutError(HOLDOUT_UNSUPPORTED,
                           f"the training split is not a Subset of a dataset "
                           f"({type(subset).__name__}); the full split cannot be recovered, "
                           f"so no holdout can be drawn over it")
    full = subset.dataset
    if spec.get(HOLDOUT_RULE) is None:
        positions = holdout_positions(len(full), spec["count"], spec["seed"])
    else:
        positions = holdout_positions_for(full, spec)
    held = set(positions)
    kept = [int(i) for i in subset.indices if int(i) not in held]
    if not kept:
        raise ReadoutError(HOLDOUT_EMPTIES_A_PEER,
                           f"the holdout removes every one of this peer's "
                           f"{len(subset.indices)} training examples")
    splits = {"train": Subset(full, kept)}
    for name in ("val", "test"):
        other = partition.get(name)
        if other is not None:
            splits[name] = getattr(other, "dataset", other)
    rebuilt = ClientData(partition.client_cfg, **splits)
    if getattr(partition, "applied_partition", None) is not None:
        rebuilt.applied_partition = dict(partition.applied_partition)
    return declare_holdout(rebuilt, spec, Subset(full, list(positions)))


def require_applied_holdout(data: Any, schedule: Mapping[str, Any], peer_id: int,
                            builder: str) -> None:
    """A stated schedule whose holdout a builder did not apply is refused by name.

    Otherwise the run's training data would still contain the examples its readouts are
    scored on, while the scenario says they were held out."""
    applied = getattr(data, "applied_holdout", None)
    held = getattr(data, "readout_holdout", None)
    if applied is None or held is None or applied.get("spec") != schedule["holdout"]:
        raise ReadoutError(
            HOLDOUT_NOT_APPLIED,
            f"peer {peer_id}: the scenario states readout holdout {schedule['holdout']!r} and "
            f"the builder {builder!r} returned a partition that declares "
            f"{'no holdout' if applied is None else repr(applied.get('spec'))}; a builder "
            f"must apply the stated holdout (federatedscope.contrib.decentralized.simulation.readout.apply_holdout "
            f"or declare_holdout) and say so, or the scenario must not state one")


def holdout_loader(data: Any, batch_size: Optional[int] = None):
    """The holdout in fixed order, through FederatedScope's loader builder (the ``test`` split
    rule: never shuffled). ``batch_size`` overrides the configured batch size."""
    from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader

    holdout = getattr(data, "readout_holdout", None)
    if holdout is None:
        raise ReadoutError(HOLDOUT_NOT_APPLIED, "the peer's data carries no readout holdout")
    cfg = data.client_cfg
    if batch_size is not None:
        cfg = cfg.clone()
        cfg.defrost()
        cfg.dataloader.batch_size = int(batch_size)
    return get_dataloader(holdout, cfg, "test")


def holdout_labels(data: Any) -> "Any":
    import numpy as np

    holdout = data.readout_holdout
    return np.array([int(holdout[i]["labels"]) for i in range(len(holdout))], dtype=np.int64)


def forward_logits(algorithm: Any, loader: Iterable, device: str = "cpu") -> "Any":
    """Every holdout example's logits, in loader order: the trainer's evaluation forward
    (``GLUEBaseTrainer._hook_on_batch_forward``'s call, without labels, which only add a
    loss) in eval mode under ``no_grad``. The model is put back where it rests and in the
    mode it was in."""
    import numpy as np
    import torch

    trainer = getattr(algorithm, "trainer", None)
    model = getattr(getattr(trainer, "ctx", None), "model", None)
    if model is None:
        raise ReadoutError(HOLDOUT_UNSUPPORTED, "the peer has no trainer model to read out")
    was_training = model.training
    model.to(device)
    model.eval()
    outputs = []
    try:
        with torch.no_grad():
            for batch in loader:
                token_type_ids = batch.get("token_type_ids", None)
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    token_type_ids=None if token_type_ids is None else token_type_ids.to(device),
                ).logits
                outputs.append(logits.detach().to(torch.float32).cpu())
    finally:
        model.train(was_training)
        trainer.discharge_model()
    return np.ascontiguousarray(torch.cat(outputs).numpy())


def logits_sha256(logits: "Any") -> str:
    import numpy as np

    array = np.ascontiguousarray(logits)
    return hashlib.sha256(f"{array.dtype}:{array.shape}:".encode("utf-8") + array.tobytes()).hexdigest()
