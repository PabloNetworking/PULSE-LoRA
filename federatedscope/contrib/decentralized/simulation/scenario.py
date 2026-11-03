"""A scenario: one simulated run, described as a JSON document.

A scenario names the method, the peers, the contact trace, the run seed, the configuration
overrides (for the run and per peer), the data builder and partition, the training RNG rule,
the readout schedule and the common construction. Everything a run does is derived from it.
``driver.run_scenario`` executes it.

Every peer is built by :func:`construct_peer`: a per-peer torch reseed, the compute device, the
common canonical construction (``lora/canonical_init.py``: every LoRA component, the classifier
and the pooler start from the run's canonical values, identical across peers, methods and
ranks), and the training RNG rule.
"""
import copy
import hashlib
import importlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from federatedscope.contrib.decentralized.simulation.partition import PartitionError, validate_partition_spec
from federatedscope.contrib.decentralized.simulation.partition import STRATIFIED as STRATIFIED_SPLITTER
from federatedscope.contrib.decentralized.simulation.readout import ReadoutError, validate_readout_schedule
from federatedscope.contrib.decentralized.simulation.training import (
    TRAINING_RNG_POLICIES,
    build_data,
    install_training_rng,
)

SCENARIO_VERSION = 1

#: The ``data.splitter`` values that leave FederatedScope's split IID.
PLAIN_SPLITTERS = ("", "iid")

#: Every key a scenario document may hold.
DOCUMENT_KEYS = ("scenario_version", "method", "peers", "trace", "seed", "peer_seed_base",
                 "adapter", "adapter_rank", "cfg_overrides", "peer_overrides", "data",
                 "training_rng", "data_partition", "readout_schedule", "construction")


class ScenarioError(ValueError):
    """A run description is malformed, or does not describe what it claims."""


@dataclass(frozen=True)
class Scenario:
    """One run: everything the peers must agree on."""

    method: str
    peers: Tuple[int, ...]
    trace: Dict[str, Any]
    seed: int = 11
    #: The torch seed applied immediately before a peer is built is ``peer_seed_base + id``.
    peer_seed_base: int = 0
    #: Dotted path to ``(rank, seed) -> model``, a model handed to the method instead of the
    #: one it builds from the configuration (the tests use small models this way). The
    #: common construction then identifies the backbone by this path. None for a real backbone.
    adapter: Optional[str] = None
    adapter_rank: int = 4
    cfg_overrides: Dict[str, Any] = field(default_factory=dict)
    #: Per-peer configuration overlays, merged the way the runner merges a ``client_<id>``
    #: section: the run's configuration, then the overlay.
    peer_overrides: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    #: Dotted path to ``builder(scenario, peer_id, cfg) -> ClientData``; None for a run
    #: without data. ``glue_data.GLUE_BUILDER`` for a GLUE task.
    data: Optional[str] = None
    #: None (torch's global generator, as the runner uses it) or ``"per_peer_episode"``
    #: (training.py). Readouts inside a run require ``"per_peer_episode"``.
    training_rng: Optional[str] = None
    #: How the training data is split between peers (partition.py); None is
    #: FederatedScope's IID split.
    data_partition: Optional[Dict[str, Any]] = None
    #: When peers are read out and on which holdout (readout.py); None reads nothing out.
    readout_schedule: Optional[Dict[str, Any]] = None
    #: ``{"mode": "common_canonical_v1", "model_init_seed": s}`` (lora/canonical_init.py).
    #: Required.
    construction: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.data_partition is not None:
            try:
                normalised = validate_partition_spec(self.data_partition)
            except PartitionError as exc:
                raise ScenarioError(f"data_partition: {exc}") from exc
            object.__setattr__(self, "data_partition", normalised)
        if self.readout_schedule is not None:
            try:
                schedule = validate_readout_schedule(self.readout_schedule)
            except ReadoutError as exc:
                raise ScenarioError(f"readout_schedule: {exc}") from exc
            object.__setattr__(self, "readout_schedule", schedule)
        if (self.data_partition is not None
                and self.data_partition.get("splitter") == STRATIFIED_SPLITTER):
            from federatedscope.contrib.decentralized.simulation.readout import STRATIFIED_HOLDOUT

            holdout = (self.readout_schedule or {}).get("holdout") or {}
            if holdout.get("rule") != STRATIFIED_HOLDOUT:
                raise ScenarioError(
                    f"data_partition: a {STRATIFIED_SPLITTER} split deals what the scenario's "
                    f"{STRATIFIED_HOLDOUT} readout holdout leaves; this scenario states the "
                    f"holdout {holdout!r}")
        if self.data_partition is not None and not self.data:
            raise ScenarioError("data_partition: a scenario without data has nothing to split")
        splitter = (self.cfg_overrides or {}).get("data.splitter", "")
        if splitter not in PLAIN_SPLITTERS:
            raise ScenarioError(
                f"cfg_overrides sets data.splitter {splitter!r}; FederatedScope's IID split is the "
                f"only one get_data may make here, and any other split is stated as data_partition")
        if self.training_rng not in TRAINING_RNG_POLICIES:
            raise ScenarioError(f"training_rng {self.training_rng!r} is not one of "
                                f"{list(TRAINING_RNG_POLICIES)}")
        from federatedscope.contrib.decentralized.lora.canonical_init import (
            ConstructionError,
            validate_spec,
        )
        if self.construction is None:
            raise ScenarioError("a scenario must state its construction: "
                                "{\"mode\": \"common_canonical_v1\", \"model_init_seed\": <int>}")
        try:
            canonical = validate_spec(self.construction)
        except ConstructionError as exc:
            raise ScenarioError(f"construction: {exc}") from exc
        object.__setattr__(self, "construction", canonical)

    def peer_seed(self, peer_id: int) -> int:
        return self.peer_seed_base + int(peer_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_version": SCENARIO_VERSION,
            "method": self.method,
            "peers": list(self.peers),
            "trace": self.trace,
            "seed": self.seed,
            "peer_seed_base": self.peer_seed_base,
            "adapter": self.adapter,
            "adapter_rank": self.adapter_rank,
            "cfg_overrides": dict(self.cfg_overrides),
            "peer_overrides": {str(p): dict(v) for p, v in sorted(self.peer_overrides.items())},
            "data": self.data,
            "training_rng": self.training_rng,
            "data_partition": None if self.data_partition is None else dict(self.data_partition),
            "readout_schedule": None if self.readout_schedule is None else {
                "every_frames": self.readout_schedule["every_frames"],
                "initial": self.readout_schedule["initial"],
                "holdout": dict(self.readout_schedule["holdout"])},
            "construction": dict(self.construction),
        }

    @property
    def digest(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def scenario_from_dict(payload: Dict[str, Any]) -> Scenario:
    if type(payload) is not dict:
        raise ScenarioError("a scenario document must be a plain mapping")
    version = payload.get("scenario_version")
    if version != SCENARIO_VERSION:
        raise ScenarioError(
            f"scenario schema version {version!r} is not the expected "
            f"{SCENARIO_VERSION}; refusing to guess what the fields mean")
    unknown = sorted(set(payload) - set(DOCUMENT_KEYS))
    if unknown:
        raise ScenarioError(f"unknown scenario keys {unknown}")
    for required in ("method", "peers", "trace", "construction"):
        if required not in payload:
            raise ScenarioError(f"scenario is missing {required!r}")
    peers = payload["peers"]
    if (type(peers) is not list or not peers
            or any(type(p) is not int or type(p) is bool or p < 1 for p in peers)):
        raise ScenarioError("scenario peers must be positive integers")
    if len(set(peers)) != len(peers):
        raise ScenarioError(f"scenario peers repeat: {peers}")
    for name, default in (("seed", 11), ("peer_seed_base", 0), ("adapter_rank", 4)):
        value = payload.get(name, default)
        if type(value) is not int:
            raise ScenarioError(f"{name} must be an integer, not {value!r}")
    data = payload.get("data")
    if data is not None and (type(data) is not str or not data):
        raise ScenarioError("data must be a dotted builder path, or absent")
    training_rng = payload.get("training_rng")
    if type(training_rng) not in (type(None), str) or training_rng not in TRAINING_RNG_POLICIES:
        raise ScenarioError(f"training_rng {training_rng!r} is not one of "
                            f"{list(TRAINING_RNG_POLICIES)}")
    return Scenario(
        method=payload["method"],
        peers=tuple(peers),
        trace=payload["trace"],
        seed=int(payload.get("seed", 11)),
        peer_seed_base=int(payload.get("peer_seed_base", 0)),
        adapter=payload.get("adapter"),
        adapter_rank=int(payload.get("adapter_rank", 4)),
        cfg_overrides=dict(payload.get("cfg_overrides") or {}),
        peer_overrides={int(p): dict(v)
                        for p, v in (payload.get("peer_overrides") or {}).items()},
        data=data,
        training_rng=training_rng,
        data_partition=payload.get("data_partition"),
        readout_schedule=payload.get("readout_schedule"),
        construction=payload.get("construction"),
    )


def save_scenario(scenario: Scenario, path: str) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(scenario.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def load_scenario(path: str) -> Scenario:
    with open(path, "r", encoding="utf-8") as handle:
        return scenario_from_dict(json.load(handle))


# -- deriving the run ----------------------------------------------------------------------
#
# Everything below is a pure function of the scenario.

def build_cfg(scenario: Scenario, peer_id: Optional[int] = None):
    """The configuration of this run, or of one peer if ``peer_id`` is given.

    With ``peer_id`` the peer's overlay is merged the way
    ``DecentralizedRunner._resolve_peer_config`` merges a ``client_<id>`` section.
    """
    from federatedscope.core.configs.config import global_cfg

    cfg = global_cfg.clone()
    settings = {
        "federate.method": scenario.method,
        "federate.client_num": len(scenario.peers),
        "decentralized.enabled": True,
        "decentralized.seed": scenario.seed,
        "outdir": "",
        # Sessions are admitted in the serialized bytes of the socket_v1 protocol
        # (wire_quote.py) unless the scenario's cfg_overrides choose another wire-cost model.
        "decentralized.transport.wire_cost": "socket_v1",
    }
    # Copied: merge_from_list keeps a non-string value by reference, so without the copy
    # every configuration built from one Scenario would share the document's own lists.
    settings.update(copy.deepcopy(dict(scenario.cfg_overrides)))
    flat = []
    for key, value in settings.items():
        flat.extend([key, value])
    cfg.merge_from_list(flat)
    if peer_id is not None and scenario.peer_overrides.get(int(peer_id)):
        cfg.merge_from_other_cfg(client_section(scenario, int(peer_id)))
    return cfg


#: A run-level setting that a peer overlay may not change.
ENVELOPE_KEY = "decentralized.transport.envelope_bytes"


def client_section(scenario: Scenario, peer_id: int):
    """One peer's overlay as the CfgNode section the runner expects."""
    from federatedscope.core.configs.config import CN

    if ENVELOPE_KEY in scenario.peer_overrides.get(int(peer_id), {}):
        raise ScenarioError(
            f"peer_overrides[{peer_id}] sets {ENVELOPE_KEY}: the envelope is one run-level "
            f"value that both endpoints of every session price under - set it in "
            f"cfg_overrides; a per-peer envelope is refused")
    nested: Dict[str, Any] = {}
    for dotted, value in sorted(scenario.peer_overrides.get(int(peer_id), {}).items()):
        value = copy.deepcopy(value)            # never alias the document; see build_cfg
        node = nested
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return CN(nested)


def client_configs(scenario: Scenario):
    """Every peer's overlay, keyed as the runner reads them; None when there are none."""
    if not scenario.peer_overrides:
        return None
    return {f"client_{p}": client_section(scenario, p) for p in sorted(scenario.peer_overrides)}


def build_factory(scenario: Scenario, cfg=None):
    """The method's registered factory."""
    from federatedscope.contrib.decentralized import bootstrap, registry

    bootstrap.load_methods()
    if not registry.is_decentralized_method(scenario.method):
        raise ScenarioError(
            f"{scenario.method!r} is not a registered decentralized method")
    return registry.get_factory(scenario.method)(cfg if cfg is not None
                                                 else build_cfg(scenario))


def _adapter_for(scenario: Scenario, peer_id: int):
    if not scenario.adapter:
        return None
    module_name, _, attribute = scenario.adapter.rpartition(".")
    builder = getattr(importlib.import_module(module_name), attribute)
    return builder(rank=scenario.adapter_rank, seed=scenario.peer_seed(peer_id))


def construct_peer(scenario: Scenario, peer_id: int, context, create_peer,
                   training_rng: Any = "scenario", device: Optional[str] = None):
    """Build one peer through ``create_peer``, with everything a scenario adds to it.

      - a torch reseed with ``scenario.peer_seed(peer_id)``, so a peer does not depend on what
        was built before it;
      - the scenario's ``adapter`` model, if it names one, through ``context.extras["model"]``;
      - the compute device of a peer that holds data: ``device`` (``"cpu"`` by default, or a
        CUDA device such as ``"cuda:0"``). Models rest on the host and the trainer moves them
        to the compute device for each training routine;
      - the common construction: supplied to the method through
        ``context.extras[canonical_init.EXTRAS_KEY]`` before it builds the peer, and applied to
        the built model before any optimizer exists;
      - the training RNG rule (training.py); ``training_rng`` overrides the scenario's.

    Data is not built here: it must exist before the reseed, because a data builder may reseed.
    """
    import torch

    from federatedscope.contrib.decentralized.lora.canonical_init import (
        EXTRAS_KEY,
        CommonConstruction,
        ConstructionError,
        construction_for,
    )

    if int(getattr(context.config, "device", -1)) >= 0:
        raise ScenarioError(f"peer {peer_id}: cfg.device is {context.config.device}; models are "
                            f"built on the host (cfg.device -1) and trained on the compute "
                            f"device the driver is given")
    torch.manual_seed(scenario.peer_seed(peer_id))
    adapter = _adapter_for(scenario, peer_id)
    if adapter is not None:
        context.extras["model"] = adapter
    if context.data is not None:
        context.device = str(device) if device is not None else "cpu"
    try:
        if scenario.adapter:
            construction = CommonConstruction(
                model_init_seed=scenario.construction["model_init_seed"],
                backbone_digest=hashlib.sha256(f"adapter:{scenario.adapter}".encode("utf-8")).hexdigest(),
                backbone_path="")
        else:
            construction = construction_for(scenario.construction, context.config)
    except ConstructionError as exc:
        raise ScenarioError(f"peer {peer_id}: {exc}") from exc
    context.extras[EXTRAS_KEY] = construction
    algorithm = create_peer(context)
    try:
        construction.apply(algorithm, context, who=f"peer {peer_id}")
    except ConstructionError as exc:
        raise ScenarioError(str(exc)) from exc
    try:
        install_training_rng(algorithm, scenario, peer_id, training_rng)
    except ValueError as exc:
        raise ScenarioError(str(exc)) from exc
    return algorithm


def build_peer(scenario: Scenario, peer_id: int, cfg=None, factory=None,
               topology=None, data=None, device: Optional[str] = None):
    """Build one peer outside a runner, deterministically, whatever else was built before.

    ``topology`` is handed over where the runner hands its own, through
    ``extras["topology"]``: TAD-LoRA reads recent contact degree from it. ``data`` is this
    peer's data if the caller built it; otherwise a scenario with a data builder builds it
    here, before the construction reseed.
    """
    from federatedscope.contrib.decentralized.core.interfaces import PeerContext

    if peer_id not in scenario.peers:
        raise ScenarioError(
            f"peer {peer_id} is not part of this scenario {list(scenario.peers)}")
    if scenario.peer_overrides.get(int(peer_id)):
        cfg = build_cfg(scenario, peer_id)
    cfg = cfg if cfg is not None else build_cfg(scenario)
    factory = factory if factory is not None else build_factory(scenario, cfg)
    if data is not None and not scenario.data:
        raise ScenarioError("data was supplied for a dataset-free scenario")
    if scenario.data and data is None:
        data = build_data(scenario, peer_id, cfg)

    extras = {}
    if topology is not None:
        extras["topology"] = topology
    context = (PeerContext(peer_id=peer_id, config=cfg, extras=extras) if data is None else
               PeerContext(peer_id=peer_id, config=cfg, data=data, device="cpu", extras=extras))
    return construct_peer(scenario, peer_id, context, factory.create_peer, device=device)


def build_trace(scenario: Scenario):
    from federatedscope.contrib.decentralized.core.traces import load_trace

    return load_trace(scenario.trace)


# -- state digests ---------------------------------------------------------------------------

def learning_state_digest(algorithm) -> str:
    """A digest of the adapter tensors and the committed state version.

    Unlike :func:`exact_state_digest` it ignores session bookkeeping, which a failed session
    legitimately changes; a clean rollback leaves this digest unchanged.
    """
    digest = hashlib.sha256()
    adapter = algorithm.state_dict().get("adapter", {}) or {}
    for key in sorted(adapter):
        value = adapter[key]
        digest.update(str(key).encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes()
                      if hasattr(value, "detach") else repr(value).encode("utf-8"))
    digest.update(str(algorithm.state_version()).encode("utf-8"))
    return digest.hexdigest()


def learned_tensor_digest(algorithm) -> str:
    """A digest of the adapter tensors alone: it moves if and only if a learned value moved.

    Each tensor's path (in sorted order), dtype, shape and host bytes go in. The state version,
    session bookkeeping and optimizer and scheduler state are excluded. The same values give
    the same digest on the host and on a GPU.
    """
    import torch

    digest = hashlib.sha256()
    adapter = algorithm.state_dict().get("adapter", {}) or {}
    for key in sorted(adapter):
        value = adapter[key]
        digest.update(f"path:{key}:".encode("utf-8"))
        if hasattr(value, "detach"):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode("utf-8"))
            # Raw bytes through a reinterpreting view, whatever the dtype.
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(f"plain:{value!r}".encode("utf-8"))
    return digest.hexdigest()


#: Leaves whose repr is their exact value.
_EXACT_SCALARS = (type(None), bool, int, float, str, bytes)


def _leaf_digest(value) -> str:
    import numpy
    import torch

    digest = hashlib.sha256()
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode("utf-8"))
        # Raw bytes, whatever the dtype: a reinterpreting view, never a repr.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, numpy.ndarray):
        array = numpy.ascontiguousarray(value)
        digest.update(f"ndarray:{array.dtype}:{array.shape}:".encode("utf-8"))
        digest.update(array.tobytes())
    elif isinstance(value, (set, frozenset)):
        # A set's repr follows string hashing, which differs between processes.
        if any(not isinstance(item, _EXACT_SCALARS) for item in value):
            raise TypeError(f"a set holding {sorted({type(i).__name__ for i in value})} "
                            f"cannot be digested exactly")
        digest.update(f"{type(value).__name__}:".encode("utf-8"))
        digest.update(repr(sorted(repr(item) for item in value)).encode("utf-8"))
    elif isinstance(value, _EXACT_SCALARS):
        digest.update(f"{type(value).__name__}:{value!r}".encode("utf-8"))
    else:
        # A repr would be lossy: an object holding a tensor would be compared to four digits.
        raise TypeError(f"{type(value).__name__} is not a leaf the exact digest handles")
    return digest.hexdigest()


def exact_state_paths(algorithm) -> Dict[str, str]:
    """Every leaf of the peer's ``state_dict()``, by path, digested exactly.

    Every tensor at every depth contributes its raw bytes, dtype and shape; mappings are walked
    in sorted key order; every container records its type and length, so an empty entry that
    one side has and the other lacks is a difference. Paths rather than one number, so a
    disagreement names what disagrees.
    """
    paths: Dict[str, str] = {}

    def walk(path: str, value) -> None:
        if isinstance(value, dict):
            paths[path] = f"dict:{len(value)}"
            for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
                walk(f"{path}/{key!r}", value[key])
        elif isinstance(value, (list, tuple)):
            paths[path] = f"{type(value).__name__}:{len(value)}"
            for position, item in enumerate(value):
                walk(f"{path}[{position}]", item)
        else:
            try:
                paths[path] = _leaf_digest(value)
            except TypeError as exc:
                raise TypeError(f"state path {path or '/'}: {exc}") from exc

    walk("", algorithm.state_dict())
    return paths


def exact_state_digest(algorithm) -> str:
    """One digest over :func:`exact_state_paths`."""
    digest = hashlib.sha256()
    for path, value in sorted(exact_state_paths(algorithm).items()):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
