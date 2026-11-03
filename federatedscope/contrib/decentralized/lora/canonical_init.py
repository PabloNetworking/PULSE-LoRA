"""Common canonical construction, ``common_canonical_v1``.

Every LoRA component of a run gets one initial value, the same whichever peer, method, active
rank or support asks for it:

  - a canonical rank universe of 200 component ids. A PULSE-LoRA peer with ``r_max`` 200 and a
    rank-r Dec-LoRA or TAD-LoRA peer (ids 0..r-1) address the same components;
  - for target module ``m`` and component ``p``: ``A[m][p, :]`` independent uniform values in
    ``[-1/sqrt(d_in), +1/sqrt(d_in)]`` (the marginal of peft's ``kaiming_uniform_(a=sqrt(5))``),
    drawn by an isolated CPU ``torch.Generator``; ``B[m][:, p]`` zero;
  - the generator's seed: the first eight bytes, unsigned big-endian, of the sha256 of the
    UTF-8 canonical JSON (sorted keys, compact separators) of ``{namespace, schema_id,
    model_init_seed, module, component_id}``, reduced modulo 2**63;
  - the classifier and the frozen pooler: copied from one CPU donor model per (backbone,
    model-init seed), built through ``get_model`` at the canonical rank-200 configuration under
    the model-init seed, in an isolated RNG scope.

The schema id identifies the canonical schema and nothing else: the pretrained backbone (its
``config.json`` and weight files, by sha256), the sorted target modules with their dimensions,
the dtype, the effective scale 2.0, the rank universe 200 and the initialisation version. It
never includes a method, a peer, an active rank, a support or the model-init seed. Because
the backbone files are hashed, the canonical values depend on the exact files in the model
directory.

A scenario states its construction as ``{"mode": "common_canonical_v1", "model_init_seed": s}``.
``simulation.scenario.construct_peer`` hands it to the method through ``PeerContext.extras``
(:data:`EXTRAS_KEY`) before the peer is built, and applies it to the built model before any
optimizer exists.

Values are cached per process, keyed by every input that determines them, and handed out as
copies: no tensor is shared between two peers.

Shared initial ids do not guarantee that factors keep identical meanings after local training.
"""
import copy
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch

MODE = "common_canonical_v1"
NAMESPACE = "pulse_common_component_v1"
#: The initialisation rule's version, part of the schema id.
INIT_VERSION = "common_canonical_v1:a=uniform(+-1/sqrt(d_in)),b=0"
RANK_UNIVERSE = 200
EFFECTIVE_SCALE = 2.0
SPEC_KEYS = ("mode", "model_init_seed")
#: The ``PeerContext.extras`` key a method reads its run's construction from.
EXTRAS_KEY = "common_construction"
#: The method hook :meth:`CommonConstruction.apply` calls:
#: ``() -> (adapter_bridge, component ids in compact slot order)``. A method without it
#: cannot take a common construction.
TARGETS_HOOK = "common_construction_targets"
#: peft's wrapper prefix; a canonical module name is the backbone-relative name after it.
PEFT_INNER = "base_model.model."
#: The newly initialised, non-LoRA tensors copied from the donor.
AUXILIARY_PREFIXES = ("classifier.", "pooler.")
#: Files identifying the pretrained backbone (weights and architecture).
WEIGHT_SUFFIXES = (".bin", ".safetensors")
CONFIG_FILE = "config.json"
#: The donor's canonical adapter configuration (rank 200, alpha 400: scale 2.0).
DONOR_RANK = 200
DONOR_ALPHA = 400


class ConstructionError(ValueError):
    """A common construction that cannot be applied, with the reason."""


def canonical_json(value: Any) -> bytes:
    """UTF-8 canonical JSON: sorted keys, compact separators."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                      ).encode("utf-8")


def sha256_hex(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def validate_spec(spec: Any) -> Dict[str, Any]:
    """``{"mode": "common_canonical_v1", "model_init_seed": int in [0, 2**63)}`` and nothing else."""
    if type(spec) is not dict:
        raise ConstructionError(f"construction must be a plain mapping with keys "
                                f"{list(SPEC_KEYS)}, not {type(spec).__name__}")
    unknown = sorted(set(spec) - set(SPEC_KEYS))
    missing = [key for key in SPEC_KEYS if key not in spec]
    if unknown or missing:
        raise ConstructionError(f"construction must have exactly the keys {list(SPEC_KEYS)}; "
                                f"unknown {unknown}, missing {missing}")
    if spec["mode"] != MODE:
        raise ConstructionError(f"construction.mode {spec['mode']!r} is not {MODE!r}")
    seed = spec["model_init_seed"]
    if type(seed) is not int or seed < 0 or seed >= 2 ** 63:
        raise ConstructionError(f"construction.model_init_seed must be an integer in "
                                f"[0, 2**63), not {seed!r}")
    return {"mode": MODE, "model_init_seed": int(seed)}


def canonical_module(name: str) -> str:
    """The backbone-relative module name: everything after peft's ``base_model.model.``."""
    marker = name.rfind(PEFT_INNER)
    return name[marker + len(PEFT_INNER):] if marker >= 0 else name


def component_seed(schema_id: str, model_init_seed: int, module: str, component_id: int) -> int:
    material = canonical_json({"namespace": NAMESPACE, "schema_id": str(schema_id),
                               "model_init_seed": int(model_init_seed), "module": str(module),
                               "component_id": int(component_id)})
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2 ** 63)


# -- the pinned backbone ---------------------------------------------------------------------

_BACKBONE_CACHE: Dict[Tuple[str, Tuple[Tuple[str, int, int], ...]], Dict[str, Any]] = {}


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def backbone_identity(model_dir: str) -> Dict[str, Any]:
    """The pinned pretrained backbone: ``config.json`` and every weight file, by sha256.

    Cached per process by the directory's real path and its files' sizes and modification
    times, so a changed file is hashed again rather than trusted."""
    root = os.path.realpath(model_dir)
    if not os.path.isdir(root):
        raise ConstructionError(f"{model_dir!r} is not a directory; {MODE} identifies its "
                                f"backbone by the pinned pretrained files")
    names = sorted(name for name in os.listdir(root)
                   if os.path.isfile(os.path.join(root, name))
                   and (name == CONFIG_FILE or name.endswith(WEIGHT_SUFFIXES)))
    if CONFIG_FILE not in names or not any(n.endswith(WEIGHT_SUFFIXES) for n in names):
        raise ConstructionError(f"{root} holds no {CONFIG_FILE} and weight file pair; the "
                                f"backbone cannot be identified")
    stats = tuple((name, os.path.getsize(os.path.join(root, name)),
                   os.stat(os.path.join(root, name)).st_mtime_ns) for name in names)
    key = (root, stats)
    if key not in _BACKBONE_CACHE:
        files = [{"file": name, "bytes": int(size), "sha256": _file_sha256(os.path.join(root, name))}
                 for name, size, _ in stats]
        _BACKBONE_CACHE[key] = {"path": root, "files": files,
                                "digest": sha256_hex(canonical_json(files))}
    return copy.deepcopy(_BACKBONE_CACHE[key])


def model_directory(cfg) -> str:
    """The backbone directory a GLUE configuration loads: ``model.type`` before ``@``."""
    model_type = str(getattr(getattr(cfg, "model", None), "type", "") or "")
    return model_type.split("@", 1)[0]


# -- the canonical schema ----------------------------------------------------------------------

def schema_document(modules: Iterable[Tuple[str, int, int]], backbone_digest: str,
                    dtype: str) -> Dict[str, Any]:
    """Everything the schema id identifies - and nothing it must not (see the module docstring)."""
    rows = sorted((str(m), int(i), int(o)) for m, i, o in modules)
    names = [row[0] for row in rows]
    if len(set(names)) != len(names):
        raise ConstructionError(f"two target modules share a canonical name: {names}")
    return {"init_version": INIT_VERSION, "backbone": str(backbone_digest),
            "modules": [list(row) for row in rows], "dtype": str(dtype),
            "effective_scale": EFFECTIVE_SCALE, "rank_universe": RANK_UNIVERSE}


def schema_id(document: Mapping[str, Any]) -> str:
    return sha256_hex(canonical_json(dict(document)))


_ROW_CACHE: Dict[Tuple[str, int, str, int], torch.Tensor] = {}


@dataclass(frozen=True)
class CanonicalInit:
    """The canonical initial values of one schema under one model-init seed.

    Frozen and cheap: every peer binds its own. ``a_row``/``b_column`` hand out copies."""

    schema_id: str
    model_init_seed: int
    dims: Tuple[Tuple[str, int, int], ...]

    def _dims(self, module: str) -> Tuple[int, int]:
        name = canonical_module(module)
        for row in self.dims:
            if row[0] == name:
                return int(row[1]), int(row[2])
        raise ConstructionError(f"module {module!r} ({name!r}) is not in canonical schema "
                                f"{self.schema_id[:16]}")

    def a_row(self, module: str, component: int) -> torch.Tensor:
        """``A[module][component, :]``, float32, on the CPU - a fresh copy every call."""
        component = int(component)
        if component not in range(RANK_UNIVERSE):
            raise ConstructionError(f"component {component} outside 0..{RANK_UNIVERSE - 1}")
        name = canonical_module(module)
        d_in, _ = self._dims(name)
        key = (self.schema_id, int(self.model_init_seed), name, component)
        cached = _ROW_CACHE.get(key)
        if cached is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(component_seed(self.schema_id, self.model_init_seed, name,
                                                 component))
            raw = torch.rand(d_in, generator=generator, dtype=torch.float32)
            cached = (raw * 2 - 1) * (1.0 / math.sqrt(d_in))
            _ROW_CACHE[key] = cached
        return cached.clone()

    def b_column(self, module: str, component: int) -> torch.Tensor:
        if int(component) not in range(RANK_UNIVERSE):
            raise ConstructionError(f"component {component} outside 0..{RANK_UNIVERSE - 1}")
        _, d_out = self._dims(module)
        return torch.zeros(d_out, dtype=torch.float32)


# -- the auxiliary-state donor -----------------------------------------------------------------

_DONOR_CACHE: Dict[Tuple[Any, ...], Dict[str, torch.Tensor]] = {}


def _donor_config(cfg):
    """The run's configuration at the canonical rank-200 adapter (alpha 400, scale 2.0)."""
    donor = cfg.clone()
    if hasattr(donor, "defrost"):
        donor.defrost()
    args = copy.deepcopy(list(donor.glue.adapter.args))
    if len(args) != 1 or not isinstance(args[0], dict):
        raise ConstructionError("the donor needs exactly one adapter argument mapping")
    args[0]["r"], args[0]["lora_alpha"] = DONOR_RANK, DONOR_ALPHA
    donor.merge_from_list(["glue.adapter.args", args, "glue.adapter.max_rank", DONOR_RANK])
    return donor


def _donor_key(cfg, backbone_digest: str, seed: int) -> Tuple[Any, ...]:
    args = list(cfg.glue.adapter.args)[0] if list(cfg.glue.adapter.args) else {}
    return (str(backbone_digest), int(seed), str(cfg.model.type),
            int(getattr(cfg.model, "out_channels", 0) or 0),
            json.dumps(sorted(args.get("target_modules") or []), sort_keys=True),
            float(args.get("lora_dropout", 0.0)), int(getattr(cfg, "device", -1)),
            torch.__version__)


def auxiliary_name(name: str) -> Optional[str]:
    """``classifier.*`` / ``pooler.*`` relative to the backbone, or None for anything else."""
    relative = canonical_module(name)
    return relative if relative.startswith(AUXILIARY_PREFIXES) else None


def donor_auxiliary(cfg, seed: int, backbone_digest: str, *, builder=None
                    ) -> Dict[str, torch.Tensor]:
    """The donor's classifier and pooler tensors (parameters and buffers), copied.

    Built once per key, through ``get_model`` (the constructor peers are built with) at the
    canonical rank-200 configuration, with the CPU generator seeded by the model-init seed
    inside a scope that restores the CPU torch, python and numpy states afterwards - so
    construction consumes nothing from any stream the run trains with. Only the CPU generator
    is seeded: the donor is built on the host, and ``torch.manual_seed`` would also queue a
    CUDA reseed."""
    key = _donor_key(cfg, backbone_digest, seed)
    if key not in _DONOR_CACHE:
        import numpy as np

        donor_cfg = _donor_config(cfg)
        if int(getattr(donor_cfg, "device", -1)) >= 0:
            raise ConstructionError("the donor is built on the host; cfg.device must be -1")
        if builder is None:
            from federatedscope.core.auxiliaries.model_builder import get_model

            def builder(c):
                return get_model(c, local_data=None)
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(int(seed))
                model = builder(donor_cfg)
                tensors = {}
                for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
                    relative = auxiliary_name(name)
                    if relative is not None:
                        if relative in tensors:
                            raise ConstructionError(f"the donor holds two tensors named {relative}")
                        tensors[relative] = tensor.detach().to("cpu").clone()
                del model
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        if not tensors:
            raise ConstructionError("the donor holds no classifier or pooler tensor")
        _DONOR_CACHE[key] = tensors
    return {name: tensor.clone() for name, tensor in _DONOR_CACHE[key].items()}


# -- one run's construction ------------------------------------------------------------------------

@dataclass(frozen=True)
class CommonConstruction:
    """One run's ``common_canonical_v1``: the specification and the backbone identity.

    Frozen and cheap; ``construct_peer`` makes one per peer, so no peer shares one."""

    model_init_seed: int
    backbone_digest: str
    backbone_path: str

    @property
    def spec(self) -> Dict[str, Any]:
        return {"mode": MODE, "model_init_seed": int(self.model_init_seed)}

    def bind(self, lora_schema) -> CanonicalInit:
        """The canonical values of ``lora_schema``'s modules (rank-independent: only their
        canonical names and dimensions enter)."""
        modules = [(canonical_module(spec.module), int(spec.in_features), int(spec.out_features))
                   for spec in lora_schema.modules]
        document = schema_document(modules, self.backbone_digest, lora_schema.dtype)
        return CanonicalInit(schema_id=schema_id(document), model_init_seed=int(self.model_init_seed),
                             dims=tuple(tuple(row) for row in document["modules"]))

    def schema_document(self, lora_schema) -> Dict[str, Any]:
        modules = [(canonical_module(spec.module), int(spec.in_features), int(spec.out_features))
                   for spec in lora_schema.modules]
        return schema_document(modules, self.backbone_digest, lora_schema.dtype)

    def apply(self, algorithm, context, *, who: str, donor_builder=None) -> None:
        """Write the canonical values into ``algorithm``'s built model.

        Called by ``construct_peer`` right after the method's ``create_peer``, before any
        optimizer exists (the trainer builds its optimizer at the first fit)."""
        hook = getattr(algorithm, TARGETS_HOOK, None)
        if not callable(hook):
            raise ConstructionError(f"{who}: {type(algorithm).__name__} does not implement {MODE} "
                                    f"(no {TARGETS_HOOK}); a common construction it cannot apply "
                                    f"is refused rather than silently skipped")
        bridge, ids = hook()
        ids = tuple(int(c) for c in ids)
        if len(set(ids)) != len(ids):
            raise ConstructionError(f"{who}: repeated component ids {ids}")
        init = self.bind(bridge.schema)
        live = bridge.live_scale()
        if abs(live - EFFECTIVE_SCALE) > 1e-12:
            raise ConstructionError(f"{who}: the adapter applies scale {live}, not the canonical "
                                    f"{EFFECTIVE_SCALE}")
        trainer = getattr(algorithm, "trainer", None)
        if getattr(trainer, "optimizer", None) is not None:
            raise ConstructionError(f"{who}: an optimizer exists before the initial tensors are "
                                    f"final")
        with torch.no_grad():
            for spec in bridge.schema.modules:
                a = bridge._params[spec.a_key]
                b = bridge._params[spec.b_key]
                if tuple(a.shape) != (len(ids), int(spec.in_features)) or \
                        tuple(b.shape) != (int(spec.out_features), len(ids)):
                    raise ConstructionError(f"{who}: {spec.module} holds A {tuple(a.shape)} and B "
                                            f"{tuple(b.shape)} for {len(ids)} components")
                rows = torch.stack([init.a_row(spec.module, c) for c in ids])
                a.copy_(rows.to(device=a.device, dtype=a.dtype))
                b.zero_()
        model = getattr(algorithm, "model", None)
        named = list(model.named_parameters()) + list(model.named_buffers())
        if not any(auxiliary_name(name) is not None for name, _ in named):
            return          # no classifier or pooler: nothing to copy from a donor
        auxiliary = donor_auxiliary(context.config, self.model_init_seed, self.backbone_digest,
                                    builder=donor_builder)
        seen = set()
        with torch.no_grad():
            for name, tensor in named:
                relative = auxiliary_name(name)
                if relative is None:
                    continue
                if relative not in auxiliary:
                    raise ConstructionError(f"{who}: {relative} has no donor value")
                value = auxiliary[relative]
                if tuple(value.shape) != tuple(tensor.shape) or value.dtype != tensor.dtype:
                    raise ConstructionError(f"{who}: {relative} is {tuple(tensor.shape)} "
                                            f"{tensor.dtype}; the donor's is {tuple(value.shape)} "
                                            f"{value.dtype}")
                tensor.copy_(value.to(tensor.device))
                seen.add(relative)
        if seen != set(auxiliary):
            raise ConstructionError(f"{who}: donor tensors {sorted(set(auxiliary) - seen)} have "
                                    f"no place in this model")


def construction_for(spec: Mapping[str, Any], cfg) -> CommonConstruction:
    """The run's construction for ``cfg``'s backbone; refused if it has no pinned directory."""
    spec = validate_spec(dict(spec))
    identity = backbone_identity(model_directory(cfg))
    return CommonConstruction(model_init_seed=spec["model_init_seed"],
                              backbone_digest=identity["digest"], backbone_path=identity["path"])
