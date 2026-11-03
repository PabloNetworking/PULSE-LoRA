"""Shared LoRA compatibility schema.

Each peer builds it from its own PEFT model; its hashes are what partners and
checkpoints are checked against. Modules are ordered by full name rather than by
registration order, so the hashes do not depend on how the model was assembled.

The schema records alpha, rank and the resulting scale explicitly. peft 0.4.0
computes ``scaling = lora_alpha / r`` inside ``update_layer`` and reallocates the
factor modules, so a rank change rescales the adapter. ``compatibility_hash``
covers rank and scale, for methods that need one common rank;
``structural_hash`` leaves them out, for methods whose peers hold different
ranks.
"""
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

logger = logging.getLogger(__name__)

LORA_A_MARKER = ".lora_A."
LORA_B_MARKER = ".lora_B."

#: peft's ModulesToSaveWrapper keeps two copies of a saved head: the live
#: ``modules_to_save.<adapter>.*`` used by the forward pass, and
#: ``original_module.*``, the pre-adapter copy. In peft 0.4.0 both are left with
#: requires_grad=True even though only the first participates in the forward
#: pass. Communicating the shadow would double head traffic and mix a tensor
#: that affects nothing, so it is excluded.
SHADOW_MODULE_MARKER = ".original_module."


class SchemaError(ValueError):
    """The adapter does not match the shared compatibility contract."""


@dataclass(frozen=True)
class LoRAModuleSpec:
    module: str
    a_key: str
    b_key: str
    in_features: int
    out_features: int
    rank: int
    dtype: str
    scaling: Optional[float] = None


@dataclass(frozen=True)
class LoRASchema:
    adapter_name: str
    modules: Tuple[LoRAModuleSpec, ...]
    rank: int
    lora_alpha: Optional[float]
    scaling: Optional[float]
    dtype: str
    auxiliary_keys: Tuple[str, ...]
    shadow_keys: Tuple[str, ...]
    compatibility_hash: str
    #: Identity of the injected structure with rank and scale excluded.
    #: Two peers holding different active ranks over the same backbone and the
    #: same injected modules share this, which lets a heterogeneous-rank method
    #: define a component namespace they both agree on. ``compatibility_hash``
    #: includes the rank, because a common-rank method must reject a mismatch.
    structural_hash: str

    def lora_keys(self) -> Tuple[str, ...]:
        keys: List[str] = []
        for spec in self.modules:
            keys.append(spec.a_key)
            keys.append(spec.b_key)
        return tuple(keys)

    def payload_keys(self, include_auxiliary: bool) -> FrozenSet[str]:
        keys = set(self.lora_keys())
        if include_auxiliary:
            keys |= set(self.auxiliary_keys)
        return frozenset(keys)

    def shape_of(self, key: str) -> Tuple[int, ...]:
        for spec in self.modules:
            if key == spec.a_key:
                return (spec.rank, spec.in_features)
            if key == spec.b_key:
                return (spec.out_features, spec.rank)
        raise KeyError(key)

    def is_compatible_with(self, other: "LoRASchema") -> bool:
        return self.compatibility_hash == other.compatibility_hash

    def require_compatible(self, other: "LoRASchema") -> None:
        if not self.is_compatible_with(other):
            raise SchemaError(
                f"incompatible LoRA schemas: {self.compatibility_hash[:12]}... vs "
                f"{other.compatibility_hash[:12]}...; peers must share backbone, "
                f"injected modules, rank, dtype and scale"
            )


def _module_scaling(model, module_name: str, adapter: str) -> Optional[float]:
    for name, module in model.named_modules():
        if name.endswith(module_name) and hasattr(module, "scaling"):
            scaling = getattr(module, "scaling")
            if isinstance(scaling, dict) and adapter in scaling:
                return float(scaling[adapter])
    return None


def build_schema(model, include_auxiliary: bool = True) -> LoRASchema:
    """Derive the shared schema from a live PEFT model.

    Rejects a model with no trainable LoRA factors, more than one adapter,
    incomplete A/B pairs, asymmetric A/B ranks, mixed ranks or dtypes across
    modules, or modules that disagree on scale: any of these would make
    full-state exchange lossy. With ``include_auxiliary=False`` the schema
    lists no auxiliary keys.
    """
    a_parts: Dict[str, object] = {}
    b_parts: Dict[str, object] = {}
    adapters = set()
    auxiliary: List[str] = []
    shadowed: List[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if LORA_A_MARKER in name:
            module, rest = name.split(LORA_A_MARKER, 1)
            adapters.add(rest.rsplit(".", 1)[0])
            a_parts[module] = (name, param)
        elif LORA_B_MARKER in name:
            module, rest = name.split(LORA_B_MARKER, 1)
            adapters.add(rest.rsplit(".", 1)[0])
            b_parts[module] = (name, param)
        elif SHADOW_MODULE_MARKER in name:
            shadowed.append(name)
        else:
            auxiliary.append(name)

    if not a_parts:
        raise SchemaError("no trainable LoRA factors found on the model")
    if len(adapters) != 1:
        raise SchemaError(f"expected exactly one adapter name, found {sorted(adapters)}")
    adapter_name = adapters.pop()

    missing_b = sorted(set(a_parts) - set(b_parts))
    missing_a = sorted(set(b_parts) - set(a_parts))
    if missing_b or missing_a:
        raise SchemaError(
            f"incomplete A/B pairs; modules missing B: {missing_b}, missing A: {missing_a}"
        )

    specs: List[LoRAModuleSpec] = []
    ranks = set()
    dtypes = set()
    for module in sorted(a_parts):  # canonical full name order, not insertion order
        a_name, a_param = a_parts[module]
        b_name, b_param = b_parts[module]
        a_rank, in_features = tuple(a_param.shape)
        out_features, b_rank = tuple(b_param.shape)
        if a_rank != b_rank:
            raise SchemaError(
                f"{module}: asymmetric A/B rank ({a_rank} vs {b_rank})"
            )
        ranks.add(int(a_rank))
        dtypes.add(str(a_param.dtype))
        dtypes.add(str(b_param.dtype))
        specs.append(LoRAModuleSpec(
            module=module, a_key=a_name, b_key=b_name,
            in_features=int(in_features), out_features=int(out_features),
            rank=int(a_rank), dtype=str(a_param.dtype),
            scaling=_module_scaling(model, module, adapter_name),
        ))

    if len(ranks) != 1:
        raise SchemaError(f"mixed LoRA ranks across modules: {sorted(ranks)}")
    if len(dtypes) != 1:
        raise SchemaError(f"mixed LoRA dtypes across modules: {sorted(dtypes)}")

    scalings = {spec.scaling for spec in specs if spec.scaling is not None}
    if len(scalings) > 1:
        raise SchemaError(f"modules disagree on LoRA scale: {sorted(scalings)}")
    scaling = scalings.pop() if scalings else None
    rank = ranks.pop()
    lora_alpha = scaling * rank if scaling is not None else None

    auxiliary_keys = tuple(sorted(auxiliary)) if include_auxiliary else ()
    shadow_keys = tuple(sorted(shadowed))
    if shadow_keys:
        logger.debug(
            "excluding %d peft original_module shadow parameters from the payload: %s",
            len(shadow_keys), ", ".join(shadow_keys))

    fingerprint = {
        "adapter": adapter_name,
        "rank": rank,
        "dtype": sorted(dtypes)[0],
        "scaling": scaling,
        "modules": [
            [s.module, s.a_key, s.b_key, s.in_features, s.out_features, s.rank]
            for s in specs
        ],
        "auxiliary": list(auxiliary_keys),
        "shadow": list(shadow_keys),
    }
    blob = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    compatibility_hash = hashlib.sha256(blob.encode("utf-8")).hexdigest()

    structural = {
        "adapter": adapter_name,
        "dtype": sorted(dtypes)[0],
        "modules": [[s.module, s.a_key, s.b_key, s.in_features, s.out_features]
                    for s in specs],
        "auxiliary": list(auxiliary_keys),
    }
    structural_hash = hashlib.sha256(
        json.dumps(structural, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return LoRASchema(
        adapter_name=adapter_name,
        modules=tuple(specs),
        rank=rank,
        lora_alpha=lora_alpha,
        scaling=scaling,
        dtype=sorted(dtypes)[0],
        auxiliary_keys=auxiliary_keys,
        shadow_keys=shadow_keys,
        compatibility_hash=compatibility_hash,
        structural_hash=structural_hash,
    )
