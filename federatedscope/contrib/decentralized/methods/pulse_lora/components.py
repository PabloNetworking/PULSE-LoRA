"""The canonical component space: PULSE-LoRA's identity for a rank-1 slice.

A canonical component ``p`` is flat across the adapter: the ``p``-th row of
every LoRA ``A`` and the ``p``-th column of every matching ``B``, over all
injected modules. It is one component across all layers, not one per layer.

The complete ``A``/``B`` block across every injected module is the atomic unit.
``A`` is never selected, transmitted, mixed, or committed without its paired ``B``.

The space is derived from the shared :class:`LoRASchema` and does not replace it,
so Dec-LoRA and other common-rank methods do not depend on it.
"""
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

from federatedscope.contrib.decentralized.lora.schema import LoRASchema, SchemaError

#: The two sides of a component block: its row of ``A`` and its column of ``B``.
SIDE_A = "A"
SIDE_B = "B"


class ComponentError(ValueError):
    pass


@dataclass(frozen=True)
class CanonicalComponentSchema:
    """Immutable component identity shared by every peer in a run."""

    lora: LoRASchema
    r_max: int
    scale: float
    #: ``component_seed``: part of the identity digest, and the seed of the
    #: disagreement sketch and of the selector's draws. Initial values come from
    #: ``init_rule``, not from this seed.
    seed: int
    #: The run's ``lora.canonical_init.CanonicalInit``: an untouched component's value is the
    #: canonical value every peer's active rows were constructed with. Not part of the
    #: identity digest or of equality.
    init_rule: Optional[Any] = field(default=None, compare=False, repr=False)

    @property
    def component_ids(self) -> Tuple[int, ...]:
        """``P = {0, ..., r_max - 1}``. Immutable for the life of the run."""
        return tuple(range(self.r_max))

    @property
    def modules(self):
        return self.lora.modules

    def identity_digest(self) -> str:
        """Peers must agree on this before any component is exchanged."""
        # structural_hash, not compatibility_hash: the latter covers the active rank
        # and scale, which differ between PULSE peers that share one namespace.
        payload = f"{self.lora.structural_hash}:{self.r_max}:{self.scale}:{self.seed}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # -- initial values of untouched components -----------------------------

    def initial_slice(self, module: str, side: str, component: int) -> torch.Tensor:
        """The value of an untouched component, computed on demand rather than stored.

        ``A`` comes from the run's common construction (``init_rule``), so peers
        that have never touched component ``p`` derive the same value for it, and
        mixing two such values leaves them unchanged.

        ``B`` is zero, as in the common construction, so an untouched component
        contributes nothing to the forward pass.
        """
        spec = self._spec(module)
        if component not in range(self.r_max):
            raise ComponentError(
                f"component {component} outside P = 0..{self.r_max - 1}")
        if side == SIDE_B:
            return torch.zeros(spec.out_features, dtype=self._dtype())
        if side != SIDE_A:
            raise ComponentError(f"unknown tensor side {side!r}")
        if self.init_rule is None:
            raise ComponentError("the component space has no initialisation rule; build it "
                                 "with the run's common construction")
        return self.init_rule.a_row(module, component).to(self._dtype())

    def initial_component(self, component: int) -> Dict[str, torch.Tensor]:
        """The complete cross-layer A/B block for one untouched component."""
        out: Dict[str, torch.Tensor] = {}
        for spec in self.lora.modules:
            out[f"{spec.module}|{SIDE_A}"] = self.initial_slice(spec.module, SIDE_A, component)
            out[f"{spec.module}|{SIDE_B}"] = self.initial_slice(spec.module, SIDE_B, component)
        return out

    # -- cost ----------------------------------------------------------------

    def component_numel(self) -> int:
        return sum(spec.in_features + spec.out_features for spec in self.lora.modules)

    def component_cost_bytes(self, identifier_bytes: int = 4) -> int:
        """Value bytes of one complete component plus a fixed-size identifier.

        Every ``A`` and ``B`` slice across all injected modules, plus
        ``identifier_bytes``. This is a lower bound on the wire cost: an encoded
        unit also carries keys and entry headers, and ``units.unit_cost_bytes``
        gives its exact size.
        """
        return self.component_numel() * self._element_size() + identifier_bytes

    # -- helpers ---------------------------------------------------------------

    def _spec(self, module: str):
        for spec in self.lora.modules:
            if spec.module == module:
                return spec
        raise ComponentError(f"unknown injected module {module!r}")

    def _dtype(self) -> torch.dtype:
        return getattr(torch, self.lora.dtype.replace("torch.", ""))

    def _element_size(self) -> int:
        return torch.empty(0, dtype=self._dtype()).element_size()


def build_component_schema(lora: LoRASchema, r_max: int, seed: int,
                           scale: float = None, init_rule=None) -> CanonicalComponentSchema:
    """Derive the component space from a validated shared LoRA schema.

    ``r_max`` must be at least the adapter's own rank: components are addressed
    by a single flat index, so a peer cannot hold a rank the namespace does not
    cover.
    """
    if r_max <= 0:
        raise ComponentError(f"r_max must be positive, got {r_max}")
    if r_max < lora.rank:
        raise ComponentError(
            f"r_max={r_max} is smaller than the adapter's rank {lora.rank}; the "
            f"canonical namespace must cover every component a peer can hold")
    if scale is None:
        # The fixed scale is a run-level constant, so it is derived from r_max and
        # never from the peer's active rank: `lora.scaling` is peft's
        # alpha / current_rank, which would give peers of different rank
        # different scales.
        if lora.lora_alpha is None:
            raise SchemaError(
                "cannot derive a fixed scale: the adapter reports no lora_alpha")
        scale = float(lora.lora_alpha) / float(r_max)
    if scale <= 0:
        raise ComponentError(f"the fixed LoRA scale must be positive, got {scale}")
    return CanonicalComponentSchema(lora=lora, r_max=int(r_max), scale=float(scale),
                                    seed=int(seed), init_rule=init_rule)
