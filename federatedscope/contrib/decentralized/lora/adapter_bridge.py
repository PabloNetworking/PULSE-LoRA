"""Validated access to a PEFT adapter's factors.

Writes do not go through ``model.load_state_dict``:
``GLUEAdapterModel.load_state_dict`` ignores the caller's ``strict`` argument and
always passes ``strict=False`` to the inner model, so keys that do not match
would be dropped without an error. The bridge resolves the target parameters
once, checks the incoming keys, shapes and dtypes against the expected set, and
copies in place.
"""
from collections import OrderedDict
from typing import Dict, Iterable, Mapping, Optional

import torch

from federatedscope.contrib.decentralized.lora.schema import LoRASchema, build_schema


class BridgeError(ValueError):
    pass


class AdapterBridge:
    def __init__(self, model, schema: Optional[LoRASchema] = None):
        self.model = model
        self.schema = schema or build_schema(model)
        self._params: Dict[str, torch.nn.Parameter] = {
            name: param for name, param in model.named_parameters()
        }
        missing = [k for k in self.schema.payload_keys(True) if k not in self._params]
        if missing:
            raise BridgeError(f"schema references parameters absent from the model: {missing}")

    # -- read ---------------------------------------------------------------

    def read(self, keys: Optional[Iterable[str]] = None, detach: bool = True
             ) -> "OrderedDict[str, torch.Tensor]":
        selected = sorted(keys) if keys is not None else sorted(self.schema.payload_keys(True))
        out = OrderedDict()
        for key in selected:
            if key not in self._params:
                raise BridgeError(f"unknown parameter {key!r}")
            tensor = self._params[key]
            out[key] = tensor.detach().clone() if detach else tensor
        return out

    def parameters(self, keys: Iterable[str]):
        return [self._params[key] for key in sorted(keys)]

    # -- write ----------------------------------------------------------------

    def expected_adapter_keys(self, include_auxiliary: bool = True) -> frozenset:
        """The adapter's key set as the live model defines it.

        Taken from the model, not from the incoming mapping, so a checkpoint
        that lacks a tensor cannot define its own reduced expected set.
        """
        return self.schema.payload_keys(include_auxiliary)

    def adapter_contract(self, rank: Optional[int] = None,
                         include_auxiliary: bool = True) -> Dict[str, tuple]:
        """``{key: (shape, dtype)}`` for a target rank, without resizing anything.

        ``rank`` describes the adapter a caller is about to restore: a method
        whose rank changes validates a checkpoint against the shape the model
        will have, before it resizes anything.

        Shapes come from the live parameters, so with no ``rank`` this describes
        the model as it is; only the rank dimension is substituted. Callers check
        the structural hash first, which fixes the other dimensions.
        """
        contract: Dict[str, tuple] = {}
        for spec in self.schema.modules:
            a, b = self._params[spec.a_key], self._params[spec.b_key]
            a_shape, b_shape = list(a.shape), list(b.shape)   # (r, in), (out, r)
            if rank is not None:
                a_shape[0] = b_shape[-1] = int(rank)
            contract[spec.a_key] = (tuple(a_shape), a.dtype)
            contract[spec.b_key] = (tuple(b_shape), b.dtype)
        if include_auxiliary:
            for key in self.schema.auxiliary_keys:
                param = self._params[key]
                contract[key] = (tuple(param.shape), param.dtype)
        return contract

    def validate_state(self, state: Mapping[str, torch.Tensor],
                       contract: Mapping[str, tuple], allow_cast: bool = False) -> None:
        """Check a mapping against a contract. Raises, never writes."""
        expected = frozenset(contract)
        provided = frozenset(state)
        if provided != expected:
            missing = sorted(expected - provided)
            extra = sorted(provided - expected)
            raise BridgeError(
                f"payload key set does not match the expected key set; "
                f"missing={missing}, unexpected={extra}"
            )
        for key in sorted(expected):
            shape, dtype = contract[key]
            value = state[key]
            if not torch.is_tensor(value):
                raise BridgeError(
                    f"{key}: payload entry is {type(value).__name__}, not a tensor")
            if tuple(value.shape) != tuple(shape):
                raise BridgeError(
                    f"{key}: shape mismatch, payload {tuple(value.shape)} vs expected "
                    f"{tuple(shape)}")
            if not allow_cast and value.dtype != dtype:
                raise BridgeError(
                    f"{key}: dtype mismatch, payload {value.dtype} vs expected {dtype}. "
                    f"Casting silently would change what the checkpoint means and what "
                    f"it costs on the wire.")

    def write(self, state: Mapping[str, torch.Tensor], expected_keys: Iterable[str],
              allow_cast: bool = False) -> None:
        """Validate ``state`` against exactly ``expected_keys``, then write it in place."""
        expected = frozenset(expected_keys)
        full = self.adapter_contract()
        unknown = sorted(k for k in expected if k not in full)
        if unknown:
            raise BridgeError(f"expected keys name parameters the model lacks: {unknown}")
        self.validate_state(state, {k: full[k] for k in expected}, allow_cast=allow_cast)

        with torch.no_grad():
            for key in sorted(expected):
                # param.copy_, not param.data.copy_, so autograd bookkeeping and
                # the version counter stay correct.
                self._params[key].copy_(state[key].to(self._params[key].dtype))

    def component_contribution(self, module: str, position: int) -> "torch.Tensor":
        """``live_scale * outer(B[:, position], A[position, :])`` for one slot."""
        spec = next((s for s in self.schema.modules if s.module == module), None)
        if spec is None:
            raise BridgeError(f"unknown module {module!r}")
        a = self._params[spec.a_key][position, :].detach()
        b = self._params[spec.b_key][:, position].detach()
        return self.live_scale() * torch.outer(b, a)

    def refresh_schema(self) -> LoRASchema:
        """Re-derive the schema after a method has changed the adapter's rank.

        Resizing the factor tensors makes the cached shapes and the parameter
        map stale, so both are rebuilt from the model.
        """
        self.schema = build_schema(self.model)
        self._params = {name: param for name, param in self.model.named_parameters()}
        # Keep the cached scale equal to the scale the live modules apply.
        try:
            self._sync_schema_scale(self.live_scale())
        except BridgeError:
            pass
        return self.schema

    # -- scale ---------------------------------------------------------------

    def resolve_modules(self) -> Dict[str, object]:
        """Map each schema module to exactly one live PEFT module.

        Resolution is by full module name: leaf names such as ``query`` repeat
        in every transformer layer, so matching on them would hide a scale
        disagreement between layers. A missing or ambiguous module raises.
        """
        by_name = {}
        for name, module in self.model.named_modules():
            by_name.setdefault(name, []).append(module)

        resolved: Dict[str, object] = {}
        missing, ambiguous = [], []
        for spec in self.schema.modules:
            candidates = by_name.get(spec.module, [])
            if not candidates:
                missing.append(spec.module)
            elif len(candidates) > 1:
                ambiguous.append(spec.module)
            else:
                resolved[spec.module] = candidates[0]

        if missing or ambiguous:
            raise BridgeError(
                f"could not resolve every injected module one-to-one; "
                f"missing={missing}, ambiguous={ambiguous}")
        return resolved

    def current_scales(self) -> Dict[str, float]:
        """The scale each injected module currently applies, one entry each."""
        adapter = self.schema.adapter_name
        scales: Dict[str, float] = {}
        for module_name, module in self.resolve_modules().items():
            scaling = getattr(module, "scaling", None)
            if not isinstance(scaling, dict) or adapter not in scaling:
                raise BridgeError(
                    f"module {module_name} exposes no scale for adapter {adapter!r}")
            scales[module_name] = float(scaling[adapter])
        return scales

    def live_scale(self) -> float:
        """The scale the forward pass currently applies, read from the live modules.

        The cached schema goes stale when a method pins a scale or peft rebuilds
        a layer, so diagnostics read this instead. Raises if the modules
        disagree.
        """
        scales = set(self.current_scales().values())
        if not scales:
            raise BridgeError("no injected module exposed a scale")
        if len(scales) > 1:
            raise BridgeError(f"injected modules disagree on scale: {sorted(scales)}")
        return float(scales.pop())

    def _sync_schema_scale(self, scale: float) -> None:
        """Keep the cached metadata consistent with the live model."""
        import dataclasses

        self.schema = dataclasses.replace(
            self.schema,
            scaling=float(scale),
            modules=tuple(dataclasses.replace(spec, scaling=float(scale))
                          for spec in self.schema.modules),
        )

    def enforce_fixed_scale(self, scale: float) -> int:
        """Pin every injected module to one scale, independent of active rank.

        peft recomputes ``scaling = lora_alpha / r`` whenever a layer is rebuilt,
        so without a pinned scale the same canonical component would contribute
        differently at different ranks. A method that needs a rank-independent
        scale calls this after every rebuild. Returns how many modules were
        pinned.
        """
        adapter = self.schema.adapter_name
        pinned = 0
        for _name, module in self.model.named_modules():
            scaling = getattr(module, "scaling", None)
            if isinstance(scaling, dict) and adapter in scaling:
                scaling[adapter] = float(scale)
                pinned += 1
        if pinned == 0:
            raise BridgeError("no injected module exposed a scale to pin")
        # Keep the cached schema equal to the pinned scale, so it never reports
        # alpha/r while the forward pass applies another scale.
        self._sync_schema_scale(scale)
        return pinned

    def assert_fixed_scale(self, scale: float, tolerance: float = 1e-9) -> None:
        offenders = {module: value for module, value in self.current_scales().items()
                     if abs(value - scale) > tolerance}
        if offenders:
            raise BridgeError(
                f"modules are not holding the required fixed scale {scale}: {offenders}")

    # -- diagnostics ------------------------------------------------------------

    def effective_update(self, module: str) -> torch.Tensor:
        """``scaling * B @ A`` for one injected module, the quantity a rebuild
        must preserve."""
        spec = next((s for s in self.schema.modules if s.module == module), None)
        if spec is None:
            raise BridgeError(f"unknown module {module!r}")
        a = self._params[spec.a_key]
        b = self._params[spec.b_key]
        # The live scale, not the cached one, so this describes the model as it runs.
        return self.live_scale() * (b.detach() @ a.detach())

    def effective_updates(self) -> "OrderedDict[str, torch.Tensor]":
        return OrderedDict((s.module, self.effective_update(s.module))
                           for s in self.schema.modules)
