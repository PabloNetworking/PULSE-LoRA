"""PULSE's two-plane peer state.

Each peer holds:

``S_i``  the trainable support: the components in its forward pass, gradients
         and optimizer, held in the compact PEFT factors.
``M_i``  the materialised components: the active ones plus the inactive ones
         held in CPU relay storage.

Every other component has the initial value of the run's common construction,
computed on demand; nothing is stored for it (lazy materialisation).

The invariant is ``S_i ⊆ M_i``, and no component has two independently mutable
authoritative copies: a component is either active (owned by the model) or
inactive and materialised (owned by relay storage), never both.

Pruning is monotone. Components are not regrown and relay state is not evicted.
"""
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from federatedscope.contrib.decentralized.lora.adapter_bridge import AdapterBridge
from federatedscope.contrib.decentralized.methods.pulse_lora.components import (
    SIDE_A,
    SIDE_B,
    CanonicalComponentSchema,
    ComponentError,
)


#: What relay storage keeps when an exchange writes an inactive component: the mixed
#: value (``carried``), or the component's initial value (``scrubbed``, an ablation that
#: keeps relay components materialised, eligible and priced but drops what they receive).
RELAY_CARRIED = "carried"
RELAY_SCRUBBED = "scrubbed"
RELAY_CONTENT = (RELAY_CARRIED, RELAY_SCRUBBED)


class StateError(ValueError):
    pass


def _slot(module: str, side: str) -> str:
    return f"{module}|{side}"


class PulsePeerState:
    """Owns every tensor and mapping specific to one PULSE learner."""

    def __init__(self, bridge: AdapterBridge, schema: CanonicalComponentSchema,
                 support: Optional[Sequence[int]] = None,
                 relay_content: str = RELAY_CARRIED):
        self.bridge = bridge
        self.schema = schema
        if relay_content not in RELAY_CONTENT:
            raise StateError(
                f"relay_content {relay_content!r} is not one of {list(RELAY_CONTENT)}")
        self.relay_content = relay_content

        rank = bridge.schema.rank
        if support is None:
            support = list(range(rank))
        support = [int(c) for c in support]
        if len(set(support)) != len(support):
            raise StateError(f"duplicate component in the initial support: {support}")
        if len(support) != rank:
            raise StateError(
                f"initial support has {len(support)} components but the adapter holds "
                f"rank {rank}; every active slot must map to exactly one component")
        for component in support:
            if component not in range(schema.r_max):
                raise ComponentError(
                    f"component {component} outside P = 0..{schema.r_max - 1}")

        #: canonical component id -> compact row/column index in the PEFT model
        self._active: Dict[int, int] = {c: position for position, c in enumerate(support)}
        #: inactive materialised components: id -> {module|side: tensor} on CPU
        self._relay: Dict[int, Dict[str, torch.Tensor]] = {}
        # Initially M_i = S_i; untouched components stay lazy and occupy nothing.
        self._pruned_history: List[int] = []
        self.enforce_scale()

    # -- the two planes -------------------------------------------------------

    @property
    def support(self) -> Tuple[int, ...]:
        """``S_i`` in compact order, so index i is compact slot i."""
        return tuple(sorted(self._active, key=lambda c: self._active[c]))

    @property
    def materialised(self) -> Tuple[int, ...]:
        """``M_i``: active components plus those held in relay storage."""
        return tuple(sorted(set(self._active) | set(self._relay)))

    @property
    def relay_components(self) -> Tuple[int, ...]:
        return tuple(sorted(self._relay))

    def is_active(self, component: int) -> bool:
        return component in self._active

    def is_materialised(self, component: int) -> bool:
        return component in self._active or component in self._relay

    def compact_index(self, component: int) -> int:
        if component not in self._active:
            raise StateError(f"component {component} is not active on this peer")
        return self._active[component]

    def check_invariants(self) -> None:
        support = set(self._active)
        if not support <= set(self.materialised):
            raise StateError(f"S_i is not a subset of M_i: {sorted(support)}")
        overlap = support & set(self._relay)
        if overlap:
            raise StateError(
                f"components {sorted(overlap)} are both active and in relay storage; "
                f"a component must have exactly one authoritative copy")
        positions = sorted(self._active.values())
        if positions != list(range(len(positions))):
            raise StateError(
                f"the canonical->compact mapping is not a bijection onto "
                f"0..{len(positions) - 1}: {positions}")
        if self.bridge.schema.rank != len(self._active):
            raise StateError(
                f"the adapter holds rank {self.bridge.schema.rank} but {len(self._active)} "
                f"components are active")

    # -- scale -----------------------------------------------------------------

    def enforce_scale(self) -> None:
        """One fixed scale, whatever the active rank happens to be."""
        self.bridge.enforce_fixed_scale(self.schema.scale)

    # -- gather / scatter -------------------------------------------------------

    def gather(self, component: int) -> Dict[str, torch.Tensor]:
        """The complete cross-layer A/B block for one component.

        Reads from whichever plane owns it, and falls back to the common
        construction's initial value for a component this peer has never touched,
        which costs no storage and agrees with every other peer.
        """
        if component in self._active:
            return self._gather_active(component)
        if component in self._relay:
            return {slot: tensor.clone() for slot, tensor in self._relay[component].items()}
        return self.schema.initial_component(component)

    def _gather_active(self, component: int) -> Dict[str, torch.Tensor]:
        position = self._active[component]
        out: Dict[str, torch.Tensor] = {}
        for spec in self.bridge.schema.modules:
            # Read the live factors without cloning them, then clone only this component's
            # row and column: cloning both whole factors would copy every component's values
            # to keep one row and one column.
            state = self.bridge.read([spec.a_key, spec.b_key], detach=False)
            out[_slot(spec.module, SIDE_A)] = state[spec.a_key][position, :].detach().clone()
            out[_slot(spec.module, SIDE_B)] = state[spec.b_key][:, position].detach().clone()
        return out

    def scatter(self, component: int, block: Mapping[str, torch.Tensor]) -> None:
        """Write a complete component back into whichever plane owns it."""
        self._validate_block(component, block)
        if component in self._active:
            self._scatter_active(component, block)
        else:
            # An inactive component is materialised in relay storage only; it must
            # not enter the forward pass or acquire gradients.
            if self.scrub_relay(component):
                block = self.schema.initial_component(component)
            self._relay[component] = self._to_relay(block)
        self.check_invariants()

    def _scatter_active(self, component: int, block: Mapping[str, torch.Tensor]) -> None:
        position = self._active[component]
        with torch.no_grad():
            for spec in self.bridge.schema.modules:
                a_param = self.bridge._params[spec.a_key]
                b_param = self.bridge._params[spec.b_key]
                a_param[position, :].copy_(block[_slot(spec.module, SIDE_A)])
                b_param[:, position].copy_(block[_slot(spec.module, SIDE_B)])

    def informative_relay_components(self) -> Tuple[int, ...]:
        """Relay components whose stored value differs from their initial value.

        This separates relay that carries content from relay that is only
        materialised. Under ``relay_content='scrubbed'`` every value an exchange
        writes to relay storage is the initial one, so only a component pruned
        since its last exchange can count here (pruning keeps the trained value).
        """
        informative = []
        for component, stored in self._relay.items():
            initial = self.schema.initial_component(component)
            if any(not torch.equal(stored[slot].cpu(), initial[slot].cpu())
                   for slot in initial):
                informative.append(int(component))
        return tuple(sorted(informative))

    def scrub_relay(self, component: int) -> bool:
        """True if a value written to this component must be replaced by its initial value.

        This is the ``relay_content='scrubbed'`` ablation. Relay components stay
        materialised and eligible, and are selected, sent and charged as usual; only
        the content an exchange writes to an inactive component is dropped. Values
        that enter relay storage through :meth:`prune` are kept.

        Unlike ``survivor_only``, which restricts the candidates to ``S_i & S_j`` and so
        also changes eligibility, byte counts and ages, scrubbing changes only the
        stored content.
        """
        return self.relay_content == RELAY_SCRUBBED and component not in self._active

    def materialise(self, component: int, block: Mapping[str, torch.Tensor]) -> None:
        """Take ownership of a received component without activating it."""
        if component in self._active:
            raise StateError(
                f"component {component} is active; materialising it into relay storage "
                f"would create a second authoritative copy")
        self.scatter(component, block)

    def _validate_block(self, component: int, block: Mapping[str, torch.Tensor]) -> None:
        if component not in range(self.schema.r_max):
            raise ComponentError(f"component {component} outside P")
        expected = set()
        for spec in self.bridge.schema.modules:
            expected.add(_slot(spec.module, SIDE_A))
            expected.add(_slot(spec.module, SIDE_B))
        provided = set(block)
        if provided != expected:
            missing = sorted(expected - provided)
            extra = sorted(provided - expected)
            raise StateError(
                f"component {component} block is incomplete; missing={missing}, "
                f"unexpected={extra}. A is never accepted without its paired B, and "
                f"never for a subset of layers.")
        expected_dtype = self.schema._dtype()
        for spec in self.bridge.schema.modules:
            for side, length in ((SIDE_A, spec.in_features), (SIDE_B, spec.out_features)):
                tensor = block[_slot(spec.module, side)]
                if not torch.is_tensor(tensor):
                    raise StateError(
                        f"{spec.module} {side} slice is {type(tensor).__name__}, "
                        f"not a tensor")
                if tuple(tensor.shape) != (length,):
                    raise StateError(
                        f"{spec.module} {side} slice has shape {tuple(tensor.shape)}, "
                        f"expected {(length,)}")
                if tensor.dtype != expected_dtype:
                    # Silently keeping a float64 block under a float32 schema would
                    # make the same component mean different things on different
                    # peers, and would change its wire cost.
                    raise StateError(
                        f"{spec.module} {side} slice has dtype {tensor.dtype}, "
                        f"expected {expected_dtype}")

    def _to_relay(self, block: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inactive storage is always detached, CPU, and the canonical dtype."""
        dtype = self.schema._dtype()
        return {slot: tensor.detach().to(device="cpu", dtype=dtype).clone()
                for slot, tensor in block.items()}

    # -- transactional snapshot / restore -------------------------------------------

    def snapshot_components(self, components: Iterable[int]) -> dict:
        """Everything a mix over these components can change, captured exactly.

        A component's value lives in the compact tensors or in relay storage, and
        which of the two is itself state. The snapshot records relay membership so
        that a rollback can tell "restore the old relay value" from "this component
        was lazy and must become lazy again"; the second cannot be expressed by
        writing a value back.
        """
        components = tuple(sorted(int(c) for c in components))
        return {
            "components": components,
            "blocks": {c: self.gather(c) for c in components},
            "relay_members": frozenset(self._relay),
            "active": dict(self._active),
        }

    def restore_components(self, snapshot: Mapping[str, object]) -> None:
        """Put back exactly what :meth:`snapshot_components` captured."""
        if dict(self._active) != snapshot["active"]:
            raise StateError(
                "the active support changed while a transaction was open; PULSE "
                "mixing never alters support, so this snapshot cannot be applied")
        relay_members = snapshot["relay_members"]
        for component, block in snapshot["blocks"].items():
            if component in self._active:
                self._scatter_active(component, block)
            elif component in relay_members:
                self._relay[component] = self._to_relay(block)
            else:
                # It was lazy before the transaction. Writing the old value back
                # would leave it materialised: a different state from the one
                # that was snapshotted, with a larger M_i.
                self._relay.pop(component, None)
        self.check_invariants()

    def restrict_initial_support(self, support: Sequence[int], optimizer=None) -> None:
        """Narrow a freshly built adapter down to its configured initial support.

        Distinct from :meth:`prune`: the components dropped here were never
        trained, so they must stay *lazy* rather than entering relay storage.
        Initial materialisation is exactly ``M_i = S_i``; anything else would put
        untouched values on the relay plane and inflate M_i from the first frame.
        """
        wanted = [int(c) for c in support]
        if len(set(wanted)) != len(wanted):
            raise StateError(f"duplicate component in the requested support: {wanted}")
        if not wanted:
            raise StateError("the initial support cannot be empty")
        for component in wanted:
            if component not in range(self.schema.r_max):
                raise ComponentError(
                    f"component {component} outside P = 0..{self.schema.r_max - 1}")
        missing = [c for c in wanted if c not in self._active]
        if missing:
            raise StateError(
                f"cannot select components {missing} as the initial support: the "
                f"adapter was built holding {sorted(self._active)}, and components are "
                f"never grown")
        if list(self.support) == wanted:
            return
        self._compact(wanted, optimizer=optimizer)
        self.check_invariants()

    # -- monotone pruning ---------------------------------------------------------

    def prune(self, components: Iterable[int], optimizer=None) -> Tuple[int, ...]:
        """Retire components from the support, keeping their latest values.

        The order matters and is the reason this is one operation: each pruned
        component's current active factors are copied into relay storage *before*
        the compact tensors are rebuilt, so nothing is lost. Pruning never removes
        a component from ``M_i`` - relay eligibility is not a local decision.
        """
        doomed = sorted({int(c) for c in components})
        if not doomed:
            return ()
        unknown = [c for c in doomed if c not in self._active]
        if unknown:
            raise StateError(f"cannot prune components that are not active: {unknown}")
        if len(doomed) >= len(self._active):
            raise StateError(
                f"pruning {len(doomed)} of {len(self._active)} components would leave "
                f"an empty support")

        # 1. preserve the latest active values in relay storage
        preserved = {component: self._gather_active(component) for component in doomed}

        # 2. compact the model down to the survivors, in canonical order
        survivors = [c for c in self.support if c not in set(doomed)]
        self._compact(survivors, optimizer=optimizer)

        # 3. the pruned components are now inactive but still materialised
        for component, block in preserved.items():
            self._relay[component] = self._to_relay(block)

        self._pruned_history.extend(doomed)
        self.check_invariants()
        return tuple(doomed)

    def _compact(self, survivors: Sequence[int], optimizer=None) -> None:
        """Rebuild the compact tensors to hold exactly ``survivors``."""
        keep = [self._active[c] for c in survivors]

        with torch.no_grad():
            for spec in self.bridge.schema.modules:
                a_param = self.bridge._params[spec.a_key]
                b_param = self.bridge._params[spec.b_key]
                # index_select needs the index on the tensor's device, which is a
                # GPU when the peer trains on one.
                new_a = a_param.data.index_select(
                    0, torch.tensor(keep, dtype=torch.long, device=a_param.device)).clone()
                new_b = b_param.data.index_select(
                    1, torch.tensor(keep, dtype=torch.long, device=b_param.device)).clone()
                self._replace_parameter(spec.a_key, new_a, optimizer, keep, dim=0)
                self._replace_parameter(spec.b_key, new_b, optimizer, keep, dim=1)

        self._active = {component: position for position, component in enumerate(survivors)}
        self.bridge.refresh_schema()
        # The scale must not follow the active rank (peft sets alpha/r whenever it
        # rebuilds a layer), so re-pin the fixed scale after every rank change.
        self.enforce_scale()

    def _replace_parameter(self, key: str, new_value: torch.Tensor, optimizer,
                           keep: Sequence[int], dim: int) -> None:
        """Resize a parameter in place and migrate its optimizer state.

        Optimizer state follows the *canonical* components that survive, not the
        raw slot positions: dropping the moments entirely would silently reset
        momentum for components that were never pruned.
        """
        param = self.bridge._params[key]
        old_state = None
        if optimizer is not None and param in optimizer.state:
            old_state = optimizer.state.pop(param)

        param.data = new_value

        if old_state is not None:
            migrated = {}
            for name, value in old_state.items():
                if torch.is_tensor(value) and value.shape and value.dim() > dim:
                    index = torch.tensor(keep, dtype=torch.long, device=value.device)
                    migrated[name] = value.index_select(dim, index).clone()
                else:
                    migrated[name] = value
            optimizer.state[param] = migrated

    # -- checkpointing --------------------------------------------------------------

    def state_dict(self) -> dict:
        state = {
            "active": dict(self._active),
            "relay": {component: {slot: tensor.clone() for slot, tensor in block.items()}
                      for component, block in self._relay.items()},
            "pruned_history": list(self._pruned_history),
            "identity": self.schema.identity_digest(),
        }
        return state

    def plan_restore(self, state: dict) -> dict:
        """Validate a checkpoint completely, mutating nothing.

        Kept separate from :meth:`apply_restore` so that a malformed checkpoint is
        rejected before the adapter is compacted or optimizer state is migrated.
        Returns the validated plan for :meth:`apply_restore`.
        """
        if state.get("identity") != self.schema.identity_digest():
            raise StateError(
                "checkpoint component identity does not match this peer's schema")

        raw_active = state.get("active")
        if not isinstance(raw_active, dict) or not raw_active:
            raise StateError("checkpoint has an empty or malformed active mapping")

        positions, components = [], []
        seen_active_keys: Dict[int, object] = {}
        for raw_component, raw_position in raw_active.items():
            component, position = int(raw_component), int(raw_position)
            # 1 and "1" normalise to the same component but are distinct mapping
            # keys, so a serialised checkpoint can carry both and silently lose one.
            if component in seen_active_keys:
                raise StateError(
                    f"checkpoint active mapping has colliding identifiers for "
                    f"component {component}: {seen_active_keys[component]!r} and "
                    f"{raw_component!r}")
            seen_active_keys[component] = raw_component
            if component not in range(self.schema.r_max):
                raise ComponentError(
                    f"checkpoint active component {component} is outside "
                    f"P = 0..{self.schema.r_max - 1}")
            if position < 0:
                raise StateError(
                    f"checkpoint gives component {component} negative position {position}")
            components.append(component)
            positions.append(position)

        if len(set(components)) != len(components):
            raise StateError(f"checkpoint repeats an active component: {components}")
        if sorted(positions) != list(range(len(positions))):
            # duplicated, gapped or out-of-range slots all land here
            raise StateError(
                f"checkpoint active positions {sorted(positions)} are not a bijection "
                f"onto 0..{len(positions) - 1}")

        target = [component for component, _ in
                  sorted(zip(components, positions), key=lambda pair: pair[1])]

        held = set(self._active)
        missing = [c for c in target if c not in held]
        if missing:
            raise StateError(
                f"checkpoint support {target} contains components {missing} this peer "
                f"does not hold; supports never grow, so a checkpoint can only "
                f"describe a subset of the support this peer holds")

        raw_relay = state.get("relay", {})
        if not isinstance(raw_relay, dict):
            raise StateError("checkpoint relay record is malformed")

        relay: Dict[int, Dict[str, torch.Tensor]] = {}
        target_set = set(target)
        seen_relay_keys: Dict[int, object] = {}
        for raw_component, block in raw_relay.items():
            component = int(raw_component)
            if component in seen_relay_keys:
                raise StateError(
                    f"checkpoint relay record has colliding identifiers for component "
                    f"{component}: {seen_relay_keys[component]!r} and {raw_component!r}")
            seen_relay_keys[component] = raw_component
            if component not in range(self.schema.r_max):
                raise ComponentError(
                    f"checkpoint relay component {component} is outside P")
            if component in target_set:
                raise StateError(
                    f"checkpoint holds component {component} in relay storage while it "
                    f"is also active; that would be two authoritative copies")
            # exactly the rules a live block must satisfy
            self._validate_block(component, block)
            relay[component] = block

        history = state.get("pruned_history", [])
        if not isinstance(history, (list, tuple)):
            raise StateError("checkpoint pruned_history is malformed")
        normalised_history = []
        for component in history:
            value = int(component)
            if value not in range(self.schema.r_max):
                raise ComponentError(
                    f"checkpoint pruned_history names component {component}, outside P")
            normalised_history.append(value)

        # Pruning is monotone, so a component can be retired at most once...
        duplicates = sorted({c for c in normalised_history
                             if normalised_history.count(c) > 1})
        if duplicates:
            raise StateError(
                f"checkpoint pruned_history repeats components {duplicates}; pruning is "
                f"monotone, so a component can be retired at most once")

        # ...it cannot be active again...
        stranded = [c for c in normalised_history if c in target_set]
        if stranded:
            raise StateError(
                f"checkpoint lists components {stranded} as pruned while also holding "
                f"them in the active support")

        # ...and its value is retained rather than evicted, so it must be in relay.
        lost = [c for c in normalised_history if c not in relay]
        if lost:
            raise StateError(
                f"checkpoint lists components {lost} as pruned but holds no relay value "
                f"for them; pruning retains the latest value and relay state is never "
                f"evicted")

        return {"support": target, "relay": relay, "pruned_history": normalised_history}

    def apply_restore(self, plan: dict, optimizer=None) -> None:
        """Apply an already-validated plan. The first mutation happens here."""
        target = plan["support"]
        if target != list(self.support) or len(target) != self.bridge.schema.rank:
            self._compact(target, optimizer=optimizer)
        else:
            self._active = {c: i for i, c in enumerate(target)}
        self._relay = {component: self._to_relay(block)
                       for component, block in plan["relay"].items()}
        self._pruned_history = list(plan["pruned_history"])
        self.enforce_scale()
        self.check_invariants()

    def load_state_dict(self, state: dict, optimizer=None) -> None:
        """Validate the whole checkpoint, then apply it. Never the other way round."""
        self.apply_restore(self.plan_restore(state), optimizer=optimizer)

    # -- metrics ----------------------------------------------------------------------

    def relay_memory_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size()
                   for block in self._relay.values() for tensor in block.values())

    def metrics(self) -> dict:
        return {
            "pulse_active_rank": len(self._active),
            "pulse_materialised": len(self.materialised),
            "pulse_relay_components": len(self._relay),
            "pulse_relay_memory_bytes": self.relay_memory_bytes(),
            "pulse_pruned_total": len(self._pruned_history),
            "pulse_scale": self.schema.scale,
        }
