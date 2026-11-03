"""PULSE transactional mixing: stage, commit, roll back.

The generic transaction engine (``core/transaction.py``) sequences the two
endpoints and names no method. This module supplies what the engine asks a
plugin for: turn the partner's bytes into a reversible staged change, apply it,
and put it back exactly if anything downstream fails.

Staging does not write. The engine may stage one endpoint and then fail before
the other, so a stage that wrote would leave a peer changed by a transaction
that never happened. Staging therefore snapshots, computes the mixed values,
and stores both; ``commit`` is the first write.

A commit stays undoable. The partner can fail after this endpoint has
committed, so the snapshot is kept until ``discard``, which the runtime calls
on both the committed and the failed paths.

Support never changes. Selected components are materialised at both peers and
their trainable supports stay as they were. A component outside ``S_i`` lands
on the relay plane, where it receives no gradients and never enters the forward
pass; that is how a peer relays a component it does not train.
"""
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import torch

from federatedscope.contrib.decentralized.lora.mixing.factor_gossip import bilateral_gossip
from federatedscope.contrib.decentralized.lora.optimizer_state import (
    clear_moment_slices, read_moment_slices, write_moment_slices,
)


class MixError(ValueError):
    """A payload cannot be mixed. Raised before anything is written."""


@dataclass
class StagedMix:
    """One reversible mix, held by the peer and keyed by an inert scalar token."""
    components: Tuple[int, ...]
    mixed: Dict[int, Dict[str, torch.Tensor]]
    state_snapshot: dict
    moments: list
    version: int
    committed: bool = False


def plan_mix(state, components: Sequence[int],
             remote_blocks: Mapping[int, Mapping[str, torch.Tensor]],
             gamma: float) -> Dict[int, Dict[str, torch.Tensor]]:
    """``(1-g) x_i + g x_j`` per component, through the shared kernel.

    ``bilateral_gossip`` is also the kernel Dec-LoRA's pairwise profile mixes
    whole adapters with: PULSE differs in what it selects, not in how two values
    are combined, and one kernel keeps one mean-preserving implementation.
    """
    mixed = {}
    for component in components:
        local_block = state.gather(component)
        remote_block = remote_blocks[component]
        # The remote block was decoded from bytes and is on the CPU, while gather()
        # returns the local block on the device of the plane that holds it: the
        # model's device for an active component, the CPU for a relay or untouched
        # one. Align the remote block to it here so the shared kernel needs no
        # device handling.
        aligned = {
            slot: tensor.to(device=local_block[slot].device,
                            dtype=local_block[slot].dtype)
            for slot, tensor in remote_block.items()
            if slot in local_block
        }
        mixed[component] = dict(bilateral_gossip(local_block, aligned, gamma))
    return mixed


def moment_slices(state, bridge, components: Sequence[int]):
    """``(param, dim, index)`` for every active component's own row and column.

    Keyed on the component's compact position at this instant. A component owns
    one row of each ``A`` and one column of each ``B``, so anything coarser would
    reset momentum for the r-1 other components sharing the tensor.
    """
    slices = []
    for component in components:
        if not state.is_active(component):
            continue                      # inactive components hold no optimizer state
        position = state.compact_index(component)
        for spec in bridge.schema.modules:
            slices.append((bridge._params[spec.a_key], 0, position))
            slices.append((bridge._params[spec.b_key], 1, position))
    return slices


class MixEngine:
    """The plugin half of the transaction. Owns snapshots; the engine owns none."""

    def __init__(self, state, bridge, components_schema, gamma: float,
                 optimizer_source=None):
        self.state = state
        self.bridge = bridge
        self.schema = components_schema
        self.gamma = float(gamma)
        self._optimizer_source = optimizer_source or (lambda: None)
        self._staged: Dict[int, StagedMix] = {}
        self._next_token = 1

    # -- the three hooks ------------------------------------------------------------

    def stage(self, components: Sequence[int],
              remote_blocks: Mapping[int, Mapping[str, torch.Tensor]],
              version: int) -> int:
        """Snapshot and compute. Nothing is written until :meth:`commit`."""
        components = tuple(sorted(int(c) for c in components))
        missing = [c for c in components if c not in remote_blocks]
        if missing:
            raise MixError(
                f"the payload is missing components {missing} that were agreed for "
                f"this session; a partial exchange is not a smaller exchange")

        mixed = plan_mix(self.state, components, remote_blocks, self.gamma)

        optimizer = self._optimizer_source()
        slices = moment_slices(self.state, self.bridge, components)
        token = self._next_token
        self._next_token += 1
        self._staged[token] = StagedMix(
            components=components,
            mixed=mixed,
            state_snapshot=self.state.snapshot_components(components),
            moments=read_moment_slices(optimizer, slices) if optimizer else [],
            version=int(version),
        )
        return token

    def commit(self, token: int) -> None:
        """Write every component or none, then reset exactly the touched moments."""
        staged = self._staged[token]

        # One pass over an all-or-nothing write. scatter() validates each block
        # and routes it to whichever plane owns the component, so an active
        # component reaches the compact tensors and an inactive one reaches
        # relay - the support itself is never touched.
        try:
            for component in staged.components:
                self.state.scatter(component, staged.mixed[component])
        except Exception:
            # A block that fails validation half way through would leave the
            # peer holding some mixed and some original components. Put it back
            # before letting the engine see the failure.
            self.state.restore_components(staged.state_snapshot)
            raise

        optimizer = self._optimizer_source()
        if optimizer is not None:
            clear_moment_slices(
                optimizer, moment_slices(self.state, self.bridge, staged.components))
        staged.committed = True

    def rollback(self, token: int) -> None:
        """Restore learning state, relay membership and moments exactly."""
        staged = self._staged.pop(token, None)
        if staged is None:
            return                        # already discarded; nothing to undo
        self.state.restore_components(staged.state_snapshot)
        optimizer = self._optimizer_source()
        if optimizer is not None and staged.moments:
            write_moment_slices(optimizer, staged.moments)

    def discard(self, token: int) -> None:
        self._staged.pop(token, None)

    # -- introspection for tests and metrics ------------------------------------------

    @property
    def open_tokens(self) -> Tuple[int, ...]:
        return tuple(sorted(self._staged))

    def staged_components(self, token: int) -> Tuple[int, ...]:
        return self._staged[token].components

    def was_committed(self, token: int) -> bool:
        return self._staged[token].committed
