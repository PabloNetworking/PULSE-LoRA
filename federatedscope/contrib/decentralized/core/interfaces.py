"""Small, separated plugin contracts.

Deliberately several narrow protocols rather than one base class, so a method
implements only what its interaction form needs. The runtime depends on these
shapes and never on a method name.

Nothing here mentions a LoRA tensor, a component, a support, or a relay plane:
those are method concepts. Method payloads cross this boundary as opaque bytes.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol, runtime_checkable

from federatedscope.contrib.decentralized.core.types import PeerId

INTERACTION_PAIRWISE = "pairwise_session"
INTERACTION_GRAPH = "graph_round"


@dataclass
class PeerContext:
    """Everything the runtime supplies when constructing one peer's algorithm."""

    peer_id: PeerId
    config: Any
    data: Any = None
    device: Any = None
    rng: Any = None
    client_config: Any = None
    extras: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PeerAlgorithm(Protocol):
    """Owns every tensor and every piece of metadata specific to the algorithm."""

    def local_episode(self, steps: int) -> Mapping[str, Any]:
        """Run one local-compute event. Returns method-local metrics."""

    def evaluate(self, split: str = "val") -> Mapping[str, Any]:
        """Evaluate this peer. There is no global model to evaluate instead."""

    def state_version(self) -> int:
        """Monotone version, bumped on every committed change to method state."""

    def state_dict(self) -> Mapping[str, Any]:
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        ...


@runtime_checkable
class PairwiseAlgorithm(Protocol):
    """A method that interacts through bilateral endpoint sessions.

    The session runs a four-stage negotiation, all of it method-agnostic:

    1. ``build_manifest``  - each endpoint declares what it has, as opaque bytes
    2. ``consume_manifest`` - each endpoint reads the other's declaration
    3. ``plan_payload``     - each endpoint plans, given local state, the remote
                              declaration, and the directional byte budget
    4. ``acknowledge``      - each endpoint is told the transaction outcome

    The runtime never decodes a manifest. Dec-LoRA puts a schema hash and payload
    size in its own; a component-selecting method puts its component inventory and
    its selection scores or disagreement sketches in the same slot and plans a
    budgeted subset in ``plan_payload`` - without any change to the runner, the
    executor, or the transaction engine.

    The executor also calls these optional hooks when a method defines them:
    ``identity_bound``, ``payload_identities``, ``planning_identities``,
    ``received_identities`` and ``auxiliary_payload_bytes``.
    """

    def build_manifest(self, session) -> bytes:
        """Opaque control declaration. Its encoded bytes are budgeted."""

    def consume_manifest(self, remote_manifest: bytes, session):
        """Decode and validate the peer's declaration; return a remote view."""

    def plan_payload(self, session, budget, remote_view) -> Optional[bytes]:
        """Deterministically choose this endpoint's payload, or ``None``.

        Must respect ``budget.payload_credit()``. Returning ``None`` yields a
        control-only session that changes no state at either endpoint.
        """

    def acknowledge(self, session, committed: bool, reason: str = "") -> None:
        """Explicit transaction outcome. Called exactly once per session."""

    def stage(self, incoming: bytes, session) -> object:
        """Snapshot internally and return an inert scalar token (int or UUID)."""

    def commit(self, token: object) -> None:
        """Apply the staged change, but keep the snapshot.

        A commit is not final until the whole transaction finishes: if the other
        endpoint fails afterwards, this endpoint must still be undoable. The
        snapshot is released by ``discard``.
        """

    def rollback(self, token: object) -> None:
        """Restore exactly. Sent bytes stay counted; a rollback cannot un-send."""

    def discard(self, token: object) -> None:
        """Release the snapshot once the transaction has finished, committed or rolled back."""


@runtime_checkable
class GraphAlgorithm(Protocol):
    """A method that interacts through synchronous snapshot-and-mix rounds."""

    def snapshot(self) -> bytes:
        """Encode this peer's pre-round state. Taken once, before any mixing."""

    def stage_mix(self, weights: Mapping[PeerId, float],
                  snapshots: Mapping[PeerId, bytes]) -> object:
        """Stage the weighted result from pre-round snapshots only."""

    def commit(self, token: object) -> None:
        """Apply the staged change while keeping the snapshot until discard."""

    def rollback(self, token: object) -> None:
        ...

    def discard(self, token: object) -> None:
        ...


@runtime_checkable
class DecentralizedMethodFactory(Protocol):
    """Supplies peers; validates only its own config subtree."""

    name: str
    interaction_form: str

    def validate_config(self, cfg) -> None:
        ...

    def create_peer(self, context: PeerContext) -> PeerAlgorithm:
        ...


@runtime_checkable
class MethodMetricProvider(Protocol):
    def method_metrics(self) -> Mapping[str, Any]:
        """Namespaced metrics exposed without altering the runner."""
