"""Bilateral transaction engine: a pure state machine over opaque tokens.

The engine holds transaction ids, phases, endpoint ids, expected versions and
one scalar staging token per endpoint. It holds no tensors, optimizer state,
state dicts or restoration data, and no peer objects or callbacks closing over
them, since a stored callable's ``__closure__`` reaches tensors just as well as
an attribute would.

So it returns instructions (``Stage``, ``Commit``, ``Rollback``, ``Finish``)
rather than performing them. The pair-session executor resolves each peer id and
calls the plugin, which keeps its own snapshot keyed by the token.
"""
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from federatedscope.contrib.decentralized.core.types import PeerId


class TransactionPhase(IntEnum):
    CREATED = 0
    STAGING = 1
    STAGED = 2
    COMMITTING = 3
    COMMITTED = 4
    ROLLING_BACK = 5
    ROLLED_BACK = 6


class TransactionError(RuntimeError):
    pass


# --- instructions the runner executes ---------------------------------------

@dataclass(frozen=True)
class Stage:
    peer_id: PeerId


@dataclass(frozen=True)
class Commit:
    peer_id: PeerId
    token: object


@dataclass(frozen=True)
class Rollback:
    peer_id: PeerId
    token: object


@dataclass(frozen=True)
class Finish:
    committed: bool
    reason: str = ""


# --- outcomes the runner reports back ----------------------------------------

@dataclass(frozen=True)
class Staged:
    peer_id: PeerId
    token: object


@dataclass(frozen=True)
class Applied:
    peer_id: PeerId


@dataclass(frozen=True)
class Failed:
    peer_id: Optional[PeerId]
    reason: str


def _check_token(token) -> None:
    """Tokens must be inert scalars: no plugin reference, no snapshot handle."""
    if not isinstance(token, (int, uuid.UUID)) or isinstance(token, bool):
        raise TransactionError(
            f"staging token must be an int or UUID carrying no reference, got "
            f"{type(token).__name__}; a handle object would give the engine a reference "
            f"to peer state"
        )


@dataclass
class TransactionState:
    transaction_id: int
    endpoints: Tuple[PeerId, PeerId]
    phase: TransactionPhase = TransactionPhase.CREATED
    tokens: Dict[PeerId, object] = field(default_factory=dict)
    expected_versions: Dict[PeerId, int] = field(default_factory=dict)
    reason: str = ""
    #: Endpoints whose rollback itself raised. Recorded, never retried forever.
    unrecovered: List[PeerId] = field(default_factory=list)
    _pending: List[PeerId] = field(default_factory=list)
    _rollback_queue: List[PeerId] = field(default_factory=list)


class TransactionEngine:
    """Drives staging and all-or-nothing commit of bilateral transactions.

    Endpoints are always staged and committed in ascending peer id, whatever order
    they are given in; a failure rolls back every staged endpoint in reverse
    staging order.
    """

    def __init__(self):
        self._next_id = 1
        self._last_issued = 0

    def state_dict(self) -> dict:
        # The id counter is part of run state: without it a resumed run
        # renumbers its transactions and its event log stops matching.
        return {"next_id": self._next_id, "last_issued": self._last_issued}

    def load_state_dict(self, state: dict) -> None:
        self._next_id = int(state["next_id"])
        self._last_issued = int(state.get("last_issued", self._next_id - 1))

    def begin(self, endpoints: Tuple[PeerId, PeerId],
              expected_versions: Optional[Dict[PeerId, int]] = None) -> TransactionState:
        a, b = endpoints
        if a == b:
            raise TransactionError(f"cannot open a transaction from peer {a} to itself")
        ordered = (a, b) if a < b else (b, a)  # stable lock order
        if self._next_id <= self._last_issued:
            raise TransactionError(
                f"transaction id {self._next_id} would repeat an already-issued id "
                f"({self._last_issued}); ids must be unique within a run so that each "
                f"event-log record and ledger row names one transaction"
            )
        self._last_issued = self._next_id
        state = TransactionState(
            transaction_id=self._next_id,
            endpoints=ordered,
            expected_versions=dict(expected_versions or {}),
        )
        state._pending = list(ordered)
        self._next_id += 1
        return state

    def step(self, state: TransactionState, outcome=None):
        """Advance the machine and return the next instruction."""
        if isinstance(outcome, Failed):
            if state.phase == TransactionPhase.ROLLING_BACK:
                # A rollback that itself fails must not restart the rollback, which
                # would loop forever on a permanently broken endpoint. Record the
                # endpoint as unrecovered, drop it, and keep unwinding the rest.
                return self._rollback_failed(state, outcome)
            return self._begin_rollback(state, outcome.reason or "failure reported")

        if state.phase in (TransactionPhase.CREATED, TransactionPhase.STAGING):
            if isinstance(outcome, Staged):
                _check_token(outcome.token)
                state.tokens[outcome.peer_id] = outcome.token
                state._rollback_queue.append(outcome.peer_id)
                if outcome.peer_id in state._pending:
                    state._pending.remove(outcome.peer_id)
            state.phase = TransactionPhase.STAGING
            if state._pending:
                return Stage(state._pending[0])
            state.phase = TransactionPhase.STAGED
            state._pending = list(state.endpoints)
            return Commit(state._pending[0], state.tokens[state._pending[0]])

        if state.phase in (TransactionPhase.STAGED, TransactionPhase.COMMITTING):
            state.phase = TransactionPhase.COMMITTING
            if isinstance(outcome, Applied) and outcome.peer_id in state._pending:
                state._pending.remove(outcome.peer_id)
            if state._pending:
                peer = state._pending[0]
                return Commit(peer, state.tokens[peer])
            state.phase = TransactionPhase.COMMITTED
            return Finish(committed=True)

        if state.phase == TransactionPhase.ROLLING_BACK:
            if isinstance(outcome, Applied) and outcome.peer_id in state._pending:
                state._pending.remove(outcome.peer_id)
            if state._pending:
                peer = state._pending[0]
                return Rollback(peer, state.tokens[peer])
            state.phase = TransactionPhase.ROLLED_BACK
            return Finish(committed=False, reason=state.reason)

        raise TransactionError(f"transaction {state.transaction_id} already finished")

    def _rollback_failed(self, state: TransactionState, outcome: "Failed"):
        peer_id = outcome.peer_id
        if peer_id is not None and peer_id not in state.unrecovered:
            state.unrecovered.append(peer_id)
        if peer_id in state._pending:
            state._pending.remove(peer_id)
        if state._pending:
            nxt = state._pending[0]
            return Rollback(nxt, state.tokens[nxt])
        state.phase = TransactionPhase.ROLLED_BACK
        return Finish(
            committed=False,
            reason=(f"{state.reason}; rollback failed for endpoints "
                    f"{sorted(state.unrecovered)} - their state may be inconsistent"),
        )

    def _begin_rollback(self, state: TransactionState, reason: str):
        state.reason = reason
        state.phase = TransactionPhase.ROLLING_BACK
        # Unwind in reverse of the order things were staged.
        state._pending = [p for p in reversed(state._rollback_queue) if p in state.tokens]
        if not state._pending:
            state.phase = TransactionPhase.ROLLED_BACK
            return Finish(committed=False, reason=reason)
        peer = state._pending[0]
        return Rollback(peer, state.tokens[peer])


class FailureInjector:
    """Deterministic, reproducible transaction failure injection.

    Fires every ``every``-th time the named stage is reached (the pair-session
    executor asks for ``"stage"`` and ``"commit"``), so a test can target exactly
    one stage without depending on random draws.
    """

    def __init__(self, stage: str = "", every: int = 0):
        self.stage = stage or ""
        self.every = int(every)
        self._counts: Dict[str, int] = {}

    def should_fail(self, stage: str) -> bool:
        if not self.stage or self.every <= 0 or stage != self.stage:
            return False
        self._counts[stage] = self._counts.get(stage, 0) + 1
        return self._counts[stage] % self.every == 0

    def state_dict(self) -> dict:
        return {"stage": self.stage, "every": self.every, "counts": dict(self._counts)}

    def load_state_dict(self, state: dict) -> None:
        self.stage = state["stage"]
        self.every = int(state["every"])
        self._counts = dict(state["counts"])
