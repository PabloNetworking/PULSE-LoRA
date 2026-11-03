"""The transaction engine: lock order, atomicity, and the ownership boundary."""
import uuid

import pytest

from federatedscope.contrib.decentralized.core.transaction import (
    Applied,
    Commit,
    Failed,
    FailureInjector,
    Finish,
    Rollback,
    Stage,
    Staged,
    TransactionEngine,
    TransactionError,
    TransactionPhase,
)
from federatedscope.contrib.decentralized.tests.core.ownership import find_learning_state


def drive(engine, state, outcomes):
    """Feed scripted outcomes, collecting the instructions the engine emits."""
    actions = []
    action = engine.step(state)
    for outcome in outcomes:
        actions.append(action)
        action = engine.step(state, outcome)
    actions.append(action)
    return actions


def test_locks_are_acquired_in_ascending_peer_id_regardless_of_argument_order():
    engine = TransactionEngine()
    assert engine.begin((9, 4)).endpoints == (4, 9)
    assert engine.begin((4, 9)).endpoints == (4, 9)


def test_self_transaction_is_rejected():
    with pytest.raises(TransactionError, match="to itself"):
        TransactionEngine().begin((3, 3))


def test_happy_path_stages_both_then_commits_both():
    engine = TransactionEngine()
    state = engine.begin((1, 2))
    actions = drive(engine, state, [Staged(1, 11), Staged(2, 22), Applied(1), Applied(2)])
    assert [type(a).__name__ for a in actions] == \
           ["Stage", "Stage", "Commit", "Commit", "Finish"]
    assert actions[-1].committed is True
    assert state.phase == TransactionPhase.COMMITTED


def test_failure_rolls_back_every_staged_endpoint_in_reverse_order():
    engine = TransactionEngine()
    state = engine.begin((1, 2))
    engine.step(state)
    engine.step(state, Staged(1, 11))
    engine.step(state, Staged(2, 22))
    first = engine.step(state, Failed(2, "boom"))
    assert isinstance(first, Rollback) and first.peer_id == 2
    second = engine.step(state, Applied(2))
    assert isinstance(second, Rollback) and second.peer_id == 1
    done = engine.step(state, Applied(1))
    assert isinstance(done, Finish) and done.committed is False and done.reason == "boom"


def test_failure_before_any_staging_needs_no_rollback():
    engine = TransactionEngine()
    state = engine.begin((1, 2))
    engine.step(state)
    done = engine.step(state, Failed(1, "early"))
    assert isinstance(done, Finish) and done.committed is False


@pytest.mark.parametrize("token", [7, uuid.uuid4()])
def test_scalar_tokens_are_accepted(token):
    engine = TransactionEngine()
    state = engine.begin((1, 2))
    engine.step(state)
    engine.step(state, Staged(1, token))
    assert state.tokens[1] == token


@pytest.mark.parametrize("token", [object(), {"snapshot": 1}, [1, 2], True, "abc", 1.5])
def test_tokens_carrying_a_reference_are_refused(token):
    """Only int or UUID tokens are accepted, so the engine holds no reference to peer state."""
    engine = TransactionEngine()
    state = engine.begin((1, 2))
    engine.step(state)
    with pytest.raises(TransactionError, match="int or UUID"):
        engine.step(state, Staged(1, token))


def test_engine_holds_no_learning_state_while_staged():
    engine = TransactionEngine()
    state = engine.begin((1, 2))
    engine.step(state)
    engine.step(state, Staged(1, 11))
    engine.step(state, Staged(2, 22))
    assert state.phase == TransactionPhase.STAGED

    offenders = find_learning_state(engine) + find_learning_state(state)
    assert not offenders, f"transaction machinery reached learning state: {offenders}"
    # and the tokens are plain integers
    assert all(isinstance(t, int) for t in state.tokens.values())


def test_failure_injector_is_deterministic_and_stage_scoped():
    injector = FailureInjector("commit", every=2)
    assert injector.should_fail("stage") is False
    assert injector.should_fail("commit") is False
    assert injector.should_fail("commit") is True
    assert injector.should_fail("commit") is False
    assert injector.should_fail("commit") is True


def test_failure_injector_is_inert_when_unconfigured():
    injector = FailureInjector()
    assert not any(injector.should_fail(stage) for stage in ("stage", "commit") for _ in range(5))


def test_failure_injector_round_trips_through_a_checkpoint():
    injector = FailureInjector("commit", every=3)
    injector.should_fail("commit")
    restored = FailureInjector()
    restored.load_state_dict(injector.state_dict())
    assert restored.state_dict() == injector.state_dict()


# --- rollback failures and transaction ids ----------------------------------

def test_a_failing_rollback_terminates_instead_of_looping_forever():
    """A rollback that keeps failing ends the transaction uncommitted, with both
    endpoints recorded as unrecovered."""
    engine = TransactionEngine()
    state = engine.begin((1, 2))
    engine.step(state)
    engine.step(state, Staged(1, 1))
    engine.step(state, Staged(2, 2))

    action = engine.step(state, Failed(2, "boom"))
    for _ in range(10):
        if isinstance(action, Finish):
            break
        action = engine.step(state, Failed(action.peer_id, "rollback failed"))
    else:
        pytest.fail("the engine never terminated when rollback kept failing")

    assert action.committed is False
    assert "rollback failed for endpoints" in action.reason
    assert sorted(state.unrecovered) == [1, 2]


def test_a_partially_failing_rollback_still_unwinds_the_rest():
    engine = TransactionEngine()
    state = engine.begin((1, 2))
    engine.step(state)
    engine.step(state, Staged(1, 1))
    engine.step(state, Staged(2, 2))

    first = engine.step(state, Failed(2, "boom"))          # rollback peer 2
    second = engine.step(state, Failed(first.peer_id, "cannot restore"))
    assert isinstance(second, Rollback) and second.peer_id == 1, \
        "the healthy endpoint must still be rolled back"
    done = engine.step(state, Applied(1))
    assert isinstance(done, Finish) and state.unrecovered == [2]


def test_transaction_ids_are_unique_within_a_run():
    engine = TransactionEngine()
    ids = {engine.begin((1, 2)).transaction_id for _ in range(50)}
    assert len(ids) == 50


def test_a_reused_transaction_id_is_refused():
    """A repeated id would make two transactions indistinguishable in the event log."""
    engine = TransactionEngine()
    engine.begin((1, 2))
    engine.begin((1, 2))
    engine._next_id = 1  # simulate a bad restore
    with pytest.raises(TransactionError, match="would repeat an already-issued id"):
        engine.begin((1, 2))


def test_id_numbering_survives_a_checkpoint_round_trip():
    engine = TransactionEngine()
    for _ in range(3):
        engine.begin((1, 2))
    restored = TransactionEngine()
    restored.load_state_dict(engine.state_dict())
    assert restored.begin((1, 2)).transaction_id == 4
