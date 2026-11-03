"""Deterministic event ordering."""

from federatedscope.contrib.decentralized.core.events import EventQueue, structural_key
from federatedscope.contrib.decentralized.core.rng import RngService
from federatedscope.contrib.decentralized.core.types import (
    DirectionalCredits,
    EvaluationEvent,
    LocalComputeEvent,
    PairSessionEvent,
)


def pair(event_id, time_us, endpoints):
    return PairSessionEvent(event_id, time_us, 0, endpoints, 1000,
                            DirectionalCredits(10, 10), "direct")


def test_local_compute_precedes_interaction_at_equal_time():
    queue = EventQueue()
    queue.push(pair(0, 100, (1, 2)))
    queue.push(LocalComputeEvent(1, 100, 5, 0))
    kinds = [type(e).__name__ for e in queue.drain()]
    assert kinds == ["LocalComputeEvent", "PairSessionEvent"]


def test_equal_time_same_kind_breaks_by_structural_key_not_insertion():
    queue = EventQueue()
    for peer_id in (7, 3, 5, 1):
        queue.push(LocalComputeEvent(0, 100, peer_id, 0))
    assert [e.peer_id for e in queue.drain()] == [1, 3, 5, 7]


def test_ordering_is_independent_of_insertion_order():
    events = [pair(0, 5, (3, 4)), LocalComputeEvent(1, 5, 2, 0), pair(2, 1, (1, 2))]

    first, second = EventQueue(), EventQueue()
    for event in events:
        first.push(event)
    for event in reversed(events):
        second.push(event)

    assert [structural_key(e) for e in first.drain()] == \
           [structural_key(e) for e in second.drain()]


def test_time_dominates_kind():
    queue = EventQueue()
    queue.push(LocalComputeEvent(0, 200, 1, 0))
    queue.push(pair(1, 100, (1, 2)))
    assert [type(e).__name__ for e in queue.drain()] == \
           ["PairSessionEvent", "LocalComputeEvent"]


def test_events_are_never_compared_directly():
    """Two events identical up to the tiebreak must not raise on heap compare."""
    queue = EventQueue()
    queue.push(EvaluationEvent(0, 10, 0))
    queue.push(EvaluationEvent(0, 10, 0))
    assert len(list(queue.drain())) == 2


def test_isolated_rng_streams_are_deterministic():
    a = [RngService(9).stream("topology").random() for _ in range(4)]
    b = [RngService(9).stream("topology").random() for _ in range(4)]
    assert a == b
