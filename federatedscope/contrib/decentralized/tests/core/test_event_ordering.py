"""Dispatch order and the event queue.

Under event_driven scheduling a local-compute event is dated at its frame's time
plus the peer's compute latency, which can exceed one frame interval. The runner
therefore drains its queue only up to the next frame's time, which keeps dispatch
timestamps globally nondecreasing and makes pending events part of the run state.
"""
import pytest

from federatedscope.contrib.decentralized.core.events import EventQueue
from federatedscope.contrib.decentralized.core.runner import DecentralizedRunner
from federatedscope.contrib.decentralized.core.types import LocalComputeEvent
from federatedscope.contrib.decentralized.tests.core.test_runner import toy_cfg
from federatedscope.contrib.decentralized.tests.core.toy_method import ToyFactory


def dispatch_times(frames, scheduling="event_driven"):
    cfg = toy_cfg(frames=frames, **{"decentralized.execution.scheduling": scheduling})
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyFactory(cfg))
    seen = []
    original = runner._dispatch

    def spy(event, algorithms, steps):
        seen.append(event.time_us)
        return original(event, algorithms, steps)

    runner._dispatch = spy
    runner.run()
    return seen, runner


@pytest.mark.parametrize("frames", [4, 8, 12, 20])
def test_dispatch_time_never_goes_backwards(frames):
    seen, _ = dispatch_times(frames)
    violations = [(a, b) for a, b in zip(seen, seen[1:]) if b < a]
    assert not violations, f"{len(violations)} backward jumps, e.g. {violations[:3]}"


@pytest.mark.parametrize("frames", [4, 12])
def test_synchronized_scheduling_is_also_monotonic(frames):
    seen, _ = dispatch_times(frames, scheduling="synchronized")
    assert seen == sorted(seen)


def test_every_scheduled_event_is_dispatched():
    """Bounding the drain must not drop work."""
    frames, peers = 12, 4
    seen, runner = dispatch_times(frames)
    # one local-compute event per peer per frame, plus one session per offered edge
    sessions = sum(1 for e in runner.sink.events if e["event"] == "pair_session")
    assert len(seen) == frames * peers + sessions
    assert len(runner.coordinator.queue) == 0, "nothing may be left pending at the end"


def test_local_episode_counts_stay_equal_under_event_driven_scheduling():
    _, runner = dispatch_times(12)
    episodes = {pid: peer.local_episodes for pid, peer in runner.peers.items()}
    assert len(set(episodes.values())) == 1, f"uneven local work: {episodes}"


# --- the queue is run state ---------------------------------------------------

def test_pending_events_round_trip_through_a_checkpoint():
    queue = EventQueue()
    for time_us in (100, 5_000, 200):
        queue.push(LocalComputeEvent(0, time_us, 1, 0))

    restored = EventQueue()
    restored.load_state_dict(queue.state_dict())

    assert len(restored) == len(queue)
    assert [e.time_us for e in restored.pop_until(None)] == [100, 200, 5000]


def test_pop_until_respects_the_horizon():
    queue = EventQueue()
    for time_us in (100, 5_000, 200):
        queue.push(LocalComputeEvent(0, time_us, 1, 0))
    assert [e.time_us for e in queue.pop_until(1000)] == [100, 200]
    assert len(queue) == 1, "a future-dated event must stay pending"


def test_a_checkpoint_carries_pending_events():
    """A resumed run must not lose events that had not fired yet."""
    cfg = toy_cfg(frames=12, **{"decentralized.execution.scheduling": "event_driven"})
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyFactory(cfg))
    runner._build_peers()
    runner.coordinator.queue.push(LocalComputeEvent(0, 10 ** 9, 1, 0))

    state = runner.state_dict()
    assert "queue" in state
    assert state["queue"]["entries"], "the pending event must be in the checkpoint"
