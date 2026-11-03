"""Topology history records offered contacts, never method outcomes."""
from federatedscope.contrib.decentralized.core.topology import TopologyHistory


def test_records_offered_edges_and_reports_degree():
    history = TopologyHistory([1, 2, 3, 4])
    history.record_frame(0, [(1, 2), (3, 4)])
    history.record_frame(1, [(1, 3), (2, 4)])
    assert history.degree() == {1: 2, 2: 2, 3: 2, 4: 2}


def test_edges_are_normalised_so_direction_does_not_duplicate():
    history = TopologyHistory([1, 2])
    history.record_frame(0, [(2, 1)])
    assert history.recent_edges() == [(1, 2)]


def test_bounded_connectivity_over_the_retained_window():
    history = TopologyHistory([1, 2, 3, 4])
    history.record_frame(0, [(1, 2)])
    assert not history.union_is_connected()
    history.record_frame(1, [(3, 4)])
    assert not history.union_is_connected(), "two components is not connected"
    history.record_frame(2, [(2, 3)])
    assert history.union_is_connected()


def test_window_evicts_old_frames():
    history = TopologyHistory([1, 2, 3, 4], window=2)
    history.record_frame(0, [(1, 2)])
    history.record_frame(1, [(3, 4)])
    history.record_frame(2, [(2, 3)])
    assert (1, 2) not in history.recent_edges(), "frame 0 should have been evicted"


def test_single_peer_is_trivially_connected():
    assert TopologyHistory([1]).union_is_connected()


def test_round_trips_through_a_checkpoint():
    history = TopologyHistory([1, 2, 3], window=4)
    history.record_frame(0, [(1, 2)])
    history.record_frame(1, [(2, 3)])

    restored = TopologyHistory([1, 2, 3])
    restored.load_state_dict(history.state_dict())
    assert restored.recent_edges() == history.recent_edges()
    assert restored.degree() == history.degree()
