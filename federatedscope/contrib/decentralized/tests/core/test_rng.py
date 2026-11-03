"""Named RNG streams: isolation, derivation, and restore-in-place."""
import pytest

from federatedscope.contrib.decentralized.core.rng import RngService, derive_seed


def test_streams_are_derived_deterministically_from_the_master_seed():
    assert derive_seed(7, "topology") == derive_seed(7, "topology")
    assert derive_seed(7, "topology") != derive_seed(7, "opportunities")
    assert derive_seed(7, "topology") != derive_seed(8, "topology")


def test_two_services_with_the_same_seed_agree_stream_by_stream():
    left, right = RngService(11), RngService(11)
    assert left.stream("a").random() == right.stream("a").random()
    assert float(left.numpy_stream("b").random()) == float(right.numpy_stream("b").random())


def test_the_process_global_generators_are_never_touched():
    import random as _random
    _random.seed(1234)
    before = _random.random()
    _random.seed(1234)
    service = RngService(99)
    [service.stream("noise").random() for _ in range(50)]
    service.numpy_stream("noise").random(50)
    assert _random.random() == before


# -- restore must keep the generator objects a consumer already holds -----------------
#
# Consumers capture their stream by reference: DecentralizedRunner hands a peer
# rng.stream(f"method:{m}:peer{id}") when it builds the peer, and it builds peers
# before loading a checkpoint. A restore therefore sets the state of the existing
# generator objects instead of replacing them.

def test_a_held_python_stream_survives_restore_with_its_identity():
    service = RngService(7)
    held = service.stream("method:pulse_lora:peer1")
    [held.random() for _ in range(3)]
    snapshot = service.state_dict()
    expected = held.random()

    service.load_state_dict(snapshot)

    assert service.stream("method:pulse_lora:peer1") is held, "the object was replaced"
    assert held.random() == expected, "the held reference did not replay"


def test_a_held_numpy_stream_survives_restore_with_its_identity():
    service = RngService(7)
    held = service.numpy_stream("partition")
    held.random(3)
    snapshot = service.state_dict()
    expected = float(held.random())

    service.load_state_dict(snapshot)

    assert service.numpy_stream("partition") is held, "the object was replaced"
    assert float(held.random()) == expected, "the held reference did not replay"


def test_named_streams_stay_isolated_across_a_restore():
    service = RngService(7)
    left, right = service.stream("topology"), service.stream("opportunities")
    numpy_stream = service.numpy_stream("partition")
    [left.random() for _ in range(2)]
    [right.random() for _ in range(5)]
    numpy_stream.random(4)
    snapshot = service.state_dict()
    expected = (left.random(), right.random(), float(numpy_stream.random()))

    service.load_state_dict(snapshot)

    assert (left.random(), right.random(), float(numpy_stream.random())) == expected
    assert left is service.stream("topology")
    assert right is service.stream("opportunities")


def test_a_stream_only_in_the_checkpoint_is_created_and_restored():
    """A resumed run must not silently start it from the master seed."""
    source = RngService(7)
    stream = source.stream("late")
    [stream.random() for _ in range(4)]
    snapshot = source.state_dict()
    expected = stream.random()

    target = RngService(7)                     # has never touched "late"
    target.load_state_dict(snapshot)

    assert target.stream("late").random() == expected


def test_a_stream_missing_from_the_checkpoint_is_reset_not_left_advanced():
    """Leaving it would make the resumed run depend on pre-checkpoint work."""
    service = RngService(7)
    snapshot = service.state_dict()            # taken before "later" exists
    later = service.stream("later")
    [later.random() for _ in range(5)]

    service.load_state_dict(snapshot)

    assert service.stream("later") is later, "the object should still be reusable"
    assert later.random() == RngService(7).stream("later").random()


def test_a_malformed_checkpoint_leaves_every_stream_untouched():
    service = RngService(7)
    held = service.stream("keep")
    [held.random() for _ in range(3)]
    expected = held.getstate()

    with pytest.raises(Exception):
        service.load_state_dict({"master_seed": 7, "python": {"keep": ("not", "a", "state")}})

    assert held.getstate() == expected, "a refused checkpoint half-restored a stream"


# -- the same three reconciliation cases, mirrored for NumPy generators ----------------

def test_a_numpy_stream_only_in_the_checkpoint_is_created_and_restored():
    source = RngService(7)
    stream = source.numpy_stream("late")
    stream.random(4)
    snapshot = source.state_dict()
    expected = float(stream.random())

    target = RngService(7)                     # has never touched "late"
    target.load_state_dict(snapshot)

    assert float(target.numpy_stream("late").random()) == expected


def test_a_numpy_stream_missing_from_the_checkpoint_is_reset_not_left_advanced():
    service = RngService(7)
    snapshot = service.state_dict()            # taken before "later" exists
    later = service.numpy_stream("later")
    later.random(5)

    service.load_state_dict(snapshot)

    assert service.numpy_stream("later") is later
    assert float(later.random()) == float(RngService(7).numpy_stream("later").random())


def test_a_malformed_numpy_state_leaves_every_stream_untouched():
    """The failure is placed after a valid python stream, so a partial apply shows."""
    service = RngService(7)
    python_held = service.stream("keep")
    numpy_held = service.numpy_stream("keep")
    [python_held.random() for _ in range(3)]
    numpy_held.random(3)
    expected_python = python_held.getstate()
    expected_numpy = dict(numpy_held.bit_generator.state)

    snapshot = service.state_dict()
    snapshot["numpy"]["keep"] = {"bit_generator": "NotAGenerator"}

    with pytest.raises(Exception):
        service.load_state_dict(snapshot)

    assert python_held.getstate() == expected_python, "a valid python stream was applied"
    assert numpy_held.bit_generator.state == expected_numpy


def test_both_generator_kinds_share_one_master_seed_but_stay_independent():
    service = RngService(7)
    python_stream, numpy_stream = service.stream("shared"), service.numpy_stream("shared")
    snapshot = service.state_dict()
    expected = (python_stream.random(), float(numpy_stream.random()))

    service.load_state_dict(snapshot)

    assert (python_stream.random(), float(numpy_stream.random())) == expected
