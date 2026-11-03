"""Contact traces: normalisation, digests and validation.

Runs are compared on the same trace through its digest, so the digest must be
stable and independent of the declared units, and no unit conversion may drift.
"""
import copy
import json
import os
import tempfile

import pytest

from federatedscope.contrib.decentralized.core import traces as T
from federatedscope.contrib.decentralized.core.rng import RngService
from federatedscope.contrib.decentralized.core.types import gross_credit_bytes

BASE = {
    "trace_schema_version": 1,
    "time_unit": "ms",
    "duration_unit": "ms",
    "goodput_unit": "kbps",
    "peers": [1, 2, 3, 4],
    "frames": [
        {"index": 0, "time": 0, "edges": [
            {"endpoints": [1, 2], "duration": 1000, "goodput": 1000, "transport": "direct"},
            {"endpoints": [3, 4], "duration": 1000, "goodput": 2000, "transport": "direct"},
        ]},
        {"index": 1, "time": 1000, "edges": [
            {"endpoints": [1, 3], "duration": 1000, "goodput": 1000, "transport": "infrastructure"},
        ]},
    ],
}


def payload(**overrides):
    out = copy.deepcopy(BASE)
    out.update(overrides)
    return out


# --- capacity -> byte conversion -------------------------------------------

def test_bit_to_byte_factor_in_isolation():
    # 8 bits/s for 1 s == 1 byte
    assert gross_credit_bytes(8, 1_000_000) == 1
    # 1 Mbit/s for 1 s == 125000 bytes
    assert gross_credit_bytes(1_000_000, 1_000_000) == 125_000


def test_conversion_floors_and_never_over_credits():
    # 7 bits/s for 1 s is 0.875 bytes -> 0, never 1
    assert gross_credit_bytes(7, 1_000_000) == 0
    # 15 bits/s for 1 s is 1.875 bytes -> 1
    assert gross_credit_bytes(15, 1_000_000) == 1


def test_credit_path_is_integer_only():
    trace = T.load_trace(payload())
    credits = trace.frame(0).opportunities[0].gross_credits()
    assert isinstance(credits.forward_bytes, int)
    assert isinstance(credits.reverse_bytes, int)
    assert credits.forward_bytes == 125_000  # 1000 kbps for 1000 ms


def test_declared_units_are_rejected_when_not_exactly_integral():
    bad = payload()
    bad["duration_unit"] = "us"
    bad["frames"][0]["edges"][0]["duration"] = 1.5
    with pytest.raises(T.TraceValidationError, match="does not normalise to an integer"):
        T.load_trace(bad)


# --- digest -----------------------------------------------------------------

def test_same_physical_trace_in_different_units_has_the_same_digest():
    in_kbps = T.load_trace(payload())

    other = payload()
    other["goodput_unit"] = "bps"
    other["duration_unit"] = "us"
    other["time_unit"] = "us"
    for frame in other["frames"]:
        frame["time"] *= 1000
        for edge in frame["edges"]:
            edge["duration"] *= 1000
            edge["goodput"] *= 1000
    in_bps = T.load_trace(other)

    assert in_kbps.trace_id == in_bps.trace_id
    assert in_kbps.normalised == in_bps.normalised


def test_digest_is_stable_across_repeated_loads():
    assert T.load_trace(payload()).trace_id == T.load_trace(payload()).trace_id


@pytest.mark.parametrize("mutate,label", [
    (lambda p: p["frames"][0]["edges"][0].__setitem__("endpoints", [1, 4]), "endpoint"),
    (lambda p: p["frames"][0]["edges"][0].__setitem__("goodput", 999), "capacity"),
    (lambda p: p["frames"][0]["edges"][0].__setitem__("duration", 999), "duration"),
    (lambda p: p["frames"][1].__setitem__("time", 5000), "time"),
    (lambda p: p["frames"][0]["edges"][0].__setitem__("transport", "infrastructure"), "transport"),
    (lambda p: p["frames"][0]["edges"][0].__setitem__("available", False), "availability"),
])
def test_digest_changes_when_any_exogenous_field_changes(mutate, label):
    original = T.load_trace(payload()).trace_id
    mutated = payload()
    mutate(mutated)
    assert T.load_trace(mutated).trace_id != original, f"{label} change did not alter the digest"


def test_declared_endpoint_order_does_not_change_the_digest_but_direction_is_preserved():
    forward = payload()
    forward["frames"][0]["edges"][0] = {
        "endpoints": [1, 2], "duration": 1000,
        "goodput_forward": 1000, "goodput_reverse": 2000, "transport": "direct",
    }
    reversed_decl = payload()
    reversed_decl["frames"][0]["edges"][0] = {
        "endpoints": [2, 1], "duration": 1000,
        "goodput_forward": 2000, "goodput_reverse": 1000, "transport": "direct",
    }
    a = T.load_trace(forward)
    b = T.load_trace(reversed_decl)
    assert a.trace_id == b.trace_id
    edge = a.frame(0).opportunities[0]
    assert edge.endpoints == (1, 2)
    assert edge.goodput_bps_forward == 1_000_000  # 1->2
    assert edge.goodput_bps_reverse == 2_000_000  # 2->1


# --- validation --------------------------------------------------------------

@pytest.mark.parametrize("mutate,expected", [
    (lambda p: p.__setitem__("trace_schema_version", 99), "unsupported trace_schema_version"),
    (lambda p: p.__setitem__("peers", []), "non-empty peers"),
    (lambda p: p.__setitem__("peers", [1, 1, 2]), "duplicate peer id"),
    (lambda p: p["frames"][0]["edges"][0].__setitem__("endpoints", [1, 1]), "self-loop"),
    (lambda p: p["frames"][0]["edges"][0].__setitem__("endpoints", [1, 99]), "not in declared peers"),
    (lambda p: p["frames"][0]["edges"].append(
        {"endpoints": [2, 1], "duration": 1000, "goodput": 1000}), "duplicate edge"),
    (lambda p: p["frames"][1].__setitem__("time", 0) or p["frames"][0].__setitem__("time", 5000),
     "precedes previous frame"),
    (lambda p: p["frames"][1].__setitem__("index", 7), "out of order"),
    (lambda p: p["frames"][0]["edges"][0].__setitem__("duration", 0), "duration must be positive"),
    (lambda p: p["frames"][0]["edges"][0].__setitem__("goodput", 0), "positive goodput"),
    (lambda p: p["frames"][0]["edges"][0].__setitem__("transport", "carrier-pigeon"), "unknown transport"),
    (lambda p: p["frames"][0]["edges"][0].pop("goodput"), "needs goodput"),
    (lambda p: p["frames"][0]["edges"][0].pop("duration"), "missing duration"),
    (lambda p: p.pop("duration_unit"), "must declare duration_unit"),
    (lambda p: p.pop("goodput_unit"), "must declare goodput_unit"),
])
def test_malformed_traces_are_rejected(mutate, expected):
    bad = payload()
    mutate(bad)
    with pytest.raises(T.TraceValidationError, match=expected):
        T.load_trace(bad)


def test_unavailable_edge_may_have_zero_capacity():
    ok = payload()
    ok["frames"][0]["edges"][0]["goodput"] = 0
    ok["frames"][0]["edges"][0]["available"] = False
    trace = T.load_trace(ok)
    assert trace.frame(0).opportunities[0].available is False


# --- matching profile ---------------------------------------------------------

def test_matching_profile_rejects_a_peer_in_two_edges():
    overlapping = payload()
    overlapping["frames"][0]["edges"].append(
        {"endpoints": [2, 3], "duration": 1000, "goodput": 1000})
    trace = T.load_trace(overlapping)  # a valid trace, just not a matching
    with pytest.raises(T.TraceValidationError, match="requires a matching"):
        T.validate_matching_profile(trace)


def test_matching_profile_accepts_a_disjoint_frame():
    T.validate_matching_profile(T.load_trace(payload()))


def test_matching_profile_ignores_unavailable_edges():
    p = payload()
    p["frames"][0]["edges"].append(
        {"endpoints": [2, 3], "duration": 1000, "goodput": 0, "available": False})
    T.validate_matching_profile(T.load_trace(p))


# --- file backing --------------------------------------------------------------

def test_file_loading_round_trips_and_matches_the_in_memory_digest():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "trace.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload(), handle)
        assert T.load_trace_file(path).trace_id == T.load_trace(payload()).trace_id


def test_missing_file_raises_and_never_falls_back():
    with pytest.raises(T.TraceValidationError, match="trace file not found"):
        T.load_trace_file("/nonexistent/definitely/not/here.json")


def test_malformed_json_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        with pytest.raises(T.TraceValidationError, match="not valid JSON"):
            T.load_trace_file(path)


# --- generation ------------------------------------------------------------------

def test_generation_is_deterministic_and_isolated_from_the_global_rng():
    import random

    random.seed(11)
    before = random.random()
    a = T.generate_matching_trace([1, 2, 3, 4], 5, RngService(3).stream("opportunities"))
    b = T.generate_matching_trace([1, 2, 3, 4], 5, RngService(3).stream("opportunities"))
    random.seed(11)
    assert before == random.random(), "generation perturbed the global RNG"
    assert a.trace_id == b.trace_id


def test_generated_trace_is_a_valid_matching_even_with_an_odd_peer_count():
    trace = T.generate_matching_trace([1, 2, 3, 4, 5], 4, RngService(5).stream("opportunities"))
    T.validate_matching_profile(trace)
    for frame in trace.frames:
        assert len(frame.opportunities) == 2  # one peer sits out, legitimately
