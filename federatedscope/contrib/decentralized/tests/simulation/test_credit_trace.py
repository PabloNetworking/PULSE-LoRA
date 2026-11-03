"""Credit traces (simulation/credit_trace.py) and the reference scenarios (scripts/make_scenario.py)."""
import importlib.util
import json
import os

import pytest

from federatedscope.contrib.decentralized.core.traces import load_trace
from federatedscope.contrib.decentralized.simulation.credit_trace import (
    bps_to_credit_bytes,
    contact_credits,
    credit_bytes_to_bps,
    credit_trace,
    draw_factors,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 5))
REFERENCE = {"b_ref": {"forward": 101661258, "reverse": 135554034}, "peers": list(range(1, 11)),
             "frames": 120, "p": 0.1, "trace_seed": 21, "capacity_seed": 21, "low": 0.6, "high": 1.8}
#: The digest of the reference trace's contents (``scripts/make_scenario.py`` defaults).
REFERENCE_TRACE_ID = "ce7f7b4065e58adb24a8c7d0ce6e2c9ed576c390e7790337818f216c92b8c6a6"


def reference_trace():
    args = dict(REFERENCE)
    return credit_trace(args.pop("b_ref"), **args)


def test_the_reference_trace_is_reproduced():
    trace = reference_trace()
    assert trace["generator"]["trace_id"] == REFERENCE_TRACE_ID
    assert load_trace(trace).trace_id == REFERENCE_TRACE_ID


def test_every_credit_is_the_reference_scaled_into_the_factor_range():
    rows = contact_credits(reference_trace())
    assert rows
    for row in rows:
        for direction in ("forward", "reverse"):
            ratio = row[direction] / REFERENCE["b_ref"][direction]
            assert 0.6 - 1e-9 <= ratio < 1.8


@pytest.mark.parametrize("credit", [0, 1, 7, 101661258, 2 ** 40 + 3])
def test_a_credit_round_trips_through_the_goodput_exactly(credit):
    assert bps_to_credit_bytes(credit_bytes_to_bps(credit)) == credit


def test_inexact_contacts_and_bad_ranges_are_refused():
    with pytest.raises(ValueError):
        credit_bytes_to_bps(5, duration_ms=1500)
    with pytest.raises(ValueError):
        credit_bytes_to_bps(-1)
    with pytest.raises(ValueError):
        draw_factors(3, seed=1, low=1.8, high=0.6)


def test_the_factors_depend_on_the_seed_only():
    assert draw_factors(5, 21, 0.6, 1.8) == draw_factors(5, 21, 0.6, 1.8)
    assert draw_factors(5, 21, 0.6, 1.8) != draw_factors(5, 22, 0.6, 1.8)


def _make_scenario():
    spec = importlib.util.spec_from_file_location(
        "make_scenario", os.path.join(REPO, "scripts", "make_scenario.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("method", ["pulse_lora", "dec_lora", "tad_lora"])
def test_the_reference_scenarios_carry_the_reference_trace(method, tmp_path):
    script = _make_scenario()
    out = tmp_path / f"{method}.json"
    script.main([method, "--out", str(out)])
    document = json.loads(out.read_text())
    assert document["trace"]["generator"]["trace_id"] == REFERENCE_TRACE_ID
    assert document["construction"] == {"mode": "common_canonical_v1", "model_init_seed": 31001}
    assert document["training_rng"] == "per_peer_episode"
    if method == "pulse_lora":
        ranks = {int(p): v["decentralized.methods.pulse_lora.initial_rank"]
                 for p, v in document["peer_overrides"].items()}
        assert ranks == {1: 64, 2: 64, 3: 150, 4: 150, 5: 120, 9: 120, 10: 120}
