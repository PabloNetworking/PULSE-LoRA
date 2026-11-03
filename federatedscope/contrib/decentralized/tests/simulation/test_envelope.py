"""The message envelope (core/wire_cost.py) and what the ``socket_v1`` pricing and the session
executors do with a message over it.

Every message of a run must fit one envelope: 64 MiB by default, at most 512 MiB when the
run's configuration sets ``decentralized.transport.envelope_bytes``. The pricing model prices a
pair session or graph-round edge that holds a larger message at ``OVER_CAP``, which no credit
admits: a planner that searches finds the largest payload that fits, and a session or an edge
over the envelope is refused as not fitting, never with an exception. The size functions and
pinned quotes are in test_socket_v1.py.
"""
import dataclasses
import json

import pytest

from federatedscope.contrib.decentralized.core.accounting import ByteLedger, SessionBudget
from federatedscope.contrib.decentralized.core.types import DirectionalCredits
from federatedscope.contrib.decentralized.core.wire_cost import (
    DEFAULT_ENVELOPE_BYTES,
    FORWARD,
    MAX_ENVELOPE_BYTES,
    OVER_CAP,
    REVERSE,
    GraphEdgeWireContext,
    SessionWireContext,
    resolve_graph_wire_cost,
    resolve_wire_cost,
    validate_envelope_bytes,
)
from federatedscope.contrib.decentralized.simulation import wire_quote  # noqa: F401 - registers socket_v1
from federatedscope.contrib.decentralized.simulation.scenario import (
    ENVELOPE_KEY,
    ScenarioError,
    build_cfg,
    build_factory,
    client_configs,
)
from federatedscope.contrib.decentralized.simulation.socket_v1 import (
    envelope_encoded_size,
    message_id,
)
from federatedscope.contrib.decentralized.tests.simulation.training_fixtures import (
    training_scenario,
)

MiB = 1024 * 1024
LARGE = 512 * MiB


def _session_ctx(envelope=DEFAULT_ENVELOPE_BYTES, gross=10 ** 12, width=3, count=200):
    return SessionWireContext(lo=1, hi=2, event_id=7, checksum=True,
                              manifest_bytes={FORWARD: 900, REVERSE: 900},
                              ack_bytes={FORWARD: 50, REVERSE: 50},
                              gross_bytes={FORWARD: gross, REVERSE: gross},
                              identity_width=width, identity_count_bound=count,
                              envelope_bytes=envelope)


def _graph_ctx(forward, reverse, envelope=DEFAULT_ENVELOPE_BYTES, gross=10 ** 12):
    return GraphEdgeWireContext(lo=1, hi=2, event_id=7, checksum=True,
                                manifest_bytes={FORWARD: 300, REVERSE: 300},
                                snapshot_bytes={FORWARD: forward, REVERSE: reverse},
                                gross_bytes={FORWARD: gross, REVERSE: gross},
                                auxiliary_present={FORWARD: False, REVERSE: False},
                                envelope_bytes=envelope)


def _with_envelope(scenario, envelope):
    return dataclasses.replace(scenario, cfg_overrides=dict(scenario.cfg_overrides,
                                                            **{ENVELOPE_KEY: envelope}))


# -- the configuration ---------------------------------------------------------------------------

def test_the_configuration_default_is_64_mib():
    from federatedscope.core.configs.config import global_cfg

    assert DEFAULT_ENVELOPE_BYTES == 64 * MiB and MAX_ENVELOPE_BYTES == LARGE
    assert int(global_cfg.clone().decentralized.transport.envelope_bytes) == DEFAULT_ENVELOPE_BYTES
    assert resolve_wire_cost("socket_v1").envelope_bytes == DEFAULT_ENVELOPE_BYTES
    assert resolve_graph_wire_cost("socket_v1").envelope_bytes == DEFAULT_ENVELOPE_BYTES


@pytest.mark.parametrize("value", [1.5, "64", True, DEFAULT_ENVELOPE_BYTES - 1,
                                   MAX_ENVELOPE_BYTES + 1])
def test_a_malformed_or_out_of_range_envelope_is_refused(value):
    with pytest.raises(ValueError, match="envelope_bytes"):
        validate_envelope_bytes(value, "socket_v1")


def test_a_larger_envelope_under_the_identity_model_is_refused():
    assert validate_envelope_bytes(MAX_ENVELOPE_BYTES, "socket_v1") == MAX_ENVELOPE_BYTES
    assert validate_envelope_bytes(DEFAULT_ENVELOPE_BYTES, "logical") == DEFAULT_ENVELOPE_BYTES
    with pytest.raises(ValueError, match="prices no envelope"):
        validate_envelope_bytes(LARGE, "logical")
    with pytest.raises(ValueError, match="prices no envelope"):
        resolve_wire_cost("logical", envelope_bytes=LARGE)


def test_the_configuration_refuses_an_envelope_outside_the_allowed_range():
    scenario = training_scenario("pulse_lora", stochastic=False)
    assert int(build_cfg(_with_envelope(scenario, LARGE)).decentralized.transport.envelope_bytes) \
        == LARGE
    with pytest.raises(ValueError, match="envelope_bytes"):
        build_cfg(_with_envelope(scenario, MAX_ENVELOPE_BYTES + 1))


def test_a_per_peer_envelope_is_refused_by_name():
    """Both endpoints of a session price it under one envelope, so it is a run-level value."""
    scenario = training_scenario("pulse_lora", stochastic=False)
    peer = sorted(scenario.peers)[0]
    bad = dataclasses.replace(scenario, peer_overrides={peer: {ENVELOPE_KEY: LARGE}})
    with pytest.raises(ScenarioError, match="per-peer envelope is refused"):
        build_cfg(bad, peer)
    with pytest.raises(ScenarioError, match="per-peer envelope is refused"):
        client_configs(bad)


@pytest.mark.parametrize("method", ["pulse_lora", "tad_lora"])
def test_the_run_prices_its_sessions_and_edges_under_the_configured_envelope(method, tmp_path):
    from federatedscope.contrib.decentralized.core.runner import DecentralizedRunner

    trace = tmp_path / "trace.json"
    for envelope in (DEFAULT_ENVELOPE_BYTES, LARGE):
        scenario = _with_envelope(training_scenario(method, stochastic=False), envelope)
        trace.write_text(json.dumps(scenario.trace))
        cfg = build_cfg(scenario)
        cfg.merge_from_list(["decentralized.opportunities.source", "file",
                             "decentralized.opportunities.path", str(trace)])
        runner = DecentralizedRunner(data={}, config=cfg, factory=build_factory(scenario, cfg))
        for model in (runner.wire_cost, runner.graph_wire_cost):
            assert model.name == "socket_v1" and model.envelope_bytes == envelope


def test_the_envelope_enters_a_wire_context_and_the_manifest_only_when_it_is_not_the_default():
    from federatedscope.contrib.decentralized.core import manifest

    for build in (lambda **kw: _session_ctx(**kw), lambda **kw: _graph_ctx(10, 20, **kw)):
        default = build(envelope=DEFAULT_ENVELOPE_BYTES)
        assert "envelope_bytes" not in default.as_dict()
        large = build(envelope=LARGE)
        assert large.as_dict()["envelope_bytes"] == LARGE
        assert large.digest() != default.digest()
        assert type(default).from_dict(large.as_dict()).envelope_bytes == LARGE
        assert type(default).from_dict(default.as_dict()).envelope_bytes == DEFAULT_ENVELOPE_BYTES

    class Runner:
        def __init__(self, envelope):
            self.wire_cost = resolve_wire_cost("socket_v1", envelope_bytes=envelope)

    assert manifest._envelope_manifest(Runner(DEFAULT_ENVELOPE_BYTES)) == {}
    assert manifest._envelope_manifest(Runner(LARGE)) == {"transport_envelope_bytes": LARGE}


# -- pricing --------------------------------------------------------------------------------------

def test_a_large_edge_below_a_configured_envelope_is_priced_and_admitted():
    """A 100 MB snapshot each way: over the default envelope, under a 512 MiB one."""
    forward = reverse = 100_000_000
    default = resolve_graph_wire_cost("socket_v1")
    large = resolve_graph_wire_cost("socket_v1", envelope_bytes=LARGE)
    for model, envelope, fits in ((default, DEFAULT_ENVELOPE_BYTES, False), (large, LARGE, True)):
        ctx = _graph_ctx(forward, reverse, envelope=envelope)
        budget = SessionBudget((1, 2), DirectionalCredits(10 ** 12, 10 ** 12))
        budget.attach_wire(model, ctx)
        quote = budget.wire_quote(forward, reverse)
        assert all(quote[d]["fits"] is fits for d in (FORWARD, REVERSE)), quote
        assert all(quote[d]["over_cap"] is (not fits) for d in (FORWARD, REVERSE))


def test_one_byte_over_the_envelope_is_priced_out_of_reach():
    """The coordinator's snapshot frame travels raw in one request; at exactly the envelope
    it is priced, one byte more and it is ``OVER_CAP``."""
    model = resolve_graph_wire_cost("socket_v1")
    mid = message_id(7, 3, "receive_snapshot")
    at_cap = DEFAULT_ENVELOPE_BYTES - envelope_encoded_size(mid, 0)
    assert envelope_encoded_size(mid, at_cap) == DEFAULT_ENVELOPE_BYTES
    ok = model.payload(_graph_ctx(at_cap, 10), FORWARD, at_cap, None)
    over = model.payload(_graph_ctx(at_cap + 1, 10), FORWARD, at_cap + 1, None)
    assert ok < OVER_CAP and over == OVER_CAP
    large = resolve_graph_wire_cost("socket_v1", envelope_bytes=LARGE)
    assert large.payload(_graph_ctx(at_cap + 1, 10, envelope=LARGE), FORWARD, at_cap + 1,
                         None) < OVER_CAP


def test_a_planner_search_past_the_envelope_returns_not_affordable_and_never_raises():
    budget = SessionBudget((1, 2), DirectionalCredits(10 ** 12, 10 ** 12))
    budget.attach_wire(resolve_wire_cost("socket_v1"), _session_ctx())
    per_unit = 1_000_000

    def cost(k):
        return 200 + per_unit * k

    slots = budget.affordable(cost, 10_000)          # the credit alone would admit 10,000
    assert 0 < slots < 10_000
    assert budget.wire_quote(cost(slots), cost(slots), units=slots)[REVERSE]["fits"]
    beyond = budget.wire_quote(cost(slots + 1), cost(slots + 1), units=slots + 1)
    assert not beyond[REVERSE]["fits"] and beyond[REVERSE]["over_cap"]
    none_fits = SessionBudget((1, 2), DirectionalCredits(10 ** 12, 10 ** 12))
    none_fits.attach_wire(resolve_wire_cost("socket_v1"), _session_ctx())
    assert none_fits.affordable(lambda k: 70_000_000 * k, 50) == 0


# -- sessions and edges over the envelope ---------------------------------------------------------

def _fat_pair(peer_id, value, extra):
    from federatedscope.contrib.decentralized.tests.core.toy_method import _PACK, ToyAlgorithm

    class Fat(ToyAlgorithm):
        """Plans one whole, fixed payload without searching, as a full-adapter method does."""

        def plan_payload(self, session, budget, remote_view):
            return _PACK.pack(self.value) + b"\0" * extra

        def stage(self, incoming, session):
            return super().stage(incoming[:8], session)

    return Fat(peer_id, value)


def test_a_fixed_payload_over_the_envelope_is_refused_as_not_fitting_and_moves_nothing():
    """A planner that does not search: the executor prices the whole plan, finds a message
    over the envelope and refuses the session by that name - no exception, no payload booked,
    both endpoints told, nothing mixed. Under the larger envelope the same session commits."""
    from federatedscope.contrib.decentralized.core.interactions import compile_pair_sessions
    from federatedscope.contrib.decentralized.core.pair_session import execute_pair_session
    from federatedscope.contrib.decentralized.core.traces import load_trace
    from federatedscope.contrib.decentralized.core.transaction import TransactionEngine

    trace = load_trace({"trace_schema_version": 1, "time_unit": "ms", "duration_unit": "ms",
                        "goodput_unit": "bps", "peers": [1, 2],
                        "frames": [{"index": 0, "time": 0, "edges": [
                            {"endpoints": [1, 2], "duration": 1000,
                             "goodput": 8 * 400_000_000}]}]})
    session = compile_pair_sessions(trace.frame(0))[0]
    for envelope, admitted in ((DEFAULT_ENVELOPE_BYTES, False), (LARGE, True)):
        algos = {1: _fat_pair(1, 10.0, DEFAULT_ENVELOPE_BYTES),
                 2: _fat_pair(2, 20.0, DEFAULT_ENVELOPE_BYTES)}
        ledger = ByteLedger()
        record = execute_pair_session(
            session, algos, TransactionEngine(), ledger,
            wire_cost=resolve_wire_cost("socket_v1", envelope_bytes=envelope))
        if admitted:
            assert record["status"] == "committed", record.get("reason")
            assert ledger.total("payload") > 2 * DEFAULT_ENVELOPE_BYTES
            assert algos[1].value == algos[2].value == 15.0
        else:
            assert record["status"] != "committed"
            assert "envelope in force" in record["reason"], record["reason"]
            assert ledger.total("payload") == 0
            assert algos[1].value == 10.0 and algos[2].value == 20.0
            assert [a for a, _ in algos[1].acknowledgements] == [False]
            assert [a for a, _ in algos[2].acknowledgements] == [False]


def _big_toy(peer_id, value, extra):
    from federatedscope.contrib.decentralized.tests.core.toy_method import _PACK, ToyAlgorithm

    class Big(ToyAlgorithm):
        def snapshot(self):
            return _PACK.pack(self.value) + b"\0" * extra

        def stage_mix(self, weights, snapshots):
            return super().stage_mix(weights, {p: s[:8] for p, s in snapshots.items()})

    return Big(peer_id, value)


def _edge_event(credit_bytes):
    from federatedscope.contrib.decentralized.core.interactions import compile_graph_round
    from federatedscope.contrib.decentralized.core.traces import load_trace

    frame = load_trace({"trace_schema_version": 1, "time_unit": "ms", "duration_unit": "ms",
                        "goodput_unit": "bps", "peers": [1, 2],
                        "frames": [{"index": 0, "time": 0, "edges": [
                            {"endpoints": [1, 2], "duration": 1000,
                             "goodput": 8 * credit_bytes}]}]}).frame(0)
    return compile_graph_round(frame, [1, 2], 0.5)


def test_an_edge_over_the_envelope_moves_no_payload_and_the_ledger_reconciles():
    from federatedscope.contrib.decentralized.core.graph_mixing import execute_graph_round

    extra = DEFAULT_ENVELOPE_BYTES + 4096
    event = _edge_event(200_000_000)
    algos = {1: _big_toy(1, 10.0, extra), 2: _big_toy(2, 20.0, 0)}
    ledger = ByteLedger()
    report = execute_graph_round(event, algos, ledger, wire_cost=resolve_graph_wire_cost("socket_v1"))
    assert len(report["skipped_edges"]) == 1
    assert "envelope in force" in report["skipped_edges"][0]["reason"]
    assert report["realized_matrix"] == [[1.0, 0.0], [0.0, 1.0]]
    assert report["payload_bytes"] == 0 and ledger.total("payload") == 0
    assert ledger.total("control") == report["control_bytes"] > 0
    assert algos[1].value == 10.0 and algos[2].value == 20.0

    # The same edge under the larger envelope is admitted, priced and mixed.
    algos = {1: _big_toy(1, 10.0, extra), 2: _big_toy(2, 20.0, 0)}
    ledger = ByteLedger()
    report = execute_graph_round(event, algos, ledger, wire_cost=resolve_graph_wire_cost(
        "socket_v1", envelope_bytes=LARGE))
    assert report["skipped_edges"] == [] and report["affordable_edges"] == [[1, 2]]
    assert ledger.total("payload") == report["payload_bytes"] > DEFAULT_ENVELOPE_BYTES
    assert algos[1].value == algos[2].value == 15.0
