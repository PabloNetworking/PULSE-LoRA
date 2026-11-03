"""The method-agnostic control negotiation of a pair session.

Each endpoint sends a manifest, reads the remote one, plans its payload against
the session budget and its view of the remote, and receives one acknowledgement.
A component-selecting double runs through the same executor.
"""
import struct

import pytest

from federatedscope.contrib.decentralized.core.accounting import ByteLedger
from federatedscope.contrib.decentralized.core.interactions import compile_pair_sessions
from federatedscope.contrib.decentralized.core.pair_session import (
    ACK_COMMITTED,
    ACK_FAILED,
    execute_pair_session,
)
from federatedscope.contrib.decentralized.core.traces import load_trace
from federatedscope.contrib.decentralized.core.transaction import TransactionEngine
from federatedscope.contrib.decentralized.core.wire import (
    control_frame,
    decode_frame,
    encoded_size,
)
from federatedscope.contrib.decentralized.tests.core.toy_method import ToyAlgorithm


def session(goodput_bps=1_000_000):
    trace = load_trace({
        "trace_schema_version": 1, "time_unit": "ms", "duration_unit": "ms",
        "goodput_unit": "bps", "peers": [1, 2],
        "frames": [{"index": 0, "time": 0, "edges": [
            {"endpoints": [1, 2], "duration": 1000, "goodput": goodput_bps}]}],
    })
    return compile_pair_sessions(trace.frame(0))[0]


def toys():
    return {1: ToyAlgorithm(1, 10.0), 2: ToyAlgorithm(2, 20.0)}


def run(algorithms, goodput_bps=1_000_000, ledger=None):
    ledger = ledger or ByteLedger()
    record = execute_pair_session(session(goodput_bps), algorithms,
                                  TransactionEngine(), ledger)
    return record, ledger


# --- the four stages ------------------------------------------------------------

def test_manifests_are_produced_consumed_and_budgeted():
    algorithms = toys()
    seen = {}
    for pid, algorithm in algorithms.items():
        original = algorithm.consume_manifest
        algorithm.consume_manifest = (
            lambda blob, s, _p=pid, _o=original: seen.setdefault(_p, blob) or _o(blob, s))

    record, ledger = run(algorithms)

    assert set(seen) == {1, 2}, "both endpoints must consume the remote manifest"
    assert all(len(blob) > 0 for blob in seen.values()), "manifests must not be empty"
    assert record["control_bytes"] > 0
    assert record["control_bytes_forward"] > 0 and record["control_bytes_reverse"] > 0


def test_control_is_charged_before_payload_credit_is_computed():
    record, _ = run(toys())
    offered = record["offered_forward_bytes"]
    assert record["offered_payload_credit"] == offered - record["control_bytes"] // 2


def test_acknowledgement_is_delivered_exactly_once_per_endpoint():
    algorithms = toys()
    record, _ = run(algorithms)
    assert record["status"] == "committed"
    for algorithm in algorithms.values():
        assert len(algorithm.acknowledgements) == 1
        committed, _reason = algorithm.acknowledgements[0]
        assert committed is True


def test_acknowledgement_reports_failure_too():
    algorithms = toys()
    record, _ = run(algorithms, goodput_bps=200)
    assert record["status"] == "control_exceeds_capacity"
    for algorithm in algorithms.values():
        assert algorithm.acknowledgements[0][0] is False


def test_ack_frames_have_a_fixed_width_so_they_can_be_budgeted_up_front():
    """The status is unknown when the frame is budgeted, so width must not depend on it."""
    committed = control_frame("ack", 1, 2, 0, ACK_COMMITTED)
    failed = control_frame("ack", 1, 2, 0, ACK_FAILED)
    assert encoded_size(committed) == encoded_size(failed)
    assert decode_frame(committed.encode() if hasattr(committed, "encode") else
                        __import__("federatedscope.contrib.decentralized.core.wire",
                                   fromlist=["encode_frame"]).encode_frame(committed)).body \
        == ACK_COMMITTED


def test_a_rejected_manifest_stops_the_session_before_any_payload():
    algorithms = toys()

    def refuse(blob, s):
        raise ValueError("incompatible peer")

    algorithms[1].consume_manifest = refuse
    before = {pid: a.value for pid, a in algorithms.items()}

    record, ledger = run(algorithms)

    assert record["status"] == "control_only"
    assert "rejected the remote manifest" in record["reason"]
    assert record["payload_bytes"] == 0
    assert {pid: a.value for pid, a in algorithms.items()} == before
    assert ledger.total("control") > 0, "the manifests were still sent"


def test_control_exceeding_capacity_is_reported_distinctly_from_no_payload():
    tiny, _ = run(toys(), goodput_bps=200)
    assert tiny["status"] == "control_exceeds_capacity"
    assert "short by" in tiny["reason"]


# --- a component-selecting method ---------------------------------------------------

class InventoryAlgorithm(ToyAlgorithm):
    """A component-selecting double.

    Its manifest declares the age and value of each component. It plans the
    subset that fits the budget, ranked by disagreement with the remote's
    declared values, then by the remote's age, oldest first. It runs through the
    same pair-session executor and transaction engine as any method.
    """

    COMPONENT_BYTES = 8

    def __init__(self, peer_id, value, components=None, **kwargs):
        super().__init__(peer_id, value, **kwargs)
        self.components = components or {p: float(p * peer_id) for p in range(6)}
        self.ages = {p: p for p in self.components}
        self.selected = None

    def build_manifest(self, session):
        return struct.pack(f"!I{len(self.components)}I{len(self.components)}d",
                           len(self.components),
                           *[self.ages[p] for p in sorted(self.components)],
                           *[self.components[p] for p in sorted(self.components)])

    def consume_manifest(self, remote_manifest, session):
        (count,) = struct.unpack_from("!I", remote_manifest, 0)
        rest = struct.unpack_from(f"!{count}I{count}d", remote_manifest, 4)
        return {"ages": list(rest[:count]), "values": list(rest[count:])}

    def plan_payload(self, session, budget, remote_view):
        # disagreement against the remote's declared values, oldest-first tie-break
        scored = sorted(
            self.components,
            key=lambda p: (-abs(self.components[p] - remote_view["values"][p]),
                           -remote_view["ages"][p], p),
        )
        affordable = max(0, (budget.payload_credit() - 64) // self.COMPONENT_BYTES)
        self.selected = sorted(scored[:affordable])
        if not self.selected:
            return None
        return struct.pack(f"!I{len(self.selected)}I", len(self.selected), *self.selected)

    def stage(self, incoming, session):
        (count,) = struct.unpack_from("!I", incoming, 0)
        ids = struct.unpack_from(f"!{count}I", incoming, 4)
        token = self._next_token
        self._next_token += 1
        self._staged[token] = {"previous": dict(self.components), "received": list(ids),
                               "next": dict(self.components), "committed": False}
        return token

    def commit(self, token):
        staged = self._staged[token]
        self.components = staged["next"]
        staged["committed"] = True
        self._version += 1


def inventory_peers():
    return {1: InventoryAlgorithm(1, 10.0), 2: InventoryAlgorithm(2, 20.0)}


def test_a_component_selecting_method_runs_through_the_pair_session_executor():
    algorithms = inventory_peers()
    record, _ = run(algorithms)

    assert record["status"] == "committed"
    for algorithm in algorithms.values():
        assert algorithm.selected, "the plugin selected components from remote metadata"


def test_selection_shrinks_with_the_budget():
    """The budget reaches the plugin, so a tighter session selects fewer components.

    2000 bps gives 250 B per direction; the handshake takes 152 B of it, which
    leaves room for four of the six components.
    """
    wide = inventory_peers()
    run(wide, goodput_bps=1_000_000)
    generous = len(wide[1].selected)
    assert generous == 6, "a generous budget should carry every component"

    narrow = inventory_peers()
    record, _ = run(narrow, goodput_bps=2_000)
    constrained = len(narrow[1].selected or [])

    assert 0 < constrained < generous, f"expected a partial selection, got {constrained}"
    assert record["status"] == "committed", record["reason"]


def test_selection_is_deterministic():
    first = inventory_peers()
    second = inventory_peers()
    run(first)
    run(second)
    assert first[1].selected == second[1].selected
