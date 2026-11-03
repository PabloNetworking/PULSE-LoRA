"""Graph rounds admitted in the wire denomination.

``execute_graph_round`` takes a ``wire_cost`` model, as ``execute_pair_session`` does.
Once both snapshot sizes of an edge are known it builds a ``GraphEdgeWireContext``, and
it drops at planning an edge whose snapshot exchange fits the ledger but not the wire,
naming the direction and the model. Under the logical model, the identity, admission is
the ledger's ``can_afford`` to the byte.
"""

import pytest

from federatedscope.contrib.decentralized.core.accounting import ByteLedger, SessionBudget
from federatedscope.contrib.decentralized.core.graph_mixing import execute_graph_round
from federatedscope.contrib.decentralized.core.interactions import compile_graph_round
from federatedscope.contrib.decentralized.core.traces import load_trace
from federatedscope.contrib.decentralized.core.types import DirectionalCredits
from federatedscope.contrib.decentralized.core.wire_cost import (
    FORWARD,
    LOGICAL,
    REVERSE,
    GraphEdgeWireContext,
    LogicalWireCost,
    SessionWireContext,
    WireCostModel,
    register_graph_wire_cost,
    registered_graph_wire_costs,
    resolve_graph_wire_cost,
)
from federatedscope.contrib.decentralized.tests.core.toy_method import ToyAlgorithm


def frame(goodput_bps=1_000_000, peers=(1, 2)):
    """One matching frame; goodput in bps so credits can be set to the byte. With the
    toy's 8-byte payload a control frame is 46 B and a snapshot frame 54 B."""
    ordered = list(peers)
    edges = [{"endpoints": [ordered[i], ordered[i + 1]], "duration": 1000,
              "goodput": goodput_bps} for i in range(0, len(ordered) - 1, 2)]
    return load_trace({
        "trace_schema_version": 1, "time_unit": "ms", "duration_unit": "ms",
        "goodput_unit": "bps", "peers": sorted(ordered),
        "frames": [{"index": 0, "time": 0, "edges": edges}]}).frame(0)


def algorithms(values):
    return {pid: ToyAlgorithm(pid, value) for pid, value in values.items()}


class Inflating(WireCostModel):
    """A wire-cost model that doubles the reverse payload and keeps a reverse reserve."""

    name = "inflating"

    def __init__(self, factor=2, reserve=100):
        self.factor = factor
        self._reserve = reserve

    def fixed(self, ctx, direction):
        return int(ctx.manifest_bytes[direction])

    def payload(self, ctx, direction, logical_frame_bytes, units):
        return int(logical_frame_bytes) * (self.factor if direction == REVERSE else 1)

    def reserve(self, ctx, direction):
        return self._reserve if direction == REVERSE else 0


# -- the logical model admits exactly what the ledger admits ---------------------------------

def test_the_identity_model_reproduces_the_ledger_admission_to_the_byte():
    """Credits from 25 B to 174 B, one byte at a time, against the ledger's rule stated
    from the frame sizes: the snapshot is admitted exactly when the credit less the
    manifest frame holds the snapshot frame. The logical model, with no reserve and no
    payload expansion, neither refuses an edge that rule admits nor admits one it
    refuses, and the record, ledger rows and state equal those of the executor's
    default, which is the logical model."""
    from federatedscope.contrib.decentralized.core.types import gross_credit_bytes
    from federatedscope.contrib.decentralized.core.wire import (
        control_frame, encoded_size, payload_frame)

    manifest = encoded_size(control_frame("graph_manifest", 1, 2, 0, checksum=True))
    snapshot = encoded_size(payload_frame("graph_snapshot", 1, 2, 0, ToyAlgorithm(1, 1.0).snapshot(),
                                          checksum=True))
    assert (manifest, snapshot) == (46, 54)
    decisions = []
    for goodput in range(200, 1400, 8):
        credit = gross_credit_bytes(goodput, 1_000_000)
        ledger_admits = credit - manifest >= snapshot
        event = compile_graph_round(frame(goodput), [1, 2], 0.5)
        plain, priced = algorithms({1: 10.0, 2: 20.0}), algorithms({1: 10.0, 2: 20.0})
        plain_ledger, priced_ledger = ByteLedger(), ByteLedger()
        without = execute_graph_round(event, plain, plain_ledger)
        with_identity = execute_graph_round(event, priced, priced_ledger,
                                            wire_cost=LogicalWireCost())
        assert bool(with_identity["skipped_edges"]) == (not ledger_admits), (
            goodput, credit, with_identity["skipped_edges"])
        assert with_identity == without, goodput
        assert priced_ledger.records == plain_ledger.records, goodput
        assert [a.value for a in priced.values()] == [a.value for a in plain.values()]
        decisions.append(ledger_admits)
    assert decisions.count(False) > 0 and decisions.count(True) > 0
    # the boundary itself: exactly manifest + snapshot bytes of credit admits
    boundary = compile_graph_round(frame(8 * (manifest + snapshot)), [1, 2], 0.5)
    assert not execute_graph_round(boundary, algorithms({1: 1.0, 2: 2.0}), ByteLedger(),
                                   wire_cost=LogicalWireCost())["skipped_edges"]


# -- fits the ledger, not the wire: dropped at planning, naming direction and model ---------

def test_a_snapshot_that_fits_the_ledger_but_not_the_wire_is_dropped_at_planning():
    # 1,200 bps for 1 s = 150 B per direction: 46 B of control and a 54 B snapshot fit
    # the ledger (150 - 46 = 104 >= 54) and the forward wire (46 + 54 = 100), but not
    # the inflating reverse wire (46 + 108 + 100 > 150).
    event = compile_graph_round(frame(1_200), [1, 2], 0.5)
    control = execute_graph_round(event, algorithms({1: 10.0, 2: 20.0}), ByteLedger())
    assert not control["skipped_edges"], "precondition: the ledger admits this edge"

    algos, ledger = algorithms({1: 10.0, 2: 20.0}), ByteLedger()
    report = execute_graph_round(event, algos, ledger, wire_cost=Inflating())

    assert len(report["skipped_edges"]) == 1
    skipped = report["skipped_edges"][0]
    assert skipped["endpoints"] == [1, 2]
    assert "on the wire reverse" in skipped["reason"]
    assert "priced by the inflating model" in skipped["reason"]
    assert skipped["needed_bytes"] == control["payload_bytes"] // 2
    assert report["realized_matrix"] == [[1.0, 0.0], [0.0, 1.0]]
    assert report["mixed_peers"] == [] and report["payload_bytes"] == 0
    assert ledger.total("payload") == 0 and ledger.total("control") == control["control_bytes"]
    assert algos[1].value == 10.0 and algos[2].value == 20.0


def test_the_ledger_drop_comes_first_and_keeps_its_own_reason():
    # 400 bps for 1 s = 50 B: after 46 B of control the 54 B snapshot does not fit
    event = compile_graph_round(frame(400), [1, 2], 0.5)
    report = execute_graph_round(event, algorithms({1: 1.0, 2: 2.0}), ByteLedger(),
                                 wire_cost=Inflating())
    assert report["skipped_edges"][0]["reason"] == \
        "complete snapshot does not fit after control overhead"


# -- the graph-edge wire context ---------------------------------------------------------------

def test_graph_and_pairwise_contexts_never_share_a_digest_and_round_trip():
    graph = GraphEdgeWireContext(
        lo=1, hi=2, event_id=4, checksum=True,
        manifest_bytes={FORWARD: 46, REVERSE: 46}, snapshot_bytes={FORWARD: 54, REVERSE: 54},
        gross_bytes={FORWARD: 150, REVERSE: 150},
        auxiliary_present={FORWARD: False, REVERSE: False})
    pair = SessionWireContext(
        lo=1, hi=2, event_id=4, checksum=True,
        manifest_bytes={FORWARD: 46, REVERSE: 46}, ack_bytes={FORWARD: 0, REVERSE: 0},
        gross_bytes={FORWARD: 150, REVERSE: 150},
        auxiliary_present={FORWARD: False, REVERSE: False})
    assert graph.digest() != pair.digest()
    assert graph.as_dict()["form"] == "graph_round"
    assert GraphEdgeWireContext.from_dict(graph.as_dict()) == graph
    assert graph.ack_bytes == {FORWARD: 0, REVERSE: 0}, "a graph edge sends no acks"
    other = GraphEdgeWireContext.from_dict(dict(graph.as_dict(),
                                                snapshot_bytes={FORWARD: 55, REVERSE: 54}))
    assert other.digest() != graph.digest()


def test_the_identity_prices_a_graph_edge_as_the_control_the_ledger_charged():
    ctx = GraphEdgeWireContext(
        lo=1, hi=2, event_id=0, checksum=True,
        manifest_bytes={FORWARD: 46, REVERSE: 47}, snapshot_bytes={FORWARD: 54, REVERSE: 54},
        gross_bytes={FORWARD: 150, REVERSE: 150},
        auxiliary_present={FORWARD: False, REVERSE: False})
    budget = SessionBudget((1, 2), DirectionalCredits(150, 150))
    budget.charge_control(1, 46)
    budget.charge_control(2, 47)
    budget.attach_wire(LogicalWireCost(), ctx)
    quote = budget.wire_quote(54, 54)
    assert quote[FORWARD]["fixed"] == 46 and quote[REVERSE]["fixed"] == 47
    assert quote[FORWARD]["reserve"] == 0 and quote[REVERSE]["payload"] == 54
    assert budget.wire_fits(54, 54) == budget.can_afford(54)


# -- the registry: resolved by name, refused by name --------------------------------------------

def test_the_graph_registry_is_resolved_by_name_and_refused_by_name():
    assert isinstance(resolve_graph_wire_cost(LOGICAL), LogicalWireCost)
    assert LOGICAL in registered_graph_wire_costs()
    with pytest.raises(ValueError, match="graph round.*not registered"):
        resolve_graph_wire_cost("no_such_transport")

    class Fake(WireCostModel):
        name = "fake_graph_model"

    register_graph_wire_cost("fake_graph_model", Fake)
    assert isinstance(resolve_graph_wire_cost("fake_graph_model"), Fake)
    with pytest.raises(ValueError, match="already registered"):
        register_graph_wire_cost("fake_graph_model", Inflating)
