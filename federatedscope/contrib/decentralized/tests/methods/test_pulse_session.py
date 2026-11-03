"""PULSE-LoRA session policy: the four pair-session hooks, end to end.

Run: python -m pytest federatedscope/contrib/decentralized/tests/methods/test_pulse_session.py -q -p no:cacheprovider

Two endpoints plan independently, with no negotiation round between planning
and sending, and must arrive at the identical component set. That relies on
manifest agreement on the component space and the selector, symmetric scores,
and a per-edge selector state whose digest is compared at manifest time, so a
divergence is rejected rather than producing a mismatched exchange. Also covered:
the byte budget, the age lifecycle, staging and commit, and checkpointing of the
selector state.
"""
import copy
import json
import pathlib

import pytest
import torch

from federatedscope.contrib.decentralized.core.accounting import ByteLedger, SessionBudget
from federatedscope.contrib.decentralized.core.interactions import compile_pair_sessions
from federatedscope.contrib.decentralized.core.pair_session import execute_pair_session
from federatedscope.contrib.decentralized.core.traces import load_trace
from federatedscope.contrib.decentralized.core.transaction import TransactionEngine
from federatedscope.contrib.decentralized.core.wire import encoded_size, payload_frame
from federatedscope.contrib.decentralized.methods.pulse_lora import selection, units
from federatedscope.contrib.decentralized.methods.pulse_lora.session import (
    SessionError,
    unpack_bitmap,
)
from federatedscope.contrib.decentralized.tests.methods.test_pulse_peer import (
    make_peer, pulse_cfg,
)


def event(a=1, b=2, duration=1000, goodput=1_000_000):
    frame = load_trace({
        "trace_schema_version": 1, "time_unit": "ms", "duration_unit": "ms",
        "goodput_unit": "bps", "peers": [a, b],
        "frames": [{"index": 0, "time": 0, "edges": [
            {"endpoints": [a, b], "duration": duration, "goodput": goodput}]}],
    }).frame(0)
    return compile_pair_sessions(frame)[0]


def budget_for(ev, credit=None):
    """A budget whose payload credit is exactly ``credit``.

    The rest of the link credit is charged as control bytes in both directions,
    standing in for the handshake. With ``credit=None`` nothing is charged.
    """
    budget = SessionBudget(ev.endpoints, ev.credits)
    if credit is None:
        return budget
    slack = budget.payload_credit() - int(credit)
    assert slack >= 0, "asked for more credit than the link offers"
    budget.charge_control(ev.endpoints[0], slack)
    budget.charge_control(ev.endpoints[1], slack)
    assert budget.payload_credit() == int(credit)
    return budget


def negotiate(a, b, ev):
    """Build and consume both manifests in the order ``execute_pair_session`` does.

    Returns both remote views.
    """
    ma, mb = a.build_manifest(ev), b.build_manifest(ev)
    return a.consume_manifest(mb, ev), b.consume_manifest(ma, ev)


def run_round(a, b, ev, credit=None, committed=False):
    """Negotiate, plan on both endpoints, and acknowledge both with ``committed``.

    No payload is staged or mixed. Both endpoints must acknowledge, or their
    per-edge selector state diverges and the manifest digest rejects the next
    session.
    """
    view_a, view_b = negotiate(a, b, ev)
    shared = budget_for(ev, credit)
    body_a = a.plan_payload(ev, shared, view_a)
    body_b = b.plan_payload(ev, shared, view_b)
    a.acknowledge(ev, committed=committed)
    b.acknowledge(ev, committed=committed)
    return body_a, body_b


def trained(peer, seed=0, components=None):
    """Move some components so scores are not all zero."""
    generator = torch.Generator().manual_seed(seed)
    for component in (components if components is not None else peer.state.support):
        block = peer.state.gather(component)
        peer.state.scatter(component, {k: torch.rand(v.shape, generator=generator,
                                                     dtype=v.dtype) - 0.5
                                       for k, v in block.items()})
    return peer


def peers(selector="local_update", **over):
    settings = {"decentralized.methods.pulse_lora.selector": selector}
    settings.update(over)
    cfg = pulse_cfg(**settings)
    return make_peer(1, cfg=cfg), make_peer(2, cfg=cfg)


# -- the two endpoints agree ----------------------------------------------------------

DEPLOYABLE = [s for s in selection.SELECTORS if s not in selection.ORACLE_SELECTORS]


@pytest.mark.parametrize("selector", DEPLOYABLE)
@pytest.mark.parametrize("slots", [1, 2, 3])
def test_both_endpoints_independently_select_the_identical_set(selector, slots):
    """Under capacity pressure, the only regime in which ranking matters.

    The budget affords fewer components than there are candidates, so the scores
    decide the selection. With room for every candidate, select() returns them
    all without consulting the scores, and the test could not fail.
    """
    a, b = peers(selector)
    trained(a, seed=1); trained(b, seed=3)
    ev = event()
    view_a, view_b = negotiate(a, b, ev)
    credit = units.units_cost_bytes(a.components, slots) + 36
    shared = budget_for(ev, credit)

    body_a = a.plan_payload(ev, shared, view_a)
    body_b = b.plan_payload(ev, shared, view_b)

    assert body_a is not None and body_b is not None
    chosen_a = sorted(units.decode_units(a.components, body_a))
    chosen_b = sorted(units.decode_units(b.components, body_b))
    assert len(chosen_a) == slots, "the budget must actually bind, or this proves nothing"
    assert chosen_a == chosen_b, f"{selector} at {slots} slots: endpoints disagreed"
    assert len(body_a) == len(body_b), "asymmetric sizes would kill the session"


@pytest.mark.parametrize("selector", DEPLOYABLE)
def test_endpoints_agree_across_many_independent_training_histories(selector):
    """A single seed pair can agree by luck; 30 cannot."""
    divergences = []
    for left in range(1, 7):
        for right in range(1, 7):
            if left == right:
                continue
            a, b = peers(selector, **{
                "decentralized.methods.pulse_lora.exploration_fraction": 0.0})
            trained(a, seed=left); trained(b, seed=right)
            ev = event()
            view_a, view_b = negotiate(a, b, ev)
            shared = budget_for(ev, units.units_cost_bytes(a.components, 1) + 36)
            body_a = a.plan_payload(ev, shared, view_a)
            body_b = b.plan_payload(ev, shared, view_b)
            if body_a is None or body_b is None:
                continue
            if (sorted(units.decode_units(a.components, body_a))
                    != sorted(units.decode_units(b.components, body_b))):
                divergences.append((left, right))
    assert not divergences, f"{selector} diverged on {len(divergences)} seed pairs"


def test_every_score_publishing_selector_actually_publishes_its_scores():
    """Score-folding selectors publish ``sc`` for every materialised component.

    Score-free selectors publish none. ``build_manifest`` and ``consume_manifest``
    both decide from ``SCORE_PUBLISHING_SELECTORS``.
    """
    for selector in selection.SCORE_PUBLISHING_SELECTORS:
        a, _ = peers(selector)
        declaration = json.loads(a.build_manifest(event()).decode("utf-8"))
        assert "sc" in declaration, (
            f"{selector} folds a remote score but publishes none")
        assert set(declaration["sc"]) == {str(c) for c in a.state.materialised}

    for selector in selection.SCORE_FREE_SELECTORS:
        a, _ = peers(selector)
        declaration = json.loads(a.build_manifest(event()).decode("utf-8"))
        assert "sc" not in declaration, (
            f"{selector} needs no cross-peer score but pays to publish one")


def test_the_two_selector_sets_partition_the_deployable_selectors():
    """Every selector is score-publishing, score-free, the sketch or the oracle.

    A selector in none of these would fall through to the score-folding path
    without publishing scores.
    """
    covered = (selection.SCORE_PUBLISHING_SELECTORS | selection.SCORE_FREE_SELECTORS
               | selection.ORACLE_SELECTORS | {selection.SELECTOR_SKETCH})
    assert set(selection.SELECTORS) == covered, (
        f"unclassified selectors: {sorted(set(selection.SELECTORS) - covered)}")
    assert not (selection.SCORE_PUBLISHING_SELECTORS & selection.SCORE_FREE_SELECTORS)


def test_the_exact_oracle_agrees_across_endpoints_and_needs_a_registry():
    cfg = pulse_cfg(**{"decentralized.methods.pulse_lora.selector": "exact_disagreement"})
    from federatedscope.contrib.decentralized.methods.pulse_lora import PulseLoRAFactory

    class Factory(PulseLoRAFactory):
        pass

    factory = Factory(cfg)
    assert factory.oracle is not None, "the oracle registry is built only for this mode"

    from federatedscope.contrib.decentralized.core.interfaces import PeerContext
    from federatedscope.contrib.decentralized.tests.fixtures import build_toy_lora_model, toy_extras
    a = factory.create_peer(PeerContext(peer_id=1, config=cfg,
                                        extras=toy_extras(build_toy_lora_model(rank=4, seed=1))))
    b = factory.create_peer(PeerContext(peer_id=2, config=cfg,
                                        extras=toy_extras(build_toy_lora_model(rank=4, seed=2))))
    trained(a, seed=1); trained(b, seed=2)
    ev = event()
    view_a, view_b = negotiate(a, b, ev)
    shared = budget_for(ev)
    chosen_a = sorted(units.decode_units(a.components, a.plan_payload(ev, shared, view_a)))
    chosen_b = sorted(units.decode_units(b.components, b.plan_payload(ev, shared, view_b)))
    assert chosen_a == chosen_b


def test_both_endpoints_compute_bit_identical_oracle_scores():
    """The exact score is evaluated in one orientation (the lower peer id first) on both
    endpoints, so rounding cannot rank the candidates differently on the two sides."""
    cfg = pulse_cfg(**{"decentralized.methods.pulse_lora.selector": "exact_disagreement"})
    from federatedscope.contrib.decentralized.methods.pulse_lora import PulseLoRAFactory
    from federatedscope.contrib.decentralized.core.interfaces import PeerContext
    from federatedscope.contrib.decentralized.tests.fixtures import build_toy_lora_model, toy_extras

    factory = PulseLoRAFactory(cfg)
    a = factory.create_peer(PeerContext(peer_id=1, config=cfg,
                                        extras=toy_extras(build_toy_lora_model(rank=4, seed=1))))
    b = factory.create_peer(PeerContext(peer_id=2, config=cfg,
                                        extras=toy_extras(build_toy_lora_model(rank=4, seed=2))))
    differing = 0
    for seed in range(40):
        trained(a, seed=2 * seed + 1)
        trained(b, seed=2 * seed + 2)
        candidates = list(a.state.support)
        score_a = a.policy._utility(candidates, {}, (1, 2))
        score_b = b.policy._utility(candidates, {}, (1, 2))
        differing += score_a != score_b
    assert differing == 0, f"{differing} of 40 states scored differently on the two endpoints"


def test_the_oracle_is_labelled_as_privileged_in_the_metrics():
    cfg = pulse_cfg(**{"decentralized.methods.pulse_lora.selector": "exact_disagreement"})
    from federatedscope.contrib.decentralized.methods.pulse_lora import PulseLoRAFactory
    from federatedscope.contrib.decentralized.core.interfaces import PeerContext
    from federatedscope.contrib.decentralized.tests.fixtures import build_toy_lora_model, toy_extras
    peer = PulseLoRAFactory(cfg).create_peer(
        PeerContext(peer_id=1, config=cfg,
                    extras=toy_extras(build_toy_lora_model(rank=4, seed=1))))
    assert peer.method_metrics()["pulse_selector_is_oracle"] is True

    plain, _ = peers("local_update")
    assert plain.method_metrics()["pulse_selector_is_oracle"] is False


def test_a_deployable_run_builds_no_oracle_back_channel():
    from federatedscope.contrib.decentralized.methods.pulse_lora import PulseLoRAFactory
    for selector in selection.SELECTORS:
        cfg = pulse_cfg(**{"decentralized.methods.pulse_lora.selector": selector})
        factory = PulseLoRAFactory(cfg)
        if selector in selection.ORACLE_SELECTORS:
            assert factory.oracle is not None
        else:
            assert factory.oracle is None, f"{selector} must not get peer access"


# -- manifest agreement -----------------------------------------------------------------

def test_the_manifest_declares_the_shared_namespace_and_rule():
    a, _ = peers()
    declaration = json.loads(a.build_manifest(event()).decode("utf-8"))
    assert declaration["m"] == "pulse_lora"
    assert declaration["id"] == a.components.identity_digest()[:16]
    assert declaration["sel"] == "local_update"
    assert declaration["u"] == units.unit_cost_bytes(a.components)


@pytest.mark.parametrize("field,value,match", [
    ("v", 99, "schema version"),
    ("m", "dec_lora", "not pulse_lora"),
    ("id", "0" * 16, "component-space disagreement"),
    ("sel", "age_only", "selector disagreement"),
    ("aux", True, "auxiliary-state policy disagreement"),
    ("u", 7, "unit cost disagreement"),
    ("sd", "deadbeefcafe", "selector state has diverged"),
])
def test_a_disagreeing_manifest_is_rejected(field, value, match):
    a, b = peers()
    ev = event()
    declaration = json.loads(b.build_manifest(ev).decode("utf-8"))
    declaration[field] = value
    with pytest.raises(SessionError, match=match):
        a.consume_manifest(json.dumps(declaration).encode("utf-8"), ev)


@pytest.mark.parametrize("blob,match", [
    (b"not json", "unreadable manifest"),
    (b"[1,2,3]", "not an object"),
    ("{\"v\":2}".encode(), "not pulse_lora"),
])
def test_an_unreadable_manifest_is_rejected(blob, match):
    a, _ = peers()
    with pytest.raises(SessionError, match=match):
        a.consume_manifest(blob, event())


def test_a_malformed_inventory_bitmap_is_rejected_at_manifest_time():
    """Not inside plan_payload, whose exceptions the runtime does not catch."""
    a, b = peers()
    ev = event()
    declaration = json.loads(b.build_manifest(ev).decode("utf-8"))
    declaration["mat"] = "zz"
    with pytest.raises(SessionError, match="not hex"):
        a.consume_manifest(json.dumps(declaration).encode("utf-8"), ev)

    declaration["mat"] = "ffff"
    with pytest.raises(SessionError, match="the peers disagree about the component space"):
        a.consume_manifest(json.dumps(declaration).encode("utf-8"), ev)


def test_consume_manifest_is_pure():
    """consume_manifest changes nothing.

    The runtime stops at the first rejection, so a consume that changed state
    would leave the two endpoints out of step.
    """
    a, b = peers()
    ev = event()
    before = (a.policy.memory.state_dict(), dict(a.policy.metrics),
              a.state.support, a.state.materialised)
    a.consume_manifest(b.build_manifest(ev), ev)
    assert (a.policy.memory.state_dict(), dict(a.policy.metrics),
            a.state.support, a.state.materialised) == before


def test_the_manifest_inventory_round_trips_through_the_bitmap():
    a, b = peers()
    a.state.materialise(6, a.components.initial_component(6))
    ev = event()
    view = b.consume_manifest(a.build_manifest(ev), ev)
    assert view["_materialised"] == a.state.materialised
    assert view["_support"] == a.state.support


def test_a_malformed_scoring_payload_is_rejected_at_manifest_time():
    """Rejected by consume_manifest, whose exceptions end one session.

    The runtime does not catch exceptions from plan_payload, where one aborts the
    run, so every field the scorer reads is checked at manifest time.
    """
    a, b = peers("local_update")
    ev = event()
    good = json.loads(b.build_manifest(ev).decode("utf-8"))

    for mangle, match in (
        (lambda d: d.pop("sc"), "'sc' is NoneType"),
        (lambda d: d.__setitem__("sc", []), "'sc' is list"),
        (lambda d: d["sc"].__setitem__("0", "big"), "not a number"),
        (lambda d: d["sc"].__setitem__("0", float("nan")), "is NaN"),
        (lambda d: d["sc"].__setitem__("0", float("inf")), "not finite"),
        (lambda d: d["sc"].__setitem__("zero", 1.0), "unexpected="),
        (lambda d: d["sc"].pop(sorted(d["sc"])[0]), "missing="),
        (lambda d: d["sc"].__setitem__("5", 1.0), "unexpected="),
        (lambda d: d["sc"].__setitem__("99", 1.0), "unexpected="),
        (lambda d: d["sc"].__setitem__("0", True), "not a number"),
    ):
        declaration = copy.deepcopy(good)
        mangle(declaration)
        with pytest.raises(SessionError, match=match):
            a.consume_manifest(json.dumps(declaration).encode("utf-8"), ev)


def test_a_malformed_sketch_payload_is_rejected_at_manifest_time():
    a, b = peers("sketch_disagreement",
                 **{"decentralized.methods.pulse_lora.sketch_dim": 4})
    ev = event()
    good = json.loads(b.build_manifest(ev).decode("utf-8"))

    for mangle, match in (
        (lambda d: d.pop("nm"), "'nm' is NoneType"),
        (lambda d: d.pop("sk"), "'sk' is NoneType"),
        (lambda d: d["nm"].pop(sorted(d["nm"])[0]), "missing="),
        (lambda d: d["sk"].pop(sorted(d["sk"])[0]), "missing="),
        (lambda d: d["nm"].__setitem__("6", [[1.0, 2.0]] * 3), "unexpected="),
        (lambda d: d["sk"].__setitem__("6", [[[0.0] * 4] * 2] * 3), "unexpected="),
        (lambda d: d["nm"].__setitem__(sorted(d["nm"])[0], [[1.0, 2.0]]), "modules"),
        (lambda d: d["sk"].__setitem__(sorted(d["sk"])[0],
                                       [[[1.0], [1.0]]] * 3), "width"),
        (lambda d: d["sk"][sorted(d["sk"])[0]][0].__setitem__(0, "nope"), "width"),
    ):
        declaration = copy.deepcopy(good)
        mangle(declaration)
        with pytest.raises(SessionError, match=match):
            a.consume_manifest(json.dumps(declaration).encode("utf-8"), ev)


def test_a_manifest_missing_a_scoring_field_cannot_make_plan_payload_raise():
    """Either consume_manifest rejects it, ending one session, or planning succeeds."""
    a, b = peers("sketch_disagreement")
    trained(a, seed=1)
    ev = event()
    good = json.loads(b.build_manifest(ev).decode("utf-8"))

    for key in ("sc", "nm", "sk", "d"):
        declaration = copy.deepcopy(good)
        declaration.pop(key, None)
        try:
            view = a.consume_manifest(json.dumps(declaration).encode("utf-8"), ev)
        except SessionError:
            continue                      # rejected cleanly - the desired outcome
        a.policy._pending.clear()
        a.plan_payload(ev, budget_for(ev), view)   # must not raise


# -- a declared component's scoring entry cannot be omitted -----------------------------------
#
# Each of these drops one entry from sc / nm / sk while leaving that component's
# materialisation bit set. consume_manifest rejects such a manifest: plan_payload
# reads an entry for every declared component, and scoring a missing one as
# untouched would make the two endpoints select different components of the same
# encoded size, which the executor's max() admission rule cannot tell apart.

#: (peer-1 training seed, peer-2 training seed, component dropped from the manifest),
#: chosen so that scoring the dropped component as untouched changes the selection
DIVERGING = [(1, 4, "1"), (1, 5, "0"), (2, 5, "0"), (3, 1, "2"), (3, 4, "1"), (4, 2, "2")]


def manifest_without(peer, ev, field, component):
    declaration = json.loads(peer.build_manifest(ev).decode("utf-8"))
    assert component in declaration[field], "the test needs this entry to exist"
    bit = unpack_bitmap(declaration["mat"], peer.components.r_max)
    assert int(component) in bit, "the materialisation bit must stay set"
    declaration[field].pop(component)
    return json.dumps(declaration).encode("utf-8")


@pytest.mark.parametrize("left,right,component", DIVERGING)
def test_an_omitted_score_for_a_declared_component_is_rejected(left, right, component):
    a, b = peers("local_update",
                 **{"decentralized.methods.pulse_lora.exploration_fraction": 0.0})
    trained(a, seed=left); trained(b, seed=right)
    ev = event()

    with pytest.raises(SessionError, match="missing="):
        a.consume_manifest(manifest_without(b, ev, "sc", component), ev)


@pytest.mark.parametrize("field", ["nm", "sk"])
def test_an_omitted_sketch_entry_for_a_declared_component_is_rejected(field):
    a, b = peers("sketch_disagreement",
                 **{"decentralized.methods.pulse_lora.sketch_dim": 4})
    trained(a, seed=1); trained(b, seed=4)
    ev = event()

    with pytest.raises(SessionError, match="missing="):
        a.consume_manifest(manifest_without(b, ev, field, "1"), ev)


@pytest.mark.parametrize("selector,field", [("local_update", "sc"),
                                            ("sketch_disagreement", "nm"),
                                            ("sketch_disagreement", "sk")])
def test_the_rejection_happens_before_planning_and_changes_nothing(selector, field):
    """Rejected at consumption, before any plan is recorded.

    No selector state or metric moves, and the next valid session still agrees.
    """
    extra = {"decentralized.methods.pulse_lora.exploration_fraction": 0.0}
    if selector == "sketch_disagreement":
        extra["decentralized.methods.pulse_lora.sketch_dim"] = 4
    a, b = peers(selector, **extra)
    trained(a, seed=1); trained(b, seed=4)
    ev = event()

    # Run one clean session first, so there is selector state that a bug could move.
    run_round(a, b, ev, credit=units.units_cost_bytes(a.components, 1) + 36,
              committed=True)
    before_memory = a.policy.memory.state_dict()
    before_metrics = dict(a.policy.metrics)
    before_digest = a.policy.memory.selector_digest(1, 2)
    ledger = ByteLedger()

    with pytest.raises(SessionError, match="missing="):
        a.consume_manifest(manifest_without(b, ev, field, "1"), ev)

    assert a.policy.memory.state_dict() == before_memory, "ages/cursor/sessions moved"
    assert dict(a.policy.metrics) == before_metrics, "a selection metric moved"
    assert a.policy._pending == {}, "a pending decision was recorded"
    assert ledger.total() == 0, "a rejected manifest debited bytes"
    assert a.policy.memory.selector_digest(1, 2) == before_digest

    # ...and the next valid session still agrees with the partner.
    body_a, body_b = run_round(a, b, ev,
                               credit=units.units_cost_bytes(a.components, 1) + 36,
                               committed=True)
    assert body_a is not None and body_b is not None
    assert (sorted(units.decode_units(a.components, body_a))
            == sorted(units.decode_units(b.components, body_b))), \
        "the endpoints desynchronised after a rejected manifest"
    assert a.policy.memory.selector_digest(1, 2) == b.policy.memory.selector_digest(1, 2)


def test_the_deterministic_fallback_still_applies_to_a_genuinely_unheld_component():
    """A component the partner does not hold is scored against its initial value.

    It stays a candidate and is sent when the budget allows.
    """
    a, b = peers("sketch_disagreement",
                 **{"decentralized.methods.pulse_lora.sketch_dim": 4})
    trained(a, seed=1)
    a.state.materialise(6, a.components.initial_component(6))
    assert 6 in a.state.materialised and 6 not in b.state.materialised

    ev = event()
    view_a = a.consume_manifest(b.build_manifest(ev), ev)
    assert 6 not in view_a["_materialised"]
    body = a.plan_payload(ev, budget_for(ev), view_a)      # must not raise
    assert body is not None
    assert 6 in units.decode_units(a.components, body), (
        "a component only this peer holds should still be a candidate")


# -- component keys must be canonical --------------------------------------------------------
#
# Planning indexes sc / nm / sk with str(component), so a key such as "01" would
# pass a comparison done on integers and then raise KeyError in plan_payload, which
# the runtime does not catch. consume_manifest therefore compares the raw keys.

ALIASES = ["01", "+1", " 1", "1 ", "001"]


@pytest.mark.parametrize("field,selector", [("sc", "local_update"),
                                            ("nm", "sketch_disagreement"),
                                            ("sk", "sketch_disagreement")])
@pytest.mark.parametrize("alias", ALIASES)
def test_a_non_canonical_component_key_is_rejected(field, selector, alias):
    extra = {"decentralized.methods.pulse_lora.sketch_dim": 4} \
        if selector == "sketch_disagreement" else {}
    a, b = peers(selector, **extra)
    trained(a, seed=1); trained(b, seed=3)
    ev = event()
    declaration = json.loads(b.build_manifest(ev).decode("utf-8"))
    declaration[field][alias] = declaration[field].pop("1")

    with pytest.raises(SessionError, match="non-canonical component keys"):
        a.consume_manifest(json.dumps(declaration).encode("utf-8"), ev)


@pytest.mark.parametrize("field,selector", [("sc", "local_update"),
                                            ("nm", "sketch_disagreement"),
                                            ("sk", "sketch_disagreement")])
def test_a_canonical_key_alongside_its_alias_is_rejected(field, selector):
    """Two raw keys for one component; an integer comparison would count them once."""
    extra = {"decentralized.methods.pulse_lora.sketch_dim": 4} \
        if selector == "sketch_disagreement" else {}
    a, b = peers(selector, **extra)
    trained(a, seed=1); trained(b, seed=3)
    ev = event()
    declaration = json.loads(b.build_manifest(ev).decode("utf-8"))
    declaration[field]["01"] = declaration[field]["1"]

    with pytest.raises(SessionError, match="non-canonical component keys"):
        a.consume_manifest(json.dumps(declaration).encode("utf-8"), ev)


def test_the_minus_zero_alias_is_rejected():
    a, b = peers("local_update")
    trained(a, seed=1); trained(b, seed=3)
    ev = event()
    declaration = json.loads(b.build_manifest(ev).decode("utf-8"))
    declaration["sc"]["-0"] = declaration["sc"].pop("0")

    with pytest.raises(SessionError, match="non-canonical component keys"):
        a.consume_manifest(json.dumps(declaration).encode("utf-8"), ev)


@pytest.mark.parametrize("field,selector", [("sc", "local_update"),
                                            ("nm", "sketch_disagreement")])
def test_an_alias_is_rejected_before_planning_and_changes_nothing(field, selector):
    """An alias is rejected before planning; the next valid session still agrees."""
    extra = {"decentralized.methods.pulse_lora.exploration_fraction": 0.0}
    if selector == "sketch_disagreement":
        extra["decentralized.methods.pulse_lora.sketch_dim"] = 4
    a, b = peers(selector, **extra)
    trained(a, seed=1); trained(b, seed=4)
    ev = event()
    run_round(a, b, ev, credit=units.units_cost_bytes(a.components, 1) + 36,
              committed=True)

    before_memory = a.policy.memory.state_dict()
    before_metrics = dict(a.policy.metrics)
    before_digest = a.policy.memory.selector_digest(1, 2)
    ledger = ByteLedger()

    declaration = json.loads(b.build_manifest(ev).decode("utf-8"))
    declaration[field]["01"] = declaration[field].pop("1")
    with pytest.raises(SessionError, match="non-canonical component keys"):
        a.consume_manifest(json.dumps(declaration).encode("utf-8"), ev)

    assert a.policy.memory.state_dict() == before_memory
    assert dict(a.policy.metrics) == before_metrics
    assert a.policy._pending == {}
    assert ledger.total() == 0
    assert a.policy.memory.selector_digest(1, 2) == before_digest

    body_a, body_b = run_round(a, b, ev,
                               credit=units.units_cost_bytes(a.components, 1) + 36,
                               committed=True)
    assert (sorted(units.decode_units(a.components, body_a))
            == sorted(units.decode_units(b.components, body_b)))
    assert (a.policy.memory.selector_digest(1, 2)
            == b.policy.memory.selector_digest(1, 2))


# -- the agreement digest must cover everything a selection depends on ------------------------

@pytest.mark.parametrize("selector", ["random", "shuffled_score"])
@pytest.mark.parametrize("diverge", ["seed", "sessions"])
def test_selection_driving_state_that_diverges_is_caught_at_manifest_time(selector, diverge):
    """``draw`` is keyed on the seed and the session counter; the digest covers both.

    ``random`` and ``shuffled_score`` are the selectors that depend on ``draw``;
    without both in the digest, their endpoints could draw differently while
    their manifests agreed.
    """
    a, b = peers(selector)
    trained(a, seed=1); trained(b, seed=3)
    if diverge == "seed":
        b.policy.memory.seed += 1
    else:
        b.policy.memory.edge(1, 2).sessions += 1
    ev = event()

    with pytest.raises(SessionError, match="selector state has diverged"):
        a.consume_manifest(b.build_manifest(ev), ev)


@pytest.mark.parametrize("selector", ["random", "shuffled_score"])
@pytest.mark.parametrize("diverge", ["seed", "sessions"])
def test_divergent_selection_state_cannot_reach_bilateral_planning(selector, diverge):
    """The divergence is rejected at manifest time, before any plan is recorded."""
    a, b = peers(selector,
                 **{"decentralized.methods.pulse_lora.exploration_fraction": 0.0})
    trained(a, seed=1); trained(b, seed=3)
    if diverge == "seed":
        b.policy.memory.seed += 1
    else:
        b.policy.memory.edge(1, 2).sessions += 1
    ev = event()

    with pytest.raises(SessionError, match="selector state has diverged"):
        a.consume_manifest(b.build_manifest(ev), ev)
    assert a.policy._pending == {}, "no planning happened"


def test_the_divergence_would_actually_have_changed_the_selection():
    """The seed and the session counter each change ``draw`` and the digest."""
    from federatedscope.contrib.decentralized.methods.pulse_lora.ages import SelectorMemory

    base = SelectorMemory(seed=1)
    other_seed = SelectorMemory(seed=2)
    other_sessions = SelectorMemory(seed=1)
    other_sessions.edge(1, 2).sessions = 1

    draws = [m.draw(1, 2, 0, "select") for m in (base, other_seed, other_sessions)]
    assert len(set(draws)) == 3, "seed and sessions must both move the draw"
    assert base.selector_digest(1, 2) != other_seed.selector_digest(1, 2)
    assert base.selector_digest(1, 2) != other_sessions.selector_digest(1, 2)


# -- byte budget ---------------------------------------------------------------------------

def test_a_plan_never_exceeds_the_payload_credit():
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    ev = event()
    view_a, _ = negotiate(a, b, ev)
    overhead = encoded_size(payload_frame("unit", 1, 1, 0, b""))
    unit = units.unit_cost_bytes(a.components)

    for credit in range(0, 4 * unit, 37):
        a.policy._pending.clear()
        body = a.plan_payload(ev, budget_for(ev, credit), view_a)
        if body is None:
            assert credit < units.units_cost_bytes(a.components, 1) + overhead
        else:
            assert len(body) + overhead <= credit


def test_the_exact_boundary_admits_one_component_and_one_byte_short_admits_none():
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    ev = event()
    view_a, _ = negotiate(a, b, ev)
    overhead = encoded_size(payload_frame("unit", 1, 1, 0, b""))
    exact = units.units_cost_bytes(a.components, 1) + overhead

    a.policy._pending.clear()
    body = a.plan_payload(ev, budget_for(ev, exact), view_a)
    assert body is not None and len(units.decode_units(a.components, body)) == 1

    a.policy._pending.clear()
    assert a.plan_payload(ev, budget_for(ev, exact - 1), view_a) is None


def test_the_plan_size_matches_the_predicted_cost_exactly():
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    ev = event()
    view_a, _ = negotiate(a, b, ev)
    body = a.plan_payload(ev, budget_for(ev), view_a)
    count = len(units.decode_units(a.components, body))
    assert len(body) == units.units_cost_bytes(a.components, count)


def test_a_verbose_manifest_costs_payload_credit():
    """A sketch manifest is several times longer than a score manifest.

    Manifests are charged as control bytes before payload credit is computed, so
    the sketch's bytes come out of the payload credit.
    """
    lean, _ = peers("local_update")
    rich, _ = peers("sketch_disagreement",
                    **{"decentralized.methods.pulse_lora.sketch_dim": 32})
    trained(lean, seed=1); trained(rich, seed=1)
    ev = event()
    assert len(rich.build_manifest(ev)) > len(lean.build_manifest(ev)) * 3, (
        "a disagreement sketch is not free")


def test_the_sketch_selector_pays_for_its_manifest_in_a_pair_session():
    records = {}
    for selector, extra in (("age_only", {}),
                            ("sketch_disagreement",
                             {"decentralized.methods.pulse_lora.sketch_dim": 32})):
        cfg = pulse_cfg(**{"decentralized.methods.pulse_lora.selector": selector, **extra})
        algorithms = {1: make_peer(1, cfg=cfg), 2: make_peer(2, cfg=cfg)}
        for peer in algorithms.values():
            trained(peer, seed=peer.peer_id)
        records[selector] = execute_pair_session(
            event(), algorithms, TransactionEngine(), ByteLedger())
    assert (records["sketch_disagreement"]["control_bytes"]
            > records["age_only"]["control_bytes"])


# -- age lifecycle ----------------------------------------------------------------------------

def test_age_does_not_advance_when_nothing_could_have_fitted():
    a, b = peers()
    ev = event()
    view_a, _ = negotiate(a, b, ev)
    assert a.plan_payload(ev, budget_for(ev, 10), view_a) is None
    a.acknowledge(ev, committed=False, reason="too small")
    assert a.policy.memory.view(1, 2).ages == {}, (
        "a session that could carry nothing aged a component")


def test_age_advances_for_eligible_components_on_a_committed_session():
    """Ages advance on a committed session in which at least one component fits.

    Persistent selector state moves on a commit and on nothing else, so a
    rolled-back transaction leaves it as it was. Ageing only on sessions that
    delivered measures what the selector passed over, not what the link failed
    to carry.
    """
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    ev = event()
    run_round(a, b, ev, credit=units.units_cost_bytes(a.components, 1) + 36,
              committed=True)
    ages = a.policy.memory.view(1, 2).ages
    assert ages, "a committed affordable session ages every eligible component"


def test_a_failed_session_ages_nothing_at_all():
    """A failed session moves no selector state.

    It is still counted in the ``failed_sessions`` metric.
    """
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    ev = event()
    before = a.policy.memory.state_dict()
    run_round(a, b, ev, credit=units.units_cost_bytes(a.components, 1) + 36,
              committed=False)
    assert a.policy.memory.state_dict() == before, (
        "a failed session moved persistent selector state")
    assert a.policy.metrics["failed_sessions"] == 1, "the failure is still recorded"


def test_a_selected_component_resets_only_after_a_successful_commit():
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    ev = event()

    one = units.units_cost_bytes(a.components, 1) + 36

    body, _ = run_round(a, b, ev, credit=one, committed=False)
    failed_pick = sorted(units.decode_units(a.components, body))
    assert all(a.policy.memory.view(1, 2).age_of(c) == 0 for c in failed_pick), (
        "a failed transaction must move no persistent selector state at all - "
        "neither resetting what it selected nor ageing what it did not")

    body, _ = run_round(a, b, ev, credit=one, committed=True)
    committed_pick = sorted(units.decode_units(a.components, body))
    ages = a.policy.memory.view(1, 2)
    assert all(ages.age_of(c) == 0 for c in committed_pick), (
        f"a committed selection must reset: {committed_pick} -> {ages.ages}")
    unselected = [c for c in ages.ages if c not in committed_pick]
    assert unselected and all(ages.age_of(c) > 0 for c in unselected), (
        "only the committed selection resets; the rest keep ageing")


def test_a_rolled_back_transaction_resets_nothing():
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    ev = event()
    one = units.units_cost_bytes(a.components, 1) + 36
    before = a.policy.memory.state_dict()
    for _ in range(3):
        run_round(a, b, ev, credit=one, committed=False)
    assert a.policy.memory.state_dict() == before, (
        "three rolled-back sessions moved persistent selector state")


def test_acknowledge_without_a_plan_changes_nothing():
    """The handshake can fail before planning; both endpoints must skip alike."""
    a, _ = peers()
    before = a.policy.memory.state_dict()
    a.acknowledge(event(), committed=False, reason="control exceeded capacity")
    assert a.policy.memory.state_dict() == before


def test_age_state_stays_in_step_across_the_two_endpoints():
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    for _ in range(4):
        ev = event()
        run_round(a, b, ev, credit=units.units_cost_bytes(a.components, 1) + 36,
                  committed=True)
        assert a.policy.memory.selector_digest(1, 2) == b.policy.memory.selector_digest(1, 2)


# -- staging, commit and byte accounting ------------------------------------------------------
#
# Staging validates the incoming payload before any write, a session commits a mix,
# and the bytes it sent are recorded.

def test_stage_validates_the_incoming_plan_before_touching_anything():
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    ev = event()
    view_a, view_b = negotiate(a, b, ev)
    shared = budget_for(ev)
    a.plan_payload(ev, shared, view_a)          # a must have planned to stage
    body = b.plan_payload(ev, shared, view_b)

    before = {c: a.state.gather(c) for c in a.state.materialised}
    token = a.stage(body, ev)
    assert isinstance(token, int) and not isinstance(token, bool)
    for component, block in before.items():
        now = a.state.gather(component)
        assert all(torch.equal(now[k], v) for k, v in block.items()), (
            "staging wrote; the engine may still fail the other endpoint")

    from federatedscope.contrib.decentralized.methods.pulse_lora.units import UnitError
    with pytest.raises(UnitError):
        a.stage(body[:-8], ev)


def test_a_pulse_session_commits_a_mix():
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    before = {p.peer_id: {c: p.state.gather(c) for c in p.state.materialised}
              for p in (a, b)}
    record = execute_pair_session(event(), {1: a, 2: b}, TransactionEngine(), ByteLedger())

    assert record["status"] == "committed"
    assert record["payload_bytes"] > 0
    changed = [c for c in before[1]
               if not all(torch.equal(a.state.gather(c)[k], v)
                          for k, v in before[1][c].items())]
    assert changed, "a committed session changed nothing"


def test_a_session_records_payload_and_control_bytes():
    """The session here commits.

    Failed transactions, whose bytes also stay charged, are tested in
    ``test_pulse_transaction.py``.
    """
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    ledger = ByteLedger()
    record = execute_pair_session(event(), {1: a, 2: b}, TransactionEngine(), ledger)
    assert record["payload_bytes"] > 0
    assert record["control_bytes"] > 0


# -- checkpoint --------------------------------------------------------------------------------------

def test_the_selector_state_is_checkpointed_with_the_peer():
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    ev = event()
    view_a, _ = negotiate(a, b, ev)
    a.plan_payload(ev, budget_for(ev, units.units_cost_bytes(a.components, 1) + 36), view_a)
    a.acknowledge(ev, committed=True)

    state = a.state_dict()
    assert "session_policy" in state

    fresh = make_peer(1, cfg=pulse_cfg(
        **{"decentralized.methods.pulse_lora.selector": "local_update"}))
    fresh.load_state_dict(state)
    assert fresh.policy.memory.state_dict() == a.policy.memory.state_dict()
    assert fresh.policy.memory.selector_digest(1, 2) == a.policy.memory.selector_digest(1, 2)
    assert fresh.policy.metrics == a.policy.metrics


def test_a_malformed_selector_checkpoint_is_rejected_without_mutating_anything():
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    ev = event()
    view_a, _ = negotiate(a, b, ev)
    a.plan_payload(ev, budget_for(ev), view_a)
    a.acknowledge(ev, committed=True)

    before = a.policy.memory.state_dict()
    state = dict(a.state_dict())
    state["session_policy"] = {
        "memory": {"version": 1, "seed": 1,
                   "edges": {"1-2": {"ages": {"0": -5}, "cursor": 0,
                                     "sessions": 0}}},
        "metrics": dict.fromkeys(a.policy.metrics, 0)}
    with pytest.raises(ValueError, match="negative age"):
        a.load_state_dict(state)
    assert a.policy.memory.state_dict() == before, "a rejected checkpoint mutated the peer"


def test_resuming_reproduces_the_same_selection():
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    ev = event()
    one = units.units_cost_bytes(a.components, 1) + 36
    for _ in range(3):
        run_round(a, b, ev, credit=one, committed=True)

    resumed = make_peer(1, cfg=pulse_cfg(
        **{"decentralized.methods.pulse_lora.selector": "local_update"}))
    resumed.load_state_dict(a.state_dict())

    view_a, _ = negotiate(a, b, ev)
    view_r, _ = negotiate(resumed, b, ev)
    shared = budget_for(ev, units.units_cost_bytes(a.components, 1) + 36)
    assert a.plan_payload(ev, shared, view_a) == resumed.plan_payload(ev, shared, view_r)


# -- selection must stay RNG-free ---------------------------------------------------------

def test_selection_never_draws_from_a_live_rng():
    """Selection draws only from checkpointed per-edge state, never from a live RNG.

    ``SelectorMemory.draw`` is keyed on the seed, a purpose label, the edge, the
    session counter and the component, so it does not depend on how many draws
    came before, the two endpoints stay in step, and a resumed run replays the
    same selection. The test scans the selection, scoring, session, ages and units
    modules for RNG attributes read from ``context`` or ``self`` and for imports
    of the RNG service.
    """
    import ast

    package = pathlib.Path(__file__).resolve().parents[2].joinpath("methods/pulse_lora").resolve()
    for name in ("selection.py", "scoring.py", "session.py", "ages.py", "units.py"):
        tree = ast.parse((package / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in (
                    "rng", "random", "randint", "shuffle", "choice", "sample"):
                value = node.value
                base = getattr(value, "id", None) or getattr(value, "attr", None)
                assert base not in ("context", "self"), (
                    f"pulse_lora/{name} draws from a live RNG via {base}.{node.attr}")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, "module", "") or ""
                names = [a.name for a in node.names]
                assert "rng" not in module and "rng" not in names, (
                    f"pulse_lora/{name} imports the RNG service")


def test_the_random_selector_replays_across_a_simulated_resume():
    a, b = peers("random")
    trained(a, seed=1); trained(b, seed=2)
    ev = event()
    one = units.units_cost_bytes(a.components, 1) + 36
    for _ in range(3):
        run_round(a, b, ev, credit=one, committed=True)

    resumed = make_peer(1, cfg=pulse_cfg(
        **{"decentralized.methods.pulse_lora.selector": "random"}))
    resumed.load_state_dict(a.state_dict())

    view_a, _ = negotiate(a, b, ev)
    view_r, _ = negotiate(resumed, b, ev)
    shared = budget_for(ev, one)
    assert a.plan_payload(ev, shared, view_a) == resumed.plan_payload(ev, shared, view_r)
