"""Candidate construction, scoring modes, exploration and per-edge selector state.

Run: python -m pytest federatedscope/contrib/decentralized/tests/methods/test_pulse_selection.py -q -p no:cacheprovider

Covers the candidate set ``Q_ij = M_i | M_j`` and its ``survivor_only``
restriction, the symmetric score fold, bounded exploration, and the ages, cursor,
deterministic draws and checkpoint of ``SelectorMemory``. How ages move across
sessions is tested in ``test_pulse_session.py``.
"""
import pytest
import torch

from federatedscope.contrib.decentralized.methods.pulse_lora import selection
from federatedscope.contrib.decentralized.methods.pulse_lora.ages import (
    SELECTOR_CHECKPOINT_VERSION, SelectorMemory, edge_key,
)
from federatedscope.contrib.decentralized.methods.pulse_lora.session import METRIC_NAMES
from federatedscope.contrib.decentralized.methods.pulse_lora.selection import (
    SELECTORS, SelectionError,
)
from federatedscope.contrib.decentralized.tests.methods.test_pulse_peer import make_peer


def complete_checkpoint(**edge_overrides):
    """A checkpoint with every required field, so a test can corrupt exactly one.

    ``load_state_dict`` rejects a checkpoint with a missing field, which would
    otherwise mask the fault under test.
    """
    edge = {"ages": {"0": 3}, "cursor": 1, "sessions": 2}
    second = {"ages": {}, "cursor": 0, "sessions": 0}
    second.update(edge_overrides)
    return {"version": SELECTOR_CHECKPOINT_VERSION, "seed": 999,
            "edges": {"1-2": edge, "2-3": second}}


# -- candidates -------------------------------------------------------------------

def test_candidates_are_the_union_of_the_two_materialisations():
    got = selection.candidate_components("local_update", [0, 1, 3], [1, 4], [0], [1])
    assert got == (0, 1, 3, 4)


def test_a_component_lazy_at_both_endpoints_is_excluded():
    """Not by a special case - it is in neither M, so the union never contains it."""
    got = selection.candidate_components("local_update", [0, 1], [1, 2], [0], [1])
    assert 7 not in got and got == (0, 1, 2)


def test_a_component_untouched_at_both_endpoints_would_be_a_provable_no_op():
    """Both peers derive the same value for a component neither holds."""
    a, b = make_peer(1), make_peer(2)
    lazy = 7
    assert lazy not in a.state.materialised and lazy not in b.state.materialised
    left, right = a.state.gather(lazy), b.state.gather(lazy)
    assert all(torch.equal(left[k], right[k]) for k in left), (
        "two peers derive the identical deterministic value for an untouched component")


def test_one_sided_materialisation_still_yields_a_usable_value():
    """A component held by one endpoint is a candidate.

    The endpoint that lacks it uses the deterministic initial value.
    """
    a, b = make_peer(1), make_peer(2)
    component = 7
    a.state.materialise(component, a.components.initial_component(component))
    assert component in a.state.materialised
    assert component not in b.state.materialised
    assert component in selection.candidate_components(
        "local_update", a.state.materialised, b.state.materialised,
        a.state.support, b.state.support)
    # The peer that lacks it can still price and score it.
    assert set(b.state.gather(component)) == set(a.state.gather(component))


def test_survivor_only_restricts_to_components_active_at_both_ends():
    got = selection.candidate_components("survivor_only", [0, 1, 3], [1, 4], [0, 1], [1, 3])
    assert got == (1,), "relay eligibility is switched off in this ablation"


def test_survivor_only_and_the_union_differ_whenever_relay_state_exists():
    union = selection.candidate_components("local_update", [0, 1, 3], [1, 4], [0, 1], [1])
    survivor = selection.candidate_components("survivor_only", [0, 1, 3], [1, 4], [0, 1], [1])
    assert set(survivor) < set(union)


# -- scores are symmetric ------------------------------------------------------------

def test_the_combined_score_is_symmetric_in_the_two_endpoints():
    left = {0: 1.0, 1: 5.0}
    right = {0: 4.0, 1: 2.0}
    forward = selection.combine_scores("local_update", [0, 1], left, right)
    backward = selection.combine_scores("local_update", [0, 1], right, left)
    assert forward == backward == {0: 4.0, 1: 5.0}


def test_a_missing_contribution_counts_as_zero_not_as_an_error():
    assert selection.combine_scores("local_update", [0, 9], {0: 1.0}, {}) == {0: 1.0, 9: 0.0}


@pytest.mark.parametrize("mode", SELECTORS)
def test_every_selector_is_deterministic(mode):
    memory = SelectorMemory(seed=5)
    edge, components = (1, 2), (0, 1, 2, 3)
    utility = {0: 0.3, 1: 0.9, 2: 0.1, 3: 0.5}
    first = selection.score_candidates(mode, memory, edge, components, utility)
    second = selection.score_candidates(mode, memory, edge, components, utility)
    assert first == second


def test_an_unknown_selector_is_rejected():
    with pytest.raises(SelectionError, match="unknown selector"):
        selection.score_candidates("vibes", SelectorMemory(), (1, 2), (0,), {})


def test_shuffled_score_preserves_the_score_multiset_but_breaks_the_pairing():
    memory = SelectorMemory(seed=3)
    scores = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
    shuffled = selection.shuffled(memory, (1, 2), scores)
    assert sorted(shuffled.values()) == sorted(scores.values())
    assert shuffled != scores, "the shuffle left every score on its own component"


def test_full_exchange_expresses_no_preference():
    scores = selection.score_candidates("full_exchange", SelectorMemory(), (1, 2),
                                        (0, 1, 2), {0: 9.0, 1: 1.0})
    assert set(scores.values()) == {0.0}


# -- exploration ---------------------------------------------------------------------

@pytest.mark.parametrize("fraction,slots,expected", [
    (0.0, 4, 0), (0.0, 1, 0),            # the no-exploration ablation reserves nothing
    (0.25, 4, 1), (0.5, 4, 2), (1.0, 4, 4),
    (0.01, 4, 1),                        # any positive fraction reserves at least one
    (0.5, 1, 1), (0.5, 0, 0),
])
def test_exploration_reservation(fraction, slots, expected):
    assert selection.exploration_slots(fraction, slots) == expected


@pytest.mark.parametrize("slots", [2, 3, 4, 6])
def test_bounded_exploration_never_starves_utility(slots):
    """The reservation is a floor for age and a ceiling on itself.

    The top ``slots - reserved`` candidates by utility are always selected,
    however old the low-utility ones get. The candidate set is larger than the
    slot count; with fewer candidates than slots every candidate is sent.
    """
    memory = SelectorMemory(seed=1)
    edge, components = (1, 2), tuple(range(8))
    assert slots < len(components), "otherwise every candidate is selected anyway"

    # The oldest components are exactly the least useful ones, so a selector that
    # let exploration run unbounded would pick nothing worth sending.
    for _ in range(5):
        memory.edge(*edge).advance([6, 7])
    scores = {c: float(8 - c) for c in components}

    for fraction in (0.1, 0.25, 0.5):
        chosen = selection.select("local_update", memory, edge, components, scores,
                                  slots, fraction)
        reserved = selection.exploration_slots(fraction, slots)
        assert len(chosen) == slots
        guaranteed = sorted(components, key=lambda c: (-scores[c], c))[:slots - reserved]
        assert set(guaranteed) <= set(chosen), (
            f"utility was starved at {fraction=}: wanted {guaranteed}, got {chosen}")
        assert slots - reserved > 0, "some slot must always remain for utility"


def test_with_no_exploration_every_slot_goes_to_utility():
    memory = SelectorMemory(seed=1)
    edge, components = (1, 2), tuple(range(8))
    for _ in range(9):
        memory.edge(*edge).advance([6, 7])
    scores = {c: float(8 - c) for c in components}
    assert selection.select("local_update", memory, edge, components, scores,
                            3, 0.0) == (0, 1, 2)


def test_exploration_eventually_serves_the_oldest_component():
    """Without exploration a low-utility component is never selected.

    With exploration, the oldest component is served.
    """
    memory = SelectorMemory(seed=1)
    edge, components = (1, 2), (0, 1, 2)
    scores = {0: 9.0, 1: 8.0, 2: 0.0}

    starved = SelectorMemory(seed=1)
    for _ in range(10):
        starved.edge(*edge).advance([2])
        assert 2 not in selection.select("local_update", starved, edge, components,
                                         scores, 2, 0.0)

    memory.edge(*edge).advance([2])
    assert 2 in selection.select("local_update", memory, edge, components, scores, 2, 0.5)


def test_selection_never_exceeds_the_slots_or_the_candidates():
    memory = SelectorMemory(seed=1)
    for slots in range(0, 6):
        chosen = selection.select("local_update", memory, (1, 2), (0, 1, 2),
                                  {0: 1.0, 1: 2.0, 2: 3.0}, slots, 0.5)
        assert len(chosen) == min(slots, 3)
        assert len(set(chosen)) == len(chosen), "no component is selected twice"


def test_selection_is_returned_in_canonical_component_order():
    memory = SelectorMemory(seed=1)
    memory.edge(1, 2).advance([3])
    chosen = selection.select("local_update", memory, (1, 2), (0, 1, 2, 3),
                              {0: 9.0, 1: 8.0, 2: 7.0, 3: 0.0}, 3, 0.5)
    assert list(chosen) == sorted(chosen), "wire order must not reveal why one was picked"


def test_ties_break_on_component_id():
    memory = SelectorMemory(seed=1)
    flat = {c: 1.0 for c in range(6)}
    assert selection.select("local_update", memory, (1, 2), tuple(range(6)), flat,
                            3, 0.0) == (0, 1, 2)


# -- per-edge state -------------------------------------------------------------------

def test_an_edge_is_unordered():
    memory = SelectorMemory()
    memory.edge(5, 2).advance([1])
    assert memory.edge(2, 5).age_of(1) == 1
    assert edge_key(5, 2) == edge_key(2, 5) == (2, 5)


def test_edges_do_not_share_age_state():
    memory = SelectorMemory()
    memory.edge(1, 2).advance([0])
    assert memory.edge(1, 3).age_of(0) == 0


def test_the_digest_changes_with_the_ages_and_agrees_across_endpoints():
    a, b = SelectorMemory(seed=1), SelectorMemory(seed=1)
    assert a.selector_digest(1, 2) == b.selector_digest(2, 1)
    a.edge(1, 2).advance([0])
    assert a.selector_digest(1, 2) != b.selector_digest(2, 1)
    b.edge(2, 1).advance([0])
    assert a.selector_digest(1, 2) == b.selector_digest(2, 1)


def test_selector_memory_survives_a_checkpoint_round_trip_exactly():
    memory = SelectorMemory(seed=9)
    memory.edge(1, 2).advance([0, 3]); memory.edge(1, 2).reset([0])
    memory.edge(2, 3).advance([1]); memory.edge(2, 3).cursor = 2
    memory.edge(2, 3).sessions = 7

    restored = SelectorMemory()
    restored.load_state_dict(memory.state_dict())
    assert restored.state_dict() == memory.state_dict()
    assert restored.selector_digest(1, 2) == memory.selector_digest(1, 2)
    assert restored.draw(2, 3, 1, "select") == memory.draw(2, 3, 1, "select")


@pytest.mark.parametrize("corrupt,match", [
    (complete_checkpoint(ages={"0": -1}), "negative age"),
    (complete_checkpoint(cursor=-1), "negative cursor"),
    (complete_checkpoint(sessions=-4), "negative sessions"),
    (complete_checkpoint(ages={0: 1, "0": 2}), "colliding identifiers"),
    (dict(complete_checkpoint(), edges={"oops": {"ages": {}, "cursor": 0,
                                                 "sessions": 0}}),
     "unreadable edge key"),
])
def test_a_malformed_selector_checkpoint_is_rejected(corrupt, match):
    with pytest.raises(ValueError, match=match):
        SelectorMemory().load_state_dict(corrupt)


def test_a_checkpoint_missing_only_the_version_is_rejected():
    """A checkpoint without a version is refused, not read as the current version.

    Every other field is present, so the missing version is the only fault, and
    the refusal leaves the memory, its digest and its draws unchanged.
    """
    memory = populated_memory()
    before = memory.state_dict()
    before_digest = memory.selector_digest(1, 2)
    before_draws = [memory.draw(1, 2, c, "select") for c in range(4)]

    complete_but_unversioned = {
        "seed": 42,
        "edges": {"1-2": {"ages": {"0": 3}, "cursor": 1, "sessions": 2}},
    }
    assert "version" not in complete_but_unversioned

    with pytest.raises(ValueError, match="missing 'version'"):
        memory.load_state_dict(complete_but_unversioned)

    assert memory.state_dict() == before, "a refused checkpoint mutated the memory"
    assert memory.selector_digest(1, 2) == before_digest
    assert [memory.draw(1, 2, c, "select") for c in range(4)] == before_draws


def test_the_version_field_is_validated_like_every_other_schema_field():
    for value, match in ((None, "not an integer"), (True, "not a counter"),
                         (1.5, "not an exact integer"), ("1", "not an integer")):
        checkpoint = dict(complete_checkpoint(), version=value)
        with pytest.raises(ValueError, match=match):
            SelectorMemory().load_state_dict(checkpoint)


def test_an_integral_float_counter_is_accepted_but_a_fractional_one_is_not():
    """2.0 is read as the integer 2; 2.5 is refused rather than truncated."""
    memory = SelectorMemory()
    memory.load_state_dict({"version": SELECTOR_CHECKPOINT_VERSION, "seed": 1.0,
                            "edges": {"1-2": {"ages": {"0": 2.0}, "cursor": 0.0,
                                              "sessions": 0.0}}})
    assert memory.seed == 1 and memory.view(1, 2).age_of(0) == 2

    with pytest.raises(ValueError, match="not an exact integer"):
        SelectorMemory().load_state_dict(complete_checkpoint(ages={"0": 2.5}))


def test_our_own_state_dict_round_trips_under_the_strict_schema():
    """The writer must satisfy the reader, or resume is impossible."""
    memory = populated_memory()
    restored = SelectorMemory()
    restored.load_state_dict(memory.state_dict())
    assert restored.state_dict() == memory.state_dict()
    assert restored.selector_digest(1, 2) == memory.selector_digest(1, 2)


def test_the_deterministic_draw_does_not_depend_on_call_count():
    """Counter-based, so resuming mid-run cannot desynchronise it."""
    memory = SelectorMemory(seed=4)
    first = memory.draw(1, 2, 0, "select")
    for _ in range(50):
        memory.draw(1, 2, 3, "select")
    assert memory.draw(1, 2, 0, "select") == first


def test_the_draw_changes_as_sessions_advance():
    memory = SelectorMemory(seed=4)
    before = memory.draw(1, 2, 0, "select")
    memory.edge(1, 2).sessions += 1
    assert memory.draw(1, 2, 0, "select") != before


# -- a refused checkpoint changes nothing ------------------------------------------------
#
# SelectorMemory.load_state_dict and PulseSessionPolicy.load_state_dict validate the
# whole checkpoint before writing anything, whoever calls them (PulseLoRAPeer also
# validates through a throwaway policy first). Most faults below come after a valid
# seed and a valid first edge, so a partial apply would be visible.

def populated_memory():
    memory = SelectorMemory(seed=11)
    memory.edge(1, 2).advance([0, 1])
    memory.edge(1, 2).reset([0])
    memory.edge(2, 3).cursor = 4
    memory.edge(2, 3).sessions = 9
    return memory


@pytest.mark.parametrize("checkpoint,match", [
    # every one of these is complete apart from the single late fault under test
    (complete_checkpoint(ages={"1": -7}), "negative age"),
    (complete_checkpoint(cursor=-1), "negative cursor"),
    (complete_checkpoint(sessions=-1), "negative sessions"),
    (complete_checkpoint(ages={"x": 1}), "non-numeric component key"),
    (complete_checkpoint(ages={"01": 1}), "not the canonical form"),
    (complete_checkpoint(ages={"0": True}), "not a counter"),
    (complete_checkpoint(ages={"0": 1.9}), "not an exact integer"),
    (complete_checkpoint(cursor=2.8), "not an exact integer"),
    (complete_checkpoint(sessions=3.7), "not an exact integer"),
    # structural faults, still after a valid first edge
    (dict(complete_checkpoint(), edges={"1-2": {"ages": {"0": 3}, "cursor": 1,
                                                "sessions": 2},
                                        "oops": {"ages": {}, "cursor": 0,
                                                 "sessions": 0}}),
     "unreadable edge key"),
    (dict(complete_checkpoint(), edges={"1-2": {"ages": {"0": 3}, "cursor": 1,
                                                "sessions": 2},
                                        "2-3": "not a mapping"}), "not a mapping"),
    (dict(complete_checkpoint(), edges={"1-2": {"ages": {"0": 3}, "cursor": 1,
                                                "sessions": 2},
                                        "3-2": {"ages": {}, "cursor": 0,
                                                "sessions": 0}}),
     "not the canonical form"),
    # missing selection-driving fields are refused, never defaulted
    ({"version": SELECTOR_CHECKPOINT_VERSION, "edges": {}}, "missing 'seed'"),
    ({"version": SELECTOR_CHECKPOINT_VERSION, "seed": 1}, "missing 'edges'"),
    (dict(complete_checkpoint(), edges={"1-2": {"ages": {}, "sessions": 0}}),
     "missing 'cursor'"),
    (dict(complete_checkpoint(), edges={"1-2": {"ages": {}, "cursor": 0}}),
     "missing 'sessions'"),
    (dict(complete_checkpoint(), edges={"1-2": {"cursor": 0, "sessions": 0}}),
     "missing 'ages'"),
    ({"version": SELECTOR_CHECKPOINT_VERSION, "seed": True, "edges": {}},
     "not a counter"),
    # a checkpoint of another version is refused rather than reinterpreted
    ({"version": 99, "seed": 1, "edges": {}}, "explicit migration"),
])
def test_a_late_failure_leaves_the_selector_memory_byte_for_byte_unchanged(checkpoint, match):
    memory = populated_memory()
    before = memory.state_dict()
    before_digest = memory.selector_digest(1, 2)

    with pytest.raises(ValueError, match=match):
        memory.load_state_dict(checkpoint)

    assert memory.state_dict() == before, "a refused checkpoint mutated the memory"
    assert memory.seed == before["seed"], "the seed was applied before validation"
    assert memory.selector_digest(1, 2) == before_digest


def test_a_valid_checkpoint_still_applies_completely():
    memory = populated_memory()
    other = SelectorMemory(seed=42)
    other.edge(4, 5).advance([2, 3])
    memory.load_state_dict(other.state_dict())
    assert memory.state_dict() == other.state_dict()
    assert memory.seed == 42


def full_metrics(**overrides):
    """Every counter the policy declares, so only the override is under test.

    Read from METRIC_NAMES rather than from a constructed peer: these calls run
    inside @parametrize at collection time, before the autouse registry-isolation
    fixture, and building a peer there would leave the toy factory registered
    for every later test.
    """
    base = dict.fromkeys(METRIC_NAMES, 0)
    base.update(overrides)
    return base


@pytest.mark.parametrize("metrics,match", [
    (full_metrics(units_selected="not-an-int"), "not an integer"),
    (full_metrics(units_selected=-4), "negative"),
    (full_metrics(units_selected=True), "not a counter"),
    (full_metrics(units_selected=1.5), "not an exact integer"),
])
def test_a_late_metric_failure_leaves_the_policy_memory_unchanged(metrics, match):
    """A valid memory with invalid metrics.

    Applying the memory before checking the metrics would leave it changed here.
    """
    from federatedscope.contrib.decentralized.methods.pulse_lora.session import SessionError
    from federatedscope.contrib.decentralized.tests.methods.test_pulse_peer import make_peer

    peer = make_peer(1)
    policy = peer.policy
    policy.memory.edge(1, 2).advance([0, 3])
    policy.metrics["units_selected"] = 5
    before_memory = policy.memory.state_dict()
    before_metrics = dict(policy.metrics)

    with pytest.raises(SessionError, match=match):
        policy.load_state_dict({
            "memory": {"version": SELECTOR_CHECKPOINT_VERSION, "seed": 77,
                       "edges": {"4-5": {"ages": {"2": 9}, "cursor": 0,
                                         "sessions": 0}}},
            "metrics": metrics,
        })

    assert policy.memory.state_dict() == before_memory, "the memory was applied first"
    assert dict(policy.metrics) == before_metrics


def test_a_policy_checkpoint_with_a_bad_memory_leaves_the_metrics_unchanged():
    from federatedscope.contrib.decentralized.methods.pulse_lora.session import SessionError
    from federatedscope.contrib.decentralized.tests.methods.test_pulse_peer import make_peer

    peer = make_peer(1)
    policy = peer.policy
    policy.metrics["units_selected"] = 5
    before = (policy.memory.state_dict(), dict(policy.metrics))

    with pytest.raises(ValueError, match="negative age"):
        policy.load_state_dict({
            "memory": {"version": SELECTOR_CHECKPOINT_VERSION, "seed": 3, "edges": {
                "1-2": {"ages": {"0": 1}, "cursor": 0, "sessions": 0},
                "2-3": {"ages": {"1": -2}, "cursor": 0, "sessions": 0}}},
            "metrics": full_metrics(units_selected=9)})

    assert (policy.memory.state_dict(), dict(policy.metrics)) == before
