"""Eligible candidates, selector scores, and the budgeted choice of components.

The candidates for a session between ``i`` and ``j`` are the eligible set
``Q_ij = M_i | M_j``, the components materialized at either endpoint. A
component neither endpoint has materialized is in neither set: both peers
derive the same deterministic initial value for it, so exchanging it would
change nothing.

The two endpoints select independently, with no negotiation between planning
and sending (``core/pair_session.py`` calls ``plan_payload`` on each peer in
turn), so selection is a deterministic function of the two manifests, the
shared per-edge state and the budget. If the endpoints chose different sets,
staging would reject the partner's payload and the session would fail. The
scores are therefore symmetric in the two endpoints (the exact oracle up to
floating-point rounding, see ``scoring.exact_disagreement``), and ties are
broken by component id order (rotated by the per-edge round-robin cursor in
the exploration step).
"""
import math
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

from federatedscope.contrib.decentralized.methods.pulse_lora.ages import SelectorMemory

#: Selector names: ``local_update`` (the default) and ``sketch_disagreement``,
#: the ``exact_disagreement`` oracle, and the ablations ``age_only``,
#: ``random``, ``shuffled_score``, ``full_exchange`` and ``survivor_only``.
SELECTOR_EXACT = "exact_disagreement"
SELECTOR_SKETCH = "sketch_disagreement"
SELECTOR_LOCAL = "local_update"
SELECTOR_AGE = "age_only"
SELECTOR_RANDOM = "random"
SELECTOR_SHUFFLED = "shuffled_score"
SELECTOR_FULL = "full_exchange"
SELECTOR_SURVIVOR = "survivor_only"

SELECTORS = (
    SELECTOR_EXACT, SELECTOR_SKETCH, SELECTOR_LOCAL, SELECTOR_AGE,
    SELECTOR_RANDOM, SELECTOR_SHUFFLED, SELECTOR_FULL, SELECTOR_SURVIVOR,
)

#: Selectors that read the partner's tensors directly. Only the single-process
#: simulation can do that, so they are oracles, not deployable protocols.
ORACLE_SELECTORS = frozenset({SELECTOR_EXACT})

#: Selectors whose utility folds a per-component score that both endpoints
#: publish in the manifest (``sc``: the local-update magnitude of every
#: materialized component). ``survivor_only`` is one of them: it restricts the
#: candidates to the shared trainable support and ranks them by the same
#: utility as ``local_update``.
SCORE_PUBLISHING_SELECTORS = frozenset(
    {SELECTOR_LOCAL, SELECTOR_SHUFFLED, SELECTOR_SURVIVOR})

#: Selectors that need no cross-peer utility at all.
SCORE_FREE_SELECTORS = frozenset({SELECTOR_AGE, SELECTOR_RANDOM, SELECTOR_FULL})


class SelectionError(ValueError):
    pass


@dataclass(frozen=True)
class Candidate:
    component: int
    score: float
    age: int


def candidate_components(mode: str,
                         local_materialised: Sequence[int],
                         remote_materialised: Sequence[int],
                         local_support: Sequence[int],
                         remote_support: Sequence[int]) -> Tuple[int, ...]:
    """The eligible set ``Q_ij = M_i | M_j``, or ``S_i & S_j`` under ``survivor_only``.

    ``survivor_only`` is the ablation without relay eligibility: a peer may only
    send components in its own trainable support, so a bilateral exchange needs
    the component active at both ends. Sorted by component id.
    """
    if mode == SELECTOR_SURVIVOR:
        return tuple(sorted(set(map(int, local_support)) & set(map(int, remote_support))))
    return tuple(sorted(set(map(int, local_materialised)) | set(map(int, remote_materialised))))


def combine_scores(mode: str, components: Sequence[int],
                   local: Mapping[int, float], remote: Mapping[int, float]) -> Dict[int, float]:
    """Fold the two endpoints' scores into one symmetric score: their ``max``.

    A component either peer has moved far from its initial value ranks high
    even if the other peer has not touched it. A component missing from a map
    contributes 0.
    """
    return {int(c): max(float(local.get(int(c), 0.0)), float(remote.get(int(c), 0.0)))
            for c in components}


def shuffled(memory: SelectorMemory, edge: Tuple[int, int],
             scores: Mapping[int, float]) -> Dict[int, float]:
    """The ``shuffled_score`` ablation: the same scores, permuted across components.

    The permutation comes from a per-session draw, so the distribution of scores
    is kept while the link between each component and its score is broken.
    """
    components = sorted(scores)
    values = sorted(scores.values())
    order = sorted(components,
                   key=lambda c: (memory.draw(edge[0], edge[1], c, "shuffle"), c))
    return {component: values[index] for index, component in enumerate(order)}


def score_candidates(mode: str, memory: SelectorMemory, edge: Tuple[int, int],
                     components: Sequence[int],
                     utility: Mapping[int, float]) -> Dict[int, float]:
    """The score each candidate is ranked by under ``mode``.

    ``age_only``: its age on this edge; ``random``: a per-session draw;
    ``full_exchange``: 0 for every candidate; ``shuffled_score``: the utility,
    permuted; every other selector: the utility itself.
    """
    if mode not in SELECTORS:
        raise SelectionError(f"unknown selector {mode!r}; expected one of {list(SELECTORS)}")

    if mode == SELECTOR_AGE:
        return {int(c): float(memory.view(*edge).age_of(c)) for c in components}
    if mode == SELECTOR_RANDOM:
        return {int(c): memory.draw(edge[0], edge[1], c, "select") for c in components}
    if mode == SELECTOR_FULL:
        # No preference at all: every candidate is equally worth sending, so the
        # ordering is purely the id tie-break and capacity decides the rest.
        return {int(c): 0.0 for c in components}
    if mode == SELECTOR_SHUFFLED:
        return shuffled(memory, edge, {int(c): float(utility.get(int(c), 0.0))
                                       for c in components})
    return {int(c): float(utility.get(int(c), 0.0)) for c in components}


def exploration_slots(fraction: float, slots: int) -> int:
    """The exploration quota ``min(slots, max(1, ceil(fraction * slots)))``.

    These slots go to the oldest eligible components. A fraction of 0 (or
    below) reserves nothing, which is the setting without exploration; a
    fraction above 1 raises. No slots means no quota.
    """
    if slots <= 0:
        return 0
    if fraction <= 0.0:
        return 0
    if not 0.0 < float(fraction) <= 1.0:
        raise SelectionError(
            f"exploration fraction must satisfy 0 <= f <= 1, got {fraction}")
    return min(slots, max(1, math.ceil(float(fraction) * slots)))


def select_with_provenance(mode: str, memory: SelectorMemory, edge: Tuple[int, int],
                           components: Sequence[int], scores: Mapping[int, float],
                           slots: int, exploration_fraction: float):
    """Like :func:`select`, but returns ``(chosen, explored)``.

    ``explored`` holds the components the exploration quota picked; both are
    sorted by component id. The planning log records it, because the chosen set
    alone cannot tell an exploration pick from a utility pick of the same
    component.
    """
    chosen, explored = _select(mode, memory, edge, components, scores, slots,
                               exploration_fraction)
    return chosen, explored


def select(mode: str, memory: SelectorMemory, edge: Tuple[int, int],
           components: Sequence[int], scores: Mapping[int, float],
           slots: int, exploration_fraction: float) -> Tuple[int, ...]:
    """Choose ``min(slots, len(components))`` components.

    The exploration quota (:func:`exploration_slots`) goes to the oldest
    eligible components; the remaining slots go to the highest scores among the
    rest, ties broken by component id. Returned sorted by component id, so the
    payload order does not depend on why each component was chosen.
    """
    return _select(mode, memory, edge, components, scores, slots,
                   exploration_fraction)[0]


def _select(mode, memory, edge, components, scores, slots, exploration_fraction):
    components = tuple(sorted(int(c) for c in components))
    if slots <= 0 or not components:
        return (), ()
    slots = min(int(slots), len(components))

    reserved = exploration_slots(exploration_fraction, slots)
    state = memory.view(*edge)

    chosen, explored = [], []
    if reserved:
        # Oldest first. Among equal ages the round-robin cursor rotates the
        # starting point, so a persistently tied set does not always hand the
        # slot to the same lowest id.
        rotated = _rotate(components, state.cursor)
        by_age = sorted(rotated, key=lambda c: (-state.age_of(c), rotated.index(c)))
        chosen = by_age[:reserved]
        explored = list(chosen)

    remaining = slots - len(chosen)
    if remaining > 0:
        rest = [c for c in components if c not in set(chosen)]
        by_utility = sorted(rest, key=lambda c: (-float(scores.get(c, 0.0)), c))
        chosen.extend(by_utility[:remaining])

    return tuple(sorted(chosen)), tuple(sorted(explored))


def _rotate(components: Sequence[int], cursor: int) -> Tuple[int, ...]:
    if not components:
        return ()
    offset = int(cursor) % len(components)
    return tuple(components[offset:]) + tuple(components[:offset])
