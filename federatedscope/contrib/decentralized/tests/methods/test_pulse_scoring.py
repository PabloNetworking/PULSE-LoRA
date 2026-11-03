"""Exact effective-update disagreement and its deployable sketch.

Run: python -m pytest federatedscope/contrib/decentralized/tests/methods/test_pulse_scoring.py -q -p no:cacheprovider

The exact score is computed from inner products without allocating the rank-one
outer products, so it is first checked against the allocated definition. The
remaining tests check the invariances of a distance between two peers' values of
one component, and how closely the sketch tracks the exact score.
"""
import math

import pytest
import torch

from federatedscope.contrib.decentralized.methods.pulse_lora import scoring
from federatedscope.contrib.decentralized.methods.pulse_lora.scoring import ScoringError
from federatedscope.contrib.decentralized.tests.methods.test_pulse_peer import make_peer


def randomise(block, seed):
    generator = torch.Generator().manual_seed(seed)
    return {k: torch.rand(v.shape, generator=generator, dtype=v.dtype) - 0.5
            for k, v in block.items()}


def brute_force(schema, local, remote):
    """The definition, with the outer products actually allocated."""
    total = 0.0
    for spec in schema.lora.modules:
        ai = local[f"{spec.module}|A"].double(); bi = local[f"{spec.module}|B"].double()
        aj = remote[f"{spec.module}|A"].double(); bj = remote[f"{spec.module}|B"].double()
        total += float(((torch.outer(bi, ai) - torch.outer(bj, aj)) ** 2).sum())
    return float(schema.scale) * math.sqrt(total)


@pytest.fixture
def pair():
    peer = make_peer(1)
    base = peer.state.gather(0)
    return peer.components, randomise(base, 11), randomise(base, 12)


# -- the closed form is the definition ------------------------------------------------

def test_the_closed_form_matches_allocated_outer_products(pair):
    schema, local, remote = pair
    exact = scoring.exact_disagreement(schema, local, remote)
    reference = brute_force(schema, local, remote)
    assert exact == pytest.approx(reference, rel=1e-12)
    assert exact > 0


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_the_closed_form_matches_over_many_draws(seed):
    peer = make_peer(1)
    base = peer.state.gather(0)
    local, remote = randomise(base, seed), randomise(base, seed + 100)
    assert scoring.exact_disagreement(peer.components, local, remote) == pytest.approx(
        brute_force(peer.components, local, remote), rel=1e-12)


# -- invariances ------------------------------------------------------------------------

def test_a_component_does_not_disagree_with_itself(pair):
    schema, local, _ = pair
    assert scoring.exact_disagreement(schema, local, local) == 0.0


def test_disagreement_is_symmetric(pair):
    schema, local, remote = pair
    assert (scoring.exact_disagreement(schema, local, remote)
            == pytest.approx(scoring.exact_disagreement(schema, remote, local), rel=1e-15))


def test_disagreement_is_non_negative(pair):
    schema, local, remote = pair
    assert scoring.exact_disagreement(schema, local, remote) >= 0.0


def test_flipping_the_sign_of_both_factors_changes_nothing(pair):
    """``b a^T`` is invariant under ``(a, b) -> (-a, -b)``, so the score must be too.

    This is the invariance that catches a score computed on the factors rather
    than on the effective update they produce.
    """
    schema, local, remote = pair
    flipped = {k: -v for k, v in local.items()}
    assert (scoring.exact_disagreement(schema, flipped, remote)
            == pytest.approx(scoring.exact_disagreement(schema, local, remote), rel=1e-12))


def test_two_untouched_components_never_disagree():
    """Both endpoints derive the same initial value, so exchanging it is a no-op."""
    peer = make_peer(1)
    for component in (0, 3, 7):
        initial = peer.components.initial_component(component)
        assert scoring.exact_disagreement(peer.components, initial, initial) == 0.0


def test_components_with_zero_b_do_not_disagree_however_different_their_a():
    """``B = 0`` contributes nothing to the forward pass, so ``A`` cannot matter."""
    peer = make_peer(1)
    base = peer.state.gather(0)
    left = randomise(base, 21)
    right = randomise(base, 22)
    for key in list(left):
        if key.endswith("|B"):
            left[key] = torch.zeros_like(left[key])
            right[key] = torch.zeros_like(right[key])
    assert scoring.exact_disagreement(peer.components, left, right) == pytest.approx(0.0)


def test_disagreement_scales_linearly_with_the_fixed_scale(pair):
    schema, local, remote = pair
    import dataclasses
    base = scoring.exact_disagreement(schema, local, remote)
    doubled = dataclasses.replace(schema, scale=schema.scale * 2)
    assert scoring.exact_disagreement(doubled, local, remote) == pytest.approx(2 * base)


def test_local_update_magnitude_is_zero_for_an_untouched_component():
    peer = make_peer(1)
    for component in (0, 5):
        block = peer.components.initial_component(component)
        assert scoring.local_update_magnitude_for(
            peer.components, component, block) == pytest.approx(0.0)


def test_local_update_magnitude_grows_as_the_peer_moves():
    peer = make_peer(1)
    base = peer.components.initial_component(0)
    previous = 0.0
    for factor in (0.1, 0.5, 2.0, 10.0):
        moved = {k: v + factor for k, v in base.items()}
        value = scoring.local_update_magnitude_for(peer.components, 0, moved)
        assert value > previous
        previous = value


# -- the deployable sketch -----------------------------------------------------------------

def test_both_peers_build_the_identical_projection_without_exchanging_it():
    a, b = make_peer(1), make_peer(2)
    for side in ("A", "B"):
        pa = scoring.projection(a.components, a.components.lora.modules[0].module,
                                side, 8, 4)
        pb = scoring.projection(b.components, b.components.lora.modules[0].module,
                                side, 8, 4)
        assert torch.equal(pa, pb)


def test_the_sketch_tracks_a_one_sided_component_closely_enough_to_rank_it():
    """A component one peer has never touched is far from the other's value.

    The sketch estimates that gap within 15% of the exact score, and ranks it
    above a component the two peers nearly agree on.
    """
    peer = make_peer(1)
    schema = peer.components
    base = peer.state.gather(0)
    untouched = schema.initial_component(0)
    moved = randomise(base, 31)

    exact = scoring.exact_disagreement(schema, untouched, moved)
    estimate = scoring.sketch_disagreement(
        schema, scoring.component_norms(schema, untouched),
        scoring.component_sketch(schema, untouched, 64),
        scoring.component_norms(schema, moved),
        scoring.component_sketch(schema, moved, 64))
    assert estimate == pytest.approx(exact, rel=0.15)

    # ...and it still ranks above a component the two peers nearly agree on.
    near = {k: v + 1e-3 * torch.randn(v.shape, generator=torch.Generator().manual_seed(3),
                                      dtype=v.dtype) for k, v in moved.items()}
    close_estimate = scoring.sketch_disagreement(
        schema, scoring.component_norms(schema, moved),
        scoring.component_sketch(schema, moved, 64),
        scoring.component_norms(schema, near),
        scoring.component_sketch(schema, near, 64))
    assert estimate > close_estimate


@pytest.mark.parametrize("delta", [0.002, 0.01, 0.05])
def test_the_sketch_carries_real_signal_in_the_close_regime(delta):
    """Small gaps, the common case between peers that gossip.

    The sketch estimates the difference ``P(a_j - a_i)``, so its noise scales with
    the gap rather than with the component's magnitude: the estimate stays
    positive and within 35% of the exact score.
    """
    peer = make_peer(1)
    schema = peer.components
    base = peer.state.gather(0)
    generator = torch.Generator().manual_seed(7)
    local = randomise(base, 41)
    remote = {k: v + delta * torch.randn(v.shape, generator=generator, dtype=v.dtype)
              for k, v in local.items()}

    exact = scoring.exact_disagreement(schema, local, remote)
    estimate = scoring.sketch_disagreement(
        schema, scoring.component_norms(schema, local),
        scoring.component_sketch(schema, local, 16),
        scoring.component_norms(schema, remote),
        scoring.component_sketch(schema, remote, 16))
    assert estimate > 0.0, "the estimator collapsed to the perfect-agreement value"
    assert estimate == pytest.approx(exact, rel=0.35)


def test_the_sketch_ranks_the_most_disagreeing_component_in_the_close_regime():
    """Ranking is the only thing selection needs, so measure ranking directly."""
    peer = make_peer(1)
    schema = peer.components
    base = peer.state.gather(0)

    agreed = 0
    trials = 20
    for trial in range(trials):
        generator = torch.Generator().manual_seed(5000 + trial)
        local = {k: torch.rand(v.shape, generator=generator, dtype=v.dtype) - 0.5
                 for k, v in base.items()}
        exact, sketch = {}, {}
        for index, delta in enumerate((0.01, 0.02, 0.04, 0.08)):
            remote = {k: v + delta * torch.randn(v.shape, generator=generator,
                                                 dtype=v.dtype)
                      for k, v in local.items()}
            exact[index] = scoring.exact_disagreement(schema, local, remote)
            sketch[index] = scoring.sketch_disagreement(
                schema, scoring.component_norms(schema, local),
                scoring.component_sketch(schema, local, 16),
                scoring.component_norms(schema, remote),
                scoring.component_sketch(schema, remote, 16))
        agreed += max(exact, key=exact.get) == max(sketch, key=sketch.get)
    assert agreed >= trials - 2, f"ranked correctly only {agreed}/{trials}"


def test_the_sketch_converges_on_the_exact_score_as_it_widens(pair):
    schema, local, remote = pair
    exact = scoring.exact_disagreement(schema, local, remote)
    ln = scoring.component_norms(schema, local)
    rn = scoring.component_norms(schema, remote)

    def error(dim):
        return abs(scoring.sketch_disagreement(
            schema, ln, scoring.component_sketch(schema, local, dim),
            rn, scoring.component_sketch(schema, remote, dim)) - exact) / exact

    narrow = sum(error(2) for _ in range(1)) / 1
    wide = sum(error(d) for d in (256, 512, 1024)) / 3
    assert wide < narrow, f"widening did not help: {narrow=} {wide=}"
    assert wide < 0.05, f"a wide sketch should track the exact score closely: {wide}"


def test_the_sketch_is_symmetric(pair):
    schema, local, remote = pair
    ln, rn = scoring.component_norms(schema, local), scoring.component_norms(schema, remote)
    ls, rs = (scoring.component_sketch(schema, local, 16),
              scoring.component_sketch(schema, remote, 16))
    assert (scoring.sketch_disagreement(schema, ln, ls, rn, rs)
            == pytest.approx(scoring.sketch_disagreement(schema, rn, rs, ln, ls)))


def test_a_sketch_of_the_wrong_width_is_rejected(pair):
    schema, local, remote = pair
    ln, rn = scoring.component_norms(schema, local), scoring.component_norms(schema, remote)
    with pytest.raises(ScoringError, match="sketch widths disagree"):
        scoring.sketch_disagreement(schema, ln, scoring.component_sketch(schema, local, 4),
                                    rn, scoring.component_sketch(schema, remote, 8))


def test_a_sketch_covering_the_wrong_module_count_is_rejected(pair):
    schema, local, remote = pair
    ln = scoring.component_norms(schema, local)
    with pytest.raises(ScoringError, match="covers 1 modules"):
        scoring.sketch_disagreement(schema, ln[:1],
                                    scoring.component_sketch(schema, local, 4),
                                    scoring.component_norms(schema, remote),
                                    scoring.component_sketch(schema, remote, 4))


def test_a_non_positive_sketch_width_is_rejected(pair):
    schema, local, _ = pair
    with pytest.raises(ScoringError, match="sketch dim must be positive"):
        scoring.component_sketch(schema, local, 0)


# -- numerical stability and clamp reporting ---------------------------------------------

@pytest.mark.parametrize("delta", [1e-7, 1e-6, 1e-5, 1e-3])
def test_the_exact_score_does_not_cancel_when_the_peers_are_nearly_identical(delta):
    """The expanded form cancels when the peers are close; the difference form does not.

    For nearly identical peers the score stays positive and matches the allocated
    definition to 1e-9.
    """
    peer = make_peer(1)
    schema = peer.components
    base = peer.state.gather(0)
    generator = torch.Generator().manual_seed(7)
    local = randomise(base, 51)
    remote = {k: v + delta * torch.randn(v.shape, generator=generator, dtype=v.dtype)
              for k, v in local.items()}

    got = scoring.exact_disagreement(schema, local, remote)
    reference = brute_force(schema, local, remote)
    assert got > 0.0, "a genuinely different pair scored as perfect agreement"
    assert got == pytest.approx(reference, rel=1e-9)


def test_a_clamped_estimate_is_reported_rather_than_hidden():
    """0.0 is what perfect agreement scores, so a clamp must stay distinguishable."""
    peer = make_peer(1)
    schema = peer.components
    block = peer.state.gather(0)

    report = {}
    value = scoring.exact_disagreement(schema, block, block, report)
    assert value == 0.0
    assert report["clamped"] is False, "genuine agreement is not a clamp"
    assert report["raw_total"] == pytest.approx(0.0, abs=1e-12)

    # A hand-built impossible total exercises the reporting path directly.
    forced = {}
    assert scoring._finish(schema, -1.0, forced) == 0.0
    assert forced["clamped"] is True and forced["raw_total"] == -1.0


def test_the_session_counts_clamped_scores_in_its_metrics():
    from federatedscope.contrib.decentralized.tests.methods.test_pulse_peer import pulse_cfg
    cfg = pulse_cfg(**{"decentralized.methods.pulse_lora.selector": "sketch_disagreement"})
    peer = make_peer(1, cfg=cfg)
    assert "clamped_scores" in peer.policy.metrics
    assert peer.method_metrics()["pulse_clamped_scores"] == 0
