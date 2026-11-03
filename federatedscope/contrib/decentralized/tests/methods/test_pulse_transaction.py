"""PULSE-LoRA transactional mixing: stage, commit, rollback.

Run: python -m pytest federatedscope/contrib/decentralized/tests/methods/test_pulse_transaction.py -q -p no:cacheprovider

The transaction engine sequences the two endpoints; these tests cover the peer's
side: staging writes nothing, a commit is all-or-nothing across every injected
layer, and a failure anywhere restores the peer exactly, optimizer moments and
relay membership included, while the bytes stay charged.
"""
import ast
import os
import pathlib

import pytest
import torch

from federatedscope.contrib.decentralized.core.accounting import ByteLedger
from federatedscope.contrib.decentralized.core.pair_session import execute_pair_session
from federatedscope.contrib.decentralized.core.transaction import (
    FailureInjector, TransactionEngine,
)
from federatedscope.contrib.decentralized.methods.pulse_lora import units
from federatedscope.contrib.decentralized.methods.pulse_lora.mixing import MixError
from federatedscope.contrib.decentralized.tests.methods.test_pulse_peer import (
    make_peer, pulse_cfg,
)
from federatedscope.contrib.decentralized.tests.methods.test_pulse_session import (
    budget_for, event, negotiate, trained,
)

#: Contact duration (ms) giving 875 bytes per direction at the default goodput:
#: about 560 bytes of payload credit after the handshake, which affords exactly
#: one component.
ONE_COMPONENT = 7


def peers(selector="full_exchange", **over):
    settings = {"decentralized.methods.pulse_lora.selector": selector}
    settings.update(over)
    cfg = pulse_cfg(**settings)
    return make_peer(1, cfg=cfg), make_peer(2, cfg=cfg)


def with_optimizer(peer, seed=0):
    """A real AdamW with live moments, so moment handling is actually exercised."""
    params = peer.bridge.parameters(peer.bridge.schema.lora_keys())
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    device = next(peer.bridge.model.parameters()).device
    inputs = torch.randn(4, 8, generator=torch.Generator().manual_seed(seed)).to(device)
    peer.bridge.model(inputs).pow(2).mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    class Trainer:
        pass

    trainer = Trainer()
    trainer.optimizer = optimizer
    peer.trainer = trainer
    return optimizer


MOMENT_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")


def canonical_moments(peer):
    """Every optimizer moment buffer, by canonical parameter name and moment key.

    Compared tensor by tensor rather than by norms: a permutation of rows, which
    is what confusing a component's slot with its identity produces, leaves the
    norms unchanged. ``exp_avg_sq`` and ``max_exp_avg_sq`` are included wherever
    the optimizer keeps them.
    """
    optimizer = getattr(peer.trainer, "optimizer", None) if peer.trainer else None
    if optimizer is None:
        return {}
    by_id = {id(param): key for key, param in peer.bridge._params.items()}
    out = {}
    for param, state in optimizer.state.items():
        name = by_id.get(id(param), f"<untracked:{id(param)}>")
        for moment in MOMENT_KEYS:
            buffer = state.get(moment)
            if torch.is_tensor(buffer):
                out[f"{name}|{moment}"] = buffer.detach().cpu().clone()
    return out


def fingerprint(peer):
    """Everything a transaction could change, including the selector state.

    The selector state is peer state too: it decides what the next session sends.
    """
    return {
        "support": peer.state.support,
        "materialised": peer.state.materialised,
        "relay": peer.state.relay_components,
        "version": peer.state_version(),
        "blocks": {c: {k: v.detach().cpu().clone()
                       for k, v in peer.state.gather(c).items()}
                   for c in range(peer.components.r_max)},
        "moments": canonical_moments(peer),
        "scale": peer.bridge.live_scale(),
        "selector_memory": peer.policy.memory.state_dict(),
        "selector_digest": peer.policy.memory.selector_digest(1, 2),
        "pending": dict(peer.policy._pending),
    }


def assert_identical(peer, before, why):
    after = fingerprint(peer)
    for field in ("support", "materialised", "relay", "version",
                  "selector_memory", "selector_digest"):
        assert after[field] == before[field], \
            f"{why}: {field} changed\n  before={before[field]}\n  after ={after[field]}"
    assert after["scale"] == pytest.approx(before["scale"]), f"{why}: fixed scale changed"

    assert set(after["moments"]) == set(before["moments"]), \
        f"{why}: the set of optimizer moment buffers changed"
    for name, tensor in before["moments"].items():
        assert torch.equal(after["moments"][name], tensor), \
            f"{why}: optimizer buffer {name} was not restored exactly"

    for component, block in before["blocks"].items():
        for key, tensor in block.items():
            assert torch.equal(after["blocks"][component][key], tensor), \
                f"{why}: component {component} slice {key} changed"
    peer.state.check_invariants()


# -- the transaction engine is method-agnostic -------------------------------------------

def test_no_pulse_branch_exists_in_the_transaction_engine():
    engine = pathlib.Path(__file__).resolve().parents[2].joinpath("core/transaction.py").resolve()
    text = engine.read_text()
    for name in ("pulse", "component", "relay", "materialis", "gossip"):
        assert name not in text.lower(), f"core/transaction.py names {name!r}"
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", "") or ""
            assert "methods" not in module and "lora" not in module


def test_the_engine_holds_no_learning_state_mid_transaction():
    """No learning state is reachable from the engine while a mix is staged.

    ``find_learning_state`` scans the engine and its transaction state for
    tensors, modules, optimizers and peers; the snapshot lives in the peer's mixer.
    """
    from federatedscope.contrib.decentralized.tests.core.ownership import (
        find_learning_state,
    )

    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    with_optimizer(a); with_optimizer(b)
    ev = event()
    view_a, view_b = negotiate(a, b, ev)
    shared = budget_for(ev)
    a.plan_payload(ev, shared, view_a)
    body_b = b.plan_payload(ev, shared, view_b)

    engine = TransactionEngine()
    state = engine.begin(ev.endpoints)
    token = a.stage(body_b, ev)
    assert token in a.mixer.open_tokens, "the snapshot must live in the peer's mixer"

    offenders = find_learning_state(engine) + find_learning_state(state)
    assert not offenders, (
        f"the engine or its state reached learning objects mid-transaction: "
        f"{sorted(set(offenders))[:8]}")


# -- staging writes nothing ------------------------------------------------------------

def test_staging_does_not_mutate_either_plane():
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    with_optimizer(a); with_optimizer(b)
    ev = event()
    view_a, view_b = negotiate(a, b, ev)
    shared = budget_for(ev)
    a.plan_payload(ev, shared, view_a)
    body_b = b.plan_payload(ev, shared, view_b)

    before = fingerprint(a)
    a.stage(body_b, ev)
    assert_identical(a, before, "staging wrote")


def test_the_staging_token_is_an_inert_scalar():
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    ev = event()
    view_a, view_b = negotiate(a, b, ev)
    shared = budget_for(ev)
    a.plan_payload(ev, shared, view_a)
    token = a.stage(b.plan_payload(ev, shared, view_b), ev)
    assert isinstance(token, int) and not isinstance(token, bool)


# -- rejections mutate nothing -----------------------------------------------------------

def staged_pair():
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    with_optimizer(a); with_optimizer(b)
    ev = event()
    view_a, view_b = negotiate(a, b, ev)
    shared = budget_for(ev)
    a.plan_payload(ev, shared, view_a)
    return a, b, ev, b.plan_payload(ev, shared, view_b)


@pytest.mark.parametrize("mangle,error", [
    (lambda body: body[:-8], Exception),                 # truncated
    (lambda body: body + b"\x00" * 8, Exception),        # trailing bytes
    (lambda body: b"", Exception),                       # empty
])
def test_a_malformed_payload_mutates_nothing(mangle, error):
    a, b, ev, body = staged_pair()
    before = fingerprint(a)
    with pytest.raises(error):
        a.stage(mangle(body), ev)
    assert_identical(a, before, "a malformed payload")


def test_a_payload_carrying_a_different_component_set_is_rejected():
    """Both endpoints apply the agreed set, or neither does."""
    a, b, ev, body = staged_pair()
    decoded = units.decode_units(b.components, body)
    trimmed = {c: block for c, block in list(decoded.items())[:-1]}
    before = fingerprint(a)
    with pytest.raises(MixError, match="agreed on"):
        a.stage(units.encode_units(b.components, trimmed), ev)
    assert_identical(a, before, "a mismatched component set")


def test_a_payload_for_an_unplanned_session_is_rejected():
    a, b, ev, body = staged_pair()
    a.policy._pending.clear()
    before = fingerprint(a)
    with pytest.raises(MixError, match="planned nothing"):
        a.stage(body, ev)
    assert_identical(a, before, "an unplanned session")


def test_a_stale_local_version_is_rejected():
    """The mix was computed against a state that has since moved."""
    a, b, ev, body = staged_pair()
    a._version += 1
    before = fingerprint(a)
    with pytest.raises(MixError, match="between planning and staging"):
        a.stage(body, ev)
    assert_identical(a, before, "a stale version")


def test_a_partial_cross_layer_block_can_never_be_observed():
    """A block missing one module's A slice is refused and the peer is unchanged.

    The refusal comes from ``encode_units``, which validates the block before
    ``stage`` is reached.
    """
    a, b, ev, body = staged_pair()
    decoded = units.decode_units(b.components, body)
    victim = sorted(decoded)[0]
    module = a.components.lora.modules[1].module
    decoded[victim].pop(f"{module}|A")
    before = fingerprint(a)
    with pytest.raises(Exception):
        a.stage(units.encode_units(b.components, decoded), ev)
    assert_identical(a, before, "a partial cross-layer block")


# -- commit --------------------------------------------------------------------------------

def test_a_commit_applies_exactly_the_agreed_component_and_nothing_else():
    """Not merely "one component moved" - the one that moved is the agreed one.

    The selection is captured from the peer's own pending decision, so the test
    names which component should change rather than inferring it from what did.
    """
    a, b = peers("local_update",
                 **{"decentralized.methods.pulse_lora.exploration_fraction": 0.0})
    trained(a, seed=5); trained(b, seed=6)
    before = {c: {k: v.clone() for k, v in a.state.gather(c).items()}
              for c in range(a.components.r_max)}

    selected = {}

    original = a.policy.acknowledge

    def capture(session, committed, reason=""):
        edge = a.policy.edge_of(session)
        pending = a.policy.pending_for(edge)
        if pending is not None:
            selected["components"] = pending.selected
        original(session, committed, reason)

    a.policy.acknowledge = capture
    try:
        record = execute_pair_session(event(1, 2, duration=ONE_COMPONENT), {1: a, 2: b},
                                      TransactionEngine(), ByteLedger())
    finally:
        del a.policy.acknowledge

    assert record["status"] == "committed"
    agreed = selected["components"]
    assert len(agreed) == 1, f"the budget should afford exactly one, got {agreed}"

    moved = [c for c in before
             if not all(torch.equal(a.state.gather(c)[k], v)
                        for k, v in before[c].items())]
    assert moved == list(agreed), (
        f"the agreed component was {list(agreed)} but {moved} changed")

    unselected = [c for c in before if c not in agreed]
    assert len(unselected) == a.components.r_max - 1
    for component in unselected:
        for key, tensor in before[component].items():
            assert torch.equal(a.state.gather(component)[key], tensor), (
                f"unselected component {component} slice {key} was not bitwise unchanged")
    a.state.check_invariants()


def test_a_commit_leaves_the_active_support_unchanged():
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    supports = (a.state.support, b.state.support)
    execute_pair_session(event(), {1: a, 2: b}, TransactionEngine(), ByteLedger())
    assert (a.state.support, b.state.support) == supports


def test_a_commit_preserves_the_pair_sum_of_every_mixed_component():
    """The kernel is mean-preserving, and the commit must not disturb that."""
    a, b = peers()
    trained(a, seed=3); trained(b, seed=4)
    before = {c: {k: a.state.gather(c)[k] + b.state.gather(c)[k]
                  for k in a.state.gather(c)}
              for c in a.state.support}

    execute_pair_session(event(), {1: a, 2: b}, TransactionEngine(), ByteLedger())

    for component, sums in before.items():
        for key, total in sums.items():
            now = a.state.gather(component)[key] + b.state.gather(component)[key]
            assert torch.allclose(now, total, atol=1e-6), \
                f"component {component} slice {key} did not conserve"


def test_repeated_mixing_reaches_consensus():
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)

    def gap():
        return sum(float((a.state.gather(c)[k] - b.state.gather(c)[k]).norm())
                   for c in a.state.support for k in a.state.gather(c))

    start = gap()
    assert start > 0
    execute_pair_session(event(), {1: a, 2: b}, TransactionEngine(), ByteLedger())
    assert gap() < start * 1e-6, f"gamma=0.5 averages exactly; gap went {start} -> {gap()}"


def test_the_fixed_scale_survives_a_commit():
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    execute_pair_session(event(), {1: a, 2: b}, TransactionEngine(), ByteLedger())
    for peer in (a, b):
        peer.bridge.assert_fixed_scale(peer.components.scale)


def test_a_committed_mix_bumps_the_state_version():
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    before = (a.state_version(), b.state_version())
    execute_pair_session(event(), {1: a, 2: b}, TransactionEngine(), ByteLedger())
    assert (a.state_version(), b.state_version()) == (before[0] + 1, before[1] + 1)


# -- the shared comparison coefficient is actually consumed ------------------------------

@pytest.mark.parametrize("gamma", [0.1, 0.25, 0.5])
def test_the_mix_follows_the_shared_comparison_gamma(gamma):
    """PULSE-LoRA has no gamma key; it mixes at ``decentralized.comparison.gamma``.

    Each committed component equals ``(1 - g) x_i + g x_j`` at that ``g``.
    """
    a, b = peers(**{"decentralized.comparison.gamma": gamma,
                    # Kept equal to comparison.gamma. Config validation compares
                    # only the running method's gamma, so a pulse_lora run does
                    # not require it.
                    "decentralized.methods.dec_lora.gamma": gamma})
    trained(a, seed=11); trained(b, seed=12)

    before_a = {c: {k: v.clone() for k, v in a.state.gather(c).items()}
                for c in a.state.support}
    before_b = {c: {k: v.clone() for k, v in b.state.gather(c).items()}
                for c in b.state.support}

    record = execute_pair_session(event(), {1: a, 2: b},
                                  TransactionEngine(), ByteLedger())
    assert record["status"] == "committed"

    for component in before_a:
        for key in before_a[component]:
            mine, theirs = before_a[component][key], before_b[component][key]
            assert torch.allclose(a.state.gather(component)[key],
                                  (1 - gamma) * mine + gamma * theirs, atol=1e-6), (
                f"peer 1 component {component} does not match (1-g)x_i + g x_j at g={gamma}")
            assert torch.allclose(b.state.gather(component)[key],
                                  (1 - gamma) * theirs + gamma * mine, atol=1e-6), (
                f"peer 2 component {component} does not match the mirrored form")


@pytest.mark.parametrize("gamma", [0.1, 0.25, 0.5])
def test_every_gamma_preserves_the_pair_sum(gamma):
    a, b = peers(**{"decentralized.comparison.gamma": gamma,
                    # Kept equal to comparison.gamma; not required for pulse_lora.
                    "decentralized.methods.dec_lora.gamma": gamma})
    trained(a, seed=13); trained(b, seed=14)
    before = {c: {k: a.state.gather(c)[k] + b.state.gather(c)[k]
                  for k in a.state.gather(c)} for c in a.state.support}

    execute_pair_session(event(), {1: a, 2: b}, TransactionEngine(), ByteLedger())

    for component, sums in before.items():
        for key, total in sums.items():
            now = a.state.gather(component)[key] + b.state.gather(component)[key]
            assert torch.allclose(now, total, atol=1e-6)


@pytest.mark.parametrize("gamma,expected", [(0.1, 0.8), (0.25, 0.5), (0.5, 0.0)])
def test_every_gamma_contracts_disagreement_by_the_expected_factor(gamma, expected):
    """One bilateral step multiplies the gap by exactly ``|1 - 2g|``."""
    a, b = peers(**{"decentralized.comparison.gamma": gamma,
                    # Kept equal to comparison.gamma; not required for pulse_lora.
                    "decentralized.methods.dec_lora.gamma": gamma})
    trained(a, seed=15); trained(b, seed=16)

    def gap():
        return sum(float((a.state.gather(c)[k] - b.state.gather(c)[k]).norm())
                   for c in a.state.support for k in a.state.gather(c))

    start = gap()
    assert start > 0
    execute_pair_session(event(), {1: a, 2: b}, TransactionEngine(), ByteLedger())
    assert gap() == pytest.approx(start * expected, abs=1e-5)


# -- moments -----------------------------------------------------------------------------------

def test_only_the_mixed_active_components_have_their_moments_reset():
    a, b = peers("local_update",
                 **{"decentralized.methods.pulse_lora.exploration_fraction": 0.0})
    trained(a, seed=5); trained(b, seed=6)
    optimizer = with_optimizer(a, seed=3)
    with_optimizer(b, seed=4)

    param = a.bridge._params[a.bridge.schema.modules[0].a_key]
    before = optimizer.state[param]["exp_avg"].clone()

    record = execute_pair_session(event(1, 2, duration=ONE_COMPONENT), {1: a, 2: b},
                                  TransactionEngine(), ByteLedger())
    assert record["status"] == "committed"

    after = optimizer.state[param]["exp_avg"]
    zeroed = [row for row in range(after.shape[0]) if bool((after[row] == 0).all())]
    assert len(zeroed) == 1, f"expected exactly one row reset, got {zeroed}"
    for row in range(after.shape[0]):
        if row not in zeroed:
            assert torch.equal(after[row], before[row]), \
                f"row {row} lost its momentum but was not mixed"


def test_a_relay_component_gets_no_optimizer_state_from_a_mix():
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    a.state.prune([3])
    optimizer = with_optimizer(a)
    tracked = len(optimizer.state)

    execute_pair_session(event(), {1: a, 2: b}, TransactionEngine(), ByteLedger())

    assert 3 in a.state.relay_components
    assert len(optimizer.state) == tracked, "mixing created optimizer state"
    a.state.check_invariants()


def test_surviving_moments_follow_component_identity_not_slot_position():
    """Prune first, so identity and compact position genuinely disagree."""
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    optimizer = with_optimizer(a)
    a.state.prune([1], optimizer=optimizer)          # 0,2,3 survive at slots 0,1,2
    assert a.state.support == (0, 2, 3)
    assert a.state.compact_index(2) == 1, "identity 2 now lives in slot 1"

    param = a.bridge._params[a.bridge.schema.modules[0].a_key]
    before = optimizer.state[param]["exp_avg"].clone()

    a.clear_moments_for([2])

    after = optimizer.state[param]["exp_avg"]
    assert bool((after[1] == 0).all()), "component 2's own slot was not the one reset"
    assert torch.equal(after[0], before[0]) and torch.equal(after[2], before[2])


# -- rollback ------------------------------------------------------------------------------------

@pytest.mark.parametrize("stage", ["stage", "commit"])
@pytest.mark.parametrize("every", [1, 2])
def test_failure_at_every_boundary_restores_everything_exactly(stage, every):
    """``every=2`` fails the second endpoint, after the first has succeeded."""
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    with_optimizer(a, seed=3); with_optimizer(b, seed=4)
    before = {1: fingerprint(a), 2: fingerprint(b)}

    ledger = ByteLedger()
    record = execute_pair_session(event(), {1: a, 2: b}, TransactionEngine(), ledger,
                                  failure_injector=FailureInjector(stage, every))

    assert record["status"] == "failed"
    assert_identical(a, before[1], f"failure at {stage} #{every}")
    assert_identical(b, before[2], f"failure at {stage} #{every}")
    assert record["payload_bytes"] > 0, "a rollback cannot un-send"


@pytest.mark.parametrize("stage", ["stage", "commit"])
def test_a_failed_transaction_still_charges_the_bytes(stage):
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    ledger = ByteLedger()
    record = execute_pair_session(event(), {1: a, 2: b}, TransactionEngine(), ledger,
                                  failure_injector=FailureInjector(stage, 1))
    assert record["status"] == "failed"
    assert record["payload_bytes"] > 0 and record["control_bytes"] > 0
    assert ledger.total() > 0


def test_a_rollback_un_materialises_a_component_that_was_lazy_before():
    """Restoring a value is not enough - membership of M_i is state too."""
    a, b = peers()
    trained(a, seed=1)
    a.state.prune([2, 3])                       # a holds 0,1 active and 2,3 in relay
    trained(b, seed=2)
    b.state.materialise(6, b.components.initial_component(6))
    before = a.state.materialised
    assert 6 not in before

    record = execute_pair_session(event(), {1: a, 2: b}, TransactionEngine(), ByteLedger(),
                                  failure_injector=FailureInjector("commit", 2))
    assert record["status"] == "failed"
    assert a.state.materialised == before, (
        "a rolled-back mix left a component materialised")
    a.state.check_invariants()


def test_a_rollback_restores_the_version_counter():
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    before = (a.state_version(), b.state_version())
    execute_pair_session(event(), {1: a, 2: b}, TransactionEngine(), ByteLedger(),
                         failure_injector=FailureInjector("commit", 2))
    assert (a.state_version(), b.state_version()) == before


def test_a_failure_part_way_through_a_commit_restores_everything():
    """A commit that fails after writing one component leaves the peer as it was.

    The failure injector fires between endpoints, not inside one endpoint's
    write, so ``scatter`` is wrapped to fail on the second component after the
    first has been written.
    """
    a, b, ev, body = staged_pair()
    before = fingerprint(a)
    token = a.stage(body, ev)
    staged = a.mixer._staged[token]
    assert len(staged.components) >= 2, "this test needs at least two components"

    victim = staged.components[1]
    original = a.state.scatter
    written = []

    def scatter(component, block):
        if component == victim:
            raise RuntimeError("injected mid-commit failure")
        written.append(component)
        original(component, block)

    a.state.scatter = scatter
    try:
        with pytest.raises(RuntimeError, match="injected mid-commit"):
            a.mixer.commit(token)
    finally:
        del a.state.scatter

    assert written, "the first component was never written, so nothing was half-applied"
    assert_identical(a, before, "a mid-commit failure")


def test_the_snapshot_is_released_only_by_discard():
    """A committed endpoint stays undoable until the transaction finishes."""
    a, b, ev, body = staged_pair()
    token = a.stage(body, ev)
    a.commit(token)
    assert token in a.mixer.open_tokens, "commit released the snapshot too early"
    a.discard(token)
    assert token not in a.mixer.open_tokens


def test_rolling_back_after_a_commit_undoes_it():
    a, b, ev, body = staged_pair()
    before = fingerprint(a)
    token = a.stage(body, ev)
    a.commit(token)
    now = fingerprint(a)["blocks"]
    assert any(not torch.equal(now[c][k], v)
               for c in before["blocks"] for k, v in before["blocks"][c].items()), \
        "the commit changed nothing, so the rollback below would prove nothing"
    a.rollback(token)
    assert_identical(a, before, "rollback after commit")


# -- CUDA: the relay plane is CPU, the compact tensors are not -------------------------
#
# Mixing gathers from both planes at once: gather() returns CUDA slices for active
# components and CPU slices for relay ones, and decoded payloads are on the CPU, so
# plan_mix moves each remote block to the device of the local one before mixing.

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def on_cuda(peer):
    device = torch.device(f"cuda:{os.environ.get('PULSE_TEST_GPU', '0')}")
    peer.bridge.model.to(device)
    peer.bridge.refresh_schema()
    return peer


@CUDA
def test_a_mix_commits_correctly_with_the_model_on_cuda():
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    on_cuda(a); on_cuda(b)

    before = {c: {k: v.detach().cpu().clone() for k, v in a.state.gather(c).items()}
              for c in a.state.support}
    record = execute_pair_session(event(), {1: a, 2: b},
                                  TransactionEngine(), ByteLedger())

    assert record["status"] == "committed"
    moved = [c for c in before
             if not all(torch.equal(a.state.gather(c)[k].detach().cpu(), v)
                        for k, v in before[c].items())]
    assert moved, "nothing mixed on CUDA"
    a.state.check_invariants()
    a.bridge.assert_fixed_scale(a.components.scale)


@CUDA
def test_a_cuda_rollback_restores_exactly():
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    on_cuda(a); on_cuda(b)
    with_optimizer(a, seed=3); with_optimizer(b, seed=4)
    before = {1: fingerprint(a), 2: fingerprint(b)}

    record = execute_pair_session(event(), {1: a, 2: b}, TransactionEngine(),
                                  ByteLedger(), failure_injector=FailureInjector("commit", 2))

    assert record["status"] == "failed"
    assert_identical(a, before[1], "cuda rollback")
    assert_identical(b, before[2], "cuda rollback")


@CUDA
def test_relay_storage_stays_on_cpu_after_a_cuda_mix():
    """Inactive components must not pin GPU memory."""
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    a.state.prune([3])
    on_cuda(a); on_cuda(b)

    execute_pair_session(event(), {1: a, 2: b}, TransactionEngine(), ByteLedger())

    assert 3 in a.state.relay_components
    for tensor in a.state.gather(3).values():
        assert tensor.device.type == "cpu", "a relay slice moved onto the GPU"


@CUDA
def test_moment_migration_survives_a_prune_then_a_mix_on_cuda():
    a, b = peers()
    trained(a, seed=1); trained(b, seed=2)
    on_cuda(a); on_cuda(b)
    optimizer = with_optimizer(a, seed=5)
    a.state.prune([1], optimizer=optimizer)
    assert a.state.support == (0, 2, 3)

    record = execute_pair_session(event(), {1: a, 2: b},
                                  TransactionEngine(), ByteLedger())
    assert record["status"] == "committed"
    a.state.check_invariants()
    for spec in a.bridge.schema.modules:
        buffer = optimizer.state[a.bridge._params[spec.a_key]]["exp_avg"]
        assert buffer.shape[0] == 3, "moment buffer did not track the compacted rank"
