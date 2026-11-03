"""One designated component, traced from source to far peer.

The source, middle and far peers form a chain, and the middle peer does not
train the designated component. The component is tagged at the source with a
value no other peer holds and followed hop by hop, and every session's
bilateral sums are checked before and after.
"""
import pytest
import torch

from federatedscope.contrib.decentralized.core.accounting import ByteLedger
from federatedscope.contrib.decentralized.core.pair_session import execute_pair_session
from federatedscope.contrib.decentralized.core.transaction import TransactionEngine
from federatedscope.contrib.decentralized.tests.methods.test_pulse_peer import (
    make_peer, pulse_cfg,
)
from federatedscope.contrib.decentralized.tests.methods.test_pulse_session import (
    event, trained,
)

#: The component tagged at the source and followed to the far peer. It is active
#: on the source and the far peer, and pruned away on the middle one.
TAGGED = 3
#: The middle peer's support after pruning.
MIDDLE_SUPPORT = (0, 1)


def chain(selector, exploration=0.5):
    """source -- middle -- far, with the middle peer pruned to MIDDLE_SUPPORT."""
    cfg = pulse_cfg(**{
        "decentralized.methods.pulse_lora.selector": selector,
        "decentralized.methods.pulse_lora.exploration_fraction": exploration,
    })
    source, middle, far = (make_peer(1, cfg=cfg), make_peer(2, cfg=cfg),
                           make_peer(3, cfg=cfg))
    trained(source, seed=11)
    trained(far, seed=13)
    middle.state.prune([c for c in middle.state.support if c not in MIDDLE_SUPPORT])
    assert middle.state.support == MIDDLE_SUPPORT
    return source, middle, far


def tag(peer, component=TAGGED, marker=7.25):
    """A value nothing else could produce, so its arrival is unambiguous."""
    block = peer.state.gather(component)
    tagged = {key: torch.full_like(value, marker + index * 0.5)
              for index, (key, value) in enumerate(sorted(block.items()))}
    peer.state.scatter(component, tagged)
    return {k: v.clone() for k, v in tagged.items()}


def block_of(peer, component):
    return {k: v.clone() for k, v in peer.state.gather(component).items()}


def pair_sums(left, right, components):
    return {c: {k: left.state.gather(c)[k] + right.state.gather(c)[k]
                for k in left.state.gather(c)} for c in components}


def assert_sums_conserved(before, left, right, label):
    for component, sums in before.items():
        for key, total in sums.items():
            now = left.state.gather(component)[key] + right.state.gather(component)[key]
            assert torch.allclose(now, total, atol=1e-6), (
                f"{label}: component {component} slice {key} did not conserve")


def hop(a, b, peer_ids, watched):
    """One session; the bilateral sums of the watched components must be conserved."""
    before = pair_sums(a, b, watched)
    record = execute_pair_session(event(*peer_ids), {peer_ids[0]: a, peer_ids[1]: b},
                                  TransactionEngine(), ByteLedger())
    assert_sums_conserved(before, a, b, f"hop {peer_ids}")
    return record


# -- the tagged component travels the whole chain ---------------------------------------

@pytest.mark.parametrize("selector", ["local_update", "full_exchange"])
def test_the_tagged_component_reaches_the_far_peer_through_an_inactive_middle(selector):
    source, middle, far = chain(selector)
    tagged_value = tag(source)
    far_before = block_of(far, TAGGED)
    middle_before = block_of(middle, TAGGED)

    first = hop(source, middle, (1, 2), watched=range(source.components.r_max))
    assert first["status"] == "committed"

    # The middle peer took it into relay and did not activate it.
    assert middle.state.support == MIDDLE_SUPPORT, "the middle peer grew a support"
    assert TAGGED in middle.state.relay_components
    middle_after_first = block_of(middle, TAGGED)
    assert any(not torch.equal(middle_after_first[k], v)
               for k, v in middle_before.items()), (
        "the middle peer did not take up the tagged component at all")

    second = hop(middle, far, (2, 3), watched=range(middle.components.r_max))
    assert second["status"] == "committed"

    far_after = block_of(far, TAGGED)
    assert any(not torch.equal(far_after[k], v) for k, v in far_before.items()), (
        "the tagged component never reached the far peer")

    # It moved towards the tagged value, which only the source held.
    for key, value in tagged_value.items():
        before_gap = float((far_before[key] - value).norm())
        after_gap = float((far_after[key] - value).norm())
        assert after_gap < before_gap, (
            f"{key}: the far peer did not move towards the tagged value "
            f"({before_gap:.4f} -> {after_gap:.4f})")

    assert middle.state.support == MIDDLE_SUPPORT
    middle.state.check_invariants()
    far.state.check_invariants()


def test_every_component_outside_the_selected_set_is_bitwise_unchanged():
    """The unselected set is fixed in advance from the far peer's own selection.

    The selected set is read from the peer's pending plan before the session
    runs, rather than inferred from which components moved, so the check on the
    complement can fail.
    """
    source, middle, far = chain("local_update", exploration=0.0)
    tag(source)
    hop(source, middle, (1, 2), watched=range(source.components.r_max))

    before = {c: block_of(far, c) for c in range(far.components.r_max)}

    ev = event(2, 3)
    manifest_far = far.build_manifest(ev)
    manifest_mid = middle.build_manifest(ev)
    view_mid = middle.consume_manifest(manifest_far, ev)
    view_far = far.consume_manifest(manifest_mid, ev)
    from federatedscope.contrib.decentralized.core.accounting import SessionBudget
    budget = SessionBudget(ev.endpoints, ev.credits)
    middle.plan_payload(ev, budget, view_mid)
    far.plan_payload(ev, budget, view_far)
    selected = set(far.policy.pending_for(far.policy.edge_of(ev)).selected)
    assert selected, "nothing was selected, so the complement is everything"
    unselected = [c for c in before if c not in selected]
    assert unselected, "everything was selected, so no component is left to stay unchanged"

    record = execute_pair_session(ev, {2: middle, 3: far},
                                  TransactionEngine(), ByteLedger())
    assert record["status"] == "committed"

    for component in unselected:
        for key, tensor in before[component].items():
            assert torch.equal(far.state.gather(component)[key], tensor), (
                f"component {component} was not selected yet slice {key} changed")
    changed = [c for c in selected
               if not all(torch.equal(far.state.gather(c)[k], v)
                          for k, v in before[c].items())]
    assert changed, f"the selected components {sorted(selected)} did not change"


# -- without relay eligibility or the second hop, the component does not arrive --------

def test_survivor_only_strands_the_tagged_component():
    """Relay eligibility disabled: the middle peer cannot carry what it lacks."""
    source, middle, far = chain("survivor_only", exploration=0.0)
    tag(source)
    middle_before = block_of(middle, TAGGED)
    far_before = block_of(far, TAGGED)

    hop(source, middle, (1, 2), watched=range(source.components.r_max))
    hop(middle, far, (2, 3), watched=range(middle.components.r_max))

    assert all(torch.equal(middle.state.gather(TAGGED)[k], v)
               for k, v in middle_before.items()), (
        "survivor_only updated a relay-only component; the ablation is not "
        "isolating relay eligibility")
    assert all(torch.equal(far.state.gather(TAGGED)[k], v)
               for k, v in far_before.items()), (
        "the tagged component reached the far peer without relay eligibility")


def test_without_the_second_hop_the_far_peer_never_sees_it():
    """Without the second session the far peer's copy stays unchanged."""
    source, middle, far = chain("local_update")
    tag(source)
    far_before = block_of(far, TAGGED)

    hop(source, middle, (1, 2), watched=range(source.components.r_max))
    # ...and no second hop.

    assert all(torch.equal(far.state.gather(TAGGED)[k], v)
               for k, v in far_before.items()), (
        "the far peer changed without ever being in a session")


def test_the_middle_peer_never_activates_what_it_relays():
    source, middle, far = chain("full_exchange")
    tag(source)
    hop(source, middle, (1, 2), watched=range(source.components.r_max))
    hop(middle, far, (2, 3), watched=range(middle.components.r_max))

    assert middle.state.support == MIDDLE_SUPPORT
    assert middle.bridge.schema.rank == len(MIDDLE_SUPPORT)
    for component in middle.state.relay_components:
        assert not middle.state.is_active(component)
        for tensor in middle.state.gather(component).values():
            assert not tensor.requires_grad


def test_every_transfer_conserves_its_bilateral_sum():
    """Checked per hop by ``hop``, not inferred from the endpoint state."""
    source, middle, far = chain("local_update")
    tag(source)
    watched = range(source.components.r_max)

    first = hop(source, middle, (1, 2), watched=watched)
    second = hop(middle, far, (2, 3), watched=watched)
    assert first["status"] == second["status"] == "committed", (
        "no transfer happened, so conservation was not exercised")
