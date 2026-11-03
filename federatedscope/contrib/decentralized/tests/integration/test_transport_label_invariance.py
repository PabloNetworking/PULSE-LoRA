"""A contact's transport label (``direct`` or ``infrastructure``) does not change learning.

A contact is described by its end-to-end conditions. Two traces that differ only
in transport label give identical adapters and byte totals, and event logs that
differ only in the ``transport`` field, while the label still enters the trace
digest because it is part of the offered trace.
"""
import json

import torch

from federatedscope.contrib.decentralized.core.runner import DecentralizedRunner
from federatedscope.contrib.decentralized.tests.methods.test_dec_lora import (
    ToyLoRAFactory,
    dec_cfg,
)


def run_with_transport(transport, tmp_path=None):
    cfg = dec_cfg(frames=6, auxiliary="local",
                  **{"decentralized.opportunities.transport": transport})
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyLoRAFactory(cfg))
    results = runner.run()
    adapters = {pid: peer.algorithm.bridge.read() for pid, peer in runner.peers.items()}
    return runner, results, adapters


def test_learning_outcome_is_identical_across_transport_labels():
    _, direct_results, direct_adapters = run_with_transport("direct")
    _, infra_results, infra_adapters = run_with_transport("infrastructure")

    assert set(direct_adapters) == set(infra_adapters)
    for pid in direct_adapters:
        for key in direct_adapters[pid]:
            assert torch.equal(direct_adapters[pid][key], infra_adapters[pid][key]), \
                f"peer {pid} diverged between transport labels"

    assert direct_results["distribution"] == infra_results["distribution"]


def test_byte_accounting_is_identical_across_transport_labels():
    direct, _, _ = run_with_transport("direct")
    infra, _, _ = run_with_transport("infrastructure")
    assert direct.ledger.total() == infra.ledger.total()
    assert direct.ledger.total("control") == infra.ledger.total("control")
    assert direct.ledger.total("payload") == infra.ledger.total("payload")


def test_the_label_still_changes_the_trace_digest():
    """The label is recorded in the trace digest, but learning does not depend on it."""
    direct, _, _ = run_with_transport("direct")
    infra, _, _ = run_with_transport("infrastructure")
    assert direct.trace.trace_id != infra.trace.trace_id


def test_event_logs_differ_only_in_the_transport_field():
    direct, _, _ = run_with_transport("direct")
    infra, _, _ = run_with_transport("infrastructure")

    def scrub(events):
        return json.dumps([{k: v for k, v in e.items() if k != "transport"}
                           for e in events], sort_keys=True)

    assert scrub(direct.sink.events) == scrub(infra.sink.events)
