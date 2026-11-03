"""Run labels for a method whose published graph is fixed (``native_graph="fixed"``).

The Dec-LoRA paper mixes synchronously over a fixed ring or a fixed Erdos-Renyi graph,
so a Dec-LoRA run on a fixed matrix is native: ``dec-native-ring`` / ``dec-native-er``.
The TAD-LoRA paper mixes over a time-varying graph, so the same fixed matrix gives
``tad-<topology>-static``, never a native label.
"""
from federatedscope.contrib.decentralized.core.interactions import (
    GRAPH_MIXING_DYNAMIC, METROPOLIS_RULE)
from federatedscope.contrib.decentralized.core.manifest import run_label


def _manifest(**fields):
    base = {"method": "dec_lora", "interaction_form": "graph_round"}
    base.update(fields)
    return base


def test_dec_lora_on_a_fixed_ring_is_dec_native_ring():
    manifest = _manifest(fixed_mixing_source="topology:ring:gamma=0.5")
    assert run_label(manifest, native_graph="fixed") == "dec-native-ring"


def test_dec_lora_on_an_er_matrix_is_dec_native_er_only_when_the_caller_says_er():
    manifest = _manifest(fixed_mixing_source="explicit:file:er_n10.json")
    assert run_label(manifest, native_graph="fixed", topology="er") == "dec-native-er"
    # An explicit matrix whose origin the caller does not name is a fixed Q, not ER.
    assert run_label(manifest, native_graph="fixed") == "dec-native-fixed-q"


def test_tad_on_a_fixed_ring_stays_tad_ring_static():
    manifest = dict(_manifest(fixed_mixing_source="topology:ring:gamma=0.5"), method="tad_lora")
    assert run_label(manifest) == "tad-ring-static"
    assert run_label(manifest, native_graph="dynamic") == "tad-ring-static"


def test_a_fixed_graph_method_under_a_dynamic_w_t_is_adapted_not_native():
    manifest = _manifest(graph_mixing={"mode": GRAPH_MIXING_DYNAMIC, "rule": METROPOLIS_RULE},
                         trace_generator={"name": "er_gnp"})
    assert run_label(manifest, native_graph="fixed") == "dec-dynamic-er-adapted"
    assert run_label(dict(manifest, method="tad_lora")) == "tad-native-dynamic-er"


def test_degree_one_execution_is_pairwise_adapted_whatever_the_native_graph():
    manifest = _manifest(interaction_form="pairwise_session")
    assert run_label(manifest, native_graph="fixed") == "dec-pairwise-adapted"


def test_an_unknown_native_graph_is_refused_by_name():
    import pytest

    with pytest.raises(ValueError, match="native_graph"):
        run_label(_manifest(), native_graph="sometimes")
