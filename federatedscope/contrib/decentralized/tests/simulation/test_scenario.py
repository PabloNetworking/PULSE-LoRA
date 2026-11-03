"""The scenario document (simulation/scenario.py): its format and digest, the configuration it
derives, the peers it builds and the state digests."""
import json

import pytest
import torch

from federatedscope.contrib.decentralized.core.accounting import ByteLedger
from federatedscope.contrib.decentralized.core.interactions import compile_pair_sessions
from federatedscope.contrib.decentralized.core.interfaces import INTERACTION_PAIRWISE
from federatedscope.contrib.decentralized.core.pair_session import execute_pair_session
from federatedscope.contrib.decentralized.core.transaction import TransactionEngine
from federatedscope.contrib.decentralized.simulation.scenario import (
    DOCUMENT_KEYS,
    SCENARIO_VERSION,
    Scenario,
    ScenarioError,
    build_cfg,
    build_factory,
    build_peer,
    build_trace,
    client_configs,
    exact_state_digest,
    exact_state_paths,
    learned_tensor_digest,
    learning_state_digest,
    load_scenario,
    save_scenario,
    scenario_from_dict,
)
from federatedscope.contrib.decentralized.simulation.training import PER_PEER_EPISODE
from federatedscope.contrib.decentralized.tests.simulation.training_fixtures import (
    ALPHA,
    CONSTRUCTION,
    DETERMINISTIC,
    LOCAL_STEPS,
    PARTITION,
    RANK,
    TARGETS,
    TinyClassifier,
    training_scenario,
)

MODULE = "federatedscope.contrib.decentralized.tests.simulation.test_scenario"
UNSEEDED = f"{MODULE}.unseeded_classifier"

TRACE = {"trace_schema_version": 1, "time_unit": "ms", "duration_unit": "ms",
         "goodput_unit": "bps", "peers": [1, 2],
         "frames": [{"index": 0, "time": 0, "edges": [
             {"endpoints": [1, 2], "duration": 1000, "goodput": 10_000_000}]}]}

OVERRIDES = {"decentralized.execution.interaction": INTERACTION_PAIRWISE,
             "decentralized.auxiliary_state.policy": "local",
             "decentralized.methods.dec_lora.rank": RANK}


def scenario(**kwargs):
    """Two Dec-LoRA peers without data, on the tiny deterministic classifier."""
    base = dict(method="dec_lora", peers=(1, 2), trace=TRACE, adapter=DETERMINISTIC,
                adapter_rank=RANK, cfg_overrides=dict(OVERRIDES), construction=dict(CONSTRUCTION))
    base.update(kwargs)
    return Scenario(**base)


def full_scenario():
    """A scenario that states every optional field."""
    return scenario(seed=12, peer_seed_base=100,
                    peer_overrides={2: {"train.optimizer.lr": 0.002}},
                    data=PARTITION, training_rng=PER_PEER_EPISODE,
                    data_partition={"splitter": "lda", "alpha": 0.3, "seed": 4},
                    readout_schedule={"every_frames": 2, "initial": True,
                                      "holdout": {"source": "train", "count": 12, "seed": 5}})


def unseeded_classifier(rank=RANK, seed=0):
    """The tiny classifier drawn from torch's global generator as it stands. Unlike the
    fixtures' builders it ignores ``seed``, so only the reseed before a peer is built fixes it."""
    from peft import LoraConfig, get_peft_model

    return get_peft_model(TinyClassifier(), LoraConfig(
        r=rank, lora_alpha=ALPHA, lora_dropout=0.0, target_modules=list(TARGETS),
        modules_to_save=["head"], bias="none"))


def model_tensors(algorithm):
    """Every parameter and buffer of a peer's model, frozen backbone weights included."""
    return {name: tensor.detach().clone() for name, tensor in algorithm.model.state_dict().items()}


def same_tensors(left, right):
    return left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)


# -- the document ---------------------------------------------------------------------------

@pytest.mark.parametrize("build", [scenario, full_scenario], ids=["minimal", "full"])
def test_a_scenario_round_trips_through_a_file(build, tmp_path):
    original = build()
    path = save_scenario(original, str(tmp_path / "s.json"))
    assert load_scenario(path) == original
    assert load_scenario(path).digest == original.digest


def test_the_document_writes_every_field():
    document = scenario().to_dict()
    assert sorted(document) == sorted(DOCUMENT_KEYS)
    assert document["scenario_version"] == SCENARIO_VERSION
    for key in ("data", "training_rng", "data_partition", "readout_schedule"):
        assert document[key] is None, key
    assert document["peer_overrides"] == {}
    assert document["construction"] == CONSTRUCTION


#: One change per document field; every field but the version is in it.
CHANGES = {
    "method": {"method": "pulse_lora"},
    "peers": {"peers": (1, 3)},
    "trace": {"trace": dict(TRACE, frames=[dict(TRACE["frames"][0], time=5)])},
    "seed": {"seed": 12},
    "peer_seed_base": {"peer_seed_base": 5},
    "adapter": {"adapter": UNSEEDED},
    "adapter_rank": {"adapter_rank": 8},
    "cfg_overrides": {"cfg_overrides": dict(OVERRIDES, **{"decentralized.seed": 99})},
    "peer_overrides": {"peer_overrides": {2: {"train.optimizer.lr": 0.002}}},
    "data": {"data": PARTITION},
    "training_rng": {"training_rng": PER_PEER_EPISODE},
    "data_partition": {"data_partition": {"splitter": "lda", "alpha": 0.3, "seed": 4}},
    "readout_schedule": {"readout_schedule": {"every_frames": 2, "initial": True, "holdout": {
        "source": "train", "count": 12, "seed": 5}}},
    "construction": {"construction": dict(CONSTRUCTION, model_init_seed=2)},
}


def test_the_changes_cover_every_field_but_the_version():
    assert set(CHANGES) == set(DOCUMENT_KEYS) - {"scenario_version"}


@pytest.mark.parametrize("field", sorted(CHANGES))
def test_the_digest_covers_every_field(field):
    # A partition needs data, so it is compared against a base that has data.
    base = {"data": PARTITION} if field == "data_partition" else {}
    changed = scenario(**base, **CHANGES[field])
    assert changed.digest != scenario(**base).digest
    assert scenario_from_dict(changed.to_dict()).digest == changed.digest


def test_the_digest_ignores_how_the_document_is_spelled():
    """Key order and whitespace are not part of a run."""
    left = scenario()
    document = left.to_dict()
    respelled = json.loads(json.dumps(dict(reversed(list(document.items()))), indent=3))
    assert scenario_from_dict(respelled).digest == left.digest


@pytest.mark.parametrize("broken, match", [
    ({"scenario_version": 99}, "schema version"),
    ({"scenario_version": None}, "schema version"),
    ({"peers": []}, "positive integers"),
    ({"peers": [0, 1]}, "positive integers"),
    ({"peers": [1, True]}, "positive integers"),
    ({"peers": "12"}, "positive integers"),
    ({"peers": [1, 1]}, "repeat"),
    ({"seed ": 12}, "unknown scenario keys"),
    ({"data": ""}, "dotted builder path"),
    ({"data": 7}, "dotted builder path"),
    ({"training_rng": "global"}, "training_rng"),
    ({"construction": None}, "must state its construction"),
    ({"construction": {"mode": "common_canonical_v1"}}, "construction"),
    ({"construction": {"mode": "peft_default", "model_init_seed": 1}}, "construction"),
    ({"construction": {"mode": "common_canonical_v1", "model_init_seed": -1}}, "construction"),
    ({"construction": {"mode": "common_canonical_v1", "model_init_seed": True}}, "construction"),
    ({"construction": dict(CONSTRUCTION, seed=1)}, "construction"),
    ({"data_partition": "lda"}, "data_partition"),
    ({"readout_schedule": [2]}, "readout_schedule"),
])
def test_a_malformed_document_is_refused(broken, match):
    payload = scenario().to_dict()
    payload.update(broken)
    with pytest.raises(ScenarioError, match=match):
        scenario_from_dict(payload)


@pytest.mark.parametrize("key", ["method", "peers", "trace", "construction"])
def test_a_document_missing_a_required_key_is_refused(key):
    payload = {k: v for k, v in scenario().to_dict().items() if k != key}
    with pytest.raises(ScenarioError, match="missing"):
        scenario_from_dict(payload)


@pytest.mark.parametrize("field,value", [
    ("seed", 12.7), ("seed", "12"), ("seed", True), ("peer_seed_base", 1.5), ("adapter_rank", "4"),
])
def test_an_integer_field_that_is_not_an_integer_is_refused(field, value):
    payload = scenario().to_dict()
    payload[field] = value
    with pytest.raises(ScenarioError, match=field):
        scenario_from_dict(payload)


def test_a_partition_without_data_is_refused():
    with pytest.raises(ScenarioError, match="data_partition"):
        scenario(data_partition={"splitter": "lda", "alpha": 0.3, "seed": 4})


@pytest.mark.parametrize("splitter", ["lda", "meta"])
def test_a_split_stated_outside_data_partition_is_refused(splitter):
    with pytest.raises(ScenarioError, match="data.splitter"):
        scenario(cfg_overrides=dict(OVERRIDES, **{"data.splitter": splitter}))


@pytest.mark.parametrize("splitter", ["", "iid"])
def test_the_iid_splitter_may_be_stated(splitter):
    scenario(cfg_overrides=dict(OVERRIDES, **{"data.splitter": splitter}))


def test_a_scenario_built_in_python_must_state_its_construction():
    with pytest.raises(ScenarioError, match="must state its construction"):
        Scenario(method="dec_lora", peers=(1, 2), trace=TRACE)


def test_a_method_that_is_not_registered_is_refused():
    with pytest.raises(ScenarioError, match="not a registered"):
        build_factory(scenario_from_dict(dict(scenario().to_dict(), method="fedavg")))


def test_a_document_that_is_not_a_mapping_is_refused():
    for bad in ("not a scenario", 7, [1, 2], None):
        with pytest.raises(ScenarioError):
            scenario_from_dict(bad)


# -- the configuration ------------------------------------------------------------------------

def test_a_peer_overlay_reaches_that_peer_only():
    setup = full_scenario()
    assert float(build_cfg(setup, 2).train.optimizer.lr) == 0.002
    run_lr = float(build_cfg(setup).train.optimizer.lr)
    assert run_lr != 0.002 and float(build_cfg(setup, 1).train.optimizer.lr) == run_lr
    sections = client_configs(setup)
    assert set(sections) == {"client_2"}
    assert float(sections["client_2"].train.optimizer.lr) == 0.002
    assert client_configs(scenario()) is None


def test_the_configuration_does_not_alias_the_document():
    """A list in the document is copied into the configuration, not shared with it."""
    splits = [0.9, 0.1]
    setup = scenario(cfg_overrides=dict(OVERRIDES, **{"data.splits": splits}))
    cfg = build_cfg(setup)
    assert list(cfg.data.splits) == splits
    cfg.data.splits.append(0.0)
    assert setup.cfg_overrides["data.splits"] == [0.9, 0.1]
    assert list(build_cfg(setup).data.splits) == [0.9, 0.1]


def test_the_trace_id_is_a_function_of_the_trace_alone():
    assert build_trace(scenario()).trace_id == build_trace(scenario(seed=12)).trace_id
    assert build_trace(scenario()).trace_id != build_trace(scenario(**CHANGES["trace"])).trace_id


# -- the peers --------------------------------------------------------------------------------

def test_a_peer_is_the_same_whatever_was_built_before_it():
    """The torch reseed before a peer is built fixes everything drawn while it is built."""
    torch.manual_seed(1)
    first = unseeded_classifier()
    torch.manual_seed(2)
    assert not same_tensors(first.state_dict(), unseeded_classifier().state_dict()), \
        "the adapter alone must depend on the global generator for this test to say anything"

    setup = scenario(adapter=UNSEEDED)
    cfg = build_cfg(setup)
    factory = build_factory(setup, cfg)
    alone = build_peer(setup, 2, cfg, factory)
    build_peer(setup, 1, cfg, factory)
    torch.rand(1000)
    after_another = build_peer(setup, 2, cfg, factory)
    assert exact_state_digest(after_another) == exact_state_digest(alone)
    assert same_tensors(model_tensors(after_another), model_tensors(alone))


def test_two_peers_start_from_the_common_construction_and_differ_elsewhere():
    """Every LoRA factor starts at the run's canonical value (B zero) on every peer; what the
    peer seed draws - here the classifier head - differs."""
    setup = scenario(adapter=UNSEEDED)
    cfg = build_cfg(setup)
    factory = build_factory(setup, cfg)
    one, two = (build_peer(setup, p, cfg, factory).state_dict()["adapter"] for p in (1, 2))
    assert one.keys() == two.keys()
    factors = [key for key in one if ".lora_A." in key or ".lora_B." in key]
    heads = [key for key in one if key not in factors]
    assert len(factors) == 2 * len(TARGETS) and heads
    assert all(torch.equal(one[key], two[key]) for key in factors)
    assert all(not torch.any(one[key]) for key in factors if ".lora_B." in key)
    assert all(not torch.equal(one[key], two[key]) for key in heads)


@pytest.mark.parametrize("rule, independent", [(PER_PEER_EPISODE, True), (None, False)],
                         ids=["per_peer_episode", "global"])
def test_under_the_per_peer_episode_rule_an_episode_ignores_the_global_generator(rule, independent):
    """With dropout and a shuffled loader an episode draws from torch's global generator. The
    rule reseeds it before every episode from (run seed, peer, episode), so what was drawn
    before the episode does not matter."""
    setup = training_scenario("dec_lora", stochastic=True, training_rng=rule)
    learned = []
    for draws in (1, 1000):
        peer = build_peer(setup, 2)
        torch.rand(draws)
        peer.local_episode(LOCAL_STEPS)
        learned.append(learned_tensor_digest(peer))
    assert (learned[0] == learned[1]) is independent


def test_the_per_peer_episode_rule_refuses_a_peer_without_a_trainer():
    with pytest.raises(ScenarioError, match="has no trainer"):
        build_peer(scenario(training_rng=PER_PEER_EPISODE), 1)


def test_a_peer_outside_the_scenario_is_refused():
    with pytest.raises(ScenarioError, match="not part of this scenario"):
        build_peer(scenario(), 7)


def test_data_for_a_dataset_free_scenario_is_refused():
    with pytest.raises(ScenarioError, match="dataset-free"):
        build_peer(scenario(), 1, data=object())


# -- the state digests ------------------------------------------------------------------------

DIGESTS = (learning_state_digest, learned_tensor_digest, exact_state_digest)


@pytest.mark.parametrize("policy, learned_moves", [("local", False), ("pairwise_full", True)])
def test_a_committed_session_moves_the_digests_that_cover_what_it_changed(policy, learned_moves):
    """Both peers start with the same canonical factors. Under the ``local`` policy only those
    are mixed, so no learned value moves while the committed state version does; under
    ``pairwise_full`` the two heads are averaged as well."""
    setup = scenario(cfg_overrides=dict(OVERRIDES, **{"decentralized.auxiliary_state.policy": policy}))
    cfg = build_cfg(setup)
    factory = build_factory(setup, cfg)
    peers = {p: build_peer(setup, p, cfg, factory) for p in setup.peers}
    untouched = build_peer(setup, 1, cfg, factory)
    before = [digest(peers[1]) for digest in DIGESTS]
    assert [digest(untouched) for digest in DIGESTS] == before

    [event] = compile_pair_sessions(build_trace(setup).frame(0))
    record = execute_pair_session(event, peers, TransactionEngine(), ByteLedger())
    assert record["status"] == "committed", record.get("reason")

    learning, learned, exact = (digest(peers[1]) for digest in DIGESTS)
    assert learning != before[0]
    assert (learned != before[1]) is learned_moves
    assert exact != before[2]
    assert [digest(untouched) for digest in DIGESTS] == before
    changed = {path for path, value in exact_state_paths(peers[1]).items()
               if exact_state_paths(untouched).get(path) != value}
    assert "/'version'" in changed


def test_the_exact_digest_refuses_a_leaf_it_cannot_digest_exactly():
    class Holder:
        def __init__(self, state):
            self.state = state

        def state_dict(self):
            return self.state

    assert exact_state_paths(Holder({"a": [1, 2.5], "b": {"c": None}})) == exact_state_paths(
        Holder({"b": {"c": None}, "a": [1, 2.5]}))
    assert exact_state_digest(Holder({"a": []})) != exact_state_digest(Holder({"a": ()}))
    with pytest.raises(TypeError, match="state path /'model'"):
        exact_state_paths(Holder({"model": object()}))
    with pytest.raises(TypeError, match="cannot be digested exactly"):
        exact_state_paths(Holder({"ids": {(1, 2)}}))
