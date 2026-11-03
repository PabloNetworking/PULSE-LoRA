"""The runner: per-peer configuration, the final evaluation, and metric direction."""
import pytest

from federatedscope.contrib.decentralized.core.metrics import metric_lower_is_better
from federatedscope.contrib.decentralized.core.runner import DecentralizedRunner
from federatedscope.contrib.decentralized.tests.core.test_runner import build, toy_cfg
from federatedscope.contrib.decentralized.tests.core.toy_method import ToyFactory
from federatedscope.core.configs.config import CN


# --- per-peer configuration ----------------------------------------------------

def peer_configs():
    """Two peers with different learning rates."""
    client_cfgs = CN()
    client_cfgs.client_1 = CN()
    client_cfgs.client_1.train = CN()
    client_cfgs.client_1.train.optimizer = CN()
    client_cfgs.client_1.train.optimizer.lr = 0.001
    client_cfgs.client_2 = CN()
    client_cfgs.client_2.train = CN()
    client_cfgs.client_2.train.optimizer = CN()
    client_cfgs.client_2.train.optimizer.lr = 0.007
    return client_cfgs


def test_each_peer_receives_its_own_resolved_config():
    cfg = toy_cfg(client_num=2)
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyFactory(cfg),
                                 client_configs=peer_configs())

    first = runner._resolve_peer_config(1)
    second = runner._resolve_peer_config(2)

    assert first.train.optimizer.lr == 0.001
    assert second.train.optimizer.lr == 0.007
    assert first.train.optimizer.lr != second.train.optimizer.lr


def test_peers_are_built_with_their_own_config_not_the_whole_mapping():
    """Handing a peer the full client-config mapping would erase heterogeneity."""
    cfg = toy_cfg(client_num=2)
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyFactory(cfg),
                                 client_configs=peer_configs())
    runner._build_peers()

    lrs = {pid: peer.context.config.train.optimizer.lr
           for pid, peer in runner.peers.items()}
    assert lrs == {1: 0.001, 2: 0.007}

    for pid, peer in runner.peers.items():
        # the context carries this peer's own section, never the whole mapping
        assert not hasattr(peer.context.client_config, "client_1") or pid == 1
        assert peer.context.config is not runner.cfg


def test_resolution_is_a_noop_without_client_configs():
    cfg = toy_cfg(client_num=2)
    runner = DecentralizedRunner(data={}, config=cfg, factory=ToyFactory(cfg))
    assert runner._resolve_peer_config(1).train.optimizer.lr == cfg.train.optimizer.lr


# --- final evaluation ------------------------------------------------------------

def test_peers_are_evaluated_once_at_the_end_not_repeatedly():
    runner = build(frames=2, **{"decentralized.execution.eval_every_frames": 0})
    runner._build_peers()

    calls = {pid: 0 for pid in runner.peer_ids}
    for pid, peer in runner.peers.items():
        original = peer.algorithm.evaluate

        def counted(split="val", _pid=pid, _orig=original):
            calls[_pid] += 1
            return _orig(split)

        peer.algorithm.evaluate = counted

    runner._build_peers = lambda: None  # peers already built
    runner.run()

    assert set(calls.values()) == {1}, f"expected one final evaluation per peer, got {calls}"


# --- metric direction --------------------------------------------------------------

@pytest.mark.parametrize("key,lower", [
    ("val_acc", False), ("val_f1", False), ("val_mcc", False), ("val_pearson", False),
    ("val_loss", True), ("val_avg_loss", True), ("train_loss", True),
    ("perplexity", True),
])
def test_metric_direction_classification(key, lower):
    assert metric_lower_is_better(key) is lower


def test_worst_is_direction_aware():
    """For a loss the worst peer has the HIGHEST value, not the lowest."""
    runner = build(frames=1)
    runner._build_peers()
    for pid, peer in runner.peers.items():
        peer.algorithm.evaluate = (
            lambda split="val", _p=pid: {"val_acc": 0.1 * _p, "val_loss": 1.0 * _p})

    runner._build_peers = lambda: None
    results = runner.run()

    accuracy = results["distribution"]["val_acc"]
    loss = results["distribution"]["val_loss"]

    assert accuracy["direction"] == "higher_is_better"
    assert accuracy["worst"] == min(accuracy["values"])
    assert accuracy["best"] == max(accuracy["values"])

    assert loss["direction"] == "lower_is_better"
    assert loss["worst"] == max(loss["values"]), "worst loss must be the largest loss"
    assert loss["best"] == min(loss["values"])
