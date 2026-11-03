"""Run one scenario: build the data and the peers, run every frame of the trace, read the peers
out on the scenario's schedule and write the results.

A readout evaluates every peer's current model on the holdout (readout.py) right after the
frame it follows, inside the run. Iterating a data loader draws from torch's global generator,
so readouts inside a run require ``training_rng = "per_peer_episode"``, which reseeds torch
before every local episode: with it, a run with readouts trains exactly as the same run without
them.

Output directory::

    scenario.json, trace.json   the scenario as run
    sessions.jsonl              one row per pair session or graph-round edge, priced (meter.py)
    readouts.jsonl              one row per capture: ensemble and per-peer accuracy, bytes, time
    readout/labels.npy          holdout labels
    readout/peer<p>/frame<NNNN>.npy   peer p's holdout logits after NNNN completed frames
    peers/peer<p>.json          per peer: episodes (losses, steps), traffic, method metrics,
                                final learned-tensor digest
    summary.json                totals and the accuracy summary
    final/peer<p>.pt            with ``save_state``: the peer's final state, without the
                                trainer's optimizer and scheduler state
    runner/                     the runner's own event log, manifest and summary

Accuracy is that of the uniform probability ensemble: the mean of the peers' softmax
probabilities, then the arg-max. The ``window`` summary is the mean over the last five captures.
"""
import json
import os
import platform
import time
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from federatedscope.contrib.decentralized.simulation.glue_data import (
    GLUE_BUILDER,
    central_glue_data,
)
from federatedscope.contrib.decentralized.simulation.meter import WireMeter
from federatedscope.contrib.decentralized.simulation.readout import (
    capture_due,
    forward_logits,
    holdout_labels,
    holdout_loader,
)
from federatedscope.contrib.decentralized.simulation.scenario import (
    Scenario,
    ScenarioError,
    build_cfg,
    build_factory,
    client_configs,
    construct_peer,
    learned_tensor_digest,
    save_scenario,
)
from federatedscope.contrib.decentralized.simulation.training import (
    PER_PEER_EPISODE,
    build_data,
    plain,
)

#: Number of final captures the ``window`` accuracy is averaged over.
WINDOW = 5


def softmax(logits) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def probability_ensemble(per_peer: Mapping[Any, Any]) -> np.ndarray:
    """The uniform mean of every peer's softmax probabilities. Peers are summed in a fixed
    (string) order, so the floating-point result is reproducible."""
    arrays = [softmax(per_peer[p]) for p in sorted(per_peer, key=str)]
    return np.mean(np.stack(arrays), axis=0)


def accuracy(scores, labels) -> float:
    return float(np.mean(np.argmax(np.asarray(scores), axis=1) == np.asarray(labels)))


def build_scenario_data(scenario: Scenario, cfg) -> Dict[int, Any]:
    """Every peer's data: one ``get_data`` call for the GLUE builder, the scenario's own
    builder per peer otherwise, nothing for a run without data."""
    if not scenario.data:
        return {}
    if scenario.data == GLUE_BUILDER:
        return central_glue_data(scenario, cfg)
    return {int(p): build_data(scenario, int(p), build_cfg(scenario, int(p))) for p in scenario.peers}


def _write_json(path: str, value) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=1, sort_keys=True, default=str)
        handle.write("\n")


class ScenarioRun:
    """One run of a scenario. ``run()`` executes it and returns the summary.

    It is the runner's observer: readouts at ``on_peers_built`` (frame 0) and ``on_frame_end``,
    byte metering at ``on_event``.
    """

    def __init__(self, scenario: Scenario, out_dir: str, device: str = "cpu",
                 save_state: bool = False, log=print):
        if scenario.readout_schedule is not None and not scenario.data:
            raise ScenarioError("a readout schedule needs data with a holdout")
        if scenario.readout_schedule is not None and scenario.training_rng != PER_PEER_EPISODE:
            raise ScenarioError(f"readouts inside a run require training_rng {PER_PEER_EPISODE!r}; "
                                f"this scenario states {scenario.training_rng!r}")
        self.scenario = scenario
        self.out_dir = out_dir
        self.device = str(device)
        self.save_state = bool(save_state)
        self.log = log
        self.frames = len(scenario.trace["frames"])
        self.episodes: Dict[int, List[dict]] = {int(p): [] for p in scenario.peers}
        self.compute_s: Dict[int, float] = {int(p): 0.0 for p in scenario.peers}
        self.captures: List[dict] = []
        self.readout_s = 0.0
        self.meter: Optional[WireMeter] = None
        self.loaders: Dict[int, Any] = {}
        self.labels = None
        self.started = None

    # -- the runner's observer -------------------------------------------------------------

    def on_peers_built(self, runner) -> None:
        self._capture(runner, 0)

    def on_event(self, runner, event, record) -> None:
        self.meter.on_event(runner, event, record)

    def on_frame_end(self, runner, frame_index: int) -> None:
        self._capture(runner, int(frame_index) + 1)

    # -- readouts -------------------------------------------------------------------------

    def _capture(self, runner, frames_completed: int) -> None:
        schedule = self.scenario.readout_schedule
        if schedule is None or not capture_due(schedule, frames_completed, self.frames):
            return
        began = time.perf_counter()
        logits = {}
        for pid, peer in sorted(runner.peers.items()):
            logits[int(pid)] = forward_logits(peer.algorithm, self.loaders[int(pid)], self.device)
            directory = os.path.join(self.out_dir, "readout", f"peer{int(pid)}")
            os.makedirs(directory, exist_ok=True)
            np.save(os.path.join(directory, f"frame{frames_completed:04d}.npy"), logits[int(pid)])
        self.readout_s += time.perf_counter() - began
        row = {
            "frames_completed": frames_completed,
            "accuracy": accuracy(probability_ensemble(logits), self.labels),
            "per_peer_accuracy": {str(p): accuracy(v, self.labels) for p, v in sorted(logits.items())},
            "upload_bytes_total": int(sum(self.meter.sent.values())),
            "per_peer": {str(pid): {"optimizer_steps": int(peer.local_steps),
                                    "local_episodes": int(peer.local_episodes),
                                    "sessions": int(peer.sessions),
                                    "upload_bytes": int(self.meter.sent[int(pid)]),
                                    "download_bytes": int(self.meter.received[int(pid)]),
                                    "compute_s": self.compute_s[int(pid)]}
                         for pid, peer in sorted(runner.peers.items())},
            "elapsed_s": time.perf_counter() - self.started - self.readout_s,
        }
        self.captures.append(row)
        with open(os.path.join(self.out_dir, "readouts.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        self.log(f"frame {frames_completed}/{self.frames}: ensemble accuracy {row['accuracy']:.4f}, "
                 f"{row['upload_bytes_total']} bytes uploaded")

    # -- peers ------------------------------------------------------------------------------

    def _factory(self, cfg):
        factory = build_factory(self.scenario, cfg)
        create = factory.create_peer

        def create_peer(context):
            algorithm = construct_peer(self.scenario, context.peer_id, context, create,
                                       device=self.device)
            peer_id, episode = int(context.peer_id), algorithm.local_episode

            def recorded_episode(steps):
                if self.device.startswith("cuda"):
                    import torch
                    torch.cuda.synchronize()
                began = time.perf_counter()
                metrics = episode(steps)
                if self.device.startswith("cuda"):
                    import torch
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - began
                self.compute_s[peer_id] += seconds
                self.episodes[peer_id].append(dict(plain(metrics), seconds=seconds))
                return metrics

            algorithm.local_episode = recorded_episode
            return algorithm

        factory.create_peer = create_peer
        return factory

    # -- the run ------------------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        from federatedscope.contrib.decentralized.core.runner import DecentralizedRunner

        os.makedirs(self.out_dir, exist_ok=False)
        save_scenario(self.scenario, os.path.join(self.out_dir, "scenario.json"))
        trace_path = os.path.join(self.out_dir, "trace.json")
        _write_json(trace_path, self.scenario.trace)
        cfg = build_cfg(self.scenario)
        cfg.merge_from_list([
            "decentralized.opportunities.source", "file",
            "decentralized.opportunities.path", trace_path,
            "decentralized.execution.total_frames", self.frames,
            "decentralized.execution.eval_every_frames", 0,
            "decentralized.execution.final_evaluation", False,
            "outdir", os.path.join(self.out_dir, "runner"),
        ])
        self.started = time.perf_counter()
        data = build_scenario_data(self.scenario, cfg)
        if self.scenario.readout_schedule is not None:
            self.loaders = {int(p): holdout_loader(data[int(p)]) for p in self.scenario.peers}
            self.labels = holdout_labels(data[int(self.scenario.peers[0])])
            os.makedirs(os.path.join(self.out_dir, "readout"), exist_ok=True)
            np.save(os.path.join(self.out_dir, "readout", "labels.npy"), self.labels)
        runner = DecentralizedRunner(data=data, config=cfg, factory=self._factory(cfg),
                                     client_configs=client_configs(self.scenario), observer=self)
        self.meter = WireMeter(self.scenario.peers, runner.wire_cost, runner.graph_wire_cost)
        if self.scenario.data_partition is not None:
            # Recorded in the runner's manifest.
            runner.data_partition = dict(self.scenario.data_partition)
        runner.run()
        wall_s = time.perf_counter() - self.started

        with open(os.path.join(self.out_dir, "sessions.jsonl"), "w", encoding="utf-8") as handle:
            for row in self.meter.rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        os.makedirs(os.path.join(self.out_dir, "peers"), exist_ok=True)
        for pid, peer in sorted(runner.peers.items()):
            metrics = getattr(peer.algorithm, "method_metrics", None)
            _write_json(os.path.join(self.out_dir, "peers", f"peer{int(pid)}.json"), {
                "peer_id": int(pid),
                "episodes": self.episodes[int(pid)],
                "traffic": self.meter.traffic(pid),
                "compute_s": self.compute_s[int(pid)],
                "method_metrics": plain(metrics()) if callable(metrics) else None,
                "final_learned_tensor_digest": learned_tensor_digest(peer.algorithm),
            })
            if self.save_state:
                self._save_state(pid, peer.algorithm)
        summary = self._summary(runner, wall_s)
        _write_json(os.path.join(self.out_dir, "summary.json"), summary)
        return summary

    def _save_state(self, peer_id, algorithm) -> None:
        import torch

        state = {key: value for key, value in algorithm.state_dict().items() if key != "trainer"}
        directory = os.path.join(self.out_dir, "final")
        os.makedirs(directory, exist_ok=True)
        torch.save(state, os.path.join(directory, f"peer{int(peer_id)}.pt"))

    def _summary(self, runner, wall_s: float) -> Dict[str, Any]:
        import peft
        import torch
        import transformers

        rows = self.meter.rows
        summary = {
            "method": self.scenario.method,
            "scenario_digest": self.scenario.digest,
            "trace_id": (self.scenario.trace.get("generator") or {}).get("trace_id"),
            "peers": len(self.scenario.peers),
            "frames": self.frames,
            "exchanges": len(rows),
            "committed": sum(1 for r in rows if r.get("status") == "committed"),
            "upload_bytes_total": int(sum(self.meter.sent.values())),
            "device": self.device,
            "wall_s": wall_s,
            "readout_s": self.readout_s,
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "transformers": transformers.__version__, "peft": peft.__version__,
                         "numpy": np.__version__},
        }
        if self.captures:
            window = self.captures[-WINDOW:]
            values = [c["accuracy"] for c in window]
            summary["accuracy"] = {
                "final": self.captures[-1]["accuracy"],
                "window_frames": [c["frames_completed"] for c in window],
                "window_mean": float(np.mean(values)),
                "window_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
        return summary


def run_scenario(scenario: Scenario, out_dir: str, device: str = "cpu", save_state: bool = False,
                 log=print) -> Dict[str, Any]:
    """Run ``scenario`` and write its results to ``out_dir`` (which must not exist)."""
    return ScenarioRun(scenario, out_dir, device=device, save_state=save_state, log=log).run()
