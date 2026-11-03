"""``DecentralizedRunner``: runs one decentralized method over a contact trace.

The runner builds one generic peer per client id from the method's factory and, frame by
frame, schedules every peer's local episode and the frame's pair sessions or graph round.
It dispatches on the factory's declared interaction form, never on a method name, and
constructs no server and no client. Its own random draws come from named ``RngService``
streams, not from the process-global generators.
"""
import json
import logging
import os
from typing import Any, Dict, Optional

from federatedscope.contrib.decentralized.core.accounting import ByteLedger
from federatedscope.contrib.decentralized.core.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from federatedscope.contrib.decentralized.core.coordinator import (
    DecentralizedSimulationCoordinator,
)
from federatedscope.contrib.decentralized.core.graph_mixing import execute_graph_round
from federatedscope.contrib.decentralized.core.wire_cost import (
    DEFAULT_ENVELOPE_BYTES,
    resolve_graph_wire_cost,
    resolve_wire_cost,
)
from federatedscope.contrib.decentralized.core.interactions import (
    GRAPH_MIXING_DYNAMIC,
    compile_dynamic_graph_round,
    compile_fixed_graph_round,
    compile_graph_round,
    compile_pair_sessions,
)
from federatedscope.contrib.decentralized.core.interfaces import (
    INTERACTION_GRAPH,
    INTERACTION_PAIRWISE,
    PeerContext,
)
from federatedscope.contrib.decentralized.core.manifest import (
    build_offered_frames, write_manifest)
from federatedscope.contrib.decentralized.core.metrics import (
    MetricSink,
    metric_lower_is_better,
)
from federatedscope.contrib.decentralized.core.pair_session import execute_pair_session
from federatedscope.contrib.decentralized.core.peer import Peer
from federatedscope.contrib.decentralized.core.rng import (
    STREAM_LOCAL_COMPUTE,
    STREAM_OPPORTUNITIES,
    RngService,
)
from federatedscope.contrib.decentralized.core.traces import (
    generate_fixed_graph_trace,
    generate_matching_trace,
    load_trace_file,
    profile_requires_matching,
    validate_matching_profile,
)
from federatedscope.contrib.decentralized.core.transaction import (
    FailureInjector,
    TransactionEngine,
)
from federatedscope.contrib.decentralized.core.types import EventKind, LocalComputeEvent

logger = logging.getLogger(__name__)


def _framework_rng_state() -> dict:
    import random as _random

    import numpy as _np
    import torch as _torch

    state = {
        "torch": _torch.get_rng_state(),
        "python": _random.getstate(),
        "numpy": _np.random.get_state(),
    }
    if _torch.cuda.is_available():
        state["torch_cuda"] = _torch.cuda.get_rng_state_all()
    return state


def _restore_framework_rng(state: dict) -> None:
    import random as _random

    import numpy as _np
    import torch as _torch

    _torch.set_rng_state(state["torch"])
    _random.setstate(state["python"])
    _np.random.set_state(state["numpy"])
    if state.get("torch_cuda") is not None and _torch.cuda.is_available():
        _torch.cuda.set_rng_state_all(state["torch_cuda"])


class DecentralizedRunner:
    """Runs one decentralized method over a contact trace, frame by frame.

    ``observer`` is optional. The runner calls whichever of these it defines:

      - ``on_peers_built(runner)``: once, after every peer is built and before the first frame;
      - ``on_event(runner, event, record)``: after each pair session or graph round, with the
        record the runner logged;
      - ``on_frame_end(runner, frame_index)``: after the frame's events (every event dated
        before the next frame) and the periodic evaluation, if one is due.

    None of them is called for a local episode, and none may change the run's state.
    ``simulation/driver.py`` uses the observer for readouts and wire pricing.
    """

    def __init__(self, data=None, config=None, client_configs=None, factory=None,
                 observer=None):
        if factory is None:
            raise ValueError("DecentralizedRunner requires a method factory")
        self.cfg = config
        self.observer = observer
        # The wire-cost model of every pair session of this run, resolved by name: a run
        # asking for a model that is not registered is refused rather than silently priced
        # by the logical model.
        # The envelope in force: 64 MiB unless the run sets a larger one.
        envelope = int(getattr(config.decentralized.transport, "envelope_bytes",
                               DEFAULT_ENVELOPE_BYTES))
        self.wire_cost = resolve_wire_cost(
            getattr(config.decentralized.transport, "wire_cost", "logical"),
            envelope_bytes=envelope)
        # The same denomination for every graph round of this run, resolved in the
        # graph-round registry: the identity under "logical", otherwise the graph-round
        # model registered under the same name (refused by name if there is none).
        self.graph_wire_cost = resolve_graph_wire_cost(
            getattr(config.decentralized.transport, "wire_cost", "logical"),
            envelope_bytes=envelope)
        self.data = data or {}
        self.client_cfgs = client_configs
        self.factory = factory

        dec = config.decentralized
        self.peer_ids = tuple(range(1, int(config.federate.client_num) + 1))
        seed = int(dec.seed) if int(dec.seed) else int(getattr(config, "seed", 0))
        self.rng = RngService(seed)
        self.ledger = ByteLedger()
        self.engine = TransactionEngine()
        self.injector = FailureInjector(
            dec.transaction.failure_injection_stage,
            dec.transaction.failure_injection_every,
        )
        # A graph method may declare a fixed mixing matrix. Asked through a hook so
        # the runner never learns which method it is.
        self.fixed_mixing = None
        declare = getattr(factory, "fixed_mixing_matrix", None)
        if callable(declare):
            self.fixed_mixing = declare(self.peer_ids)
        # How a graph round's W_t is built when the method declares no fixed matrix: the
        # matching's pair block (the default) or, under dynamic_metropolis, per-frame
        # Metropolis weights at any degree. A fixed Q and a dynamic W_t are different
        # mixing operators, so a method that declares a matrix is refused under
        # dynamic_metropolis rather than one of them silently taking precedence.
        self.graph_mixing = str(getattr(dec.execution, "graph_mixing", "pair_block"))
        if self.graph_mixing == GRAPH_MIXING_DYNAMIC and self.fixed_mixing is not None:
            raise ValueError(
                f"decentralized.execution.graph_mixing is {GRAPH_MIXING_DYNAMIC!r} but "
                f"{factory.name} declares a fixed mixing matrix ({self.fixed_mixing[3]}); "
                f"a fixed Q and a per-frame renormalised dynamic W_t are two different "
                f"experiments - choose one")
        self.trace = self._build_trace()
        # What the run manifest records about the trace, taken once here.
        self._trace_id = self.trace.trace_id
        self._trace_frame_count = len(self.trace.frames)
        self._trace_schema_version = self.trace.schema_version
        self._offered_frames = build_offered_frames(self.trace)
        # The trace document's generator block as a plain dict, or None if it had none.
        provenance = getattr(self.trace, "provenance", None)
        self._trace_provenance = dict(provenance) if provenance else None
        self.sink = MetricSink()
        self.coordinator = DecentralizedSimulationCoordinator(
            trace=self.trace, rng=self.rng, peer_ids=self.peer_ids, sink=self.sink,
            topology_window=int(dec.topology.history_window),
        )
        self.peers: Dict[int, Peer] = {}

        # Log where federatedscope was imported from, so a run that picked up another
        # installed copy of the package shows it in its log.
        import federatedscope

        logger.info("decentralized runtime importing federatedscope from %s",
                    federatedscope.__file__)

    def _compute_latency(self, peer_id: int) -> int:
        """How long this peer's local episode takes, in microseconds.

        ``compute_latency_us`` plus a jitter drawn uniformly from
        ``[0, compute_latency_jitter_us]``: a configured duration, independent of the
        simulation clock, so shifting every timestamp of a trace by a constant changes
        nothing. The jitter comes from the peer's own stream, so it is deterministic and
        adding a peer does not change another peer's draws.
        """
        execution = self.cfg.decentralized.execution
        base = int(execution.compute_latency_us)
        jitter = int(execution.compute_latency_jitter_us)
        if jitter <= 0:
            return base
        stream = self.rng.stream(f"{STREAM_LOCAL_COMPUTE}:peer{peer_id}")
        return base + stream.randrange(0, jitter + 1)

    def effective_gamma(self) -> float:
        """The mixing coefficient in force, ``decentralized.comparison.gamma``. The
        configuration check refuses an active method's own ``gamma`` that differs from it."""
        return float(self.cfg.decentralized.comparison.gamma)

    # -- setup ---------------------------------------------------------------

    def _build_trace(self):
        dec = self.cfg.decentralized
        opportunities = dec.opportunities
        if opportunities.source == "file":
            trace = load_trace_file(opportunities.path)
        elif self.fixed_mixing is not None:
            # A fixed connected graph, offered in full every round - not a matching.
            _matrix, edges, digest, source = self.fixed_mixing
            reverse = opportunities.goodput_kbps_reverse
            trace = generate_fixed_graph_trace(
                peer_ids=self.peer_ids, edges=edges,
                num_frames=int(dec.execution.total_frames),
                duration_ms=int(opportunities.duration_ms),
                goodput_kbps=int(opportunities.goodput_kbps),
                transport=opportunities.transport,
                frame_interval_ms=int(opportunities.frame_interval_ms),
                goodput_kbps_reverse=None if int(reverse) < 0 else int(reverse),
            )
            logger.info("fixed mixing matrix %s (%s), %d edges",
                        digest[:16], source, len(edges))
        else:
            reverse = opportunities.goodput_kbps_reverse
            trace = generate_matching_trace(
                peer_ids=self.peer_ids,
                num_frames=int(dec.execution.total_frames),
                rng=self.rng.stream(STREAM_OPPORTUNITIES),
                duration_ms=int(opportunities.duration_ms),
                goodput_kbps=int(opportunities.goodput_kbps),
                transport=opportunities.transport,
                frame_interval_ms=int(opportunities.frame_interval_ms),
                goodput_kbps_reverse=None if int(reverse) < 0 else int(reverse),
            )

        if profile_requires_matching(dec.comparison.profile):
            validate_matching_profile(trace)

        expected = dec.comparison.expected_trace_digest
        if expected and expected != trace.trace_id:
            raise ValueError(
                f"expected trace digest {expected} but loaded {trace.trace_id}; "
                f"refusing to start a common-comparison run on a different trace"
            )
        logger.info("decentralized trace %s (%d frames)", trace.trace_id[:16], len(trace))
        return trace

    def _resolve_peer_config(self, peer_id: int):
        """Clone the global config and overlay this peer's ``client_<id>`` section.

        Clone, defrost, merge the peer's own section, freeze: each peer gets its own
        settings, not the whole client-config mapping. Without a section the plain
        clone is returned.
        """
        resolved = self.cfg.clone()
        section = None
        if self.client_cfgs:
            section = self.client_cfgs.get(f"client_{peer_id}")
        if section is not None:
            resolved.defrost()
            try:
                # Both calls run every registered config check, so an invalid peer
                # overlay fails while the peers are being built; the error names the
                # peer.
                resolved.merge_from_other_cfg(section)
                resolved.freeze(save=False)
            except ValueError as exc:
                raise ValueError(f"peer {peer_id}: {exc}") from exc
        return resolved

    def _build_peers(self) -> None:
        method = self.factory.name
        for peer_id in self.peer_ids:
            peer_config = self._resolve_peer_config(peer_id)
            context = PeerContext(
                peer_id=peer_id,
                config=peer_config,
                data=self.data.get(peer_id) if hasattr(self.data, "get") else None,
                device=getattr(peer_config, "device", None),
                rng=self.rng.stream(f"method:{method}:peer{peer_id}"),
                client_config=self.client_cfgs.get(f"client_{peer_id}") if self.client_cfgs else None,
                # Every peer of every method is handed the same recent-topology
                # history; what a method does with it is up to the method.
                extras={"topology": self.coordinator.topology},
            )
            self.peers[peer_id] = Peer(peer_id, self.factory.create_peer(context), context)
        logger.info("constructed %d peers hosting %s; no server exists",
                    len(self.peers), method)

        # Ask each peer for its initial-state digest (an optional method hook) now,
        # once every peer is fully built and before any session: a method computes it
        # on the first call and returns that value when the manifest asks again.
        for peer in self.peers.values():
            capture = getattr(peer.algorithm, "initial_state_digest", None)
            if callable(capture):
                capture()

    # -- run -----------------------------------------------------------------

    def _notify(self, hook: str, *args) -> None:
        callback = getattr(self.observer, hook, None)
        if callback is not None:
            callback(self, *args)

    def run(self, resume_from: Optional[str] = None) -> dict:
        self._build_peers()
        self._notify("on_peers_built")
        dec = self.cfg.decentralized
        interaction = self.factory.interaction_form
        local_steps = int(dec.execution.local_steps)
        eval_every = int(dec.execution.eval_every_frames)
        event_driven = dec.execution.scheduling == "event_driven"
        algorithms = {pid: peer.algorithm for pid, peer in self.peers.items()}

        start_frame = 0
        if resume_from:
            self.load_state_dict(load_checkpoint(resume_from))
            start_frame = self.coordinator.frame_cursor + 1
            logger.info("resumed from %s at frame %d", resume_from, start_frame)

        self._open_outputs()

        frames = self.trace.frames
        for position in range(start_frame, len(frames)):
            frame = frames[position]
            self.coordinator.advance_clock(frame.time_us)
            self.coordinator.frame_cursor = frame.index

            # Local compute is scheduled per peer independently of contact degree,
            # so a high-degree peer gets no extra optimisation steps. Under
            # event_driven scheduling each peer completes after its own compute
            # latency, so completion is not locked to the contact instant, while
            # the episode count stays identical for every peer.
            for peer_id in self.peer_ids:
                offset = self._compute_latency(peer_id) if event_driven else 0
                self.coordinator.queue.push(LocalComputeEvent(
                    event_id=self.coordinator.allocate_event_id(),
                    time_us=frame.time_us + offset, peer_id=peer_id,
                    episode=frame.index,
                ))

            if interaction == INTERACTION_PAIRWISE:
                for event in compile_pair_sessions(frame, self.coordinator.next_event_id):
                    self.coordinator.next_event_id = event.event_id + 1
                    self.coordinator.queue.push(event)
            elif interaction == INTERACTION_GRAPH:
                if self.fixed_mixing is not None:
                    event = compile_fixed_graph_round(
                        frame, self.peer_ids, self.fixed_mixing[0],
                        self.coordinator.allocate_event_id(),
                        require_complete_frame=bool(
                            dec.execution.require_complete_graph_frames),
                    )
                elif self.graph_mixing == GRAPH_MIXING_DYNAMIC:
                    event = compile_dynamic_graph_round(
                        frame, self.peer_ids, self.coordinator.allocate_event_id(),
                    )
                else:
                    event = compile_graph_round(
                        frame, self.peer_ids, self.effective_gamma(),
                        self.coordinator.allocate_event_id(),
                    )
                self.coordinator.queue.push(event)
            else:
                raise ValueError(f"unknown interaction form {interaction!r}")

            self.coordinator.topology.record_frame(
                frame.index,
                [o.endpoints for o in frame.opportunities if o.available],
            )

            # Drain only up to the next frame's timestamp. Draining everything here
            # would dispatch an event dated beyond the next frame before that
            # frame's own events existed, so dispatch timestamps could go backwards.
            horizon = frames[position + 1].time_us if position + 1 < len(frames) else None
            for event in self.coordinator.queue.pop_until(horizon):
                self._dispatch(event, algorithms, local_steps)

            if eval_every > 0 and (frame.index + 1) % eval_every == 0:
                self._evaluate(frame.index)

            self._notify("on_frame_end", frame.index)
            self._maybe_checkpoint(frame.index)

        final_evaluation = (self._evaluate(len(self.trace) - 1, final=True)
                            if bool(dec.execution.final_evaluation) else {})
        results = self._results(final_evaluation)
        self._close_outputs(results)
        return results

    # -- outputs -------------------------------------------------------------

    def _output_dir(self) -> Optional[str]:
        outdir = getattr(self.cfg, "outdir", "") or ""
        return outdir or None

    def _open_outputs(self) -> None:
        directory = self._output_dir()
        if directory and self.cfg.decentralized.metrics.event_log:
            self.sink.path = os.path.join(directory, "decentralized_events.jsonl")
            self.sink.open()
        if directory:
            write_manifest(self, directory)

    def _close_outputs(self, results: dict) -> None:
        directory = self._output_dir()
        if directory:
            os.makedirs(directory, exist_ok=True)
            summary_path = os.path.join(directory, "decentralized_summary.json")
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump({
                    "trace_id": results["trace_id"],
                    "method": results["method"],
                    "interaction": results["interaction"],
                    "comparison_profile": results["comparison_profile"],
                    "distribution": results["distribution"],
                    "peer_records": results["peer_records"],
                    "summary": results["summary"],
                    "total_bytes": results["total_bytes"],
                }, handle, indent=2, sort_keys=True, default=str)
        self.sink.close()

    def _maybe_checkpoint(self, frame_index: int) -> Optional[str]:
        checkpoint = self.cfg.decentralized.checkpoint
        if not checkpoint.enabled or int(checkpoint.every_frames) <= 0:
            return None
        if (frame_index + 1) % int(checkpoint.every_frames) != 0:
            return None
        directory = checkpoint.dir or (self._output_dir() or ".")
        path = os.path.join(directory, f"decentralized_frame{frame_index}.ckpt")
        save_checkpoint(path, self.state_dict())
        logger.info("checkpointed frame %d to %s", frame_index, path)
        return path

    def _dispatch(self, event, algorithms, local_steps: int) -> None:
        kind = getattr(event, "kind", None)
        if kind == EventKind.LOCAL_COMPUTE:
            self.peers[event.peer_id].local_episode(local_steps)
        elif kind == EventKind.PAIR_SESSION:
            record = execute_pair_session(
                event, algorithms, self.engine, self.ledger,
                checksum=bool(self.cfg.decentralized.transport.checksum),
                failure_injector=self.injector,
                wire_cost=self.wire_cost,
            )
            for endpoint in event.endpoints:
                self.peers[endpoint].sessions += 1
            self.sink.record_event(record)
            self._notify("on_event", event, record)
        elif kind == EventKind.GRAPH_MIXING:
            record = execute_graph_round(
                event, algorithms, self.ledger,
                checksum=bool(self.cfg.decentralized.transport.checksum),
                skipped_edge_policy=self.cfg.decentralized.execution.skipped_edge_policy,
                wire_cost=self.graph_wire_cost,
            )
            self.sink.record_event(record)
            self._notify("on_event", event, record)
        else:
            raise ValueError(f"unhandled event kind {kind!r}")

    # -- evaluation ------------------------------------------------------------

    def _evaluate(self, frame_index: int, final: bool = False) -> Dict[int, dict]:
        """Every peer is evaluated. There is no global adapter to evaluate."""
        per_peer = {pid: dict(peer.evaluate()) for pid, peer in self.peers.items()}
        self.sink.record_event({
            "event": "evaluation",
            "frame_index": frame_index,
            "final": final,
            "per_peer": per_peer,
        })
        return per_peer

    def _results(self, per_peer=None) -> dict:
        # Reuse the final evaluation rather than evaluating every peer again: a second
        # pass costs a full evaluation and, with a stochastic eval split, would not
        # return the same numbers.
        if per_peer is None:
            per_peer = self._evaluate(len(self.trace) - 1, final=True)
        per_peer = {pid: dict(values) for pid, values in per_peer.items()}
        for pid, peer in self.peers.items():
            self.sink.record_peer(pid, {
                "local_episodes": peer.local_episodes,
                "local_steps": peer.local_steps,
                "sessions": peer.sessions,
                "sent_control_bytes": self.ledger.sent_by(pid, "control"),
                "sent_payload_bytes": self.ledger.sent_by(pid, "payload"),
                "state_version": peer.algorithm.state_version(),
            })

        distribution = {}
        numeric_keys = {k for values in per_peer.values() for k, v in values.items()
                        if isinstance(v, (int, float))}
        for key in sorted(numeric_keys):
            values = [v[key] for v in per_peer.values() if key in v]
            if not values:
                continue
            lower_is_better = metric_lower_is_better(key)
            distribution[key] = {
                "mean": sum(values) / len(values),
                # For a lower-is-better metric (a loss) the worst peer is the one
                # with the highest value.
                "worst": max(values) if lower_is_better else min(values),
                "best": min(values) if lower_is_better else max(values),
                "direction": "lower_is_better" if lower_is_better else "higher_is_better",
                "values": values,
            }

        return {
            "trace_id": self.trace.trace_id,
            "method": self.factory.name,
            "interaction": self.factory.interaction_form,
            "comparison_profile": self.cfg.decentralized.comparison.profile,
            "per_peer": per_peer,
            "distribution": distribution,
            "peer_records": dict(self.sink.peers),
            "summary": self.sink.summary(),
            "total_bytes": self.ledger.total(),
        }

    # -- checkpointing -----------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "coordinator": self.coordinator.state_dict(),
            "queue": self.coordinator.queue.state_dict(),
            "ledger": self.ledger.state_dict(),
            "engine": self.engine.state_dict(),
            "injector": self.injector.state_dict(),
            "peers": {pid: peer.state_dict() for pid, peer in self.peers.items()},
            # The runtime's own decisions use named streams, but training draws on
            # the framework's global generators: data-loader shuffling (the
            # ReIterator is reset at each fit, so each episode reshuffles) and LoRA
            # dropout. Without them a resumed run would train on different batches
            # with different dropout masks than an uninterrupted one.
            "framework_rng": _framework_rng_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.coordinator.load_state_dict(state["coordinator"])
        if "queue" in state:
            self.coordinator.queue.load_state_dict(state["queue"])
        self.ledger.load_state_dict(state["ledger"])
        self.engine.load_state_dict(state["engine"])
        self.injector.load_state_dict(state["injector"])
        for pid, peer_state in state["peers"].items():
            self.peers[int(pid)].load_state_dict(peer_state)
        if state.get("framework_rng"):
            _restore_framework_rng(state["framework_rng"])
