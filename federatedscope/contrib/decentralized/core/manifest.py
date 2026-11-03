"""Reproducibility manifest for one decentralized run.

Records what a reader needs to decide whether two results are comparable: the
contact-trace identity, the local-work schedule, precision, auxiliary-state
policy, mixing coefficient, wire version and wire-cost model, and per-peer method
settings and digests. A cross-method comparison is only meaningful if these
agree, so they are written down rather than assumed. The runner writes the
manifest into its output directory (``decentralized_manifest.json``), and
:func:`assert_comparable` checks two manifests against each other.

The manifest records the offered communication; the realized communication is
reported in the run summary.
"""
import hashlib
import json
import os
import platform
from typing import Any, Dict, Optional


def _applied_scheduling(dec) -> Dict[str, Any]:
    """The scheduling actually in force, with unused latency canonicalised to zero.

    Compute latency only affects ``event_driven`` scheduling. Recording a
    configured-but-ignored value would make two synchronized runs look
    incomparable over a setting neither of them applied, so under any other
    scheduling mode the applied values are zero.
    """
    scheduling = dec.execution.scheduling
    event_driven = scheduling == "event_driven"
    return {
        "scheduling": scheduling,
        "compute_latency_us": int(dec.execution.compute_latency_us) if event_driven else 0,
        "compute_latency_jitter_us": (
            int(dec.execution.compute_latency_jitter_us) if event_driven else 0),
    }


def _peer_method_settings(runner) -> Dict[str, Any]:
    """Each peer's ``method_metrics()``, keyed by peer id.

    Recorded per peer, so a heterogeneous assignment (different ranks or
    supports on different peers) is reproducible from the manifest.
    """
    peers = getattr(runner, "peers", None)
    if not peers:
        return {}
    out: Dict[str, Any] = {}
    for peer_id, peer in sorted(peers.items()):
        describe = getattr(peer.algorithm, "method_metrics", None)
        if callable(describe):
            out[str(peer_id)] = dict(describe())
    return out


def _value_precision(runner) -> Any:
    """The numeric precision the run exchanges values at.

    Read through the optional ``value_precision()`` method hook, so ``core/``
    asks no LoRA question directly. ``None`` when no peer's method provides the
    hook, which only matches another run without it; ``"<heterogeneous>"`` when
    peers report different values.

    It is a comparability axis because two runs at different precision exchange
    different numbers of bytes for the same component and accumulate different
    round-off.
    """
    peers = getattr(runner, "peers", None)
    if not peers:
        return None
    reported = set()
    for _, peer in sorted(peers.items()):
        describe = getattr(peer.algorithm, "value_precision", None)
        reported.add(describe() if callable(describe) else None)
    if len(reported) == 1:
        return reported.pop()
    return "<heterogeneous>"


def _initial_state_digests(runner) -> Dict[str, Any]:
    """Per-peer digest of the state each peer started from.

    Read through the optional ``initial_state_digest()`` method hook (``None``
    for a peer without it). A shared seed does not imply a shared starting
    point: two methods can hold the same seed and begin from different effective
    updates.
    """
    peers = getattr(runner, "peers", None)
    if not peers:
        return {}
    out = {}
    for peer_id, peer in sorted(peers.items()):
        describe = getattr(peer.algorithm, "initial_state_digest", None)
        out[str(peer_id)] = describe() if callable(describe) else None
    return out


def _partition_digests(runner) -> Dict[str, Any]:
    """Per-peer digest of the data each peer holds.

    Built from per-example identities (:func:`_example_identities`), not from
    lengths: two partitions of equal size are not the same partition. A peer
    whose partition cannot be identified gets :data:`UNPROVEN_PARTITION`. A run
    without data that declares itself dataset-free
    (``decentralized.comparison.dataset_free``) gets :data:`NO_DATASET_PARTITIONS`
    for every peer, provided no peer holds data.
    """
    peer_ids = tuple(getattr(runner, "peer_ids", ()) or ())
    data = getattr(runner, "data", None)
    if not data:
        declared = bool(getattr(runner.cfg.decentralized.comparison,
                                "dataset_free", False))
        if not declared:
            # No data and no dataset-free declaration: nothing identifies the
            # partitions.
            return {str(peer_id): UNPROVEN_PARTITION for peer_id in peer_ids}
        # Declared dataset-free: check that no peer holds data.
        holders = [pid for pid, peer in getattr(runner, "peers", {}).items()
                   if getattr(peer, "data", None) is not None
                   or getattr(getattr(peer, "context", None), "data", None) is not None]
        if holders:
            return {str(peer_id): UNPROVEN_PARTITION for peer_id in peer_ids}
        return {str(peer_id): NO_DATASET_PARTITIONS for peer_id in peer_ids}
    out: Dict[str, Any] = {}
    for peer_id in getattr(runner, "peer_ids", ()):
        split = data.get(peer_id) if hasattr(data, "get") else None
        if split is None:
            # No split for this peer: recorded as unproven, not as a digest.
            out[str(peer_id)] = UNPROVEN_PARTITION
            continue
        material = []
        unproven = False
        for name in sorted(dir(split)) if not isinstance(split, dict) else sorted(split):
            if isinstance(split, dict):
                value = split[name]
            elif name.startswith("_"):
                continue
            else:
                value = getattr(split, name, None)
            identities = _example_identities(value)
            if identities == UNPROVEN_PARTITION:
                # One unidentifiable field makes the whole peer unproven: hashing
                # the sentinel with the rest would turn it into an ordinary digest.
                unproven = True
                break
            if identities is not None:
                material.append(f"{name}:{identities}")
        if unproven:
            out[str(peer_id)] = UNPROVEN_PARTITION
            continue
        if not material:
            # Nothing about this split's contents was discoverable. A type name
            # is not a partition digest: two different partitions of the same
            # class would share it.
            out[str(peer_id)] = UNPROVEN_PARTITION
            continue
        blob = "|".join(material)
        out[str(peer_id)] = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]
    return out


#: Recorded when a peer's partition cannot be identified (its example
#: identities cannot be discovered). A sentinel, not a digest.
UNPROVEN_PARTITION = "<unproven:identities-not-discoverable>"

#: Recorded for every peer of a run that declares itself dataset-free
#: (``decentralized.comparison.dataset_free``) when no peer holds data.
NO_DATASET_PARTITIONS = "<verified:dataset-free-scenario>"


def _example_identities(value) -> Optional[str]:
    """A fingerprint of every example a split contains, in order.

    ``idx:`` and the SHA-256 of the full index list for a split that exposes
    ``indices`` (a ``Subset``), otherwise ``items:`` and the SHA-256 of every
    item's stable representation. Every example is hashed, not a sample: two
    same-length partitions that differ in one position must not collide.
    ``None`` for a value that is not a dataset, ``"empty"`` for an empty one, and
    :data:`UNPROVEN_PARTITION` when an item cannot be read.
    """
    dataset = getattr(value, "dataset", value)
    if dataset is None or isinstance(dataset, (str, int, float, bool)):
        return None
    try:
        size = len(dataset)
    except TypeError:
        return None
    if size == 0:
        return "empty"

    indices = getattr(value, "indices", None)
    if indices is not None:
        # Every index, in order: a Subset is identified by its index list.
        return "idx:" + hashlib.sha256(
            ",".join(str(int(i)) for i in indices).encode("utf-8")).hexdigest()

    digest = hashlib.sha256()
    digest.update(f"n={size}|".encode("utf-8"))
    for index in range(size):
        try:
            item = dataset[index]
        except Exception:
            return UNPROVEN_PARTITION
        digest.update(_stable_repr(item).encode("utf-8"))
        digest.update(b"\x1e")
    return "items:" + digest.hexdigest()


def _stable_repr(item) -> str:
    """A representation that is stable across processes for the usual shapes."""
    import torch

    if torch.is_tensor(item):
        return f"t{tuple(item.shape)}:{item.detach().cpu().numpy().tobytes().hex()}"
    if isinstance(item, dict):
        return "{" + ",".join(f"{k}={_stable_repr(item[k])}"
                              for k in sorted(item, key=str)) + "}"
    if isinstance(item, (list, tuple)):
        return "[" + ",".join(_stable_repr(v) for v in item) + "]"
    return repr(item)


def _offered_communication_digest(frames) -> str:
    """SHA-256 of the offered frames (:func:`build_offered_frames`).

    It restates ``trace_id``, the digest of the normalised trace, from the frames
    the runner holds and over exactly the fields a comparison holds equal: each
    frame's index and time and, per edge, the endpoints in canonical order,
    duration, both directional goodputs, availability and transport label. A
    mismatch then names the offered communication rather than an opaque trace id.
    """
    blob = json.dumps(frames, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_offered_frames(trace) -> list:
    """The offered communication as plain built-ins, one entry per frame.

    Each frame's index, time and offered edges (endpoints, duration, both
    directional goodputs, availability, transport). The runner builds it once,
    when it loads the trace, and the manifest digests it. Frames that offer
    nothing are kept: they are still frames of the trace.
    """
    frames = []
    for frame in trace.frames:
        offered = [{
            "endpoints": list(opportunity.endpoints),
            "duration_us": int(opportunity.duration_us),
            "goodput_bps_forward": int(opportunity.goodput_bps_forward),
            "goodput_bps_reverse": int(opportunity.goodput_bps_reverse),
            "available": bool(opportunity.available),
            "transport": opportunity.transport,
        } for opportunity in frame.opportunities]
        # Canonical edge order, so the same offered set listed in another order
        # gives the same digest. The order of frames is part of the trace and
        # is kept.
        offered.sort(key=lambda o: (o["endpoints"], o["transport"]))
        frames.append({"index": int(frame.index), "time_us": int(frame.time_us),
                       "offered": offered})
    return frames


def _graph_mixing_manifest(runner) -> Dict[str, Any]:
    """``graph_mixing`` (mode and rule) for a run under ``dynamic_metropolis``.

    Nothing for any other run. A reader who finds the key knows the run mixed by
    a per-frame renormalised matrix rather than by the pair block or a fixed Q.
    """
    from federatedscope.contrib.decentralized.core.interactions import (
        GRAPH_MIXING_DYNAMIC, METROPOLIS_RULE)

    if getattr(runner, "graph_mixing", None) != GRAPH_MIXING_DYNAMIC:
        return {}
    return {"graph_mixing": {"mode": GRAPH_MIXING_DYNAMIC, "rule": METROPOLIS_RULE}}


def _envelope_manifest(runner) -> Dict[str, Any]:
    """``transport_envelope_bytes``, the envelope in force, when it is not the 64 MiB
    default; nothing otherwise."""
    from federatedscope.contrib.decentralized.core.wire_cost import DEFAULT_ENVELOPE_BYTES

    envelope = int(getattr(getattr(runner, "wire_cost", None), "envelope_bytes",
                           DEFAULT_ENVELOPE_BYTES))
    if envelope == DEFAULT_ENVELOPE_BYTES:
        return {}
    return {"transport_envelope_bytes": envelope}


def _trace_generator_manifest(runner) -> Dict[str, Any]:
    """``trace_generator``: the generator block the trace document carried (its name,
    version, seed, p, n, trace digest, ...), when it carried one. A hand-written or
    generated matching trace records nothing here."""
    provenance = getattr(runner, "_trace_provenance", None)
    if not provenance:
        return {}
    return {"trace_generator": dict(provenance)}


#: The generator name ``traces.er_trace_document`` records.
_ER_GENERATOR = "er_gnp"


NATIVE_GRAPHS = ("dynamic", "fixed")


def run_label(manifest: Dict[str, Any], native_form: str = "graph_round",
              native_graph: str = "dynamic", topology: str = None) -> str:
    """A run's label, derived from its manifest and the method's published setting.

    ``native_form`` is the method's published interaction form: ``graph_round`` for
    a graph-native baseline, ``pairwise_session`` for a method whose native form is
    the pair session. ``native_graph`` is the graph its paper mixes over: ``dynamic``
    (a time-varying graph) or ``fixed`` (a fixed ring or a fixed Erdos-Renyi graph).
    ``topology`` names a fixed matrix whose origin the manifest cannot say, an
    explicit ER matrix for instance. Core names no method; the caller says which it
    has. With ``<m>`` the method name without its ``_lora`` suffix:

    1. a pair-session run is ``<m>-native`` when the method's native form is the pair
       session and ``<m>-pairwise-adapted`` otherwise; a pair-block graph round
       (degree-one execution from a matching) is ``<m>-pairwise-adapted`` too;
    2. a run on a fixed matrix is ``<m>-native-<topology>`` (``dec-native-ring``,
       ``dec-native-er``) for a fixed-graph method and ``<m>-<topology>-static`` for a
       dynamic-graph one, where ``<topology>`` is the named topology, ``fixed-q`` for
       an explicit matrix, or the ``topology`` argument when one is given;
    3. a run under ``graph_mixing: dynamic_metropolis`` is ``<m>-native-dynamic-<g>``
       for a dynamic-graph method and ``<m>-dynamic-<g>-adapted`` for a fixed-graph
       one, where ``<g>`` is ``er`` when the trace document records the ER generator
       (:data:`_ER_GENERATOR`, as its ``name`` or, for a credit trace built on it, its
       ``er_name``) and ``graph`` otherwise.
    """
    if native_graph not in NATIVE_GRAPHS:
        raise ValueError(f"native_graph {native_graph!r} is not one of {list(NATIVE_GRAPHS)}")
    method = str(manifest["method"])
    short = method[:-len("_lora")] if method.endswith("_lora") else method
    form = manifest.get("interaction_form")
    if form == "pairwise_session":
        return f"{short}-native" if native_form == "pairwise_session" \
            else f"{short}-pairwise-adapted"
    mixing = manifest.get("graph_mixing") or {}
    if mixing.get("mode") == "dynamic_metropolis":
        generator = manifest.get("trace_generator") or {}
        graph = "er" if _ER_GENERATOR in (generator.get("name"), generator.get("er_name")) \
            else "graph"
        if native_graph == "fixed":
            return f"{short}-dynamic-{graph}-adapted"
        return f"{short}-native-dynamic-{graph}"
    source = manifest.get("fixed_mixing_source")
    if source:
        named = source.split(":")[1] if str(source).startswith("topology:") else "fixed-q"
        if topology is not None:
            named = str(topology)
        if native_graph == "fixed":
            return f"{short}-native-{named}"
        return f"{short}-{named}-static"
    return f"{short}-pairwise-adapted"


def _applied_method_settings(runner) -> Dict[str, Any]:
    """Run-level summary of the per-peer method settings.

    A value on which every peer agrees is reported as it is; one that differs
    between peers is reported as ``"<heterogeneous>"`` here, and its per-peer
    values are in ``peer_method_settings``. Descriptive only, not a comparability
    axis (see :func:`assert_comparable`).
    """
    per_peer = _peer_method_settings(runner)
    if not per_peer:
        return {}
    keys = set()
    for settings in per_peer.values():
        keys |= set(settings)

    summary: Dict[str, Any] = {}
    for key in sorted(keys):
        values = [settings.get(key) for settings in per_peer.values()]
        first = values[0]
        summary[key] = first if all(v == first for v in values) else "<heterogeneous>"
    return summary


def build_manifest(runner) -> Dict[str, Any]:
    """The manifest of ``runner``'s run, as a JSON-serializable dict."""
    cfg = runner.cfg
    dec = cfg.decentralized
    import torch

    from federatedscope.contrib.decentralized.core.wire import WIRE_VERSION

    manifest = {
        "method": runner.factory.name,
        "plugin_version": dec.plugin_version,
        "interaction_form": runner.factory.interaction_form,
        "comparison_profile": dec.comparison.profile,
        "gamma": float(dec.comparison.gamma),
        "auxiliary_state_policy": dec.auxiliary_state.policy,

        # Offered communication identity - the thing that must match across methods.
        "trace_id": runner._trace_id,
        "offered_communication_digest": _offered_communication_digest(runner._offered_frames),
        "trace_schema_version": runner._trace_schema_version,
        "trace_frames": runner._trace_frame_count,
        "trace_source": dec.opportunities.source,
        "expected_trace_digest": dec.comparison.expected_trace_digest or None,

        # Local-work schedule - equally necessary for comparability.
        "peer_ids": list(runner.peer_ids),
        "local_steps_per_episode": int(dec.execution.local_steps),
        "total_frames": int(dec.execution.total_frames),
        **_applied_scheduling(dec),

        # Accounting and precision.
        # What the run actually applied, not merely what was configured.
        "skipped_edge_policy": dec.execution.skipped_edge_policy,
        "fixed_mixing_digest": (runner.fixed_mixing[2]
                                if getattr(runner, "fixed_mixing", None) else None),
        "fixed_mixing_source": (runner.fixed_mixing[3]
                                if getattr(runner, "fixed_mixing", None) else None),
        # Present only when they apply: dynamic graph mixing, a trace generator block.
        **_graph_mixing_manifest(runner),
        **_trace_generator_manifest(runner),
        # Recorded so a heterogeneous assignment can be reproduced. Not a
        # comparability axis: differing method policy is the thing under
        # comparison, while the shared trace and local-work axes are what have to
        # match. See assert_comparable.
        "applied_method_settings": _applied_method_settings(runner),
        "peer_method_settings": _peer_method_settings(runner),

        "wire_version": WIRE_VERSION,
        # The wire-cost model that admission was priced in. Two runs under
        # different models sized k_e (the number of components a session carries)
        # in different bytes and are not comparable.
        "wire_cost": str(dec.transport.wire_cost),
        **_envelope_manifest(runner),
        "checksum": bool(dec.transport.checksum),
        "seed": int(dec.seed) if int(dec.seed) else int(getattr(cfg, "seed", 0)),
        "value_precision": _value_precision(runner),
        "initial_state_digests": _initial_state_digests(runner),
        "partition_digests": _partition_digests(runner),

        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }
    # The data-partition specification, read off the runner by name and recorded only
    # when the runner carries one. DecentralizedRunner itself sets no such attribute.
    spec = getattr(runner, "data_partition", None)
    if spec is not None:
        manifest["data_partition"] = dict(spec)
    return manifest


def write_manifest(runner, directory: str) -> str:
    """Write the manifest to ``<directory>/decentralized_manifest.json``; return the path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "decentralized_manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(build_manifest(runner), handle, indent=2, sort_keys=True)
    return path


def assert_comparable(left: Dict[str, Any], right: Dict[str, Any]) -> None:
    """Refuse to treat two runs as directly comparable unless the axes align.

    Payload and local-update *policy* may differ - that is the thing being
    compared. Offered communication and local work may not. Method settings,
    including per-peer heterogeneity, are recorded in the manifest but are
    deliberately absent from these axes. Raises ``ValueError`` naming every
    axis that differs.
    """
    axes = (
        "trace_id", "gamma", "auxiliary_state_policy", "peer_ids",
        "local_steps_per_episode", "total_frames", "wire_version", "wire_cost",
        "comparison_profile",
        # The local-work schedule is one of the axes comparability rests on, and
        # *when* a peer finishes computing is part of that schedule - not only how
        # many steps it takes. Recorded values are canonicalised, so an ignored
        # latency setting cannot create a false mismatch.
        "scheduling", "compute_latency_us", "compute_latency_jitter_us",
        # A direct comparison also holds these equal:
        #   seed          - the run's named random streams derive from it
        #   checksum      - changes the encoded size of every frame, so it moves
        #                   the byte axis a byte-normalised comparison rests on
        #   precision     - changes both the bytes per component and the
        #                   round-off, i.e. two different experiments
        #   offered digest- the structural restatement of the offered conditions
        "seed", "checksum", "value_precision", "offered_communication_digest",
        # A per-frame renormalised W_t is a different mixing operator from the
        # pair block or a fixed Q. Absent (None) on a run that does not use it.
        "graph_mixing",
        # The envelope in force; absent (None) at the 64 MiB default.
        "transport_envelope_bytes",
        # The data-partition specification. Absent on both sides is equal (neither
        # run stated one); stated on one side only, or stated differently, is a
        # mismatch.
        "data_partition",
    )
    mismatched = {
        axis: (left.get(axis), right.get(axis))
        for axis in axes
        if left.get(axis) != right.get(axis)
    }
    if mismatched:
        raise ValueError(
            "runs are not directly comparable; differing axes: "
            + ", ".join(f"{k}={v[0]!r} vs {v[1]!r}" for k, v in mismatched.items())
        )
