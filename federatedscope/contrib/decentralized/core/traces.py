"""The immutable contact trace, ``CommunicationOpportunityTrace``.

One trace represents *offered* network conditions and nothing else: per frame,
which pairs of peers may hold a session, for how long, and at what goodput in
each direction. It is generated or loaded once, before any peer is built, and
identified by a digest over its normalised contents rather than a file path.
The runtime compiles each frame into pair sessions or a graph round; a method
decides what it sends, but never regenerates, reinterprets or resamples the
trace. Which offered edges a method actually used is an outcome of the run,
not part of the trace.
"""
import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from federatedscope.contrib.decentralized.core.rng import derive_seed
from federatedscope.contrib.decentralized.core.types import (
    DURATION_UNITS,
    GOODPUT_UNITS,
    KNOWN_TRANSPORTS,
    TRANSPORT_DIRECT,
    CommunicationOpportunity,
    OpportunityFrame,
    PeerId,
    UnitError,
    normalise_exact,
)

TRACE_SCHEMA_VERSION = 1

#: The seeded G(n, p) generator (:func:`er_trace_document`). A document it writes
#: carries a ``generator`` block with this name and version, the seed, ``p``,
#: ``n``, the frame count and the trace digest.
ER_GENERATOR_NAME = "er_gnp"
ER_GENERATOR_VERSION = 1


class TraceValidationError(ValueError):
    """A trace is malformed. Never recovered from, never silently defaulted."""


@dataclass(frozen=True)
class CommunicationOpportunityTrace:
    schema_version: int
    peer_ids: Tuple[PeerId, ...]
    frames: Tuple[OpportunityFrame, ...]
    trace_id: str
    normalised: dict
    #: The ``generator`` block the document carried, or None. Not part of the
    #: normalised contents, so not part of the digest: it describes how the offered
    #: conditions were made. A block that records a ``trace_id`` is checked against
    #: the digest of the contents (load_trace). Recorded in the run manifest only
    #: when present.
    provenance: Optional[dict] = None

    def frame(self, index: int) -> OpportunityFrame:
        return self.frames[index]

    def __len__(self) -> int:
        return len(self.frames)


def compute_trace_digest(normalised: dict) -> str:
    """SHA-256 over canonical JSON of the normalised trace.

    Because normalisation converts to integer microseconds and bits/second, two
    files describing identical physical conditions in different declared units
    produce the same digest, while any change to an endpoint, capacity,
    duration, timestamp, availability, or transport label changes it.
    """
    blob = json.dumps(normalised, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TraceValidationError(message)


def normalise_trace_payload(payload: dict) -> dict:
    """Validate a raw trace payload and return its canonical normalised form."""
    _require(isinstance(payload, dict), "trace payload must be a mapping")

    version = payload.get("trace_schema_version")
    _require(
        version == TRACE_SCHEMA_VERSION,
        f"unsupported trace_schema_version {version!r}; expected {TRACE_SCHEMA_VERSION}",
    )

    time_unit = payload.get("time_unit", payload.get("duration_unit"))
    duration_unit = payload.get("duration_unit")
    goodput_unit = payload.get("goodput_unit")
    _require(duration_unit is not None, "trace must declare duration_unit")
    _require(goodput_unit is not None, "trace must declare goodput_unit")

    raw_peers = payload.get("peers")
    _require(isinstance(raw_peers, list) and raw_peers, "trace must declare a non-empty peers list")
    _require(all(isinstance(p, int) and p > 0 for p in raw_peers),
             "peer ids must be positive integers")
    _require(len(set(raw_peers)) == len(raw_peers), "duplicate peer id in peers list")
    peers = sorted(raw_peers)
    peer_set = set(peers)

    raw_frames = payload.get("frames")
    _require(isinstance(raw_frames, list), "trace must declare a frames list")

    frames: List[dict] = []
    previous_time = None
    for position, raw_frame in enumerate(raw_frames):
        _require(isinstance(raw_frame, dict), f"frame {position} must be a mapping")
        index = raw_frame.get("index", position)
        _require(index == position,
                 f"frame index {index!r} is out of order at position {position}")

        try:
            time_us = normalise_exact(raw_frame.get("time", 0), time_unit,
                                      DURATION_UNITS, f"frame {position} time")
        except UnitError as exc:
            raise TraceValidationError(str(exc)) from exc
        _require(time_us >= 0, f"frame {position}: negative time")
        if previous_time is not None:
            _require(time_us >= previous_time,
                     f"frame {position}: time {time_us}us precedes previous frame")
        previous_time = time_us

        raw_edges = raw_frame.get("edges", [])
        _require(isinstance(raw_edges, list), f"frame {position}: edges must be a list")

        seen_edges = set()
        edges: List[dict] = []
        for edge_pos, raw_edge in enumerate(raw_edges):
            where = f"frame {position} edge {edge_pos}"
            _require(isinstance(raw_edge, dict), f"{where}: must be a mapping")

            endpoints = raw_edge.get("endpoints")
            _require(isinstance(endpoints, (list, tuple)) and len(endpoints) == 2,
                     f"{where}: endpoints must be a pair")
            a, b = endpoints
            _require(isinstance(a, int) and isinstance(b, int),
                     f"{where}: endpoints must be integers")
            _require(a != b, f"{where}: self-loop on peer {a}")
            _require(a in peer_set and b in peer_set,
                     f"{where}: endpoint not in declared peers {peers}")
            lo, hi = (a, b) if a < b else (b, a)
            forward_is_declared_order = (a, b) == (lo, hi)
            _require((lo, hi) not in seen_edges,
                     f"{where}: duplicate edge ({lo},{hi}) within the frame")
            seen_edges.add((lo, hi))

            try:
                duration_us = normalise_exact(raw_edge["duration"], duration_unit,
                                              DURATION_UNITS, f"{where} duration")
            except KeyError as exc:
                raise TraceValidationError(f"{where}: missing duration") from exc
            except UnitError as exc:
                raise TraceValidationError(str(exc)) from exc
            _require(duration_us > 0, f"{where}: duration must be positive")

            if "goodput" in raw_edge:
                raw_fwd = raw_rev = raw_edge["goodput"]
            else:
                _require("goodput_forward" in raw_edge and "goodput_reverse" in raw_edge,
                         f"{where}: needs goodput, or goodput_forward and goodput_reverse")
                raw_fwd = raw_edge["goodput_forward"]
                raw_rev = raw_edge["goodput_reverse"]
            try:
                fwd = normalise_exact(raw_fwd, goodput_unit, GOODPUT_UNITS,
                                      f"{where} goodput_forward")
                rev = normalise_exact(raw_rev, goodput_unit, GOODPUT_UNITS,
                                      f"{where} goodput_reverse")
            except UnitError as exc:
                raise TraceValidationError(str(exc)) from exc

            # Declared direction is relative to the declared endpoint order;
            # canonical storage is always (lo, hi), so swap when they disagree.
            if not forward_is_declared_order:
                fwd, rev = rev, fwd

            transport = raw_edge.get("transport", TRANSPORT_DIRECT)
            _require(transport in KNOWN_TRANSPORTS,
                     f"{where}: unknown transport {transport!r}")
            available = bool(raw_edge.get("available", True))
            if available:
                _require(fwd > 0 and rev > 0,
                         f"{where}: an available edge needs positive goodput in both "
                         f"directions (use available=false instead)")

            edges.append({
                "endpoints": [lo, hi],
                "duration_us": duration_us,
                "goodput_bps_forward": fwd,
                "goodput_bps_reverse": rev,
                "transport": transport,
                "available": available,
            })

        edges.sort(key=lambda e: (e["endpoints"][0], e["endpoints"][1]))
        frames.append({"index": position, "time_us": time_us, "edges": edges})

    return {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "peers": peers,
        "frames": frames,
    }


def _plain_generator_block(raw) -> dict:
    """The document's ``generator`` block as fresh built-ins, or a refusal.

    Only JSON scalars under string keys are accepted, because the block is copied
    into the run manifest as it is."""
    if type(raw) is not dict:
        raise TraceValidationError("the trace's generator block is not a plain mapping")
    block = {}
    for key, value in raw.items():
        if type(key) is not str:
            raise TraceValidationError("the trace's generator block has a non-string key")
        if value is not None and type(value) not in (bool, int, float, str):
            raise TraceValidationError(
                f"the trace's generator block carries a non-scalar under {key!r}")
        block[key] = value
    return block


def _trace_from_normalised(normalised: dict,
                           provenance: Optional[dict] = None) -> CommunicationOpportunityTrace:
    frames = tuple(
        OpportunityFrame(
            index=frame["index"],
            time_us=frame["time_us"],
            opportunities=tuple(
                CommunicationOpportunity(
                    endpoints=(edge["endpoints"][0], edge["endpoints"][1]),
                    duration_us=edge["duration_us"],
                    goodput_bps_forward=edge["goodput_bps_forward"],
                    goodput_bps_reverse=edge["goodput_bps_reverse"],
                    transport=edge["transport"],
                    available=edge["available"],
                )
                for edge in frame["edges"]
            ),
        )
        for frame in normalised["frames"]
    )
    return CommunicationOpportunityTrace(
        schema_version=normalised["trace_schema_version"],
        peer_ids=tuple(normalised["peers"]),
        frames=frames,
        trace_id=compute_trace_digest(normalised),
        normalised=normalised,
        provenance=provenance,
    )


def load_trace(payload: dict) -> CommunicationOpportunityTrace:
    normalised = normalise_trace_payload(payload)
    provenance = None
    if isinstance(payload, dict) and payload.get("generator") is not None:
        provenance = _plain_generator_block(payload["generator"])
        recorded = provenance.get("trace_id")
        computed = compute_trace_digest(normalised)
        if recorded is not None and recorded != computed:
            # The digest inside the document is a claim about its contents; a
            # document edited after it was written, or one carrying another's block,
            # is refused rather than loaded under a name it does not have.
            raise TraceValidationError(
                f"the trace document records trace_id {str(recorded)[:16]}... but its "
                f"contents digest to {computed[:16]}...; refusing a document whose "
                f"recorded identity is not its own")
    return _trace_from_normalised(normalised, provenance)


def load_trace_file(path: str) -> CommunicationOpportunityTrace:
    """Load a file-backed trace. Never falls back to generation on failure."""
    if not os.path.isfile(path):
        raise TraceValidationError(f"trace file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise TraceValidationError(f"trace file {path} is not valid JSON: {exc}") from exc
    return load_trace(payload)


def profile_requires_matching(profile: str) -> bool:
    """Only the ``common_matching_trace`` profile requires every frame to be a matching;
    ``common_study`` and ``paper_native`` admit a peer in several edges of one frame."""
    return str(profile) == "common_matching_trace"


def validate_matching_profile(trace: CommunicationOpportunityTrace) -> None:
    """Under ``common_matching_trace`` no peer may appear in two available edges."""
    for frame in trace.frames:
        seen: Dict[PeerId, Tuple[PeerId, PeerId]] = {}
        for opportunity in frame.opportunities:
            if not opportunity.available:
                continue
            for endpoint in opportunity.endpoints:
                if endpoint in seen:
                    raise TraceValidationError(
                        f"frame {frame.index}: peer {endpoint} appears in overlapping "
                        f"edges {seen[endpoint]} and {opportunity.endpoints}; the "
                        f"common_matching_trace profile requires a matching"
                    )
                seen[endpoint] = opportunity.endpoints


def generate_fixed_graph_trace(
    peer_ids: Sequence[PeerId],
    edges: Sequence[Tuple[PeerId, PeerId]],
    num_frames: int,
    duration_ms: int = 1000,
    goodput_kbps: int = 1000,
    transport: str = TRANSPORT_DIRECT,
    frame_interval_ms: int = 1000,
    goodput_kbps_reverse: Optional[int] = None,
) -> CommunicationOpportunityTrace:
    """Offer the same fixed edge set in every frame.

    This is what a paper-native fixed-graph formulation assumes: a connected
    graph whose edges are all available each round. It is not a matching, so it
    needs no RNG and is not validated as one.
    """
    peers = sorted(peer_ids)
    _require(len(peers) >= 2, "a fixed graph needs at least two peers")
    _require(bool(edges), "a fixed graph needs at least one edge")
    reverse = goodput_kbps if goodput_kbps_reverse is None else goodput_kbps_reverse

    normalised_edges = sorted({tuple(sorted(edge)) for edge in edges})
    frames = []
    for index in range(num_frames):
        frames.append({
            "index": index,
            "time": index * frame_interval_ms,
            "edges": [
                {"endpoints": [lo, hi], "duration": duration_ms,
                 "goodput_forward": goodput_kbps, "goodput_reverse": reverse,
                 "transport": transport}
                for lo, hi in normalised_edges
            ],
        })

    return load_trace({
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "time_unit": "ms", "duration_unit": "ms", "goodput_unit": "kbps",
        "peers": peers, "frames": frames,
    })


def er_trace_document(
    peer_ids: Sequence[PeerId],
    num_frames: int,
    p,
    seed: int,
    duration_ms: int = 1000,
    goodput_kbps: int = 1000,
    transport: str = TRANSPORT_DIRECT,
    frame_interval_ms: int = 1000,
    goodput_kbps_reverse: Optional[int] = None,
) -> dict:
    """The G(n, p) contact-trace document: a plain dict that :func:`load_trace` accepts.

    Every frame draws each of the n(n-1)/2 pairs independently with probability
    ``p`` (Erdos-Renyi edge activation, time-varying) from one generator seeded by
    name from ``seed`` (``rng.derive_seed``), so the same arguments give the same
    document byte for byte. Pairs are drawn in canonical order, so the draw
    sequence is a function of the arguments alone.

    The ``generator`` block records the generator's name and version, ``seed``,
    ``p``, ``n``, the frame count and the trace digest of the normalised contents
    (the ``trace_id`` the runner and the manifest see), which ``load_trace`` checks
    against the contents.
    """
    peers = sorted(int(peer) for peer in peer_ids)
    _require(len(peers) >= 2, "an ER trace needs at least two peers")
    _require(len(set(peers)) == len(peers), "duplicate peer id")
    _require(int(num_frames) >= 1, "an ER trace needs at least one frame")
    try:
        probability = float(p)
    except (TypeError, ValueError) as exc:
        raise TraceValidationError(f"p must be a probability, got {p!r}") from exc
    _require(0.0 <= probability <= 1.0, f"p must lie in [0, 1], got {p!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TraceValidationError(f"seed must be an integer, got {seed!r}")
    reverse = goodput_kbps if goodput_kbps_reverse is None else goodput_kbps_reverse

    rng = random.Random(derive_seed(int(seed), f"{ER_GENERATOR_NAME}:v{ER_GENERATOR_VERSION}"))
    pairs = [(peers[i], peers[j]) for i in range(len(peers)) for j in range(i + 1, len(peers))]
    frames = []
    for index in range(int(num_frames)):
        edges = [{"endpoints": [lo, hi], "duration": int(duration_ms),
                  "goodput_forward": int(goodput_kbps), "goodput_reverse": int(reverse),
                  "transport": transport}
                 for lo, hi in pairs if rng.random() < probability]
        frames.append({"index": index, "time": index * int(frame_interval_ms), "edges": edges})

    document = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "time_unit": "ms", "duration_unit": "ms", "goodput_unit": "kbps",
        "peers": peers, "frames": frames,
    }
    digest = compute_trace_digest(normalise_trace_payload(document))
    document["generator"] = {
        "name": ER_GENERATOR_NAME, "version": ER_GENERATOR_VERSION,
        "seed": int(seed), "p": probability, "n": len(peers),
        "num_frames": int(num_frames), "trace_id": digest,
    }
    return document


def generate_er_trace(peer_ids: Sequence[PeerId], num_frames: int, p, seed: int,
                      **options) -> CommunicationOpportunityTrace:
    """:func:`er_trace_document`, loaded - digest checked, provenance kept."""
    return load_trace(er_trace_document(peer_ids, num_frames, p, seed, **options))


def generate_matching_trace(
    peer_ids: Sequence[PeerId],
    num_frames: int,
    rng,
    duration_ms: int = 1000,
    goodput_kbps: int = 1000,
    transport: str = TRANSPORT_DIRECT,
    frame_interval_ms: int = 1000,
    goodput_kbps_reverse: Optional[int] = None,
) -> CommunicationOpportunityTrace:
    """Generate a trace of random matchings, one per frame, from an injected RNG.

    The RNG must be a stream of :class:`~federatedscope.contrib.decentralized.core.rng.RngService`;
    the process-global generators are never used. With an odd peer count one peer
    sits out each frame, which is still a matching.
    """
    peers = sorted(peer_ids)
    _require(len(peers) >= 2, "a matching trace needs at least two peers")
    reverse = goodput_kbps if goodput_kbps_reverse is None else goodput_kbps_reverse

    frames = []
    for index in range(num_frames):
        shuffled = list(peers)
        rng.shuffle(shuffled)
        edges = []
        for position in range(0, len(shuffled) - 1, 2):
            a, b = shuffled[position], shuffled[position + 1]
            lo, hi = (a, b) if a < b else (b, a)
            edges.append({
                "endpoints": [lo, hi],
                "duration": duration_ms,
                "goodput_forward": goodput_kbps,
                "goodput_reverse": reverse,
                "transport": transport,
            })
        frames.append({"index": index, "time": index * frame_interval_ms, "edges": edges})

    trace = load_trace({
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "time_unit": "ms",
        "duration_unit": "ms",
        "goodput_unit": "kbps",
        "peers": peers,
        "frames": frames,
    })
    validate_matching_profile(trace)
    return trace
