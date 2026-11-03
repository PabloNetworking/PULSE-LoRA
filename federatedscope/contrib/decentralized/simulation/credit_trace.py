"""Contact traces with byte credits.

A credit trace says which pairs of peers can hold a session in each frame and how many bytes
each direction of that session may carry:

  - contacts: an independent Erdos-Renyi G(n, p) draw per frame from one seeded generator
    (``core.traces.er_trace_document``), every contact one second long;
  - credits: for every contact direction ``d``, ``floor(b_ref[d] * f)`` bytes with ``f`` drawn
    uniformly from ``[low, high)`` by a seeded stream, one factor per contact direction in
    frame, edge and direction order. A credit is written as the goodput (bits per second) that
    carries exactly that many bytes in one second.

``b_ref`` is the reference credit per direction, for example the serialized size of one complete
session at a reference rank, so the factor range says how many complete sessions a contact can
carry. The trace's ``generator`` block records every parameter and the trace digest.
"""
import math
import random
from typing import Any, Dict, List, Mapping, Sequence

from federatedscope.contrib.decentralized.core.rng import derive_seed
from federatedscope.contrib.decentralized.core.traces import (
    compute_trace_digest,
    er_trace_document,
    normalise_trace_payload,
)
from federatedscope.contrib.decentralized.core.types import gross_credit_bytes

FORWARD, REVERSE = "forward", "reverse"
DIRECTIONS = (FORWARD, REVERSE)
GENERATOR = "er_credit_trace"
GENERATOR_VERSION = 1
DURATION_MS = 1000
#: The label of the seeded stream the capacity factors are drawn from.
CAPACITY_STREAM = "c1b-capacity:v1"


def credit_bytes_to_bps(credit_bytes: int, duration_ms: int = DURATION_MS) -> int:
    """The goodput (bits per second) whose contact of ``duration_ms`` carries exactly
    ``credit_bytes`` under the runtime's ``gross_credit_bytes``. The duration must be a whole
    number of seconds that divides the credit's bits; anything else is refused rather than
    rounded."""
    credit_bytes, duration_ms = int(credit_bytes), int(duration_ms)
    if credit_bytes < 0:
        raise ValueError("a credit is non-negative")
    if duration_ms <= 0 or duration_ms % 1000:
        raise ValueError(f"exact byte credits need whole-second contacts, not {duration_ms} ms")
    seconds = duration_ms // 1000
    bits = credit_bytes * 8
    if bits % seconds:
        raise ValueError(f"{credit_bytes} B over {seconds} s is not a whole bps")
    return bits // seconds


def bps_to_credit_bytes(goodput_bps: int, duration_ms: int = DURATION_MS) -> int:
    """The credit the runtime gives a contact of this goodput and duration."""
    return gross_credit_bytes(int(goodput_bps), int(duration_ms) * 1000)


def draw_factors(count: int, seed: int, low: float, high: float) -> List[float]:
    """``count`` factors, uniform on [low, high), from one generator seeded by name."""
    if not (0 < low < high):
        raise ValueError(f"need 0 < low < high, got {low}, {high}")
    rng = random.Random(derive_seed(int(seed), CAPACITY_STREAM))
    return [rng.uniform(float(low), float(high)) for _ in range(int(count))]


def credit_trace(b_ref: Mapping[str, int], *, peers: Sequence[int], frames: int, p: float,
                 trace_seed: int, capacity_seed: int, low: float, high: float) -> Dict[str, Any]:
    """The trace document: ``frames`` frames of G(n, p) contacts with byte credits."""
    base = er_trace_document(list(peers), int(frames), float(p), int(trace_seed),
                             duration_ms=DURATION_MS, goodput_kbps=1)
    count = sum(len(frame["edges"]) for frame in base["frames"]) * 2
    factors = iter(draw_factors(count, capacity_seed, low, high))
    out_frames = []
    for frame in base["frames"]:
        edges = []
        for edge in frame["edges"]:
            credit = {d: int(math.floor(int(b_ref[d]) * next(factors))) for d in DIRECTIONS}
            edges.append({"endpoints": list(edge["endpoints"]), "duration": DURATION_MS,
                          "goodput_forward": credit_bytes_to_bps(credit[FORWARD]),
                          "goodput_reverse": credit_bytes_to_bps(credit[REVERSE]),
                          "transport": edge["transport"]})
        out_frames.append({"index": frame["index"], "time": frame["time"], "edges": edges})
    document = {"trace_schema_version": base["trace_schema_version"], "time_unit": "ms",
                "duration_unit": "ms", "goodput_unit": "bps",
                "peers": list(base["peers"]), "frames": out_frames}
    digest = compute_trace_digest(normalise_trace_payload(document))
    er = base["generator"]
    document["generator"] = {
        "name": GENERATOR, "version": GENERATOR_VERSION,
        "er_name": er["name"], "er_version": er["version"], "er_trace_id": er["trace_id"],
        "seed": int(trace_seed), "p": float(p), "n": len(base["peers"]),
        "num_frames": int(frames), "capacity_seed": int(capacity_seed),
        "capacity_distribution": "uniform", "capacity_low": float(low),
        "capacity_high": float(high), "b_ref_forward_bytes": int(b_ref[FORWARD]),
        "b_ref_reverse_bytes": int(b_ref[REVERSE]),
        "credit_unit": "bytes per direction per one-second contact",
        "trace_id": digest}
    return document


def contact_credits(trace: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Every contact of a credit trace: frame, endpoints and the credit per direction."""
    rows = []
    for frame in trace["frames"]:
        for edge in frame["edges"]:
            rows.append({"frame_index": int(frame["index"]), "endpoints": list(edge["endpoints"]),
                         FORWARD: bps_to_credit_bytes(edge["goodput_forward"], edge["duration"]),
                         REVERSE: bps_to_credit_bytes(edge["goodput_reverse"], edge["duration"])})
    return rows
