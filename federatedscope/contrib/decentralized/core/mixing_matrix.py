"""Fixed mixing matrices for graph methods that mix over a fixed graph.

A method whose published formulation mixes over a *fixed* connected graph with a
symmetric doubly stochastic ``Q`` gets that matrix built or parsed and validated
once, then uses it unchanged for every round. A per-frame matrix compiled from a
matching trace is a different mixing rule (``interactions.compile_graph_round``).

Method-agnostic: this module knows about peers and weights, not about LoRA.
"""
import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from federatedscope.contrib.decentralized.core.types import PeerId

TOPOLOGY_RING = "ring"
TOPOLOGY_COMPLETE = "complete"
TOPOLOGY_STAR = "star"
NAMED_TOPOLOGIES = (TOPOLOGY_RING, TOPOLOGY_COMPLETE, TOPOLOGY_STAR)

TOLERANCE = 1e-9


class MixingMatrixError(ValueError):
    pass


#: Marks a JSON matrix literal in a ``str`` option. A bare literal cannot pass
#: through the config loader: yacs ``_decode_cfg_value`` literal-evaluates every
#: string, so ``"[[0.5, 0.5], [0.5, 0.5]]"`` arrives as a list and is refused as a
#: type mismatch against the ``str`` default. ``json:[[...]]`` is no Python
#: literal, so it stays a string and is parsed here.
JSON_PREFIX = "json:"


def _rows_or_refuse(rows, where: str):
    """A square list of rows of numbers, or a refusal that names the route."""
    if not isinstance(rows, list) or not rows or not all(isinstance(r, list) for r in rows):
        raise MixingMatrixError(f"{where}: an explicit matrix must be a list of rows")
    if any(len(row) != len(rows) for row in rows):
        raise MixingMatrixError(
            f"{where}: an explicit matrix must be square, got {len(rows)} rows of "
            f"lengths {[len(row) for row in rows]}")
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MixingMatrixError(
                    f"{where}: entry ({i},{j}) is {value!r}, not a number")
    return rows


def _unpack(payload, where: str) -> Tuple[list, Optional[list]]:
    """``[rows]`` or ``{"matrix": rows, "peers": [...]}`` -> (rows, declared peers)."""
    declared = None
    if isinstance(payload, dict):
        declared = payload.get("peers")
        payload = payload.get("matrix")
        if declared is not None and not (isinstance(declared, list)
                                         and all(isinstance(p, int) and not isinstance(p, bool)
                                                 for p in declared)):
            raise MixingMatrixError(f"{where}: peers must be a list of integers")
    return _rows_or_refuse(payload, where), declared


def parse_explicit_q(spec: str) -> Tuple[list, Optional[list], str]:
    """Parse an explicit matrix given in a ``str`` configuration option.

    ``fixed_q`` / ``fixed_wt`` are ``str`` options, and the config loader refuses a
    list on the path that serves a scenario's ``cfg_overrides`` (and literal-evaluates
    a bare JSON literal into a list, see :data:`JSON_PREFIX`). So an explicit matrix
    travels in the string itself, in one of two forms:

    * ``json:`` followed by a JSON matrix literal - a list of rows, or
      ``{"matrix": [...], "peers": [...]}``;
    * the path of a JSON file holding the same.

    ``peers``, when given, must be the run's own (checked by :func:`resolve_fixed_q`).
    Returns ``(rows, declared_peers, source)``; the matrix is validated by the caller
    exactly as a list supplied in code is. A bare ``[...]`` literal is accepted for
    callers that bypass the config loader; through the loader it never arrives as a
    string.
    """
    text = str(spec).strip()
    literal = text[len(JSON_PREFIX):] if text.startswith(JSON_PREFIX) else (
        text if text.startswith("[") else None)
    if literal is not None:
        try:
            payload = json.loads(literal)
        except json.JSONDecodeError as exc:
            raise MixingMatrixError(f"fixed matrix literal is not valid JSON: {exc}") from exc
        rows, declared = _unpack(payload, "fixed matrix literal")
        return rows, declared, "explicit:json"
    if os.path.isfile(text):
        with open(text, "r", encoding="utf-8") as handle:
            try:
                payload = json.load(handle)
            except json.JSONDecodeError as exc:
                raise MixingMatrixError(
                    f"fixed matrix file {text} is not valid JSON: {exc}") from exc
        rows, declared = _unpack(payload, f"fixed matrix file {os.path.basename(text)}")
        return rows, declared, f"explicit:file:{os.path.basename(text)}"
    raise MixingMatrixError(
        f"{spec!r} is neither a named topology {list(NAMED_TOPOLOGIES)}, a "
        f"'{JSON_PREFIX}' JSON matrix literal, nor the path of a JSON matrix file")


def _neighbours(topology: str, peers: Sequence[PeerId]) -> List[Tuple[PeerId, PeerId]]:
    count = len(peers)
    if count < 2:
        raise MixingMatrixError("a fixed mixing graph needs at least two peers")
    if topology == TOPOLOGY_RING:
        if count == 2:
            return [(peers[0], peers[1])]
        return [(peers[i], peers[(i + 1) % count]) for i in range(count)]
    if topology == TOPOLOGY_COMPLETE:
        return [(peers[i], peers[j]) for i in range(count) for j in range(i + 1, count)]
    if topology == TOPOLOGY_STAR:
        return [(peers[0], peers[j]) for j in range(1, count)]
    raise MixingMatrixError(
        f"unknown topology {topology!r}; expected one of {list(NAMED_TOPOLOGIES)}")


def build_named_q(topology: str, peer_ids: Sequence[PeerId], gamma: float):
    """Q on a named topology, with the same weight ``gamma / max_degree`` on every edge.

    Dividing by the maximum degree keeps the matrix doubly stochastic with a
    non-negative diagonal for any degree, so a ring and a complete graph can share
    one gamma. Returns ``(matrix, edges)``.
    """
    peers = sorted(peer_ids)
    edges = _neighbours(topology, peers)
    index = {peer: position for position, peer in enumerate(peers)}
    size = len(peers)

    degree: Dict[PeerId, int] = {peer: 0 for peer in peers}
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    max_degree = max(degree.values())
    weight = float(gamma) / max_degree

    matrix = [[0.0] * size for _ in range(size)]
    for a, b in edges:
        matrix[index[a]][index[b]] = weight
        matrix[index[b]][index[a]] = weight
    for peer in peers:
        position = index[peer]
        matrix[position][position] = 1.0 - weight * degree[peer]

    return tuple(tuple(row) for row in matrix), tuple(sorted(edges))


def normalise_explicit_q(rows, peer_ids: Sequence[PeerId]):
    peers = sorted(peer_ids)
    if len(rows) != len(peers):
        raise MixingMatrixError(
            f"explicit Q has {len(rows)} rows but there are {len(peers)} peers")
    matrix = tuple(tuple(float(v) for v in row) for row in rows)
    edges = []
    for i, peer_a in enumerate(peers):
        for j in range(i + 1, len(peers)):
            if abs(matrix[i][j]) > TOLERANCE:
                edges.append((peer_a, peers[j]))
    return matrix, tuple(edges)


def validate_q(matrix, peer_ids: Sequence[PeerId]) -> None:
    """Square, symmetric, non-negative, doubly stochastic, and connected."""
    peers = sorted(peer_ids)
    size = len(peers)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise MixingMatrixError(f"Q must be {size}x{size}")

    for i in range(size):
        for j in range(size):
            if matrix[i][j] < -TOLERANCE:
                raise MixingMatrixError(
                    f"Q has a negative weight at ({i},{j}): {matrix[i][j]}")
            if abs(matrix[i][j] - matrix[j][i]) > TOLERANCE:
                raise MixingMatrixError(
                    f"Q is not symmetric at ({i},{j}): "
                    f"{matrix[i][j]} vs {matrix[j][i]}")

    for i in range(size):
        row = sum(matrix[i])
        column = sum(matrix[r][i] for r in range(size))
        if abs(row - 1.0) > TOLERANCE:
            raise MixingMatrixError(f"Q row {i} sums to {row}, not 1")
        if abs(column - 1.0) > TOLERANCE:
            raise MixingMatrixError(f"Q column {i} sums to {column}, not 1")

    # Connectivity: mixing over a disconnected Q cannot reach consensus.
    adjacency = {i: {j for j in range(size) if i != j and abs(matrix[i][j]) > TOLERANCE}
                 for i in range(size)}
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbour in adjacency[node]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    if len(seen) != size:
        raise MixingMatrixError(
            f"Q is not connected: {len(seen)} of {size} peers are reachable from the "
            f"first. A disconnected fixed graph cannot reach consensus.")


def q_digest(matrix, peer_ids: Sequence[PeerId]) -> str:
    """Stable identity for the manifest, rounded so float noise cannot change it."""
    payload = {
        "peers": [int(p) for p in sorted(peer_ids)],
        "matrix": [[round(float(v), 12) for v in row] for row in matrix],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def resolve_fixed_q(spec, peer_ids: Sequence[PeerId], gamma: float):
    """Build Q from config: a named topology string, or an explicit matrix.

    Returns ``(matrix, edges, digest, source)``.
    """
    peers = sorted(peer_ids)
    if isinstance(spec, str) and spec in NAMED_TOPOLOGIES:
        matrix, edges = build_named_q(spec, peers, gamma)
        source = f"topology:{spec}:gamma={gamma}"
    elif isinstance(spec, str) and spec:
        # An explicit matrix in the str option: a JSON literal or a file path.
        rows, declared, source = parse_explicit_q(spec)
        if declared is not None and sorted(int(p) for p in declared) != [int(p) for p in peers]:
            raise MixingMatrixError(
                f"the fixed matrix declares peers {sorted(declared)} but this run's "
                f"peers are {list(peers)}")
        matrix, edges = normalise_explicit_q(rows, peers)
    elif spec:
        matrix, edges = normalise_explicit_q(spec, peers)
        source = "explicit"
    else:
        raise MixingMatrixError(
            "a fixed mixing matrix was requested but none was supplied: expected "
            f"either a topology name {list(NAMED_TOPOLOGIES)} or an explicit matrix")

    validate_q(matrix, peers)
    return matrix, edges, q_digest(matrix, peers), source
