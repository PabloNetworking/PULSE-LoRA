"""Compile one offered opportunity frame into interaction events.

The compilers are deterministic and must not change any exogenous fact: the
same frame always yields the same endpoints, order, capacities, durations,
transport labels, and mixing coefficient, whichever interaction form is used.
That equality is what makes a cross-method comparison meaningful.
"""
from typing import Dict, Iterable, List, Sequence, Tuple

from federatedscope.contrib.decentralized.core.types import (
    GraphEdgeInfo,
    GraphMixingEvent,
    OpportunityFrame,
    PairSessionEvent,
    PeerId,
)

#: How a graph round's ``W_t`` is obtained when the method declares no fixed matrix
#: (``decentralized.execution.graph_mixing``). ``pair_block`` is the matching-derived
#: pair-block matrix at ``comparison.gamma`` (degree one only). ``dynamic_metropolis``
#: is a per-frame renormalised ``W_t`` over the frame's available edges at any degree,
#: the time-varying mixing of TAD-LoRA and D-PSGD.
GRAPH_MIXING_PAIR_BLOCK = "pair_block"
GRAPH_MIXING_DYNAMIC = "dynamic_metropolis"
GRAPH_MIXING_MODES = (GRAPH_MIXING_PAIR_BLOCK, GRAPH_MIXING_DYNAMIC)

#: The ``dynamic_metropolis`` rule, recorded in the manifest of every run that uses it.
#: Metropolis-Hastings weights over the frame's available edges: an edge (i, j) gets
#: 1 / (1 + max(d_i, d_j)) where d is the peer's degree in this frame, and every peer
#: keeps the remainder of its row on the diagonal. Symmetric by construction, every
#: off-diagonal is non-negative, and the diagonal is at least 1 / (1 + d_i) > 0, so
#: the matrix is doubly stochastic at any degree without a gamma to tune.
METROPOLIS_RULE = "w_ij = 1/(1+max(d_i,d_j)) on an available edge; w_ii = 1 - sum_j w_ij"


class CompilationError(ValueError):
    pass


def compile_pair_sessions(frame: OpportunityFrame, first_event_id: int = 0
                          ) -> Tuple[PairSessionEvent, ...]:
    """One bilateral session per available offered edge, in canonical order."""
    events: List[PairSessionEvent] = []
    event_id = first_event_id
    for opportunity in frame.opportunities:
        if not opportunity.available:
            continue
        events.append(PairSessionEvent(
            event_id=event_id,
            time_us=frame.time_us,
            frame_index=frame.index,
            endpoints=opportunity.endpoints,
            duration_us=opportunity.duration_us,
            credits=opportunity.gross_credits(),
            transport=opportunity.transport,
        ))
        event_id += 1
    return tuple(events)


def compile_graph_round(frame: OpportunityFrame, peer_ids: Sequence[PeerId],
                        gamma: float, event_id: int = 0) -> GraphMixingEvent:
    """Compile the same frame into a synchronous round with a pair-block ``W_t``.

    For an offered pair (i, j) the block is::

        [[1-gamma, gamma],
         [gamma, 1-gamma]]

    and every unmatched peer keeps self-weight one. The result is symmetric and
    doubly stochastic, so it is mean-preserving.
    """
    if not 0.0 < gamma <= 0.5:
        raise CompilationError(f"gamma must satisfy 0 < gamma <= 0.5, got {gamma}")

    order = tuple(sorted(peer_ids))
    index = {peer: position for position, peer in enumerate(order)}
    size = len(order)
    matrix = [[0.0] * size for _ in range(size)]
    for position in range(size):
        matrix[position][position] = 1.0

    edges: List[GraphEdgeInfo] = []
    touched = set()
    for opportunity in frame.opportunities:
        if not opportunity.available:
            continue
        a, b = opportunity.endpoints
        if a not in index or b not in index:
            raise CompilationError(f"edge {opportunity.endpoints} references an unknown peer")
        if a in touched or b in touched:
            raise CompilationError(
                f"frame {frame.index}: peer reused across edges; a graph round compiled "
                f"from a matching requires disjoint edges"
            )
        touched.add(a)
        touched.add(b)

        ia, ib = index[a], index[b]
        matrix[ia][ia] = 1.0 - gamma
        matrix[ib][ib] = 1.0 - gamma
        matrix[ia][ib] = gamma
        matrix[ib][ia] = gamma

        edges.append(GraphEdgeInfo(
            endpoints=opportunity.endpoints,
            duration_us=opportunity.duration_us,
            credits=opportunity.gross_credits(),
            transport=opportunity.transport,
        ))

    return GraphMixingEvent(
        event_id=event_id,
        time_us=frame.time_us,
        frame_index=frame.index,
        peer_order=order,
        matrix=tuple(tuple(row) for row in matrix),
        edges=tuple(edges),
    )


def compile_fixed_graph_round(frame: OpportunityFrame, peer_ids: Sequence[PeerId],
                              matrix, event_id: int = 0,
                              require_complete_frame: bool = False) -> GraphMixingEvent:
    """Attach this frame's offered capacities to a matrix supplied from outside.

    The matrix is used unchanged: it is the method's fixed Q, not something
    derived from the frame. Only per-edge capacity, duration and transport come
    from the trace.

    A nonzero off-diagonal weight whose edge this frame does not offer (absent
    entirely, or ``available=False``) is reported in ``unavailable_edges``, so
    that the executor's skipped-edge policy removes it instead of mixing over a
    link that carries no bytes. With ``require_complete_frame`` such a frame is
    rejected instead: the strict reading, in which the fixed graph is present
    every round.
    """
    order = tuple(sorted(peer_ids))
    index = {peer: position for position, peer in enumerate(order)}

    available: Dict[Tuple[PeerId, PeerId], object] = {}
    for opportunity in frame.opportunities:
        a, b = opportunity.endpoints
        if a not in index or b not in index:
            raise CompilationError(f"edge {opportunity.endpoints} references an unknown peer")
        if opportunity.available:
            available[opportunity.endpoints] = opportunity

    edges: List[GraphEdgeInfo] = []
    unavailable: List[Tuple[PeerId, PeerId]] = []
    for i, peer_a in enumerate(order):
        for j in range(i + 1, len(order)):
            if abs(matrix[i][j]) <= 1e-12:
                continue  # Q does not use this pair
            endpoints = (peer_a, order[j])
            opportunity = available.get(endpoints)
            if opportunity is None:
                unavailable.append(endpoints)
                continue
            edges.append(GraphEdgeInfo(
                endpoints=endpoints,
                duration_us=opportunity.duration_us,
                credits=opportunity.gross_credits(),
                transport=opportunity.transport,
            ))

    if unavailable and require_complete_frame:
        raise CompilationError(
            f"frame {frame.index} offers no available opportunity for fixed-graph "
            f"edges {unavailable}. decentralized.execution.require_complete_graph_frames "
            f"is set, so the incomplete frame is rejected rather than silently mixing "
            f"over links that carry no bytes."
        )

    return GraphMixingEvent(
        event_id=event_id, time_us=frame.time_us, frame_index=frame.index,
        peer_order=order, matrix=tuple(tuple(row) for row in matrix), edges=tuple(edges),
        unavailable_edges=tuple(unavailable),
    )


def metropolis_hastings_matrix(order: Sequence[PeerId],
                               edges: Iterable[Tuple[PeerId, PeerId]]):
    """``W`` over ``edges`` by :data:`METROPOLIS_RULE`, rows and columns in ``order``.

    A pure function of the edge set: the order edges are given in does not change a
    single entry, and an empty edge set is the identity.
    """
    index = {peer: position for position, peer in enumerate(order)}
    size = len(order)
    pairs = sorted({(min(a, b), max(a, b)) for a, b in edges})
    degree: Dict[PeerId, int] = {peer: 0 for peer in order}
    for a, b in pairs:
        degree[a] += 1
        degree[b] += 1
    matrix = [[0.0] * size for _ in range(size)]
    for a, b in pairs:
        weight = 1.0 / (1.0 + max(degree[a], degree[b]))
        matrix[index[a]][index[b]] = weight
        matrix[index[b]][index[a]] = weight
    for position in range(size):
        # The diagonal absorbs whatever the row does not hand to its neighbours - an
        # unavailable edge contributes nothing here, so its weight stays with the peer.
        matrix[position][position] = 1.0 - sum(
            matrix[position][column] for column in range(size) if column != position)
    return tuple(tuple(row) for row in matrix)


def compile_dynamic_graph_round(frame: OpportunityFrame, peer_ids: Sequence[PeerId],
                                event_id: int = 0) -> GraphMixingEvent:
    """Compile this frame's available edges into a renormalised ``W_t`` at any degree.

    Selected by ``decentralized.execution.graph_mixing: dynamic_metropolis``. Unlike
    :func:`compile_graph_round` it accepts a peer on several edges, and unlike
    :func:`compile_fixed_graph_round` the matrix is derived from this frame, so a
    time-varying trace gives a time-varying ``W_t``. The rule is
    :data:`METROPOLIS_RULE`; the result is checked with
    :func:`assert_doubly_stochastic` before it is returned.

    An edge the frame does not offer, or offers as unavailable, is left out of the
    matrix (weight zero, the diagonal keeps the rest of the row), so
    ``unavailable_edges`` is always empty: a matrix built from the frame never names a
    link that carries no bytes.
    """
    order = tuple(sorted(peer_ids))
    index = {peer: position for position, peer in enumerate(order)}

    edges: List[GraphEdgeInfo] = []
    seen = set()
    for opportunity in frame.opportunities:
        a, b = opportunity.endpoints
        if a not in index or b not in index:
            raise CompilationError(f"edge {opportunity.endpoints} references an unknown peer")
        if not opportunity.available:
            continue  # weight zero; the diagonal absorbs it
        key = (min(a, b), max(a, b))
        if key in seen:
            raise CompilationError(
                f"frame {frame.index}: edge {key} is offered twice; one weight per edge")
        seen.add(key)
        edges.append(GraphEdgeInfo(
            endpoints=opportunity.endpoints,
            duration_us=opportunity.duration_us,
            credits=opportunity.gross_credits(),
            transport=opportunity.transport,
        ))

    matrix = metropolis_hastings_matrix(order, [edge.endpoints for edge in edges])
    assert_doubly_stochastic(matrix)
    return GraphMixingEvent(
        event_id=event_id,
        time_us=frame.time_us,
        frame_index=frame.index,
        peer_order=order,
        matrix=matrix,
        edges=tuple(edges),
    )


def assert_doubly_stochastic(matrix, tolerance: float = 1e-12) -> None:
    size = len(matrix)
    for position in range(size):
        row = sum(matrix[position])
        column = sum(matrix[r][position] for r in range(size))
        if abs(row - 1.0) > tolerance:
            raise CompilationError(f"row {position} sums to {row}, not 1")
        if abs(column - 1.0) > tolerance:
            raise CompilationError(f"column {position} sums to {column}, not 1")
    for i in range(size):
        for j in range(size):
            if abs(matrix[i][j] - matrix[j][i]) > tolerance:
                raise CompilationError(f"matrix is not symmetric at ({i},{j})")
