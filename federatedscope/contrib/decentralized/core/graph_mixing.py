"""Synchronous snapshot-route-mix-commit executor for graph rounds.

Every participant's pre-round state is snapshotted *once*, before any mixing,
and each peer's update is computed from those same pre-round snapshots.
Sequential in-place pair updates are a different algorithm and are not used here.

Offered and realized matrices are distinguished: an edge that cannot carry the
method's complete payload, or a fixed-Q edge the frame does not offer, is removed
from the realized matrix by the skipped-edge policy, while the offered matrix and
the trace are untouched.
"""
from typing import Dict, Mapping

from federatedscope.contrib.decentralized.core.accounting import (
    CATEGORY_CONTROL,
    CATEGORY_PAYLOAD,
    ByteLedger,
    SessionBudget,
)
from federatedscope.contrib.decentralized.core.types import GraphMixingEvent, PeerId
from federatedscope.contrib.decentralized.core.wire import (
    control_frame,
    encoded_size,
    payload_frame,
)
from federatedscope.contrib.decentralized.core.wire_cost import (
    DIRECTIONS as WIRE_DIRECTIONS,
    FORWARD,
    REVERSE,
    GraphEdgeWireContext,
    LogicalWireCost,
)


#: What happens to a neighbour's weight when its edge cannot carry the payload.
#:
#: ``self_weight``    - the skipped edge is removed symmetrically and its weight is
#:                      reassigned to both endpoints' diagonals. The realized matrix
#:                      stays symmetric and doubly stochastic, so the round remains
#:                      mean-preserving; an unreachable neighbour simply contributes
#:                      nothing and the peer keeps that much of its own state.
#: ``identity_round`` - if any edge of the round is skipped, the whole round becomes
#:                      the identity. Stricter, and appropriate when a partial round
#:                      would not correspond to the method's published update.
SKIPPED_EDGE_SELF_WEIGHT = "self_weight"
SKIPPED_EDGE_IDENTITY_ROUND = "identity_round"
SKIPPED_EDGE_POLICIES = (SKIPPED_EDGE_SELF_WEIGHT, SKIPPED_EDGE_IDENTITY_ROUND)

_STOCHASTIC_TOLERANCE = 1e-9


class GraphMixingError(ValueError):
    pass


def _assert_realized_is_valid(matrix, peer_order) -> None:
    """Check that every row and column of the realized matrix sums to one.

    A matrix that is not doubly stochastic rescales peer state: at degree > 1 (a
    fixed Q, or dynamic mixing), dropping one edge of several without reassigning
    its weight leaves a row summing to less than one, and that peer shrinks its
    own adapter.
    """
    size = len(matrix)
    for position in range(size):
        row = sum(matrix[position])
        if abs(row - 1.0) > _STOCHASTIC_TOLERANCE:
            raise GraphMixingError(
                f"realized matrix row for peer {peer_order[position]} sums to {row}, "
                f"not 1; the skipped-edge policy left it invalid")
        column = sum(matrix[r][position] for r in range(size))
        if abs(column - 1.0) > _STOCHASTIC_TOLERANCE:
            raise GraphMixingError(
                f"realized matrix column for peer {peer_order[position]} sums to "
                f"{column}, not 1")


def _optional(algorithm, name):
    return getattr(algorithm, name, None)


def _edge_credit(edge) -> Dict[str, int]:
    """The trace's credit of one edge, per direction."""
    return {FORWARD: int(edge.credits.forward_bytes), REVERSE: int(edge.credits.reverse_bytes)}


def execute_graph_round(event: GraphMixingEvent, algorithms: Mapping[PeerId, object],
                        ledger: ByteLedger, checksum: bool = True,
                        skipped_edge_policy: str = SKIPPED_EDGE_SELF_WEIGHT,
                        wire_cost=None) -> dict:
    """Run one graph round and return its record. ``algorithms`` maps peer id -> GraphAlgorithm.

    ``wire_cost`` is the wire-cost model admission is priced in: the logical model when None,
    under which admission is exactly the ledger's ``can_afford``. An edge whose snapshot
    exchange fits the ledger but not the wire is dropped at planning, before a payload row is
    debited, and the reason names the direction and the model. The record's
    ``wire_contexts`` holds the context of every edge priced on the wire, so the round can be
    priced again from the log.
    """
    # 1. Barrier: one pre-round snapshot per participant, taken before anything mixes.
    snapshots: Dict[PeerId, bytes] = {
        peer_id: algorithms[peer_id].snapshot()
        for peer_id in event.peer_order
        if peer_id in algorithms
    }

    if skipped_edge_policy not in SKIPPED_EDGE_POLICIES:
        raise GraphMixingError(
            f"unknown skipped-edge policy {skipped_edge_policy!r}; expected one of "
            f"{list(SKIPPED_EDGE_POLICIES)}")

    offered = [list(row) for row in event.matrix]
    # Start from the offered matrix and remove what cannot be carried, rather than
    # starting from identity and adding back only matched blocks: the latter is
    # only correct when every peer has degree one.
    realized = [list(row) for row in event.matrix]
    index = {peer: position for position, peer in enumerate(event.peer_order)}

    # Directional totals use the same convention as a pair session: "forward" is
    # endpoints[0] -> endpoints[1] on each edge. The run summary adds them to the
    # pair sessions' directional totals.
    control_bytes = 0
    payload_bytes = 0
    auxiliary_bytes = 0
    directional = {"control_forward": 0, "control_reverse": 0,
                   "payload_forward": 0, "payload_reverse": 0,
                   "auxiliary_forward": 0, "auxiliary_reverse": 0}
    affordable_edges = []
    # The wire context of every edge that was priced, so the round can be priced again from
    # the log.
    wire_contexts = []
    skipped = []

    def drop_edge(endpoints, reason, needed_bytes=None, payload_credit=0):
        """Remove an edge symmetrically and hand its weight to both diagonals."""
        a_, b_ = endpoints
        ia_, ib_ = index[a_], index[b_]
        weight = offered[ia_][ib_]
        skipped.append({
            "endpoints": list(endpoints),
            "reason": reason,
            "needed_bytes": needed_bytes,
            "payload_credit": payload_credit,
            "reassigned_weight": weight,
        })
        realized[ia_][ib_] = 0.0
        realized[ib_][ia_] = 0.0
        realized[ia_][ia_] += weight
        realized[ib_][ib_] += weight

    # A nonzero Q edge the frame never offered carries no bytes, so its weight
    # must not survive into the realized matrix. Resolved by the same policy as an
    # edge that was offered but turned out unaffordable.
    for endpoints in getattr(event, "unavailable_edges", ()):
        drop_edge(endpoints, "no available opportunity for this edge in the frame")

    # 2. Per edge: charge control, then test affordability of the complete payload.
    for edge in event.edges:
        a, b = edge.endpoints
        budget = SessionBudget(edge.endpoints, edge.credits)

        manifest_sizes: Dict[PeerId, int] = {}
        for sender, receiver in ((a, b), (b, a)):
            frame = control_frame("graph_manifest", sender, receiver, event.event_id,
                                  checksum=checksum)
            size = ledger.debit_frame(frame, CATEGORY_CONTROL)
            budget.charge_control(sender, size)
            manifest_sizes[sender] = size
            control_bytes += size
            directional["control_forward" if sender == a else "control_reverse"] += size

        frame_ab = payload_frame("graph_snapshot", a, b, event.event_id, snapshots[a],
                                 checksum=checksum)
        frame_ba = payload_frame("graph_snapshot", b, a, event.event_id, snapshots[b],
                                 checksum=checksum)
        need = max(encoded_size(frame_ab), encoded_size(frame_ba))

        if not budget.can_afford(need):
            drop_edge(edge.endpoints,
                      "complete snapshot does not fit after control overhead",
                      needed_bytes=need, payload_credit=budget.payload_credit())
            continue

        # The wire view for this edge, priced once both manifest frames are debited and
        # both snapshot frames are measured, by the run's wire-cost model. Under the
        # logical model (the identity) this never refuses what can_afford admitted. The
        # ledger above stays logical; this decides whether the exchange fits the credit
        # in serialized bytes, and drops the edge before any payload is debited.
        size_ab, size_ba = encoded_size(frame_ab), encoded_size(frame_ba)
        identity_bound = 0
        for endpoint in edge.endpoints:
            bound = _optional(algorithms.get(endpoint), "identity_bound")
            if bound is not None:
                identity_bound = max(identity_bound, int(bound()))
        model = wire_cost if wire_cost is not None else LogicalWireCost()
        budget.attach_wire(
            model,
            GraphEdgeWireContext(
                lo=a, hi=b, event_id=event.event_id, checksum=checksum,
                manifest_bytes={FORWARD: manifest_sizes[a], REVERSE: manifest_sizes[b]},
                snapshot_bytes={FORWARD: size_ab, REVERSE: size_ba},
                gross_bytes=_edge_credit(edge),
                # Presence only: the wire-cost model needs to know whether an endpoint
                # sends auxiliary state, not how much.
                auxiliary_present={
                    FORWARD: _optional(algorithms.get(a), "auxiliary_payload_bytes") is not None,
                    REVERSE: _optional(algorithms.get(b), "auxiliary_payload_bytes") is not None},
                identity_width=len(str(identity_bound - 1)) if identity_bound else 0,
                identity_count_bound=identity_bound,
                envelope_bytes=int(model.envelope_bytes)))
        wire_contexts.append(budget.wire_context.as_dict())
        if not budget.wire_fits(size_ab, size_ba):
            quote = budget.wire_quote(size_ab, size_ba)
            tight = [d for d in WIRE_DIRECTIONS if not quote[d]["fits"]][0]
            if quote[tight].get("over_cap"):
                drop_edge(edge.endpoints,
                          f"complete snapshot of {need}B cannot be carried: a message of the "
                          f"edge exceeds the {budget.wire.envelope_bytes}-byte envelope in "
                          f"force; priced by the {budget.wire.name} model",
                          needed_bytes=need, payload_credit=budget.payload_credit())
                continue
            drop_edge(edge.endpoints,
                      f"complete snapshot of {need}B needs {quote[tight]['total']}B on the "
                      f"wire {tight} ({quote[tight]['fixed']}B control, "
                      f"{quote[tight]['payload']}B payload, {quote[tight]['reserve']}B "
                      f"reserve) but the edge offers {quote[tight]['offered']}B; priced by "
                      f"the {budget.wire.name} model",
                      needed_bytes=need, payload_credit=budget.payload_credit())
            continue
        forward_size = ledger.debit_frame(frame_ab, CATEGORY_PAYLOAD)
        reverse_size = ledger.debit_frame(frame_ba, CATEGORY_PAYLOAD)
        payload_bytes += forward_size + reverse_size
        directional["payload_forward"] += forward_size
        directional["payload_reverse"] += reverse_size
        budget.charge_payload(need)

        for endpoint, direction in ((a, "auxiliary_forward"), (b, "auxiliary_reverse")):
            if endpoint not in algorithms:
                continue
            hook = getattr(algorithms[endpoint], "auxiliary_payload_bytes", None)
            size = hook() if hook else 0
            auxiliary_bytes += size
            directional[direction] += size

        affordable_edges.append(edge.endpoints)

    if skipped and skipped_edge_policy == SKIPPED_EDGE_IDENTITY_ROUND:
        realized = [[1.0 if i == j else 0.0 for j in range(len(event.peer_order))]
                    for i in range(len(event.peer_order))]

    _assert_realized_is_valid(realized, event.peer_order)

    # 3. Stage every peer from the same pre-round snapshots, then commit together.
    tokens: Dict[PeerId, object] = {}
    status = "failed"
    try:
        for peer_id in event.peer_order:
            if peer_id not in algorithms:
                continue
            position = index[peer_id]
            weights = {
                other: realized[position][index[other]]
                for other in event.peer_order
                if realized[position][index[other]] != 0.0
            }
            if set(weights) == {peer_id}:
                continue  # self-weight one: nothing routed, nothing to stage
            tokens[peer_id] = algorithms[peer_id].stage_mix(weights, snapshots)
        for peer_id, token in tokens.items():
            algorithms[peer_id].commit(token)
        status = "committed"
    except Exception:
        for peer_id, token in tokens.items():
            algorithms[peer_id].rollback(token)
        raise

    # The round committed, so the held snapshots can be released.
    for peer_id, token in tokens.items():
        discard = getattr(algorithms[peer_id], "discard", None)
        if callable(discard):
            discard(token)

    return {
        "event": "graph_round",
        "frame_index": event.frame_index,
        "time_us": event.time_us,
        "status": status,
        "offered_matrix": [list(row) for row in event.matrix],
        "realized_matrix": realized,
        "offered_edges": [list(e.endpoints) for e in event.edges],
        "affordable_edges": [list(e) for e in affordable_edges],
        "skipped_edges": skipped,
        "skipped_edge_policy": skipped_edge_policy,
        "control_bytes": control_bytes,
        "control_bytes_forward": directional["control_forward"],
        "control_bytes_reverse": directional["control_reverse"],
        "payload_bytes": payload_bytes,
        "payload_bytes_forward": directional["payload_forward"],
        "payload_bytes_reverse": directional["payload_reverse"],
        "auxiliary_bytes": auxiliary_bytes,
        "auxiliary_bytes_forward": directional["auxiliary_forward"],
        "auxiliary_bytes_reverse": directional["auxiliary_reverse"],
        "mixed_peers": sorted(tokens),
        "wire_contexts": wire_contexts,
    }
