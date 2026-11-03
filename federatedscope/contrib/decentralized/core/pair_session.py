"""Bilateral pair-session executor.

Runs one offered contact through a four-stage negotiation, affordability checks,
the generic transaction engine, and an explicit acknowledgement. It never
decodes a manifest or a payload: both are opaque bytes whose *encoded length* is
what the ledger charges.

Byte order of operations matters and is fixed here: every control frame the
session will send - both manifests and both acknowledgements - is encoded and
debited before payload capacity is computed, so the payload credit is what
remains after the encoded control overhead. A payload must then fit both that
credit and the run's wire-cost model, which prices the session in serialized
bytes; the session record carries the wire context, so the session can be
priced again from the log.
"""
from typing import Mapping, Optional

from federatedscope.contrib.decentralized.core.accounting import (
    CATEGORY_CONTROL,
    CATEGORY_PAYLOAD,
    ByteLedger,
    SessionBudget,
)
from federatedscope.contrib.decentralized.core.transaction import (
    Applied,
    Commit,
    Failed,
    Finish,
    Rollback,
    Stage,
    Staged,
    TransactionEngine,
)
from federatedscope.contrib.decentralized.core.types import (
    STATUS_COMMITTED,
    STATUS_CONTROL_EXCEEDS_CAPACITY,
    STATUS_CONTROL_ONLY,
    STATUS_FAILED,
    PairSessionEvent,
    PeerId,
)
from federatedscope.contrib.decentralized.core.wire import (
    control_frame,
    encoded_size,
    payload_frame,
)
from federatedscope.contrib.decentralized.core.wire_cost import (
    DIRECTIONS as WIRE_DIRECTIONS,
    FORWARD,
    REVERSE,
    LogicalWireCost,
    SessionWireContext,
)

#: Acknowledgement body: a single status byte. Fixed width so the frame can be
#: budgeted up front, before the outcome that determines its value is known.
ACK_COMMITTED = b"\x01"
ACK_FAILED = b"\x00"



def _optional(algorithm, name):
    return getattr(algorithm, name, None)


def execute_pair_session(event: PairSessionEvent, algorithms: Mapping[PeerId, object],
                         engine: TransactionEngine, ledger: ByteLedger,
                         checksum: bool = True, failure_injector=None,
                         wire_cost=None) -> dict:
    """Run one pair session between ``event.endpoints`` and return its record.

    ``wire_cost`` is the wire-cost model admission is priced in (the logical model when
    None) against the trace's credit for the session, per direction. The record holds
    the status and reason, bytes per category and direction, the wire context once the
    handshake fits the credit, and on commit each endpoint's declared plan."""
    a, b = event.endpoints
    budget = SessionBudget(event.endpoints, event.credits)
    credit = {FORWARD: int(event.credits.forward_bytes),
              REVERSE: int(event.credits.reverse_bytes)}

    record = {
        "event": "pair_session",
        "event_id": event.event_id,
        # Every frame and every ledger row of this session carries this id.
        "session_id": event.event_id,
        "transaction_id": None,
        "frame_index": event.frame_index,
        "time_us": event.time_us,
        "endpoints": [a, b],
        "transport": event.transport,
        "duration_us": event.duration_us,
        "offered_forward_bytes": event.credits.forward_bytes,
        "offered_reverse_bytes": event.credits.reverse_bytes,
        "control_bytes": 0,
        "control_bytes_forward": 0,
        "control_bytes_reverse": 0,
        "payload_bytes": 0,
        "payload_bytes_forward": 0,
        "payload_bytes_reverse": 0,
        "auxiliary_bytes": 0,
        "auxiliary_bytes_forward": 0,
        "auxiliary_bytes_reverse": 0,
        "status": STATUS_CONTROL_ONLY,
        "reason": "",
    }

    def charge_control(frame):
        size = ledger.debit_frame(frame, CATEGORY_CONTROL)
        budget.charge_control(frame.sender, size)
        record["control_bytes"] += size
        key = "control_bytes_forward" if frame.sender == a else "control_bytes_reverse"
        record[key] += size
        return size

    # --- stage 1: each endpoint declares what it has ------------------------
    manifests = {}
    manifest_bytes = {}
    for sender, receiver in ((a, b), (b, a)):
        build = _optional(algorithms[sender], "build_manifest")
        body = build(event) if build is not None else b""
        manifests[sender] = body
        manifest_bytes[sender] = charge_control(
            control_frame("manifest", sender, receiver, event.event_id, body,
                          checksum=checksum))

    # Acknowledgements are budgeted now, not after the fact: a session must not
    # spend payload credit it will need to close the transaction.
    ack_frames = {
        sender: control_frame("ack", sender, receiver, event.event_id,
                              ACK_FAILED, checksum=checksum)
        for sender, receiver in ((a, b), (b, a))
    }
    for frame in ack_frames.values():
        charge_control(frame)

    if budget.control_overspent():
        record["status"] = STATUS_CONTROL_EXCEEDS_CAPACITY
        record["reason"] = (
            f"the handshake needs {budget.control_bytes}B but the session offers "
            f"{event.credits.forward_bytes}/{event.credits.reverse_bytes}B; short by "
            f"{budget.control_deficit()}B"
        )
        _acknowledge(algorithms, event, committed=False, reason=record["reason"])
        return record

    record["offered_payload_credit"] = budget.payload_credit()

    # The wire view. Priced here, once both manifests and both acknowledgements are
    # known, by the run's wire-cost model (the logical model unless another was handed
    # in). The ledger above stays logical; this decides whether a payload fits. Both
    # endpoints see the same context, so both plan the same number of components.
    identity_bound = 0
    for endpoint in (a, b):
        bound = _optional(algorithms[endpoint], "identity_bound")
        if bound is not None:
            identity_bound = max(identity_bound, int(bound()))
    model = wire_cost if wire_cost is not None else LogicalWireCost()
    budget.attach_wire(
        model,
        SessionWireContext(
            lo=a, hi=b, event_id=event.event_id, checksum=checksum,
            manifest_bytes={FORWARD: manifest_bytes[a], REVERSE: manifest_bytes[b]},
            ack_bytes={FORWARD: encoded_size(ack_frames[a]),
                       REVERSE: encoded_size(ack_frames[b])},
            gross_bytes=dict(credit),
            # Presence only: the wire-cost model needs to know whether an endpoint
            # sends auxiliary state, not how much.
            auxiliary_present={
                FORWARD: _optional(algorithms[a], "auxiliary_payload_bytes") is not None,
                REVERSE: _optional(algorithms[b], "auxiliary_payload_bytes") is not None},
            identity_width=len(str(identity_bound - 1)) if identity_bound else 0,
            identity_count_bound=identity_bound,
            envelope_bytes=int(model.envelope_bytes)))
    # The record carries the context, so the session can be priced again from the log.
    record["wire_context"] = budget.wire_context.as_dict()

    # --- stage 2: each endpoint reads the other's declaration ----------------
    remote_views = {}
    for me, other in ((a, b), (b, a)):
        consume = _optional(algorithms[me], "consume_manifest")
        try:
            remote_views[me] = consume(manifests[other], event) if consume else None
        except Exception as exc:  # noqa: BLE001 - a rejected peer is a normal outcome
            record["reason"] = f"peer {me} rejected the remote manifest: {exc}"
            _acknowledge(algorithms, event, committed=False, reason=record["reason"])
            return record

    # --- stage 3: each endpoint plans against local state, remote view, budget --
    offers = {}
    for me in (a, b):
        plan = _optional(algorithms[me], "plan_payload")
        offers[me] = plan(event, budget, remote_views[me]) if plan else None

    if offers[a] is None or offers[b] is None:
        record["reason"] = record["reason"] or "an endpoint planned no payload"
        _acknowledge(algorithms, event, committed=False, reason=record["reason"])
        return record

    frame_ab = payload_frame("unit", a, b, event.event_id, offers[a], checksum=checksum)
    frame_ba = payload_frame("unit", b, a, event.event_id, offers[b], checksum=checksum)
    need = max(encoded_size(frame_ab), encoded_size(frame_ba))
    record["payload_needed_bytes"] = need

    if not budget.can_afford(need):
        record["reason"] = (
            f"planned payload needs {need}B but only {budget.payload_credit()}B "
            f"remains after control overhead"
        )
        _acknowledge(algorithms, event, committed=False, reason=record["reason"])
        return record

    # The same plan, priced by the wire-cost model. A planner that used
    # budget.affordable never trips this; a method that plans a fixed payload (a whole
    # adapter) can be refused here even though the ledger credit admitted it.
    units = None
    for endpoint in (a, b):
        carried = _optional(algorithms[endpoint], "payload_identities")
        identities = carried(event) if carried else None
        if identities is not None:
            units = max(units or 0, len(identities))
    if not budget.wire_fits(encoded_size(frame_ab), encoded_size(frame_ba), units=units):
        quote = budget.wire_quote(encoded_size(frame_ab), encoded_size(frame_ba), units=units)
        tight = [d for d in WIRE_DIRECTIONS if not quote[d]["fits"]][0]
        if quote[tight].get("over_cap"):
            record["reason"] = (
                f"planned payload of {need}B cannot be carried: a message of the session "
                f"exceeds the {budget.wire.envelope_bytes}-byte envelope in force; priced by "
                f"the {budget.wire.name} model")
        else:
            record["reason"] = (
                f"planned payload of {need}B needs {quote[tight]['total']}B on the wire "
                f"{tight} ({quote[tight]['fixed']}B control, {quote[tight]['payload']}B payload, "
                f"{quote[tight]['reserve']}B reserve) but the session offers "
                f"{quote[tight]['offered']}B; priced by the {budget.wire.name} model"
            )
        _acknowledge(algorithms, event, committed=False, reason=record["reason"])
        return record

    # The transaction is opened here, before any payload leaves, so payload rows
    # can be stamped with it. Control traffic above legitimately has no
    # transaction id: on a control-only session no transaction is ever opened.
    state = engine.begin(event.endpoints)
    record["transaction_id"] = state.transaction_id

    # Bytes are spent the moment frames go out; a later rollback cannot un-send them.
    #
    # Each payload row is stamped with the identities its frame carried, as the
    # sending endpoint reports them (``payload_identities``; the runtime decodes
    # nothing). Sizes cannot distinguish one component from another, so without
    # this a schedule that starves a component would look the same as one that
    # serves it every session.
    for frame, sender, direction in ((frame_ab, a, "forward"),
                                     (frame_ba, b, "reverse")):
        carried = _optional(algorithms[sender], "payload_identities")
        size = ledger.debit_frame(frame, CATEGORY_PAYLOAD,
                                  transaction_id=state.transaction_id,
                                  identities=carried(event) if carried else None)
        record["payload_bytes"] += size
        record[f"payload_bytes_{direction}"] += size
    budget.charge_payload(need)

    for endpoint, direction in ((a, "forward"), (b, "reverse")):
        auxiliary = _optional(algorithms[endpoint], "auxiliary_payload_bytes")
        size = auxiliary() if auxiliary else 0
        record["auxiliary_bytes"] += size
        record[f"auxiliary_bytes_{direction}"] += size

    # --- stage 4: transaction, then an explicit acknowledgement ----------------
    incoming = {a: offers[b], b: offers[a]}
    action = engine.step(state)
    outcome = None

    while True:
        if isinstance(action, Finish):
            for peer_id, token in state.tokens.items():
                discard = _optional(algorithms[peer_id], "discard")
                if discard:
                    discard(token)
            record["status"] = (
                STATUS_COMMITTED if action.committed else STATUS_FAILED)
            record["reason"] = action.reason or record["reason"]

            # On commit, record what each endpoint planned (``planning_identities``)
            # in the session record. It is asked for here, while the plan is still
            # live and before ``_acknowledge`` lets the method fold it into its own
            # state; the identities come from the method, since core cannot decode a
            # method's payload.
            if action.committed:
                declared = {}
                for endpoint in (a, b):
                    identities = _optional(algorithms[endpoint],
                                           "planning_identities")
                    if identities is None:
                        continue
                    declared[str(endpoint)] = identities(event)
                if declared:
                    record["planning"] = declared

                # What each receiver decoded from its partner's payload
                # (``received_identities``), recorded against the ledger row that
                # carried the frame.
                for receiver, sender in ((a, b), (b, a)):
                    observed = _optional(algorithms[receiver], "received_identities")
                    if observed is None:
                        continue
                    ledger.observe_identities(event.event_id, sender, receiver,
                                              CATEGORY_PAYLOAD, observed(event))

            # The acknowledgement frames were debited up front with a placeholder
            # status. Rebuild them with the real status and check that the encoded
            # width is unchanged.
            for sender, receiver in ((a, b), (b, a)):
                body = ACK_COMMITTED if action.committed else ACK_FAILED
                sent = control_frame("ack", sender, receiver, event.event_id, body,
                                     checksum=checksum)
                assert encoded_size(sent) == encoded_size(ack_frames[sender])
            _acknowledge(algorithms, event, action.committed, record["reason"])
            return record

        peer_id = action.peer_id
        try:
            if isinstance(action, Stage):
                if failure_injector is not None and failure_injector.should_fail("stage"):
                    raise RuntimeError("injected failure at stage")
                outcome = Staged(peer_id, algorithms[peer_id].stage(incoming[peer_id], event))
            elif isinstance(action, Commit):
                if failure_injector is not None and failure_injector.should_fail("commit"):
                    raise RuntimeError("injected failure at commit")
                algorithms[peer_id].commit(action.token)
                outcome = Applied(peer_id)
            elif isinstance(action, Rollback):
                algorithms[peer_id].rollback(action.token)
                outcome = Applied(peer_id)
            else:
                raise RuntimeError(f"unknown transaction action {action!r}")
        except Exception as exc:  # noqa: BLE001 - reported to the transaction engine
            outcome = Failed(peer_id, str(exc))

        action = engine.step(state, outcome)


def _acknowledge(algorithms, event, committed: bool, reason: str = "") -> None:
    """Call ``acknowledge`` on the two endpoints of this session, and only on them.

    Other peers took no part in the session, so their counters and per-edge
    state must not move.
    """
    for peer_id in event.endpoints:
        algorithm = algorithms.get(peer_id)
        if algorithm is None:
            continue
        acknowledge = _optional(algorithm, "acknowledge")
        if acknowledge:
            acknowledge(event, committed, reason)
