"""``socket_v1`` pricing: the size functions equal the encoders, and quotes are pinned.

``socket_v1_quotes.json`` holds quotes of pair-session and graph-edge contexts (both envelopes,
with and without identities, auxiliary state and credits, including messages over the
envelope). Any change to a quote changes which sessions are admitted, so it must be deliberate.
"""
import json
import os

import pytest

from federatedscope.contrib.decentralized.core.wire_cost import (
    OVER_CAP,
    GraphEdgeWireContext,
    SessionWireContext,
    resolve_graph_wire_cost,
    resolve_wire_cost,
)
from federatedscope.contrib.decentralized.simulation import socket_v1
from federatedscope.contrib.decentralized.simulation import wire_quote  # noqa: F401

QUOTES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "socket_v1_quotes.json")
PAYLOAD_KEYS = {"payload_1_0": (1, 0), "payload_6013_1": (6013, 1),
                "payload_40000000_64": (40_000_000, 64), "payload_101000000_200": (101_000_000, 200),
                "payload_600000000_200": (600_000_000, 200)}


def quotes():
    with open(QUOTES, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.mark.parametrize("index", range(9))
def test_pair_session_quotes_are_pinned(index):
    row = quotes()["pair"][index]
    model = resolve_wire_cost("socket_v1", envelope_bytes=row["env"])
    ctx = SessionWireContext.from_dict(row["ctx"])
    assert model.fixed(ctx, row["d"]) == row["fixed"]
    assert model.reserve(ctx, row["d"]) == row["reserve"]
    for key, (logical, units) in PAYLOAD_KEYS.items():
        assert model.payload(ctx, row["d"], logical, units) == row[key], key


@pytest.mark.parametrize("index", range(7))
def test_graph_edge_quotes_are_pinned(index):
    row = quotes()["graph"][index]
    model = resolve_graph_wire_cost("socket_v1", envelope_bytes=row["env"])
    ctx = GraphEdgeWireContext.from_dict(row["ctx"])
    assert model.fixed(ctx, row["d"]) == row["fixed"]
    assert model.reserve(ctx, row["d"]) == row["reserve"]
    assert model.payload(ctx, row["d"], int(ctx.snapshot_bytes[row["d"]]), None) == row["payload"]
    assert model.payload(ctx, row["d"], 700_000_000, None) == row["payload_big"]


def test_a_message_over_the_envelope_is_priced_out_of_reach():
    over = [r for r in quotes()["pair"] if r["payload_600000000_200"] >= OVER_CAP]
    assert over, "the pinned quotes include a payload over the envelope"


@pytest.mark.parametrize("n", [0, 1, 2, 3, 57, 4096, 65537])
def test_the_size_functions_equal_the_encoders(n):
    mid = socket_v1.message_id(12, 3, "fetch")
    payload = bytes(n)
    envelope = socket_v1.Envelope(mid, 1, 2, payload)
    assert socket_v1.request_socket_bytes_for(mid, n) == socket_v1.request_socket_bytes(envelope)
    assert (socket_v1.predicted_reply_socket_bytes_for(mid, 1, 2, 2, n)
            == socket_v1.predicted_reply_socket_bytes(envelope, 2, payload))


def test_the_session_reserve_covers_the_abort_and_the_completion_paths():
    for pairwise in (True, False):
        reserve = socket_v1.session_reserve(7, 1, 2, pairwise, True)
        assert reserve["forward"] > 0 and reserve["reverse"] > 0


def test_a_value_the_protocol_cannot_carry_is_refused():
    with pytest.raises(socket_v1.EncodingError):
        socket_v1.encode_call("hook", {"bad": object()})
