"""The encoder is the sole byte authority, and budgets are conservative."""
import pytest

from federatedscope.contrib.decentralized.core.accounting import (
    CATEGORY_CONTROL,
    CATEGORY_PAYLOAD,
    ByteLedger,
    SessionBudget,
)
from federatedscope.contrib.decentralized.core.types import DirectionalCredits
from federatedscope.contrib.decentralized.core.wire import (
    WIRE_VERSION,
    Frame,
    WireError,
    control_frame,
    decode_frame,
    encode_frame,
    encoded_size,
    payload_frame,
)


@pytest.mark.parametrize("body_len", [0, 1, 17, 1024, 65536])
def test_encoded_size_equals_the_real_encoded_length(body_len):
    frame = payload_frame("unit", 1, 2, 3, b"\x5a" * body_len)
    assert encoded_size(frame) == len(encode_frame(frame))


@pytest.mark.parametrize("body_len", [0, 1, 17, 1024])
def test_ledger_debits_exactly_the_encoded_length(body_len):
    """Every byte the ledger records is a byte of an encoded frame."""
    ledger = ByteLedger()
    frame = payload_frame("unit", 1, 2, 3, b"q" * body_len)
    debited = ledger.debit_frame(frame, CATEGORY_PAYLOAD)
    assert debited == len(encode_frame(frame)) == ledger.total()


def test_frames_round_trip_exactly():
    frame = payload_frame("unit", 4, 9, 77, b"payload-bytes")
    decoded = decode_frame(encode_frame(frame))
    assert decoded.body == frame.body
    assert decoded.kind == frame.kind
    assert (decoded.sender, decoded.receiver) == (4, 9)
    assert decoded.session_id == 77


def test_checksum_detects_corruption():
    raw = bytearray(encode_frame(payload_frame("unit", 1, 2, 3, b"abcdef")))
    raw[-5] ^= 0xFF
    with pytest.raises(WireError, match="checksum mismatch"):
        decode_frame(bytes(raw))


def test_truncated_frame_is_rejected():
    with pytest.raises(WireError):
        decode_frame(b"short")


def test_bad_magic_is_rejected():
    raw = b"XXXX" + encode_frame(control_frame("m", 1, 2, 3))[4:]
    with pytest.raises(WireError, match="bad magic"):
        decode_frame(raw)


def test_unsupported_version_is_rejected():
    raw = bytearray(encode_frame(control_frame("m", 1, 2, 3)))
    raw[4] = WIRE_VERSION + 7
    with pytest.raises(WireError, match="unsupported wire version"):
        decode_frame(bytes(raw))


def test_oversized_kind_is_rejected():
    with pytest.raises(WireError, match="255 bytes"):
        Frame(0, "k" * 300, 1, 2, 3)


# --- budgets ------------------------------------------------------------------

def test_control_is_debited_before_payload_credit_is_computed():
    budget = SessionBudget((1, 2), DirectionalCredits(1000, 1000))
    assert budget.payload_credit() == 1000
    budget.charge_control(1, 100)
    assert budget.payload_credit() == 900


def test_payload_credit_is_the_conservative_minimum_of_both_directions():
    budget = SessionBudget((1, 2), DirectionalCredits(1000, 400))
    budget.charge_control(1, 100)  # forward -> 900
    budget.charge_control(2, 50)   # reverse -> 350
    assert budget.remaining_forward() == 900
    assert budget.remaining_reverse() == 350
    assert budget.payload_credit() == 350


def test_affordability_boundary_is_exact():
    budget = SessionBudget((1, 2), DirectionalCredits(100, 100))
    assert budget.can_afford(100)
    assert not budget.can_afford(101)
    budget.charge_payload(100)
    assert budget.payload_credit() == 0
    assert not budget.can_afford(1)


def test_overspending_raises_rather_than_silently_truncating():
    budget = SessionBudget((1, 2), DirectionalCredits(10, 10))
    with pytest.raises(ValueError, match="exceeds credit"):
        budget.charge_payload(11)


def test_control_can_exhaust_the_budget_entirely():
    budget = SessionBudget((1, 2), DirectionalCredits(50, 50))
    budget.charge_control(1, 80)
    assert budget.remaining_forward() == 0
    assert budget.payload_credit() == 0


def test_ledger_rows_are_joinable_to_their_session_and_transaction():
    """A row with only one id cannot be joined back to what produced it."""
    ledger = ByteLedger()
    ledger.debit_frame(control_frame("manifest", 1, 2, 55), CATEGORY_CONTROL)
    ledger.debit_frame(payload_frame("unit", 1, 2, 55, b"x" * 8), CATEGORY_PAYLOAD,
                       transaction_id=900)

    assert len(ledger.rows_for_session(55)) == 2, "both rows belong to the session"
    assert len(ledger.rows_for_transaction(900)) == 1, "only the payload has a transaction"
    control_row = next(r for r in ledger.records if r["category"] == CATEGORY_CONTROL)
    assert control_row["session_id"] == 55
    assert control_row["transaction_id"] is None, \
        "control precedes any transaction, so it must not claim one"


def test_ledger_is_directional_and_categorised():
    ledger = ByteLedger()
    ledger.debit_frame(control_frame("m", 1, 2, 0), CATEGORY_CONTROL)
    ledger.debit_frame(payload_frame("u", 1, 2, 0, b"x" * 50), CATEGORY_PAYLOAD)
    ledger.debit_frame(payload_frame("u", 2, 1, 0, b"y" * 10), CATEGORY_PAYLOAD)

    assert ledger.sent_by(1, CATEGORY_PAYLOAD) > ledger.sent_by(2, CATEGORY_PAYLOAD)
    assert ledger.received_by(2, CATEGORY_PAYLOAD) == ledger.sent_by(1, CATEGORY_PAYLOAD)
    assert set(ledger.by_direction()) == {(1, 2), (2, 1)}
    assert ledger.total() == ledger.total(CATEGORY_CONTROL) + ledger.total(CATEGORY_PAYLOAD)


def test_negative_debit_is_refused():
    with pytest.raises(ValueError):
        ByteLedger().debit(1, 2, CATEGORY_CONTROL, -1)


def test_ledger_round_trips_through_a_checkpoint():
    ledger = ByteLedger()
    ledger.debit_frame(payload_frame("u", 1, 2, 0, b"z" * 30), CATEGORY_PAYLOAD)
    restored = ByteLedger()
    restored.load_state_dict(ledger.state_dict())
    assert restored.total() == ledger.total()
