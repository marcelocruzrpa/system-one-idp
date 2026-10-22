"""Three-way match: supplier and currency, quantities summed per PO line, and nothing skipped silently."""

from idp.clients.jev import JevResult
from idp.config import schema as load_schema
from idp.erp import Erp
from idp.validate.match import norm_currency, three_way_match
from tests.test_rules import ext, failed

VENDORS = [{"vendor_id": "V1", "name": "Northwind Supply Co.", "tax_id": "77-0421953"},
           {"vendor_id": "V2", "name": "Bluefin Office Supplies LLC"}]


def erp(*, received=(2, 4), invoiced=0, receipts=1) -> Erp:
    """PO-1 from Northwind in USD: 2 servos at 18.50 and 4 drivers at 6.25."""
    po = {"po_number": "PO-1", "vendor_id": "V1", "vendor": "Northwind Supply Co.", "currency": "USD",
          "lines": [{"line": 1, "description": "Servo motor", "quantity": 2, "unit_price": 18.5, "invoiced": invoiced},
                    {"line": 2, "description": "Stepper driver", "quantity": 4, "unit_price": 6.25}]}
    # The received quantities are split evenly over `receipts` goods receipts.
    grs = [{"po_number": "PO-1", "lines": [{"line": i, "received": q / receipts} for i, q in enumerate(received, start=1)]}
           for _ in range(receipts)] if received else []
    return Erp({"vendors": VENDORS, "purchase_orders": [po], "goods_receipts": grs})


class LineJev:
    """Matches invoice line n (1-based) to PO line `po_lines[n - 1]`."""

    def __init__(self, po_lines: list[int], confidence: float = 0.95) -> None:
        self.po_lines = po_lines
        self.confidence = confidence
        self.calls = 0

    def ask(self, state, questions, *, images=None) -> JevResult:
        self.calls += 1
        answers = {name: {"type": "choice", "choice": f"po_line_{self.po_lines[int(name[5:]) - 1]}",
                          "confidence": self.confidence, "probabilities": {}} for name in questions}
        return JevResult(model="fake-jev", answers=answers, input_tokens=10)


INVOICE = dict(vendor_name="Northwind Supply Co.", po_number="PO-1", currency="USD",
               line_descriptions=["Servo", "Driver"], line_quantities=[2, 4], line_unit_prices=[18.5, 6.25])


def match(erp_: Erp, po_lines=(1, 2), **values) -> set[str]:
    checks, _ = three_way_match(LineJev(list(po_lines)), load_schema("invoice"), ext(**{**INVOICE, **values}), erp_)
    return failed(checks)


def test_clean_match():
    assert match(erp()) == set()
    assert match(erp(received=(2, 4), receipts=2)) == set()  # two partial receipts add up


def test_missing_lines_and_receipt_are_flagged():
    flags = match(erp(received=()), line_descriptions=None, line_quantities=None, line_unit_prices=None)
    assert flags == {"unverified:three_way_match", "no_goods_receipt"}


def test_missing_quantity_is_unverified():
    assert match(erp(), line_quantities=[None, 4]) == {"unverified:line_values"}


def test_quantities_are_summed_per_po_line():
    # Two rows of 2 servos each against 2 ordered and received.
    flags = match(erp(), po_lines=(1, 1), line_descriptions=["Servo", "Servo"], line_quantities=[2, 2],
                  line_unit_prices=[18.5, 18.5])
    assert flags == {"qty_exceeds_po", "qty_exceeds_received"}


def test_quantity_billed_on_earlier_invoices_counts():
    assert match(erp(invoiced=1)) == {"qty_exceeds_po", "qty_exceeds_received"}


def test_supplier_and_currency_must_match_the_po():
    assert match(erp(), vendor_name="Bluefin Office Supplies LLC") == {"po_vendor_mismatch"}
    assert match(erp(), vendor_name="Initech Automation Inc.") == {"unknown_vendor"}
    assert match(erp(), vendor_name="NORTHWIND SUPPLY COMPANY") == set()
    assert match(erp(), currency="EUR") == {"currency_mismatch"}
    assert match(erp(), currency="€") == {"currency_mismatch"}
    assert match(erp(), currency=None) == {"unverified:currency"}


def test_same_text_lines_match_without_the_model():
    jev = LineJev([2, 1])  # would be wrong if asked
    checks, usage = three_way_match(jev, load_schema("invoice"),
                                    ext(**{**INVOICE, "line_descriptions": ["Servo motor", "Stepper driver"]}), erp())
    assert failed(checks) == set() and jev.calls == 0 and usage.jev_requests == 0
    matches = next(c for c in checks if c.id == "po_not_found").data["matches"]
    assert {m["po_line"] for m in matches.values()} == {1, 2} and all(m["how"] == "same text" for m in matches.values())


def test_uncertain_match_is_a_quality_flag_not_a_mismatch():
    uncertain = LineJev([1, 2], confidence=0.45)
    checks, _ = three_way_match(uncertain, load_schema("invoice"), ext(**INVOICE), erp())
    assert failed(checks) == {"unverified:line_match"}
    detail = next(c for c in checks if c.id == "unverified:line_match").detail
    assert "probably PO line 1" in detail and "p=0.45" in detail


def test_norm_currency():
    assert norm_currency(" usd ") == "USD"
    assert norm_currency("€") == "EUR"
    assert norm_currency("$") is None
