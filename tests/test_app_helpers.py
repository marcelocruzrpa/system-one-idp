"""The pure helpers behind the Streamlit demo (app/ui_helpers.py), imported the way `streamlit run` does."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from ui_helpers import (check_label, erp_url, extraction_rows, find_evidence, line_item_rows, lines_table,  # noqa: E402
                        po_number_from_detail, replayed_from_cache, split_reasons, truth_caption)

from idp.models import Extraction, FieldAnswer, Usage  # noqa: E402


def test_check_label_known_prefixed_and_fallback():
    assert check_label("price_mismatch") == "Unit price differs from PO"
    assert check_label("qty_exceeds_received") == "Billed more than received"
    assert check_label("missing:invoice_number") == "Missing: invoice_number"
    assert check_label("low_confidence:total") == "Low confidence: total"
    assert check_label("unverified:totals") == "Could not verify: totals"
    assert check_label("unverified:total_mismatch") == "Could not verify: subtotal + tax equals the total"
    assert check_label("brand_new:flag_id") == "brand new: flag id"


def test_check_label_passed_reads_as_what_was_established():
    assert check_label("duplicate_invoice", passed=True) == "Not posted before"
    assert check_label("duplicate_invoice") == "Already posted"
    assert check_label("unsupported:total", passed=True) == "Supported by the document: total"
    assert check_label("unsupported:total") == "Not supported by the document: total"
    assert check_label("required_fields", passed=True) == "Required fields present"


def test_split_reasons_keeps_order_and_colons_in_details():
    reasons = ["duplicate_invoice: Invoice X was already posted on 2026-08-07 (Northwind, USD 904.35).",
               "price_mismatch: Unit price differs from the PO by more than 2%: line 1 265.00 vs PO 245.00.",
               "missing:po_number: Required field `po_number` not found (p_null=0.9).",
               "unverified:line_match: PO line could not be determined: line 2 'x' -> probably PO line 1 (p=0.55)."]
    business, quality = split_reasons(reasons)
    assert [cid for cid, _ in business] == ["duplicate_invoice", "price_mismatch"]
    assert business[1][1].startswith("Unit price differs from the PO by more than 2%: line 1")
    assert [cid for cid, _ in quality] == ["missing:po_number", "unverified:line_match"]
    assert quality[0][1] == "Required field `po_number` not found (p_null=0.9)."


def test_truth_caption_matched_and_differs():
    truth = {"expected_route": "review", "expected_flags": ["price_mismatch"]}
    assert truth_caption("review", {"price_mismatch"}, truth) == ("Expected: review (price_mismatch): matched", True)
    text, ok = truth_caption("auto_approve", {"total_mismatch"}, truth)
    assert not ok
    assert text == ("Expected: review (price_mismatch): differs: routed to auto approve; missed price_mismatch; "
                    "extra total_mismatch")
    clean = {"expected_route": "auto_approve", "expected_flags": []}
    assert truth_caption("auto_approve", set(), clean) == ("Expected: auto approve (no seeded defects): matched", True)


def test_find_evidence_number_forms():
    text = "Subtotal | 1,234.56\nSales tax | 95.00\nTotal due | 1,329.56"
    assert find_evidence(text, 1234.56, "number").startswith("...Subtotal | [1,234.56] / Sales tax | 95.00 / Total")
    assert "[95.00]" in find_evidence(text, 95.0, "number")
    assert "[2.472,00]" in find_evidence("Zwischensumme | 2.472,00 EUR", 2472, "number")
    assert "[10]" in find_evidence("Qty | 10 | Encoder", 10.0, "number")
    assert find_evidence(text, 999.0, "number") == ""
    assert find_evidence("", 1.0, "number") == ""
    assert find_evidence(text, [1.0], "number") == ""


def test_find_evidence_does_not_match_inside_a_longer_number():
    assert "| [100.00]" in find_evidence("Amount 1100.00 | 100.00", 100.0, "number")
    assert find_evidence("Total 1,234.56 only", 1234.0, "number") == ""
    assert find_evidence("Ref 195.00 only", 95.0, "number") == ""


def test_find_evidence_dates_and_text():
    assert "[September 15, 2026]" in find_evidence("Invoice date: September 15, 2026", "2026-09-15", "date")
    assert "[16.09.2026]" in find_evidence("Rechnungsdatum: 16.09.2026", "2026-09-16", "date")
    assert "[2026-09-15]" in find_evidence("issued 2026-09-15", "2026-09-15", "date")
    assert "[Sep 8, 2026]" in find_evidence("Posted Sep 8, 2026", "2026-09-08", "date")
    assert find_evidence("no dates here", "2026-09-15", "date") == ""
    assert "[Net 30]" in find_evidence("Terms: Net 30 days", "Net 30", "text")


def test_extraction_rows_lists_have_no_confidence_and_a_note():
    text = "Invoice INV-7 | Total | 1,329.56\nWidget | 2 | 10.00 | 20.00"
    ext = Extraction(state_text=text, fields={
        "invoice_number": FieldAnswer(type="text", value="INV-7", confidence=0.93, p_null=0.01, span=[8, 13]),
        "total": FieldAnswer(type="number", value=1329.56, confidence=0.88, p_null=0.02),
        "line_amounts": FieldAnswer(type="list", value=[20.0], guess=[20.0], confidence=0.4, confidences=[0.9], source="rows"),
        "line_descriptions": FieldAnswer(type="list", value=["Widget"], confidence=0.05),
    })
    schema = {"fields": {"invoice_number": {"type": "text"}, "total": {"type": "number"},
                         "line_amounts": {"type": "list", "items": "number"},
                         "line_descriptions": {"type": "list", "items": "text"}}}
    rows = {r["field"]: r for r in extraction_rows(ext, schema)}
    assert rows["invoice_number"]["evidence"].startswith("...Invoice [INV-7] | Total | 1,329.56 / Widge")
    assert rows["invoice_number"]["confidence"] == 0.93 and rows["invoice_number"]["note"] == ""
    assert "[1,329.56]" in rows["total"]["evidence"]
    assert rows["line_amounts"]["confidence"] is None and rows["line_amounts"]["note"] == "rebuilt from page rows"
    assert rows["line_descriptions"]["note"] == "n/a (list)" and rows["line_descriptions"]["evidence"] == ""
    assert rows["line_amounts"]["value"] == "[20.0]"
    assert "matches truth" not in rows["total"]

    truth = {"fields": {"invoice_number": "INV-7", "total": 1329.56, "line_amounts": [20.0], "line_descriptions": ["Bracket"]}}
    with_truth = {r["field"]: r["matches truth"] for r in extraction_rows(ext, schema, truth)}
    assert with_truth == {"invoice_number": True, "total": True, "line_amounts": True, "line_descriptions": False}


def test_line_item_rows_follow_the_schema_columns():
    invoice = {"rows": [{"line": 1, "description": "Servo", "quantities": 10.0, "unit_prices": 129.5, "amounts": 1295.0,
                         "text": "Servo | 10 | 129.50 | 1,295.00", "item_code": "SRV-1"},
                        {"line": 2, "description": "Gearbox", "quantities": 6.0, "unit_prices": 412.0, "amounts": 2472.0,
                         "text": None, "item_code": None}]}
    rows = line_item_rows(invoice)
    assert list(rows[0]) == ["line", "item code", "description", "quantity", "unit price", "amount"]
    assert rows[1]["amount"] == 2472.0 and rows[1]["item code"] is None

    delivery = {"rows": [{"line": 1, "description": "Encoder", "ordered": 10.0, "shipped": 8.0, "text": "x", "item_code": None}]}
    rows = line_item_rows(delivery)
    assert list(rows[0]) == ["line", "description", "ordered", "shipped"]
    assert line_item_rows({"rows": []}) == []


def test_lines_table_picks_the_useful_columns():
    price = [{"line": 1, "po_line": 1, "description": "Chair", "invoiced": 265.0, "po": 245.0, "delta_pct": 8.2}]
    assert lines_table("price_mismatch", price) == [{"line": 1, "description": "Chair", "invoiced": 265.0, "PO price": 245.0,
                                                     "delta %": 8.2}]
    qty = [{"po_line": 2, "description": "Encoder", "invoice_lines": [1], "invoiced": 10.0, "previously_invoiced": 0,
            "received": 8.0}]
    assert lines_table("qty_exceeds_received", qty) == [{"PO line": 2, "description": "Encoder", "invoiced": 10.0,
                                                         "received": 8.0}]
    unmatched = [{"line": 3, "description": "Freight", "po_line": None, "status": "no_match", "choice": "no_match",
                  "confidence": 0.91, "candidate": None}]
    assert lines_table("unmatched_line", unmatched) == [{"line": 3, "description": "Freight", "confidence": 0.91}]
    uncertain = [{"line": 2, "description": "Enc.", "po_line": None, "status": "uncertain", "choice": "po_line_1",
                  "confidence": 0.55, "candidate": 1, "candidate_description": "Rotary encoder"}]
    assert lines_table("unverified:line_match", uncertain)[0]["closest PO line"] == "Rotary encoder"
    generic = [{"line": 1, "quantity": 2.0, "unit_price": 10.0, "amount": 25.0, "expected": 20.0, "nested": {"a": 1}}]
    assert lines_table("something_else", generic) == [{"line": 1, "quantity": 2.0, "unit price": 10.0, "amount": 25.0,
                                                       "expected": 20.0}]
    assert lines_table("unverified:line_math", [1, 2]) == []


def test_po_number_from_detail_and_erp_url():
    assert po_number_from_detail("PO-4501 found (Northwind Supply Co.).") == "PO-4501"
    assert po_number_from_detail("PO 1 found (Vendor (Europe) GmbH).") == "PO 1"
    assert po_number_from_detail("PO PO-9 is not in the ERP.") is None
    url = erp_url(Path("out") / "erp.html", "PO 1")
    assert url.startswith("file:///") and url.endswith("/out/erp.html#/pos/PO%201")


def test_replayed_from_cache_needs_every_request_cached():
    cached = {"classify": Usage(jev_input_tokens=100, jev_requests=1, jev_cached_tokens=100),
              "extract": Usage(clairic_input_tokens=200, clairic_requests=1, clairic_cached_tokens=200)}
    assert replayed_from_cache(cached)
    live = {**cached, "match": Usage(jev_input_tokens=50, jev_requests=1)}
    assert not replayed_from_cache(live)
    assert not replayed_from_cache({})
    assert not replayed_from_cache({"classify": Usage(jev_requests=1)})
