"""Scoring must not reward near-miss identifiers, credit the verifier for fields it never checked, or hide ingest time."""

import json

import pytest

from idp import ingest
from idp.config import SAMPLES_DIR, schema as load_schema
from idp.evaluate import (NoMatchingSamples, align_items, evaluate_doc, failed_row, score_field, scorecard,
                          select_entries)
from idp.pipeline import Pipeline
from tests.conftest import load_truth

TEXT = {"type": "text"}
ITEMS = {"type": "list", "items": "text"}
NEEDS_SAMPLES = pytest.mark.skipif(not (SAMPLES_DIR / "manifest.json").exists(), reason="run `idp generate` first")


def process(fakes, doc_id: str, clairic_truth: dict | None = None):
    """invoice `doc_id` through the pipeline with the fakes answering from (optionally doctored) ground truth."""
    jev, clairic = fakes
    truth = load_truth(doc_id)
    jev.truth, clairic.truth = truth, clairic_truth or truth
    result = Pipeline(jev=jev, clairic=clairic).process(doc_id, "text", doc=ingest.load(SAMPLES_DIR / "digital" / f"{doc_id}.pdf"))
    result.timings = {"ingest": 0.25, **result.timings}
    return truth, result


def test_identifiers_need_an_exact_match():
    assert score_field(TEXT, "INV-NW-10231", "INV", "invoice_number") == 0.0
    assert score_field(TEXT, "PO-4501", "PO-45010", "po_number") == 0.0
    assert score_field(TEXT, "PO-4501", "po 4501", "po_number") == 1.0
    assert score_field(TEXT, "USD", "US", "currency") == 0.0
    assert score_field(TEXT, "77-0421953", "Tax ID: 77-0421953", "vendor_tax_id") == 0.0
    items = {"type": "list", "items": "text"}
    assert score_field(items, ["INV-1", "INV-2"], ["INV-1", "INV"], "invoice_numbers") == 0.5


def test_free_text_tolerates_small_wording_differences_only():
    assert score_field(TEXT, "Acme Robotics Inc.", "ACME ROBOTICS INC", "bill_to_name") == 1.0
    assert score_field(TEXT, "Acme Robotics Inc.", "Acme Robotics Inc. Accounts Payable", "bill_to_name") == 1.0
    assert score_field(TEXT, "Acme Robotics Inc.", "Acme", "bill_to_name") == 0.0
    assert score_field(TEXT, "Northwind Supply Co.", "Northwind Supply Co. 455 Industrial Way Fremont CA", "vendor_name") == 0.0


@pytest.mark.skipif(not (SAMPLES_DIR / "manifest.json").exists(), reason="run `idp generate` first")
def test_verifier_counts_only_checked_fields_and_ingest_time_is_reported(fakes):
    jev, clairic = fakes
    truth = load_truth("invoice-01")
    jev.truth = clairic.truth = truth
    result = Pipeline(jev=jev, clairic=clairic).process(
        "invoice-01", "text", doc=ingest.load(SAMPLES_DIR / "digital" / "invoice-01.pdf"))
    result.timings = {"ingest": 0.25, **result.timings}

    card = scorecard([evaluate_doc(truth, result)])
    checked = len(load_schema("invoice")["verify_fields"])
    fields = len(load_schema("invoice")["fields"])
    assert sum(v for k, v in card["verifier"].items() if k != "fields_not_verified") == checked
    assert card["verifier"]["fields_not_verified"] == fields - checked
    assert sum(card["extraction_flags"].values()) == fields
    assert card["seconds_per_document"]["ingest"] == pytest.approx(0.25, abs=0.01)
    assert card["seconds_per_document"]["end_to_end"] >= card["seconds_per_document"]["ingest"]

    # Without semantic checks, the verifier gets no credit at all.
    result.checks = [c for c in result.checks if c.stage != "semantic"]
    card = scorecard([evaluate_doc(truth, result)])
    assert card["verifier"] == {"fields_not_verified": fields}


def test_scorecard_with_no_rows():
    card = scorecard([])
    assert card["documents"] == 0 and card["documents_failed"] == 0 and card["errors"] == []
    assert card["classification_accuracy"] is None and card["route_accuracy"] is None
    assert card["field_accuracy"] is None and card["scalar_field_accuracy"] is None
    assert card["list_items"]["recall"] is None and card["cache"]["hit_rate"] is None
    assert card["cost_usd"]["per_document"] is None and card["cost_usd"]["per_1000_documents"] is None
    assert card["latency"] == {"p50": None, "p95": None, "max": None}
    assert card["safety"]["needs_attention_total"] == 0 and card["seconds_per_document"]["pipeline"] is None
    json.dumps(card)


def test_select_entries_matches_types_ids_and_prefixes():
    manifest = [{"doc_id": "invoice-01", "doc_type": "invoice"}, {"doc_id": "bank_statement-01", "doc_type": "bank_statement"},
                {"doc_id": "credit_note-02", "doc_type": "credit_note"}]
    assert select_entries(manifest, None) == manifest and select_entries(manifest, []) == manifest
    assert select_entries(manifest, ["Bank-Statement"]) == [manifest[1]]
    assert select_entries(manifest, ["credit_note-02,INVOICE"]) == [manifest[0], manifest[2]]  # manifest order kept
    assert select_entries(manifest, ["bank"]) == [manifest[1]]
    with pytest.raises(NoMatchingSamples, match="bank_statement, credit_note, invoice"):
        select_entries(manifest, ["invoices"])
    with pytest.raises(NoMatchingSamples, match="'receipt'"):
        select_entries(manifest, ["invoice", "receipt"])


def test_align_items_tolerates_dropped_and_stray_rows():
    truth = ["Servo Motor", "Stepper Driver", "Battery Pack"]
    pred = ["Tax", "Servo Motor", "Battery Pack"]
    assert align_items("text", truth, pred) == {"aligned": 2, "extra": 1, "missing": 1}
    assert align_items("number", [40.0, 60.0, 12.0], [40.0, 60.0, 12.0]) == {"aligned": 3, "extra": 0, "missing": 0}
    assert align_items("text", ["INV-1", "INV-2"], ["INV-2", "INV-1"], identifier=True)["aligned"] == 1  # order matters
    assert align_items("text", [], []) == {"aligned": 0, "extra": 0, "missing": 0}
    # The positional score keeps punishing the shift; the alignment is reported next to it, not instead of it.
    assert score_field(ITEMS, truth, pred) == pytest.approx(1 / 3)


@NEEDS_SAMPLES
def test_scorecard_splits_scalars_lists_abstentions_and_cache(fakes):
    truth = load_truth("invoice-01")
    doctored = json.loads(json.dumps(truth))
    doctored["fields"]["payment_terms"] = None  # the fake answers null with p_null 0.9: an abstention, not a misread
    truth, result = process(fakes, "invoice-01", doctored)
    # Drop a row after the pipeline ran: `lines.reconcile` would otherwise rebuild it from the page's table rows.
    descriptions = result.extraction.fields["line_descriptions"]
    descriptions.value = descriptions.value[1:]
    row = evaluate_doc(truth, result)
    row["result"] = result.model_dump(mode="json", exclude={"text"})
    assert row["fields"]["payment_terms"] == {**row["fields"]["payment_terms"], "score": 0.0, "abstained": True, "list": False}
    assert row["fields"]["line_descriptions"]["list"] and row["fields"]["line_descriptions"]["missing"] == 1
    assert row["fields"]["line_descriptions"]["aligned"] == 2 and row["fields"]["line_descriptions"]["score"] == 0.0

    card = scorecard([row])
    scalars = [f for f in row["fields"].values() if not f["list"]]
    assert card["scalar_field_accuracy"] == pytest.approx((len(scalars) - 1) / len(scalars))
    assert card["list_field_accuracy"] == pytest.approx(3 / 4) and card["abstained"] == 1
    assert card["list_items"] == {"recall": pytest.approx(11 / 12), "precision": 1.0, "extra_items": 0, "missing_items": 1}
    assert card["confidence_calibration"]["0.9-1.0"]["n"] == len(scalars)
    assert card["confidence_calibration_lists"]["0.9-1.0"]["n"] == 4
    assert card["modes"] == {"text": 1} and card["models"] == {"clairic": ["clairic"], "jev": ["fake-jev"]}
    # Fakes are never cached: 3 Jev requests (classify, semantic, match) and 1 Clairic request, all live.
    assert card["cache"] == {"hit_rate": 0.0, "cached_requests": 0, "live_requests": 4}
    assert card["latency"]["p50"] == pytest.approx(row["pipeline_seconds"], abs=0.001)
    assert card["latency"]["max"] == pytest.approx(row["pipeline_seconds"], abs=0.001)


@NEEDS_SAMPLES
def test_failed_documents_are_listed_not_scored(fakes):
    truth, result = process(fakes, "invoice-01")
    good = evaluate_doc(truth, result)
    bad = failed_row({"doc_id": "invoice-02", "doc_type": "invoice", "expected_route": "review",
                      "expected_flags": ["qty_exceeds_received"]}, RuntimeError("Cloudflare unreachable"))
    card = scorecard([good, bad])
    assert card["documents"] == 1 and card["documents_failed"] == 1
    assert card["errors"] == [{"doc_id": "invoice-02", "error": "RuntimeError: Cloudflare unreachable"}]
    assert card["classification_accuracy"] == 1.0 and card["route_accuracy"] == 1.0
    assert card["safety"] == {"needs_attention_caught": 0, "needs_attention_total": 0, "false_approvals": 0,
                              "false_rejects": 0, "clean_auto_approved": 1, "clean_total": 1}
    assert card["cost_usd"]["per_document"] == pytest.approx(result.cost.total, abs=1e-6)
    json.dumps(card)
