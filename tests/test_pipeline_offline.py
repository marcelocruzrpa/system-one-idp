"""End-to-end over every sample with perfect fake models: seeded defects must surface as the expected flags and route."""

import json

import pytest

from idp import ingest
from idp.config import SAMPLES_DIR
from idp.erp import norm_ref
from idp.evaluate import business_flags
from idp.pipeline import Pipeline, pick_mode
from tests.conftest import load_truth

MANIFEST = json.loads((SAMPLES_DIR / "manifest.json").read_text(encoding="utf-8")) if (SAMPLES_DIR / "manifest.json").exists() else []


@pytest.mark.skipif(not MANIFEST, reason="run `idp generate` first")
@pytest.mark.parametrize("entry", MANIFEST, ids=[m["doc_id"] for m in MANIFEST])
def test_sample_flags_and_route(entry, fakes):
    jev, clairic = fakes
    truth = load_truth(entry["doc_id"])
    jev.truth = clairic.truth = truth
    pipeline = Pipeline(jev=jev, clairic=clairic)
    doc = ingest.load(SAMPLES_DIR / entry["digital"])
    # The fakes answer from ground truth, so check separately that ingestion read the document's references.
    text = norm_ref(doc.text)
    for name, value in truth["fields"].items():
        if name.endswith("_number") and isinstance(value, str):
            assert norm_ref(value) in text, name
    result = pipeline.process(entry["digital"], "text", doc=doc)

    assert result.classification.doc_type == truth["doc_type"]
    assert business_flags(result) == set(truth["expected_flags"]), result.reasons
    assert result.route == truth["expected_route"]


@pytest.mark.skipif(not MANIFEST, reason="run `idp generate` first")
def test_image_mode_sends_page_images(fakes):
    jev, clairic = fakes
    jev.truth = clairic.truth = load_truth("invoice-01")
    doc = ingest.load(SAMPLES_DIR / "digital" / "invoice-01.pdf")
    Pipeline(jev=jev, clairic=clairic).process("invoice-01", "image", doc=doc)
    classify_call = jev.calls[0]
    assert classify_call["images"] and "document" not in classify_call["state"]
    assert "`pages`" in classify_call["questions"]["doc_type"].instructions


@pytest.mark.skipif(not MANIFEST, reason="run `idp generate` first")
@pytest.mark.parametrize(("sample", "expected"), [("digital/invoice-01.pdf", "text"), ("scanned/invoice-01.png", "hybrid"),
                                                  ("scanned/invoice-07.pdf", "hybrid")])
def test_auto_mode_follows_text_source(sample, expected):
    assert pick_mode(ingest.load(SAMPLES_DIR / sample))[0] == expected


@pytest.mark.skipif(not MANIFEST, reason="run `idp generate` first")
def test_auto_mode_sends_ocr_text_and_images_for_scans(fakes):
    jev, clairic = fakes
    jev.truth = clairic.truth = load_truth("invoice-01")
    doc = ingest.load(SAMPLES_DIR / "scanned" / "invoice-01.png")
    result = Pipeline(jev=jev, clairic=clairic).process("invoice-01", "auto", doc=doc)
    assert result.mode == "hybrid"
    classify_call = jev.calls[0]
    assert classify_call["images"] and classify_call["state"]["document"] == doc.text


def run_clean_invoice(jev, clairic, mode="text", pages=1):
    """invoice-01 (clean, auto-approved) with the first page repeated to `pages` pages."""
    truth = load_truth("invoice-01")
    jev.truth = clairic.truth = truth
    doc = ingest.load(SAMPLES_DIR / "digital" / "invoice-01.pdf")
    doc = doc.model_copy(update={"pages": [doc.pages[0].model_copy(update={"number": i}) for i in range(1, pages + 1)]})
    return Pipeline(jev=jev, clairic=clairic).process("invoice-01", mode, doc=doc)


@pytest.mark.skipif(not MANIFEST, reason="run `idp generate` first")
def test_incomplete_input_is_never_auto_approved(fakes):
    jev, clairic = fakes
    assert run_clean_invoice(jev, clairic).route == "auto_approve"

    clairic.truncated = True
    result = run_clean_invoice(jev, clairic)
    assert result.route == "review" and "extraction_truncated" in result.flags
    clairic.truncated = False

    result = run_clean_invoice(jev, clairic, mode="image", pages=5)
    assert result.route == "review" and "pages_not_sent" in result.flags
    assert run_clean_invoice(jev, clairic, mode="hybrid", pages=5).route == "auto_approve"

    jev.signals = {"multiple_documents": 0.99}
    result = run_clean_invoice(jev, clairic)
    assert result.route == "review" and "multiple_documents" in result.flags


@pytest.mark.skipif(not MANIFEST, reason="run `idp generate` first")
def test_invoice_without_lines_is_not_auto_approved(fakes):
    jev, clairic = fakes
    truth = load_truth("invoice-01")
    for name in ("line_descriptions", "line_quantities", "line_unit_prices", "line_amounts"):
        truth["fields"][name] = None
    jev.truth = clairic.truth = truth
    result = Pipeline(jev=jev, clairic=clairic).process("invoice-01", "text",
                                                        doc=ingest.load(SAMPLES_DIR / "digital" / "invoice-01.pdf"))
    assert result.route == "review"
    assert {"unverified:three_way_match", "unverified:line_items"} <= set(result.flags)
    assert not {"line_sum_mismatch", "line_math_mismatch", "unmatched_line"} & set(result.flags)


@pytest.mark.skipif(not MANIFEST, reason="run `idp generate` first")
def test_stray_list_items_are_realigned_from_the_page(fakes):
    """Clairic's text list with strays and SKU cells must not shift rows: the three-way match reads rows from the page."""
    jev, clairic = fakes
    truth = load_truth("invoice-02")
    truth["fields"]["line_descriptions"] = ["Tax", *truth["fields"]["line_descriptions"], "HC-SFU1204-500"]
    jev.truth = clairic.truth = truth
    doc = ingest.load(SAMPLES_DIR / "digital" / "invoice-02.pdf")
    result = Pipeline(jev=jev, clairic=clairic).process("invoice-02", "text", doc=doc)
    assert business_flags(result) == set(load_truth("invoice-02")["expected_flags"]), result.reasons
    assert result.line_items and result.line_items["source"] == "rows" and len(result.line_items["rows"]) == 3
    assert "Tax" in result.line_items["dropped"]


@pytest.mark.skipif(not MANIFEST, reason="run `idp generate` first")
def test_unsupported_doc_type_is_its_own_check(fakes):
    jev, clairic = fakes
    truth = load_truth("quote-01")
    truth["doc_type"] = "other"
    jev.truth = clairic.truth = truth
    result = Pipeline(jev=jev, clairic=clairic).process("quote-01", "text",
                                                        doc=ingest.load(SAMPLES_DIR / "digital" / "quote-01.pdf"))
    assert result.route == "review" and result.extraction is None
    assert "unsupported_doc_type" in result.flags and "uncertain_doc_type" not in result.flags
    assert business_flags(result) == set()


@pytest.mark.skipif(not MANIFEST, reason="run `idp generate` first")
def test_model_failure_in_validation_degrades_to_review(fakes):
    jev, clairic = fakes
    truth = load_truth("invoice-01")
    jev.truth = clairic.truth = truth
    calls = {"n": 0}
    original = jev.ask

    def flaky(state, questions, *, images=None):
        calls["n"] += 1
        if calls["n"] > 1:  # classification succeeds, every later Jev request fails
            raise RuntimeError("503 overloaded")
        return original(state, questions, images=images)

    jev.ask = flaky
    result = Pipeline(jev=jev, clairic=clairic).process("invoice-01", "text",
                                                        doc=ingest.load(SAMPLES_DIR / "digital" / "invoice-01.pdf"))
    assert result.route == "review"
    assert {"unverified:semantic", "unverified:match"} <= set(result.flags) and len(result.errors) == 2
