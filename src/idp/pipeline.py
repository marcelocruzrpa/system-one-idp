"""End-to-end: ingest -> classify (Jev) -> extract (Clairic) -> validate (rules, Jev, 3-way match) -> route."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from idp import ingest
from idp.classify import MAX_PAGES_AS_IMAGES, classify
from idp.clients import Clairic, Jev
from idp.config import schema as load_schema, threshold
from idp.erp import Erp
from idp.extract import extract
from idp.lines import reconcile
from idp.models import Check, Classification, Document, Mode, ModeChoice, ProcessResult, Usage
from idp.route import route
from idp.validate import run_rules, semantic_checks, three_way_match
from idp.validate.rules import demote_unsupported


def pick_mode(doc: Document) -> tuple[Mode, str]:
    """`text` when every page has a text layer (exact, cheapest); `hybrid` when any page was OCR'd, so the models can
    read past OCR errors on the page images. `image` alone is never picked: it drops pages beyond MAX_PAGES_AS_IMAGES
    and the OCR text that evidence spans index into."""
    ocr = [str(p.number) for p in doc.pages if p.text_source == "ocr"]
    if not ocr:
        return "text", "every page has a text layer"
    return "hybrid", "the text came from OCR" if len(ocr) == len(doc.pages) else f"page(s) {', '.join(ocr)} came from OCR"


def classification_checks(classification: Classification, schema: dict | None) -> list[Check]:
    """Is the type certain, and is it one the pipeline can process? Two different questions, two checks."""
    min_conf = threshold("classify_min_confidence")
    c = classification
    checks = [Check(id="uncertain_doc_type", stage="classify", passed=c.confidence >= min_conf,
                    detail=f"{c.doc_type} (confidence {c.confidence:.2f}, minimum {min_conf:.2f})",
                    probability=round(1 - c.confidence, 4), data={"doc_type": c.doc_type, "confidence": c.confidence})]
    if schema is None:
        checks.append(Check(id="unsupported_doc_type", stage="classify", passed=False,
                            detail=f"Classified as `{c.doc_type}` (confidence {c.confidence:.2f}): outside the supported "
                                   "document types, no extraction schema, needs a human."))
    return checks


def input_checks(doc: Document, mode: Mode, classification: Classification) -> list[Check]:
    """Did the models see the whole input, and is it a single document?"""
    p = classification.signals["multiple_documents"]
    checks = [Check(id="multiple_documents", stage="classify", passed=p < threshold("semantic_flag"),
                    detail=f"P(more than one document in the file) = {p:.2f}", probability=p)]
    pages, sent = len(doc.pages), min(len(doc.pages), MAX_PAGES_AS_IMAGES)
    if mode == "image" and pages > sent:
        checks.append(Check(id="pages_not_sent", stage="classify", passed=False,
                            detail=f"Image mode sends {sent} page images; pages {sent + 1}-{pages} were never read."))
    elif mode == "hybrid" and pages > sent:
        checks.append(Check(id="pages_not_sent", stage="classify", passed=True,
                            detail=f"All {pages} pages sent as text; only the first {sent} also as images."))
    else:
        checks.append(Check(id="pages_not_sent", stage="classify", passed=True, detail=f"All {pages} page(s) sent."))
    return checks


class Pipeline:
    def __init__(self, *, jev: Jev | None = None, clairic: Clairic | None = None, erp: Erp | None = None,
                 use_cache: bool = True) -> None:
        self.jev = jev or Jev(use_cache=use_cache)
        self.clairic = clairic or Clairic(use_cache=use_cache)
        self.erp = erp or Erp.load()

    def process(self, path: str | Path, mode: ModeChoice = "auto", *, doc: Document | None = None,
                force_ocr: bool = False) -> ProcessResult:
        timings: dict[str, float] = {}
        stage_usage: dict[str, Usage] = {}
        errors: list[str] = []
        models: dict[str, str] = {}

        def timed(stage: str, fn, *args, **kwargs):
            start = time.perf_counter()
            out = fn(*args, **kwargs)
            timings[stage] = round(time.perf_counter() - start, 3)
            return out

        doc = doc or timed("ingest", ingest.load, path, force_ocr=force_ocr)
        if mode == "auto":
            mode, _ = pick_mode(doc)
        classification, stage_usage["classify"] = timed("classify", classify, self.jev, doc, mode)
        if classification.model:
            models["jev"] = classification.model

        schema = load_schema(classification.doc_type)
        checks = classification_checks(classification, schema)
        checks += input_checks(doc, mode, classification)

        extraction = None
        line_items = None
        if schema is not None:
            extraction, stage_usage["extract"], models["clairic"] = timed("extract", extract, self.clairic, doc, schema, mode)
            checks.append(Check(id="extraction_truncated", stage="extract", passed=not extraction.truncated,
                                detail="Clairic truncated the input, so later fields may be missing or wrong."
                                       if extraction.truncated else "Clairic read the whole input."))
            checks += timed("rules", run_rules, schema, extraction, self.erp)
            table = reconcile(schema, extraction)
            line_items = table.as_dict() if table.relevant else None

            # Semantic checks and line matching are independent Jev requests: run them concurrently. A failed model
            # call degrades that stage to `unverified:<stage>` (review) instead of losing the whole document.
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {"semantic": pool.submit(semantic_checks, self.jev, doc, mode, schema, extraction, classification)}
                if any(r["check"] == "three_way_match" for r in schema.get("rules", [])):
                    futures["match"] = pool.submit(three_way_match, self.jev, schema, extraction, self.erp)
                for stage, future in futures.items():
                    try:
                        stage_checks, stage_usage[stage] = future.result()
                    except Exception as exc:  # noqa: BLE001 - any model/transport failure becomes a review flag
                        errors.append(f"{stage}: {type(exc).__name__}: {exc}")
                        stage_checks = [Check(id=f"unverified:{stage}", stage=stage, passed=False,
                                              detail=f"The {stage} checks could not run ({type(exc).__name__}: "
                                                     f"{str(exc)[:200]}); sent to review.")]
                        stage_usage[stage] = Usage()
                    checks += stage_checks
            timings["validate_models"] = round(time.perf_counter() - start, 3)
            checks = demote_unsupported(schema, extraction, checks)

        usage = Usage()
        for u in stage_usage.values():
            usage.add(u)
        decision, reasons = route(checks)
        return ProcessResult(
            path=str(path), mode=mode, text=doc.text, page_count=len(doc.pages),
            classification=classification, extraction=extraction, checks=checks,
            route=decision, reasons=reasons, line_items=line_items, usage=usage, stage_usage=stage_usage,
            timings=timings, models=models, errors=errors,
        )

    def close(self) -> None:
        self.jev.close()
        self.clairic.close()
