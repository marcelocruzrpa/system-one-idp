"""Offline fakes for Jev and Clairic that answer from sample ground truth."""

import json
from typing import Any

import pytest

from idp.clients.jev import JevResult
from idp.config import ROOT, SAMPLES_DIR, schema as load_schema
from idp.models import Check, Classification, Extraction, FieldAnswer, ProcessResult
from idp.route import route

SIGNAL_FLAGS = {"has_paid_marking": "already_paid", "requests_bank_detail_change": "bank_detail_change"}


def load_truth(doc_id: str) -> dict[str, Any]:
    return json.loads((SAMPLES_DIR / "ground_truth" / f"{doc_id}.json").read_text(encoding="utf-8"))


def make_result(*failed: str, path: str = "samples/digital/invoice-03.pdf", mode: str = "text") -> ProcessResult:
    """A processed invoice whose only failed checks are `failed`, routed by the real `route()`."""
    checks = [Check(id="required_fields", stage="rules", passed=True, detail="All required fields present.")]
    checks += [Check(id=f, stage="rules", passed=False, detail=f"{f} detail: with a colon.") for f in failed]
    decision, reasons = route(checks)
    return ProcessResult(path=str(ROOT / path), mode=mode, text="Invoice INV-7 | Total | 1,329.56", page_count=1,
                         classification=Classification(doc_type="invoice", confidence=0.97, probabilities={"invoice": 0.97}),
                         extraction=Extraction(fields={"total": FieldAnswer(type="number", value=1329.56, confidence=0.9,
                                                                            p_null=0.01)}),
                         checks=checks, route=decision, reasons=reasons)


class FakeJev:
    """Answers as a perfect model would for the current document's ground truth."""

    def __init__(self) -> None:
        self.truth: dict[str, Any] = {}
        self.calls: list[dict[str, Any]] = []
        self.signals: dict[str, float] = {}
        """Override the answer to a speculative signal, e.g. {"multiple_documents": 0.99}."""

    def ask(self, state, questions, *, images=None) -> JevResult:
        self.calls.append({"state": state, "questions": questions, "images": images})
        flags = set(self.truth["expected_flags"])
        answers = {}
        for name in questions:
            if name == "doc_type":
                answers[name] = {"type": "choice", "choice": self.truth["doc_type"], "confidence": 0.97,
                                 "probabilities": {self.truth["doc_type"]: 0.97}}
            elif name.startswith("line_"):
                # Invoice lines are generated in PO order, so invoice line n (1-based) is PO line n.
                answers[name] = {"type": "choice", "choice": f"po_line_{int(name[5:])}", "confidence": 0.95,
                                 "probabilities": {}}
            elif name.startswith("signal_"):
                default = 0.95 if SIGNAL_FLAGS.get(name[7:]) in flags else 0.03
                answers[name] = {"type": "noul", "noul": self.signals.get(name[7:], default)}
            else:
                answers[name] = {"type": "noul", "noul": 0.95 if name in flags else 0.03}
        return JevResult(model="fake-jev", answers=answers, input_tokens=100)

    def close(self) -> None:
        pass


class FakeClairic:
    """Returns ground-truth values for every field of the requested schema."""

    def __init__(self) -> None:
        self.truth: dict[str, Any] = {}
        self.truncated = False

    def extract(self, fields, *, state="", images=None, reference_date=None) -> dict[str, Any]:
        answers = {}
        for name, spec in fields.items():
            value = self.truth["fields"].get(name)
            answers[name] = {"type": spec["type"], "value": value, "guess": value,
                             "confidence": 0.95, "p_null": 0.9 if value is None else 0.01}
        return {"answers": answers, "truncated": self.truncated, "usage": {"input_tokens": 200, "output_tokens": 0}}

    def close(self) -> None:
        pass


@pytest.fixture
def fakes():
    return FakeJev(), FakeClairic()


@pytest.fixture
def invoice_schema():
    return load_schema("invoice")
