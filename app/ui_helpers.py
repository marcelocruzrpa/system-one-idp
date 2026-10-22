"""Pure helpers behind the Streamlit demo: check labels, reason splitting, evidence search and table shaping.

Nothing here imports Streamlit, so `tests/test_app_helpers.py` exercises it without a browser session.
"""

import json
import re
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote

from idp.evaluate import score_field
from idp.models import Extraction, FieldAnswer, Usage
from idp.route import is_quality_flag

# What a FAILED check means, in the words of an AP clerk. Prefixed ids (`missing:<field>`, `unverified:<rule>`, ...)
# are built in `check_label`; anything unknown falls back to the id with its separators spelled out.
LABELS: dict[str, str] = {
    "uncertain_doc_type": "Uncertain document type",
    "unsupported_doc_type": "Unsupported document type",
    "multiple_documents": "More than one document in the file",
    "pages_not_sent": "Pages the models never saw",
    "extraction_truncated": "Clairic truncated the input",
    "unverified:line_items": "Line items could not be aligned",
    "total_mismatch": "Subtotal + tax does not equal the total",
    "line_sum_mismatch": "Line amounts do not add up to the subtotal",
    "line_math_mismatch": "Quantity x unit price differs from the line amount",
    "unverified:line_math": "Line math not checked on every line",
    "due_before_issue": "Due date before the issue date",
    "delivery_before_order": "Delivery date before the order date",
    "valid_until_before_issue": "Quote expires before its own date",
    "period_end_before_start": "Statement period ends before it starts",
    "future_date": "Dated in the future",
    "duplicate_invoice": "Already posted",
    "unverified:duplicate_invoice": "Possibly already posted (vendor unknown)",
    "unverified:vendor_identity": "Vendor name and tax id disagree",
    "wrong_role:vendor_tax_id": "Vendor tax id is our own",
    "missing_po": "No purchase order number",
    "original_invoice_not_found": "Original invoice not in the ERP",
    "unverified:original_invoice_exists": "Original invoice could not be looked up",
    "short_shipment": "Shipped less than ordered",
    "unverified:short_shipment": "Shipment not checked on every line",
    "balance_mismatch": "Balances do not reconcile",
    "remittance_sum_mismatch": "Paid amounts do not add up to the payment total",
    "not_addressed_to_us": "Not addressed to us",
    "not_issued_by_us": "Purchase order not issued by us",
    "already_paid": "Marked as paid",
    "bank_detail_change": "Bank details change requested",
    "short_payment": "Paid less than invoiced",
    "non_reimbursable_items": "Non-reimbursable items",
    "unverified:semantic": "Semantic checks could not run",
    "unknown_vendor": "Vendor not in the vendor master",
    "po_vendor_mismatch": "PO was issued to another vendor",
    "unverified:currency": "Currency could not be compared",
    "currency_mismatch": "Currency differs from the PO",
    "po_not_found": "Purchase order not in the ERP",
    "no_goods_receipt": "No goods receipt for the PO",
    "unverified:three_way_match": "No invoice lines to match",
    "unmatched_line": "Invoice line not on the PO",
    "unverified:line_match": "Uncertain PO line match",
    "unverified:line_values": "Matched lines with a doubtful quantity or price",
    "price_mismatch": "Unit price differs from PO",
    "qty_exceeds_po": "Billed more than ordered",
    "qty_exceeds_received": "Billed more than received",
    "unverified:match": "Three-way match could not run",
}
# What a PASSED check established.
PASSED: dict[str, str] = {
    "uncertain_doc_type": "Document type is certain",
    "multiple_documents": "One document in the file",
    "pages_not_sent": "Every page was sent",
    "extraction_truncated": "Whole input read",
    "required_fields": "Required fields present",
    "line_items": "Line items aligned",
    "total_mismatch": "Subtotal + tax equals the total",
    "line_sum_mismatch": "Line amounts add up",
    "line_math_mismatch": "Quantity x unit price equals each line amount",
    "due_before_issue": "Due date after the issue date",
    "delivery_before_order": "Delivery date after the order date",
    "valid_until_before_issue": "Quote validity after its date",
    "period_end_before_start": "Statement period in order",
    "future_date": "Not dated in the future",
    "duplicate_invoice": "Not posted before",
    "vendor_identity": "Vendor name and tax id agree",
    "missing_po": "Purchase order referenced",
    "original_invoice_not_found": "Original invoice is in the ERP",
    "short_shipment": "Shipped in full",
    "balance_mismatch": "Balances reconcile",
    "remittance_sum_mismatch": "Paid amounts add up",
    "not_addressed_to_us": "Addressed to us",
    "not_issued_by_us": "Purchase order issued by us",
    "already_paid": "Not marked as paid",
    "bank_detail_change": "No bank details change",
    "short_payment": "Paid in full",
    "non_reimbursable_items": "Reimbursable items only",
    "po_vendor_mismatch": "PO vendor matches",
    "currency_mismatch": "Currency matches the PO",
    "po_not_found": "Purchase order found",
    "no_goods_receipt": "Goods receipt recorded",
    "unmatched_line": "Every line is on the PO",
    "price_mismatch": "Unit prices match the PO",
    "qty_exceeds_po": "Quantities within the PO",
    "qty_exceeds_received": "Quantities within what was received",
}
PREFIXES = {"missing": "Missing", "low_confidence": "Low confidence", "unsupported": "Not supported by the document",
            "wrong_role": "Wrong role", "unverified": "Could not verify"}
PASSED_PREFIXES = {"unsupported": "Supported by the document", "wrong_role": "Right role"}

# Columns worth reading in the `data["lines"]` of a failed per-line check, in display order.
LINE_COLUMNS: dict[str, list[str]] = {
    "price_mismatch": ["line", "description", "invoiced", "po", "delta_pct"],
    "qty_exceeds_po": ["po_line", "description", "invoiced", "previously_invoiced", "ordered"],
    "qty_exceeds_received": ["po_line", "description", "invoiced", "previously_invoiced", "received"],
    "unmatched_line": ["line", "description", "candidate_description", "confidence"],
    "unverified:line_match": ["line", "description", "candidate_description", "confidence"],
    "unverified:line_values": ["line", "po_line", "description", "quantity", "unit_price", "would_flag"],
    "line_math_mismatch": ["line", "quantity", "unit_price", "amount", "expected"],
    "short_shipment": ["line", "description", "ordered", "shipped"],
}
HEADERS = {"po_line": "PO line", "po": "PO price", "delta_pct": "delta %", "candidate_description": "closest PO line",
           "previously_invoiced": "billed earlier", "unit_price": "unit price", "would_flag": "would flag"}
# Line-item column key (schema `lines` block) -> table header.
ITEM_COLUMNS = {"quantities": "quantity", "unit_prices": "unit price", "amounts": "amount", "ordered": "ordered",
                "shipped": "shipped"}
CONTEXT = 25


def humanise(check_id: str) -> str:
    return check_id.replace("_", " ").replace(":", ": ")


def check_label(check_id: str, passed: bool = False) -> str:
    """A reader-facing name for a check: what the flag means when it failed, what was established when it passed."""
    table = PASSED if passed else LABELS
    if check_id in table:
        return table[check_id]
    kind, _, name = check_id.partition(":")
    prefixes = PASSED_PREFIXES if passed else PREFIXES
    if name and kind in prefixes:
        if kind == "unverified":
            inner = PASSED.get(name) or humanise(name)
            if len(inner) > 1 and inner[1].islower():
                inner = inner[0].lower() + inner[1:]
            return f"{prefixes[kind]}: {inner}"
        return f"{prefixes[kind]}: {name}"
    return humanise(check_id)


def split_reasons(reasons: list[str]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Route reasons (`<id>: <detail>`, already ordered by weight) as (id, detail) pairs: business flags first,
    extraction / input quality flags second."""
    business: list[tuple[str, str]] = []
    quality: list[tuple[str, str]] = []
    for reason in reasons:
        check_id, _, detail = reason.partition(": ")
        (quality if is_quality_flag(check_id) else business).append((check_id, detail))
    return business, quality


def truth_caption(route: str, flags: set[str], truth: dict[str, Any]) -> tuple[str, bool]:
    """Compare a result with the sample's ground truth on route and business flags."""
    expected = set(truth["expected_flags"])
    defects = ", ".join(sorted(expected)) or "no seeded defects"
    head = f"Expected: {truth['expected_route'].replace('_', ' ')} ({defects})"
    problems = []
    if route != truth["expected_route"]:
        problems.append(f"routed to {route.replace('_', ' ')}")
    if expected - flags:
        problems.append("missed " + ", ".join(sorted(expected - flags)))
    if flags - expected:
        problems.append("extra " + ", ".join(sorted(flags - expected)))
    if problems:
        return f"{head}: differs: {'; '.join(problems)}", False
    return f"{head}: matched", True


# ------------------------------------------------------------------------------------------------ evidence
def snippet(text: str, start: int, end: int) -> str:
    return (f"...{text[max(0, start - CONTEXT):start]}[{text[start:end]}]{text[end:end + CONTEXT]}..."
            .replace("\n", " / "))


def number_forms(value: Any) -> list[str]:
    """Ways a document prints a number, most specific first: 1,234.56 | 1.234,56 | 1234.56 | 1234,56 | 1,234 | 1234."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return [str(value)]
    us = f"{v:,.2f}"
    forms = [us, us.translate(str.maketrans(",.", ".,")), f"{v:.2f}", f"{v:.2f}".replace(".", ",")]
    if v.is_integer():
        forms += [f"{int(v):,}", f"{int(v):,}".replace(",", "."), str(int(v))]
    else:
        forms.append(f"{v:g}")
    return list(dict.fromkeys(forms))


def date_forms(value: Any) -> list[str]:
    """Ways a document prints a date: ISO, `September 15, 2026`, `15 Sep 2026`, `15.09.2026`, `09/15/2026`, ..."""
    try:
        d = date.fromisoformat(str(value)[:10])
    except ValueError:
        return [str(value)]
    return list(dict.fromkeys([
        d.isoformat(), f"{d:%B} {d.day}, {d.year}", f"{d:%b} {d.day}, {d.year}", f"{d:%B %d, %Y}", f"{d:%b %d, %Y}",
        f"{d.day} {d:%B} {d.year}", f"{d.day} {d:%b} {d.year}", f"{d:%d %B %Y}", f"{d:%d %b %Y}",
        f"{d:%d.%m.%Y}", f"{d:%d/%m/%Y}", f"{d:%m/%d/%Y}", f"{d:%d-%m-%Y}", f"{d:%Y/%m/%d}",
    ]))


def _find_standalone(text: str, needle: str) -> int | None:
    """First occurrence of `needle` that is not part of a longer number or word (`95.00` is not inside `195.00`,
    `1,234` is not the head of `1,234.56`)."""
    if not needle:
        return None
    pos = text.find(needle)
    while pos != -1:
        end = pos + len(needle)
        before, before2 = text[pos - 1:pos], text[pos - 2:pos - 1] if pos > 1 else ""
        after, after2 = text[end:end + 1], text[end + 1:end + 2]
        glued_before = before.isalnum() or (before in ".," and before2.isdigit())
        glued_after = after.isalnum() or (after in ".," and after2.isdigit())
        if not glued_before and not glued_after:
            return pos
        pos = text.find(needle, pos + 1)
    return None


def find_evidence(text: str, value: Any, kind: str) -> str:
    """The first place `text` prints `value`, as a bracketed snippet. Number and date fields carry no Clairic span,
    so the value is searched for in the ways documents print it; anything else is searched for verbatim."""
    if not text or value is None or isinstance(value, bool | list | dict):
        return ""
    forms = number_forms(value) if kind == "number" else date_forms(value) if kind == "date" else [str(value)]
    for form in forms:
        pos = _find_standalone(text, form)
        if pos is not None:
            return snippet(text, pos, pos + len(form))
    return ""


def field_evidence(ext: Extraction, name: str, kind: str) -> str:
    """Clairic's span when it gave one (text scalars), else a search for the value; nothing for lists."""
    answer = ext.fields[name]
    text = ext.state_text or ""
    if not text or answer.value is None or answer.is_list:
        return ""
    if answer.span and len(answer.span) == 2:
        start, end = answer.span
        if 0 <= start < end <= len(text):
            return snippet(text, start, end)
    return find_evidence(text, answer.value, kind)


def field_note(answer: FieldAnswer) -> str:
    if not answer.is_list:
        return ""
    return "rebuilt from page rows" if answer.source == "rows" else "n/a (list)"


# ------------------------------------------------------------------------------------------------ tables
def extraction_rows(ext: Extraction, schema: dict[str, Any], truth: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """One row per extracted field. Lists show no confidence (Clairic's joint list probability is not comparable to a
    scalar's) and say where they came from; `truth` adds a `matches truth` column."""
    specs = schema.get("fields", {})
    rows = []
    for name, a in ext.fields.items():
        spec = specs.get(name) or {"type": a.type}
        kind = spec.get("type", a.type)
        value = json.dumps(a.value, ensure_ascii=False) if a.is_list else a.value
        row: dict[str, Any] = {"field": name, "value": None if value is None else str(value),
                               "confidence": None if a.is_list else a.confidence, "p_null": a.p_null,
                               "note": field_note(a), "evidence": field_evidence(ext, name, kind)}
        if truth is not None:
            row["matches truth"] = score_field(spec, truth["fields"].get(name), a.value, name) == 1.0
        rows.append(row)
    return rows


def line_item_rows(line_items: dict[str, Any]) -> list[dict[str, Any]]:
    """`ProcessResult.line_items["rows"]` as table rows: line, item code, description and the schema's numeric
    columns (quantity / unit price / amount, or ordered / shipped)."""
    rows = line_items.get("rows") or []
    numeric = [key for key in ITEM_COLUMNS if any(key in r for r in rows)]
    with_code = any(r.get("item_code") for r in rows)
    out = []
    for r in rows:
        row: dict[str, Any] = {"line": r.get("line")}
        if with_code:
            row["item code"] = r.get("item_code")
        row["description"] = r.get("description")
        row.update({ITEM_COLUMNS[key]: r.get(key) for key in numeric})
        out.append(row)
    return out


def lines_table(check_id: str, lines: list[Any]) -> list[dict[str, Any]]:
    """The rows behind a failed per-line check, limited to the columns worth reading; [] when nothing is tabular."""
    rows = [line for line in lines if isinstance(line, dict)]
    if not rows:
        return []
    columns = LINE_COLUMNS.get(check_id) or [k for k, v in rows[0].items() if v is None or isinstance(v, str | int | float)]
    kept = []
    for column in columns:
        values = [r.get(column) for r in rows]
        if all(v is None for v in values):
            continue  # e.g. no `candidate_description` on an `unmatched_line`
        if column == "previously_invoiced" and not any(values):
            continue
        kept.append(column)
    return [{HEADERS.get(c, c.replace("_", " ")): r.get(c) for c in kept} for r in rows]


# ------------------------------------------------------------------------------------------------ misc
def po_number_from_detail(detail: str) -> str | None:
    """`PO-4501 found (Northwind Supply Co.).` -> `PO-4501`: the passed `po_not_found` check names the PO only there."""
    m = re.match(r"^(?P<po>\S.*?) found \(", detail)
    return m.group("po") if m else None


def erp_url(page: Path, po_number: str) -> str:
    """File URL of the mock ERP page opened on one purchase order (`#/pos/<po>` is the page's own router)."""
    return f"{page.resolve().as_uri()}#/pos/{quote(po_number, safe='')}"


def replayed_from_cache(stage_usage: dict[str, Usage]) -> bool:
    """True when every model request of every stage was answered from the response cache, so the timings say
    nothing about the models (a cached request counts its tokens as cached)."""
    seen = False
    for usage in stage_usage.values():
        for model in ("jev", "clairic"):
            if getattr(usage, f"{model}_requests"):
                seen = True
                tokens, cached = getattr(usage, f"{model}_input_tokens"), getattr(usage, f"{model}_cached_tokens")
                if not tokens or cached < tokens:
                    return False
    return seen
