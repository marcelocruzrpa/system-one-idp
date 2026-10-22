"""Three-way match (PO <-> invoice <-> goods receipt).

Code checks that the invoice comes from the PO's vendor in the PO's currency. Jev decides which PO line
each invoice line refers to (descriptions are worded differently across documents); code then compares
quantities (summed per PO line, plus what earlier invoices billed) and prices against the PO and the
goods receipts. Invoice lines come from `idp.lines` (rows read from the document), never from index-paired lists.

A Jev answer below `line_match_min_confidence` is an uncertain match, reported as `unverified:line_match` (review);
only a confident `no_match` becomes the business flag `unmatched_line`.
"""

import re
from typing import Any

from typesafe_sdk import Choice

from idp.clients import Jev
from idp.config import policy, threshold
from idp.erp import Erp, norm_ref
from idp.lines import reconcile
from idp.models import Check, Extraction, Usage
from idp.validate.rules import doubtful

NO_MATCH = "no_match"
# A bare `$` is ambiguous (USD, CAD, AUD, ...) and is not mapped.
CURRENCY_SYMBOLS = {"€": "EUR", "£": "GBP", "US$": "USD", "¥": "JPY"}


def norm_currency(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    text = CURRENCY_SYMBOLS.get(text, text)
    return text if re.fullmatch(r"[A-Z]{3}", text) else None


def invoice_lines(schema: dict, ext: Extraction) -> list[dict[str, Any]]:
    """The invoice's line items as rows (description, quantity, unit price); empty when no consistent rows exist."""
    table = reconcile(schema, ext)
    if not table.usable:
        return []
    return [{"line": item.line, "description": item.description, "quantity": item.quantity,
             "unit_price": item.unit_price, "item_code": item.item_code} for item in table.items]


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", str(text or "").casefold())).strip()


def exact_matches(lines: list[dict], po: dict) -> dict[int, int]:
    """Invoice line -> PO line when the description (or the item code) is the same text: no model needed."""
    by_text: dict[str, list[int]] = {}
    by_code: dict[str, int] = {}
    for pl in po["lines"]:
        by_text.setdefault(_norm(pl["description"]), []).append(pl["line"])
        if pl.get("sku"):
            by_code[norm_ref(pl["sku"])] = pl["line"]
    found = {}
    for line in lines:
        code = norm_ref(line.get("item_code"))
        candidates = by_text.get(_norm(line["description"]), []) if line.get("description") else []
        if code and code in by_code:
            found[line["line"]] = by_code[code]
        elif len(candidates) == 1:
            found[line["line"]] = candidates[0]
    return found


def match_lines(jev: Jev, lines: list[dict], po: dict) -> tuple[dict[int, dict], Usage]:
    """Map invoice line number -> {po_line, status, choice, confidence, candidate}.

    Lines whose description is the PO line's text are matched in code. For the rest, one Jev Choice per line
    (question `line_<invoice line number>`): `status` is `matched` (po_line set), `no_match` (Jev is confident the
    item is not on the PO) or `uncertain` (below `line_match_min_confidence`, po_line None, `candidate` = the most
    likely PO line).
    """
    matches: dict[int, dict] = {}
    for number, po_line in exact_matches(lines, po).items():
        matches[number] = {"po_line": po_line, "status": "matched", "choice": f"po_line_{po_line}", "confidence": 1.0,
                           "candidate": po_line, "how": "same text"}
    pending = [l for l in lines if l["line"] not in matches]
    if not pending:
        return matches, Usage()

    criteria = {f"po_line_{pl['line']}": pl["description"] for pl in po["lines"]}
    criteria[NO_MATCH] = "None of the purchase order lines is the same product or service."
    state = {"invoice_lines": [{k: v for k, v in (("line", l["line"]), ("description", l["description"]),
                                                   ("item_code", l.get("item_code"))) if v is not None} for l in pending]}
    questions = {
        f"line_{l['line']}": Choice(
            instructions=f"Which purchase order line is the same product or service as the entry with `line` = {l['line']} "
                         "in `invoice_lines`? Wording may differ (abbreviations, word order, part numbers vs names); "
                         "ignore quantities and prices.",
            criteria=criteria,
        )
        for l in pending
    }
    result = jev.ask(state, questions)
    min_conf = threshold("line_match_min_confidence")
    for line in pending:
        answer = result.choice(f"line_{line['line']}")
        confident = answer["confidence"] >= min_conf
        probabilities = answer.get("probabilities") or {}
        candidate = max((k for k in probabilities if k != NO_MATCH), key=probabilities.get, default=None)
        if answer["choice"] != NO_MATCH:
            candidate = answer["choice"]
        po_line = None
        if answer["choice"] != NO_MATCH and confident:
            status, po_line = "matched", int(answer["choice"].removeprefix("po_line_"))
        elif answer["choice"] == NO_MATCH and confident:
            status = "no_match"
        else:
            status = "uncertain"
        matches[line["line"]] = {"po_line": po_line, "status": status, "choice": answer["choice"],
                                 "confidence": answer["confidence"], "how": "jev",
                                 "candidate": int(candidate.removeprefix("po_line_")) if candidate else None}
    return matches, result.usage()


def supplier_checks(ext: Extraction, po: dict, erp: Erp) -> list[Check]:
    """The invoice must come from the PO's vendor and be billed in the PO's currency."""
    checks = []
    name, tax_id = ext.value("vendor_name"), ext.value("vendor_tax_id")
    candidates = erp.vendor_candidates(name, tax_id)
    by_tax, by_name = candidates["tax_id"], candidates["name"]
    vendor = by_tax or by_name
    if by_tax and by_name and by_tax["vendor_id"] != by_name["vendor_id"]:
        checks.append(Check(id="unverified:vendor_identity", stage="match", passed=False,
                            detail=f"Tax id {tax_id} belongs to {by_tax['name']} but the name {name!r} is {by_name['name']}; "
                                   f"the supplier of PO {po['po_number']} cannot be confirmed."))
    elif vendor is None:
        checks.append(Check(id="unknown_vendor", stage="match", passed=False,
                            detail=f"Vendor {name!r} is not in the ERP vendor master."))
    elif vendor["vendor_id"] != po.get("vendor_id"):
        checks.append(Check(id="po_vendor_mismatch", stage="match", passed=False,
                            detail=f"Invoice is from {vendor['name']}, but PO {po['po_number']} was issued to {po['vendor']}."))
    else:
        checks.append(Check(id="po_vendor_mismatch", stage="match", passed=True,
                            detail=f"Invoice and PO {po['po_number']} are both for {vendor['name']}."))
    currency, po_currency = norm_currency(ext.value("currency")), norm_currency(po.get("currency"))
    if currency is None or po_currency is None:
        checks.append(Check(id="unverified:currency", stage="match", passed=False,
                            detail=f"Cannot compare currencies: invoice {ext.value('currency')!r}, PO {po.get('currency')!r}."))
    else:
        checks.append(Check(id="currency_mismatch", stage="match", passed=currency == po_currency,
                            detail=f"Invoice currency {currency}, PO currency {po_currency}."))
    return checks


def three_way_match(jev: Jev, schema: dict, ext: Extraction, erp: Erp) -> tuple[list[Check], Usage]:
    po_number = ext.value("po_number")
    if not po_number:
        return [], Usage()  # `missing_po` covers this
    po = erp.find_po(po_number)
    if po is None:
        return [Check(id="po_not_found", stage="match", passed=False, detail=f"PO {po_number} is not in the ERP.")], Usage()
    found_po = Check(id="po_not_found", stage="match", passed=True, detail=f"{po['po_number']} found ({po['vendor']}).")
    checks = [found_po, *supplier_checks(ext, po, erp)]
    received = erp.received_quantities(po_number)
    checks.append(Check(id="no_goods_receipt", stage="match", passed=bool(received),
                        detail=f"{'Goods receipt' if received else 'No goods receipt'} recorded for PO {po['po_number']}."))
    lines = invoice_lines(schema, ext)
    if not lines:
        checks.append(Check(id="unverified:three_way_match", stage="match", passed=False,
                            detail="No invoice lines could be read, so nothing was matched to the PO."))
        return checks, Usage()
    currency_differs = any(c.id == "currency_mismatch" and not c.passed for c in checks)

    matches, usage = match_lines(jev, lines, po)
    found_po.data = {"matches": matches}
    po_lines = {pl["line"]: pl for pl in po["lines"]}
    tolerances = policy()["tolerances"]
    price_pct, qty_tol = tolerances["price_pct"], tolerances.get("quantity_abs", 1e-6)
    issues: dict[str, list[dict]] = {"unmatched_line": [], "unverified:line_match": [], "unverified:line_values": [],
                                     "price_mismatch": [], "qty_exceeds_po": [], "qty_exceeds_received": []}

    # Several invoice lines can bill the same PO line, so quantities are compared per PO line.
    billed: dict[int, dict[str, Any]] = {}
    for line in lines:
        match = matches[line["line"]]
        pl = po_lines.get(match["po_line"]) if match["po_line"] else None
        if pl is None:
            bucket = "unmatched_line" if match["status"] == "no_match" else "unverified:line_match"
            entry = {"line": line["line"], "description": line["description"], **match}
            if bucket == "unverified:line_match" and match["candidate"] in po_lines:
                entry["candidate_description"] = po_lines[match["candidate"]]["description"]
            issues[bucket].append(entry)
            continue
        ref = {"line": line["line"], "po_line": pl["line"], "description": line["description"]}
        qty, price = line["quantity"], line["unit_price"]
        if qty is None or price is None:
            issues["unverified:line_values"].append({**ref, "quantity": qty, "unit_price": price})
        # Prices in different currencies are not comparable; the currency flag already sends the document to review.
        if price is not None and not currency_differs and abs(price - pl["unit_price"]) > price_pct * pl["unit_price"]:
            delta = (price - pl["unit_price"]) / pl["unit_price"] if pl["unit_price"] else None
            issues["price_mismatch"].append({**ref, "invoiced": price, "po": pl["unit_price"],
                                             "delta_pct": None if delta is None else round(100 * delta, 1)})
        if qty is not None:
            total = billed.setdefault(pl["line"], {"po_line": pl["line"], "description": pl["description"],
                                                   "invoice_lines": [], "invoiced": 0.0})
            total["invoice_lines"].append(line["line"])
            total["invoiced"] += qty

    for total in billed.values():
        pl = po_lines[total["po_line"]]
        # Quantity billed on earlier invoices against this PO line counts too.
        cumulative = total["invoiced"] + pl.get("invoiced", 0)
        ref = {**total, "invoiced": round(total["invoiced"], 6), "previously_invoiced": pl.get("invoiced", 0)}
        if cumulative - pl["quantity"] > qty_tol:
            issues["qty_exceeds_po"].append({**ref, "ordered": pl["quantity"]})
        got = received.get(pl["line"], 0)
        if received and cumulative - got > qty_tol:
            issues["qty_exceeds_received"].append({**ref, "received": round(got, 6)})

    # Quantities and prices Clairic itself doubted (lists not rebuilt from the document) cannot assert a mismatch.
    cols = schema["lines"]
    weak = doubtful(ext, cols.get("quantities", ""), cols.get("unit_prices", ""))
    if weak:
        for check_id in ("price_mismatch", "qty_exceeds_po", "qty_exceeds_received"):
            if issues[check_id]:
                issues["unverified:line_values"].extend({**f, "would_flag": check_id, "doubtful": weak} for f in issues[check_id])
                issues[check_id] = []

    labels = {
        "unmatched_line": "Invoice line(s) not on the PO",
        "unverified:line_match": "PO line could not be determined confidently for invoice line(s)",
        "unverified:line_values": "Matched invoice line(s) whose quantity or unit price could not be trusted",
        "price_mismatch": f"Unit price differs from the PO by more than {price_pct:.0%}",
        "qty_exceeds_po": "Invoiced quantity exceeds the ordered quantity",
        "qty_exceeds_received": "Invoiced quantity exceeds the received quantity",
    }
    passes = {
        "unmatched_line": "Every invoice line is on the PO.",
        "price_mismatch": f"Unit prices within {price_pct:.0%} of the PO.",
        "qty_exceeds_po": "Invoiced quantities within the ordered quantities.",
        "qty_exceeds_received": "Invoiced quantities within the received quantities.",
    }
    for check_id, found in issues.items():
        if found:
            if check_id.startswith("qty_"):
                where = f"PO lines {[f['po_line'] for f in found]}"
            elif check_id == "price_mismatch":
                where = "; ".join(f"line {f['line']} {f['invoiced']:.2f} vs PO {f['po']:.2f}"
                                  + (f" ({f['delta_pct']:+.1f}%)" if f.get("delta_pct") is not None else "") for f in found)
            elif check_id == "unverified:line_match":
                where = "; ".join(f"line {f['line']} {f['description']!r} -> probably PO line {f['candidate']} "
                                  f"(p={f['confidence']:.2f})" if f.get("candidate") else
                                  f"line {f['line']} {f['description']!r} (p={f['confidence']:.2f})" for f in found)
            else:
                where = f"lines {[f['line'] for f in found]}"
            checks.append(Check(id=check_id, stage="match", passed=False, detail=f"{labels[check_id]}: {where}.",
                                data={"lines": found}))
        elif check_id in passes:
            checks.append(Check(id=check_id, stage="match", passed=True, detail=passes[check_id]))
    return checks, usage
