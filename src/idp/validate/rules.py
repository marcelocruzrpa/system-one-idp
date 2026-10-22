"""Deterministic checks: arithmetic, dates, required fields and ERP lookups. Math stays in code, not in the model.

A rule returns no checks when its inputs are missing; `run_rules` then raises `unverified:<rule>` (the rule's
`flag` if it has one) so a check that could not run never looks like one that passed. Schema rules marked
`optional: true` may be skipped silently.

Two more conventions keep model doubt from turning into accusations against the document:
- Line items come from `idp.lines` (rows parsed from the text, Clairic's lists as cross-check). When no consistent
  rows exist, one `unverified:line_items` check replaces every per-line rule.
- An arithmetic rule that FAILS on a value Clairic read below `rule_input_min_confidence` reports `unverified:<rule>`
  instead of a business flag; a rule that reconciles still passes (the arithmetic corroborates its inputs).
"""

from collections.abc import Callable
from datetime import date
from typing import Any

from idp.config import policy, threshold
from idp.erp import Erp, norm_ref
from idp.lines import LineTable, reconcile
from idp.models import Check, Extraction


def as_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def numbers(values: Any) -> list[float | None]:
    return [as_number(v) for v in values] if isinstance(values, list) else []


def close(a: float, b: float, tol: float | None = None) -> bool:
    tol = policy()["tolerances"]["amount_abs"] if tol is None else tol
    return abs(a - b) <= tol + 1e-9


def _fail(check_id: str, detail: str, **data: Any) -> Check:
    return Check(id=check_id, stage="rules", passed=False, detail=detail, data=data)


def _pass(check_id: str, detail: str, inputs: list[str] | None = None) -> Check:
    """A passed check; `inputs` names the fields this particular pass rested on (for the verifier to withdraw it)."""
    return Check(id=check_id, stage="rules", passed=True, detail=detail, data={"inputs": inputs} if inputs else {})


def present(ext: Extraction, name: str) -> bool:
    """Clairic returned a value and does not think the field is absent (the same test as the required-field check)."""
    answer = ext.fields.get(name)
    return answer is not None and answer.value is not None and (answer.p_null or 0) <= threshold("field_max_p_null")


def doubtful(ext: Extraction, *names: str) -> dict[str, float]:
    """Inputs Clairic itself read below `rule_input_min_confidence`, lowest first.

    Number lists use their per-item confidences (or the list confidence when Clairic gave none); text lists are never
    judged by confidence (Clairic reports one joint probability per list, near zero even when every item is right);
    columns rebuilt from the document's table rows are not model values at all.
    """
    floor = threshold("rule_input_min_confidence")
    weak: dict[str, float] = {}
    for name in names:
        answer = ext.fields.get(name) if name else None
        if answer is None or answer.value is None or answer.source == "rows":
            continue
        if answer.is_list:
            if any(isinstance(v, str) for v in answer.value):
                continue
            confidence = answer.item_confidence()
            confidence = answer.confidence if confidence is None else confidence
        else:
            confidence = answer.confidence
        if confidence is not None and confidence < floor:
            weak[name] = confidence
    return dict(sorted(weak.items(), key=lambda kv: kv[1]))


def _fail_or_unverified(rule: str, check_id: str, detail: str, ext: Extraction, inputs: tuple[str, ...], **data: Any) -> Check:
    """The business flag, unless one of the rule's inputs is a value the model itself doubts."""
    weak = doubtful(ext, *inputs)
    if not weak:
        return _fail(check_id, detail, **data)
    name, confidence = next(iter(weak.items()))
    value = ext.value(name)
    shown = f"{len(value)} items" if isinstance(value, list) else repr(value)
    return _fail(f"unverified:{rule}", f"{detail} Not asserted: `{name}` = {shown} was read at confidence "
                 f"{confidence:.2f}, so the extraction is the likelier error.", doubtful=weak, **data)


def check_required(schema: dict, ext: Extraction) -> list[Check]:
    """Missing or low-confidence required fields (extraction quality, not business flags)."""
    min_conf = threshold("field_min_confidence")
    checks = []
    for name in schema.get("required", []):
        answer = ext.fields.get(name)
        if not present(ext, name):
            p_null = None if answer is None else answer.p_null
            checks.append(Check(id=f"missing:{name}", stage="extract", passed=False,
                                detail=f"Required field `{name}` not found (p_null={p_null}).", data={"p_null": p_null}))
        elif not answer.is_list and answer.confidence is not None and answer.confidence < min_conf:
            checks.append(Check(id=f"low_confidence:{name}", stage="extract", passed=False,
                                detail=f"`{name}` = {answer.value!r} has low confidence {answer.confidence:.2f}.",
                                probability=round(1 - answer.confidence, 4)))
    if not checks:
        checks.append(Check(id="required_fields", stage="extract", passed=True, detail="All required fields extracted confidently."))
    return checks


def check_list_items(schema: dict, ext: Extraction) -> list[Check]:
    """Number lists whose per-item confidences Clairic reported below `field_min_confidence`."""
    min_conf = threshold("field_min_confidence")
    checks = []
    for name, spec in schema.get("fields", {}).items():
        answer = ext.fields.get(name)
        if spec.get("type") != "list" or answer is None or not answer.confidences or answer.source == "rows":
            continue
        low = [i + 1 for i, c in enumerate(answer.confidences) if c < min_conf]
        if low:
            checks.append(Check(id=f"low_confidence:{name}", stage="extract", passed=False,
                                detail=f"`{name}` items {low} were read below confidence {min_conf:.2f} "
                                       f"(lowest {min(answer.confidences):.2f}).",
                                probability=round(1 - min(answer.confidences), 4), data={"items": low}))
    return checks


def line_table(schema: dict, ext: Extraction) -> LineTable:
    return reconcile(schema, ext)


def line_items_check(schema: dict, ext: Extraction) -> list[Check]:
    """One extraction-quality check for the line items: aligned rows, or `unverified:line_items` (review)."""
    table = line_table(schema, ext)
    if not table.relevant:
        return []
    if table.usable:
        return [Check(id="line_items", stage="extract", passed=True, detail=table.detail,
                      data={"source": table.source, "lengths": table.lengths, "agreement": table.agreement,
                            "dropped": table.dropped})]
    return [Check(id="unverified:line_items", stage="extract", passed=False,
                  detail=f"{table.detail} Per-line checks (line math, line sums, PO matching) were not run.",
                  data={"lengths": table.lengths})]


def line_amounts(schema: dict, ext: Extraction) -> list[float]:
    table = line_table(schema, ext)
    return [a for a in table.column("amounts") if a is not None] if table.usable else []


def _tax(ext: Extraction) -> float | None:
    """The extracted tax, or None when Clairic did not find one (a missing tax line is not a zero tax)."""
    return as_number(ext.value("tax_amount")) if present(ext, "tax_amount") else None


def totals(schema: dict, ext: Extraction, erp: Erp, rule: dict) -> list[Check]:
    subtotal, total = as_number(ext.value("subtotal")), as_number(ext.value("total"))
    tax = _tax(ext)
    label, amounts = "Subtotal", line_amounts(schema, ext)
    if subtotal is None and amounts:
        subtotal, label = sum(amounts), "Line amounts"
    if subtotal is None or total is None:
        return []
    inputs = ("subtotal", "tax_amount", "total", schema.get("lines", {}).get("amounts", ""))
    if tax is None:
        if close(subtotal, total):
            return [_pass("total_mismatch", f"{label} {subtotal:.2f} = total {total:.2f} (no tax line found).")]
        return []  # a missed tax line or a wrong total: cannot tell -> unverified:totals
    expected = subtotal + tax
    if close(expected, total):
        return [_pass("total_mismatch", f"{label} {subtotal:.2f} + tax {tax:.2f} = total {total:.2f}.")]
    return [_fail_or_unverified("totals", "total_mismatch",
                                f"{label} {subtotal:.2f} + tax {tax:.2f} = {expected:.2f}, but total is {total:.2f}.",
                                ext, inputs, expected=round(expected, 2), total=total)]


def line_sum(schema: dict, ext: Extraction, erp: Erp, rule: dict) -> list[Check]:
    amounts = line_amounts(schema, ext)
    subtotal, total = as_number(ext.value("subtotal")), as_number(ext.value("total"))
    tax = _tax(ext)
    # Line amounts are before tax: without a subtotal, compare them with the total less tax.
    if subtotal is not None:
        target_name, target = "subtotal", subtotal
    elif total is not None and tax is not None:
        target_name, target = "total less tax", total - tax
    elif total is not None and close(sum(amounts), total):
        target_name, target = "total", total  # no tax line found and the lines add up to the total: a tax-free document
    else:
        return []
    if not amounts or target is None:
        return []
    line_total = sum(amounts)
    if close(line_total, target):
        return [_pass("line_sum_mismatch", f"{len(amounts)} line amounts sum to the {target_name} {target:.2f}.")]
    inputs = (schema.get("lines", {}).get("amounts", ""), "subtotal", "tax_amount", "total")
    return [_fail_or_unverified("line_sum", "line_sum_mismatch",
                                f"{len(amounts)} line amounts sum to {line_total:.2f}, but the {target_name} is {target:.2f}.",
                                ext, inputs, line_total=round(line_total, 2), target=target)]


def line_math(schema: dict, ext: Extraction, erp: Erp, rule: dict) -> list[Check]:
    table = line_table(schema, ext)
    if not table.usable or not table.items:
        return []
    bad, incomplete = [], []
    for item in table.items:
        q, p, a = item.quantity, item.unit_price, item.amount
        if None in (q, p, a):
            incomplete.append(item.line)
        elif not close(q * p, a, tol=0.01 + 0.0005 * abs(a)):
            bad.append({"line": item.line, "quantity": q, "unit_price": p, "amount": a, "expected": round(q * p, 2)})
    checks = []
    cols = schema["lines"]
    if bad:
        checks.append(_fail_or_unverified("line_math", "line_math_mismatch",
                                          f"Quantity x unit price != amount on line(s) {[b['line'] for b in bad]}.", ext,
                                          (cols.get("quantities", ""), cols.get("unit_prices", ""), cols.get("amounts", "")),
                                          lines=bad))
    elif len(incomplete) < len(table.items):
        checks.append(_pass("line_math_mismatch", "Quantity x unit price = amount on every complete line."))
    if incomplete:
        checks.append(_fail("unverified:line_math", f"Line(s) {incomplete} lack a quantity, unit price or amount.",
                            lines=incomplete))
    return checks


def date_order(schema: dict, ext: Extraction, erp: Erp, rule: dict) -> list[Check]:
    earlier, later = as_date(ext.value(rule["earlier"])), as_date(ext.value(rule["later"]))
    if earlier is None or later is None:
        return []
    if later < earlier:
        return [_fail(rule["flag"], f"`{rule['later']}` {later} is before `{rule['earlier']}` {earlier}.")]
    return [_pass(rule["flag"], f"`{rule['earlier']}` {earlier} <= `{rule['later']}` {later}.")]


def not_future(schema: dict, ext: Extraction, erp: Erp, rule: dict, today: date | None = None) -> list[Check]:
    today = today or policy().get("reference_date") or date.today()
    today = as_date(today) or date.today()
    checks = []
    for name in rule["fields"]:
        value = as_date(ext.value(name))
        if value and value > today:
            checks.append(_fail("future_date", f"`{name}` {value} is in the future.", field=name))
        elif value:
            checks.append(_pass("future_date", f"`{name}` {value} is not in the future."))
    return checks


def duplicate_invoice(schema: dict, ext: Extraction, erp: Erp, rule: dict) -> list[Check]:
    """A duplicate is the same invoice number from the same vendor; another vendor may use the same number."""
    number = ext.value("invoice_number")
    if not number:
        return []
    if not erp.posted_invoices(number):
        return [_pass("duplicate_invoice", f"Invoice {number} has not been posted before.", inputs=["invoice_number"])]
    vendor = erp.resolve_vendor(ext.value("vendor_name"), ext.value("vendor_tax_id"))
    if vendor is None:
        return [_fail("unverified:duplicate_invoice", f"Invoice number {number} is already posted, and vendor "
                      f"{ext.value('vendor_name')!r} could not be identified in the vendor master, so this may be a duplicate.")]
    posted = erp.posted_invoices(number, vendor["vendor_id"])
    if posted:
        return [_fail("duplicate_invoice", f"Invoice {number} was already posted on {posted[0].get('posted_on')} "
                      f"({posted[0].get('vendor')}, {posted[0].get('amount')}).", posted=posted[0])]
    return [_pass("duplicate_invoice", f"Invoice {number} has not been posted for {vendor['name']} "
                  "(only another vendor used this number).", inputs=["invoice_number", "vendor_name", "vendor_tax_id"])]


def vendor_identity(schema: dict, ext: Extraction, erp: Erp, rule: dict) -> list[Check]:
    """The extracted vendor tax id must belong to the seller: not to us, and not to a different vendor than the name."""
    tax_id, name = ext.value("vendor_tax_id"), ext.value("vendor_name")
    if not norm_ref(tax_id):
        return []
    ours = norm_ref(policy()["our_company"].get("tax_id"))
    if ours and norm_ref(tax_id) == ours:
        return [_fail("unverified:vendor_identity",
                      f"`vendor_tax_id` = {tax_id!r} is our own tax id (read from the bill-to block), not the seller's.")]
    candidates = erp.vendor_candidates(name, tax_id)
    by_tax, by_name = candidates["tax_id"], candidates["name"]
    if by_tax and by_name and by_tax["vendor_id"] != by_name["vendor_id"]:
        return [_fail("unverified:vendor_identity", f"Tax id {tax_id} belongs to {by_tax['name']} in the vendor master, "
                      f"but the name {name!r} is {by_name['name']}.")]
    if by_name and not by_tax and by_name.get("tax_id"):
        return [_fail("unverified:vendor_identity", f"{by_name['name']} is on file with tax id {by_name['tax_id']}, "
                      f"but the invoice shows {tax_id}.")]
    vendor = by_tax or by_name
    if vendor is None:
        return [_pass("vendor_identity", f"Tax id {tax_id} and name {name!r} are both unknown to the vendor master.")]
    return [_pass("vendor_identity", f"Tax id {tax_id} and name {name!r} both identify {vendor['name']}.")]


def require_po(schema: dict, ext: Extraction, erp: Erp, rule: dict) -> list[Check]:
    if not policy().get("require_po_for_invoices"):
        return [_pass("missing_po", "Policy does not require a PO number on invoices.")]
    if ext.value("po_number"):
        return [_pass("missing_po", f"References PO {ext.value('po_number')}.")]
    return [_fail("missing_po", "Policy requires a PO number on invoices, but none was found.")]


def original_invoice_exists(schema: dict, ext: Extraction, erp: Erp, rule: dict) -> list[Check]:
    number = ext.value("original_invoice_number")
    if not number:
        return [_fail("original_invoice_not_found", "No original invoice number found on the credit note.")]
    vendor = erp.resolve_vendor(ext.value("vendor_name"), ext.value("vendor_tax_id"))
    if vendor is None:
        return [_fail("unverified:original_invoice_exists",
                      f"Vendor {ext.value('vendor_name')!r} could not be identified in the vendor master, so original "
                      f"invoice {number} cannot be looked up.")]
    if erp.posted_invoices(number, vendor["vendor_id"]):
        return [_pass("original_invoice_not_found", f"Original invoice {number} from {vendor['name']} is posted in the ERP.")]
    return [_fail("original_invoice_not_found", f"Original invoice {number} from {vendor['name']} is not in the ERP.")]


def short_shipment(schema: dict, ext: Extraction, erp: Erp, rule: dict) -> list[Check]:
    table = line_table(schema, ext)
    if not table.usable or not table.items:
        return []
    short, incomplete = [], []
    for item in table.items:
        ordered, shipped = item.values.get("ordered"), item.values.get("shipped")
        if ordered is None or shipped is None:
            incomplete.append(item.line)
        elif shipped < ordered:
            short.append({"line": item.line, "description": item.description, "ordered": ordered, "shipped": shipped})
    checks = []
    cols = schema["lines"]
    if short:
        checks.append(_fail_or_unverified("short_shipment", "short_shipment",
                                          f"Shipped less than ordered on line(s) {[s['line'] for s in short]}.", ext,
                                          (cols.get("ordered", ""), cols.get("shipped", "")), lines=short))
    elif len(incomplete) < len(table.items):
        checks.append(_pass("short_shipment", f"Every one of {len(table.items)} lines shipped in full."))
    if incomplete:
        checks.append(_fail("unverified:short_shipment", f"Line(s) {incomplete} lack an ordered or shipped quantity.",
                            lines=incomplete))
    return checks


def balance(schema: dict, ext: Extraction, erp: Erp, rule: dict) -> list[Check]:
    opening, credits, debits, closing = (as_number(ext.value(n)) for n in
                                         ("opening_balance", "total_credits", "total_debits", "closing_balance"))
    if None in (opening, credits, debits, closing):
        return []
    # Statements print debits as positive totals or as negative amounts: either way they reduce the balance.
    expected = opening + abs(credits) - abs(debits)
    if close(expected, closing):
        return [_pass("balance_mismatch", f"Opening {opening:.2f} + credits {abs(credits):.2f} - debits {abs(debits):.2f} "
                      f"= closing {closing:.2f}.")]
    return [_fail_or_unverified("balance", "balance_mismatch",
                                f"Opening + credits - debits = {expected:.2f}, but closing balance is {closing:.2f}.", ext,
                                ("opening_balance", "total_credits", "total_debits", "closing_balance"),
                                expected=round(expected, 2), closing=closing)]


def remittance_sum(schema: dict, ext: Extraction, erp: Erp, rule: dict) -> list[Check]:
    paid = [p for p in numbers(ext.value("paid_amounts")) if p is not None]
    total = as_number(ext.value("payment_total"))
    if not paid or total is None:
        return []
    if close(sum(paid), total):
        return [_pass("remittance_sum_mismatch", f"{len(paid)} paid amounts sum to the payment total {total:.2f}.")]
    return [_fail_or_unverified("remittance_sum", "remittance_sum_mismatch",
                                f"Paid amounts sum to {sum(paid):.2f}, but the payment total is {total:.2f}.", ext,
                                ("paid_amounts", "payment_total"))]


RULES: dict[str, Callable[[dict, Extraction, Erp, dict], list[Check]]] = {
    "totals": totals,
    "line_sum": line_sum,
    "line_math": line_math,
    "date_order": date_order,
    "not_future": not_future,
    "duplicate_invoice": duplicate_invoice,
    "vendor_identity": vendor_identity,
    "require_po": require_po,
    "original_invoice_exists": original_invoice_exists,
    "short_shipment": short_shipment,
    "balance": balance,
    "remittance_sum": remittance_sum,
}
# Rules that need model calls run elsewhere (validate/match.py).
MODEL_RULES = {"three_way_match"}
# Rules that consume line items: replaced by the single `unverified:line_items` check when no rows could be built.
LINE_RULES = {"line_sum", "line_math", "short_shipment"}

# The fields each rule asserts on, by rule name: (schema, rule config) -> field names. Used to withdraw a passed rule
# when the Jev verifier rejects one of its inputs.
RULE_INPUTS: dict[str, Callable[[dict, dict], list[str]]] = {
    "totals": lambda s, r: ["subtotal", "tax_amount", "total"],
    "line_sum": lambda s, r: ["subtotal", "tax_amount", "total"],
    "date_order": lambda s, r: [r["earlier"], r["later"]],
    "not_future": lambda s, r: list(r["fields"]),
    "duplicate_invoice": lambda s, r: ["invoice_number", "vendor_name", "vendor_tax_id"],
    "vendor_identity": lambda s, r: ["vendor_name", "vendor_tax_id"],
    "require_po": lambda s, r: ["po_number"],
    "original_invoice_exists": lambda s, r: ["original_invoice_number", "vendor_name"],
    "balance": lambda s, r: ["opening_balance", "total_credits", "total_debits", "closing_balance"],
    "remittance_sum": lambda s, r: ["paid_amounts", "payment_total"],
}
RULE_CHECK_IDS: dict[str, Callable[[dict], list[str]]] = {
    "totals": lambda r: ["total_mismatch"], "line_sum": lambda r: ["line_sum_mismatch"],
    "date_order": lambda r: [r["flag"]], "not_future": lambda r: ["future_date"],
    "duplicate_invoice": lambda r: ["duplicate_invoice"], "vendor_identity": lambda r: ["vendor_identity"],
    "require_po": lambda r: ["missing_po"], "original_invoice_exists": lambda r: ["original_invoice_not_found"],
    "balance": lambda r: ["balance_mismatch"], "remittance_sum": lambda r: ["remittance_sum_mismatch"],
}


def demote_unsupported(schema: dict, ext: Extraction, checks: list[Check]) -> list[Check]:
    """A rule that passed on a value the Jev verifier found unsupported (or in the wrong role) is not a pass: the
    document was not checked, the model's reading was. Rewrite it as `unverified:<flag>` with the verifier's verdict."""
    rejected: dict[str, Check] = {}
    for c in checks:
        kind, _, name = c.id.partition(":")
        if not c.passed and c.stage == "semantic" and kind in ("unsupported", "wrong_role") and name:
            rejected.setdefault(name, c)
    if not rejected:
        return checks
    out = []
    for c in checks:
        rule = next((r for r in schema.get("rules", []) if r["check"] in RULE_INPUTS and c.id in RULE_CHECK_IDS[r["check"]](r)), None)
        if c.passed and c.stage == "rules" and rule is not None:
            inputs = c.data.get("inputs") or RULE_INPUTS[rule["check"]](schema, rule)
            bad = [f for f in inputs if f in rejected]
            if bad:
                name, verdict = bad[0], rejected[bad[0]]
                p = f" (P={verdict.probability:.2f})" if verdict.probability is not None else ""
                detail = (f"`{c.id}` passed on `{name}` = {ext.value(name)!r}, but the verifier found that value "
                          f"{'in the wrong role' if verdict.id.startswith('wrong_role') else 'unsupported by the document'}"
                          f"{p}: {c.detail}")
                out.append(Check(id=f"unverified:{c.id}", stage="rules", passed=False, detail=detail,
                                 probability=verdict.probability, data={"input": name, "verifier": verdict.id}))
                continue
        out.append(c)
    return out


def run_rules(schema: dict, ext: Extraction, erp: Erp) -> list[Check]:
    table = line_table(schema, ext)  # first: rebuilds the line-item columns the other checks read
    checks = check_required(schema, ext) + check_list_items(schema, ext) + line_items_check(schema, ext)
    for rule in schema.get("rules", []):
        if rule["check"] in MODEL_RULES:
            continue
        if rule["check"] in LINE_RULES and table.relevant and not table.usable:
            continue  # covered by unverified:line_items
        found = RULES[rule["check"]](schema, ext, erp, rule)
        if not found and not rule.get("optional"):
            found = [_fail(f"unverified:{rule.get('flag', rule['check'])}",
                           f"`{rule['check']}` could not run: the values it needs were not extracted.")]
        checks.extend(found)
    return checks
