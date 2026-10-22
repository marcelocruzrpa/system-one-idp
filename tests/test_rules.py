from idp.config import schema as load_schema
from idp.erp import Erp, norm_ref
from idp.models import Check, Extraction, FieldAnswer
from idp.route import route
from idp.validate.rules import run_rules


def ext(**values) -> Extraction:
    return Extraction(fields={k: FieldAnswer(type="x", value=v, confidence=0.95, p_null=0.01 if v is not None else 0.9)
                              for k, v in values.items()})


def failed(checks) -> set[str]:
    return {c.id for c in checks if not c.passed}


ERP = Erp({
    "vendors": [{"vendor_id": "V1", "name": "Vendor Supply Company", "tax_id": "11-1111111"},
                {"vendor_id": "V2", "name": "Other Vendor GmbH"}],
    "purchase_orders": [{"po_number": "PO-1", "vendor_id": "V1", "vendor": "Vendor Supply Company", "currency": "USD",
                         "lines": [{"line": 1, "description": "x", "quantity": 2, "unit_price": 5.0}]}],
    "posted_invoices": [{"invoice_number": "INV-9", "vendor_id": "V1", "vendor": "V", "amount": "1", "posted_on": "2026-01-01"},
                        {"invoice_number": "INV-7", "vendor_id": "V2", "vendor": "O", "amount": "1", "posted_on": "2026-01-01"}],
})

GOOD_INVOICE = dict(invoice_number="INV-1", issue_date="2026-09-01", due_date="2026-10-01", vendor_name="Vendor Supply Co.",
                    bill_to_name="Acme Robotics Inc.", po_number="PO-1", currency="USD", subtotal=10.0, tax_amount=0.85,
                    total=10.85, line_descriptions=["x"], line_quantities=[2], line_unit_prices=[5.0], line_amounts=[10.0])
LINE_FIELDS = ("line_descriptions", "line_quantities", "line_unit_prices", "line_amounts")


def test_clean_invoice_passes():
    assert failed(run_rules(load_schema("invoice"), ext(**GOOD_INVOICE), ERP)) == set()


def test_invoice_defects():
    bad = {**GOOD_INVOICE, "invoice_number": "INV 9", "due_date": "2026-08-01", "total": 11.85, "po_number": None,
           "line_amounts": [12.0]}
    assert failed(run_rules(load_schema("invoice"), ext(**bad), ERP)) == {
        "duplicate_invoice", "due_before_issue", "total_mismatch", "missing_po", "line_sum_mismatch", "line_math_mismatch"}


def test_missing_required_field():
    checks = run_rules(load_schema("invoice"), ext(**{**GOOD_INVOICE, "vendor_name": None}), ERP)
    assert "missing:vendor_name" in failed(checks)


def test_rules_that_cannot_run_are_unverified_not_passed():
    no_lines = {**GOOD_INVOICE, **dict.fromkeys(LINE_FIELDS), "subtotal": None}
    flags = failed(run_rules(load_schema("invoice"), ext(**no_lines), ERP))
    # One quality flag stands in for every per-line rule; none of them may look like a pass.
    assert {"unverified:line_items", "unverified:totals"} <= flags
    assert not {"line_sum_mismatch", "line_math_mismatch"} & flags
    gap = {**GOOD_INVOICE, "line_quantities": [None]}
    assert "unverified:line_math" in failed(run_rules(load_schema("invoice"), ext(**gap), ERP))


def test_misaligned_lists_without_table_text_are_unverified_not_mismatched():
    # Clairic returned a stray description: five descriptions for three numeric rows and no text to realign them.
    stray = {**GOOD_INVOICE, "line_descriptions": ["Tax", "x", "y"], "line_quantities": [2], "line_unit_prices": [5.0],
             "line_amounts": [10.0]}
    flags = failed(run_rules(load_schema("invoice"), ext(**stray), ERP))
    assert "unverified:line_items" in flags
    assert not {"line_math_mismatch", "line_sum_mismatch", "line_count_mismatch"} & flags


def test_rows_are_read_from_the_document_text():
    from idp.lines import reconcile
    text = ("Northwind Supply Co. | INVOICE\nTax ID: 77-0421953 | Invoice date | September 8, 2026\n\n"
            "Item | Description | Qty | Unit price | Amount (USD)\n"
            "NW-1 | Servo motor MG996R | 2 | 18.50 | 37.00\n"
            "NW-2 | Stepper driver TMC2209 | 4 | 6.25 | 25.00\n\n"
            "Subtotal | 62.00\nSales tax 8.5% | 5.27\nTotal due (USD) | 67.27\n")
    # Clairic's text list carries a stray ("Tax" from the tax id line) and the SKU column; its numeric lists are right.
    e = ext(**{**GOOD_INVOICE, "line_descriptions": ["Tax", "NW-1", "Servo motor MG996R", "Stepper driver TMC2209"],
               "line_quantities": [2, 4], "line_unit_prices": [18.5, 6.25], "line_amounts": [37.0, 25.0],
               "subtotal": 62.0, "tax_amount": 5.27, "total": 67.27})
    e.state_text = text
    table = reconcile(load_schema("invoice"), e)
    assert table.source == "rows" and [i.description for i in table.items] == ["Servo motor MG996R", "Stepper driver TMC2209"]
    assert table.column("amounts") == [37.0, 25.0] and table.column("quantities") == [2.0, 4.0]
    assert table.dropped == ["Tax"] and [i.item_code for i in table.items] == ["NW-1", "NW-2"]
    assert e.value("line_descriptions") == ["Servo motor MG996R", "Stepper driver TMC2209"]  # written back
    assert e.fields["line_descriptions"].guess == ["Tax", "NW-1", "Servo motor MG996R", "Stepper driver TMC2209"]
    assert failed(run_rules(load_schema("invoice"), e, ERP)) == set()


def test_failed_arithmetic_on_doubted_values_is_unverified():
    bs = ext(bank_name="B", account_holder="A", period_start="2026-09-01", period_end="2026-09-30",
             opening_balance=100.0, total_credits=50.0, total_debits=20.0, closing_balance=131.0)
    bs.fields["total_debits"].confidence = 0.0003
    flags = failed(run_rules(load_schema("bank_statement"), bs, ERP))
    assert "unverified:balance" in flags and "balance_mismatch" not in flags
    # A reconciling statement passes whatever the confidence: the arithmetic corroborates the inputs.
    ok = ext(bank_name="B", account_holder="A", period_start="2026-09-01", period_end="2026-09-30",
             opening_balance=100.0, total_credits=50.0, total_debits=-20.0, closing_balance=130.0)
    ok.fields["total_debits"].confidence = 0.1
    assert failed(run_rules(load_schema("bank_statement"), ok, ERP)) == set()


def test_missing_tax_is_not_zero_tax():
    taxed = {**GOOD_INVOICE, "tax_amount": None, "total": 10.85}
    flags = failed(run_rules(load_schema("invoice"), ext(**taxed), ERP))
    assert "unverified:totals" in flags and "total_mismatch" not in flags
    tax_free = {**GOOD_INVOICE, "tax_amount": None, "total": 10.0}
    assert failed(run_rules(load_schema("invoice"), ext(**tax_free), ERP)) == set()


def test_vendor_identity():
    schema = load_schema("invoice")
    ours = {**GOOD_INVOICE, "vendor_tax_id": "94-3187265"}  # the bill-to party's id (policy.our_company.tax_id)
    assert "unverified:vendor_identity" in failed(run_rules(schema, ext(**ours), ERP))
    conflict = {**GOOD_INVOICE, "vendor_name": "Other Vendor GmbH", "vendor_tax_id": "11-1111111"}
    assert "unverified:vendor_identity" in failed(run_rules(schema, ext(**conflict), ERP))
    assert ERP.resolve_vendor("Other Vendor GmbH", "11-1111111") is None
    fine = {**GOOD_INVOICE, "vendor_tax_id": "11-1111111"}
    assert failed(run_rules(schema, ext(**fine), ERP)) == set()


def test_verifier_rejection_withdraws_a_passed_rule():
    from idp.validate.rules import demote_unsupported
    schema = load_schema("invoice")
    e = ext(**GOOD_INVOICE)
    checks = run_rules(schema, e, ERP)
    checks.append(Check(id="unsupported:due_date", stage="semantic", passed=False, detail="P(problem) = 0.95", probability=0.95))
    demoted = demote_unsupported(schema, e, checks)
    ids = {c.id for c in demoted if not c.passed}
    assert "unverified:due_before_issue" in ids and not any(c.id == "due_before_issue" and c.passed for c in demoted)
    # A pass that never used the rejected field stays a pass: duplicate_invoice passed on the number alone.
    checks = run_rules(schema, e, ERP)
    checks.append(Check(id="unsupported:vendor_tax_id", stage="semantic", passed=False, detail="", probability=0.9))
    assert any(c.id == "duplicate_invoice" and c.passed for c in demote_unsupported(schema, e, checks))


def test_optional_rule_may_skip_and_passing_rules_say_so():
    checks = run_rules(load_schema("invoice"), ext(**{**GOOD_INVOICE, "due_date": None}), ERP)
    assert failed(checks) == set()
    assert any(c.id == "future_date" and c.passed for c in checks)


def test_taxed_invoice_without_subtotal_reconciles_against_total_less_tax():
    checks = run_rules(load_schema("invoice"), ext(**{**GOOD_INVOICE, "subtotal": None}), ERP)
    assert failed(checks) == set()
    assert {c.id for c in checks} >= {"line_sum_mismatch", "total_mismatch"}


def test_duplicate_means_same_number_from_same_vendor():
    schema = load_schema("invoice")
    # INV-7 was posted by another vendor: not a duplicate.
    assert failed(run_rules(schema, ext(**{**GOOD_INVOICE, "invoice_number": "INV-7"}), ERP)) == set()
    # Matched by tax ID even when the name is written differently.
    by_tax_id = {**GOOD_INVOICE, "invoice_number": "INV-9", "vendor_name": "VSC", "vendor_tax_id": "111111111"}
    assert "duplicate_invoice" in failed(run_rules(schema, ext(**by_tax_id), ERP))
    # Unknown vendor and a posted number: can't rule out a duplicate.
    unknown = {**GOOD_INVOICE, "invoice_number": "INV-9", "vendor_name": "Someone Else Ltd"}
    assert "unverified:duplicate_invoice" in failed(run_rules(schema, ext(**unknown), ERP))


def test_receipts_and_postings_accumulate():
    erp = Erp({"goods_receipts": [{"po_number": "PO-1", "lines": [{"line": 1, "received": 1}]},
                                  {"po_number": "po 1", "lines": [{"line": 1, "received": 1}, {"line": 2, "received": 3}]}],
               "posted_invoices": [{"invoice_number": "A-1", "vendor_id": "V1"}, {"invoice_number": "A-1", "vendor_id": "V2"}]})
    assert erp.received_quantities("PO-1") == {1: 2, 2: 3}
    assert len(erp.posted_invoices("A1")) == 2
    assert [p["vendor_id"] for p in erp.posted_invoices("A-1", "V2")] == ["V2"]


def test_resolve_vendor():
    assert ERP.resolve_vendor("VENDOR SUPPLY CO")["vendor_id"] == "V1"
    assert ERP.resolve_vendor("Other Vendor")["vendor_id"] == "V2"
    assert ERP.resolve_vendor("anything", tax_id="11 1111111")["vendor_id"] == "V1"
    assert ERP.resolve_vendor("Vendor Supply Logistics") is None
    assert ERP.resolve_vendor(None) is None


def test_short_shipment_and_balance_and_remittance():
    dn = ext(delivery_note_number="DN", delivery_date="2026-09-01", supplier_name="S", customer_name="C",
             line_descriptions=["a", "b"], line_ordered_quantities=[6, 10], line_shipped_quantities=[6, 8])
    assert "short_shipment" in failed(run_rules(load_schema("delivery_note"), dn, ERP))
    # Lists of different lengths cannot be paired: never a silent pass.
    lopsided = ext(delivery_note_number="DN", delivery_date="2026-09-01", supplier_name="S", customer_name="C",
                   line_ordered_quantities=[6, 10, 8], line_shipped_quantities=[8])
    assert "unverified:line_items" in failed(run_rules(load_schema("delivery_note"), lopsided, ERP))
    bs = ext(bank_name="B", account_holder="A", period_start="2026-09-01", period_end="2026-09-30",
             opening_balance=100.0, total_credits=50.0, total_debits=20.0, closing_balance=131.0)
    assert failed(run_rules(load_schema("bank_statement"), bs, ERP)) == {"balance_mismatch"}
    ra = ext(payer_name="P", payee_name="A", payment_date="2026-10-01", payment_total=100.0, paid_amounts=[60.0, 40.0])
    assert failed(run_rules(load_schema("remittance_advice"), ra, ERP)) == set()


def test_norm_ref():
    assert norm_ref("po 4501") == norm_ref("PO-4501") == "PO4501"


def test_route_priorities():
    ok = Check(id="a", stage="rules", passed=True, detail="")
    review = Check(id="total_mismatch", stage="rules", passed=False, detail="")
    reject = Check(id="duplicate_invoice", stage="rules", passed=False, detail="")
    assert route([ok])[0] == "auto_approve"
    assert route([ok, review])[0] == "review"
    assert route([review, reject])[0] == "reject"
