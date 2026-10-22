from decimal import Decimal

from idp.erp import Erp
from idp.erp_view import parse_amount, render, view

ERP = Erp({
    "vendors": [{"vendor_id": "V1", "name": "Vendor Supply Company", "tax_id": "11-1111111"}],
    "purchase_orders": [{"po_number": "PO-1", "vendor_id": "V1", "vendor": "Vendor Supply Company", "currency": "USD",
                         "lines": [{"line": 1, "description": "Servo <script>alert(1)</script>", "quantity": 10,
                                    "unit_price": 129.5},
                                   {"line": 2, "description": "Gearbox", "quantity": 6, "unit_price": 412.0}]}],
    "goods_receipts": [{"po_number": "PO 1", "lines": [{"line": 1, "received": 8}, {"line": 2, "received": 6}]}],
    "posted_invoices": [{"invoice_number": "INV-9", "vendor_id": "V1", "vendor": "Vendor Supply Company",
                         "amount": "USD 1,486.22", "posted_on": "2026-08-07"},
                        {"invoice_number": "INV-10", "vendor_id": "V1", "vendor": "Vendor Supply Company",
                         "amount": "n/a", "posted_on": "2026-08-09"}],
})


def test_view_computes_totals_and_statuses():
    po = view(ERP)["purchase_orders"][0]
    assert po["net"] == 3767  # 10 x 129.50 + 6 x 412.00
    assert [line["open"] for line in po["lines"]] == [2, 0]
    assert (po["receipt_status"], po["invoice_status"]) == ("Partially received", "Not invoiced")


def test_parse_amount_reads_currency_and_number():
    assert parse_amount("USD 904.35") == ("USD", Decimal("904.35"))
    assert parse_amount("EUR 3,120.00") == ("EUR", Decimal("3120.0"))
    assert parse_amount("2.472,00 EUR") == ("EUR", Decimal("2472.0"))
    assert parse_amount(12.5) == ("", Decimal("12.5"))
    assert parse_amount("n/a") == ("", None)
    assert parse_amount(None) == ("", None)


def test_page_shows_po_and_escapes_text():
    html = render(ERP)
    assert "PO-1" in html and "3,767.00" in html and "2 open" in html
    assert "<script>alert(1)" not in html and "&lt;script&gt;alert(1)" in html


def test_page_formats_posted_invoice_amounts_and_has_no_dead_controls():
    html = render(ERP)
    assert '<span class="cur">USD</span>1,486.22' in html
    assert "INV-10" in html and "n/a" in html  # an unparsable amount is shown as it is
    assert "<span>Purchase orders</span><b>1</b>" in html and "Open purchase orders" not in html
    assert "Export" not in html and "New purchase order" not in html
    assert "fonts.googleapis.com" not in html


def test_real_erp_renders_every_po():
    erp = Erp.load()
    html = render(erp)
    assert erp.purchase_orders and all(f'id="po-{po["po_number"]}"' in html for po in erp.purchase_orders.values())
