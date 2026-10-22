"""Mock-ERP web page over samples/erp.json: purchase orders, goods receipts, vendors and posted invoices.

A self-contained HTML file for demos (`idp erp`), so the data the three-way match reads can be shown as the
ERP screens an AP clerk would see. The pipeline never reads it.
"""

import re
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from idp.config import OUT_DIR
from idp.erp import Erp, norm_ref
from idp.lines import parse_number

TEMPLATES = Path(__file__).parent / "templates"
# Status label -> chip colour in the template.
TONES = {"Fully received": "ok", "Partially received": "warn", "Not received": "muted",
         "Fully invoiced": "ok", "Partially invoiced": "info", "Not invoiced": "muted", "Active": "ok", "Posted": "ok"}
# Posted-invoice amounts in erp.json are strings such as `USD 904.35` or `3,120.00 EUR`.
AMOUNT_RE = re.compile(r"^(?:(?P<before>[A-Z]{3})\s*)?(?P<number>[-+]?[\d.,]+)(?:\s*(?P<after>[A-Z]{3}))?$")


def money(x: Any) -> str:
    return f"{Decimal(str(x)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):,.2f}"


def parse_amount(value: Any) -> tuple[str, Decimal | None]:
    """`"USD 904.35"` -> ("USD", Decimal("904.35")); a bare number keeps an empty currency; anything else -> ("", None)."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return "", Decimal(str(value))
    m = AMOUNT_RE.match(str(value or "").strip())
    number = parse_number(m.group("number")) if m else None
    if m is None or number is None:
        return "", None
    return m.group("before") or m.group("after") or "", Decimal(str(number))


def qty(x: Any) -> str:
    return str(int(x)) if float(x).is_integer() else money(x)


def day(value: str | None) -> str:
    """ISO date -> `Sep 8, 2026`."""
    if not value:
        return "—"
    d = date.fromisoformat(value)
    return f"{d:%b} {d.day}, {d.year}"


def _status(done: list[bool], started: bool, verb: str) -> str:
    if all(done):
        return f"Fully {verb}"
    return f"Partially {verb}" if started else f"Not {verb}"


def _purchase_order(erp: Erp, po: dict[str, Any], vendor: dict[str, Any]) -> dict[str, Any]:
    received = erp.received_quantities(po["po_number"])
    lines = []
    for line in po.get("lines", []):
        got = received.get(line["line"], 0)
        lines.append({"line": line["line"], "description": line["description"], "quantity": line["quantity"],
                      "unit_price": line["unit_price"], "received": got, "open": max(line["quantity"] - got, 0),
                      "invoiced": line.get("invoiced", 0),
                      "amount": Decimal(str(line["quantity"])) * Decimal(str(line["unit_price"]))})
    receipts = [{"gr_number": gr.get("gr_number", "—"), "posted_on": gr.get("posted_on"),
                 "lines": len(gr.get("lines", [])), "quantity": sum(l["received"] for l in gr.get("lines", []))}
                for gr in erp.receipts.get(norm_ref(po["po_number"]), [])]
    return {
        "po_number": po["po_number"], "vendor": po.get("vendor") or vendor.get("name", "—"),
        "vendor_id": po.get("vendor_id", "—"), "tax_id": vendor.get("tax_id"), "address": vendor.get("address", []),
        "currency": po.get("currency", ""), "po_date": po.get("po_date"), "delivery_date": po.get("delivery_date"),
        "payment_terms": po.get("payment_terms") or "—", "lines": lines, "receipts": receipts,
        "net": sum((l["amount"] for l in lines), Decimal(0)),
        "ordered_qty": sum(l["quantity"] for l in lines), "open_qty": sum(l["open"] for l in lines),
        "receipt_status": _status([l["open"] == 0 for l in lines], any(l["received"] for l in lines), "received"),
        "invoice_status": _status([l["invoiced"] >= l["quantity"] for l in lines], any(l["invoiced"] for l in lines),
                                  "invoiced"),
    }


def view(erp: Erp) -> dict[str, Any]:
    """Everything the template shows, with totals and statuses computed here rather than in the page."""
    vendors = {v["vendor_id"]: v for v in erp.vendors}
    pos = [_purchase_order(erp, po, vendors.get(po.get("vendor_id"), {})) for po in erp.purchase_orders.values()]
    by_number = {norm_ref(po["po_number"]): po for po in pos}

    net_by_currency: dict[str, Decimal] = {}
    for po in pos:
        net_by_currency[po["currency"]] = net_by_currency.get(po["currency"], Decimal(0)) + po["net"]
    awaiting = [po for po in pos if po["open_qty"]]

    receipts = []
    for po_ref, grs in erp.receipts.items():
        po = by_number.get(po_ref)
        for gr in grs:
            receipts.append({"gr_number": gr.get("gr_number", "—"), "posted_on": gr.get("posted_on"),
                             "po_number": gr["po_number"], "vendor": po["vendor"] if po else "—",
                             "vendor_id": po["vendor_id"] if po else "—", "lines": len(gr.get("lines", [])),
                             "quantity": sum(l["received"] for l in gr.get("lines", [])),
                             "po_status": po["receipt_status"] if po else None})

    vendor_rows = []
    for v in erp.vendors:
        own = [po for po in pos if po["vendor_id"] == v["vendor_id"]]
        vendor_rows.append({"vendor_id": v["vendor_id"], "name": v["name"], "tax_id": v.get("tax_id") or "—",
                            "address": v.get("address", []), "email": v.get("email"),
                            "currency": v.get("currency") or (own[0]["currency"] if own else ""),
                            "po_count": len(own), "po_value": sum((po["net"] for po in own), Decimal(0))})

    invoices = []
    for found in erp.posted.values():
        for inv in found:
            currency, amount = parse_amount(inv.get("amount"))
            invoices.append({"invoice_number": inv["invoice_number"], "vendor": inv.get("vendor", "—"),
                             "vendor_id": inv.get("vendor_id", "—"), "currency": currency, "amount": amount,
                             "amount_text": str(inv.get("amount") or "—"), "posted_on": inv.get("posted_on")})

    return {"purchase_orders": pos, "receipts": receipts, "vendors": vendor_rows, "invoices": invoices,
            "kpis": {"vendor_count": len({po["vendor_id"] for po in pos}), "net_by_currency": net_by_currency,
                     "awaiting": awaiting, "open_units": sum(po["open_qty"] for po in awaiting),
                     "not_invoiced": sum(po["invoice_status"] != "Fully invoiced" for po in pos)}}


def render(erp: Erp) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=True, undefined=StrictUndefined)
    env.filters.update(money=money, qty=qty, day=day)
    return env.get_template("erp.html.j2").render(**view(erp), tones=TONES)


def write(path: Path = OUT_DIR / "erp.html") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(Erp.load()), encoding="utf-8")
    return path
