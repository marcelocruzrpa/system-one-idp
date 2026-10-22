"""Sample document specs: what each document shows, its ground truth, and the mock ERP.

Everything printed on a document and everything in its ground truth is computed from the same
values here, so the two cannot drift apart. Defects are seeded on purpose and listed in
`expected_flags`.
"""

import zlib
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Any


def r2(x: Decimal | float | str) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def pct(rate: str) -> str:
    return f"{(Decimal(rate) * 100).normalize():f}"


@dataclass(frozen=True)
class Style:
    decimal: str = "."
    thousands: str = ","
    date_format: str = "long"  # long: "September 4, 2026" | eu: "04.09.2026" | us: "09/04/2026" | iso | short: "Sep 4, 2026"

    def money(self, x: Decimal | float) -> str:
        s = f"{r2(x):,.2f}"
        return s.replace(",", "\0").replace(".", self.decimal).replace("\0", self.thousands)

    def qty(self, x: float) -> str:
        return str(int(x)) if float(x).is_integer() else self.money(x)

    def date(self, d: date) -> str:
        match self.date_format:
            case "long":
                return f"{d:%B} {d.day}, {d.year}"
            case "short":
                return f"{d:%b} {d.day}, {d.year}"
            case "eu":
                return f"{d:%d.%m.%Y}"
            case "us":
                return f"{d:%m/%d/%Y}"
        return d.isoformat()


US = Style()
EU = Style(decimal=",", thousands=".", date_format="eu")


@dataclass(frozen=True)
class Party:
    name: str
    address: tuple[str, ...]
    tax_label: str = "Tax ID"
    tax_id: str | None = None
    contact: tuple[str, ...] = ()

    def lines(self, with_tax: bool = True) -> list[str]:
        out = [self.name, *self.address]
        if with_tax and self.tax_id:
            out.append(f"{self.tax_label}: {self.tax_id}")
        return out


@dataclass(frozen=True)
class Line:
    description: str
    quantity: float
    unit_price: float
    sku: str = ""

    @property
    def amount(self) -> Decimal:
        return r2(Decimal(str(self.quantity)) * Decimal(str(self.unit_price)))


ACME = Party("Acme Robotics Inc.", ("1200 Harbor Point Blvd, Suite 400", "Oakland, CA 94607", "USA"),
             tax_id="94-3187265", contact=("ap@acmerobotics.example", "+1 (510) 555-0142"))
APEX = Party("Apex Robotix B.V.", ("Keizersgracht 221", "1016 DV Amsterdam", "Netherlands"),
             tax_label="VAT ID", tax_id="NL859321476B01")
NORTHWIND = Party("Northwind Supply Co.", ("455 Industrial Way", "Fremont, CA 94538", "USA"),
                  tax_id="77-0421953", contact=("billing@northwind-supply.example", "+1 (510) 555-0199"))
HELIOS = Party("Helios Components GmbH", ("Industriestraße 18", "70565 Stuttgart", "Germany"),
               tax_label="VAT ID", tax_id="DE287456913", contact=("rechnung@helios-components.example", "+49 711 555 2040"))
BLUEFIN = Party("Bluefin Office Supplies LLC", ("88 Market Street", "San Jose, CA 95113", "USA"),
                tax_id="46-5523190", contact=("ar@bluefin-office.example", "+1 (408) 555-0117"))
GLOBEX = Party("Globex Manufacturing Corp.", ("9000 Commerce Pkwy", "Plano, TX 75024", "USA"))
INITECH = Party("Initech Automation Inc.", ("4120 Freidrich Lane", "Austin, TX 78744", "USA"))

# Truth for the PO `ship_to` field: the street line of the Ship to block (a span-based text field returns one
# line of a printed address block, never a comma-joined composite).
ACME_SHIP_TO = "1200 Harbor Point Blvd, Suite 400"

# Acme's suppliers in the ERP vendor master.
VENDOR_IDS = {NORTHWIND: "V-1001", HELIOS: "V-1002", BLUEFIN: "V-1003"}


def currency_of(vendor: Party) -> str:
    return "EUR" if vendor is HELIOS else "USD"


@dataclass
class Spec:
    id: str
    doc_type: str
    template: str
    paper: str  # Letter | A4 | receipt
    context: dict[str, Any]
    fields: dict[str, Any]
    expected_flags: list[str] = field(default_factory=list)
    note: str = ""
    links: dict[str, str] = field(default_factory=dict)

    @property
    def expected_route(self) -> str:
        if {"not_addressed_to_us", "duplicate_invoice"} & set(self.expected_flags):
            return "reject"
        return "review" if self.expected_flags else "auto_approve"


# --------------------------------------------------------------------------------------------
# Commercial documents (invoice, pro forma, credit note, quote, purchase order)
# --------------------------------------------------------------------------------------------

LAYOUTS = {
    NORTHWIND.name: ("classic", US, "#1f3b63"),
    HELIOS.name: ("eu", EU, "#9a3412"),
    BLUEFIN.name: ("minimal", US, "#0f766e"),
    ACME.name: ("classic", Style(date_format="iso"), "#4338ca"),
}


@dataclass
class Totals:
    subtotal: Decimal
    tax: Decimal
    total: Decimal


def compute_totals(lines: list[Line], tax_rate: str) -> Totals:
    subtotal = r2(sum((l.amount for l in lines), Decimal(0)))
    tax = r2(subtotal * Decimal(tax_rate))
    return Totals(subtotal, tax, subtotal + tax)


def commercial(*, issuer: Party, title: str, meta: list[tuple[str, str]], blocks: list[tuple[str, list[str]]],
               lines: list[Line], tax_label: str, totals: Totals, total_label: str = "Total due",
               currency: str, notes: list[str] = (), stamp: str | None = None, footer: list[str] | None = None,
               show_sku: bool = False, layout_of: Party | None = None) -> dict[str, Any]:
    layout, style, accent = LAYOUTS[(layout_of or issuer).name]
    columns = (["Item"] if show_sku else []) + ["Description", "Qty", "Unit price", f"Amount ({currency})"]
    rows = [([l.sku] if show_sku else []) + [l.description, style.qty(l.quantity), style.money(l.unit_price), style.money(l.amount)]
            for l in lines]
    contact = list(issuer.contact)
    return {
        "layout": layout, "accent": accent, "title": title,
        "issuer": {"name": issuer.name, "address": list(issuer.address),
                   "tax_line": f"{issuer.tax_label}: {issuer.tax_id}" if issuer.tax_id else "", "contact": contact},
        "meta": meta, "blocks": [{"heading": h, "lines": ls} for h, ls in blocks],
        "columns": columns, "rows": rows,
        "totals": [
            {"label": "Subtotal", "value": style.money(totals.subtotal)},
            {"label": tax_label, "value": style.money(totals.tax)},
            {"label": f"{total_label} ({currency})", "value": style.money(totals.total), "strong": True},
        ],
        "notes": list(notes), "stamp": stamp, "footer": footer or [],
    }


def line_truth(lines: list[Line]) -> dict[str, list]:
    return {
        "line_descriptions": [l.description for l in lines],
        "line_quantities": [float(l.quantity) for l in lines],
        "line_unit_prices": [float(l.unit_price) for l in lines],
        "line_amounts": [float(l.amount) for l in lines],
    }


def money_truth(t: Totals) -> dict[str, float]:
    return {"subtotal": float(t.subtotal), "tax_amount": float(t.tax), "total": float(t.total)}


def helios_footer() -> list[str]:
    return ["Helios Components GmbH · Registered office: Stuttgart · Commercial register HRB 774512 · Managing director: Katrin Vogel",
            "Bank: Commerzbank Stuttgart · IBAN DE89 3704 0044 0532 0130 00 · BIC COBADEFFXXX"]


def invoice(*, id: str, issuer: Party, bill_to: Party, number: str, issued: date, due: date, terms: str,
            po: str | None, lines: list[Line], tax_rate: str, tax_label: str, currency: str,
            printed: Totals | None = None, flags: list[str] = (), note: str = "", notes: list[str] = (),
            stamp: str | None = None, footer: list[str] | None = None, show_sku: bool = False) -> Spec:
    _, style, _ = LAYOUTS[issuer.name]
    totals = printed or compute_totals(lines, tax_rate)
    meta = [("Invoice No.", number), ("Invoice date", style.date(issued)), ("Due date", style.date(due)),
            ("Payment terms", terms)]
    if po:
        meta.append(("Your PO No.", po))
    ctx = commercial(issuer=issuer, title="Invoice" if style is EU else "INVOICE", meta=meta,
                     blocks=[("Bill to", bill_to.lines())], lines=lines, tax_label=tax_label, totals=totals,
                     currency=currency, notes=notes, stamp=stamp, footer=footer, show_sku=show_sku)
    truth = {"invoice_number": number, "issue_date": issued.isoformat(), "due_date": due.isoformat(),
             "vendor_name": issuer.name, "vendor_tax_id": issuer.tax_id, "bill_to_name": bill_to.name,
             "po_number": po, "currency": currency, "payment_terms": terms,
             **money_truth(totals), **line_truth(lines)}
    return Spec(id, "invoice", "commercial.html.j2", "A4" if style is EU else "Letter", ctx, truth, list(flags), note,
                {"po": po} if po else {})


# Purchase orders (ERP master data; three of them are also rendered as documents).
PO_LINES: dict[str, tuple[Party, date, date, str, str, list[Line]]] = {
    "PO-4501": (NORTHWIND, date(2026, 8, 24), date(2026, 9, 8), "Net 30", "0.085", [
        Line("Servo motor MG996R", 40, 18.50, "NW-SRV-996"),
        Line("Stepper driver module TMC2209", 60, 6.25, "NW-DRV-2209"),
        Line("Li-ion battery pack 24 V / 10 Ah", 12, 189.00, "NW-BAT-2410"),
    ]),
    "PO-4502": (HELIOS, date(2026, 8, 18), date(2026, 9, 15), "30 days net", "0", [
        Line("Planetary gearbox PLE60, i=10", 6, 412.00, "HC-PLE60-10"),
        Line("Absolute encoder 17-bit AE17", 10, 129.50, "HC-AE17"),
        Line("Ball screw SFU1204 L=500mm", 8, 74.90, "HC-SFU1204-500"),
    ]),
    "PO-4503": (BLUEFIN, date(2026, 9, 1), date(2026, 9, 12), "Net 15", "0.09375", [
        Line("Ergonomic mesh office chair", 8, 245.00, "BF-CH-220"),
        Line("Dual monitor arm", 8, 89.00, "BF-MA-2"),
        Line('Standing desk converter, 32"', 4, 179.00, "BF-SD-32"),
    ]),
    "PO-4504": (NORTHWIND, date(2026, 9, 3), date(2026, 9, 17), "Net 30", "0.085", [
        Line("Aluminum extrusion 2020, 1m", 100, 4.80),
        Line("M5x12 socket head screws, box/100", 20, 7.40),
        Line("Linear bearing LM8UU", 50, 1.95),
    ]),
    "PO-4506": (NORTHWIND, date(2026, 7, 20), date(2026, 8, 3), "Net 30", "0.085", [
        Line("Brushless DC motor 57BL55", 10, 64.00),
        Line("GT2 timing belt 6mm, 10m roll", 15, 12.90),
    ]),
}

OFFICE_ITEMS = [
    ("Copy paper Letter 20 lb, case of 10 reams", 6, 48.99), ("Ballpoint pens blue, box of 60", 4, 14.25),
    ("Sticky notes 3x3 in, 24 pads", 5, 18.40), ("Whiteboard markers, assorted (12)", 6, 11.90),
    ("Stapler heavy duty", 3, 32.50), ("Staples standard, 5000 ct", 6, 4.75),
    ("Binder clips medium, 144 ct", 4, 9.80), ("Hanging file folders, letter (25)", 5, 21.60),
    ("Desk organizer mesh", 8, 16.99), ("Notebooks A5 ruled (pack of 5)", 10, 12.49),
    ("Toner cartridge HP 58A", 4, 112.00), ("Toner cartridge HP 58X high yield", 2, 189.00),
    ("Ink cartridge Epson 252 black", 6, 24.99), ("Label maker tape 12 mm", 8, 13.75),
    ("USB-C hub 7-in-1", 6, 39.90), ("Wireless mouse", 12, 21.99),
    ("Keyboard wired full size", 6, 29.50), ("Laptop stand aluminum", 8, 34.95),
    ("LED desk lamp", 8, 42.00), ("Cable management tray", 10, 27.80),
    ("Surge protector 8 outlet", 10, 24.40), ("HDMI cable 2 m", 15, 7.95),
    ("Ethernet cable Cat6 3 m", 20, 5.60), ("Shredder cross-cut 12 sheet", 1, 219.00),
    ("Paper towels, case of 12", 4, 31.20), ("Hand sanitizer 1 L", 6, 9.95),
    ("Coffee pods medium roast (96)", 5, 52.90), ("Disposable cups 12 oz (500)", 2, 44.80),
    ("Trash bags 33 gal (100)", 3, 26.70), ("First aid kit 100 pc", 2, 38.60),
    ("Fire-resistant document box", 1, 89.99), ("Visitor badge holders (50)", 2, 17.30),
]
PO_LINES["PO-4507"] = (BLUEFIN, date(2026, 9, 20), date(2026, 10, 2), "Net 30", "0.09375",
                       [Line(d, q, p, f"BF-{i:03d}") for i, (d, q, p) in enumerate(OFFICE_ITEMS, start=101)])

# What the warehouse actually received (goods receipts). PO-4502 line 2 is short by 2 units.
RECEIVED = {po: [l.quantity for l in spec[5]] for po, spec in PO_LINES.items()}
RECEIVED["PO-4502"] = [6, 8, 8]

POSTED_INVOICES = [
    {"invoice_number": "INV-NW-10198", "vendor_id": VENDOR_IDS[NORTHWIND], "vendor": NORTHWIND.name,
     "amount": "USD 904.35", "posted_on": "2026-08-07"},
    {"invoice_number": "RE-2026-0398", "vendor_id": VENDOR_IDS[HELIOS], "vendor": HELIOS.name,
     "amount": "EUR 3,120.00", "posted_on": "2026-07-10"},
    {"invoice_number": "INV-BF-0087", "vendor_id": VENDOR_IDS[BLUEFIN], "vendor": BLUEFIN.name,
     "amount": "USD 1,486.22", "posted_on": "2026-08-21"},
]


def erp_data() -> dict[str, Any]:
    return {
        "vendors": [{"vendor_id": vid, "name": vendor.name, "aliases": [], "tax_id": vendor.tax_id,
                     "currency": currency_of(vendor), "address": list(vendor.address), "email": vendor.contact[0]}
                    for vendor, vid in VENDOR_IDS.items()],
        # `invoiced` = quantity already billed on earlier invoices (none yet in the samples).
        "purchase_orders": [
            {"po_number": po, "vendor_id": VENDOR_IDS[vendor], "vendor": vendor.name, "currency": currency_of(vendor),
             "po_date": d.isoformat(), "delivery_date": deliver_by.isoformat(), "payment_terms": terms,
             "lines": [{"line": i, "description": l.description, "quantity": l.quantity, "unit_price": l.unit_price,
                        "invoiced": 0}
                       for i, l in enumerate(lines, start=1)]}
            for po, (vendor, d, deliver_by, terms, _, lines) in PO_LINES.items()
        ],
        # One receipt per PO, posted on the PO's deliver-by date (matches the delivery notes' ship dates).
        "goods_receipts": [
            {"gr_number": f"GR-{5001 + n}", "po_number": po, "posted_on": PO_LINES[po][2].isoformat(),
             "lines": [{"line": i, "received": q} for i, q in enumerate(qtys, start=1)]}
            for n, (po, qtys) in enumerate(RECEIVED.items())
        ],
        "posted_invoices": POSTED_INVOICES,
    }


def purchase_order(id: str, po: str) -> Spec:
    vendor, issued, delivery, terms, rate, lines = PO_LINES[po]
    _, style, _ = LAYOUTS[ACME.name]
    currency = currency_of(vendor)
    totals = compute_totals(lines, rate)
    tax_label = "VAT 0% (export)" if rate == "0" else f"Est. sales tax {pct(rate)}%"
    ctx = commercial(issuer=ACME, title="PURCHASE ORDER",
                     meta=[("PO Number", po), ("PO date", style.date(issued)), ("Deliver by", style.date(delivery)),
                           ("Payment terms", terms), ("Buyer", "J. Okafor, Procurement")],
                     blocks=[("Vendor", vendor.lines()), ("Ship to", [ACME.name, "Receiving Dock B", *ACME.address])],
                     lines=lines, tax_label=tax_label, totals=totals, total_label="Order total", currency=currency,
                     notes=["Please reference the PO number on all invoices and delivery documents.",
                            "Partial shipments allowed. Invoices without a valid PO number will be returned."],
                     show_sku=True)
    truth = {"po_number": po, "po_date": issued.isoformat(), "buyer_name": ACME.name, "vendor_name": vendor.name,
             "delivery_date": delivery.isoformat(), "ship_to": ACME_SHIP_TO, "payment_terms": terms,
             "currency": currency, **money_truth(totals), **line_truth(lines)}
    return Spec(id, "purchase_order", "commercial.html.j2", "Letter", ctx, truth,
                note=f"PO issued by Acme to {vendor.name}", links={"po": po})


def renamed(lines: list[Line], names: list[str], **overrides: dict[int, float]) -> list[Line]:
    """Vendor-side wording of PO lines (so line matching needs semantics), with optional price overrides."""
    prices = overrides.get("prices", {})
    return [Line(n, l.quantity, prices.get(i, l.unit_price), l.sku) for i, (l, n) in enumerate(zip(lines, names))]


def invoices() -> list[Spec]:
    po1 = PO_LINES["PO-4501"][5]
    po2 = PO_LINES["PO-4502"][5]
    po3 = PO_LINES["PO-4503"][5]
    po4 = PO_LINES["PO-4504"][5]
    po6 = PO_LINES["PO-4506"][5]
    po7 = PO_LINES["PO-4507"][5]
    us_tax = "Sales tax 8.5%"
    sj_tax = "Sales tax 9.375%"
    eu_tax = "VAT 0% (tax-free export delivery)"

    l1 = renamed(po1, ["MG-996R High-Torque Servo Motor", "TMC2209 Stepper Motor Driver Module",
                       "Lithium Battery Pack 24V 10Ah"])
    l2 = renamed(po2, ["Planetengetriebe / Planetary gearbox PLE60 ratio 10:1", "Absolute encoder AE-17, 17 bit",
                       "Precision ball screw SFU1204, length 500 mm"])
    l3 = renamed(po3, ["Mesh-back ergonomic task chair", "Monitor arm, dual mount",
                       "Sit-stand desk converter 32 in"], prices={0: 265.00})
    l4 = renamed(po4, ["2020 aluminum extrusion profile 1000 mm", "Socket head cap screw M5 x 12 (100/box)",
                       "LM8UU linear ball bearing"])
    l6 = renamed(po6, ["57BL55 brushless DC motor 24V", "GT2 timing belt, 6 mm wide, 10 m"])
    l7 = [Line(l.description, l.quantity, l.unit_price, l.sku) for l in po7]
    apex_lines = [Line("Harmonic drive unit HD-14-50", 2, 1180.00), Line("Torque sensor TS-200 Nm", 1, 640.00)]

    t4 = compute_totals(l4, "0.085")
    t4_printed = Totals(t4.subtotal, t4.tax, t4.total + 100)  # arithmetic error on the printed total

    return [
        invoice(id="invoice-01", issuer=NORTHWIND, bill_to=ACME, number="INV-NW-10231", issued=date(2026, 9, 10),
                due=date(2026, 10, 10), terms="Net 30", po="PO-4501", lines=l1, tax_rate="0.085", tax_label=us_tax,
                currency="USD", note="Clean invoice; matches PO-4501 and its full delivery",
                notes=["Thank you for your business! Please include the invoice number with your payment."]),
        invoice(id="invoice-02", issuer=HELIOS, bill_to=ACME, number="RE-2026-0457", issued=date(2026, 9, 16),
                due=date(2026, 10, 16), terms="30 days net", po="PO-4502", lines=l2, tax_rate="0", tax_label=eu_tax,
                currency="EUR", flags=["qty_exceeds_received"], footer=helios_footer(), show_sku=True,
                note="Bills 10 encoders but only 8 were received (short shipment on DN-HE-2210)"),
        invoice(id="invoice-03", issuer=BLUEFIN, bill_to=ACME, number="INV-BF-0099", issued=date(2026, 9, 15),
                due=date(2026, 9, 30), terms="Net 15", po="PO-4503", lines=l3, tax_rate="0.09375", tax_label=sj_tax,
                currency="USD", flags=["price_mismatch"], note="Chair billed at 265.00 vs 245.00 on the PO"),
        invoice(id="invoice-04", issuer=NORTHWIND, bill_to=ACME, number="INV-NW-10240", issued=date(2026, 9, 22),
                due=date(2026, 10, 22), terms="Net 30", po="PO-4504", lines=l4, tax_rate="0.085", tax_label=us_tax,
                currency="USD", printed=t4_printed, flags=["total_mismatch"],
                note="Printed total is 100.00 higher than subtotal + tax"),
        invoice(id="invoice-05", issuer=HELIOS, bill_to=APEX, number="RE-2026-0461", issued=date(2026, 9, 25),
                due=date(2026, 10, 25), terms="30 days net", po=None, lines=apex_lines, tax_rate="0",
                tax_label="VAT 0% (reverse charge, Art. 196 VAT Directive)", currency="EUR",
                flags=["not_addressed_to_us", "missing_po"], footer=helios_footer(),
                note="Billed to a different company (Apex Robotix B.V.) and has no PO"),
        invoice(id="invoice-06", issuer=NORTHWIND, bill_to=ACME, number="INV-NW-10198", issued=date(2026, 8, 5),
                due=date(2026, 9, 4), terms="Net 30", po="PO-4506", lines=l6, tax_rate="0.085", tax_label=us_tax,
                currency="USD", flags=["duplicate_invoice", "already_paid"], stamp="PAID 09/03/2026",
                note="Re-sent copy of an invoice already posted and paid"),
        invoice(id="invoice-07", issuer=BLUEFIN, bill_to=ACME, number="INV-BF-0112", issued=date(2026, 10, 5),
                due=date(2026, 9, 4), terms="Net 30", po="PO-4507", lines=l7, tax_rate="0.09375", tax_label=sj_tax,
                currency="USD", flags=["due_before_issue", "bank_detail_change"], show_sku=True,
                notes=["IMPORTANT: Effective immediately our banking details have changed. Please update your records "
                       "and remit all future payments to our NEW account: Coastal Trust Bank, routing 322271724, "
                       "account 7700193345. Payments sent to the old account will be returned."],
                note="Two-page invoice; due date before invoice date; requests a bank detail change"),
    ]


def proformas() -> list[Spec]:
    specs = []
    for id, issuer, number, issued, valid, lines, rate, tax_label, currency, notes in [
        ("proforma-01", HELIOS, "PF-HE-2026-019", date(2026, 10, 6), date(2026, 10, 20),
         [Line("Torque sensor TS-200 Nm", 4, 640.00), Line("Calibration certificate (DAkkS)", 4, 85.00)],
         "0", "VAT 0% (export)", "EUR",
         ["Advance payment required before shipment.", "This pro forma invoice is not a tax invoice; a commercial "
          "invoice will be issued upon shipment."]),
        ("proforma-02", NORTHWIND, "PF-NW-0007", date(2026, 9, 29), date(2026, 10, 29),
         [Line("Brushless DC motor 57BL55 (evaluation samples)", 3, 64.00), Line("Shipping & handling", 1, 45.00)],
         "0.085", "Sales tax 8.5%", "USD",
         ["PRO FORMA - for prepayment and customs purposes only. This is not a demand for payment."]),
    ]:
        _, style, _ = LAYOUTS[issuer.name]
        totals = compute_totals(lines, rate)
        ctx = commercial(issuer=issuer, title="PRO FORMA INVOICE",
                         meta=[("Pro forma No.", number), ("Date", style.date(issued)), ("Valid until", style.date(valid))],
                         blocks=[("Prepared for", ACME.lines())], lines=lines, tax_label=tax_label, totals=totals,
                         total_label="Total", currency=currency, notes=notes,
                         footer=helios_footer() if issuer is HELIOS else None)
        truth = {"proforma_number": number, "issue_date": issued.isoformat(), "valid_until": valid.isoformat(),
                 "vendor_name": issuer.name, "bill_to_name": ACME.name, "currency": currency,
                 **money_truth(totals), **line_truth(lines)}
        specs.append(Spec(id, "proforma_invoice", "commercial.html.j2", "A4" if style is EU else "Letter", ctx, truth,
                          note="Pro forma (looks like an invoice, but is not a demand for payment)"))
    return specs


def credit_notes() -> list[Spec]:
    specs = []
    for id, issuer, number, issued, original, reason, lines, rate, tax_label, currency, printed_sub, flags, note in [
        ("credit_note-01", NORTHWIND, "CN-NW-0042", date(2026, 8, 28), "INV-NW-10198",
         "Return of 2 timing belts (wrong width)", [Line("GT2 timing belt 6mm, 10m roll - returned", 2, 12.90)],
         "0.085", "Sales tax 8.5%", "USD", None, [], "Clean credit note for a posted invoice"),
        ("credit_note-02", HELIOS, "GS-2026-0031", date(2026, 10, 2), "RE-2026-0398",
         "Agreed price adjustment and freight refund",
         [Line("Price adjustment absolute encoder AE17 (agreed discount)", 20, 6.50),
          Line("Refund of freight overcharge", 1, 45.00)],
         "0", "VAT 0%", "EUR", Decimal("185.00"), ["line_sum_mismatch"],
         "Printed subtotal 185.00 but lines sum to 175.00"),
    ]:
        _, style, _ = LAYOUTS[issuer.name]
        totals = compute_totals(lines, rate)
        if printed_sub is not None:
            totals = Totals(printed_sub, r2(printed_sub * Decimal(rate)), printed_sub + r2(printed_sub * Decimal(rate)))
        ctx = commercial(issuer=issuer, title="CREDIT NOTE",
                         meta=[("Credit note No.", number), ("Date", style.date(issued)),
                               ("Original invoice", original), ("Reason", reason)],
                         blocks=[("Customer", ACME.lines())], lines=lines, tax_label=tax_label, totals=totals,
                         total_label="Total credit", currency=currency,
                         notes=["This amount will be credited against your account. No payment is required."],
                         footer=helios_footer() if issuer is HELIOS else None)
        truth = {"credit_note_number": number, "issue_date": issued.isoformat(), "vendor_name": issuer.name,
                 "customer_name": ACME.name, "original_invoice_number": original, "reason": reason,
                 "currency": currency, **money_truth(totals), **line_truth(lines)}
        specs.append(Spec(id, "credit_note", "commercial.html.j2", "A4" if style is EU else "Letter", ctx, truth,
                          flags, note, {"invoice": original}))
    return specs


def quotes() -> list[Spec]:
    specs = []
    for id, issuer, title, number, issued, valid, lines, rate, tax_label in [
        ("quote-01", NORTHWIND, "QUOTATION", "Q-NW-2026-118", date(2026, 10, 12), date(2026, 11, 11),
         [Line("Brushless DC motor 57BL55", 25, 61.50), Line("Motor controller BLD-300B", 25, 38.00),
          Line("Encoder cable 2 m", 25, 4.20)], "0.085", "Sales tax 8.5%"),
        ("quote-02", BLUEFIN, "ESTIMATE", "EST-BF-7745", date(2026, 10, 1), date(2026, 10, 31),
         [Line("Conference table 10 ft, walnut", 1, 1450.00), Line("Conference chair, upholstered", 10, 189.00),
          Line("Delivery and installation", 1, 250.00)], "0.09375", "Sales tax 9.375%"),
    ]:
        _, style, _ = LAYOUTS[issuer.name]
        totals = compute_totals(lines, rate)
        ctx = commercial(issuer=issuer, title=title,
                         meta=[(f"{title.title()} No.", number), ("Date", style.date(issued)),
                               ("Valid until", style.date(valid))],
                         blocks=[("Prepared for", ACME.lines())], lines=lines, tax_label=tax_label, totals=totals,
                         total_label="Total", currency="USD",
                         notes=["Prices valid until the date shown. To accept, please send a purchase order "
                                "referencing this quote number."])
        truth = {"quote_number": number, "quote_date": issued.isoformat(), "valid_until": valid.isoformat(),
                 "vendor_name": issuer.name, "customer_name": ACME.name, "currency": "USD",
                 **money_truth(totals), **line_truth(lines)}
        specs.append(Spec(id, "quote", "commercial.html.j2", "Letter", ctx, truth, note="Quote (nothing owed yet)"))
    return specs


# --------------------------------------------------------------------------------------------
# Delivery notes, receipts, bank statements, remittance advices
# --------------------------------------------------------------------------------------------

def delivery_notes() -> list[Spec]:
    specs = []
    for id, po, number, shipped_on, carrier, names, flags, note in [
        ("delivery_note-01", "PO-4501", "DN-7781", date(2026, 9, 8), "UPS Ground · 1Z 84F 2E0 03 4471 2290",
         ["MG-996R servo motor", "TMC2209 driver module", "Battery pack 24V 10Ah"], [], "Full delivery for PO-4501"),
        ("delivery_note-02", "PO-4502", "DN-HE-2210", date(2026, 9, 14), "DHL Freight · 4471 2290 18",
         ["Planetary gearbox PLE60 i=10", "Absolute encoder AE17", "Ball screw SFU1204 500 mm"],
         ["short_shipment"], "Encoder short-shipped: 8 of 10, 2 on backorder"),
        ("delivery_note-03", "PO-4503", "DN-BF-553", date(2026, 9, 11), "Bluefin delivery van",
         ["Ergonomic mesh chair", "Dual monitor arm", "Desk converter 32 in"], [], "Full delivery for PO-4503"),
    ]:
        vendor, *_, lines = PO_LINES[po]
        _, style, accent = LAYOUTS[vendor.name]
        received = RECEIVED[po]
        rows = [[l.sku, n, style.qty(l.quantity), style.qty(r), style.qty(l.quantity - r)]
                for l, n, r in zip(lines, names, received)]
        ctx = {"accent": accent, "supplier": {"name": vendor.name, "address": list(vendor.address), "contact": list(vendor.contact)},
               "meta": [("Delivery note No.", number), ("Ship date", style.date(shipped_on)), ("Customer PO", po),
                        ("Carrier", carrier)],
               "ship_to": [ACME.name, "Receiving Dock B", *ACME.address],
               "rows": rows, "packages": f"{len(lines) + 1} packages"}
        truth = {"delivery_note_number": number, "delivery_date": shipped_on.isoformat(), "supplier_name": vendor.name,
                 "customer_name": ACME.name, "po_number": po, "line_descriptions": names,
                 "line_ordered_quantities": [float(l.quantity) for l in lines],
                 "line_shipped_quantities": [float(r) for r in received]}
        specs.append(Spec(id, "delivery_note", "delivery_note.html.j2", "A4" if style is EU else "Letter",
                          ctx, truth, flags, note, {"po": po}))
    return specs


def receipts() -> list[Spec]:
    specs = []
    for id, merchant, address, when, items, rate, method, last4, flags, note in [
        ("receipt-01", "METRO HARDWARE & SUPPLY", ["2231 Telegraph Ave", "Oakland, CA 94612", "(510) 555-0170"],
         (date(2026, 10, 8), "14:32"),
         [("Zip ties 8in 100pk", 7.99), ("Electrical tape 3pk", 5.49), ("Drill bit set 21pc", 24.99)],
         "0.1025", "VISA", "4821", [], "Hardware store receipt (business supplies)"),
        ("receipt-02", "HARBOR GRILL & TAPROOM", ["55 Embarcadero W", "Oakland, CA 94607", "(510) 555-0133"],
         (date(2026, 10, 14), "19:47"),
         [("Grilled salmon", 28.00), ("Caesar salad", 14.50), ("Hazy IPA draft", 8.00), ("Hazy IPA draft", 8.00),
          ("Sparkling water", 4.50)],
         "0.1025", "MASTERCARD", "7730", ["non_reimbursable_items"], "Restaurant receipt including alcohol"),
    ]:
        subtotal = r2(sum(Decimal(str(p)) for _, p in items))
        tax = r2(subtotal * Decimal(rate))
        total = subtotal + tax
        ctx = {"merchant": merchant, "address": address, "date": f"{when[0]:%m/%d/%Y}", "time": when[1],
               "items": [(d, US.money(p)) for d, p in items], "subtotal": US.money(subtotal),
               "tax_label": f"TAX {pct(rate)}%", "tax": US.money(tax), "total": US.money(total),
               "method": method, "last4": last4, "auth": f"{zlib.crc32(id.encode()) % 900000 + 100000}",
               "txn": f"{when[0]:%m%d}-{last4}-0{len(items)}"}
        truth = {"merchant_name": merchant, "transaction_date": when[0].isoformat(), "transaction_time": when[1],
                 "currency": "USD", "subtotal": float(subtotal), "tax_amount": float(tax), "total": float(total),
                 "payment_method": method, "card_last4": last4,
                 "item_descriptions": [d for d, _ in items], "item_amounts": [p for _, p in items]}
        specs.append(Spec(id, "receipt", "receipt.html.j2", "receipt", ctx, truth, flags, note))
    return specs


def bank_statements() -> list[Spec]:
    specs = []
    for id, bank, product, account, opening, txns, variant in [
        ("bank_statement-01", "First Harbor Bank", "Business Checking", "****9083", "184220.15", [
            (2, "ACH CREDIT GLOBEX MANUFACTURING CORP", "12450.00"),
            (4, "ACH DEBIT NORTHWIND SUPPLY CO INV-NW-10198", "-904.35"),
            (8, "INTL WIRE OUT HELIOS COMPONENTS GMBH", "-3412.80"),
            (10, "CARD PURCHASE AWS EMEA CLOUD SERVICES", "-2184.37"),
            (15, "TRANSFER TO PAYROLL ****2210", "-96500.00"),
            (18, "ACH CREDIT INITECH AUTOMATION INC", "18725.00"),
            (22, "ACH DEBIT HARBOR POINT PROPERTIES RENT", "-14800.00"),
            (25, "ACH CREDIT UMBRELLA LOGISTICS LLC", "7960.00"),
            (29, "MONTHLY SERVICE FEE", "-45.00"),
            (30, "INTEREST PAID", "38.12"),
        ], "summary_top"),
        ("bank_statement-02", "Pacific Commerce Bank", "Payroll Account", "****2210", "12310.40", [
            (15, "TRANSFER FROM ****9083", "96500.00"),
            (15, "PAYROLL BATCH #0915 DIRECT DEPOSITS", "-94882.16"),
            (30, "PAYROLL SERVICE FEE", "-149.00"),
            (30, "INTEREST", "3.27"),
        ], "summary_bottom"),
    ]:
        opening_d = Decimal(opening)
        credits = sum((Decimal(a) for *_, a in txns if Decimal(a) > 0), Decimal(0))
        debits = -sum((Decimal(a) for *_, a in txns if Decimal(a) < 0), Decimal(0))
        rows, balance = [], opening_d
        for day, desc, amount in txns:
            a = Decimal(amount)
            balance += a
            rows.append([f"09/{day:02d}", desc, US.money(-a) if a < 0 else "", US.money(a) if a > 0 else "", US.money(balance)])
        closing = opening_d + credits - debits
        start, end = date(2026, 9, 1), date(2026, 9, 30)
        ctx = {"bank": bank, "product": product, "account": account, "variant": variant,
               "holder": ACME.lines(with_tax=False), "period": f"{US.date(start)} through {US.date(end)}",
               "opening": US.money(opening_d), "credits": US.money(credits), "debits": US.money(debits),
               "closing": US.money(closing), "rows": rows, "n_credits": sum(1 for *_, a in txns if Decimal(a) > 0),
               "n_debits": sum(1 for *_, a in txns if Decimal(a) < 0)}
        truth = {"bank_name": bank, "account_holder": ACME.name, "account_number": account, "currency": "USD",
                 "period_start": start.isoformat(), "period_end": end.isoformat(),
                 "opening_balance": float(opening_d), "total_credits": float(credits), "total_debits": float(debits),
                 "closing_balance": float(closing)}
        specs.append(Spec(id, "bank_statement", "bank_statement.html.j2", "Letter", ctx, truth, note=f"{product} statement"))
    return specs


def remittances() -> list[Spec]:
    specs = []
    for id, payer, paid_on, method, ref, items, flags, note in [
        ("remittance_advice-01", GLOBEX, date(2026, 10, 9), "ACH", "ACH-889120",
         [("ACM-INV-3301", "09/02/2026", "12450.00", "0", ""), ("ACM-INV-3307", "09/09/2026", "8200.00", "0", "")],
         [], "Full payment of two invoices"),
        ("remittance_advice-02", INITECH, date(2026, 10, 13), "Wire transfer", "WT-2026101300457",
         [("ACM-INV-3290", "08/28/2026", "5600.00", "0", ""),
          ("ACM-INV-3312", "09/15/2026", "9800.00", "490.00", "3 units returned damaged")],
         ["short_payment"], "Deduction taken on one invoice"),
    ]:
        rows, paid_amounts = [], []
        for number, inv_date, amount, deduction, reason in items:
            paid = Decimal(amount) - Decimal(deduction)
            paid_amounts.append(float(paid))
            rows.append([number, inv_date, US.money(Decimal(amount)),
                         US.money(Decimal(deduction)) if Decimal(deduction) else "-", US.money(paid), reason])
        total = sum((Decimal(str(p)) for p in paid_amounts), Decimal(0))
        ctx = {"payer": payer.lines(with_tax=False), "payee": ACME.lines(with_tax=False),
               "meta": [("Payment date", US.date(paid_on)), ("Payment method", method), ("Reference", ref),
                        ("Currency", "USD")],
               "rows": rows, "total": US.money(total)}
        truth = {"payer_name": payer.name, "payee_name": ACME.name, "payment_date": paid_on.isoformat(),
                 "payment_reference": ref, "currency": "USD", "payment_total": float(total),
                 "invoice_numbers": [i[0] for i in items], "paid_amounts": paid_amounts}
        specs.append(Spec(id, "remittance_advice", "remittance.html.j2", "Letter", ctx, truth, flags, note))
    return specs


def all_specs() -> list[Spec]:
    return [
        *invoices(),
        *(purchase_order(f"purchase_order-0{i}", po) for i, po in enumerate(["PO-4501", "PO-4502", "PO-4503"], start=1)),
        *delivery_notes(),
        *receipts(),
        *credit_notes(),
        *quotes(),
        *proformas(),
        *bank_statements(),
        *remittances(),
    ]
