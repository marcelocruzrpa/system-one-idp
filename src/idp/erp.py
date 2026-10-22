"""Mock ERP: vendors, purchase orders, goods receipts and already-posted invoices (from samples/erp.json).

In a real AP flow, POs and goods receipts live in the ERP, not in scanned documents, so the
three-way match and duplicate checks read from here.
"""

import json
import re
from pathlib import Path
from typing import Any

from idp.config import SAMPLES_DIR

# Legal-form words dropped from the end of company names before comparing them.
LEGAL_SUFFIXES = {"inc", "incorporated", "llc", "ltd", "limited", "co", "company", "corp", "corporation",
                  "gmbh", "ag", "bv", "nv", "sa", "sas", "sarl", "srl", "spa", "plc", "kg", "lp", "llp"}


def norm_ref(value: Any) -> str:
    """Normalize a document reference for lookup: `po 4501`, `PO-4501` and `PO4501` all match."""
    return re.sub(r"[^0-9A-Z]", "", str(value or "").upper())


def norm_company(value: Any) -> str:
    """Normalize a company name for lookup: `Northwind Supply Co.` and `NORTHWIND SUPPLY` both match."""
    words = re.sub(r"[\W_]+", " ", str(value or "").replace(".", "").casefold()).split()
    while words and words[-1] in LEGAL_SUFFIXES:
        words.pop()
    return " ".join(words)


class Erp:
    def __init__(self, data: dict[str, Any]) -> None:
        self.vendors = data.get("vendors", [])
        self.purchase_orders = {norm_ref(po["po_number"]): po for po in data.get("purchase_orders", [])}
        # Several receipts per PO (partial deliveries) and the same invoice number from different vendors are normal.
        self.receipts: dict[str, list[dict]] = {}
        for gr in data.get("goods_receipts", []):
            self.receipts.setdefault(norm_ref(gr["po_number"]), []).append(gr)
        self.posted: dict[str, list[dict]] = {}
        for inv in data.get("posted_invoices", []):
            self.posted.setdefault(norm_ref(inv["invoice_number"]), []).append(inv)

    @classmethod
    def load(cls, path: Path | None = None) -> "Erp":
        path = path or SAMPLES_DIR / "erp.json"
        if not path.exists():
            return cls({})
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def vendor_by_tax_id(self, tax_id: Any) -> dict[str, Any] | None:
        key = norm_ref(tax_id)
        if key:
            for vendor in self.vendors:
                if vendor.get("tax_id") and norm_ref(vendor["tax_id"]) == key:
                    return vendor
        return None

    def vendor_by_name(self, name: Any) -> dict[str, Any] | None:
        key = norm_company(name)
        if key:
            for vendor in self.vendors:
                if key in {norm_company(n) for n in (vendor["name"], *vendor.get("aliases", []))}:
                    return vendor
        return None

    def vendor_candidates(self, name: Any, tax_id: Any = None) -> dict[str, dict[str, Any] | None]:
        """Vendor master records an extracted supplier may be: by tax id and by name/alias, kept apart so callers can
        see when the two disagree (a tax id read from the wrong party resolves to the wrong vendor)."""
        return {"tax_id": self.vendor_by_tax_id(tax_id), "name": self.vendor_by_name(name)}

    def resolve_vendor(self, name: Any, tax_id: Any = None) -> dict[str, Any] | None:
        """The vendor master record for an extracted supplier: by tax ID, else by name or alias; None when unknown or
        when the tax id and the name identify different vendors."""
        found = self.vendor_candidates(name, tax_id)
        by_tax, by_name = found["tax_id"], found["name"]
        if by_tax and by_name and by_tax["vendor_id"] != by_name["vendor_id"]:
            return None
        return by_tax or by_name

    def find_po(self, number: Any) -> dict[str, Any] | None:
        return self.purchase_orders.get(norm_ref(number)) if number else None

    def received_quantities(self, po_number: Any) -> dict[int, float]:
        """Quantity received per PO line, summed over every goods receipt for the PO."""
        received: dict[int, float] = {}
        for receipt in self.receipts.get(norm_ref(po_number), []):
            for line in receipt.get("lines", []):
                received[line["line"]] = received.get(line["line"], 0) + line["received"]
        return received

    def posted_invoices(self, number: Any, vendor_id: str | None = None) -> list[dict[str, Any]]:
        """Posted invoices with this number, optionally only those of one vendor."""
        found = self.posted.get(norm_ref(number), []) if number else []
        return [inv for inv in found if vendor_id is None or inv.get("vendor_id") == vendor_id]
