"""Line items come from the page's table rows; Clairic's lists only cross-check them and never shift rows."""

from idp.config import schema as load_schema
from idp.extract import realign_list
from idp.lines import build_line_table, locate_items, parse_number, parse_rows, reconcile
from idp.models import Extraction, FieldAnswer


def ext(state_text=None, **values) -> Extraction:
    fields = {k: FieldAnswer(type="list" if isinstance(v, list) else "x", value=v, confidence=0.9, p_null=0.01)
              for k, v in values.items()}
    return Extraction(fields=fields, state_text=state_text)


RECEIPT = ("HARBOR GRILL & TAPROOM\n55 Embarcadero W\nOakland, CA 94607\n10/14/2026 | 19:47\nTXN 1014-7730-05 | REG 02\n"
           "Grilled salmon | 28.00\nCaesar salad | 14.50\nHazy IPA draft | 8.00\nHazy IPA draft | 8.00\nSparkling water | 4.50\n"
           "SUBTOTAL | 63.00\nTAX 10.25% | 6.46\nTOTAL USD | 69.46\nMASTERCARD ************7730 | 69.46\nAUTH 974004 | APPROVED\n")
REMITTANCE = ("Invoice No. | Invoice date | Invoice amount | Deduction | Amount paid Remarks\n"
              "ACM-INV-3290 | 08/28/2026 | 5,600.00 | 5,600.00\n"
              "ACM-INV-3312 | 09/15/2026 | 9,800.00 | 490.00 | 9,310.00 3 units returned damaged\n"
              "Total payment: USD 14,910.00\n")
EU_INVOICE = ("Helios Components GmbH | RECHNUNG / INVOICE\nUSt-IdNr. / VAT ID: DE287456913\nBILL TO\nAcme Robotics Inc.\n"
              "Tax ID: 94-3187265\n\nArtikel / Item | Beschreibung / Description | Menge / Qty | Einzelpreis / Unit price | Betrag / Amount (EUR)\n"
              "HC-PLE60-10 | Planetengetriebe / Planetary gearbox PLE60 ratio 10:1 | 6 | 412,00 | 2.472,00\n"
              "HC-AE17 | Absolute encoder AE-17, 17 bit | 10 | 129,50 | 1.295,00\n"
              "HC-SFU1204-500 | Precision ball screw SFU1204, length 500 mm | 8 | 74,90 | 599,20\n\n"
              "Zwischensumme / Subtotal | 4.366,20\nUSt. 0% / VAT 0% (export) | 0,00\nGesamtbetrag / Total (EUR) | 4.366,20\n")
DELIVERY = ("Item | Description | Ordered | Shipped | Backordered\n"
            "HC-PLE60-10 | Planetary gearbox PLE60 i=10 | 6 | 6 | 0\n"
            "HC-AE17 | Absolute encoder AE17 | 10 | 8 | 2\n"
            "HC-SFU1204-500 | Ball screw SFU1204 500 mm | 8 | 8 | 0\n\n4 packages. Please inspect goods on arrival.\n")


def test_parse_number_formats():
    assert parse_number("1,960.00") == 1960.0
    assert parse_number("2.472,00", eu=True) == 2472.0
    assert parse_number("412,00", eu=True) == 412.0
    assert parse_number("1,450") == 1450.0
    assert parse_number("$95,031.16") == 95031.16
    assert parse_number("(20.00)") == -20.0
    assert parse_number("-") == 0.0
    for text in ("09/15/2026", "19:47", "9.375%", "REG 02", "ACM-INV-3312", "HDMI cable 2 m", "2026-08-18", "94-3187265"):
        assert parse_number(text) is None, text


def test_rows_skip_headers_totals_and_payment_lines():
    rows, _ = parse_rows(RECEIPT, 1)
    assert [r.textual[0].text for r in rows] == ["Grilled salmon", "Caesar salad", "Hazy IPA draft", "Hazy IPA draft",
                                                 "Sparkling water"]
    rows, _ = parse_rows(EU_INVOICE, 3)
    assert len(rows) == 3 and rows[0].numeric[-1].number == 2472.0 and rows[1].numeric[0].number == 10.0


def test_glued_remark_in_the_last_cell():
    rows, _ = parse_rows(REMITTANCE, 1)
    assert len(rows) == 2
    assert [c.number for c in rows[1].numeric] == [9800.0, 490.0, 9310.0]
    assert rows[1].textual[-1].text == "3 units returned damaged"


def test_receipt_rows_anchored_on_amounts_keep_repeated_items():
    schema = load_schema("receipt")
    e = ext(RECEIPT, item_descriptions=["Grilled salmon", "Caesar salad", "Hazy IPA draft", "Sparkling water", "REG 02"],
            item_amounts=[28.0, 14.5, 8.0, 8.0, 4.5], subtotal=63.0, tax_amount=6.46, total=69.46)
    table = build_line_table(schema, e)
    assert table.source == "rows"
    assert [i.description for i in table.items] == ["Grilled salmon", "Caesar salad", "Hazy IPA draft", "Hazy IPA draft",
                                                    "Sparkling water"]
    assert table.column("amounts") == [28.0, 14.5, 8.0, 8.0, 4.5] and table.dropped == ["REG 02"]


def test_remittance_identifiers_come_from_the_first_cell():
    schema = load_schema("remittance_advice")
    e = ext(REMITTANCE, invoice_numbers=["AC", "M-INV-3290", "M-INV-3312"], paid_amounts=[5600.0, 9310.0], payment_total=14910.0)
    table = build_line_table(schema, e)
    assert [i.description for i in table.items] == ["ACM-INV-3290", "ACM-INV-3312"]
    assert table.column("amounts") == [5600.0, 9310.0]


def test_eu_numbers_and_sku_column():
    schema = load_schema("invoice")
    e = ext(EU_INVOICE, line_descriptions=["Tax", "Planetengetriebe / Planetary gearbox PLE60 ratio 10:1",
                                           "Absolute encoder AE-17, 17 bit", "HC-SFU1204-500", "Precision ball screw SFU1204, length"],
            line_quantities=[6, 10, 8], line_unit_prices=[412.0, 129.5, 74.9], line_amounts=[2472.0, 1295.0, 599.2],
            subtotal=4366.2, tax_amount=0.0, total=4366.2)
    table = build_line_table(schema, e)
    assert table.source == "rows" and len(table.items) == 3
    assert table.items[2].description == "Precision ball screw SFU1204, length 500 mm"  # the full cell, not the fragment
    assert [i.item_code for i in table.items] == ["HC-PLE60-10", "HC-AE17", "HC-SFU1204-500"]
    assert table.column("unit_prices") == [412.0, 129.5, 74.9] and table.dropped == ["Tax"]
    assert table.agreement == {"quantities": [3, 3], "unit_prices": [3, 3], "amounts": [3, 3]}


def test_delivery_note_columns_are_ordered_and_shipped():
    schema = load_schema("delivery_note")
    e = ext(DELIVERY, line_descriptions=["Packing slip", "Planetary gearbox PLE60 i=10", "HC-AE17", "Absolute encoder AE17",
                                         "Ball screw SFU1204"],
            line_ordered_quantities=[6, 10, 8], line_shipped_quantities=[6, 8, 8])
    table = build_line_table(schema, e)
    assert [i.values for i in table.items] == [{"ordered": 6.0, "shipped": 6.0}, {"ordered": 10.0, "shipped": 8.0},
                                                {"ordered": 8.0, "shipped": 8.0}]
    assert table.items[2].description == "Ball screw SFU1204 500 mm"


def test_rows_win_when_clairic_lists_drift_on_long_tables():
    schema = load_schema("invoice")
    rows = "\n".join(f"BF-{100 + i} | Item {i} | {i} | 2.00 | {2 * i:.2f}" for i in range(1, 33))
    text = "Description | Qty | Unit price | Amount (USD)\n" + rows + "\n\nSubtotal | 1,056.00\n"
    # 28 drifting amounts for 32 rows, with low confidence: the rows are right, the list is not.
    e = ext(text, line_descriptions=[f"Item {i}" for i in range(1, 33)], line_quantities=list(range(1, 28)),
            line_unit_prices=[2.0] * 25, line_amounts=[2.0 * i for i in range(1, 29)], subtotal=1056.0)
    table = build_line_table(schema, e)
    assert table.source == "rows" and len(table.items) == 32
    assert table.column("amounts")[-1] == 64.0 and table.agreement == {}


def test_lists_are_the_fallback_without_table_text_and_only_when_consistent():
    schema = load_schema("invoice")
    consistent = ext(None, line_descriptions=["a", "b"], line_quantities=[1, 2], line_unit_prices=[3.0, 4.0], line_amounts=[3.0, 8.0])
    assert build_line_table(schema, consistent).source == "lists"
    stray = ext(None, line_descriptions=["Tax", "a", "b"], line_quantities=[1, 2], line_unit_prices=[3.0, 4.0], line_amounts=[3.0, 8.0])
    assert build_line_table(schema, stray).source == "none"
    nothing = ext(None, line_descriptions=None, line_quantities=None, line_unit_prices=None, line_amounts=None)
    assert not build_line_table(schema, nothing).usable
    assert not build_line_table(load_schema("bank_statement"), ext(None)).relevant


def test_model_lists_beat_a_misparsed_table_only_when_they_add_up():
    schema = load_schema("invoice")
    # A stray line that looks like a row ("Deposit received | 1 | 500.00 | 500.00") is not in Clairic's lists and
    # does not add up to the subtotal; Clairic's two rows do.
    text = ("Description | Qty | Unit price | Amount\nWidget | 2 | 10.00 | 20.00\nGadget | 1 | 5.00 | 5.00\n"
            "Deposit received | 1 | 500.00 | 500.00\nSubtotal | 25.00\n")
    e = ext(text, line_descriptions=["Widget", "Gadget"], line_quantities=[2, 1], line_unit_prices=[10.0, 5.0],
            line_amounts=[20.0, 5.0], subtotal=25.0, total=25.0)
    table = build_line_table(schema, e)
    assert table.source == "lists" and len(table.items) == 2


def test_reconcile_writes_rows_back_and_memoises():
    schema = load_schema("receipt")
    e = ext(RECEIPT, item_descriptions=["REG 02", "Grilled salmon", "Caesar salad", "Hazy IPA draft", "Sparkling water"],
            item_amounts=[28.0, 14.5, 8.0, 8.0, 4.5], subtotal=63.0)
    table = reconcile(schema, e)
    assert reconcile(schema, e) is table
    assert e.value("item_descriptions") == ["Grilled salmon", "Caesar salad", "Hazy IPA draft", "Hazy IPA draft", "Sparkling water"]
    assert e.fields["item_descriptions"].guess[0] == "REG 02" and e.fields["item_descriptions"].source == "rows"


def test_locate_items_by_span_text_not_index():
    text = "x BF-1 | Pen | 1 | 2.00 | 2.00\nBF-2 | Pencil | 1 | 1.00 | 1.00\n"
    spans = [[2, 6], [9, 12], [30, 34], [37, 43]]  # BF-1, Pen, BF-2, Pencil
    # Clairic de-duplicated nothing here, but the value order differs from the span order.
    offsets = locate_items(text, ["Pen", "Pencil", "BF-1"], spans)
    assert [text[o:o + 3] for o in offsets] == ["Pen", "Pen", "BF-"]


def test_realign_list_restores_deduplicated_occurrences():
    text = "a | 1.00\nHazy IPA draft | 8.00\nHazy IPA draft | 8.00\nWater | 4.50\n"
    first, second, water = text.index("Hazy"), text.index("Hazy", 10), text.index("Water")
    answer = FieldAnswer(type="list", value=["Hazy IPA draft", "Water"],
                         spans=[[first, first + 14], [second, second + 14], [water, water + 5]])
    fixed = realign_list(answer, text)
    assert fixed.value == ["Hazy IPA draft", "Hazy IPA draft", "Water"] and fixed.guess == ["Hazy IPA draft", "Water"]
    untouched = FieldAnswer(type="list", value=["Water", "Hazy IPA draft"], spans=[[water, water + 5], [first, first + 14]])
    assert realign_list(untouched, text).value == ["Water", "Hazy IPA draft"]
