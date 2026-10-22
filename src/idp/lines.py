"""Line items rebuilt from the document text, cross-checked against Clairic's parallel lists.

Clairic returns the line-item columns as parallel lists. Its numeric lists are reliable on short tables, but its text
lists (descriptions, invoice numbers) pick up strings from outside the table (tax ids, addresses, SKU codes, subtotal
rows) and come back de-duplicated, so pairing the lists by index shifts rows and fabricates mismatches. The text the
models read (`ocr.rebuild_text`) already separates table columns with " | ", so the rows are parsed from it here and
every per-line check consumes them; Clairic's lists serve as a cross-check and as the fallback when the text has no
table (image-only mode, free-form documents). When neither yields consistent rows, the consumers raise
`unverified:line_items` (review) instead of comparing misaligned lists.
"""

import re
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any, Literal

from idp.models import Extraction, FieldAnswer

# Numeric column keys a schema's `lines` block may name, in the order they appear on commercial documents.
NUMERIC_COLUMNS = ("quantities", "unit_prices", "amounts", "ordered", "shipped")
# A table row whose first text cell starts with one of these is a totals / payment line, not a line item.
STOP_WORDS = ("subtotal", "sub-total", "sub total", "total", "tax", "vat", "gst", "sales tax", "est. sales tax", "balance",
              "amount due", "amount paid", "grand total", "change", "cash", "tendered", "visa", "mastercard", "amex",
              "american express", "discover", "card", "auth", "payment", "page")
SKU_RE = re.compile(r"^[A-Z]{1,5}(-[A-Z0-9]+)+$")
NUMBER_RE = re.compile(r"^[-+]?(\d{1,3}([.,]\d{3})+|\d+)([.,]\d{1,2})?$")
CURRENCY_RE = re.compile(r"^(?:US\$|[$€£¥]|USD|EUR|GBP|CHF|CAD|AUD|JPY)\s*", re.IGNORECASE)
IDENTIFIER_SUFFIXES = ("_number", "_numbers", "_id", "_reference")
AMOUNT_TOLERANCE = 0.011


def detect_eu(text: str) -> bool:
    """Decimal commas (`1.234,56`) when they outnumber decimal points (`1,234.56`)."""
    commas = len(re.findall(r"\d,\d{2}(?!\d)", text))
    dots = len(re.findall(r"\d\.\d{2}(?!\d)", text))
    return commas > dots


def parse_number(cell: str, eu: bool = False) -> float | None:
    """A whole cell as a number: `1,960.00`, `2.472,00`, `412,00`, `-`, `(20.00)`, `$95,031.16`; None for anything else."""
    s = cell.strip()
    if s in ("-", "–", "—"):
        return 0.0
    s = CURRENCY_RE.sub("", s).strip()
    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1].strip()
    if not NUMBER_RE.match(s):
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        tail = s.rsplit(",", 1)[1]
        thousands = len(tail) == 3 and (not eu or s.count(",") > 1)
        s = s.replace(",", "") if thousands else s.replace(",", ".")
    elif "." in s and eu and len(s.rsplit(".", 1)[1]) == 3:
        s = s.replace(".", "")
    value = float(s)
    return -value if negative else value


def _close(a: float, b: float, tol: float = AMOUNT_TOLERANCE) -> bool:
    return abs(a - b) <= tol


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", text.casefold())).strip()


def is_stop_row(first_text_cell: str, numeric_cells: int = 1) -> bool:
    """Totals, tax and payment lines: the first text cell starts with a stop word, or (on rows with a single number,
    where a label such as `Order total (EUR)` or `Total due (USD)` is all there is) contains `total` or `tax`."""
    norm = _norm(first_text_cell)
    if any(norm == w or norm.startswith(w + " ") for w in (_norm(s) for s in STOP_WORDS)):
        return True
    return numeric_cells == 1 and bool(re.search(r"\b(total|tax|vat|balance|due)\b", norm))


@dataclass
class Cell:
    text: str
    number: float | None
    start: int
    end: int

    @property
    def is_sku(self) -> bool:
        return bool(SKU_RE.match(self.text))


@dataclass
class Row:
    index: int
    """Line index in the text."""
    start: int
    end: int
    text: str
    cells: list[Cell]

    @property
    def numeric(self) -> list[Cell]:
        return [c for c in self.cells if c.number is not None]

    @property
    def textual(self) -> list[Cell]:
        return [c for c in self.cells if c.number is None and c.text]


def _split_glued(cell: str, eu: bool) -> tuple[float | None, str, str]:
    """`9,310.00 3 units returned damaged` -> (9310.0, '9,310.00', '3 units returned damaged'): a remark glued to a cell."""
    m = re.match(r"^(\S+)\s+(\S.*)$", cell.strip())
    if m:
        number = parse_number(m.group(1), eu)
        if number is not None and re.search(r"[A-Za-z]", m.group(2)):
            return number, m.group(1), m.group(2)
    return None, cell, ""


def parse_rows(text: str, numeric_columns: int) -> tuple[list[Row], list[int]]:
    """Table rows of `text`: lines with ` | ` cells, at least `numeric_columns` numeric cells and one text cell, whose
    first text cell is not a totals keyword. Also returns every line's start offset (for span -> line lookup)."""
    eu = detect_eu(text)
    rows, starts, offset = [], [], 0
    for index, line in enumerate(text.split("\n")):
        starts.append(offset)
        start, offset = offset, offset + len(line) + 1
        if " | " not in line:
            continue
        cells, pos, parts = [], start, line.split(" | ")
        for j, part in enumerate(parts):
            raw = part.strip()
            lead = part.index(raw) if raw else 0
            cstart = pos + lead
            number = parse_number(raw, eu) if raw else None
            if number is None and raw and j == len(parts) - 1:
                glued, head, rest = _split_glued(raw, eu)
                if glued is not None:
                    cells.append(Cell(head, glued, cstart, cstart + len(head)))
                    cells.append(Cell(rest, None, cstart + len(head) + 1, cstart + len(raw)))
                    pos += len(part) + 3
                    continue
            cells.append(Cell(raw, number, cstart, cstart + len(raw)))
            pos += len(part) + 3
        row = Row(index, start, start + len(line), line, cells)
        if (len(row.numeric) >= max(numeric_columns, 1) and row.textual
                and not is_stop_row(row.textual[0].text, len(row.numeric))):
            rows.append(row)
    return rows, starts


def pick_numeric(cells: list[Cell], k: int) -> list[Cell]:
    """Which `k` numeric cells of a row are the schema's columns: all of them when the counts agree; for one column
    the rightmost (`Amount paid` after `Invoice amount`, `Deduction`); for quantity/price/amount the window where
    quantity x price = amount; otherwise the leftmost (`Ordered | Shipped | Backordered`)."""
    if len(cells) <= k:
        return cells
    if k == 1:
        return [cells[-1]]
    if k == 3:
        for i in range(len(cells) - 2):
            q, p, a = (c.number for c in cells[i:i + 3])
            if _close(q * p, a, 0.01 + 0.0005 * abs(a)):
                return cells[i:i + 3]
    return cells[:k]


@dataclass
class LineItem:
    line: int
    description: str | None
    values: dict[str, float | None]
    """Numeric columns by schema key (`quantities`, `unit_prices`, `amounts`, `ordered`, `shipped`)."""
    text: str | None = None
    """The table row the item was read from (None when it came from Clairic's lists)."""
    row: int | None = None
    item_code: str | None = None

    @property
    def quantity(self) -> float | None:
        return self.values.get("quantities")

    @property
    def unit_price(self) -> float | None:
        return self.values.get("unit_prices")

    @property
    def amount(self) -> float | None:
        return self.values.get("amounts")

    def as_dict(self) -> dict[str, Any]:
        return {"line": self.line, "description": self.description, **self.values, "text": self.text,
                "item_code": self.item_code}


@dataclass
class LineTable:
    items: list[LineItem]
    source: Literal["rows", "lists", "none"]
    detail: str
    lengths: dict[str, int] = field(default_factory=dict)
    """How many entries Clairic returned per column."""
    agreement: dict[str, list[int]] = field(default_factory=dict)
    """Per numeric column: [entries Clairic agreed on, entries compared] (compared only when the counts match)."""
    dropped: list[str] = field(default_factory=list)
    """Clairic text entries that sit on no table row (tax ids, addresses, SKU cells, totals lines)."""
    relevant: bool = True
    """False when the schema has no line-item columns at all (bank statements)."""

    @property
    def usable(self) -> bool:
        return self.source != "none"

    def column(self, key: str) -> list[float | None]:
        return [item.values.get(key) for item in self.items]

    def as_dict(self) -> dict[str, Any]:
        return {"source": self.source, "detail": self.detail, "rows": [i.as_dict() for i in self.items],
                "lengths": self.lengths, "agreement": self.agreement, "dropped": self.dropped}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _numbers(values: list[Any]) -> list[float | None]:
    out = []
    for v in values:
        if isinstance(v, bool) or v is None:
            out.append(None)
        elif isinstance(v, int | float):
            out.append(float(v))
        else:
            out.append(parse_number(str(v)))
    return out


def locate_items(text: str, items: list[str], spans: list[list[int]] | None) -> list[int | None]:
    """Character offset of each text-list entry: its own span when the span text matches, else the first unused
    occurrence in the text. Never `spans[i]` for `value[i]`: Clairic de-duplicates values but not spans."""
    span_at: dict[str, list[int]] = {}
    for span in spans or []:
        if span and len(span) == 2 and 0 <= span[0] < span[1] <= len(text):
            span_at.setdefault(text[span[0]:span[1]], []).append(span[0])
    offsets: list[int | None] = []
    used: set[int] = set()
    for item in items:
        offset = next((a for a in span_at.get(item, []) if a not in used), None)
        if offset is None:
            pos = text.find(item)
            while pos != -1 and pos in used:
                pos = text.find(item, pos + 1)
            offset = pos if pos != -1 else None
        if offset is not None:
            used.add(offset)
        offsets.append(offset)
    return offsets


def _reconciles(amounts: list[float | None], ext: Extraction) -> bool:
    """Do these line amounts add up to a total the document states?"""
    values = [a for a in amounts if a is not None]
    if not values or len(values) != len(amounts):
        return False
    total = sum(values)
    refs = []
    for name in ("subtotal", "total", "payment_total"):
        v = ext.value(name)
        if isinstance(v, int | float) and not isinstance(v, bool):
            refs.append(float(v))
    t, tax = ext.value("total"), ext.value("tax_amount")
    if isinstance(t, int | float) and isinstance(tax, int | float):
        refs.append(float(t) - float(tax))
    return any(_close(total, r) for r in refs)


def _lists_table(cols: dict[str, str], lists: dict[str, list], numeric_keys: list[str]) -> list[LineItem] | None:
    """Clairic's lists as line items, only when every non-empty list has the same length."""
    lengths = {len(v) for v in lists.values() if v}
    if len(lengths) != 1:
        return None
    n = lengths.pop()
    descriptions = lists.get("descriptions") or [None] * n
    numbers = {k: _numbers(lists[k]) if lists.get(k) else [None] * n for k in numeric_keys}
    return [LineItem(line=i + 1, description=None if descriptions[i] is None else str(descriptions[i]),
                     values={k: numbers[k][i] for k in numeric_keys}) for i in range(n)]


def _describe(row: Row, attached: list[str], identifier: bool) -> tuple[str | None, str | None]:
    """(description, item code) for a row: the text cell Clairic pointed at, else the wordiest non-SKU cell."""
    textual = row.textual
    if not textual:
        return None, None
    code = next((c.text for c in textual if c.is_sku), None)
    if identifier:
        return textual[0].text, None
    candidates = [c for c in textual if not c.is_sku] or textual
    if attached:
        wanted = [w for w in (_norm(a) for a in attached) if len(w) >= 3]
        hits = [c for c in candidates if any(w == _norm(c.text) or w in _norm(c.text) or _norm(c.text) in w for w in wanted)]
        if hits:
            return max(hits, key=lambda c: (len(c.text.split()), len(c.text))).text, code
    return max(candidates, key=lambda c: (len(c.text.split()), len(c.text))).text, code


def build_line_table(schema: dict, ext: Extraction) -> LineTable:
    cols: dict[str, str] = schema.get("lines") or {}
    if not cols:
        return LineTable([], "none", "The schema has no line-item columns.", relevant=False)
    lists = {key: _as_list(ext.value(name)) for key, name in cols.items()}
    lengths = {key: len(v) for key, v in lists.items()}
    numeric_keys = [k for k in NUMERIC_COLUMNS if k in cols]
    if not any(lists.values()):
        return LineTable([], "none", "Clairic returned no line items.", lengths=lengths)

    text = ext.state_text or ""
    rows: list[Row] = []
    starts: list[int] = []
    if text and numeric_keys:
        rows, starts = parse_rows(text, len(numeric_keys))
    k = len(numeric_keys)
    anchor = _numbers(lists["amounts"]) if "amounts" in cols and lists.get("amounts") else []

    selected: list[tuple[Row, Cell | None]] = []
    unplaced = 0
    if rows and k == 1 and anchor:
        # Receipts and remittances: one numeric column, so totals and payment lines look like rows. Keep the rows
        # that carry the amounts Clairic extracted, in order (duplicates and repeated amounts included).
        taken: set[int] = set()
        cursor = 0
        for value in anchor:
            hit = None
            if value is not None:
                order = [i for i in range(cursor, len(rows))] + [i for i in range(cursor) if i not in taken]
                for i in order:
                    if i in taken:
                        continue
                    cells = [c for c in rows[i].numeric if _close(c.number, value)]
                    if cells:
                        hit = (i, cells[-1])
                        break
            if hit is None:
                unplaced += 1
                continue
            taken.add(hit[0])
            cursor = max(cursor, hit[0] + 1)
            selected.append((rows[hit[0]], hit[1]))
        selected.sort(key=lambda rc: rc[0].index)
    else:
        selected = [(row, None) for row in rows]

    identifier = str(cols.get("descriptions", "")).endswith(IDENTIFIER_SUFFIXES)
    items: list[LineItem] = []
    dropped: list[str] = []
    if selected:
        # Place Clairic's text entries on their rows (by span text, never by index) to pick the description cell.
        desc_items = [str(v) for v in lists.get("descriptions", []) if v is not None]
        answer = ext.fields.get(cols.get("descriptions", "")) if "descriptions" in cols else None
        offsets = locate_items(text, desc_items, answer.spans if answer else None)
        by_line: dict[int, list[str]] = {}
        for item, offset in zip(desc_items, offsets):
            line = bisect_right(starts, offset) - 1 if offset is not None else None
            by_line.setdefault(line if line is not None else -1, []).append(item)
        selected_lines = {row.index for row, _ in selected}
        for line, found in by_line.items():
            if line not in selected_lines:
                dropped.extend(found)
        for n, (row, anchored) in enumerate(selected, start=1):
            numeric = pick_numeric(row.numeric, k) if anchored is None else [anchored]
            values = {key: (numeric[i].number if i < len(numeric) else None) for i, key in enumerate(numeric_keys)}
            description, code = _describe(row, by_line.get(row.index, []), identifier)
            items.append(LineItem(n, description, values, text=row.text, row=row.index, item_code=code))

    from_lists = _lists_table(cols, lists, numeric_keys)
    if items and from_lists is not None and len(items) != len(from_lists):
        # Deterministic parse and model disagree on the row count: trust whichever adds up to the stated total.
        if _reconciles([i.amount for i in from_lists], ext) and not _reconciles([i.amount for i in items], ext):
            items, dropped, unplaced = [], [], 0
            source, chosen = "lists", from_lists
        else:
            source, chosen = "rows", items
    elif items:
        source, chosen = "rows", items
    elif from_lists is not None:
        source, chosen = "lists", from_lists
    else:
        detail = (f"Clairic's line-item lists have different lengths ({lengths}) and the text has no table rows to "
                  "align them with.")
        return LineTable([], "none", detail, lengths=lengths)

    agreement: dict[str, list[int]] = {}
    for key in numeric_keys:
        got = _numbers(lists[key]) if lists.get(key) else []
        if got and len(got) == len(chosen):
            agreed = sum(1 for a, b in zip(got, (i.values.get(key) for i in chosen))
                         if a is not None and b is not None and _close(a, b))
            agreement[key] = [agreed, len(chosen)]
    if source == "rows":
        detail = f"{len(chosen)} line item(s) read from the document's table"
        if agreement:
            detail += "; Clairic agreed on " + ", ".join(f"{a}/{n} {k.replace('_', ' ')}" for k, (a, n) in agreement.items())
        if dropped:
            detail += f"; {len(dropped)} Clairic text entr{'y' if len(dropped) == 1 else 'ies'} not on a row dropped " \
                      f"({', '.join(repr(d) for d in dropped[:4])}{', ...' if len(dropped) > 4 else ''})"
        if unplaced:
            detail += f"; {unplaced} Clairic amount(s) matched no row"
    else:
        detail = f"{len(chosen)} line item(s) taken from Clairic's lists (no table rows found in the text)"
    return LineTable(chosen, source, detail + ".", lengths=lengths, agreement=agreement, dropped=dropped)


def reconcile(schema: dict, ext: Extraction) -> LineTable:
    """The document's line items (memoised on the extraction). Also writes the reconciled columns back into the
    extraction fields, keeping Clairic's raw lists in `guess`, so every consumer and the UI see aligned rows."""
    memo = ext._line_table
    if memo is not None and memo[0] == id(schema):
        return memo[1]
    table = build_line_table(schema, ext)
    if table.source == "rows":
        cols: dict[str, str] = schema.get("lines") or {}
        for key, name in cols.items():
            values = [i.description for i in table.items] if key == "descriptions" else table.column(key)
            answer = ext.fields.get(name)
            if answer is None:
                answer = ext.fields[name] = FieldAnswer(type="list")
            answer.guess = answer.value
            answer.value = values
            answer.source = "rows"
    ext._line_table = (id(schema), table)
    return table
