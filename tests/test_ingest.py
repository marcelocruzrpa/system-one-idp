"""Which PDF pages are OCR'd: scans (even with a small digital header) yes, digital and searchable-scan pages no."""

import pymupdf
from PIL import Image

from idp import ingest
from idp.ocr import Token

HEADER = "Archived copy - scanned 2026-10-01, batch 7"
BODY = " ".join(f"word{i}" for i in range(40))


def make_pdf(path, *, scan: bool, header: bool = False, hidden_text: str = "", visible_text: str = "") -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    if scan:
        page.insert_image(page.rect, stream=ingest.to_png(Image.new("RGB", (850, 1100), "white")))
    if header:
        page.insert_text((72, 30), HEADER)
    if hidden_text:
        page.insert_textbox(page.rect + (72, 72, -72, -72), hidden_text, render_mode=3)  # invisible OCR layer
    if visible_text:
        page.insert_textbox(page.rect + (72, 72, -72, -72), visible_text)
    pdf.save(path)


def sources(path, monkeypatch) -> list[str]:
    monkeypatch.setattr(ingest, "ocr_tokens", lambda png: [Token(0, 0, 50, 10, "ocr")])
    return [p.text_source for p in ingest.load(path).pages]


def test_scan_with_digital_header_is_ocred(tmp_path, monkeypatch):
    make_pdf(tmp_path / "scan.pdf", scan=True, header=True)
    assert sources(tmp_path / "scan.pdf", monkeypatch) == ["ocr"]


def test_searchable_scan_keeps_its_text_layer(tmp_path, monkeypatch):
    make_pdf(tmp_path / "searchable.pdf", scan=True, hidden_text=BODY)
    assert sources(tmp_path / "searchable.pdf", monkeypatch) == ["text_layer"]


def test_digital_page_keeps_its_text_layer(tmp_path, monkeypatch):
    make_pdf(tmp_path / "digital.pdf", scan=False, visible_text=BODY)
    assert sources(tmp_path / "digital.pdf", monkeypatch) == ["text_layer"]
