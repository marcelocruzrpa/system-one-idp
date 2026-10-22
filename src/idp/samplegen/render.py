"""Render sample specs to PDFs (Chromium), scanned image variants, ground truth and the mock ERP."""

import io
import json
import random
import shutil
from pathlib import Path

import pymupdf
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from PIL import Image, ImageFilter
from playwright.sync_api import sync_playwright

from idp.samplegen.data import Spec, all_specs, erp_data

TEMPLATES = Path(__file__).parent / "templates"
SCAN_DPI = 150
FOOTER = ('<div style="width:100%; font-size:7px; color:#9ca3af; text-align:right; padding-right:14mm;">'
          'Page <span class="pageNumber"></span> of <span class="totalPages"></span></div>')


def _pdf(page, spec: Spec, html: str) -> bytes:
    page.set_content(html, wait_until="load")
    if spec.paper == "receipt":
        height = page.evaluate("Math.ceil(document.body.getBoundingClientRect().bottom)")
        return page.pdf(width="80mm", height=f"{height + 24}px", print_background=True,
                        margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
    return page.pdf(format=spec.paper, print_background=True, display_header_footer=True,
                    header_template="<span></span>", footer_template=FOOTER,
                    margin={"top": "14mm", "bottom": "16mm", "left": "14mm", "right": "14mm"})


def _scan(pdf_bytes: bytes, seed: int) -> list[Image.Image]:
    """Grayscale 150-DPI raster with faint paper noise: looks scanned, but stays clean enough to read."""
    rng = random.Random(seed)
    pages = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page in pdf:
            pix = page.get_pixmap(dpi=SCAN_DPI, colorspace=pymupdf.csGRAY)
            image = Image.frombytes("L", (pix.width, pix.height), pix.samples)
            image = Image.blend(image, Image.effect_noise(image.size, 40), 0.06)
            pages.append(image.filter(ImageFilter.GaussianBlur(0.35)).rotate(rng.uniform(-0.25, 0.25), fillcolor=250))
    return pages


def _write_scan(images: list[Image.Image], base: Path) -> Path:
    if len(images) == 1:
        path = base.with_suffix(".png")
        images[0].save(path, optimize=True)
        return path
    path = base.with_suffix(".pdf")
    out = pymupdf.open()
    for image in images:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        w, h = (v * 72 / SCAN_DPI for v in image.size)
        out.new_page(width=w, height=h).insert_image(pymupdf.Rect(0, 0, w, h), stream=buf.getvalue())
    out.save(path)
    return path


def generate(out_dir: Path, *, only: set[str] | None = None) -> list[dict]:
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=True, undefined=StrictUndefined)
    specs = [s for s in all_specs() if not only or s.id in only]
    dirs = {name: out_dir / name for name in ("digital", "scanned", "ground_truth")}
    for d in dirs.values():
        if not only and d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    manifest = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        for i, spec in enumerate(specs):
            pdf_bytes = _pdf(page, spec, env.get_template(spec.template).render(**spec.context))
            digital = dirs["digital"] / f"{spec.id}.pdf"
            digital.write_bytes(pdf_bytes)
            scans = _scan(pdf_bytes, seed=i)
            scanned = _write_scan(scans, dirs["scanned"] / spec.id)
            truth = {"doc_id": spec.id, "doc_type": spec.doc_type, "note": spec.note, "pages": len(scans),
                     "fields": spec.fields, "expected_flags": spec.expected_flags,
                     "expected_route": spec.expected_route, "links": spec.links}
            (dirs["ground_truth"] / f"{spec.id}.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8")
            manifest.append({"doc_id": spec.id, "doc_type": spec.doc_type, "pages": len(scans),
                             "digital": digital.relative_to(out_dir).as_posix(), "scanned": scanned.relative_to(out_dir).as_posix(),
                             "expected_flags": spec.expected_flags, "expected_route": spec.expected_route, "note": spec.note})
        browser.close()

    (out_dir / "erp.json").write_text(json.dumps(erp_data(), indent=2), encoding="utf-8")
    if not only:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
