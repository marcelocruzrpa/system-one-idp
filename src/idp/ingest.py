"""Load a PDF or image into pages of text (text layer or OCR) plus model-ready page images."""

import io
from pathlib import Path

import pymupdf
from PIL import Image, ImageOps

from idp.models import Document, Page
from idp.ocr import Token, ocr_tokens, rebuild_text

# Jev image bounds (each side) — Clairic's 4 MiB/image limit is comfortably met at this size.
MAX_SIDE = 1568
MIN_SIDE = 280
OCR_DPI = 200
# A PDF page with fewer words than this is treated as a scan and OCR'd.
MIN_TEXT_LAYER_WORDS = 5
# So is a page where one image covers at least this share of the page with fewer than MIN_WORDS_OVER_SCAN text-layer
# words on it: a scan with a digital header or stamp added. Searchable scans keep their (hidden) OCR text layer.
SCAN_MIN_PAGE_SHARE = 0.5
MIN_WORDS_OVER_SCAN = 25
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
# A page image larger than this as PNG is sent as JPEG instead: Clairic caps an image at 4 MiB, and base64 adds a third.
MAX_PNG_BYTES = 2 * 1024 * 1024
JPEG_QUALITIES = (85, 75, 60)


def fit_for_models(image: Image.Image) -> Image.Image:
    """Scale so the longest side is at most MAX_SIDE, and pad so the shortest side is at least MIN_SIDE."""
    image = image.convert("RGB")
    scale = min(1.0, MAX_SIDE / max(image.size))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    if min(image.size) < MIN_SIDE:
        padded = Image.new("RGB", (max(image.width, MIN_SIDE), max(image.height, MIN_SIDE)), "white")
        padded.paste(image, (0, 0))
        image = padded
    return image


def to_png(image: Image.Image) -> bytes:
    """Lossless PNG, used for the OCR render (and for model pages that fit MAX_PNG_BYTES)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def encode_for_models(image: Image.Image) -> tuple[bytes, str]:
    """Encode a page for the models: PNG when it fits in MAX_PNG_BYTES, else JPEG at falling quality until it does.

    Returns the bytes and their content type (`image/png` or `image/jpeg`). Dense or noisy scans can exceed the PNG
    budget; JPEG at 85 keeps printed text crisp at a fraction of the size. The last quality is returned even if it
    still exceeds the budget, so a page is never dropped here (Clairic reports an oversize image itself).
    """
    png = to_png(image)
    if len(png) <= MAX_PNG_BYTES:
        return png, "image/png"
    jpeg = b""
    for quality in JPEG_QUALITIES:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
        jpeg = buf.getvalue()
        if len(jpeg) <= MAX_PNG_BYTES:
            break
    return jpeg, "image/jpeg"


def _render(page: pymupdf.Page, dpi: float) -> Image.Image:
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _render_fitted(page: pymupdf.Page) -> Image.Image:
    dpi = 72 * MAX_SIDE / max(page.rect.width, page.rect.height)
    return fit_for_models(_render(page, dpi))


def _is_scan(page: pymupdf.Page, words: list) -> bool:
    """Too little text layer, or a page-sized image that the text layer barely covers."""
    if len(words) < MIN_TEXT_LAYER_WORDS:
        return True
    for info in page.get_image_info():
        rect = pymupdf.Rect(info["bbox"]) & page.rect
        if rect.get_area() >= SCAN_MIN_PAGE_SHARE * page.rect.get_area():
            on_image = sum(rect.contains(pymupdf.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2)) for w in words)
            if on_image < MIN_WORDS_OVER_SCAN:
                return True
    return False


def _load_pdf(path: Path, force_ocr: bool) -> list[Page]:
    pages = []
    with pymupdf.open(path) as pdf:
        if pdf.needs_pass:
            raise ValueError("PDF is password-protected")
        if pdf.page_count == 0:
            raise ValueError("PDF has no pages")
        for i, page in enumerate(pdf, start=1):
            words = page.get_text("words")
            model_image = _render_fitted(page)
            if not force_ocr and not _is_scan(page, words):
                text = rebuild_text([Token(w[0], w[1], w[2], w[3], w[4]) for w in words])
                source = "text_layer"
            else:
                text = rebuild_text(ocr_tokens(to_png(_render(page, OCR_DPI))))
                source = "ocr"
            # `Page.png` holds whatever encode_for_models chose (PNG, or JPEG for an oversize page); the clients sniff it.
            encoded, _ = encode_for_models(model_image)
            pages.append(Page(number=i, text=text, png=encoded, width=model_image.width,
                              height=model_image.height, text_source=source))
    return pages


def _load_image(path: Path) -> list[Page]:
    # EXIF orientation is applied here because Jev does not apply it.
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    text = rebuild_text(ocr_tokens(to_png(image)))
    model_image = fit_for_models(image)
    encoded, _ = encode_for_models(model_image)
    return [Page(number=1, text=text, png=encoded, width=model_image.width,
                 height=model_image.height, text_source="ocr")]


def load(path: str | Path, *, force_ocr: bool = False) -> Document:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = _load_pdf(path, force_ocr)
    elif suffix in IMAGE_SUFFIXES:
        pages = _load_image(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix} (expected PDF, PNG or JPEG)")
    return Document(path=str(path), pages=pages)
