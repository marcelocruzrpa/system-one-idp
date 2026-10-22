"""OCR (RapidOCR) and reading-order line reconstruction shared by OCR and PDF text layers."""

from dataclasses import dataclass
from functools import cache
from statistics import median

# A horizontal gap wider than this many line-heights separates table columns with " | ".
COLUMN_GAP_RATIO = 1.0
# A vertical gap taller than this many line-heights inserts a blank line between blocks.
BLOCK_GAP_RATIO = 1.2


@dataclass(frozen=True)
class Token:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def h(self) -> float:
        return max(self.y1 - self.y0, 1.0)


@cache
def _engine():
    from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
    from rapidocr.utils.download_file import DownloadFileException

    # The default (Chinese) recognizer drops most spaces in English text; the English PP-OCRv5 model keeps them.
    # The models are fetched from modelscope.cn on first use; a failed download is not cached, so a later call retries.
    try:
        return RapidOCR(params={"Rec.lang_type": LangRec.EN, "Rec.ocr_version": OCRVersion.PPOCRV5,
                                "Rec.model_type": ModelType.MOBILE, "Global.log_level": "warning"})
    except DownloadFileException as exc:
        raise RuntimeError("OCR model download failed (modelscope.cn unreachable): run "
                           "`uv run idp ocr samples/scanned/invoice-01.png` once with internet access.") from exc


def ocr_tokens(image: bytes) -> list[Token]:
    result = _engine()(image)
    if result.boxes is None:
        return []
    tokens = []
    for box, text in zip(result.boxes, result.txts):
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        tokens.append(Token(min(xs), min(ys), max(xs), max(ys), text))
    return tokens


def _group_lines(tokens: list[Token]) -> list[list[Token]]:
    lines: list[list[Token]] = []
    for token in sorted(tokens, key=lambda t: (t.cy, t.x0)):
        for line in reversed(lines[-3:]):
            line_cy = median(t.cy for t in line)
            line_h = median(t.h for t in line)
            if abs(token.cy - line_cy) <= 0.5 * min(token.h, line_h):
                line.append(token)
                break
        else:
            lines.append([token])
    return sorted(lines, key=lambda line: median(t.cy for t in line))


def rebuild_text(tokens: list[Token]) -> str:
    """Rebuild reading-order text: tokens on the same visual line join; wide gaps become ` | ` column breaks."""
    tokens = [t for t in tokens if t.text.strip()]
    if not tokens:
        return ""
    out: list[str] = []
    prev_bottom: float | None = None
    for line in _group_lines(tokens):
        line.sort(key=lambda t: t.x0)
        h = median(t.h for t in line)
        top = min(t.y0 for t in line)
        if prev_bottom is not None and top - prev_bottom > BLOCK_GAP_RATIO * h:
            out.append("")
        parts = [line[0].text.strip()]
        for prev, cur in zip(line, line[1:]):
            parts.append(" | " if cur.x0 - prev.x1 > COLUMN_GAP_RATIO * h else " ")
            parts.append(cur.text.strip())
        out.append("".join(parts))
        prev_bottom = max(t.y1 for t in line)
    return "\n".join(out)
