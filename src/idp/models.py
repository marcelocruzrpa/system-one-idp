"""Data passed between pipeline stages."""

from typing import Any, Literal

from pydantic import BaseModel, Field, PrivateAttr, computed_field

from idp.config import policy

Mode = Literal["text", "image", "hybrid"]
ModeChoice = Literal["auto", "text", "image", "hybrid"]
"""What a caller may ask for: `auto` resolves to a `Mode` from the document (`pipeline.pick_mode`)."""
Route = Literal["auto_approve", "review", "reject"]


class Page(BaseModel):
    number: int
    text: str
    png: bytes = Field(repr=False)
    """The model-ready page image: PNG, or JPEG when the PNG would exceed the size budget (`ingest.encode_for_models`).
    Both clients sniff the signature, so the field name is historical."""
    width: int
    height: int
    text_source: Literal["text_layer", "ocr"]


class Document(BaseModel):
    path: str
    pages: list[Page]

    @property
    def text(self) -> str:
        if len(self.pages) == 1:
            return self.pages[0].text
        return "\n".join(f"--- page {p.number} ---\n{p.text}" for p in self.pages)


class Cost(BaseModel):
    """USD at `pricing_per_million_input` in `config/policy.yaml`."""

    jev: float = 0.0
    clairic: float = 0.0
    total: float = 0.0
    billed: float = 0.0
    """What this run actually paid: requests answered from the response cache cost nothing."""


class Usage(BaseModel):
    jev_input_tokens: int = 0
    clairic_input_tokens: int = 0
    jev_requests: int = 0
    clairic_requests: int = 0
    jev_cached_tokens: int = 0
    """Input tokens of requests answered from the response cache (counted in `jev_input_tokens`, not billed)."""
    clairic_cached_tokens: int = 0

    def add(self, other: "Usage") -> None:
        for name in type(self).model_fields:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def cost(self) -> Cost:
        prices = policy()["pricing_per_million_input"]
        jev = self.jev_input_tokens * prices["jev"] / 1e6
        clairic = self.clairic_input_tokens * prices["clairic"] / 1e6
        cached = (self.jev_cached_tokens * prices["jev"] + self.clairic_cached_tokens * prices["clairic"]) / 1e6
        # Rounded and clamped: a fully cached run is $0, never the float residue "$-0.000000".
        return Cost(jev=jev, clairic=clairic, total=jev + clairic, billed=max(0.0, round(jev + clairic - cached, 12)))


class Classification(BaseModel):
    doc_type: str
    confidence: float
    probabilities: dict[str, float]
    signals: dict[str, float] = {}
    """Speculative Noul answers (probability of yes) asked alongside the classification."""
    model: str | None = None
    """The Jev version that answered (aliases such as `jev-latest` resolve server-side)."""


class FieldAnswer(BaseModel):
    """One Clairic answer.

    Clairic returns `span` only for text scalars and `spans` only for text lists. Text lists are de-duplicated
    (`value` holds unique strings while `spans` lists every occurrence), so `spans[i]` does not belong to `value[i]`
    in general; `extract.realign_list` repairs the common case. Number lists carry per-item `confidences` and no
    spans. The list-level `confidence` is a joint probability over all items and is not comparable to a scalar's.
    """

    type: str
    value: Any = None
    guess: Any = None
    confidence: float | None = None
    confidences: list[float] | None = None
    p_null: float | None = None
    span: list[int] | None = None
    spans: list[list[int]] | None = None
    source: str | None = None
    spell_confidence: float | None = None

    @property
    def is_list(self) -> bool:
        return isinstance(self.value, list)

    def item_confidence(self) -> float | None:
        """The weakest per-item confidence of a number list; None when Clairic gave none (text lists, scalars)."""
        return min(self.confidences) if self.confidences else None


class Extraction(BaseModel):
    fields: dict[str, FieldAnswer]
    state_text: str | None = None
    """The text Clairic's spans index into (the OCR text, or Clairic's own reading in image mode)."""
    truncated: bool = False
    _line_table: Any = PrivateAttr(default=None)
    """Memo for `idp.lines.line_table`, which every per-line check shares."""

    def value(self, name: str) -> Any:
        answer = self.fields.get(name)
        return None if answer is None else answer.value

    def values(self) -> dict[str, Any]:
        return {name: answer.value for name, answer in self.fields.items()}


class Check(BaseModel):
    id: str
    """Flag id, e.g. `total_mismatch` or `unsupported:vendor_name`."""
    stage: Literal["classify", "extract", "rules", "semantic", "match"]
    passed: bool
    detail: str
    probability: float | None = None
    """For model-judged checks, the probability that the problem is present."""
    data: dict[str, Any] = {}


class ProcessResult(BaseModel):
    path: str
    mode: Mode
    text: str
    page_count: int
    classification: Classification
    extraction: Extraction | None = None
    checks: list[Check] = []
    route: Route
    reasons: list[str] = []
    line_items: dict[str, Any] | None = None
    """Line items rebuilt from the document (`idp.lines.line_table`): rows, source and what was dropped."""
    usage: Usage = Usage()
    stage_usage: dict[str, Usage] = {}
    """Model usage per stage (`classify`, `extract`, `semantic`, `match`); `usage` is their sum."""
    timings: dict[str, float] = {}
    models: dict[str, str] = {}
    """Model versions that answered, e.g. {"jev": "jev-1.14.0", "clairic": "@cf/cloudflare/clairic"}."""
    errors: list[str] = []
    """Stages whose model call failed; each produced an `unverified:<stage>` check instead of aborting the document."""

    @computed_field
    @property
    def cost(self) -> Cost:
        return self.usage.cost()

    @property
    def flags(self) -> list[str]:
        return [c.id for c in self.checks if not c.passed]
