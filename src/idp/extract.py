"""Typed field extraction with Clairic, driven by the doc type's schema."""

from typing import Any

from idp.classify import MAX_PAGES_AS_IMAGES
from idp.clients import Clairic
from idp.models import Document, Extraction, FieldAnswer, Mode, Usage


def realign_list(answer: FieldAnswer, text: str | None) -> FieldAnswer:
    """Undo Clairic's de-duplication of text lists.

    Clairic returns the unique values of a text list in first-occurrence order while `spans` lists every occurrence
    (in text mode; in image/hybrid mode `spans` may be cut to the number of unique values). When the span texts
    de-duplicate to exactly the values, the occurrences ARE the list: rebuild `value` from them so repeated rows
    ("Hazy IPA draft" twice) survive and `value[i]` belongs to `spans[i]` again. Anything else is left untouched.
    """
    if not text or not answer.spans or not isinstance(answer.value, list) or not answer.value:
        return answer
    if not all(isinstance(v, str) for v in answer.value):
        return answer
    texts = []
    for span in answer.spans:
        if len(span) != 2 or not (0 <= span[0] < span[1] <= len(text)):
            return answer
        texts.append(text[span[0]:span[1]])
    if texts == answer.value:
        return answer
    unique = list(dict.fromkeys(texts))
    if unique != answer.value[:len(unique)]:
        return answer
    rest = answer.value[len(unique):]  # occurrences whose spans were cut off: kept, located by text when needed
    answer.guess = list(answer.value) if answer.guess is None else answer.guess
    answer.value = texts + rest
    return answer


def extract(clairic: Clairic, doc: Document, schema: dict, mode: Mode) -> tuple[Extraction, Usage, str]:
    """Run the extraction. Returns the extraction, its usage and the model name Clairic reported."""
    state = doc.text if mode in ("text", "hybrid") else ""
    images = [p.png for p in doc.pages[:MAX_PAGES_AS_IMAGES]] if mode in ("image", "hybrid") else None
    result: dict[str, Any] = clairic.extract(schema["fields"], state=state, images=images)
    answers = result.get("answers", {})
    state_text = result.get("state_text") or state or None
    fields = {}
    for name in schema["fields"]:
        if name in answers:
            answer = FieldAnswer.model_validate(answers[name])
            fields[name] = realign_list(answer, state_text) if schema["fields"][name].get("type") == "list" else answer
    extraction = Extraction(fields=fields, state_text=state_text, truncated=bool(result.get("truncated")))
    tokens = (result.get("usage") or {}).get("input_tokens", 0)
    usage = Usage(clairic_input_tokens=tokens, clairic_requests=1, clairic_cached_tokens=tokens if result.get("cached") else 0)
    # The policy's full model id (the API echoes only the short name "clairic").
    return extraction, usage, str(getattr(clairic, "model", None) or result.get("model") or "clairic")
