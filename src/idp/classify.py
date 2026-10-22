"""Doc-type classification with a Jev Choice, plus speculative Nouls asked in the same request."""

from typesafe_sdk import Choice, Noul

from idp.clients import Jev
from idp.config import taxonomy
from idp.models import Classification, Document, Mode, Usage

# Models see at most this many page images (Clairic's limit; also keeps Jev requests small).
MAX_PAGES_AS_IMAGES = 4


def jev_inputs(doc: Document, mode: Mode) -> tuple[dict, list[bytes] | None, str]:
    """State, page images and a phrase naming where the document is, for the given mode."""
    images = [p.png for p in doc.pages[:MAX_PAGES_AS_IMAGES]]
    if mode == "text":
        return {"document": doc.text}, None, "the document text in `document`"
    if mode == "image":
        return {}, images, "the document page images in `pages`"
    return {"document": doc.text}, images, "the document (page images in `pages`, OCR text in `document`)"


def classify(jev: Jev, doc: Document, mode: Mode) -> tuple[Classification, Usage]:
    tax = taxonomy()
    state, images, where = jev_inputs(doc, mode)
    questions = {
        "doc_type": Choice(instructions=f"Read {where}. {tax['instructions']}", criteria=tax["doc_types"]),
        **{f"signal_{name}": Noul(instructions=f"Read {where}. {text}") for name, text in tax["speculative"].items()},
    }
    result = jev.ask(state, questions, images=images)
    answer = result.choice("doc_type")
    classification = Classification(
        doc_type=answer["choice"],
        confidence=answer["confidence"],
        probabilities=answer["probabilities"],
        signals={name: result.noul(f"signal_{name}") for name in tax["speculative"]},
        model=result.model,
    )
    return classification, result.usage()
