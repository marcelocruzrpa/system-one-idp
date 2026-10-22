"""Jev verifier: narrow yes/no judgments where `true` means "something is wrong" (SDE-cascade style)."""

from typing import Any

from typesafe_sdk import Noul

from idp.classify import jev_inputs
from idp.clients import Jev
from idp.config import policy, threshold
from idp.models import Check, Classification, Document, Extraction, Mode, Usage

OUR_COMPANY = "the company in `policy.our_company` (or any of `policy.our_company_aliases`)"

DOC_CHECKS: dict[str, dict[str, Any]] = {
    "not_addressed_to_us": {
        "instructions": f"Is {{where}} addressed to, billed to, or held by a company other than {OUR_COMPANY}? "
                        "Look at the bill-to, customer, recipient, payee or account holder.",
        "criteria": {"true": "The recipient / bill-to / account holder is clearly a different company.",
                     "false": "The recipient is our company or one of its aliases (formatting differences don't matter)."},
    },
    "not_issued_by_us": {
        "instructions": f"Was the purchase order in {{where}} issued by a company other than {OUR_COMPANY}?",
        "criteria": {"true": "The buyer issuing the order is a different company.",
                     "false": "The buyer issuing the order is our company or one of its aliases."},
    },
    "short_payment": {
        "instructions": "Does the remittance advice in {where} show that the payer paid less than the full invoiced "
                        "amount on any invoice, e.g. deductions, discounts taken, withholding, disputes or partial payment?",
    },
    "non_reimbursable_items": {
        "instructions": "Does the receipt in {where} include items that are typically not reimbursable as business "
                        "expenses under a standard corporate expense policy, such as alcoholic drinks, personal care, "
                        "entertainment, or gifts?",
    },
}

# Checks already answered by the speculative Nouls in the classification request (no extra call).
SIGNAL_BACKED = {
    "already_paid": "has_paid_marking",
    "bank_detail_change": "requests_bank_detail_change",
}


def field_questions(name: str, instructions: str, where: str) -> dict[str, Noul]:
    return {
        f"unsupported:{name}": Noul(
            instructions=f"Compare `extracted.{name}` with {where}. The field should contain: {instructions}. "
                         "Is the extracted value NOT supported by the document?",
            criteria={"true": "The document does not state this value (or anything it directly follows from).",
                      "false": "The document states this value, possibly formatted differently."},
        ),
        f"wrong_role:{name}": Noul(
            instructions=f"The field `extracted.{name}` should contain: {instructions}. In {where}, does the extracted "
                         "value come from a part of the document that plays a different role (e.g. the buyer instead "
                         "of the seller, a subtotal instead of the grand total, or a different reference number)?",
        ),
    }


def semantic_checks(jev: Jev, doc: Document, mode: Mode, schema: dict, ext: Extraction,
                    classification: Classification) -> tuple[list[Check], Usage]:
    flag_at = threshold("semantic_flag")
    state, images, where = jev_inputs(doc, mode)
    company = policy()["our_company"]
    verify = [f for f in schema.get("verify_fields", []) if ext.value(f) is not None]
    state = {
        **state,
        "extracted": {f: ext.value(f) for f in verify},
        "policy": {"our_company": company["name"], "our_company_aliases": company["aliases"]},
    }
    questions: dict[str, Noul] = {}
    for check_id in schema.get("semantic", []):
        if check_id in DOC_CHECKS:
            spec = DOC_CHECKS[check_id]
            questions[check_id] = Noul(instructions=spec["instructions"].format(where=where), criteria=spec.get("criteria"))
    for name in verify:
        questions.update(field_questions(name, schema["fields"][name].get("instructions", name), where))

    probabilities: dict[str, float] = {}
    usage = Usage()
    if questions:
        # Question names on the wire stay plain identifiers; check ids keep their `kind:field` form.
        wire = {check_id.replace(":", "__"): check_id for check_id in questions}
        result = jev.ask(state, {name: questions[check_id] for name, check_id in wire.items()}, images=images)
        probabilities = {check_id: result.noul(name) for name, check_id in wire.items()}
        usage = result.usage()
    for check_id in schema.get("semantic", []):
        if check_id in SIGNAL_BACKED:
            probabilities[check_id] = classification.signals[SIGNAL_BACKED[check_id]]

    checks = []
    for check_id, p in probabilities.items():
        name = check_id.partition(":")[2]
        detail = f"P(problem) = {p:.2f}"
        if name:
            detail += f" for `{name}` = {ext.value(name)!r}"
        checks.append(Check(id=check_id, stage="semantic", passed=p < flag_at, detail=detail, probability=p))
    return checks, usage
