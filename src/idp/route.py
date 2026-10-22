"""Route a processed document: reject on hard fails, review on any other flag, otherwise auto-approve."""

from idp.config import policy
from idp.models import Check, Route


# Failed checks with these ids or prefixes describe extraction / input quality, not a defect of the document.
QUALITY_PREFIXES = ("missing:", "low_confidence:", "unsupported:", "wrong_role:", "unverified:")
QUALITY_IDS = {"uncertain_doc_type", "unsupported_doc_type", "multiple_documents", "pages_not_sent", "extraction_truncated"}


def is_quality_flag(check_id: str) -> bool:
    return check_id in QUALITY_IDS or check_id.startswith(QUALITY_PREFIXES)


def route(checks: list[Check]) -> tuple[Route, list[str]]:
    """Reject flags first, other business flags next, quality flags last: the reasons read in order of weight."""
    failed = [c for c in checks if not c.passed]
    reject_flags = set(policy()["reject_flags"])
    rank = {c.id: (0 if c.id in reject_flags else 2 if is_quality_flag(c.id) else 1) for c in failed}
    ordered = sorted(failed, key=lambda c: rank[c.id])
    reasons = [f"{c.id}: {c.detail}" for c in ordered]
    if any(c.id in reject_flags for c in failed):
        return "reject", reasons
    if failed:
        return "review", reasons
    return "auto_approve", []
