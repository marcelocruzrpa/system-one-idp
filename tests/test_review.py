"""The review log behind the app's human-in-the-loop step: cases, decisions and the rules a decision must meet."""

from idp.config import ROOT
from idp.review import ReviewLog, case_id, decision_problems
from tests.conftest import make_result


def test_a_review_result_opens_one_pending_case(tmp_path):
    log = ReviewLog(tmp_path / "reviews.jsonl")
    result = make_result("price_mismatch")
    case = log.record(result, title="invoice-03.pdf", doc_id="invoice-03")
    assert case.pending and case.status == "pending" and case.route == "review" and case.decision is None
    assert case.case_id == case_id(ROOT / "samples" / "digital" / "invoice-03.pdf", "text", False)
    assert case.routed.path == "samples/digital/invoice-03.pdf" and case.routed.doc_id == "invoice-03"

    assert log.record(make_result("price_mismatch"), title="invoice-03.pdf").case_id == case.case_id
    assert len(log.events()) == 1, "processing the same document again appends nothing"
    assert log.record(make_result("price_mismatch"), title="invoice-03.pdf", force_ocr=True).case_id != case.case_id

    assert log.record(make_result(path="samples/digital/invoice-01.pdf"), title="invoice-01.pdf") is None
    assert len(log.cases()) == 2


def test_the_result_survives_the_round_trip(tmp_path):
    log = ReviewLog(tmp_path / "reviews.jsonl")
    result = make_result("price_mismatch", "missing:po_number")
    log.record(result, title="invoice-03.pdf")
    loaded = ReviewLog(log.path).cases()[0].routed.result
    assert loaded.model_dump() == result.model_dump()
    assert loaded.extraction.value("total") == 1329.56 and loaded.reasons[0].startswith("price_mismatch: ")


def test_a_decision_closes_the_case_and_a_changed_result_reopens_it(tmp_path):
    log = ReviewLog(tmp_path / "reviews.jsonl")
    cid = log.record(make_result("price_mismatch"), title="invoice-03.pdf").case_id
    log.decide(cid, decision="reject", reviewer=" Ana ", verdicts={"price_mismatch": "confirmed"})
    case = log.case(cid)
    assert case.status == "rejected" and not case.pending and not case.overrides
    assert case.decision.reviewer == "Ana" and case.decision.machine_route == "review"

    assert log.record(make_result("price_mismatch"), title="invoice-03.pdf").status == "rejected"
    reopened = log.record(make_result("price_mismatch", "total_mismatch"), title="invoice-03.pdf")
    assert reopened.pending and reopened.decision is None
    assert [e.kind for e in reopened.history] == ["routed", "decided", "routed"]

    closed = log.record(make_result(), title="invoice-03.pdf")
    assert closed.status == "auto_approved" and not closed.pending


def test_approving_a_reject_overrides_the_code(tmp_path):
    log = ReviewLog(tmp_path / "reviews.jsonl")
    cid = log.record(make_result("duplicate_invoice"), title="invoice-06.pdf").case_id
    assert log.case(cid).route == "reject"
    log.decide(cid, decision="approve", reviewer="Ana", note="Credit note issued",
               verdicts={"duplicate_invoice": "false_alarm"})
    assert log.case(cid).overrides and log.case(cid).status == "approved"
    log.decide(cid, decision="reject", reviewer="Ana", verdicts={"duplicate_invoice": "confirmed"})
    assert not log.case(cid).overrides and len(log.case(cid).decisions) == 2


def test_cases_list_pending_first(tmp_path):
    log = ReviewLog(tmp_path / "reviews.jsonl")
    first = log.record(make_result("price_mismatch"), title="a.pdf").case_id
    second = log.record(make_result("total_mismatch", path="samples/digital/invoice-04.pdf"), title="b.pdf").case_id
    assert [c.case_id for c in log.cases()] == [second, first]
    log.decide(second, decision="reject", reviewer="Ana", verdicts={"total_mismatch": "confirmed"})
    assert [c.case_id for c in log.cases()] == [first, second]


def test_forget_drops_the_cases_under_a_directory_and_corrupt_lines_are_skipped(tmp_path):
    log = ReviewLog(tmp_path / "reviews.jsonl")
    upload = log.record(make_result("price_mismatch", path="out/uploads/0123abcd.pdf"), title="mine.pdf").case_id
    log.record(make_result("price_mismatch"), title="invoice-03.pdf")
    log.decide(upload, decision="reject", reviewer="Ana", verdicts={"price_mismatch": "confirmed"})
    with log.path.open("a", encoding="utf-8") as f:
        f.write('{"kind": "routed", "case_id": "torn\n')
    assert len(log.cases()) == 2

    assert log.forget(ROOT / "out" / "uploads") == 1
    assert [c.routed.title for c in log.cases()] == ["invoice-03.pdf"]
    assert all(e.case_id != upload for e in log.events())
    assert log.forget(ROOT / "out" / "uploads") == 0

    log.clear()
    assert log.cases() == [] and not log.path.exists()
    log.clear()


def test_decision_problems_ask_for_a_reviewer_every_verdict_and_a_note_against_the_evidence():
    review = make_result("price_mismatch", "missing:po_number")
    assert decision_problems(review, "reject", {"price_mismatch": "confirmed"}, "Ana", "") == []
    assert decision_problems(review, "approve", {"price_mismatch": "false_alarm"}, "Ana", "") == []

    problems = decision_problems(review, "approve", {}, " ", "")
    assert len(problems) == 2 and "reviewer" in problems[0]
    assert "price_mismatch" in problems[1] and "missing:po_number" not in problems[1], "quality verdicts are optional"

    assert "note" in decision_problems(review, "approve", {"price_mismatch": "confirmed"}, "Ana", "")[0]
    assert decision_problems(review, "approve", {"price_mismatch": "confirmed"}, "Ana", "Price increase agreed") == []
    assert "note" in decision_problems(review, "reject", {"price_mismatch": "false_alarm"}, "Ana", " ")[0]

    reject = make_result("duplicate_invoice")
    assert "note" in decision_problems(reject, "approve", {"duplicate_invoice": "false_alarm"}, "Ana", "")[0]
    assert decision_problems(reject, "reject", {"duplicate_invoice": "confirmed"}, "Ana", "") == []

    quality_only = make_result("unverified:balance")
    assert quality_only.route == "review"
    assert decision_problems(quality_only, "approve", {}, "Ana", "") == []
    assert "note" in decision_problems(quality_only, "reject", {}, "Ana", "")[0]
