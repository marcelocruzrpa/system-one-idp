"""Human review of the documents the code routed to review or reject: an append-only log of what the code routed and
what a person decided, and the queue folded from it.

A case is one document processed with one set of settings (path, mode, forced OCR). Processing it appends a `Routed`
event holding the full result, because a decision is about what the machine showed, not about a re-run. Deciding
appends a `Decided` event. Nothing is ever edited in place: the state of a case is the fold of its events.
"""

import hashlib
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from idp.config import OUT_DIR, ROOT
from idp.models import Mode, ProcessResult, Route
from idp.route import is_quality_flag

REVIEW_LOG = OUT_DIR / "reviews.jsonl"
Decision = Literal["approve", "reject"]
Verdict = Literal["confirmed", "false_alarm"]
"""A person's view of one flag: the problem is real, or the flag was raised needlessly."""
Status = Literal["pending", "approved", "rejected", "auto_approved"]

# Streamlit runs every browser session in its own thread of one process.
_LOCK = threading.RLock()


def rel_path(path: str | Path) -> str:
    """A file as the log stores it: relative to the repo root, so the log survives moving the repo; absolute when
    outside it. A relative path is read as relative to the repo root."""
    p = Path(path)
    p = (p if p.is_absolute() else ROOT / p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def case_id(path: str | Path, mode: str, force_ocr: bool) -> str:
    return hashlib.sha256(f"{rel_path(path)}|{mode}|{force_ocr}".encode()).hexdigest()[:12]


def fingerprint(result: ProcessResult) -> tuple[str, tuple[str, ...]]:
    """What a decision is about: the route and its flags. Details are left out, since the probabilities they quote
    drift between live runs."""
    return result.route, tuple(sorted(result.flags))


class Routed(BaseModel):
    kind: Literal["routed"] = "routed"
    case_id: str
    at: datetime
    title: str
    """The name the person knows the document by (an upload's original file name)."""
    path: str
    """See `rel_path`."""
    mode: Mode
    force_ocr: bool = False
    doc_id: str | None = None
    """The sample's id, for the ground-truth caption; None for uploads."""
    result: ProcessResult

    @property
    def file(self) -> Path:
        p = Path(self.path)
        return p if p.is_absolute() else ROOT / p


class Decided(BaseModel):
    kind: Literal["decided"] = "decided"
    case_id: str
    at: datetime
    decision: Decision
    reviewer: str
    note: str = ""
    verdicts: dict[str, Verdict] = {}
    """Flag id -> verdict. Every business flag has one; quality flags only when the person gave one."""
    machine_route: Route


Event = Annotated[Routed | Decided, Field(discriminator="kind")]
_EVENT: TypeAdapter[Routed | Decided] = TypeAdapter(Event)


@dataclass
class Case:
    routed: Routed
    """The latest routing: the result a decision applies to."""
    decisions: list[Decided] = field(default_factory=list)
    """Decisions made since `routed`, oldest first."""
    history: list[Routed | Decided] = field(default_factory=list)

    @property
    def case_id(self) -> str:
        return self.routed.case_id

    @property
    def route(self) -> Route:
        return self.routed.result.route

    @property
    def decision(self) -> Decided | None:
        return self.decisions[-1] if self.decisions else None

    @property
    def pending(self) -> bool:
        return self.route != "auto_approve" and self.decision is None

    @property
    def status(self) -> Status:
        if self.route == "auto_approve":
            return "auto_approved"  # re-processed after a fix and routed clean: nothing left for a person
        if self.decision is None:
            return "pending"
        return "approved" if self.decision.decision == "approve" else "rejected"

    @property
    def overrides(self) -> bool:
        """Approved although the code rejected it on a hard rule."""
        return self.route == "reject" and self.decision is not None and self.decision.decision == "approve"


def fold(events: list[Routed | Decided]) -> dict[str, Case]:
    cases: dict[str, Case] = {}
    for event in events:
        case = cases.get(event.case_id)
        if isinstance(event, Routed):
            if case is None:
                case = cases[event.case_id] = Case(routed=event)
            else:
                case.routed, case.decisions = event, []
        elif case is None:
            continue  # a decision whose routing is gone
        else:
            case.decisions.append(event)
        case.history.append(event)
    return cases


def decision_problems(result: ProcessResult, decision: Decision, verdicts: dict[str, Verdict], reviewer: str,
                      note: str) -> list[str]:
    """What stops a decision from being recorded: a reviewer, a verdict on every business flag, and a note whenever
    the decision goes against the evidence (approving a reject or a real issue, rejecting with no real issue)."""
    business = [f for f in dict.fromkeys(result.flags) if not is_quality_flag(f)]
    problems = []
    if not reviewer.strip():
        problems.append("A reviewer name is required.")
    if unmarked := [f for f in business if f not in verdicts]:
        problems.append(f"Mark every flag as a real issue or a false alarm: {', '.join(unmarked)}.")
    elif not note.strip():
        real = any(verdicts[f] == "confirmed" for f in business)
        if decision == "approve" and result.route == "reject":
            problems.append("Approving a document the code rejected needs a note saying why.")
        elif decision == "approve" and real:
            problems.append("Approving despite a real issue needs a note saying why.")
        elif decision == "reject" and not real:
            problems.append("Rejecting with no real issue marked needs a note saying why.")
    return problems


class ReviewLog:
    """`out/reviews.jsonl`: one event per line, appended, never edited (only `forget` and `clear` rewrite it)."""

    def __init__(self, path: Path = REVIEW_LOG) -> None:
        self.path = path

    def events(self) -> list[Routed | Decided]:
        """Every readable event, oldest first. A torn or hand-edited line is skipped, like an unreadable cache entry."""
        with _LOCK:
            if not self.path.exists():
                return []
            lines = self.path.read_text(encoding="utf-8").splitlines()
        events = []
        for line in lines:
            if line.strip():
                try:
                    events.append(_EVENT.validate_json(line))
                except ValidationError:
                    continue
        return events

    def cases(self) -> list[Case]:
        """Pending cases first, then the rest; the most recently updated first within each (by log position, which
        two events in the same clock tick cannot tie on)."""
        events = self.events()
        last = {e.case_id: i for i, e in enumerate(events)}
        return sorted(fold(events).values(), key=lambda c: (not c.pending, -last[c.case_id]))

    def case(self, case_id: str) -> Case | None:
        return fold(self.events()).get(case_id)

    def record(self, result: ProcessResult, *, title: str, force_ocr: bool = False,
               doc_id: str | None = None) -> Case | None:
        """Log what the code routed. Idempotent: nothing is appended when the case's latest routing has the same route
        and flags, so re-processing a document keeps its decision. A changed result re-opens the case (the old
        decision stays in its history), and an `auto_approve` closes it; an `auto_approve` opens no case."""
        cid = case_id(result.path, result.mode, force_ocr)
        with _LOCK:
            case = self.case(cid)
            if case is None and result.route == "auto_approve":
                return None
            if case is not None and fingerprint(case.routed.result) == fingerprint(result):
                return case
            self._append(Routed(case_id=cid, at=datetime.now().astimezone(), title=title, path=rel_path(result.path),
                                mode=result.mode, force_ocr=force_ocr, doc_id=doc_id, result=result))
            return self.case(cid)

    def decide(self, case_id: str, *, decision: Decision, reviewer: str, note: str = "",
               verdicts: dict[str, Verdict]) -> Decided:
        with _LOCK:
            case = self.case(case_id)
            if case is None:
                raise KeyError(f"No review case {case_id}")
            event = Decided(case_id=case_id, at=datetime.now().astimezone(), decision=decision,
                            reviewer=reviewer.strip(), note=note.strip(), verdicts=verdicts, machine_route=case.route)
            self._append(event)
        return event

    def forget(self, under: Path) -> int:
        """Drop every case whose file lives under `under`, with its decisions (deleted uploads take their extracted
        values and text with them). Returns the number of cases dropped."""
        under = under.resolve()
        with _LOCK:
            events = self.events()
            gone = {e.case_id for e in events if isinstance(e, Routed) and e.file.resolve().is_relative_to(under)}
            if gone:
                self._rewrite([e for e in events if e.case_id not in gone])
        return len(gone)

    def clear(self) -> None:
        with _LOCK:
            self.path.unlink(missing_ok=True)

    def _append(self, event: Routed | Decided) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")

    def _rewrite(self, events: list[Routed | Decided]) -> None:
        """Atomically, as `ResponseCache.put` writes: a reader never sees half a log."""
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=f"{self.path.stem}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.writelines(e.model_dump_json() + "\n" for e in events)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
