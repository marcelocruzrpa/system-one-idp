"""Streamlit demo: pick a sample or upload a document, process it, inspect every decision."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from idp import erp_view, ingest
from idp.config import OUT_DIR, SAMPLES_DIR, schema as load_schema
from idp.evaluate import business_flags, load_manifest
from idp.models import Document, ProcessResult
from idp.pipeline import Pipeline, pick_mode
from idp.review import Case, ReviewLog, case_id, decision_problems
from ui_helpers import (VERDICTS, check_label, erp_url, extraction_rows, history_line, line_item_rows, lines_table,
                        po_number_from_detail, queue_label, replayed_from_cache, review_summary, split_reasons,
                        truth_caption)

st.set_page_config(page_title="IDP POC - Jev + Clairic", layout="wide")

MODES = {"auto": "Auto (from the document)", "text": "Text (OCR / PDF text layer)", "image": "Image (page images only)",
         "hybrid": "Hybrid (text + images)"}
STAGE_LABELS = {"classify": "Input and classification", "extract": "Extraction quality", "rules": "Rules (code)",
                "semantic": "Semantic checks (Jev)", "match": "Three-way match (Jev + ERP)"}
STEPS = [
    ("1 · Classify", "Jev", "One Choice names the document type; yes/no signals (paid? bank change? several documents?) "
                            "ride along in the same request."),
    ("2 · Extract", "Clairic", "Typed fields from that type's schema: value, confidence, p_null and a span into the text. "
                               "Line items are rebuilt from the page's table rows."),
    ("3 · Validate", "code + Jev", "Arithmetic, dates and ERP lookups in code; Jev verifies values and maps invoice lines "
                                   "to PO lines. Code routes: reject, review or auto-approve."),
    ("4 · Review", "human", "Review and reject routes wait for a person: approve or reject, mark each flag real or a "
                            "false alarm, add a note. Every decision is logged."),
]
QUEUE = "Review queue"
ROUTE_ASKS = {
    "review": "The code found something a person should look at: approve or reject it, and mark each flag.",
    "reject": "The code rejected this on a hard rule (`reject_flags` in config/policy.yaml): confirm the reject, or "
              "approve it with a note.",
}
MAX_REASONS = 3
UPLOAD_DIR = OUT_DIR / "uploads"
ERP_PAGE = Path(__file__).parent / "static" / "erp.html"  # served at erp_url() (server.enableStaticServing)
UPLOAD_SUFFIXES = {".pdf", *ingest.IMAGE_SUFFIXES}


@st.cache_resource
def get_pipeline(use_cache: bool) -> Pipeline:
    return Pipeline(use_cache=use_cache)


@st.cache_resource
def render_erp_page() -> Path:
    """Once per server process, like the cached Pipeline: the page shows the ERP data the match ran on."""
    return erp_view.write(ERP_PAGE)


@st.cache_resource
def review_log() -> ReviewLog:
    return ReviewLog()


@st.cache_data(show_spinner="Reading document (OCR if needed)...")
def load_doc(path: str, force_ocr: bool) -> Document:
    return ingest.load(path, force_ocr=force_ocr)


def load_truth(doc_id: str) -> dict | None:
    path = SAMPLES_DIR / "ground_truth" / f"{doc_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_upload(name: str, data: bytes) -> Path:
    """Store under a name derived from the content: the client's file name never becomes a path, and a new
    file uploaded under an old name never hits the cached document or result of the old one."""
    suffix = Path(name).suffix.lower()
    if suffix not in UPLOAD_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix or name!r}")
    path = UPLOAD_DIR / f"{hashlib.sha256(data).hexdigest()[:16]}{suffix}"
    if not path.exists():
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path


def delete_uploads() -> int:
    """Remove every uploaded file and forget the documents, results and review cases kept for them."""
    deleted = 0
    if UPLOAD_DIR.exists():
        for file in UPLOAD_DIR.iterdir():
            if file.is_file():
                file.unlink()
                deleted += 1
    load_doc.clear()
    for key in [k for k in st.session_state if str(k).startswith(str(UPLOAD_DIR))]:
        del st.session_state[key]
    review_log().forget(UPLOAD_DIR)
    return deleted


def review_section(log: ReviewLog, case: Case, result: ProcessResult) -> None:
    """4 - Human review: the decision once someone made it, else (or after "Change decision") the form."""
    st.subheader("4 - Human review")
    st.caption(ROUTE_ASKS[result.route])
    redo = f"redo|{case.case_id}"
    d = case.decision
    if d and not st.session_state.get(redo):
        verb = "Approved" if d.decision == "approve" else "Rejected"
        (st.success if d.decision == "approve" else st.error)(
            f"**{verb} by {d.reviewer}** on {d.at:%Y-%m-%d %H:%M}" + (f": {d.note}" if d.note else "")
            + "".join(f"\n- {check_label(cid)}: {VERDICTS[v]} :gray-badge[{cid}]" for cid, v in d.verdicts.items()))
        if case.overrides:
            st.badge("Overrides the code's reject", icon=":material/gavel:", color="orange")
        st.button("Change decision", icon=":material/edit:", on_click=st.session_state.__setitem__, args=(redo, True))
    else:
        decision_form(log, case, result)
        if d:
            st.button("Keep the decision", on_click=st.session_state.pop, args=(redo, None))
    with st.expander(f"History ({len(case.history)})"):
        st.markdown("".join(f"\n- {history_line(event)}" for event in case.history))


def decision_form(log: ReviewLog, case: Case, result: ProcessResult) -> None:
    """One Real issue / False alarm control per flag (business flags required, quality notes optional), a note, the
    reviewer and Approve / Reject. `decision_problems` decides whether the decision is complete."""
    business, quality = (dict(pairs) for pairs in split_reasons(result.reasons))
    previous = case.decision.verdicts if case.decision else {}
    verdicts: dict[str, str | None] = {}
    with st.form(f"review|{case.case_id}"):
        for flags, optional in ((business, False), (quality, True)):
            if flags and optional:
                st.caption("Extraction quality notes (optional)")
            for cid, detail in flags.items():
                verdicts[cid] = st.segmented_control(
                    f"**{check_label(cid)}** - {detail} :gray-badge[{cid}]", list(VERDICTS), format_func=VERDICTS.get,
                    default=previous.get(cid), key=f"verdict|{case.case_id}|{cid}")
        note = st.text_area("Note", value=case.decision.note if case.decision else "", key=f"note|{case.case_id}",
                            height=80, placeholder="Why: required to approve a reject or a real issue, or to reject "
                                                   "with no real issue")
        who, approve_col, reject_col = st.columns([2, 1, 1], vertical_alignment="bottom")
        reviewer = who.text_input("Reviewer", key="reviewer", placeholder="Your name", persist_state="session")
        approve = approve_col.form_submit_button("Approve", icon=":material/check:", width="stretch")
        reject = reject_col.form_submit_button("Reject", icon=":material/block:", width="stretch",
                                               type="primary" if result.route == "reject" else "secondary")
    if not (approve or reject):
        return
    decision = "approve" if approve else "reject"
    chosen = {cid: v for cid, v in verdicts.items() if v}
    if problems := decision_problems(result, decision, chosen, reviewer or "", note or ""):
        st.error("\n".join(f"- {p}" for p in problems))
        return
    log.decide(case.case_id, decision=decision, reviewer=reviewer, note=note or "", verdicts=chosen)
    st.session_state.pop(f"redo|{case.case_id}", None)
    st.session_state["flash"] = (f"{'Approved' if approve else 'Rejected'} {case.routed.title}",
                                 ":material/check:" if approve else ":material/block:")
    st.rerun()


if flash := st.session_state.pop("flash", None):  # set just before the st.rerun() that follows a decision
    st.toast(flash[0], icon=flash[1])

# ---------------------------------------------------------------------------------------------- headline
st.title("Intelligent Document Processing with System One Models")
st.markdown("Jev judges, Clairic extracts, code decides.")
for col, (step, who, what) in zip(st.columns(len(STEPS)), STEPS):
    col.markdown(f"**{step}** :gray[{who}]  \n:small[{what}]")

# ---------------------------------------------------------------------------------------------- sidebar
log = review_log()
cases = log.cases()
case: Case | None = None  # the review case on screen: picked in the queue, or the one the processed document opened
with st.sidebar:
    st.header("Document")
    truth = None
    doc_id = None
    source = st.radio("Input", ["Sample", "Upload", QUEUE], horizontal=True)
    if waiting := sum(c.pending for c in cases):
        st.caption(f":orange[{waiting} document{'s' if waiting > 1 else ''} waiting for a human in the review queue]")
    if source == QUEUE:
        show_decided = st.toggle("Show decided", help="Also list documents a person already approved or rejected")
        shown = {c.case_id: c for c in cases if c.pending or show_decided}
        if shown:
            case = shown[st.selectbox("Document", list(shown), format_func=lambda cid: queue_label(shown[cid]))]
        else:
            st.info("Nothing waiting for a human: documents routed to review or reject land here after Process.")
        if cases:
            st.caption(review_summary(cases))
        with st.expander("Clear the review log"):
            st.caption("Deletes `out/reviews.jsonl`: every decision and the results it was made on.")
            confirmed = st.checkbox("Yes, clear it")
            if st.button("Clear the review log", disabled=not confirmed, width="stretch"):
                log.clear()
                st.success("Review log cleared.")
                st.stop()
        if case is None:
            st.stop()
        path, title, doc_id = case.routed.file, case.routed.title, case.routed.doc_id
        if not path.exists():
            st.error(f"{title} is no longer on disk ({case.routed.path}).")
            st.stop()
        truth = load_truth(doc_id) if doc_id else None
    elif source == "Sample":
        if not (SAMPLES_DIR / "manifest.json").exists():
            st.error("No samples yet. Run `uv run idp generate`.")
            st.stop()
        manifest = load_manifest()
        variant = st.radio("Variant", ["digital", "scanned"], horizontal=True,
                           help="digital = PDF with a text layer; scanned = image only, goes through OCR")
        entry = st.selectbox("Sample", manifest, format_func=lambda m: f"{m['doc_id']} - {m['note']}")
        path = SAMPLES_DIR / entry[variant]
        title = path.name
        doc_id = entry["doc_id"]
        truth = load_truth(doc_id)
    else:
        upload = st.file_uploader("PDF, PNG or JPEG", type=["pdf", "png", "jpg", "jpeg"])
        with st.expander("Delete my uploads"):
            st.caption("Removes every file under `out/uploads` and forgets the documents and results cached for them.")
            confirmed = st.checkbox("Yes, delete them")
            if st.button("Delete my uploads", disabled=not confirmed, width="stretch"):
                st.success(f"Deleted {delete_uploads()} file(s). A file still selected in the uploader is stored again "
                           "on the next run; remove it there to keep it off disk.")
                st.stop()
        if upload is None:
            st.info("Upload a document to process it.")
            st.stop()
        try:
            path = save_upload(upload.name, upload.getvalue())
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        title = Path(upload.name).name
    if case is None:
        mode = st.radio("Model input", list(MODES), format_func=MODES.get,
                        help="Auto: text when every page has a text layer, hybrid when any page needed OCR")
        mode_note = st.empty()
        force_ocr = st.checkbox("Force OCR", help="OCR every PDF page even if it has a text layer")
        use_cache = st.checkbox("Use response cache", value=True, help="Re-use earlier model answers for identical requests")
        run = st.button("Process", type="primary", width="stretch")
    else:
        # The queue shows the result the decision is about; processing again happens under Sample or Upload.
        mode, force_ocr, run = case.routed.mode, case.routed.force_ocr, False
        st.caption(f"Processed in **{mode}** mode{' with forced OCR' if force_ocr else ''} on "
                   f"{case.routed.at:%Y-%m-%d %H:%M}. To process it again, pick it under Sample or Upload.")

try:
    doc = load_doc(str(path), force_ocr)
    if mode == "auto":
        mode, why = pick_mode(doc)
        mode_note.caption(f"Auto picked **{MODES[mode]}**: {why}")
except Exception as exc:  # noqa: BLE001 - an unreadable or corrupt file must not take the app down
    st.error(f"Could not read {title}: {exc}")
    st.stop()
key = f"{path}|{mode}|{force_ocr}"
if run:
    try:
        with st.spinner("Classifying (Jev) -> extracting (Clairic) -> validating..."):
            st.session_state[key] = get_pipeline(use_cache).process(path, mode, doc=doc)
    except Exception as exc:  # noqa: BLE001 - surface missing keys / API errors in the UI
        st.error(f"{type(exc).__name__}: {exc}")
result: ProcessResult | None = case.routed.result if case is not None else st.session_state.get(key)
if result is not None and source != QUEUE:
    # A fresh Process logs what the code routed (idempotent); a result already on screen finds its case, or opens one
    # when the log was cleared meanwhile.
    case = None if run else log.case(case_id(path, mode, force_ocr))
    case = case or log.record(result, title=title, force_ocr=force_ocr, doc_id=doc_id)

# ---------------------------------------------------------------------------------------------- layout
left, right = st.columns([2, 3], gap="large")
with left:
    st.subheader(title, anchor=False)
    for page in doc.pages:
        st.image(page.png, caption=f"Page {page.number} - text from {page.text_source.replace('_', ' ')}",
                 width="stretch")
    with st.expander("Text the models read"):
        st.code(doc.text, language=None)

with right:
    if result is None:
        st.info("Press **Process** to run the pipeline.")
        st.stop()

    # Route banner: business reasons only; quality notes (what the models could not read or verify) go below.
    business, quality = split_reasons(result.reasons)
    banner = {"auto_approve": st.success, "review": st.warning, "reject": st.error}[result.route]
    bullets = [f"- **{check_label(cid)}** - {detail} :gray-badge[{cid}]" for cid, detail in business[:MAX_REASONS]]
    if len(business) > MAX_REASONS:
        bullets.append(f"- and {len(business) - MAX_REASONS} more")
    headline = f"**Route: {result.route.replace('_', ' ')}**"
    if not business and quality:
        headline += " - no business flag; the extraction quality notes below need a look"
    banner(headline + "".join(f"\n{b}" for b in bullets))
    needs_human = case is not None and result.route != "auto_approve"
    if needs_human and (d := case.decision):
        st.badge(f"{'Approved' if d.decision == 'approve' else 'Rejected'} by {d.reviewer}", icon=":material/person:",
                 color="green" if d.decision == "approve" else "red")
    elif needs_human:
        st.badge("Waiting for a human decision: see 4 - Human review", icon=":material/person:", color="orange")
    if quality:
        with st.expander(f"Extraction quality notes ({len(quality)})"):
            for cid, detail in quality:
                st.markdown(f"- **{check_label(cid)}** - {detail} :gray-badge[{cid}]")
    if truth:
        caption, ok = truth_caption(result.route, business_flags(result), truth)
        st.caption(f":green[{caption}]" if ok else f":orange[{caption}]")

    c = result.classification
    st.subheader("1 - Classification (Jev Choice)")
    m1, m2 = st.columns(2)
    m1.metric("Document type", c.doc_type, help="Top option of the Jev Choice")
    m2.metric("Confidence", f"{c.confidence:.2f}")
    probs = pd.DataFrame(sorted(c.probabilities.items(), key=lambda kv: -kv[1])[:5], columns=["type", "probability"])
    st.bar_chart(probs, x="type", y="probability", horizontal=True, height=180)
    st.caption("Speculative signals asked in the same request: "
               + ", ".join(f"`{k}` {v:.2f}" for k, v in c.signals.items()))

    if result.extraction:
        st.subheader("2 - Extraction (Clairic)")
        schema = load_schema(c.doc_type) or {"fields": {}}
        rows = extraction_rows(result.extraction, schema, truth if truth and truth["doc_type"] == c.doc_type else None)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"confidence": st.column_config.ProgressColumn(
                                        min_value=0, max_value=1, format="%.2f",
                                        help="Scalar fields only: a list's joint confidence is not comparable"),
                                    "p_null": st.column_config.NumberColumn(format="%.2f"),
                                    "note": st.column_config.TextColumn(width="small")})
        if result.extraction.truncated:
            st.warning("Clairic reported the input as truncated.")
        if result.line_items:
            st.markdown("**Line items**")
            items = line_item_rows(result.line_items)
            if items:
                st.dataframe(pd.DataFrame(items), hide_index=True, width="stretch")
            st.caption(result.line_items.get("detail", ""))
            if result.line_items.get("dropped"):
                st.caption("Clairic text entries on no table row (ignored): "
                           + ", ".join(repr(d) for d in result.line_items["dropped"]))
    elif "unsupported_doc_type" in result.flags:
        st.info("No extraction: documents classified as `other` have no schema and go to review.")

    st.subheader("3 - Validation")
    for stage, label in STAGE_LABELS.items():
        stage_checks = [ch for ch in result.checks if ch.stage == stage]
        if not stage_checks:
            continue
        failed = sum(not ch.passed for ch in stage_checks)
        with st.expander(f"{label} - {len(stage_checks) - failed} passed, {failed} flagged", expanded=failed > 0):
            for ch in stage_checks:
                icon = ":white_check_mark:" if ch.passed else ":red_circle:"
                prob = ""
                if ch.probability is not None and not any(s in ch.detail for s in ("P(", "P=", "onfidence")):
                    prob = f" P(problem) = {ch.probability:.2f}."
                st.markdown(f"{icon} **{check_label(ch.id, ch.passed)}** - {ch.detail}{prob} :gray-badge[{ch.id}]")
                if not ch.passed and isinstance(ch.data.get("lines"), list):
                    table = lines_table(ch.id, ch.data["lines"])
                    if table:
                        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch",
                                     column_config={"delta %": st.column_config.NumberColumn(format="%.1f%%"),
                                                    "confidence": st.column_config.NumberColumn(format="%.2f")})
                if ch.id == "po_not_found" and ch.passed:
                    po = po_number_from_detail(ch.detail)
                    if po:
                        render_erp_page()
                        st.link_button(f"Open {po} in the mock ERP", erp_url(po), icon=":material/open_in_new:")

    if needs_human:
        review_section(log, case, result)

    st.subheader("Cost and latency")
    u, cost = result.usage, result.cost
    cached = replayed_from_cache(result.stage_usage)
    model_time = sum(v for k, v in result.timings.items() if k != "ingest")
    k0, k1, k2, k3, k4, k5 = st.columns(6)
    k0.metric("Cost", f"${cost.total:.6f}", help="Input tokens at `pricing_per_million_input` in config/policy.yaml; "
              "OCR runs locally and is not included")
    k1.metric("Per 1,000 documents", f"${1000 * cost.total:.2f}", help="1,000 documents like this one, every request live")
    k2.metric("Jev input tokens", f"{u.jev_input_tokens:,}", help=f"{u.jev_requests} requests, ${cost.jev:.6f}")
    k3.metric("Clairic input tokens", f"{u.clairic_input_tokens:,}", help=f"{u.clairic_requests} requests, ${cost.clairic:.6f}")
    k4.metric("Model time", "replayed" if cached else f"{model_time:.1f} s",
              help="Wall time of the model stages (classify, extract, validate), excluding ingest / OCR")
    k5.metric("Pages", result.page_count)
    if cached:
        st.badge("Replayed from the response cache (model time not representative)", icon=":material/history:", color="gray")
    st.dataframe(pd.DataFrame([{"stage": stage, "model": "Clairic" if s.clairic_requests else "Jev",
                                "input tokens": s.jev_input_tokens + s.clairic_input_tokens,
                                "cost (USD)": s.cost().total, "from cache": bool(s.jev_cached_tokens + s.clairic_cached_tokens)}
                               for stage, s in result.stage_usage.items() if s.jev_requests + s.clairic_requests]),
                 hide_index=True, width="stretch", column_config={"cost (USD)": st.column_config.NumberColumn(format="$%.6f")})
    models = ", ".join(f"{name} ({who.capitalize()})" for who, name in result.models.items()) or "not reported"
    st.caption(f"Models: {models} | Billed this run: ${max(0.0, cost.billed):.6f} (cached responses cost nothing) | "
               + " | ".join(f"{k}: {v:.2f}s" for k, v in result.timings.items()))
    if result.errors:
        st.warning("Model calls that failed (their stage was sent to review instead):"
                   + "".join(f"\n- {e}" for e in result.errors))
