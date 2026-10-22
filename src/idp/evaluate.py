"""Score pipeline results against sample ground truth: classification, fields, flags, routing, cost, latency."""

import json
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from idp import ingest
from idp.config import OUT_DIR, SAMPLES_DIR, schema as load_schema
from idp.erp import norm_ref
from idp.models import ModeChoice, ProcessResult
from idp.pipeline import Pipeline
from idp.route import QUALITY_IDS, QUALITY_PREFIXES, is_quality_flag
from idp.validate.rules import as_number

__all__ = ["NoMatchingSamples", "QUALITY_IDS", "QUALITY_PREFIXES", "align_items", "business_flags", "evaluate_doc",
           "load_manifest", "run_eval", "score_field", "scorecard", "select_entries"]
CONFIDENCE_BUCKETS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
# Fields scored by exact reference match: a near-miss invoice number or currency is simply wrong.
IDENTIFIER_SUFFIXES = ("_number", "_numbers", "_id", "_reference")
# Free text counts when one value contains the other on word boundaries and covers this share of its words.
MIN_TEXT_COVERAGE = 0.5
# A null answer with p_null above this is the model declining to read ("not on the page"), not a misread.
ABSTAIN_P_NULL = 0.5


class NoMatchingSamples(ValueError):
    """`only` named nothing in the manifest; the message lists what it could have named."""


def business_flags(result: ProcessResult) -> set[str]:
    return {f for f in result.flags if not is_quality_flag(f)}


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", str(value).casefold())).strip()


def is_identifier(name: str | None) -> bool:
    return bool(name) and (name == "currency" or name.endswith(IDENTIFIER_SUFFIXES))


def match_scalar(kind: str, truth: Any, pred: Any, *, identifier: bool = False) -> bool:
    if truth is None:
        return pred is None or pred == ""
    if pred is None:
        return False
    if kind == "number":
        p = as_number(pred)
        return p is not None and abs(p - float(truth)) <= 0.011
    if kind == "date":
        return str(pred)[:10] == str(truth)[:10]
    if kind == "time":
        return str(pred)[:5] == str(truth)[:5]
    if identifier:
        return norm_ref(truth) == norm_ref(pred)
    t, p = norm_text(truth), norm_text(pred)
    if t == p:
        return True
    # Lenient for names and addresses that add or drop a word or two ("Acme Robotics Inc." vs "Acme Robotics Inc. AP").
    short, long = sorted((t, p), key=len)
    return bool(short) and f" {short} " in f" {long} " and len(short.split()) >= MIN_TEXT_COVERAGE * len(long.split())


def score_field(spec: dict, truth: Any, pred: Any, name: str | None = None) -> float:
    """1.0 / 0.0 for scalars; fraction of positions matched for lists. Identifier fields must match exactly."""
    identifier = is_identifier(name)
    if spec["type"] != "list":
        return float(match_scalar(spec["type"], truth, pred, identifier=identifier))
    truth = truth or []
    pred = pred if isinstance(pred, list) else []
    if not truth and not pred:
        return 1.0
    hits = sum(match_scalar(spec.get("items", "text"), t, p, identifier=identifier) for t, p in zip(truth, pred))
    return hits / max(len(truth), len(pred))


def align_items(kind: str, truth: list, pred: list, identifier: bool = False) -> dict[str, int]:
    """Order-preserving alignment of predicted list items to the truth's: a longest common subsequence with
    `match_scalar` as the equality test. Returns `aligned`, `extra` (predicted items left over) and `missing` (truth
    items never matched). Unlike `score_field`, a dropped row or a stray header does not fail every item after it."""
    n, m = len(truth), len(pred)
    lcs = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if match_scalar(kind, truth[i], pred[j], identifier=identifier):
                lcs[i][j] = lcs[i + 1][j + 1] + 1
            else:
                lcs[i][j] = max(lcs[i + 1][j], lcs[i][j + 1])
    aligned = lcs[0][0]
    return {"aligned": aligned, "extra": m - aligned, "missing": n - aligned}


def evaluate_doc(truth: dict, result: ProcessResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "doc_id": truth["doc_id"], "doc_type": truth["doc_type"], "predicted_type": result.classification.doc_type,
        "type_confidence": result.classification.confidence,
        "type_correct": result.classification.doc_type == truth["doc_type"],
        "expected_route": truth["expected_route"], "route": result.route,
        "expected_flags": sorted(truth["expected_flags"]), "flags": sorted(business_flags(result)),
        "quality_flags": sorted(set(result.flags) - business_flags(result)),
        "usage": result.usage.model_dump(), "cost": result.cost.model_dump(),
        "stage_usage": {stage: u.model_dump() for stage, u in result.stage_usage.items()},
        "stage_cost": {stage: u.cost().total for stage, u in result.stage_usage.items()},
        "ingest_seconds": result.timings.get("ingest", 0.0),
        "pipeline_seconds": sum(v for k, v in result.timings.items() if k != "ingest"),
        "seconds": sum(result.timings.values()), "fields": {},
    }
    ran = {c.id for c in result.checks}
    failed = set(result.flags)
    schema = load_schema(truth["doc_type"]) or {"fields": {}}
    for name, spec in schema["fields"].items():
        answer = result.extraction.fields.get(name) if result.extraction and row["type_correct"] else None
        pred = answer.value if answer else None
        expected = truth["fields"].get(name)
        semantic = (f"unsupported:{name}", f"wrong_role:{name}")
        is_list = spec["type"] == "list"
        entry: dict[str, Any] = {
            "truth": expected, "pred": pred,
            "score": score_field(spec, expected, pred, name),
            "list": is_list,
            "confidence": answer.confidence if answer else None,
            # A confident null on a field the page does have is the model declining, scored 0 but counted apart.
            "abstained": pred is None and expected is not None and answer is not None
            and (answer.p_null or 0.0) > ABSTAIN_P_NULL,
            # Only fields the Jev verifier actually checked count towards its metrics.
            "verified": any(f in ran for f in semantic),
            "semantic_flagged": any(f in failed for f in semantic),
            "rule_flagged": any(f in failed for f in (f"missing:{name}", f"low_confidence:{name}")),
        }
        if is_list:
            entry.update(align_items(spec.get("items", "text"), expected or [], pred if isinstance(pred, list) else [],
                                     identifier=is_identifier(name)))
        row["fields"][name] = entry
    return row


def _mean(values: list[float], digits: int | None = None) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return round(mean, digits) if digits is not None else mean


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def _percentile(values: list[float], q: float) -> float | None:
    """Linearly interpolated percentile (`q` in 0..1); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo), 3)


def _empty_buckets() -> dict[str, list[int]]:
    return {f"{lo:.1f}-{min(hi, 1):.1f}": [0, 0] for lo, hi in CONFIDENCE_BUCKETS}


def _bucket(buckets: dict[str, list[int]], confidence: float, correct: bool) -> None:
    for lo, hi in CONFIDENCE_BUCKETS:
        if lo <= confidence < hi:
            bucket = buckets[f"{lo:.1f}-{min(hi, 1):.1f}"]
            bucket[0] += 1
            bucket[1] += correct


def _calibration(buckets: dict[str, list[int]]) -> dict[str, dict[str, Any]]:
    return {k: {"n": v[0], "accuracy": round(v[1] / v[0], 3) if v[0] else None} for k, v in buckets.items()}


def scorecard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics over `evaluate_doc` rows.

    Rows carrying `error` (documents the pipeline failed on) are listed under `errors` / `documents_failed` and
    excluded from every metric; `documents` counts the scored rows. With nothing to score the metrics are None or 0.
    """
    failed = [r for r in rows if r.get("error")]
    scored = [r for r in rows if not r.get("error")]
    n = len(scored)
    confusion = Counter((r["doc_type"], r["predicted_type"]) for r in scored)
    field_scores: dict[str, list[float]] = defaultdict(list)
    scalar_scores: list[float] = []
    list_scores: list[float] = []
    scalar_buckets, list_buckets = _empty_buckets(), _empty_buckets()
    verifier, extraction_flags = Counter(), Counter()
    items = Counter()
    unverified_fields = abstained = 0
    tp = fp = fn = 0
    per_flag: dict[str, Counter] = defaultdict(Counter)
    models: dict[str, set[str]] = defaultdict(set)
    live_requests = cached_requests = 0
    for r in scored:
        expected, got = set(r["expected_flags"]), set(r["flags"])
        tp, fp, fn = tp + len(expected & got), fp + len(got - expected), fn + len(expected - got)
        for f in expected & got:
            per_flag[f]["tp"] += 1
        for f in got - expected:
            per_flag[f]["fp"] += 1
        for f in expected - got:
            per_flag[f]["fn"] += 1
        for model, version in (r.get("result") or {}).get("models", {}).items():
            models[model].add(version)
        for u in r.get("stage_usage", {}).values():
            for model in ("jev", "clairic"):
                requests, tokens, cached = u[f"{model}_requests"], u[f"{model}_input_tokens"], u[f"{model}_cached_tokens"]
                if not requests:
                    continue
                # A stage's requests are all live or all cached in practice; a mixed stage is split by token share.
                hits = requests if cached >= tokens > 0 else round(requests * cached / tokens) if cached and tokens else 0
                cached_requests += hits
                live_requests += requests - hits
        for name, f in r["fields"].items():
            field_scores[f"{r['doc_type']}.{name}"].append(f["score"])
            correct = f["score"] == 1.0
            is_list = f.get("list", isinstance(f["truth"], list) or isinstance(f["pred"], list))
            (list_scores if is_list else scalar_scores).append(f["score"])
            abstained += bool(f.get("abstained"))
            if is_list:
                items.update({k: f.get(k, 0) for k in ("aligned", "extra", "missing")})
            if f["confidence"] is not None:
                _bucket(list_buckets if is_list else scalar_buckets, f["confidence"], correct)
            if r["type_correct"]:
                outcome = "wrong" if not correct else "right"
                extraction_flags[("flagged" if f["rule_flagged"] else "not_flagged", outcome)] += 1
                if f["verified"]:
                    verifier[("flagged" if f["semantic_flagged"] else "not_flagged", outcome)] += 1
                else:
                    unverified_fields += 1
    all_field_scores = [s for scores in field_scores.values() for s in scores]
    jev_tokens = sum(r["usage"]["jev_input_tokens"] for r in scored)
    clairic_tokens = sum(r["usage"]["clairic_input_tokens"] for r in scored)
    cached_tokens = sum(r["usage"]["jev_cached_tokens"] + r["usage"]["clairic_cached_tokens"] for r in scored)
    pipeline_seconds = [r["pipeline_seconds"] for r in scored]
    attention = [r for r in scored if r["expected_route"] != "auto_approve"]
    clean = [r for r in scored if r["expected_route"] == "auto_approve"]
    return {
        "documents": n,
        "documents_failed": len(failed),
        "errors": [{"doc_id": r["doc_id"], "error": r["error"]} for r in failed],
        "classification_accuracy": _ratio(sum(r["type_correct"] for r in scored), n),
        "misclassified": {f"{t} -> {p}": c for (t, p), c in confusion.items() if t != p},
        "field_accuracy": _mean(all_field_scores),
        "scalar_field_accuracy": _mean(scalar_scores),
        # Positional, like `score_field`: one dropped row fails every later item. `list_items` aligns instead.
        "list_field_accuracy": _mean(list_scores),
        "list_items": {"recall": _ratio(items["aligned"], items["aligned"] + items["missing"]),
                       "precision": _ratio(items["aligned"], items["aligned"] + items["extra"]),
                       "extra_items": items["extra"], "missing_items": items["missing"]},
        "abstained": abstained,
        "field_accuracy_by_type": {
            t: round(sum(s for k, v in field_scores.items() if k.startswith(t + ".") for s in v)
                     / max(sum(len(v) for k, v in field_scores.items() if k.startswith(t + ".")), 1), 3)
            for t in sorted({r["doc_type"] for r in scored})
        },
        "weakest_fields": sorted(((k, round(sum(v) / len(v), 3)) for k, v in field_scores.items()), key=lambda kv: kv[1])[:10],
        "flags": {"tp": tp, "fp": fp, "fn": fn, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                  "per_flag": {k: dict(v) for k, v in sorted(per_flag.items())}},
        "route_accuracy": _ratio(sum(r["route"] == r["expected_route"] for r in scored), n),
        "routes": dict(Counter(f"{r['expected_route']} -> {r['route']}" for r in scored)),
        # What matters operationally: did every document that needed a human get one, and how many clean ones did.
        "safety": {
            "needs_attention_caught": sum(r["route"] != "auto_approve" for r in attention),
            "needs_attention_total": len(attention),
            "false_approvals": sum(r["route"] == "auto_approve" for r in attention),
            "false_rejects": sum(r["route"] == "reject" for r in scored if r["expected_route"] != "reject"),
            "clean_auto_approved": sum(r["route"] == "auto_approve" for r in clean),
            "clean_total": len(clean),
        },
        "confidence_calibration": _calibration(scalar_buckets),
        # A list's confidence is a joint probability over its items, so lists are calibrated apart from scalars.
        "confidence_calibration_lists": _calibration(list_buckets),
        # Jev semantic checks on the fields they ran on; rule flags (missing / low confidence) on every field.
        "verifier": {**{f"{a}/{b}": c for (a, b), c in sorted(verifier.items())}, "fields_not_verified": unverified_fields},
        "extraction_flags": {f"{a}/{b}": c for (a, b), c in sorted(extraction_flags.items())},
        "modes": dict(Counter(r["result"]["mode"] for r in scored if r.get("result"))),
        "models": {m: sorted(v) for m, v in sorted(models.items())},
        "tokens": {"jev": jev_tokens, "clairic": clairic_tokens},
        "cache": {"hit_rate": _ratio(cached_tokens, jev_tokens + clairic_tokens),
                  "cached_requests": cached_requests, "live_requests": live_requests},
        "cost_usd": cost_summary(scored),
        # Cached model responses make `pipeline` look faster than live calls.
        "seconds_per_document": {"ingest": _mean([r["ingest_seconds"] for r in scored], 2),
                                 "pipeline": _mean(pipeline_seconds, 2),
                                 "end_to_end": _mean([r["seconds"] for r in scored], 2)},
        "latency": {"p50": _percentile(pipeline_seconds, 0.5), "p95": _percentile(pipeline_seconds, 0.95),
                    "max": max(pipeline_seconds) if pipeline_seconds else None},
    }


def cost_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimated cost at policy prices (as if every request were live) and what this run was billed."""
    rows = [r for r in rows if not r.get("error")]
    n = len(rows)
    total = sum(r["cost"]["total"] for r in rows)
    by_stage: Counter = Counter()
    by_type: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_stage.update(r["stage_cost"])
        by_type[r["doc_type"]].append(r["cost"]["total"])
    return {
        "total": round(total, 6),
        "billed": round(sum(r["cost"]["billed"] for r in rows), 6),
        "per_document": round(total / n, 6) if n else None,
        "per_1000_documents": round(1000 * total / n, 2) if n else None,
        "by_model": {m: round(sum(r["cost"][m] for r in rows), 6) for m in ("jev", "clairic")},
        "by_stage": {k: round(v, 6) for k, v in by_stage.items()},
        "per_document_by_type": {t: round(sum(v) / len(v), 6) for t, v in sorted(by_type.items())},
    }


def load_manifest() -> list[dict[str, Any]]:
    return json.loads((SAMPLES_DIR / "manifest.json").read_text(encoding="utf-8"))


def _token(value: str) -> str:
    return value.strip().casefold().replace("-", "_")


def select_entries(manifest: list[dict[str, Any]], only: list[str] | None) -> list[dict[str, Any]]:
    """Manifest entries named by `only`: doc ids, doc types or doc id prefixes, case-insensitive, `-` and `_` alike,
    comma-joined tokens split. Every entry when `only` is empty; NoMatchingSamples for a token that names nothing."""
    tokens = [raw.strip() for item in only or [] for raw in item.split(",") if raw.strip()]
    if not tokens:
        return list(manifest)
    chosen: set[str] = set()
    for raw in tokens:
        token = _token(raw)
        hits = {e["doc_id"] for e in manifest
                if _token(e["doc_id"]) == token or _token(e["doc_type"]) == token or _token(e["doc_id"]).startswith(token)}
        if not hits:
            names = sorted({e["doc_type"] for e in manifest} | {e["doc_id"].rsplit("-", 1)[0] for e in manifest})
            raise NoMatchingSamples(f"No sample matches {raw!r}. Use a doc type or id prefix ({', '.join(names)}) "
                                    f"or a doc id such as {manifest[0]['doc_id']!r}." if manifest else
                                    f"No sample matches {raw!r}: the manifest is empty (run `idp generate`).")
        chosen |= hits
    return [e for e in manifest if e["doc_id"] in chosen]


def failed_row(entry: dict[str, Any], exc: BaseException) -> dict[str, Any]:
    """The row for a document the pipeline could not process: excluded from scoring, listed under the scorecard's
    `errors`, and shaped like a scored row where the eval table needs it."""
    return {
        "doc_id": entry["doc_id"], "doc_type": entry["doc_type"], "error": f"{type(exc).__name__}: {exc}",
        "predicted_type": None, "type_correct": False, "expected_route": entry["expected_route"], "route": "error",
        "expected_flags": sorted(entry["expected_flags"]), "flags": [], "quality_flags": [], "fields": {},
    }


def run_eval(*, source: str = "digital", mode: ModeChoice = "auto", only: list[str] | None = None, workers: int = 4,
             use_cache: bool = True, progress=None) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    entries = select_entries(load_manifest(), only)
    truths = {e["doc_id"]: json.loads((SAMPLES_DIR / "ground_truth" / f"{e['doc_id']}.json").read_text(encoding="utf-8"))
              for e in entries}
    # OCR is CPU-bound and runs first, sequentially; model calls then run in parallel.
    docs, ingest_seconds = {}, {}
    for e in entries:
        start = time.perf_counter()
        docs[e["doc_id"]] = ingest.load(SAMPLES_DIR / e[source])
        ingest_seconds[e["doc_id"]] = round(time.perf_counter() - start, 3)
        if progress:
            progress(f"ingested {e['doc_id']}")
    pipeline = Pipeline(use_cache=use_cache)

    def run(e: dict) -> dict:
        doc_id = e["doc_id"]
        try:
            result = pipeline.process(SAMPLES_DIR / e[source], mode, doc=docs[doc_id])
            result.timings = {"ingest": ingest_seconds[doc_id], **result.timings}
            row = evaluate_doc(truths[doc_id], result)
            row["result"] = result.model_dump(mode="json", exclude={"text"})
        except Exception as exc:  # noqa: BLE001 - one failing document is scored as an error, not the end of the run
            row = failed_row(e, exc)
            if progress:
                progress(f"{doc_id}: FAILED {row['error']}")
            return row
        if progress:
            progress(f"{doc_id}: {row['predicted_type']} -> {row['route']} (expected {row['expected_route']})")
        return row

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(run, entries))
    finally:
        pipeline.close()

    # `auto` names the mode it resolved to for most documents, so eval-digital-auto-text and eval-scanned-auto-hybrid
    # runs can be told apart on disk.
    resolved = Counter(r["result"]["mode"] for r in rows if r.get("result"))
    label = f"{mode}-{resolved.most_common(1)[0][0]}" if mode == "auto" and resolved else mode
    out = OUT_DIR / f"eval-{source}-{label}-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    card = scorecard(rows)
    (out / "scorecard.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card, rows, out
