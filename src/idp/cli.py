"""Command line: smoke-test the APIs, generate samples, OCR, process one file, evaluate."""

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from idp.config import OUT_DIR, ROOT, SAMPLES_DIR
from idp.models import ModeChoice, ProcessResult

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
console = Console()

ModeOpt = Annotated[str, typer.Option("--mode", "-m", help="auto (text if every page has a text layer, else hybrid) "
                                                           "| text | image | hybrid")]
NoCacheOpt = Annotated[bool, typer.Option("--no-cache", help="Bypass the response cache in .cache/")]
InputPath = Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="A PDF, PNG or JPEG file")]
ROUTE_STYLE = {"auto_approve": "bold green", "review": "bold yellow", "reject": "bold red"}
KEYS_HINT = "Copy .env.example to .env and fill in TYPESAFE_API_KEY, CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN."
RETRY_HINT = "Temporary API problem: retry in a few seconds."


def _hint(exc: BaseException) -> str | None:
    """What to do about an API error, when there is something to do: fill in the keys, or simply retry."""
    from typesafe_sdk import TypeSafeAPIConnectionError, TypeSafeAPIError, TypeSafeError

    from idp.clients import ClairicConfigError, ClairicError

    if isinstance(exc, ClairicConfigError):
        return KEYS_HINT
    if isinstance(exc, TypeSafeError):
        status = exc.status if isinstance(exc, TypeSafeAPIError) else None
        if status in (401, 403) or re.search(r"api[ _]?key|authenticat|unauthori[sz]ed", str(exc), re.IGNORECASE):
            return KEYS_HINT
        if isinstance(exc, TypeSafeAPIConnectionError) or (status is not None and (status == 429 or status >= 500)):
            return RETRY_HINT
        return None
    if isinstance(exc, ClairicError) and re.search(r"HTTP (408|429|5\d\d)\b|unreachable", str(exc)):
        return RETRY_HINT
    return None


@contextmanager
def api_errors() -> Iterator[None]:
    """Turn missing keys, API failures and unreadable input into one line (plus a hint) instead of a traceback."""
    import httpx
    from typesafe_sdk import TypeSafeError

    from idp.clients import ClairicError

    try:
        yield
    except (typer.Exit, typer.Abort):
        raise  # typer's own control flow (both are RuntimeErrors); BadParameter is a plain Exception and passes through
    except (TypeSafeError, ClairicError, httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
        console.print(f"[red]{type(exc).__name__}:[/] {escape(str(exc))}")
        if hint := _hint(exc):
            console.print(f"[dim]{escape(hint)}[/]")
        raise typer.Exit(1) from None


def _mode(value: str) -> ModeChoice:
    if value not in ("auto", "text", "image", "hybrid"):
        raise typer.BadParameter("mode must be auto, text, image or hybrid")
    return value  # type: ignore[return-value]


def _source(value: str) -> str:
    if value not in ("digital", "scanned"):
        raise typer.BadParameter("source must be digital or scanned")
    return value


def _rel(path: Path) -> str:
    """A path as the user would type it from the repo root, where `uv run idp` runs."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


@app.command()
def smoke() -> None:
    """Call Jev and Clairic once each (text and image) and print the response shapes."""
    with api_errors():
        _smoke()


def _smoke() -> None:
    from typesafe_sdk import Choice, Noul

    from idp import ingest
    from idp.clients import Clairic, Jev

    doc = ingest.load(SAMPLES_DIR / "digital" / "invoice-01.pdf")
    jev = Jev(use_cache=False)
    for label, images in (("text", None), ("image", [doc.pages[0].png])):
        state = {"document": doc.text} if images is None else {}
        result = jev.ask(state, {
            "kind": Choice(instructions="What kind of document is this?", criteria={"invoice": None, "receipt": None, "other": None}),
            "paid": Noul(instructions="Does the document show it is already paid?"),
        }, images=images)
        console.print(f"[bold]Jev ({label})[/] model={result.model} tokens={result.input_tokens}", result.answers)

    clairic = Clairic(use_cache=False)
    fields = {"invoice_number": {"type": "text"}, "total": {"type": "number", "decimals": 2},
              "due_date": {"type": "date"}, "line_amounts": {"type": "list", "items": "number"}}
    for label, kwargs in (("text", {"state": doc.text}), ("image", {"images": [doc.pages[0].png]})):
        result = clairic.extract(fields, **kwargs)
        console.print(f"[bold]Clairic ({label})[/] usage={result.get('usage')} truncated={result.get('truncated')}",
                      result.get("answers"))
        if label == "image":
            console.print("[bold]state_text (image mode)[/]:", (result.get("state_text") or "")[:400])
    console.print("[green]Both APIs answered.[/]")


@app.command()
def generate(only: Annotated[list[str] | None, typer.Option(help="Only these doc ids")] = None) -> None:
    """Render sample documents, scanned variants, ground truth and the mock ERP into samples/."""
    from idp.samplegen.render import generate as render

    manifest = render(SAMPLES_DIR, only=set(only) if only else None)
    table = Table("doc", "type", "pages", "expected route", "seeded defects")
    for m in manifest:
        table.add_row(m["doc_id"], m["doc_type"], str(m["pages"]), m["expected_route"], ", ".join(m["expected_flags"]))
    console.print(table)
    console.print(f"Wrote {len(manifest)} documents to {_rel(SAMPLES_DIR)}")


@app.command()
def erp(no_open: Annotated[bool, typer.Option("--no-open", help="Only write the file")] = False) -> None:
    """Render the mock ERP (samples/erp.json) as a clickable web page in out/erp.html and open it."""
    import webbrowser

    from idp.erp_view import write

    path = write()
    console.print(f"Wrote {_rel(path)}")
    if not no_open:
        webbrowser.open(path.as_uri())


@app.command()
def ocr(path: InputPath, force: Annotated[bool, typer.Option("--force", help="OCR even if the PDF has a text layer")] = False) -> None:
    """Print the reading-order text the models will see (no API calls)."""
    from idp import ingest

    with api_errors():
        doc = ingest.load(path, force_ocr=force)
    console.print(f"[dim]{len(doc.pages)} page(s), source: {', '.join(p.text_source for p in doc.pages)}[/]")
    console.print(doc.text, markup=False, highlight=False)


def print_result(result: ProcessResult) -> None:
    c = result.classification
    console.print(f"\n[bold]{escape(Path(result.path).name)}[/]  mode={result.mode}  pages={result.page_count}")
    console.print(f"Doc type: [bold]{escape(c.doc_type)}[/] (confidence {c.confidence:.2f})  "
                  f"signals: {', '.join(f'{k}={v:.2f}' for k, v in c.signals.items())}")
    if result.extraction:
        table = Table("field", "value", "conf", "p_null")
        for name, a in result.extraction.fields.items():
            value = json.dumps(a.value, ensure_ascii=False) if a.is_list else str(a.value)
            # A list's confidence is a joint probability over its items, not comparable to a scalar's.
            conf = "n/a (list)" if a.is_list else f"{a.confidence:.2f}" if a.confidence is not None else ""
            table.add_row(escape(name), escape(value[:90]), conf, f"{a.p_null:.2f}" if a.p_null is not None else "")
        console.print(table)
    checks = Table("check", "stage", "result", "detail")
    for ch in result.checks:
        checks.add_row(escape(ch.id), ch.stage, "[green]pass[/]" if ch.passed else "[red]FLAG[/]", escape(ch.detail[:110]))
    console.print(checks)
    for error in result.errors:
        console.print(f"[yellow]Model call failed:[/] {escape(error)}")
    console.print(f"Route: [{ROUTE_STYLE[result.route]}]{result.route}[/]")
    u, cost = result.usage, result.cost
    console.print(f"Cost: [bold]${cost.total:.6f}[/] (${1000 * cost.total:.2f} per 1,000 docs; "
                  f"jev ${cost.jev:.6f}, clairic ${cost.clairic:.6f}; billed this run ${cost.billed:.6f})")
    models = " ".join(f"{name}={version}" for name, version in result.models.items()) or "n/a"
    lines = f"  line items: {len(result.line_items.get('rows', []))} rows" if result.line_items else ""
    stages = ", ".join(f"{stage} ${s.cost().total:.6f}" for stage, s in result.stage_usage.items())
    console.print("[dim]" + escape(f"models: {models}{lines}  by stage: {stages}  tokens: jev={u.jev_input_tokens} "
                                   f"clairic={u.clairic_input_tokens}  timings: {result.timings}") + "[/]")


@app.command()
def run(path: InputPath, mode: ModeOpt = "auto", no_cache: NoCacheOpt = False,
        force_ocr: Annotated[bool, typer.Option("--force-ocr", help="OCR even if the PDF has a text layer")] = False) -> None:
    """Process one PDF/PNG/JPEG end to end and save the result JSON to out/."""
    from idp.pipeline import Pipeline

    chosen = _mode(mode)
    with api_errors():
        pipeline = Pipeline(use_cache=not no_cache)
        try:
            result = pipeline.process(path, chosen, force_ocr=force_ocr)
        finally:
            pipeline.close()
    print_result(result)
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"run-{path.stem}-{result.mode}.json"
    out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"[dim]Saved {escape(_rel(out))}[/]")


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} s"


def print_eval_table(rows: list[dict[str, Any]]) -> None:
    table = Table("doc", "type", "predicted", "fields", "expected flags", "flags", "route")
    for r in rows:
        if r.get("error"):
            table.add_row(escape(r["doc_id"]), escape(r["doc_type"]), "[red]error[/]", "", ", ".join(r["expected_flags"]),
                          f"[red]{escape(r['error'][:70])}[/]", "[red]error[/]")
            continue
        scores = [f["score"] for f in r["fields"].values()]
        ok_route = r["route"] == r["expected_route"]
        table.add_row(
            escape(r["doc_id"]), escape(r["doc_type"]),
            escape(r["predicted_type"]) if r["type_correct"] else f"[red]{escape(str(r['predicted_type']))}[/]",
            f"{sum(scores) / max(len(scores), 1):.0%}", ", ".join(r["expected_flags"]),
            escape(", ".join(r["flags"])) + (f" [dim]+{len(r['quality_flags'])}q[/]" if r["quality_flags"] else ""),
            f"[{'green' if ok_route else 'red'}]{r['route']}[/]",
        )
    console.print(table)


def print_scorecard(card: dict[str, Any]) -> None:
    """The headline numbers; the full scorecard is in scorecard.json next to results.jsonl."""
    safety, cost, latency, items = card["safety"], card["cost_usd"], card["latency"], card["list_items"]
    failed = f", [red]{card['documents_failed']} failed[/]" if card["documents_failed"] else ""
    console.print(f"\n[bold]Documents:[/] {card['documents']} scored{failed}")
    for e in card["errors"]:
        console.print(f"  [red]{escape(e['doc_id'])}[/]: {escape(e['error'])}")
    misclassified = f"  misclassified: {escape(str(card['misclassified']))}" if card["misclassified"] else ""
    console.print(f"[bold]Classification:[/] {_pct(card['classification_accuracy'])}{misclassified}")
    console.print(f"[bold]Fields:[/] scalar {_pct(card['scalar_field_accuracy'])}, list (positional) "
                  f"{_pct(card['list_field_accuracy'])}; list items recall {_pct(items['recall'])}, precision "
                  f"{_pct(items['precision'])} ({items['missing_items']} missing, {items['extra_items']} extra); "
                  f"abstained {card['abstained']}")
    approvals = f"[red]{safety['false_approvals']}[/]" if safety["false_approvals"] else "0"
    console.print(f"[bold]Routing:[/] {_pct(card['route_accuracy'])}  needs-attention caught "
                  f"{safety['needs_attention_caught']}/{safety['needs_attention_total']}, false approvals {approvals}, "
                  f"false rejects {safety['false_rejects']}, clean auto-approved "
                  f"{safety['clean_auto_approved']}/{safety['clean_total']}")
    console.print(f"[bold]Flags:[/] precision {_pct(card['flags']['precision'])}, recall {_pct(card['flags']['recall'])}")
    per_1000 = cost["per_1000_documents"]
    console.print(f"[bold]Cost:[/] {'n/a' if per_1000 is None else f'${per_1000:.2f}'} per 1,000 docs "
                  f"(billed this run ${cost['billed']:.6f})  [bold]Latency:[/] p50 {_seconds(latency['p50'])}, "
                  f"p95 {_seconds(latency['p95'])}  [bold]Cache:[/] hit rate {_pct(card['cache']['hit_rate'])}")
    models = ", ".join(f"{name}={'/'.join(versions)}" for name, versions in card["models"].items()) or "n/a"
    console.print(f"[bold]Models:[/] {escape(models)}  modes: {escape(str(card['modes']))}")
    if card["weakest_fields"]:
        console.print("[bold]Weakest fields:[/] " + escape(", ".join(f"{k} {v:.0%}" for k, v in card["weakest_fields"][:5])))


@app.command(name="eval")
def eval_cmd(
    source: Annotated[str, typer.Option(help="digital (PDF text layer) | scanned (OCR)")] = "digital",
    mode: ModeOpt = "auto",
    only: Annotated[list[str] | None, typer.Option(help="Doc ids, id prefixes or doc types (repeat or comma-join)")] = None,
    workers: Annotated[int, typer.Option(min=1, help="Documents processed in parallel")] = 4,
    no_cache: NoCacheOpt = False,
) -> None:
    """Run the pipeline over the samples and print a scorecard against ground truth."""
    from idp.evaluate import NoMatchingSamples, run_eval

    source, chosen = _source(source), _mode(mode)
    with api_errors():
        try:
            card, rows, out = run_eval(source=source, mode=chosen, only=only, workers=workers, use_cache=not no_cache,
                                       progress=lambda msg: console.print(f"[dim]{escape(msg)}[/]"))
        except NoMatchingSamples as exc:
            raise typer.BadParameter(str(exc), param_hint="--only") from None
    print_eval_table(rows)
    print_scorecard(card)
    console.print(f"[dim]Saved {escape(_rel(out))}[/]")


if __name__ == "__main__":
    app()
