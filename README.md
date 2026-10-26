# IDP POC — System One models (Jev + Clairic)

Document processing without a generative LLM: **Jev judges, Clairic extracts, code decides.**

- **Jev** ([TypeSafe](https://docs.typesafe.ai)) makes the *judgments*: what type of document this is, whether
  an extracted value is real and plays the right role, whether the invoice is addressed to us, already paid,
  asking for a bank change, and which PO line an invoice line refers to.
- **Clairic** ([Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/models/clairic/),
  `@cf/cloudflare/clairic`) does the *typed extraction*: fields come back as `value`, `confidence`, `p_null`
  and a character `span` into the text.
- **Code** does everything that is a rule: arithmetic, date order, ERP lookups, thresholds, routing.

```
PDF / PNG / JPEG
  -> ingest     PDF text layer (PyMuPDF) or OCR (RapidOCR, PP-OCRv5 English) -> reading-order text with " | " columns
  -> classify   Jev Choice over 10 doc types + speculative Nouls (paid? bank change? several docs?) in ONE request
  -> extract    Clairic, fields from config/schemas/<doc_type>.yaml; line items rebuilt from the page's table rows
  -> validate   input checks (whole file read? one document?) | rules (code) | semantic verifier (Jev Nouls,
                true = problem) | three-way match (supplier + currency in code, Jev line matching, ERP quantities)
  -> route      reject (hard fail) | review (any flag, including checks that could not run) | auto_approve
  -> review     a person approves or rejects every review / reject document in the app, marks each flag a real
                issue or a false alarm, adds a note; decisions go to an append-only log (out/reviews.jsonl)
```

`--mode` chooses what the models read: `text` (OCR/text layer), `image` (page images only — Clairic takes up to 4
images; Jev images are a preview feature sent with `TypeSafe-Danger-Zone-Images: true`), or `hybrid` (both). The
default, `auto`, picks from the document: `text` when every page has a PDF text layer, `hybrid` when any page needed
OCR (the images let the models read past OCR errors). `image` is never picked automatically: it drops pages beyond 4
and the OCR text that evidence spans point into.

## Results

Evaluation over the 25 sample documents (`uv run idp eval`, 2026-10-22, jev-1.14.0), scored against their ground
truth. The scorecards behind the table are in `docs/eval/`.

| Metric | digital/text | scanned/text | scanned/hybrid |
|---|---|---|---|
| Classification | 25/25 | 25/25 | 25/25 |
| Scalar fields correct | 97.9% | 98.3% | 98.3% |
| Line items found (recall / precision) | 100% / 100% | 99.7% / 99.7% | 99.7% / 99.7% |
| Documents needing attention caught | 10/10 | 10/10 | 10/10 |
| False approvals / false rejects | 0 / 0 | 0 / 0 | 0 / 0 |
| Seeded defects flagged (precision) | 13/13 (100%) | 13/13 (100%) | 13/13 (100%) |
| Clean documents auto-approved | 14/15 | 14/15 | 14/15 |
| Cost per 1,000 documents | $0.55 | $0.55 | $2.13 |
| Model time per document (live) | ~0.9 s | ~1.1 s | ~2.8 s |

The one clean document sent to review is a scanned bank statement whose debit total Clairic misread by one dollar;
the Jev verifier flagged the value as unsupported (`unsupported:total_debits`) and the balance rule reported
`unverified:balance` instead of accusing the statement. The two remaining scalar misses are a vendor tax id read
from the bill-to block on the EU invoice in text mode (flagged by the verifier as `wrong_role:vendor_tax_id`) and
the bank statements' currency, which Clairic abstains on (it is stated only in the fine print).

Read these with the caveats in mind:

- **Synthetic data.** The 25 documents, their ground truth and the mock ERP are all generated from one spec
  (`src/idp/samplegen`), with one positive example per seeded defect type. The numbers show that the design works,
  not how it behaves on a real mailbox.
- **Preview model.** Clairic is a Cloudflare preview model; its behaviour (and these numbers) may change.
- **Cost** is at list prices for input tokens: Clairic $0.10 per M input tokens (verified on Cloudflare's pricing
  page); Jev $0.042 per M *assumed*, since TypeSafe publishes no price list. OCR runs locally and is excluded.
- **Latency** is model time with 4 documents processed in parallel, plus about 2 s of CPU OCR per scanned page.

## Setup

```powershell
uv sync
copy .env.example .env                           # PowerShell ...
cp .env.example .env                             # ... or bash; then fill in the three keys
uv run idp smoke                                 # one Jev + one Clairic call, text and image
uv run idp ocr samples/scanned/invoice-01.png   # downloads the English OCR model once (8 MB)
```

Keys: `TYPESAFE_API_KEY`; `CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_API_TOKEN` (token with Workers AI permission).

Only `idp generate` (rendering the sample documents with a headless browser) needs the extra group:
`uv sync --group samplegen` followed by `uv run playwright install chromium`.

## Commands

| Command | What it does |
|---|---|
| `uv run idp generate` | Render the 25 samples, scanned variants, ground truth and mock ERP into `samples/` (needs the `samplegen` group, see Setup) |
| `uv run idp erp` | Open the mock ERP (`samples/erp.json`) as a web page: purchase orders, goods receipts, vendors, posted invoices (`out/erp.html`) |
| `uv run idp ocr samples/scanned/invoice-02.png` | Show the text the models will read (no API calls) |
| `uv run idp run samples/scanned/invoice-03.png` | Process one file; prints fields, checks, route and cost; saves `out/run-*.json` (`--force-ocr` to ignore a PDF text layer) |
| `uv run idp eval --source digital` | Score all samples against ground truth (`--mode auto`); saves `out/eval-*/scorecard.json` |
| `uv run idp eval --source scanned --mode image --only invoice` | Same, scanned inputs, image-only mode, invoices only |
| `uv run streamlit run app/streamlit_app.py` | Interactive demo: pick a sample or upload, inspect every decision, approve or reject review / reject documents (Input: Review queue lists what is waiting) |
| `uv run pytest` | Offline tests, including the full pipeline over all samples with perfect fake models |

Model responses are cached in `.cache/` by request hash, so repeated evals cost nothing; pass `--no-cache` to refresh.

Cost metrics price input tokens at `pricing_per_million_input` in `config/policy.yaml`. Every result carries `cost`
(Jev, Clairic, total, and `billed`: what the run actually paid, since cached responses are free) and `stage_usage`
(classify / extract / semantic / match). The eval scorecard's `cost_usd` adds per 1,000 documents and breakdowns by
model, stage and doc type.

## Sample set

`samples/digital/*.pdf` have a text layer. `samples/scanned/*` are grayscale image-only scans (PNG, or an
image-only PDF for the 2-page invoice), so they go through OCR. `samples/ground_truth/*.json` hold the expected doc
type, every field value, the seeded defects (`expected_flags`) and the expected route. `samples/erp.json` is a mock ERP
with a vendor master, purchase orders (with currency and quantities already invoiced), goods receipts and
already-posted invoices (POs/receipts live in the ERP in real AP flows).

| Doc | Type | Seeded defect -> expected route |
|---|---|---|
| invoice-01 | Northwind (US) | clean, matches PO-4501 + delivery -> auto_approve |
| invoice-02 | Helios (EU, `1.234,56`, `DD.MM.YYYY`) | bills 10 encoders, 8 received -> review `qty_exceeds_received` |
| invoice-03 | Bluefin (US) | chair price 265 vs PO 245 -> review `price_mismatch` |
| invoice-04 | Northwind | printed total 100.00 too high -> review `total_mismatch` |
| invoice-05 | Helios | billed to Apex Robotix B.V., no PO -> reject `not_addressed_to_us`, `missing_po` |
| invoice-06 | Northwind | re-sent, already posted, PAID stamp -> reject `duplicate_invoice`, `already_paid` |
| invoice-07 | Bluefin, 2 pages, 32 lines | due date before issue date, bank-change notice -> review |
| purchase_order-01..03 | Acme POs | clean -> auto_approve |
| delivery_note-01..03 | packing slips | 02 short-shipped -> review `short_shipment` |
| receipt-01..02 | thermal receipts | 02 includes beer -> review `non_reimbursable_items` |
| credit_note-01..02 | | 02 lines sum to 175, subtotal says 185 -> review `line_sum_mismatch` |
| quote-01..02, proforma-01..02 | invoice look-alikes | clean -> auto_approve (tests classification) |
| bank_statement-01..02 | two layouts | clean (balance rule must reconcile) |
| remittance_advice-01..02 | | 02 takes a deduction -> review `short_payment` |

## How the pieces map to the models

| Need | Where | Model / code |
|---|---|---|
| Doc type | `classify.py`, `config/taxonomy.yaml` | Jev **Choice**, contrastive criteria (invoice vs pro forma vs quote) |
| Paid / bank change / multi-doc | `classify.py` | Jev **Nouls** asked speculatively in the same request, reused by validation |
| Whole input read (truncation, pages beyond 4 in image mode) | `pipeline.py` | code |
| Field values | `extract.py`, `config/schemas/*.yaml` | Clairic `text` / `number` / `date` / `time` / `list` fields |
| Line items | `lines.py` | code reads the table rows from the text; Clairic's parallel lists are the cross-check and the fallback |
| Missing / unsure fields | `validate/rules.py` | Clairic `p_null` and `confidence` vs `config/policy.yaml` thresholds |
| Arithmetic, dates, duplicates (vendor + number), PO present | `validate/rules.py` | code |
| Supplier and currency match the PO | `validate/match.py`, `erp.py` | code: vendor master by tax ID, then normalized name / aliases |
| Value unsupported / wrong role | `validate/semantic.py` | Jev **Nouls** per critical field (SDE-cascade pattern) |
| Addressed to us, short payment, non-reimbursable | `validate/semantic.py` | Jev **Nouls** with our company + aliases in state |
| Invoice line <-> PO line | `lines.py`, `validate/match.py` | code reads the table rows; Jev **Choice** maps each row to a PO line (or `no_match`); an uncertain match is `unverified:line_match`; code compares qty/price/received |
| Decision | `route.py` | code (`reject_flags` in policy) |
| Exceptions | `review.py`, the app's "4 - Human review" | a person: approve / reject, a verdict per flag; a note is required to approve a reject or a real issue, or to reject with none |

## Things to know

- **Clairic is a preview** model: outputs are always well-formed, not always right. Multi-step arithmetic is left to
  code on purpose.
- **Line items are rebuilt from the page's table rows** (`src/idp/lines.py`); Clairic's parallel lists (descriptions,
  quantities, prices, amounts) are a cross-check and the fallback when the text has no table. When no consistent
  rows exist, the document goes to review with `unverified:line_items` instead of comparing misaligned lists.
- **Clairic de-duplicates text lists** and returns one joint confidence per list. That number is not comparable to a
  scalar confidence and is never thresholded; number lists carry per-item confidences, which are.
- **The Jev model is pinned** (`jev-1.14.0` in `config/policy.yaml`) because aliases such as `jev-latest` move when
  TypeSafe ships a release, and the thresholds, response cache and scorecards were produced by that version.
- **Jev with images** works only behind the "Danger Zone" header, TypeSafe's opt-in for preview image support
  (280-1568 px per side). `auto` uses images only for OCR'd documents; `--mode text` is the production-safe path.
  Force `--mode text` / `image` / `hybrid` to compare.
- **OCR**: RapidOCR's default Chinese recognizer drops spaces in English text (`AcmeRoboticsInc.`); the pipeline uses
  the English PP-OCRv5 model, downloaded on first use.
- **A check that could not run is a flag**, not a pass: missing inputs raise `unverified:<rule>` (review). Schema rules
  marked `optional: true` (e.g. a due date that invoices may omit) are the only ones skipped silently.
- **Model doubt never accuses the document.** A value Clairic read below `rule_input_min_confidence` cannot raise a
  business flag when a rule fails on it: the rule reports `unverified:<rule>` instead, because the extraction is the
  likelier error. A rule that reconciles still passes.
- **A verifier-rejected input withdraws a passed rule.** When the Jev verifier finds a value unsupported by the
  document (or in the wrong role), every rule that passed on that value is rewritten as `unverified:<rule>`: the
  model's reading was checked, not the document.
- Thresholds in `config/policy.yaml` (classification 0.80, field confidence 0.60, verifier flag 0.70) are starting
  points: tune them from the eval's `confidence_calibration` and `verifier` sections. `verifier` counts only fields the
  Jev verifier actually checked (`verify_fields`); identifiers (`*_number`, `*_id`, `*_reference`, `currency`) are
  scored by exact match.

## Privacy and data

- Documents (their text and, in `image` / `hybrid` mode, their page images) go to exactly two services: TypeSafe
  (Jev) and Cloudflare Workers AI (Clairic). Nothing else leaves the machine; OCR runs locally.
- The Streamlit demo binds to `localhost` (`.streamlit/config.toml`) and sends no usage statistics.
- Uploads (`out/uploads/`), run outputs (`out/`) and cached model responses (`.cache/`, which include the document
  text sent) stay on disk until you delete them: use "Delete my uploads" in the app, or delete the `out/` and
  `.cache/` folders. Both are gitignored.
- The review log (`out/reviews.jsonl`) keeps the full result each decision was made on, extracted values and
  document text included, with the reviewer's name and note. "Delete my uploads" also removes the review cases of
  uploads; "Clear the review log" (Input: Review queue) removes the rest.
- All sample data is fictitious: companies, people, addresses, tax ids and bank details are invented.
