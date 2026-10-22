# Eval scorecards

Produced on 2026-10-22 by `uv run idp eval --source <digital|scanned> --mode <text|hybrid>` over the 25 synthetic samples
in `samples/` with jev-1.14.0 and `@cf/cloudflare/clairic` (preview). They are the numbers quoted in the README's
Results table.

| File | Command |
|---|---|
| `digital-text.scorecard.json` | `uv run idp eval --source digital --mode text` |
| `scanned-text.scorecard.json` | `uv run idp eval --source scanned --mode text` |
| `scanned-hybrid.scorecard.json` | `uv run idp eval --source scanned --mode hybrid` |

These runs replayed cached model responses, so `cost_usd.billed`, `cache.hit_rate` and the `latency` block do not
describe live calls. The live latencies measured the same day with the same requests were about 0.9 s model time per
digital document (text), 1.1 s per scanned document (text) and 2.8 s per scanned document (hybrid, p95 about 8 s for
the two-page invoice), with 3-4 documents processed in parallel.
