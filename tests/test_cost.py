"""Cost metrics: pricing from policy.yaml, cache hits not billed, per-stage breakdown sums to the total."""

import pytest

from idp import ingest
from idp.clients.jev import JevResult
from idp.config import SAMPLES_DIR, policy
from idp.evaluate import cost_summary, evaluate_doc
from idp.models import Usage
from idp.pipeline import Pipeline
from tests.conftest import load_truth

PRICES = policy()["pricing_per_million_input"]


def test_usage_cost_uses_policy_prices():
    cost = Usage(jev_input_tokens=1_000_000, clairic_input_tokens=2_000_000).cost()
    assert cost.jev == pytest.approx(PRICES["jev"])
    assert cost.clairic == pytest.approx(2 * PRICES["clairic"])
    assert cost.total == pytest.approx(PRICES["jev"] + 2 * PRICES["clairic"])
    assert cost.billed == pytest.approx(cost.total)


def test_cached_requests_are_not_billed():
    live = JevResult(model="m", answers={}, input_tokens=500).usage()
    cached = JevResult(model="m", answers={}, input_tokens=500, cached=True).usage()
    assert cached.cost().total == pytest.approx(live.cost().total)
    assert cached.cost().billed == 0
    usage = Usage()
    usage.add(live)
    usage.add(cached)
    assert usage.cost().billed == pytest.approx(live.cost().total)


@pytest.mark.skipif(not (SAMPLES_DIR / "manifest.json").exists(), reason="run `idp generate` first")
def test_pipeline_cost_by_stage(fakes):
    jev, clairic = fakes
    truth = load_truth("invoice-01")
    jev.truth = clairic.truth = truth
    doc = ingest.load(SAMPLES_DIR / "digital" / "invoice-01.pdf")
    result = Pipeline(jev=jev, clairic=clairic).process("invoice-01", "text", doc=doc)

    assert set(result.stage_usage) == {"classify", "extract", "semantic", "match"}
    # Fakes report 100 tokens per Jev request (3 here) and 200 per Clairic request.
    assert result.cost.total == pytest.approx((300 * PRICES["jev"] + 200 * PRICES["clairic"]) / 1e6)
    assert sum(u.cost().total for u in result.stage_usage.values()) == pytest.approx(result.cost.total)
    assert result.model_dump(mode="json")["cost"]["total"] == pytest.approx(result.cost.total)

    summary = cost_summary([evaluate_doc(truth, result)] * 2)
    assert summary["total"] == pytest.approx(2 * result.cost.total, abs=1e-6)
    assert summary["per_1000_documents"] == round(1000 * result.cost.total, 2)
    assert set(summary["by_stage"]) == set(result.stage_usage)
    assert summary["per_document_by_type"] == {"invoice": round(result.cost.total, 6)}
