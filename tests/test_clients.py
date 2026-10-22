"""Model clients: transient failures are retried, config problems named, httpx never leaks; the cache survives corruption."""

import json
import warnings
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image
from typesafe_sdk import Noul

from idp import ingest
from idp.clients import Clairic, ClairicConfigError, ClairicError, Jev
from idp.clients import cache as cache_module, clairic as clairic_module, jev as jev_module
from idp.clients.cache import ResponseCache
from idp.clients.clairic import content_type, image_payload

ACCOUNT = "acc0unt1d"
FIELDS = {"total": {"type": "number", "decimals": 2}}
OK = {"success": True, "result": {"answers": {"total": {"type": "number", "value": 12.5}}, "truncated": False,
                                  "usage": {"input_tokens": 5}}}


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", ACCOUNT)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "token")
    sleeps: list[float] = []
    monkeypatch.setattr(clairic_module.time, "sleep", sleeps.append)
    return sleeps


def client(handler, **kwargs) -> Clairic:
    return Clairic(use_cache=False, transport=httpx.MockTransport(handler), **kwargs)


def test_rate_limit_then_success_honours_retry_after(env):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, json={"success": False, "errors": [{"message": "rate limited"}]},
                                  headers={"Retry-After": "7"})
        return httpx.Response(200, json=OK)

    result = client(handler).extract(FIELDS, state="Total 12.50")
    assert result["answers"]["total"]["value"] == 12.5 and "cached" not in result
    assert len(calls) == 2 and env == [7.0]


def test_unreachable_api_is_a_clairic_error_not_an_httpx_error(env):
    def handler(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ClairicError, match="unreachable after 3 attempts") as info:
        client(handler, max_retries=2).extract(FIELDS, state="x")
    assert not isinstance(info.value, httpx.HTTPError)
    assert env == [1.0, 2.0]


def test_failed_envelope_is_a_clairic_error(env):
    body = {"success": False, "errors": [{"code": 7000, "message": f"no such account {ACCOUNT}"}], "result": None}
    with pytest.raises(ClairicError, match="7000") as info:
        client(lambda request: httpx.Response(200, json=body)).extract(FIELDS, state="x")
    assert not isinstance(info.value, ClairicConfigError)
    assert ACCOUNT not in str(info.value) and "<account>" in str(info.value)


def test_unauthorized_is_a_config_error(env):
    handler = lambda request: httpx.Response(401, json={"success": False, "errors": [{"message": "Invalid token"}]})  # noqa: E731
    with pytest.raises(ClairicConfigError, match="401"):
        client(handler).extract(FIELDS, state="x")
    assert env == []  # never retried


def test_server_errors_retry_then_surface(env):
    statuses = []

    def handler(request):
        statuses.append(503)
        return httpx.Response(503, text="upstream down")

    with pytest.raises(ClairicError, match="HTTP 503"):
        client(handler, max_retries=1).extract(FIELDS, state="x")
    assert statuses == [503, 503]


def test_missing_credentials_is_a_config_error(monkeypatch):
    for name in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ClairicConfigError, match="CLOUDFLARE_ACCOUNT_ID"):
        Clairic(use_cache=False)


def test_image_payload_sniffs_png_and_jpeg(monkeypatch):
    image = Image.new("RGB", (300, 300), "white")
    png, kind = ingest.encode_for_models(image)
    assert kind == "image/png" and image_payload(png)["content_type"] == "image/png"
    monkeypatch.setattr(ingest, "MAX_PNG_BYTES", 16)  # force the JPEG fallback
    jpeg, kind = ingest.encode_for_models(image)
    assert kind == "image/jpeg" and jpeg[:2] == b"\xff\xd8" and content_type(jpeg) == "image/jpeg"
    with pytest.raises(ClairicError):
        image_payload(b"GIF89a...")


def test_cache_round_trip_and_corrupt_entry_recovery(tmp_path):
    cache = ResponseCache("clairic", directory=tmp_path)
    key = cache.key({"state": "x", "images": [b"\x89PNG"]})
    assert cache.get(key) is None
    cache.put(key, {"answers": {"total": 1}})
    assert cache.get(key) == {"answers": {"total": 1}}
    assert [p.name for p in (tmp_path / "clairic").iterdir()] == [f"{key}.json"]  # no temp files left behind

    (tmp_path / "clairic" / f"{key}.json").write_text('{"answers": {"total"', encoding="utf-8")
    assert cache.get(key) is None
    assert not (tmp_path / "clairic" / f"{key}.json").exists()
    assert ResponseCache("clairic", enabled=False, directory=tmp_path).get(key) is None


class StubTypeSafeClient:
    """Stands in for the SDK client and answers as a newer Jev than the one requested."""

    def __init__(self, *, model, timeout, retry):
        self.model, self.timeout, self.retry = model, timeout, retry

    def system_one(self, *, state, questions, extra_headers=None):
        answer = SimpleNamespace(model_dump=lambda mode="json": {"type": "noul", "noul": 0.9})
        return SimpleNamespace(model="jev-1.15.0", answers={name: answer for name in questions},
                               usage=SimpleNamespace(input_tokens=42))

    def close(self):
        pass


def test_jev_warns_when_a_pinned_model_answers_as_another_version(monkeypatch, tmp_path):
    monkeypatch.setattr(jev_module, "TypeSafeClient", StubTypeSafeClient)
    monkeypatch.setattr(cache_module, "CACHE_DIR", tmp_path)
    questions = {"paid": Noul(instructions="Is it marked paid?")}
    jev = Jev(model="jev-1.14.0")
    assert jev.client.timeout == 30.0 and jev.client.retry.max_retries == 3 and jev.client.retry.timeout is None
    with pytest.warns(RuntimeWarning, match="jev-1.15.0"):
        result = jev.ask({"document": "x"}, questions)
    assert (result.model, result.input_tokens, result.cached, result.noul("paid")) == ("jev-1.15.0", 42, False, 0.9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        replay = jev.ask({"document": "x"}, questions)  # served from the cache, which kept the answering model
        alias = Jev(model="jev-latest", use_cache=False).ask({"document": "y"}, questions)  # aliases are expected to move
    assert replay.cached and replay.model == "jev-1.15.0" and alias.model == "jev-1.15.0"


def test_cache_key_ignores_nothing_that_matters(tmp_path):
    cache = ResponseCache("jev", directory=tmp_path)
    assert cache.key({"a": 1, "b": b"img"}) == cache.key({"b": b"img", "a": 1})
    assert cache.key({"a": 1, "b": b"img"}) != cache.key({"a": 1, "b": b"other"})
    json.loads(json.dumps({"k": cache.key({})}))  # keys are plain hex strings
