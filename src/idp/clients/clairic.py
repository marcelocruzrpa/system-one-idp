"""Clairic (Cloudflare Workers AI) REST client: typed field extraction from text and/or images."""

import base64
import os
import time
from typing import Any

import httpx

from idp.clients.cache import ResponseCache
from idp.config import policy

API_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
MAX_IMAGES = 4
MAX_IMAGE_BYTES = 4 * 1024 * 1024
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
# A Retry-After longer than this is not worth waiting for in a foreground run: fall back to the backoff.
MAX_RETRY_AFTER = 30.0
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8"


class ClairicError(RuntimeError):
    pass


class ClairicConfigError(ClairicError):
    """Missing or rejected credentials: the fix is in .env, not in retrying."""


def content_type(image: bytes) -> str:
    """`image/png` or `image/jpeg` from the file signature (`ingest.encode_for_models` produces either)."""
    if image.startswith(PNG_SIGNATURE):
        return "image/png"
    if image.startswith(JPEG_SIGNATURE):
        return "image/jpeg"
    raise ClairicError("Clairic images must be PNG or JPEG bytes.")


def image_payload(image: bytes) -> dict[str, str]:
    return {"content_type": content_type(image), "base64": base64.b64encode(image).decode("ascii")}


class Clairic:
    def __init__(self, *, use_cache: bool = True, timeout: float = 60.0, max_retries: int = 3,
                 transport: httpx.BaseTransport | None = None) -> None:
        account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
        token = (os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get("CLOUDFLARE_AUTH_TOKEN") or "").strip()
        if not account_id or not token:
            raise ClairicConfigError("Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN (see .env.example).")
        self.model = policy()["models"]["clairic"]
        self.account_id = account_id
        self.url = API_URL.format(account_id=account_id, model=self.model)
        self.http = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {token}"}, transport=transport)
        self.cache = ResponseCache("clairic", enabled=use_cache)
        self.max_retries = max_retries

    def extract(
        self,
        fields: dict[str, dict[str, Any]],
        *,
        state: str = "",
        images: list[bytes] | None = None,
        reference_date: str | None = None,
    ) -> dict[str, Any]:
        """Run one extraction. Returns Clairic's result object: `answers`, `state_text`, `truncated`, `usage`.

        A response served from the cache also carries `cached: True`.
        """
        images = images or []
        if len(images) > MAX_IMAGES:
            raise ClairicError(f"Clairic accepts at most {MAX_IMAGES} images, got {len(images)}.")
        if not state and not images:
            raise ClairicError("Clairic needs non-empty state text or at least one image.")
        body: dict[str, Any] = {
            "model": "clairic",
            "state": state,
            "fields": fields,
            "options": {"return_state_text": True},
        }
        if reference_date:
            body["reference_date"] = reference_date
        key = self.cache.key({**body, "images": images})
        if (cached := self.cache.get(key)) is not None:
            return {**cached, "cached": True}
        if images:
            body["images"] = [image_payload(image) for image in images]
        result = self._post(body)
        self.cache.put(key, result)
        return result

    def _redact(self, text: Any) -> str:
        """Error text never carries the account id (it is part of the request URL httpx quotes in its messages)."""
        return str(text).replace(self.account_id, "<account>")

    def _delay(self, attempt: int, response: httpx.Response | None = None) -> float:
        """Exponential backoff, unless a 429/503 names a `Retry-After` in seconds (capped at MAX_RETRY_AFTER)."""
        if response is not None and response.status_code in (429, 503):
            header = (response.headers.get("Retry-After") or "").strip()
            if header.isdigit():
                return min(float(header), MAX_RETRY_AFTER)
        return float(2**attempt)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST with retries on transport errors and retryable statuses; every failure surfaces as a ClairicError."""
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                response = self.http.post(self.url, json=body)
            except httpx.TransportError as exc:
                if attempt + 1 < attempts:
                    time.sleep(self._delay(attempt))
                    continue
                raise ClairicError(f"Cloudflare unreachable after {attempts} attempts: {self._redact(exc)}") from exc
            if response.status_code in RETRY_STATUSES and attempt + 1 < attempts:
                time.sleep(self._delay(attempt, response))
                continue
            break
        return self._parse(response)

    def _parse(self, response: httpx.Response) -> dict[str, Any]:
        if response.status_code in (401, 403):
            raise ClairicConfigError(f"HTTP {response.status_code}: Cloudflare rejected the API token or account id "
                                     f"({self._redact(response.text[:300])}). Check CLOUDFLARE_ACCOUNT_ID and "
                                     "CLOUDFLARE_API_TOKEN in .env.")
        try:
            data = response.json()
        except ValueError:
            raise ClairicError(f"HTTP {response.status_code}: {self._redact(response.text[:500])}") from None
        if not isinstance(data, dict):
            raise ClairicError(f"HTTP {response.status_code}: unexpected response {self._redact(str(data)[:500])}")
        if response.status_code >= 400 or data.get("success") is False:
            raise ClairicError(self._redact(f"HTTP {response.status_code}: {data.get('errors') or data}"))
        # The REST API wraps the model output in Cloudflare's {"result": ..., "success": ...} envelope.
        return data.get("result", data)

    def close(self) -> None:
        self.http.close()
