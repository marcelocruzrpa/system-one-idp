"""Jev (TypeSafe System One) wrapper: one cached call returning plain-dict answers."""

import warnings
from dataclasses import dataclass, field
from typing import Any

from typesafe_sdk import Image, Question, RetryPolicy, TypeSafeClient

from idp.clients.cache import ResponseCache
from idp.config import policy
from idp.models import Usage

IMAGES_HEADER = {"TypeSafe-Danger-Zone-Images": "true"}
# Model names that resolve server-side; the answering version is expected to differ from them.
ALIAS_SUFFIXES = ("-latest", "-preview")


@dataclass
class JevResult:
    model: str
    answers: dict[str, dict[str, Any]]
    input_tokens: int = 0
    cached: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def noul(self, name: str) -> float:
        return float(self.answers[name]["noul"])

    def choice(self, name: str) -> dict[str, Any]:
        return self.answers[name]

    def usage(self) -> Usage:
        return Usage(jev_input_tokens=self.input_tokens, jev_requests=1,
                     jev_cached_tokens=self.input_tokens if self.cached else 0)


class Jev:
    def __init__(self, *, model: str | None = None, use_cache: bool = True, timeout: float = 30.0) -> None:
        self.model = model or policy()["models"]["jev"]
        # The SDK reads TYPESAFE_API_KEY from the environment (loaded from .env by idp.config). Retries cover 408/429/5xx
        # and connection errors; `timeout=None` lifts the SDK's 30 s total retry budget so a retried request can
        # still use its full per-attempt timeout.
        self.client = TypeSafeClient(model=self.model, timeout=timeout, retry=RetryPolicy(max_retries=3, timeout=None))
        self.cache = ResponseCache("jev", enabled=use_cache)

    def ask(self, state: dict[str, Any], questions: dict[str, Question], *, images: list[bytes] | None = None) -> JevResult:
        """Ask independent questions over the same state in one request.

        With `images`, page images are added to the state as `pages` and the Danger Zone images header is sent.
        """
        key = self.cache.key({"model": self.model, "state": state, "images": images or [], "questions": questions})
        if (cached := self.cache.get(key)) is not None:
            return JevResult(cached["model"], cached["answers"], cached.get("input_tokens", 0), cached=True)
        headers = None
        if images:
            state = {**state, "pages": [Image(image) for image in images]}
            headers = IMAGES_HEADER
        response = self.client.system_one(state=state, questions=questions, extra_headers=headers)
        if response.model != self.model and not self.model.endswith(ALIAS_SUFFIXES):
            # The thresholds, the response cache and the eval scorecards were produced by the pinned version.
            warnings.warn(f"Jev answered as {response.model!r} although policy.yaml pins {self.model!r}; "
                          "thresholds and cached answers were tuned for the pinned version.", RuntimeWarning, stacklevel=2)
        answers = {name: answer.model_dump(mode="json") for name, answer in response.answers.items()}
        result = JevResult(response.model, answers, response.usage.input_tokens or 0)
        self.cache.put(key, {"model": result.model, "answers": answers, "input_tokens": result.input_tokens})
        return result

    def close(self) -> None:
        self.client.close()
