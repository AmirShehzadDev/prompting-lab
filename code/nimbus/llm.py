"""nimbus/llm.py — the shared LLM client used across the whole course.

Wraps the Google Gemini SDK with the production basics:
  * the model name lives in ONE place (easy to swap when a model is deprecated)
  * retries with exponential backoff on rate limits (429) and transient 5xx errors
  * a hard per-call timeout
  * token counting + USD cost estimation, accumulated across calls
  * graceful failure: a blocked/empty reply raises a clear error, not a NoneType crash

Run a quick self-test:  python -m nimbus.llm
"""
from __future__ import annotations
import time
import random
from dataclasses import dataclass

from google import genai
from google.genai import types
from google.genai import errors

# --- Config: change the model in ONE place -----------------------------------
# gemini-2.5-flash = fast + cheap + capable. It's on a deprecation schedule
# (shutdown 2026-10-16); when that lands, swap this single string for the current
# flash model. The call shape below does not change.
DEFAULT_MODEL = "gemini-2.5-flash"

# USD per 1,000,000 tokens. Verified against ai.google.dev/gemini-api/docs/pricing.
# Re-check periodically — providers change prices silently.
PRICING = {
    "gemini-2.5-flash":      {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro":        {"input": 1.25, "output": 10.00},  # flat rate <=200k-token prompts
}

# HTTP status codes worth retrying: rate limit + transient server errors.
RETRYABLE = {429, 500, 502, 503, 504}


@dataclass
class Usage:
    """Running totals so a script can report what it actually cost."""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0

    def add(self, model: str, in_tok: int, out_tok: int) -> None:
        self.calls += 1
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        price = PRICING.get(model, {"input": 0.0, "output": 0.0})
        self.usd += in_tok / 1e6 * price["input"] + out_tok / 1e6 * price["output"]

    def report(self) -> str:
        return (f"{self.calls} calls | {self.input_tokens} in + "
                f"{self.output_tokens} out tokens | ~${self.usd:.6f}")


class LLM:
    """A thin, production-minded wrapper around the Gemini client."""

    def __init__(self, model: str = DEFAULT_MODEL, max_retries: int = 5,
                 timeout_s: float = 30.0):
        # http_options.timeout is in MILLISECONDS. The client reads the API key
        # from GEMINI_API_KEY / GOOGLE_API_KEY automatically.
        self.client = genai.Client(
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
        )
        self.model = model
        self.max_retries = max_retries
        self.usage = Usage()

    def count_tokens(self, contents) -> int:
        """Input-token count BEFORE a call (note: .total_tokens, singular)."""
        return self.client.models.count_tokens(
            model=self.model, contents=contents).total_tokens

    def generate(self, contents, *, system: str | None = None,
                 temperature: float = 0.2, max_output_tokens: int = 1024,
                 **config_kwargs) -> str:
        """Text-in / text-out, with retries, timeout, cost tracking.

        Returns the reply text. Raises after retries are exhausted, or if the
        model returns no usable text (blocked / empty / non-text response).
        """
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **config_kwargs,
        )

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=contents, config=config,
                )
            except errors.APIError as e:
                # e.code is the HTTP status. Retry only transient ones.
                if e.code in RETRYABLE and attempt < self.max_retries:
                    backoff = min(2 ** attempt + random.random(), 30.0)
                    print(f"[llm] {e.code} on attempt {attempt + 1}; "
                          f"retrying in {backoff:.1f}s")
                    time.sleep(backoff)
                    continue
                raise  # 400s and exhausted retries propagate

            # --- graceful failure: a blocked/empty reply has resp.text == None ---
            if resp.text is None:
                reason = (resp.candidates[0].finish_reason
                          if resp.candidates else "no candidates")
                raise RuntimeError(f"Model returned no text (finish_reason={reason}).")

            u = resp.usage_metadata
            self.usage.add(self.model,
                           u.prompt_token_count or 0,
                           u.candidates_token_count or 0)
            return resp.text

        # Unreachable in practice, but keeps the type checker happy.
        raise RuntimeError("retry loop exited unexpectedly")

    def generate_json(self, contents, schema, *, system: str | None = None,
                      temperature: float = 0.0, max_output_tokens: int = 1024):
        """Like generate(), but returns a SCHEMA-VALIDATED object (added in Lesson 6).

        `schema` is a Pydantic model class (returns an instance) or a dict JSON
        schema (returns a dict). Raises if the model can't produce valid JSON.
        """
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_schema=schema,
        )
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=contents, config=config,
                )
            except errors.APIError as e:
                if e.code in RETRYABLE and attempt < self.max_retries:
                    time.sleep(min(2 ** attempt + random.random(), 30.0))
                    continue
                raise
            u = resp.usage_metadata
            self.usage.add(self.model,
                           u.prompt_token_count or 0,
                           u.candidates_token_count or 0)
            if resp.parsed is None:
                raise RuntimeError("Model did not return schema-valid JSON.")
            return resp.parsed
        raise RuntimeError("retry loop exited unexpectedly")


if __name__ == "__main__":
    llm = LLM()
    answer = llm.generate(
        "In one sentence, what is cloud file sync?",
        temperature=0.0,
    )
    print(answer)
    print("---")
    print(llm.usage.report())
