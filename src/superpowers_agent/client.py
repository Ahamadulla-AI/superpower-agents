"""LLM client wrapping any OpenAI-compatible endpoint.

This module provides :class:`LLMClient`, a thin, robust wrapper around the
``openai`` package that works with any OpenAI-compatible endpoint (defaulting
to OpenRouter). It adds retry-with-backoff, JSON-mode with a text-extraction
fallback, and pydantic validation with retry.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

#: Error types that are safe to retry with backoff.
_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIStatusError,
)

#: Matches the first balanced JSON object in a block of text.
_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


class LLMClientError(Exception):
    """Raised when the LLM client fails after exhausting its retries."""


def auto_detect() -> tuple[str, str, str]:
    """Auto-detect LLM provider from environment variables.

    Scans for common API keys in this order:
    1. OPENAI_API_KEY  -> base_url: https://api.openai.com/v1,  model: gpt-4o
    2. OPENROUTER_API_KEY -> base_url: https://openrouter.ai/api/v1, model: openai/gpt-4o
    3. ANTHROPIC_API_KEY -> not supported (OpenAI-compatible only)

    Also reads ``.env`` files in the current directory and the project root.

    Returns:
        A tuple of ``(api_key, base_url, model)``.

    Raises:
        LLMClientError: If no API key is found.
    """
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env", override=False)
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

    providers: list[tuple[str, str, str, str]] = [
        # env_var, api_key, base_url, default_model
        ("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"), "https://api.openai.com/v1", "gpt-4o"),
        ("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1", "openai/gpt-4o"),
        ("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"), "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-pro"),
        ("AZURE_OPENAI_API_KEY", os.getenv("AZURE_OPENAI_API_KEY"), os.getenv("AZURE_OPENAI_BASE_URL", ""), "gpt-4o"),
    ]

    for name, key, url, model in providers:
        if key and url:
            logger.info("Auto-detected LLM provider: %s", name)
            return key, url, model

    raise LLMClientError(
        "No LLM API key found. Set one of: "
        "OPENAI_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY, or "
        "AZURE_OPENAI_API_KEY (+ AZURE_OPENAI_BASE_URL). "
        "You can put your key in a .env file in your project root."
    )


class LLMClient:
    """A retrying client for any OpenAI-compatible chat-completions endpoint.

    Args:
        api_key: The API key. Falls back to auto-detection from environment
            variables (OPENAI_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY,
            AZURE_OPENAI_API_KEY).
        base_url: The endpoint base URL. Auto-detected if api_key was
            auto-detected.
        model: The model identifier to use. Auto-detected if api_key was
            auto-detected.
        timeout: Request timeout in seconds.
        max_retries: Number of retries after the first attempt.
        referer: Optional ``HTTP-Referer`` header value (OpenRouter attribution).
        app_title: Optional ``X-Title`` header value (OpenRouter attribution).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        referer: str | None = None,
        app_title: str | None = None,
    ) -> None:
        self.max_retries = max_retries
        self.timeout = timeout

        if api_key is None:
            api_key, base_url, model = auto_detect()
        if not base_url:
            base_url = "https://openrouter.ai/api/v1"
        if not model:
            raise ValueError("model must be provided (or set an API key to auto-detect)")
        self.model = model
        self.base_url = base_url
        self._api_key = api_key

        headers: dict[str, str] = {}
        if referer:
            headers["HTTP-Referer"] = referer
        if app_title:
            headers["X-Title"] = app_title
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,  # retries handled manually below
            default_headers=headers or None,
        )

    def generate_response(
        self, messages: list[dict[str, Any]], json_mode: bool = False
    ) -> str:
        """Generate a chat completion and return the text content.

        Retries on rate limits, timeouts, connection errors, and 5xx errors
        with exponential backoff. Raises :class:`LLMClientError` after the
        final attempt.

        Args:
            messages: The conversation messages.
            json_mode: Whether to request a JSON object response.

        Returns:
            The assistant's text content.

        Raises:
            LLMClientError: If all attempts fail.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except _RETRYABLE_ERRORS as exc:
                last_error = exc
                if attempt < self.max_retries:
                    delay = 2**attempt
                    logger.warning(
                        "LLM request failed (attempt %d/%d): %s; retrying in %.1fs",
                        attempt + 1,
                        self.max_retries + 1,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            except Exception as exc:  # noqa: BLE001 - surface as LLMClientError
                raise LLMClientError(f"LLM request failed: {exc}") from exc

        raise LLMClientError(
            f"LLM request failed after {self.max_retries + 1} attempts: {last_error}"
        )

    def generate_json(
        self, messages: list[dict[str, Any]], schema: type[T]
    ) -> T:
        """Generate a response and parse it into a pydantic model.

        Tries ``response_format`` JSON mode first. If the provider rejects it,
        falls back to prompting for JSON and extracting it from the text.
        Validates with pydantic and, on failure, retries up to ``max_retries``
        times, feeding the validation error back to the model.

        Args:
            messages: The conversation messages.
            schema: The pydantic model to parse the response into.

        Returns:
            An instance of ``schema``.

        Raises:
            LLMClientError: If parsing fails after all retries.
        """
        working_messages = list(messages)
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                text = self._try_json_mode(working_messages)
            except Exception as exc:  # noqa: BLE001 - provider rejected JSON mode
                logger.info("JSON mode rejected (%s); falling back to text", exc)
                text = self._try_text_mode(working_messages)

            try:
                data = _extract_json_object(text)
                return schema.model_validate(data)
            except (ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning(
                    "JSON validation failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                if attempt < self.max_retries:
                    working_messages = working_messages + [
                        {
                            "role": "assistant",
                            "content": text,
                        },
                        {
                            "role": "user",
                            "content": (
                                "Your previous response was not valid for the "
                                f"schema. Validation error: {exc}. "
                                "Please respond with only valid JSON matching "
                                "the schema."
                            ),
                        },
                    ]

        raise LLMClientError(
            f"Could not parse LLM response into {schema.__name__} after "
            f"{self.max_retries + 1} attempts: {last_error}"
        )

    def _try_json_mode(self, messages: list[dict[str, Any]]) -> str:
        """Attempt a JSON-mode completion."""
        return self.generate_response(messages, json_mode=True)

    def _try_text_mode(self, messages: list[dict[str, Any]]) -> str:
        """Attempt a plain-text completion that asks for JSON."""
        prompt_messages = list(messages) + [
            {
                "role": "user",
                "content": (
                    "Respond with only valid JSON. Do not include any other "
                    "text, markdown, or explanation."
                ),
            }
        ]
        return self.generate_response(prompt_messages, json_mode=False)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from ``text``.

    Tries to parse the whole string first, then falls back to a
    brace-counting scan to find a balanced ``{...}`` block.

    Args:
        text: The raw model output.

    Returns:
        The parsed JSON object.

    Raises:
        json.JSONDecodeError: If no JSON object can be extracted.
    """
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Brace-counting scan: find first '{', then count balanced braces
    start = stripped.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found in text", text, 0)

    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(stripped[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass

    raise json.JSONDecodeError("No JSON object found in text", text, 0)