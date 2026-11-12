"""Tests for the OpenRouter / OpenAI-compatible LLM client."""

import json
from http import HTTPStatus
from unittest.mock import MagicMock, patch

import pytest

from superpowers_agent.client import LLMClient, LLMClientError


class _FakeChoice:
    def __init__(self, content: str):
        msg = MagicMock()
        msg.content = content
        self.message = msg


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeOpenAI:
    """Minimal stand-in for the OpenAI client (no network call)."""

    def __init__(self, content: str):
        self.chat = MagicMock()
        self.chat.completions.create = MagicMock(return_value=_FakeResponse(content))


class _FailingOpenAI:
    """OpenAI stand-in that always fails with a server error."""

    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions.create = MagicMock(
            side_effect=Exception("server error")
        )


class TestLLMClient:
    def setup_method(self) -> None:
        self.client = LLMClient(api_key="test-key", model="test-model")

    def test_generate_response_returns_text(self) -> None:
        self.client._client = _FakeOpenAI("hello")
        result = self.client.generate_response([{"role": "user", "content": "hi"}])
        assert result == "hello"

    def test_generate_json_parses_model(self) -> None:
        from superpowers_agent.models import Plan

        self.client._client = _FakeOpenAI(
            json.dumps({
                "tasks": [{
                    "id": "task-1",
                    "title": "t1",
                    "description": "Do something",
                    "steps": ["step1"],
                }],
            })
        )
        plan = self.client.generate_json(
            [{"role": "user", "content": "plan"}],
            Plan,
        )
        assert isinstance(plan, Plan)
        assert plan.tasks[0].id == "task-1"

    def test_generate_json_falls_back_to_text_extraction(self) -> None:
        from superpowers_agent.models import Plan

        self.client._client = _FakeOpenAI(
            'Here is the plan: {"tasks": [{"id": "t1", "title": "t1", "description": "d", "steps": ["s"]}]} done.'
        )
        plan = self.client.generate_json(
            [{"role": "user", "content": "plan"}],
            Plan,
        )
        assert plan.tasks[0].id == "t1"

    def test_generate_response_retries_then_raises(self) -> None:
        self.client = LLMClient(api_key="test-key", model="test-model", max_retries=2)
        self.client._client = _FailingOpenAI()
        with patch("time.sleep", return_value=None):
            with pytest.raises(LLMClientError, match="LLM request failed"):
                self.client.generate_response([{"role": "user", "content": "hi"}])

    def test_default_base_url(self) -> None:
        new_client = LLMClient(api_key="test-key", model="test-model")
        assert "openrouter.ai" in str(new_client._client.base_url)

    def test_generate_with_mocked_requests(self) -> None:
        self.client._client = _FakeOpenAI("success")
        result = self.client.generate_response(
            [{"role": "user", "content": "Hello"}],
        )
        assert result == "success"

    def test_generate_handles_error_response(self) -> None:
        class _RaisingCompletions:
            def create(self, **kwargs):
                raise RuntimeError("API error")

        class _RaisingChat:
            completions = _RaisingCompletions()

        class _RaisingOpenAI:
            base_url = "https://openrouter.ai/api/v1"
            chat = _RaisingChat()

        client = LLMClient(api_key="test-key", model="test-model")
        client._client = _RaisingOpenAI()
        with pytest.raises(LLMClientError):
            client.generate_response(
                [{"role": "user", "content": "Hello"}],
            )
