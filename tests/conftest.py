"""Pytest configuration: fixtures used by multiple test modules."""

import json
from pathlib import Path
from typing import Any, Type, TypeVar

import pytest

from superpowers_agent.client import LLMClient
from superpowers_agent.models import TaskResult, Plan

T = TypeVar("T")


class MockLLMClient(LLMClient):
    """A mock LLM client for testing without real API calls."""

    def __init__(self) -> None:
        self.last_messages: list[dict[str, Any]] = []
        self._text_response: str = "Mock response"
        self._json_response: dict[str, Any] | None = None
        self._error_on_next: Exception | None = None
        self._json_error_on_next: Exception | None = None

    def set_text_response(self, text: str) -> None:
        """Set the next text response."""
        self._text_response = text

    def set_json_response(self, data: dict[str, Any]) -> None:
        """Set the next JSON response (will be parsed into the given model)."""
        self._json_response = data

    def set_error(self, exc: Exception) -> None:
        """Raise the given exception on the next call to generate()."""
        self._error_on_next = exc

    def set_json_error(self, exc: Exception) -> None:
        """Raise the given exception on the next call to generate_json()."""
        self._json_error_on_next = exc
        self._error_on_next = None  # don't trigger on generate()

    def generate(self, messages: list[dict]) -> str:
        self.last_messages = messages
        if self._error_on_next:
            exc = self._error_on_next
            self._error_on_next = None
            raise exc
        return self._text_response

    def generate_json(self, messages: list[dict], model: type[T]) -> T:
        self.last_messages = messages
        if self._json_error_on_next:
            exc = self._json_error_on_next
            self._json_error_on_next = None
            raise exc
        if self._error_on_next:
            exc = self._error_on_next
            self._error_on_next = None
            raise exc
        if self._json_response is not None:
            return model(**self._json_response)
        raise RuntimeError("No JSON response set on MockLLMClient")

    def _build_messages(self, messages: list[dict]) -> list[dict]:
        return messages


@pytest.fixture
def mock_client() -> MockLLMClient:
    """Return a fresh MockLLMClient."""
    return MockLLMClient()


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Return a temporary project root."""
    return tmp_path


@pytest.fixture
def sample_task() -> TaskResult:
    """Return a sample TaskResult for testing."""
    return TaskResult(
        task_id="task-1",
        notes="Done.",
    )


@pytest.fixture
def sample_plan() -> dict[str, Any]:
    """Return a minimal plan dict for mocking."""
    return {
        "tasks": [
            {
                "id": "task-1",
                "title": "Set up project structure",
                "description": "Create initial project files",
                "files": ["src/__init__.py"],
                "test_first": True,
                "steps": ["Create __init__.py"],
                "depends_on": [],
            }
        ]
    }
