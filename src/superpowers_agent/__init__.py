"""Superpowers Agent SDK -- public API.

A gated, subagent-driven agentic framework for AI-assisted software
development.
"""

from .client import LLMClient, LLMClientError, auto_detect
from .models import (
    Command,
    FileWrite,
    Plan,
    PlanParseError,
    Task,
    TaskResult,
    UnsafeActionError,
    WorkflowState,
    WorkflowStateError,
)
from .workflow import AgenticWorkflow
from .subagent import SubAgent

__all__ = [
    "AgenticWorkflow",
    "LLMClient",
    "LLMClientError",
    "auto_detect",
    "Plan",
    "PlanParseError",
    "SubAgent",
    "Task",
    "TaskResult",
    "UnsafeActionError",
    "WorkflowState",
    "WorkflowStateError",
]


