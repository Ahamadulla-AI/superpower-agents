"""Core data models, state machine, and custom exceptions.

This module defines the pydantic models used throughout the SDK, the
``WorkflowState`` enum that drives the gate mechanics, and the custom
exception hierarchy.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


# ── Exceptions ───────────────────────────────────────────────────────────────


class WorkflowStateError(Exception):
    """Raised when a workflow operation is called at the wrong stage."""


class PlanParseError(Exception):
    """Raised when the planner's JSON output cannot be validated."""


class UnsafeActionError(Exception):
    """Raised when a file write or command would escape the sandbox."""


# ── Workflow state ───────────────────────────────────────────────────────────


class WorkflowState(str, enum.Enum):
    """Gated state machine for the agentic workflow."""

    BRAINSTORMING = "BRAINSTORMING"
    SPEC_APPROVED = "SPEC_APPROVED"
    PLAN_APPROVED = "PLAN_APPROVED"
    EXECUTING = "EXECUTING"
    DONE = "DONE"


# ── Pydantic models (used by the LLM via generate_json) ─────────────────────


class Task(BaseModel):
    """An atomic, test-first development task produced by the planner."""

    id: str
    title: str
    description: str
    files: list[str] = Field(default_factory=list)
    test_first: bool = True
    steps: list[str]
    depends_on: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    """The implementation plan emitted by the planner."""

    tasks: list[Task]


class FileWrite(BaseModel):
    """A single file-write action returned by a subagent."""

    path: str
    content: str


class Command(BaseModel):
    """A shell command returned by a subagent."""

    cmd: str
    timeout_s: int = 60


class TaskResult(BaseModel):
    """The structured result of one subagent execution."""

    task_id: str
    file_writes: list[FileWrite] = Field(default_factory=list)
    commands: list[Command] = Field(default_factory=list)
    notes: str = ""
