"""Agentic workflow: the gated state machine.

The workflow progresses through states:

    BRAINSTORMING -> SPEC_APPROVED -> PLAN_APPROVED -> EXECUTING -> DONE

Transitions are gated: a human must approve each stage before the next
one begins.  The ``AgenticWorkflow`` class orchestrates the LLM calls
and delegates task execution to ``SubAgent``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .client import LLMClient
from .models import (
    Plan,
    PlanParseError,
    Task,
    TaskResult,
    WorkflowState,
    WorkflowStateError,
)
from .prompts import (
    BRAINSTORMING_SYSTEM,
    BRAINSTORMING_USER,
    PLAN_APPROVAL_SYSTEM,
    PLAN_SYSTEM,
    SPEC_APPROVAL_SYSTEM,
    SPEC_SYSTEM,
    SPEC_USER,
)
from .subagent import SubAgent

logger = logging.getLogger(__name__)


class AgenticWorkflow:
    """Orchestrate the gated agentic workflow."""

    def __init__(
        self,
        client: LLMClient,
        project_root: Path,
        user_requirement: str,
    ) -> None:
        self.client = client
        self.project_root = project_root
        self.user_requirement = user_requirement

        self.state = WorkflowState.BRAINSTORMING
        self.spec: str = ""
        self.plan: Plan = Plan(tasks=[])
        self.results: list[TaskResult] = []
        self._brainstorm_notes: str = ""

    # ── Public API ────────────────────────────────────────────────────────

    def brainstorm(self) -> str:
        """Generate the brainstorming output (asks clarifying questions).

        Must be called in ``BRAINSTORMING`` state.

        Returns:
            The LLM's brainstorming text.
        """
        self._require_state(WorkflowState.BRAINSTORMING)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": BRAINSTORMING_SYSTEM},
            {"role": "user", "content": BRAINSTORMING_USER.format(
                user_requirement=self.user_requirement,
            )},
        ]
        response = self.client.generate(messages)
        self._brainstorm_notes = response
        return response

    def approve_spec(self, brainstorm_notes: str = "") -> str:
        """Generate the specification from brainstorm notes.

        Must be called in ``BRAINSTORMING`` state.  Advances to
        ``SPEC_APPROVED``.

        Args:
            brainstorm_notes: Optional override of the brainstorm notes
                (e.g. the human's consolidated input after answering
                the clarifying questions).

        Returns:
            The produced specification text.
        """
        self._require_state(WorkflowState.BRAINSTORMING)

        notes = brainstorm_notes or self._brainstorm_notes
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SPEC_SYSTEM},
            {"role": "user", "content": SPEC_USER.format(brainstorm_notes=notes)},
        ]
        self.spec = self.client.generate(messages)
        self.state = WorkflowState.SPEC_APPROVED
        logger.info("Spec produced (%d chars)", len(self.spec))
        return self.spec

    def generate_plan(self) -> Plan:
        """Generate the implementation plan from the spec.

        Must be called in ``SPEC_APPROVED`` state.  Advances to
        ``PLAN_APPROVED``.

        Returns:
            The validated ``Plan``.
        """
        self._require_state(WorkflowState.SPEC_APPROVED)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": f"Create an implementation plan for: {self.spec}"},
        ]
        try:
            self.plan = self.client.generate_json(messages, Plan)
        except Exception as exc:
            raise PlanParseError(f"Failed to parse plan JSON: {exc}") from exc

        self.state = WorkflowState.PLAN_APPROVED
        logger.info("Plan generated: %d tasks", len(self.plan.tasks))
        return self.plan

    def execute_plan(self) -> list[TaskResult]:
        """Execute all tasks in the plan, in dependency order.

        Must be called in ``PLAN_APPROVED`` state.  Advances to
        ``EXECUTING``, then ``DONE`` when all tasks succeed.

        Returns:
            A list of ``TaskResult`` objects, one per completed task.
        """
        self._require_state(WorkflowState.PLAN_APPROVED)
        self.state = WorkflowState.EXECUTING

        subagent = SubAgent(self.client, self.project_root)
        ordered = self._topological_sort(self.plan.tasks)

        for task in ordered:
            logger.info("Executing task: %s", task.title)
            result = subagent.run(task)
            self.results.append(result)

        self.state = WorkflowState.DONE
        return self.results

    def get_current_state(self) -> WorkflowState:
        """Return the current workflow state."""
        return self.state

    # ── Private helpers ───────────────────────────────────────────────────

    def _require_state(self, expected: WorkflowState) -> None:
        """Raise if the workflow is not in the expected state."""
        if self.state != expected:
            raise WorkflowStateError(
                f"Expected state {expected}, but currently {self.state}"
            )

    @staticmethod
    def _topological_sort(tasks: list[Task]) -> list[Task]:
        """Sort tasks respecting ``depends_on``.

        A simple DFS-based topological sort.  Tasks with no dependencies
        come first.
        """
        task_map: dict[str, Task] = {t.id: t for t in tasks}
        visited: set[str] = set()
        order: list[Task] = []

        def visit(task_id: str) -> None:
            if task_id in visited:
                return
            visited.add(task_id)
            task = task_map[task_id]
            for dep_id in task.depends_on:
                if dep_id in task_map:
                    visit(dep_id)
            order.append(task)

        for t in tasks:
            visit(t.id)

        return order
