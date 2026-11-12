"""Tests for workflow gate mechanics (state transitions)."""

from pathlib import Path

import pytest

from superpowers_agent.client import LLMClient
from superpowers_agent.models import PlanParseError, WorkflowState, WorkflowStateError
from superpowers_agent.workflow import AgenticWorkflow

# Imported for type annotation only; the fixture is auto-discovered by pytest
from conftest import MockLLMClient


def test_start_in_brainstorming(mock_client: MockLLMClient, project_root: Path) -> None:
    wf = AgenticWorkflow(mock_client, project_root, "Build a todo app")
    assert wf.get_current_state() == WorkflowState.BRAINSTORMING


def test_brainstorm(mock_client: MockLLMClient, project_root: Path) -> None:
    mock_client.set_text_response("What framework? What language?")
    wf = AgenticWorkflow(mock_client, project_root, "Build a todo app")
    result = wf.brainstorm()
    assert "What framework" in result
    assert wf.get_current_state() == WorkflowState.BRAINSTORMING


def test_brainstorm_wrong_state(mock_client: MockLLMClient, project_root: Path) -> None:
    mock_client.set_text_response("ok")
    mock_client.set_json_response({"tasks": []})
    wf = AgenticWorkflow(mock_client, project_root, "Build a todo app")
    wf.approve_spec("Notes here")
    wf.generate_plan()
    with pytest.raises(WorkflowStateError, match=r"Expected state WorkflowState\.BRAINSTORMING"):
        wf.brainstorm()


def test_approve_spec_transitions_to_spec_approved(
    mock_client: MockLLMClient, project_root: Path
) -> None:
    mock_client.set_text_response("What framework?")
    mock_client.set_text_response("Here is the spec.")
    wf = AgenticWorkflow(mock_client, project_root, "Build a todo app")
    wf.brainstorm()
    spec = wf.approve_spec("User wants React")
    assert "spec" in spec.lower()
    assert wf.get_current_state() == WorkflowState.SPEC_APPROVED


def test_generate_plan_transitions_to_plan_approved(
    mock_client: MockLLMClient, project_root: Path
) -> None:
    mock_client.set_text_response("What framework?")
    mock_client.set_text_response("Here is the spec.")
    mock_client.set_json_response({
        "tasks": [
            {
                "id": "t1",
                "title": "Task one",
                "description": "Do something",
                "steps": ["Do it"],
            }
        ]
    })
    wf = AgenticWorkflow(mock_client, project_root, "Build a todo app")
    wf.brainstorm()
    wf.approve_spec("User wants React")
    plan = wf.generate_plan()
    assert len(plan.tasks) == 1
    assert wf.get_current_state() == WorkflowState.PLAN_APPROVED


def test_generate_plan_bad_json(mock_client: MockLLMClient, project_root: Path) -> None:
    mock_client.set_text_response("What framework?")
    mock_client.set_text_response("Here is the spec.")
    mock_client.set_json_error(ValueError("invalid json"))
    wf = AgenticWorkflow(mock_client, project_root, "Build a todo app")
    wf.brainstorm()
    wf.approve_spec("User wants React")
    with pytest.raises(PlanParseError):
        wf.generate_plan()


def test_cannot_generate_plan_before_spec(
    mock_client: MockLLMClient, project_root: Path
) -> None:
    wf = AgenticWorkflow(mock_client, project_root, "Build a todo app")
    with pytest.raises(WorkflowStateError, match=r"Expected state WorkflowState\.SPEC_APPROVED"):
        wf.generate_plan()


def test_cannot_execute_before_plan(
    mock_client: MockLLMClient, project_root: Path
) -> None:
    wf = AgenticWorkflow(mock_client, project_root, "Build a todo app")
    with pytest.raises(WorkflowStateError, match=r"Expected state WorkflowState\.PLAN_APPROVED"):
        wf.execute_plan()


def test_topological_sort_order() -> None:
    from superpowers_agent.models import Task
    from superpowers_agent.workflow import AgenticWorkflow

    tasks = [
        Task(id="a", title="A", description="", steps=[]),
        Task(id="b", title="B", description="", steps=[], depends_on=["a"]),
        Task(id="c", title="C", description="", steps=[], depends_on=["a", "b"]),
    ]
    order = AgenticWorkflow._topological_sort(tasks)
    ids = [t.id for t in order]
    assert ids.index("a") < ids.index("b")
    assert ids.index("b") < ids.index("c")
