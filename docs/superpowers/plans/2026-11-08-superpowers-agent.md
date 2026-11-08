# superpowers_agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an installable Python SDK, `superpowers_agent`, that reimplements the `obra/superpowers` methodology as a gated state machine with a robust LLM client, safe sandbox, and subagent-driven execution.

**Architecture:** A `src/`-layout package with a `WorkflowState` enum enforcing gates in code. `LLMClient` wraps any OpenAI-compatible endpoint (default OpenRouter) with retry/backoff and JSON-mode fallback. `AgenticWorkflow` holds state + conversation history and drives brainstorm → approve → plan → approve → execute. `SubAgent` executes each atomic task, applying results through a `sandbox.py` that validates paths and allowlists commands.

**Tech Stack:** Python >=3.10, `openai>=1.0`, `pydantic>=2`, `python-dotenv`, hatchling build, pytest for tests.

**Spec:** [docs/superpowers/specs/2026-11-08-superpowers-agent-design.md](../specs/2026-11-08-superpowers-agent-design.md)

## Global Constraints

- Python >= 3.10.
- Dependencies: `openai>=1.0`, `pydantic>=2`, `python-dotenv`. Dev: `pytest`.
- Build system: hatchling with `src/` layout.
- Package name `superpowers_agent`.
- No hardcoded model or endpoint; default base_url `https://openrouter.ai/api/v1` (plain string, no Markdown).
- No bare `print` in library code — use `logging`.
- Specific exceptions, never bare `except`.
- All code typed with type hints and docstrings.
- Tests use a mock client and make no network calls.

## Review Focus

- **Gate ordering:** calling `create_plan` before `approve_spec`, or `execute_plan` before `approve_plan`, must raise `WorkflowStateError` — pinned in Task 2.
- **Plan dependency validation:** a plan referencing a nonexistent task id, or containing a dependency cycle, must be rejected with `PlanParseError` — pinned in Task 3.
- **Path escape:** a file path with `..` or a symlink resolving outside `workspace_dir` must raise `UnsafeActionError` — pinned in Task 5.
- **Command allowlist:** a command not in the allowlist must raise `UnsafeActionError`; allowed commands run with `shell=False` — pinned in Task 5.
- **JSON fallback:** when the provider rejects `response_format`, `generate_json` must still return a valid pydantic model by extracting JSON from text — pinned in Task 1.

---
### Task 1: LLMClient with retry, JSON mode, and fallback

**Files:**
- Create: `src/superpowers_agent/client.py`
- Test: `tests/test_client.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `class LLMClientError(Exception)`
  - `class LLMClient` with `__init__(api_key: str | None = None, base_url: str = "https://openrouter.ai/api/v1", model: str, timeout: float = 60.0, max_retries: int = 3, referer: str | None = None, app_title: str | None = None)`
  - `generate_response(messages: list[dict], json_mode: bool = False) -> str`
  - `generate_json(messages: list[dict], schema: type[BaseModel]) -> BaseModel`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_client.py
import pytest
from pydantic import BaseModel
from superpowers_agent.client import LLMClient, LLMClientError

class Dummy(BaseModel):
    name: str

def test_generate_response_returns_text():
    client = LLMClient(model="test-model")
    client._client = _FakeOpenAI("hello")
    assert client.generate_response([{"role": "user", "content": "hi"}]) == "hello"

def test_generate_json_parses_model():
    client = LLMClient(model="test-model")
    client._client = _FakeOpenAI('{"name": "x"}')
    result = client.generate_json([{"role": "user", "content": "hi"}], Dummy)
    assert result == Dummy(name="x")

def test_generate_json_falls_back_to_text_extraction():
    client = LLMClient(model="test-model")
    client._client = _FakeOpenAI('Here you go: {"name": "x"}')
    result = client.generate_json([{"role": "user", "content": "hi"}], Dummy)
    assert result == Dummy(name="x")

def test_generate_response_retries_then_raises():
    client = LLMClient(model="test-model", max_retries=2)
    client._client = _FailingOpenAI()
    with pytest.raises(LLMClientError):
        client.generate_response([{"role": "user", "content": "hi"}])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'superpowers_agent'`

- [ ] **Step 3: Implement `LLMClient` in `src/superpowers_agent/client.py`**

Signature as in Interfaces. Use the `openai` package's `OpenAI` client with `base_url`, `api_key`, `timeout`, `max_retries=0` (retries handled manually). `generate_response` loops up to `max_retries`, catching rate-limit/timeout/5xx errors, sleeping with exponential backoff, then raising `LLMClientError`. `generate_json` first tries `response_format={"type": "json_object"}`; on exception falls back to prompting for JSON and extracting the first `{...}` block from the text; validates with pydantic and retries up to N times feeding the validation error back.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_client.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_client.py src/superpowers_agent/client.py
git commit -m "feat: add LLMClient with retry and JSON fallback"
```

---
### Task 2: Models, WorkflowState, and exceptions

**Files:**
- Create: `src/superpowers_agent/models.py`
- Test: `tests/test_workflow_gates.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class WorkflowState(Enum)`: `BRAINSTORMING`, `SPEC_APPROVED`, `PLAN_APPROVED`, `EXECUTING`, `DONE`
  - `class WorkflowStateError(Exception)`
  - `class PlanParseError(Exception)`
  - `class UnsafeActionError(Exception)`
  - `class Task(BaseModel)`: `id: str`, `title: str`, `description: str`, `files: list[str]`, `test_first: bool`, `steps: list[str]`, `depends_on: list[str] = []`
  - `class Plan(BaseModel)`: `tasks: list[Task]`
  - `class FileWrite(BaseModel)`: `path: str`, `content: str`
  - `class Command(BaseModel)`: `cmd: str`, `timeout_s: float = 60.0`
  - `class TaskResult(BaseModel)`: `task_id: str`, `file_writes: list[FileWrite] = []`, `commands: list[Command] = []`, `notes: str = ""`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_workflow_gates.py
import pytest
from superpowers_agent.models import WorkflowState, WorkflowStateError

def test_state_order():
    assert list(WorkflowState) == [
        WorkflowState.BRAINSTORMING,
        WorkflowState.SPEC_APPROVED,
        WorkflowState.PLAN_APPROVED,
        WorkflowState.EXECUTING,
        WorkflowState.DONE,
    ]

def test_workflow_state_error_is_exception():
    assert issubclass(WorkflowStateError, Exception)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow_gates.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `models.py`**

Define the enum and exceptions, and the pydantic models with the exact field names/types above. Use `pydantic.BaseModel`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_workflow_gates.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_workflow_gates.py src/superpowers_agent/models.py
git commit -m "feat: add models, WorkflowState, and exceptions"
```

---
### Task 3: Plan parsing and dependency validation

**Files:**
- Create: `src/superpowers_agent/planner.py`
- Test: `tests/test_plan_parsing.py`

**Interfaces:**
- Consumes: `Plan`, `Task`, `PlanParseError` from Task 2.
- Produces:
  - `def validate_plan(plan: Plan) -> None` — raises `PlanParseError` on unknown dependency id or cycle.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plan_parsing.py
import pytest
from superpowers_agent.models import Plan, Task, PlanParseError
from superpowers_agent.planner import validate_plan

def test_valid_plan_passes():
    plan = Plan(tasks=[
        Task(id="a", title="A", description="", files=[], test_first=True, steps=[], depends_on=[]),
        Task(id="b", title="B", description="", files=[], test_first=True, steps=[], depends_on=["a"]),
    ])
    validate_plan(plan)  # no raise

def test_unknown_dependency_rejected():
    plan = Plan(tasks=[Task(id="a", title="A", description="", files=[], test_first=True, steps=[], depends_on=["ghost"])])
    with pytest.raises(PlanParseError):
        validate_plan(plan)

def test_cycle_rejected():
    plan = Plan(tasks=[
        Task(id="a", title="A", description="", files=[], test_first=True, steps=[], depends_on=["b"]),
        Task(id="b", title="B", description="", files=[], test_first=True, steps=[], depends_on=["a"]),
    ])
    with pytest.raises(PlanParseError):
        validate_plan(plan)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_plan_parsing.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `validate_plan` in `src/superpowers_agent/planner.py`**

Check every `depends_on` id exists among task ids; raise `PlanParseError` otherwise. Detect cycles via DFS/Kahn's algorithm; raise `PlanParseError` on cycle.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_plan_parsing.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_plan_parsing.py src/superpowers_agent/planner.py
git commit -m "feat: add plan dependency validation"
```

---
### Task 4: Sandbox path and command safety

**Files:**
- Create: `src/superpowers_agent/sandbox.py`
- Test: `tests/test_sandbox.py`

**Interfaces:**
- Consumes: `UnsafeActionError` from Task 2.
- Produces:
  - `def resolve_safe_path(workspace_dir: Path, rel_path: str) -> Path` — raises `UnsafeActionError` on escape.
  - `def run_command(workspace_dir: Path, cmd: str, timeout_s: float = 60.0, allowlist: Sequence[str] = ("pytest", "python", "pip", "git", "ls")) -> str` — raises `UnsafeActionError` on non-allowlisted command; returns captured stdout.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sandbox.py
import pytest
from pathlib import Path
from superpowers_agent.sandbox import resolve_safe_path, run_command
from superpowers_agent.models import UnsafeActionError

def test_resolve_safe_path_ok(tmp_path):
    p = resolve_safe_path(tmp_path, "sub/file.txt")
    assert p == (tmp_path / "sub" / "file.txt").resolve()

def test_resolve_safe_path_rejects_dotdot(tmp_path):
    with pytest.raises(UnsafeActionError):
        resolve_safe_path(tmp_path, "../evil.txt")

def test_resolve_safe_path_rejects_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(UnsafeActionError):
        resolve_safe_path(tmp_path, "link/evil.txt")

def test_run_command_allowlisted(tmp_path):
    out = run_command(tmp_path, "ls")
    assert isinstance(out, str)

def test_run_command_rejects_unknown(tmp_path):
    with pytest.raises(UnsafeActionError):
        run_command(tmp_path, "rm -rf /")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sandbox.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `sandbox.py`**

`resolve_safe_path`: resolve the joined path, then verify it is within the resolved `workspace_dir` (using `Path.resolve()` to follow symlinks); raise `UnsafeActionError` otherwise. `run_command`: split the command, check the first token against the allowlist, run with `subprocess.run(..., shell=False, cwd=workspace_dir, timeout=timeout_s, capture_output=True, text=True)`; raise `UnsafeActionError` for non-allowlisted commands.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sandbox.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_sandbox.py src/superpowers_agent/sandbox.py
git commit -m "feat: add sandbox path and command safety"
```

---
### Task 5: SubAgent execution

**Files:**
- Create: `src/superpowers_agent/subagent.py`
- Test: `tests/test_subagent.py`

**Interfaces:**
- Consumes: `TaskResult`, `FileWrite`, `Command` from Task 2; `resolve_safe_path`, `run_command` from Task 4; `LLMClient` from Task 1.
- Produces:
  - `class SubAgent` with `__init__(client: LLMClient, workspace_dir: Path, dry_run: bool = True)`
  - `run(task: Task, context: str) -> TaskResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_subagent.py
import pytest
from pathlib import Path
from superpowers_agent.subagent import SubAgent
from superpowers_agent.models import Task, TaskResult, FileWrite, Command

class FakeClient:
    def generate_json(self, messages, schema):
        return TaskResult(task_id="t1", file_writes=[FileWrite(path="a.txt", content="hi")], commands=[Command(cmd="ls")])

def test_subagent_dry_run_writes_nothing(tmp_path):
    sub = SubAgent(FakeClient(), tmp_path, dry_run=True)
    task = Task(id="t1", title="T", description="", files=["a.txt"], test_first=True, steps=[], depends_on=[])
    result = sub.run(task, "context")
    assert result.task_id == "t1"
    assert not (tmp_path / "a.txt").exists()

def test_subagent_applies_writes(tmp_path):
    sub = SubAgent(FakeClient(), tmp_path, dry_run=False)
    task = Task(id="t1", title="T", description="", files=["a.txt"], test_first=True, steps=[], depends_on=[])
    sub.run(task, "context")
    assert (tmp_path / "a.txt").read_text() == "hi"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_subagent.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `SubAgent` in `src/superpowers_agent/subagent.py`**

`run` builds messages from `SUBAGENT_PROMPT` (Task 6) plus task/context, calls `client.generate_json(TaskResult)`, and unless `dry_run`, writes each `FileWrite` via `resolve_safe_path` and runs each `Command` via `run_command`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_subagent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_subagent.py src/superpowers_agent/subagent.py
git commit -m "feat: add SubAgent execution"
```

---
### Task 6: Prompts

**Files:**
- Create: `src/superpowers_agent/prompts.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `BRAINSTORMING_PROMPT: str`
  - `PLANNING_PROMPT: str`
  - `SUBAGENT_PROMPT: str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompts.py
from superpowers_agent.prompts import BRAINSTORMING_PROMPT, PLANNING_PROMPT, SUBAGENT_PROMPT

def test_prompts_exist():
    assert BRAINSTORMING_PROMPT.strip()
    assert PLANNING_PROMPT.strip()
    assert SUBAGENT_PROMPT.strip()

def test_planning_prompt_mentions_tdd():
    assert "test" in PLANNING_PROMPT.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `prompts.py`**

Write three distilled prompt strings. `BRAINSTORMING_PROMPT`: LLM acts as product manager, asks one or two clarifying questions per turn, never writes code, emits a final design spec in fixed Markdown structure (goals, non-goals, requirements, constraints, acceptance criteria) when the human signals readiness. `PLANNING_PROMPT`: takes approved spec, outputs only JSON matching the `Plan` schema, written for a junior engineer with no project context, enforces YAGNI/DRY/TDD. `SUBAGENT_PROMPT`: executes only the assigned task, makes no changes outside listed files, returns only JSON matching `TaskResult`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_prompts.py src/superpowers_agent/prompts.py
git commit -m "feat: add workflow prompts"
```

---
### Task 7: AgenticWorkflow state machine

**Files:**
- Create: `src/superpowers_agent/workflow.py`
- Test: `tests/test_workflow_gates.py` (extend)

**Interfaces:**
- Consumes: `LLMClient` (Task 1), `WorkflowState`, `WorkflowStateError`, `Plan`, `TaskResult` (Task 2), `validate_plan` (Task 3), `SubAgent` (Task 5), prompts (Task 6).
- Produces:
  - `class AgenticWorkflow` with `__init__(client: LLMClient, workspace_dir: Path | None = None)`
  - `brainstorm(user_intent: str) -> str`
  - `brainstorm_reply(answer: str) -> str`
  - `approve_spec(edited_spec: str | None = None) -> None`
  - `create_plan() -> Plan`
  - `approve_plan() -> None`
  - `execute_plan(dry_run: bool = True) -> list[TaskResult]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_workflow_gates.py (extend)
import pytest
from superpowers_agent.workflow import AgenticWorkflow
from superpowers_agent.models import WorkflowStateError, Plan, Task

class FakeClient:
    def generate_response(self, messages, json_mode=False):
        return "question?"
    def generate_json(self, messages, schema):
        return Plan(tasks=[Task(id="a", title="A", description="", files=[], test_first=True, steps=[], depends_on=[])])

def test_create_plan_before_spec_raises():
    wf = AgenticWorkflow(FakeClient())
    with pytest.raises(WorkflowStateError):
        wf.create_plan()

def test_execute_before_plan_raises():
    wf = AgenticWorkflow(FakeClient())
    wf.brainstorm("build x")
    wf.approve_spec("spec")
    with pytest.raises(WorkflowStateError):
        wf.execute_plan()

def test_full_flow_dry_run():
    wf = AgenticWorkflow(FakeClient())
    wf.brainstorm("build x")
    wf.approve_spec("spec")
    plan = wf.create_plan()
    wf.approve_plan()
    results = wf.execute_plan(dry_run=True)
    assert len(results) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow_gates.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `AgenticWorkflow` in `src/superpowers_agent/workflow.py`**

Hold `state` (default `BRAINSTORMING`) and `history: list[dict]`. `brainstorm` seeds history with `BRAINSTORMING_PROMPT` + user intent, returns `generate_response`. `brainstorm_reply` appends answer, returns response. `approve_spec` sets state to `SPEC_APPROVED` (store edited spec or last response). `create_plan` raises `WorkflowStateError` unless `SPEC_APPROVED`; calls `generate_json(Plan)`, `validate_plan`, stores plan, returns it (state stays `SPEC_APPROVED`). `approve_plan` sets state to `PLAN_APPROVED`. `execute_plan` raises unless `PLAN_APPROVED`; sets `EXECUTING`, runs tasks in dependency order (topological), one `SubAgent` per task with spec+plan+prior results as context; on failure stops and reports; sets `DONE`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_workflow_gates.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_workflow_gates.py src/superpowers_agent/workflow.py
git commit -m "feat: add AgenticWorkflow state machine"
```

---
### Task 8: Package scaffolding, __init__, README, example

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/superpowers_agent/__init__.py`
- Create: `example_usage.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: all modules.
- Produces:
  - `__init__.py` exports `LLMClient`, `AgenticWorkflow`, `SubAgent`, `WorkflowState`, `WorkflowStateError`, `PlanParseError`, `UnsafeActionError`, `Plan`, `Task`, `TaskResult`, `FileWrite`, `Command`, and `__version__`.
  - `tests/conftest.py` provides a `MockLLMClient` fixture returning scripted responses.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_package.py
from superpowers_agent import LLMClient, AgenticWorkflow, WorkflowState, __version__

def test_public_api_exports():
    assert LLMClient and AgenticWorkflow and WorkflowState
    assert isinstance(__version__, str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_package.py -v`
Expected: FAIL with `ModuleNotFoundError` (no `__init__.py` yet)

- [ ] **Step 3: Create `pyproject.toml`**

hatchling build backend, `[project]` name `superpowers_agent`, version, authors, `requires-python = ">=3.10"`, dependencies `openai>=1.0`, `pydantic>=2`, `python-dotenv`, `[project.optional-dependencies] dev = ["pytest"]`, `[tool.hatch.build.targets.wheel] packages = ["src/superpowers_agent"]`.

- [ ] **Step 4: Create `src/superpowers_agent/__init__.py`**

Re-export the public API and define `__version__ = "0.1.0"`.

- [ ] **Step 5: Create `tests/conftest.py`**

Define a `MockLLMClient` fixture with scripted `generate_response` / `generate_json` responses.

- [ ] **Step 6: Create `README.md` and `example_usage.py`**

README: install (`pip install -e .`), set `OPENROUTER_API_KEY`, usage. `example_usage.py`: walk through brainstorm → approve → plan → approve → dry-run execute, catching `WorkflowStateError` when a gate is skipped, using a non-OpenAI model (e.g. `meta-llama/llama-3.1-70b-instruct`).

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_package.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml README.md src/superpowers_agent/__init__.py example_usage.py tests/conftest.py tests/test_package.py
git commit -m "feat: add package scaffolding, README, and example"
```

---
### Task 9: Full verification

**Files:**
- None (verification only).

- [ ] **Step 1: Install editable**

Run: `pip install -e .`
Expected: success

- [ ] **Step 2: Run full test suite**

Run: `pytest -v`
Expected: all tests PASS

- [ ] **Step 3: Run example in dry-run**

Run: `python example_usage.py`
Expected: runs without network calls (dry-run default), demonstrates gate errors

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "chore: final verification"
```