# superpowers_agent — Design Spec

**Date:** 2026-11-08
**Status:** Approved (per user directive to start development)

## Intent and Why

Build an installable Python SDK, `superpowers_agent`, that reimplements the
ideas behind the `obra/superpowers` methodology as a **gated state machine**:
`BRAINSTORMING → SPEC_APPROVED → PLAN_APPROVED → EXECUTING → DONE`. The
methodology is a set of skill prompts and workflow conventions, so the prompts
are written fresh (distilled), not ported. The SDK works with any
OpenAI-compatible endpoint, defaulting to OpenRouter, with no hardcoded model
or endpoint.

The developer is a Python full-stack engineer building agentic systems and
needs production-quality code: typed, documented, tested, no hardcoded model
or endpoint.

## Goals

- Enforce workflow gates **in code** via a `WorkflowState` enum and
  `WorkflowStateError`; approval is always an explicit human method call.
- Enforce YAGNI, DRY, and TDD in the planner prompt (each task starts with a
  failing test, then minimal implementation, then refactor; reject speculative
  features).
- Subagent-driven development: decompose the plan into atomic tasks, delegate
  each to a fresh `SubAgent`.
- Safe execution: path resolution rejects escapes; commands run from an
  allowlist with `shell=False`, timeout, captured output.
- Robust LLM client: retry with backoff, JSON-mode with text-extraction
  fallback, pydantic validation with retry feeding errors back.
- Dry-run is the default for `execute_plan`.

## Non-Goals

- Not a port of `obra/superpowers` prompts (write distilled originals).
- No hardcoded model or endpoint.
- No bare `print` in library code (use logging).

## Anti-Goals

- LLM-triggerable approval (approval must be human-only).
- Unsafe path/command execution escaping the workspace.
- Network calls in tests (mock client only).

## Constraints

- Python >= 3.10.
- Dependencies: `openai>=1.0`, `pydantic>=2`, `python-dotenv`. Dev: `pytest`.
- Build system: hatchling with `src/` layout.
- Package name `superpowers_agent`.

## Approaches

The spec fully determines the structure, files, and behaviors, so there is one
sensible way to build it: implement the exact file layout and requirements
given. Alternatives (e.g., a single-module SDK, or enforcing gates only in
prompts) were ruled out because they contradict the explicit requirements for
a typed, tested, gated, multi-file package.

## The How (as decided)

- **Build system:** hatchling, `src/` layout.
- **Files:** exactly as specified in the structure.
- **Prompts:** fresh distilled prompts in `prompts.py`.
- **Gates:** `WorkflowState` enum + `WorkflowStateError`; approval via
  `approve_spec()` / `approve_plan()`.
- **Client:** `LLMClient` with retry/backoff, JSON mode + fallback, pydantic
  validation with retry.
- **Workflow:** `AgenticWorkflow` holding state + conversation history.
- **Subagent:** `SubAgent` per task, applies results via `sandbox.py`.
- **Sandbox:** path + command safety.
- **Verification:** install editable, run pytest, fix failures.

## What's Left to the Builder

- Exact prompt wording (distilled originals).
- Test cases for gates, plan parsing, and sandbox.
- `example_usage.py` walkthrough.