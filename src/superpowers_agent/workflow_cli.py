"""Interactive command-line runner for the gated workflow."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

from dotenv import load_dotenv

from . import AgenticWorkflow, LLMClient, LLMClientError, auto_detect


def run_workflow(requirement: str) -> None:
    """Run the workflow for a feature requirement in the current directory."""
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    load_dotenv()

    try:
        api_key, base_url, model = auto_detect()
    except LLMClientError as exc:
        raise SystemExit(
            "No LLM API key found. Set OPENAI_API_KEY, OPENROUTER_API_KEY, "
            "GOOGLE_API_KEY, or AZURE_OPENAI_API_KEY in the environment or .env."
        ) from exc

    print(f"Using {base_url} (model: {model})")
    workflow = AgenticWorkflow(
        LLMClient(api_key=api_key, base_url=base_url, model=model),
        Path.cwd().resolve(),
        requirement,
    )

    print("=== BRAINSTORMING ===")
    questions = workflow.brainstorm()
    print(questions)
    input("Review the questions above and press Enter to generate the spec...")

    print("\n=== SPEC ===")
    spec = workflow.approve_spec()
    print(spec)
    input("Review the spec above and press Enter to generate the plan...")

    print("\n=== PLAN ===")
    plan = workflow.generate_plan()
    for task in plan.tasks:
        print(f"  [{task.id}] {task.title}: {task.description}")
    input("Review the plan above and press Enter to execute...")

    print("\n=== EXECUTING ===")
    for result in workflow.execute_plan():
        print(f"  Done: {result.task_id} - {result.notes}")
    print(f"\nWorkflow complete. Final state: {workflow.get_current_state()}")


def main() -> None:
    """Preserve the legacy command that accepts a requirement directly."""
    if len(sys.argv) < 2:
        raise SystemExit('Usage: superpowers "Your feature requirement"')
    run_workflow(" ".join(sys.argv[1:]))
