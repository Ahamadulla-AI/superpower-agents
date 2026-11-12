"""CLI entry-point for Superpowers Agent SDK."""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from superpowers_agent import LLMClient, AgenticWorkflow, WorkflowState, auto_detect

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
load_dotenv()


def main() -> None:
    """Run the gated workflow in interactive mode."""
    if len(sys.argv) < 2:
        print("Usage: superpowers \"Your feature requirement\"")
        print("  Or run: python -m superpowers_agent \"Your feature requirement\"")
        sys.exit(1)

    requirement = sys.argv[1]
    project = Path(".").resolve()

    api_key, base_url, model, provider = auto_detect()
    if not api_key:
        print("Error: No LLM API key found.")
        print()
        print("  Set one of: OPENAI_API_KEY, OPENROUTER_API_KEY,")
        print("  GOOGLE_API_KEY, or AZURE_OPENAI_API_KEY")
        print("  (in your environment or a .env file)")
        sys.exit(1)

    print(f"Detected provider: {provider} (model: {model})")

    client = LLMClient()
    wf = AgenticWorkflow(client, project, requirement)

    # Gate 1: Brainstorming
    print("=== BRAINSTORMING ===")
    questions = wf.brainstorm()
    print(questions)
    input("Review the questions above and press Enter to generate the spec...")

    # Gate 2: Spec
    print("\n=== SPEC ===")
    spec = wf.approve_spec()
    print(spec)
    input("Review the spec above and press Enter to generate the plan...")

    # Gate 3: Plan
    print("\n=== PLAN ===")
    plan = wf.generate_plan()
    for task in plan.tasks:
        print(f"  [{task.id}] {task.title}: {task.description}")
    input("Review the plan above and press Enter to execute...")

    # Gate 4: Execute
    print("\n=== EXECUTING ===")
    results = wf.execute_plan()
    for r in results:
        print(f"  Done: {r.task_id} - {r.notes}")

    print(f"\nWorkflow complete. Final state: {wf.get_current_state()}")


if __name__ == "__main__":
    main()
