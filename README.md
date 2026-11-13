# Superpower Agents

Install reusable agent skills into a project and use a gated, subagent-driven development workflow with an OpenAI-compatible LLM.

## Install

```bash
pip install superpower-agents
```

## Install the skills in a project

From the root of the project where your LLM coding agent will work, run:

```bash
superpower-agents init
```

This copies the bundled skills to `.agents/skills/` in the current project. The skills cover brainstorming, specification and plan writing, implementation, testing, debugging, code review, worktree management, and other development workflows. They are plain `SKILL.md` files that coding agents supporting the `.agents/skills/` convention can discover and use.

To initialize a different project directory:

```bash
superpower-agents init --project path/to/project
```

Existing files are skipped, so rerunning `init` preserves local edits while adding missing bundled files. To replace installed copies with the package versions, use `--force`. List the skills included in your installed package with:

```bash
superpower-agents list-skills
```

## Run the interactive workflow

Set a supported API key in your environment or in a `.env` file in the project, then run:

```bash
superpower-agents run "Add user authentication"
```

The workflow guides a requirement through brainstorming, specification, planning, and execution. Review each stage before continuing. The legacy `superpowers "Add user authentication"` command is also available.

Supported environment variables include `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GOOGLE_API_KEY`, and `AZURE_OPENAI_API_KEY`. OpenAI-compatible custom endpoints can be configured through the provider variables supported by the client.

## Use it programmatically

```python
from pathlib import Path

from superpowers_agent import AgenticWorkflow, LLMClient

workflow = AgenticWorkflow(
    LLMClient(),
    Path.cwd(),
    "Add user authentication",
)
questions = workflow.brainstorm()
# Review and answer the questions, then continue through the workflow gates.
```

## Safety

- File changes are constrained to the project root by the workflow sandbox.
- Commands are validated against an allowlist.
- Skill installation skips existing files by default; `--force` explicitly replaces them.

## Development

```bash
pip install -e ".[dev]"
```

The distribution is named `superpower-agents`; the Python package remains `superpowers_agent` for compatibility.

## License

MIT. See [LICENSE](LICENSE).
