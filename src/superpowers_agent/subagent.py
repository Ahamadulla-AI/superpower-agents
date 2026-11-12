"""Subagent: execute a single atomic task.

A subagent receives one ``Task`` from the plan, reads the relevant files,
and returns a structured ``TaskResult`` with file writes and commands.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .client import LLMClient
from .models import Command, FileWrite, Task, TaskResult
from .prompts import SUBAGENT_SYSTEM, SUBAGENT_USER
from .sandbox import validate_path, validate_command

logger = logging.getLogger(__name__)


class SubAgent:
    """Execute a single task against the project files."""

    def __init__(
        self,
        client: LLMClient,
        project_root: Path,
    ) -> None:
        self.client = client
        self.project_root = project_root

    def run(
        self, task: Task, file_contents: dict[str, str] | None = None
    ) -> TaskResult:
        """Run a single task and return the structured result.

        Args:
            task: The atomic task to execute.
            file_contents: Mapping of relative file paths to their
                current contents. If ``None``, files are read from disk.

        Returns:
            A ``TaskResult`` containing the file writes and commands.
        """
        if file_contents is None:
            file_contents = self._read_task_files(task)

        # Build context with file contents
        file_context = "\n\n".join(
            f"--- {fp} ---\n{content}" for fp, content in file_contents.items()
        )

        steps_text = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(task.steps))
        files_text = ", ".join(task.files) if task.files else "(none)"

        user_content = SUBAGENT_USER.format(
            task_title=task.title,
            task_description=task.description,
            task_steps=steps_text,
            task_files=files_text,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SUBAGENT_SYSTEM},
            {"role": "user", "content": f"{user_content}\n\nCurrent file contents:\n\n{file_context}"},
        ]

        result: TaskResult = self.client.generate_json(messages, TaskResult)
        result.task_id = task.id

        # Validate all paths and commands
        for fw in result.file_writes:
            validate_path(fw.path, self.project_root)
        for c in result.commands:
            validate_command(c.cmd)

        # Apply file writes to disk
        for fw in result.file_writes:
            dest = validate_path(fw.path, self.project_root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(fw.content, encoding="utf-8")
            logger.info("Wrote file: %s", fw.path)

        # Run commands
        for c in result.commands:
            result_text = self._run_command(c)
            logger.info("Command output: %s", result_text[:500])

        return result

    # ── Private helpers ──────────────────────────────────────────────────

    def _read_task_files(self, task: Task) -> dict[str, str]:
        """Read the current contents of files relevant to the task."""
        contents: dict[str, str] = {}
        for fp in task.files:
            path = self.project_root / fp
            if path.is_file():
                contents[fp] = path.read_text(encoding="utf-8")
            else:
                contents[fp] = ""  # new file
        return contents

    def _run_command(self, command: Command) -> str:
        """Run a validated command in a subshell.

        Args:
            command: The validated command to run.

        Returns:
            The stdout output (truncated to 4000 chars).
        """
        import subprocess

        validate_command(command.cmd)
        proc = subprocess.run(
            command.cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=command.timeout_s,
            cwd=str(self.project_root),
        )
        output = proc.stdout
        if proc.stderr:
            output += "\n" + proc.stderr
        return output[:4000]
