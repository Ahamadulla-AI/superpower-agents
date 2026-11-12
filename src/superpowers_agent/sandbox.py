"""Sandbox: path validation and command allowlist.

Nothing can be written or executed outside the project root, and only a
known set of commands may be run.
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import UnsafeActionError

# Commands that are safe to run in the sandbox.
# Each pattern is a shell command name (the first word).
ALLOWED_COMMANDS: frozenset[str] = frozenset({
    "git",
    "python",
    "node",
    "npm",
    "pnpm",
    "yarn",
    "pip",
    "uv",
    "cargo",
    "java",
    "javac",
    "go",
    "gcc",
    "g++",
    "make",
    "cmake",
    "docker",
    "docker-compose",
    "cat",
    "head",
    "tail",
    "grep",
    "sed",
    "mv",
    "cp",
    "mkdir",
    "touch",
    "rm",
    "ls",
    "pwd",
    "find",
    "pytest",
    "npm",
    "jest",
    "npx",
    "eslint",
    "tsc",
    "ruff",
    "black",
    "mypy",
})

# Blocklist of destructive git subcommands
BLOCKED_GIT_SUBCOMMANDS: frozenset[str] = frozenset({
    "push",
    "am",
    "fetch",
    "merge",
    "rebase",
})

# Regex to detect absolute paths
_ABSOLUTE_PATH_RE = re.compile(r"^/[a-zA-Z0-9_\-/]+$")


def validate_path(filepath: str | Path, project_root: Path) -> Path:
    """Validate and resolve ``filepath`` to an absolute path inside ``project_root``.

    Args:
        filepath: The path to validate (relative or absolute).
        project_root: The project root directory.

    Returns:
        The resolved absolute path.

    Raises:
        UnsafeActionError: If the path escapes the project root.
    """
    filepath = Path(filepath)
    # Resolve to absolute if needed
    if not filepath.is_absolute():
        resolved = (project_root / filepath).resolve()
    else:
        resolved = filepath.resolve()

    try:
        resolved.relative_to(project_root)
    except ValueError:
        raise UnsafeActionError(
            f"Path '{filepath}' escapes the sandbox root '{project_root}'"
        )

    return resolved


def validate_command(cmd: str) -> str:
    """Validate that a shell command is allowed in the sandbox.

    Args:
        cmd: The shell command to validate.

    Returns:
        The original command string.

    Raises:
        UnsafeActionError: If the command is not allowed.
    """
    parts = cmd.strip().split()
    if not parts:
        raise UnsafeActionError("Empty command")

    executable = parts[0].split("/")[-1]  # strip any path prefix

    if executable not in ALLOWED_COMMANDS:
        raise UnsafeActionError(
            f"Command '{executable}' is not in the allowlist"
        )

    # Extra checks for git
    if executable == "git" and len(parts) > 1:
        subcommand = parts[1]
        if subcommand in BLOCKED_GIT_SUBCOMMANDS:
            raise UnsafeActionError(
                f"Git subcommand '{subcommand}' is blocked in the sandbox"
            )

    return cmd
