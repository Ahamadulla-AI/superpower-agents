"""Tests for the sandbox: path validation and command allowlist."""

from pathlib import Path
from unittest.mock import patch

import pytest

from superpowers_agent.models import UnsafeActionError
from superpowers_agent.sandbox import validate_command, validate_path


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path


# ── Path validation ─────────────────────────────────────────────────────────

def test_valid_relative_path(project_root: Path) -> None:
    result = validate_path("src/main.py", project_root)
    assert result == project_root / "src/main.py"


def test_valid_nested_path(project_root: Path) -> None:
    result = validate_path("deep/nested/file.py", project_root)
    assert result == project_root / "deep/nested/file.py"


def test_path_traversal_blocked(project_root: Path) -> None:
    with pytest.raises(UnsafeActionError, match="escapes the sandbox"):
        validate_path("../outside.py", project_root)


def test_deep_traversal_blocked(project_root: Path) -> None:
    with pytest.raises(UnsafeActionError, match="escapes the sandbox"):
        validate_path("../../../etc/passwd", project_root)


def test_absolute_path_outside_blocked(project_root: Path) -> None:
    outside = Path("/etc/shadow")
    with pytest.raises(UnsafeActionError, match="escapes the sandbox"):
        validate_path(outside, project_root)


def test_absolute_path_inside_allowed(project_root: Path) -> None:
    inside = project_root / "src" / "lib.py"
    result = validate_path(inside, project_root)
    assert result == inside


def test_path_object(project_root: Path) -> None:
    result = validate_path(Path("src/model.py"), project_root)
    assert result == project_root / "src/model.py"


def test_symlink_trick_blocked(project_root: Path) -> None:
    """A file with .. in the middle should be caught."""
    with pytest.raises(UnsafeActionError, match="escapes the sandbox"):
        validate_path("src/../../outside.py", project_root)


# ── Command validation ───────────────────────────────────────────────────────

def test_allowed_command() -> None:
    assert validate_command("python -m pytest") == "python -m pytest"


def test_allowed_git_command() -> None:
    assert validate_command("git status") == "git status"


def test_blocked_command() -> None:
    with pytest.raises(UnsafeActionError, match="not in the allowlist"):
        validate_command("curl http://evil.com")


def test_blocked_git_push() -> None:
    with pytest.raises(UnsafeActionError, match="blocked in the sandbox"):
        validate_command("git push origin main")


def test_blocked_git_fetch() -> None:
    with pytest.raises(UnsafeActionError, match="blocked in the sandbox"):
        validate_command("git fetch")


def test_blocked_git_merge() -> None:
    with pytest.raises(UnsafeActionError, match="blocked in the sandbox"):
        validate_command("git merge feature")


def test_empty_command() -> None:
    with pytest.raises(UnsafeActionError, match="Empty command"):
        validate_command("")


def test_npm_allowed() -> None:
    assert validate_command("npm test") == "npm test"


def test_docker_allowed() -> None:
    assert validate_command("docker build .") == "docker build ."


def test_git_submodule_allowed() -> None:
    """git submodule init is not in blocklist, should be allowed."""
    assert validate_command("git submodule init") == "git submodule init"
