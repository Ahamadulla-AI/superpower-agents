"""Command-line interface for installing skills and running workflows."""

from __future__ import annotations

import argparse
import importlib.resources
import shutil
import sys
from pathlib import Path


SKILL_ROOT = Path(".agents") / "skills"


def _bundled_skills() -> list[tuple[str, importlib.resources.abc.Traversable]]:
    root = importlib.resources.files("superpowers_agent").joinpath("skills")
    return sorted(
        ((item.name, item) for item in root.iterdir() if item.is_dir()),
        key=lambda entry: entry[0],
    )


def _files(
    directory: importlib.resources.abc.Traversable, relative: Path = Path()
):
    for item in directory.iterdir():
        item_relative = relative / item.name
        if item.is_dir():
            yield from _files(item, item_relative)
        elif item.is_file():
            yield item_relative, item


def _install_skills(project: Path, force: bool) -> int:
    destination_root = project / SKILL_ROOT
    installed = 0
    skipped = 0

    if not project.is_dir():
        print(f"Project directory does not exist: {project}", file=sys.stderr)
        return 2

    for skill_name, bundled_dir in _bundled_skills():
        destination = destination_root / skill_name
        for relative, item in _files(bundled_dir):
            target = destination / relative
            if target.exists() and not force:
                print(f"Skipped existing file: {target}")
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with item.open("rb") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            print(f"Installed: {target}")
            installed += 1

    if installed == 0 and skipped == 0:
        print("No bundled skills were found.", file=sys.stderr)
        return 1
    print(f"\nInstalled {installed} file(s); skipped {skipped} existing file(s).")
    return 0


def _list_skills() -> int:
    for name, skill_dir in _bundled_skills():
        description = ""
        skill_file = skill_dir.joinpath("SKILL.md")
        if skill_file.is_file():
            for line in skill_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("description:"):
                    description = line.partition(":")[2].strip()
                    break
        print(f"{name}: {description}" if description else name)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="superpower-agents",
        description="Install reusable agent skills or run the gated workflow.",
    )
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="install skills into a project")
    init_parser.add_argument("--project", type=Path, default=Path.cwd())
    init_parser.add_argument(
        "--force", action="store_true", help="replace installed files with bundled copies"
    )
    subparsers.add_parser("list-skills", help="list skills included in this package")
    run_parser = subparsers.add_parser("run", help="run the gated development workflow")
    run_parser.add_argument("requirement", nargs=argparse.REMAINDER)

    args = parser.parse_args()
    if args.command == "init":
        raise SystemExit(_install_skills(args.project.resolve(), args.force))
    if args.command == "list-skills":
        raise SystemExit(_list_skills())
    if args.command == "run":
        if not args.requirement:
            parser.error("run requires a feature requirement")
        from .workflow_cli import run_workflow

        run_workflow(" ".join(args.requirement))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
