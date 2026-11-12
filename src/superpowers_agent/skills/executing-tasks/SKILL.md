---
name: executing-tasks
description: Implement an approved plan one task at a time while respecting project conventions and scope.
---

# Executing Tasks

Use this skill only after the user approves the implementation plan.

## Process

1. Work on the next unblocked task and read the relevant files before editing.
2. Make the smallest coherent change that satisfies that task and its acceptance criteria.
3. Follow the repository's established style and preserve existing user changes.
4. Run the requested checks when authorized; report checks that could not be run.
5. Review the diff for scope, correctness, and accidental changes before moving to the next task.
6. Stop and report a concrete blocker if the plan conflicts with repository reality or user direction.

## Safety

- Keep file changes within the project unless the user explicitly requests another location.
- Do not claim a check passed unless it actually ran and passed.
- Do not commit, publish, or push unless the user explicitly requests that action.
