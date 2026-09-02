"""Validate commit subjects against the conventional-commit format cliff.toml depends on.

Usage:
    uv run python scripts/lint_commit_msg.py <commit-msg-file>
    uv run python scripts/lint_commit_msg.py --range origin/main..HEAD
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

COMMIT_TYPES = {"feat", "fix", "refactor", "chore", "docs", "test", "ci", "perf"}
MAX_SUBJECT_LENGTH = 75
SKIP_PREFIXES = ("Merge ", "Revert ", "fixup!", "squash!")

_SUBJECT_PATTERN = re.compile(
    r"^(?P<type>[a-zA-Z]+)(?P<scope>\([^)]+\))?(?P<breaking>!)?: (?P<description>.+)$"
)


def is_skippable(subject: str) -> bool:
    """Check whether a commit subject is exempt from format validation.

    Args:
        subject: The commit message subject line.

    Returns:
        True if the subject is a merge, revert, fixup, or squash commit.
    """
    return subject.startswith(SKIP_PREFIXES)


def validate_subject(subject: str) -> str | None:
    """Validate a commit subject against cliff.toml's conventional-commit format.

    Args:
        subject: The commit message subject line.

    Returns:
        None if the subject is valid, otherwise a human-readable error message.
    """
    match = _SUBJECT_PATTERN.match(subject)
    if not match:
        return (
            f"'{subject}' does not match '<type>(<scope>)?: <description>' "
            f"(allowed types: {', '.join(sorted(COMMIT_TYPES))})"
        )

    commit_type = match.group("type")
    if commit_type not in COMMIT_TYPES:
        return (
            f"'{subject}' has an unknown or wrongly-cased type '{commit_type}' "
            f"(allowed types: {', '.join(sorted(COMMIT_TYPES))})"
        )

    if len(subject) > MAX_SUBJECT_LENGTH:
        return f"'{subject}' is {len(subject)} chars, exceeds the {MAX_SUBJECT_LENGTH}-char limit"

    if subject.endswith("."):
        return f"'{subject}' must not end with a trailing period"

    return None


def _lint_message_file(path: str) -> int:
    """Validate the subject line of a commit-msg hook file.

    Args:
        path: Path to the commit message file passed by the commit-msg hook.

    Returns:
        Process exit code: 0 if the subject is valid or skippable, 1 otherwise.
    """
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()

    subject = next(
        (
            stripped
            for line in lines
            if (stripped := line.strip()) and not stripped.startswith("#")
        ),
        "",
    )

    if is_skippable(subject):
        return 0

    error = validate_subject(subject)
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    return 0


def _lint_range(commit_range: str) -> int:
    """Validate every non-merge commit subject within a git commit range.

    Args:
        commit_range: A git revision range, e.g. ``origin/main..HEAD``.

    Returns:
        Process exit code: 0 if all subjects are valid or skippable, 1 otherwise.
    """
    try:
        result = subprocess.run(
            ["git", "log", "--format=%s", "--no-merges", commit_range],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(
            f"error: invalid commit range '{commit_range}': {exc.stderr.strip()}",
            file=sys.stderr,
        )
        return 1
    subjects = [line for line in result.stdout.splitlines() if line]

    exit_code = 0
    for subject in subjects:
        if is_skippable(subject):
            continue
        error = validate_subject(subject)
        if error:
            print(f"error: {error}", file=sys.stderr)
            exit_code = 1

    return exit_code


def main() -> int:
    """Run the commit-message linter as a CLI entry point.

    Returns:
        Process exit code: 0 on success, 1 if any commit subject is invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "message_file", nargs="?", help="Path to a commit-msg hook file"
    )
    parser.add_argument(
        "--range", dest="commit_range", help="Git commit range, e.g. origin/main..HEAD"
    )
    args = parser.parse_args()

    if args.commit_range:
        return _lint_range(args.commit_range)

    if args.message_file:
        return _lint_message_file(args.message_file)

    parser.error("either a commit-msg file or --range must be given")
    return 2


if __name__ == "__main__":
    sys.exit(main())
