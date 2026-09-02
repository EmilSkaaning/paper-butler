"""Unit tests for scripts/lint_commit_msg.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lint_commit_msg import (
    COMMIT_TYPES,
    _lint_message_file,
    is_skippable,
    validate_subject,
)


class TestValidateSubject:
    """Tests for validate_subject against cliff.toml's commit type list."""

    @pytest.mark.parametrize(
        "subject",
        [
            "feat: add pdf preview",
            "fix: correct tissue mask path",
            "refactor: replace session tokens",
            "chore: bump modal to 0.67.0",
            "docs: add contributing guide",
            "test: cover empty upload case",
            "ci: cache testmon data",
            "perf: reduce embedding batch size",
            "feat(upload)!: require content-type header",
        ],
    )
    def test_returns_no_error_for_valid_subject(self, subject: str) -> None:
        # Arrange / Act
        error = validate_subject(subject)

        # Assert
        assert error is None

    @pytest.mark.parametrize(
        "subject",
        [
            "Feat: add pdf preview",
            "FIX: correct tissue mask path",
            "Perf: reduce embedding batch size",
        ],
    )
    def test_rejects_wrong_case_type_prefix(self, subject: str) -> None:
        # Arrange / Act
        error = validate_subject(subject)

        # Assert
        assert error is not None
        assert "type" in error.lower()

    def test_rejects_unknown_type(self) -> None:
        # Arrange / Act
        error = validate_subject("update: bump dependency")

        # Assert
        assert error is not None

    def test_rejects_missing_colon(self) -> None:
        # Arrange / Act
        error = validate_subject("feat add pdf preview")

        # Assert
        assert error is not None

    def test_rejects_subject_over_seventy_five_chars(self) -> None:
        # Arrange
        subject = (
            "feat: implement the new de novo alphafold model weight "
            "enumeration subsystem"
        )

        # Act
        error = validate_subject(subject)

        # Assert
        assert error is not None
        assert "75" in error

    def test_accepts_subject_between_fifty_and_seventy_five_chars(self) -> None:
        # Arrange
        subject = "feat: implement the new de novo alphafold weight enumeration"

        # Act
        error = validate_subject(subject)

        # Assert
        assert 50 < len(subject) <= 75
        assert error is None

    def test_rejects_trailing_period(self) -> None:
        # Arrange / Act
        error = validate_subject("fix: correct tissue mask path.")

        # Assert
        assert error is not None
        assert "period" in error.lower()

    def test_all_cliff_toml_types_are_covered(self) -> None:
        # Arrange
        expected = {"feat", "fix", "refactor", "chore", "docs", "test", "ci", "perf"}

        # Act / Assert
        assert COMMIT_TYPES == expected

    def test_scope_does_not_swallow_description_with_parens(self) -> None:
        # Arrange
        subject = "fix(scope): change do (x): y"

        # Act
        error = validate_subject(subject)

        # Assert
        assert error is None


class TestLintMessageFile:
    """Tests for _lint_message_file, which reads a git commit-msg hook file."""

    def test_accepts_valid_subject_on_first_line(self, tmp_path: Path) -> None:
        # Arrange
        path = tmp_path / "COMMIT_EDITMSG"
        path.write_text("feat: add pdf preview\n", encoding="utf-8")

        # Act
        exit_code = _lint_message_file(str(path))

        # Assert
        assert exit_code == 0

    def test_skips_leading_blank_and_comment_lines(self, tmp_path: Path) -> None:
        # Arrange
        path = tmp_path / "COMMIT_EDITMSG"
        path.write_text(
            "\n# Please enter the commit message for your changes. Lines starting\n"
            "# with '#' will be ignored, and an empty message aborts the commit.\n"
            "feat: add pdf preview\n",
            encoding="utf-8",
        )

        # Act
        exit_code = _lint_message_file(str(path))

        # Assert
        assert exit_code == 0

    def test_rejects_message_with_only_blank_and_comment_lines(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        path = tmp_path / "COMMIT_EDITMSG"
        path.write_text(
            "\n# Please enter the commit message for your changes.\n",
            encoding="utf-8",
        )

        # Act
        exit_code = _lint_message_file(str(path))

        # Assert
        assert exit_code == 1


class TestIsSkippable:
    """Tests for is_skippable, which exempts merge/revert/fixup commits."""

    @pytest.mark.parametrize(
        "subject",
        [
            "Merge pull request #141 from foo/bar",
            "Merge branch 'main' into feat/x",
            'Revert "feat: add pdf preview"',
            "fixup! feat: add pdf preview",
            "squash! feat: add pdf preview",
        ],
    )
    def test_skips_merge_revert_fixup_squash(self, subject: str) -> None:
        assert is_skippable(subject) is True

    def test_does_not_skip_normal_commit(self) -> None:
        assert is_skippable("feat: add pdf preview") is False
