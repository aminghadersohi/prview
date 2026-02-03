"""Tests for prview Claude CLI wrapper."""

import os
from unittest.mock import MagicMock, patch

import pytest

from prview.web.claude_cli import (
    CLAUDE_MODELS,
    DEFAULT_MODEL,
    check_claude_available,
    find_claude_cli,
)


class TestFindClaudeCli:
    def test_found_in_path(self):
        with patch("shutil.which", return_value="/usr/local/bin/claude"):
            assert find_claude_cli() == "/usr/local/bin/claude"

    def test_not_found_anywhere(self):
        with patch("shutil.which", return_value=None), patch("os.path.isfile", return_value=False):
            assert find_claude_cli() is None

    def test_found_in_common_path(self):
        with (
            patch("shutil.which", return_value=None),
            patch("os.path.isfile", side_effect=lambda p: p == "/opt/homebrew/bin/claude"),
            patch("os.access", return_value=True),
        ):
            assert find_claude_cli() == "/opt/homebrew/bin/claude"


class TestCheckClaudeAvailable:
    def test_not_found(self):
        with patch("prview.web.claude_cli.find_claude_cli", return_value=None):
            result = check_claude_available()
            assert result["available"] is False
            assert "not found" in result["error"].lower()

    def test_available(self):
        with patch("prview.web.claude_cli.find_claude_cli", return_value="/usr/bin/claude"):
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = "1.2.3"
            with patch("subprocess.run", return_value=mock_proc):
                result = check_claude_available()
                assert result["available"] is True
                assert result["version"] == "1.2.3"
                assert result["path"] == "/usr/bin/claude"

    def test_cli_returns_error(self):
        with patch("prview.web.claude_cli.find_claude_cli", return_value="/usr/bin/claude"):
            mock_proc = MagicMock()
            mock_proc.returncode = 1
            mock_proc.stderr = "some error"
            with patch("subprocess.run", return_value=mock_proc):
                result = check_claude_available()
                assert result["available"] is False

    def test_timeout(self):
        import subprocess

        with patch("prview.web.claude_cli.find_claude_cli", return_value="/usr/bin/claude"):
            with patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=10),
            ):
                result = check_claude_available()
                assert result["available"] is False
                assert "timeout" in result["error"].lower()

    def test_file_not_found(self):
        with patch("prview.web.claude_cli.find_claude_cli", return_value="/usr/bin/claude"):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                result = check_claude_available()
                assert result["available"] is False


class TestConstants:
    def test_models_list_not_empty(self):
        assert len(CLAUDE_MODELS) > 0

    def test_models_have_required_fields(self):
        for model in CLAUDE_MODELS:
            assert "id" in model
            assert "name" in model
            assert "description" in model

    def test_default_model_in_list(self):
        model_ids = [m["id"] for m in CLAUDE_MODELS]
        assert DEFAULT_MODEL in model_ids
