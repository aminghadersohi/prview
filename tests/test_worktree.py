"""Tests for prview git worktree manager."""

from pathlib import Path

import pytest

from prview.web.worktree import (
    get_bare_repo_path,
    get_worktree_path,
)


class TestPathHelpers:
    def test_bare_repo_path_org_repo(self):
        path = get_bare_repo_path("preset-io/superset-shell")
        assert path == Path.home() / ".config/prview/repos/preset-io/superset-shell.git"

    def test_bare_repo_path_single_name(self):
        path = get_bare_repo_path("myrepo")
        assert path == Path.home() / ".config/prview/repos/myrepo.git"

    def test_worktree_path(self):
        path = get_worktree_path("preset-io/superset-shell", 42)
        assert path == Path.home() / ".config/prview/worktrees/preset-io/superset-shell/42"

    def test_worktree_path_different_pr(self):
        path1 = get_worktree_path("org/repo", 1)
        path2 = get_worktree_path("org/repo", 2)
        assert path1 != path2
        assert path1.name == "1"
        assert path2.name == "2"


class TestWorktreeStatus:
    @pytest.mark.asyncio
    async def test_status_nonexistent(self):
        from prview.web.worktree import get_worktree_status

        # Use a repo/PR combo that definitely doesn't exist
        status = await get_worktree_status("nonexistent-org/nonexistent-repo", 99999)
        assert status["exists"] is False
        assert status["has_changes"] is False
        assert status["changed_files"] == []

    @pytest.mark.asyncio
    async def test_diff_nonexistent(self):
        from prview.web.worktree import get_worktree_diff

        diff = await get_worktree_diff("nonexistent-org/nonexistent-repo", 99999)
        assert diff == ""

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self):
        from prview.web.worktree import remove_worktree

        msg = await remove_worktree("nonexistent-org/nonexistent-repo", 99999)
        assert "does not exist" in msg.lower()

    @pytest.mark.asyncio
    async def test_sync_nonexistent(self):
        from prview.web.worktree import sync_worktree

        msg = await sync_worktree("nonexistent-org/nonexistent-repo", 99999)
        assert "does not exist" in msg.lower()


class TestRunCmd:
    @pytest.mark.asyncio
    async def test_run_cmd_success(self):
        from prview.web.worktree import _run_cmd

        rc, out, err = await _run_cmd(["echo", "hello"])
        assert rc == 0
        assert "hello" in out

    @pytest.mark.asyncio
    async def test_run_cmd_failure(self):
        from prview.web.worktree import _run_cmd

        rc, out, err = await _run_cmd(["false"])
        assert rc != 0

    @pytest.mark.asyncio
    async def test_run_git(self):
        from prview.web.worktree import _run_git

        rc, out, err = await _run_git(["--version"])
        assert rc == 0
        assert "git version" in out


class TestCommitAndPush:
    @pytest.mark.asyncio
    async def test_commit_and_push_no_worktree(self):
        from prview.web.worktree import commit_and_push

        result = await commit_and_push("nonexistent/repo", 99999, "test message")
        assert result["ok"] is False
        assert "does not exist" in result["error"].lower()
