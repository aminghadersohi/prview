"""Tests for prview web server routes and helper functions."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prview.web.server import (
    app,
    build_pr_context_prompt,
    build_review_prompt,
    get_ci_status,
    get_review_status,
    init_db,
    should_include_repo,
)


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """Use a temporary database for each test."""
    db_path = tmp_path / "prview.db"
    monkeypatch.setattr("prview.web.server.DB_PATH", db_path)
    init_db()
    return db_path


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# --- Unit tests for helper functions ---


class TestShouldIncludeRepo:
    def test_no_filters_includes_all(self):
        assert should_include_repo("org/repo", {}) is True

    def test_exclude_repo(self):
        config = {"exclude_repos": ["org/repo"]}
        assert should_include_repo("org/repo", config) is False
        assert should_include_repo("org/other", config) is True

    def test_exclude_org(self):
        config = {"exclude_orgs": ["blocked-org"]}
        assert should_include_repo("blocked-org/repo", config) is False
        assert should_include_repo("allowed-org/repo", config) is True

    def test_include_repos_only(self):
        config = {"include_repos": ["org/repo"]}
        assert should_include_repo("org/repo", config) is True
        assert should_include_repo("org/other", config) is False

    def test_include_orgs_only(self):
        config = {"include_orgs": ["myorg"]}
        assert should_include_repo("myorg/repo", config) is True
        assert should_include_repo("other/repo", config) is False

    def test_exclude_takes_precedence_over_include(self):
        config = {"include_orgs": ["org"], "exclude_repos": ["org/blocked"]}
        assert should_include_repo("org/repo", config) is True
        assert should_include_repo("org/blocked", config) is False


class TestGetCiStatus:
    def test_empty_checks(self):
        assert get_ci_status([]) == "none"

    def test_all_success(self):
        checks = [{"conclusion": "SUCCESS", "status": "COMPLETED"}]
        assert get_ci_status(checks) == "success"

    def test_failure(self):
        checks = [
            {"conclusion": "SUCCESS", "status": "COMPLETED"},
            {"conclusion": "FAILURE", "status": "COMPLETED"},
        ]
        assert get_ci_status(checks) == "failure"

    def test_pending(self):
        checks = [
            {"conclusion": "SUCCESS", "status": "COMPLETED"},
            {"conclusion": "", "status": "IN_PROGRESS"},
        ]
        assert get_ci_status(checks) == "pending"

    def test_skipped_is_success(self):
        checks = [{"conclusion": "SKIPPED", "status": "COMPLETED"}]
        assert get_ci_status(checks) == "success"

    def test_timed_out_is_failure(self):
        checks = [{"conclusion": "TIMED_OUT", "status": "COMPLETED"}]
        assert get_ci_status(checks) == "failure"

    def test_queued_is_pending(self):
        checks = [{"conclusion": "", "status": "QUEUED"}]
        assert get_ci_status(checks) == "pending"


class TestGetReviewStatus:
    def test_approved(self):
        assert get_review_status("APPROVED") == "approved"

    def test_changes_requested(self):
        assert get_review_status("CHANGES_REQUESTED") == "changes_requested"

    def test_review_required(self):
        assert get_review_status("REVIEW_REQUIRED") == "review_required"

    def test_unknown_defaults_to_pending(self):
        assert get_review_status("") == "pending"
        assert get_review_status("UNKNOWN") == "pending"


class TestBuildPrContextPrompt:
    def test_empty_data(self):
        result = build_pr_context_prompt({"my_prs": {}, "review_requests": []})
        assert "prview" in result
        assert "My Open Pull Requests" not in result

    def test_with_my_prs(self):
        data = {
            "my_prs": {
                "org/repo": [
                    {
                        "number": 42,
                        "title": "Fix bug",
                        "draft": False,
                        "ci_status": "success",
                        "review_status": "approved",
                        "additions": 10,
                        "deletions": 5,
                    }
                ]
            },
            "review_requests": [],
        }
        result = build_pr_context_prompt(data)
        assert "org/repo#42" in result
        assert "Fix bug" in result

    def test_with_review_requests(self):
        data = {
            "my_prs": {},
            "review_requests": [
                {
                    "repo": "org/repo",
                    "number": 99,
                    "title": "Add feature",
                    "author": "someone",
                    "ci_status": "pending",
                }
            ],
        }
        result = build_pr_context_prompt(data)
        assert "Review Requests" in result
        assert "org/repo#99" in result
        assert "someone" in result


class TestBuildReviewPrompt:
    def test_basic_prompt(self):
        pr_data = {
            "title": "Add login",
            "author": "dev",
            "files": [{"path": "auth.py"}],
            "additions": 50,
            "deletions": 10,
            "body": "Implements login flow",
            "diff": "+def login():\n+    pass",
        }
        result = build_review_prompt(pr_data, [])
        assert "Add login" in result
        assert "dev" in result
        assert "Implements login flow" in result
        assert "+def login():" in result

    def test_with_conversation(self):
        pr_data = {"title": "PR", "author": "x", "files": [], "additions": 0, "deletions": 0}
        conversation = [
            {"role": "user", "content": "Focus on security"},
            {"role": "assistant", "content": "I found an XSS issue"},
        ]
        result = build_review_prompt(pr_data, conversation)
        assert "Focus on security" in result
        assert "I found an XSS issue" in result
        assert "updated review" in result


# --- Route integration tests ---


class TestDashboardRoutes:
    def test_index_page(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "prview" in response.text

    def test_api_prs(self, client):
        response = client.get("/api/prs")
        assert response.status_code == 200
        data = response.json()
        assert "my_prs" in data
        assert "review_requests" in data
        assert "sync_status" in data

    def test_partials_prs(self, client):
        response = client.get("/partials/prs")
        assert response.status_code == 200
        assert "My Open PRs" in response.text

    def test_partials_status(self, client):
        response = client.get("/partials/status")
        assert response.status_code == 200

    def test_trigger_sync(self, client):
        with patch("prview.web.server.sync_prs"):
            response = client.post("/sync")
            assert response.status_code == 200
            assert response.json()["status"] == "syncing"


class TestSettingsRoutes:
    def test_settings_page(self, client):
        with patch("prview.web.server.check_claude_available", return_value={"available": False}):
            response = client.get("/settings")
            assert response.status_code == 200
            assert "Settings" in response.text

    def test_save_settings(self, client, tmp_path, monkeypatch):
        config_path = tmp_path / "config.yaml"
        monkeypatch.setattr("prview.web.server.CONFIG_PATH", config_path)
        response = client.post(
            "/settings/save",
            data={
                "include_orgs": "myorg",
                "exclude_orgs": "",
                "include_repos": "",
                "exclude_repos": "",
                "refresh_interval": "120",
                "max_prs_per_repo": "5",
            },
        )
        assert response.status_code == 200  # follows redirect
        assert config_path.exists()


class TestChatRoutes:
    def test_chat_page(self, client):
        with patch(
            "prview.web.server.check_claude_available",
            return_value={"available": True, "version": "1.0"},
        ):
            response = client.get("/chat")
            assert response.status_code == 200
            assert "Chat" in response.text

    def test_chat_send_empty_message(self, client):
        response = client.post("/chat/send", data={"message": "", "model": "test"})
        assert response.status_code == 400


class TestCommentsRoutes:
    def test_comments_page(self, client):
        with patch(
            "prview.web.server.get_pr_basic_info",
            return_value={"title": "Test PR", "url": "https://github.com/org/repo/pull/1"},
        ):
            response = client.get("/pr/org/repo/1/comments")
            assert response.status_code == 200
            assert "Address Comments" in response.text
            assert "Test PR" in response.text

    def test_api_pr_comments(self, client):
        with patch("prview.web.server.get_pr_review_comments", return_value=[]):
            response = client.get("/api/pr/org/repo/1/comments")
            assert response.status_code == 200
            assert response.json() == {"comments": []}

    def test_api_pr_comments_with_data(self, client):
        mock_comments = [{"id": 1, "path": "file.py", "body": "Fix this", "user": "reviewer"}]
        with patch("prview.web.server.get_pr_review_comments", return_value=mock_comments):
            response = client.get("/api/pr/org/repo/42/comments")
            assert response.status_code == 200
            assert len(response.json()["comments"]) == 1

    def test_reply_to_comment_empty_body(self, client):
        response = client.post(
            "/pr/org/repo/1/comments/123/reply",
            json={"body": ""},
        )
        assert response.status_code == 400

    def test_reply_to_comment_success(self, client):
        with patch("prview.web.server.post_comment_reply", return_value=True):
            response = client.post(
                "/pr/org/repo/1/comments/123/reply",
                json={"body": "Good point, will fix."},
            )
            assert response.status_code == 200
            assert response.json()["status"] == "ok"

    def test_reply_to_comment_failure(self, client):
        with patch("prview.web.server.post_comment_reply", return_value=False):
            response = client.post(
                "/pr/org/repo/1/comments/123/reply",
                json={"body": "Acknowledged"},
            )
            assert response.status_code == 500


class TestReviewRoutes:
    def test_review_page(self, client):
        mock_pr_data = {
            "title": "Add feature",
            "body": "Description",
            "author": "dev",
            "files": [{"path": "main.py", "additions": 10, "deletions": 2, "status": "modified"}],
            "additions": 10,
            "deletions": 2,
            "url": "https://github.com/org/repo/pull/5",
            "diff": "+code",
        }
        with patch("prview.web.server.get_pr_full_data", return_value=mock_pr_data):
            response = client.get("/pr/org/repo/5/review")
            assert response.status_code == 200
            assert "Review" in response.text
            assert "Add feature" in response.text

    def test_submit_review_empty_body(self, client):
        response = client.post(
            "/pr/org/repo/5/review/submit",
            json={"body": "", "event": "COMMENT"},
        )
        assert response.status_code == 400

    def test_submit_review_invalid_event(self, client):
        response = client.post(
            "/pr/org/repo/5/review/submit",
            json={"body": "LGTM", "event": "INVALID"},
        )
        assert response.status_code == 400

    def test_submit_review_success(self, client):
        with patch("prview.web.server.submit_pr_review", return_value=True):
            response = client.post(
                "/pr/org/repo/5/review/submit",
                json={"body": "Looks good!", "event": "APPROVE"},
            )
            assert response.status_code == 200
            assert response.json()["status"] == "ok"

    def test_submit_review_failure(self, client):
        with patch("prview.web.server.submit_pr_review", return_value=False):
            response = client.post(
                "/pr/org/repo/5/review/submit",
                json={"body": "Needs work", "event": "REQUEST_CHANGES"},
            )
            assert response.status_code == 500


class TestPostCommentReply:
    def test_success(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            from prview.web.server import post_comment_reply

            assert post_comment_reply("org/repo", 1, 123, "Thanks!") is True

    def test_failure(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            from prview.web.server import post_comment_reply

            assert post_comment_reply("org/repo", 1, 123, "Thanks!") is False

    def test_timeout(self):
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="gh", timeout=30)):
            from prview.web.server import post_comment_reply

            assert post_comment_reply("org/repo", 1, 123, "Thanks!") is False

    def test_file_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            from prview.web.server import post_comment_reply

            assert post_comment_reply("org/repo", 1, 123, "Thanks!") is False


class TestSubmitPrReview:
    def test_comment_success(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            from prview.web.server import submit_pr_review

            assert submit_pr_review("org/repo", 1, "LGTM", "COMMENT") is True
            args = mock_run.call_args[0][0]
            assert "--comment" in args

    def test_approve(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            from prview.web.server import submit_pr_review

            assert submit_pr_review("org/repo", 1, "Approved", "APPROVE") is True
            args = mock_run.call_args[0][0]
            assert "--approve" in args

    def test_request_changes(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            from prview.web.server import submit_pr_review

            assert submit_pr_review("org/repo", 1, "Fix this", "REQUEST_CHANGES") is True
            args = mock_run.call_args[0][0]
            assert "--request-changes" in args

    def test_timeout(self):
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="gh", timeout=30)):
            from prview.web.server import submit_pr_review

            assert submit_pr_review("org/repo", 1, "LGTM", "COMMENT") is False

    def test_file_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            from prview.web.server import submit_pr_review

            assert submit_pr_review("org/repo", 1, "LGTM", "COMMENT") is False


class TestGetPrReviewComments:
    def test_no_output(self):
        with patch("prview.web.server.run_gh", return_value=None):
            from prview.web.server import get_pr_review_comments

            assert get_pr_review_comments("org/repo", 1) == []

    def test_parses_json_lines(self):
        output = (
            '{"id": 1, "path": "file.py", "body": "Fix", "user": "rev", "in_reply_to_id": null}\n'
            '{"id": 2, "path": "file.py", "body": "Reply", "user": "dev", "in_reply_to_id": 1}\n'
        )
        with patch("prview.web.server.run_gh", return_value=output):
            from prview.web.server import get_pr_review_comments

            comments = get_pr_review_comments("org/repo", 1)
            # Only top-level comments (in_reply_to_id is None/null)
            assert len(comments) == 1
            assert comments[0]["id"] == 1


class TestDatabaseInit:
    def test_db_tables_created(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "prs" in tables
        assert "sync_status" in tables

    def test_sync_status_initialized(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        row = conn.execute("SELECT * FROM sync_status WHERE id=1").fetchone()
        conn.close()
        assert row is not None
        assert row[2] == 0  # is_syncing = 0
