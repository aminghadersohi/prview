"""Tests for PR chat persistence and session management."""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from prview.web.server import (
    app,
    get_pr_chat_messages,
    get_pr_session,
    init_db,
    save_chat_message,
    save_pr_session,
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


class TestChatMessagePersistence:
    def test_save_and_retrieve_messages(self):
        repo = "test-org/test-repo"
        pr_number = 42

        # Save messages
        save_chat_message(repo, pr_number, "user", "Hello agent")
        save_chat_message(
            repo,
            pr_number,
            "assistant",
            "Hello! How can I help?",
            model="claude-sonnet-4-20250514",
            session_id="sess-123",
            cost_usd=0.005,
            duration_ms=1200,
            input_tokens=50,
            output_tokens=20,
        )

        # Retrieve
        messages = get_pr_chat_messages(repo, pr_number)
        assert len(messages) == 2

        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello agent"

        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hello! How can I help?"
        assert messages[1]["model"] == "claude-sonnet-4-20250514"
        assert messages[1]["session_id"] == "sess-123"
        assert messages[1]["cost_usd"] == pytest.approx(0.005)
        assert messages[1]["input_tokens"] == 50

    def test_messages_ordered_by_time(self):
        repo = "org/repo"
        save_chat_message(repo, 1, "user", "first")
        save_chat_message(repo, 1, "assistant", "second")
        save_chat_message(repo, 1, "user", "third")

        messages = get_pr_chat_messages(repo, 1)
        assert [m["content"] for m in messages] == ["first", "second", "third"]

    def test_messages_isolated_by_pr(self):
        save_chat_message("org/repo", 1, "user", "PR 1 msg")
        save_chat_message("org/repo", 2, "user", "PR 2 msg")

        msgs1 = get_pr_chat_messages("org/repo", 1)
        msgs2 = get_pr_chat_messages("org/repo", 2)
        assert len(msgs1) == 1
        assert len(msgs2) == 1
        assert msgs1[0]["content"] == "PR 1 msg"
        assert msgs2[0]["content"] == "PR 2 msg"

    def test_empty_messages(self):
        messages = get_pr_chat_messages("nonexistent/repo", 999)
        assert messages == []


class TestSessionPersistence:
    def test_save_and_get_session(self):
        repo = "org/repo"
        save_pr_session(repo, 1, "session-abc", "claude-sonnet-4-20250514")

        session = get_pr_session(repo, 1)
        assert session is not None
        assert session["session_id"] == "session-abc"
        assert session["model"] == "claude-sonnet-4-20250514"

    def test_session_upsert(self):
        repo = "org/repo"
        save_pr_session(repo, 1, "session-old", "model-old")
        save_pr_session(repo, 1, "session-new", "model-new")

        session = get_pr_session(repo, 1)
        assert session["session_id"] == "session-new"
        assert session["model"] == "model-new"

    def test_no_session(self):
        session = get_pr_session("nonexistent/repo", 999)
        assert session is None


class TestChatRoutes:
    def test_chat_page_loads(self, client):
        with patch("prview.web.server.get_pr_basic_info") as mock_info:
            mock_info.return_value = {
                "title": "Test PR",
                "url": "https://github.com/org/repo/pull/1",
            }
            resp = client.get("/pr/org/repo/1/chat")
            assert resp.status_code == 200
            assert "Test PR" in resp.text
            assert "Agent" in resp.text or "agent" in resp.text.lower()

    def test_chat_page_with_prompt_prefill(self, client):
        with patch("prview.web.server.get_pr_basic_info") as mock_info:
            mock_info.return_value = {
                "title": "Test PR",
                "url": "https://github.com/org/repo/pull/1",
            }
            resp = client.get("/pr/org/repo/1/chat?prompt=Fix+the+bug")
            assert resp.status_code == 200
            assert "Fix the bug" in resp.text

    def test_chat_history_api(self, client):
        save_chat_message("org/repo", 1, "user", "hello")
        save_chat_message("org/repo", 1, "assistant", "hi there")

        resp = client.get("/api/pr/org/repo/1/chat/history")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["messages"]) == 2

    def test_chat_clear_api(self, client):
        save_chat_message("org/repo", 1, "user", "hello")
        save_pr_session("org/repo", 1, "sess-123")

        resp = client.post(
            "/api/pr/org/repo/1/chat/clear",
            headers={"Content-Type": "application/json"},
            content="{}",
        )
        assert resp.status_code == 200

        messages = get_pr_chat_messages("org/repo", 1)
        assert len(messages) == 0

        session = get_pr_session("org/repo", 1)
        assert session is None

    def test_chat_send_empty_message(self, client):
        resp = client.post(
            "/pr/org/repo/1/chat/send",
            headers={"Content-Type": "application/json"},
            content=json.dumps({"message": "", "model": "claude-sonnet-4-20250514"}),
        )
        assert resp.status_code == 400


class TestWorktreeRoutes:
    def test_worktree_status_no_worktree(self, client):
        resp = client.get("/api/pr/org/repo/1/worktree/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["exists"] is False

    def test_worktree_diff_no_worktree(self, client):
        resp = client.get("/api/pr/org/repo/1/worktree/diff")
        assert resp.status_code == 200
        data = resp.json()
        assert data["diff"] == ""

    def test_worktree_push_no_message(self, client):
        resp = client.post(
            "/api/pr/org/repo/1/worktree/push",
            headers={"Content-Type": "application/json"},
            content=json.dumps({"message": ""}),
        )
        assert resp.status_code == 400

    def test_worktree_setup_no_branch(self, client):
        with patch("prview.web.server.get_pr_branch_name") as mock_branch:
            mock_branch.return_value = None
            resp = client.post(
                "/api/pr/org/repo/1/worktree/setup",
                headers={"Content-Type": "application/json"},
                content="{}",
            )
            assert resp.status_code == 400
            assert "Could not determine branch" in resp.json()["error"]
