"""Tests for task management provider abstraction, Shortcut provider, and routes."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prview.web.server import (
    app,
    init_db,
)
from prview.web.task_providers import (
    PROVIDERS,
    TaskProvider,
    get_provider,
    list_providers,
    register_provider,
)
from prview.web.task_providers.shortcut import ShortcutProvider


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


# --- Provider registry tests ---


class TestProviderRegistry:
    def test_shortcut_registered(self):
        assert "shortcut" in PROVIDERS
        assert PROVIDERS["shortcut"] is ShortcutProvider

    def test_list_providers(self):
        providers = list_providers()
        ids = [p["id"] for p in providers]
        assert "shortcut" in ids

    def test_get_provider_exists(self):
        provider = get_provider("shortcut")
        assert provider is not None
        assert isinstance(provider, ShortcutProvider)

    def test_get_provider_unknown(self):
        assert get_provider("nonexistent") is None

    def test_register_provider_decorator(self):
        # Save original state
        original_providers = dict(PROVIDERS)

        @register_provider
        class FakeProvider(TaskProvider):
            id = "fake"
            name = "Fake"

            def validate(self):
                return {"valid": True, "error": None}

            def get_stories_for_pr(self, pr_number, repo):
                return []

            def link_pr_to_story(self, pr_number, repo, story_id):
                return True

            def get_story(self, story_id):
                return None

        assert "fake" in PROVIDERS
        instance = get_provider("fake")
        assert instance is not None
        assert instance.id == "fake"

        # Cleanup
        PROVIDERS.clear()
        PROVIDERS.update(original_providers)


# --- ShortcutProvider tests ---


class TestShortcutProvider:
    def test_validate_no_token(self):
        provider = ShortcutProvider()
        provider.configure({})
        result = provider.validate()
        assert result["valid"] is False
        assert "token" in result["error"].lower()

    def test_validate_success(self):
        provider = ShortcutProvider()
        provider.configure({"api_token": "tok_test"})

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"profile": {"name": "Test User"}}

        with patch("httpx.get", return_value=mock_resp):
            result = provider.validate()
            assert result["valid"] is True
            assert result["member"] == "Test User"

    def test_validate_invalid_token(self):
        provider = ShortcutProvider()
        provider.configure({"api_token": "bad_token"})

        mock_resp = MagicMock()
        mock_resp.status_code = 401

        with patch("httpx.get", return_value=mock_resp):
            result = provider.validate()
            assert result["valid"] is False
            assert "Invalid" in result["error"]

    def test_validate_timeout(self):
        import httpx

        provider = ShortcutProvider()
        provider.configure({"api_token": "tok_test"})

        with patch("httpx.get", side_effect=httpx.TimeoutException("timeout")):
            result = provider.validate()
            assert result["valid"] is False
            assert "timed out" in result["error"].lower()

    def test_validate_connect_error(self):
        import httpx

        provider = ShortcutProvider()
        provider.configure({"api_token": "tok_test"})

        with patch("httpx.get", side_effect=httpx.ConnectError("fail")):
            result = provider.validate()
            assert result["valid"] is False
            assert "connect" in result["error"].lower()

    def test_get_stories_for_pr_no_token(self):
        provider = ShortcutProvider()
        provider.configure({})
        assert provider.get_stories_for_pr(1, "org/repo") == []

    def test_get_stories_for_pr_success(self):
        provider = ShortcutProvider()
        provider.configure({"api_token": "tok_test"})

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {
                    "id": 1234,
                    "name": "Fix login bug",
                    "workflow_state": {"name": "In Progress"},
                    "story_type": "bug",
                    "app_url": "https://app.shortcut.com/org/story/1234",
                    "epic_id": None,
                    "external_links": [],
                }
            ]
        }

        with patch("httpx.get", return_value=mock_resp):
            stories = provider.get_stories_for_pr(42, "org/repo")
            assert len(stories) == 1
            assert stories[0]["id"] == "1234"
            assert stories[0]["name"] == "Fix login bug"
            assert stories[0]["state"] == "In Progress"
            assert stories[0]["type"] == "bug"

    def test_get_stories_for_pr_api_error(self):
        provider = ShortcutProvider()
        provider.configure({"api_token": "tok_test"})

        mock_resp = MagicMock()
        mock_resp.status_code = 500

        with patch("httpx.get", return_value=mock_resp):
            stories = provider.get_stories_for_pr(42, "org/repo")
            assert stories == []

    def test_get_story_success(self):
        provider = ShortcutProvider()
        provider.configure({"api_token": "tok_test"})

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": 5678,
            "name": "Add feature",
            "workflow_state": {"name": "Done"},
            "story_type": "feature",
            "app_url": "https://app.shortcut.com/org/story/5678",
            "epic_id": 10,
            "external_links": [],
        }

        with patch("httpx.get", return_value=mock_resp):
            story = provider.get_story("5678")
            assert story is not None
            assert story["id"] == "5678"
            assert story["state"] == "Done"
            assert story["epic"] == 10

    def test_get_story_not_found(self):
        provider = ShortcutProvider()
        provider.configure({"api_token": "tok_test"})

        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("httpx.get", return_value=mock_resp):
            assert provider.get_story("9999") is None

    def test_get_story_no_token(self):
        provider = ShortcutProvider()
        provider.configure({})
        assert provider.get_story("1234") is None

    def test_link_pr_to_story_no_token(self):
        provider = ShortcutProvider()
        provider.configure({})
        assert provider.link_pr_to_story(1, "org/repo", "1234") is False

    def test_link_pr_to_story_success(self):
        provider = ShortcutProvider()
        provider.configure({"api_token": "tok_test"})

        # Mock get_story to return a story without existing link
        mock_get_resp = MagicMock()
        mock_get_resp.status_code = 200
        mock_get_resp.json.return_value = {
            "id": 1234,
            "name": "Story",
            "workflow_state": {"name": "In Progress"},
            "story_type": "feature",
            "app_url": "https://app.shortcut.com/org/story/1234",
            "epic_id": None,
            "external_links": [],
        }

        mock_put_resp = MagicMock()
        mock_put_resp.status_code = 200

        with (
            patch("httpx.get", return_value=mock_get_resp),
            patch("httpx.put", return_value=mock_put_resp),
        ):
            assert provider.link_pr_to_story(42, "org/repo", "1234") is True

    def test_format_story_missing_workflow_state(self):
        provider = ShortcutProvider()
        result = provider._format_story(
            {
                "id": 1,
                "name": "Test",
                "workflow_state_id": 500,
                "story_type": "chore",
                "app_url": "",
                "external_links": [],
            }
        )
        assert result["state"] == 500  # Falls back to workflow_state_id


# --- Route tests ---


class TestTaskProviderRoutes:
    def test_list_providers_api(self, client):
        response = client.get("/api/task-providers")
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        ids = [p["id"] for p in data["providers"]]
        assert "shortcut" in ids

    def test_validate_unknown_provider(self, client):
        response = client.post(
            "/api/task-provider/validate",
            json={"provider_id": "nonexistent", "config": {}},
        )
        assert response.status_code == 400

    def test_validate_shortcut_no_token(self, client):
        response = client.post(
            "/api/task-provider/validate",
            json={"provider_id": "shortcut", "config": {}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False

    def test_validate_shortcut_success(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"profile": {"name": "Test"}}

        with patch("httpx.get", return_value=mock_resp):
            response = client.post(
                "/api/task-provider/validate",
                json={"provider_id": "shortcut", "config": {"api_token": "tok_test"}},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True

    def test_get_pr_stories(self, client):
        with patch("prview.web.server.fetch_linked_stories", return_value=[]):
            response = client.get("/api/pr/org/repo/1/stories")
            assert response.status_code == 200
            assert response.json() == {"stories": []}

    def test_get_pr_stories_with_data(self, client):
        mock_stories = [
            {"id": "1234", "name": "Fix bug", "state": "In Progress", "type": "bug", "url": ""}
        ]
        with patch("prview.web.server.fetch_linked_stories", return_value=mock_stories):
            response = client.get("/api/pr/org/repo/42/stories")
            assert response.status_code == 200
            data = response.json()
            assert len(data["stories"]) == 1
            assert data["stories"][0]["id"] == "1234"


class TestSettingsTaskProvider:
    def test_settings_page_shows_task_management(self, client):
        with patch("prview.web.server.check_claude_available", return_value={"available": False}):
            response = client.get("/settings")
            assert response.status_code == 200
            assert "Task Management" in response.text
            assert "task_provider" in response.text

    def test_save_settings_with_task_provider(self, client, tmp_path, monkeypatch):
        import yaml

        config_path = tmp_path / "config.yaml"
        monkeypatch.setattr("prview.web.server.CONFIG_PATH", config_path)
        response = client.post(
            "/settings/save",
            data={
                "include_orgs": "",
                "exclude_orgs": "",
                "include_repos": "",
                "exclude_repos": "",
                "refresh_interval": "60",
                "max_prs_per_repo": "10",
                "task_provider": "shortcut",
                "task_api_token": "tok_abc123",
            },
        )
        assert response.status_code == 200  # follows redirect
        assert config_path.exists()
        saved = yaml.safe_load(config_path.read_text())
        assert saved["task_provider"] == "shortcut"
        assert saved["task_provider_config"]["api_token"] == "tok_abc123"

    def test_save_settings_without_task_provider(self, client, tmp_path, monkeypatch):
        import yaml

        config_path = tmp_path / "config.yaml"
        monkeypatch.setattr("prview.web.server.CONFIG_PATH", config_path)
        response = client.post(
            "/settings/save",
            data={
                "include_orgs": "",
                "exclude_orgs": "",
                "include_repos": "",
                "exclude_repos": "",
                "refresh_interval": "60",
                "max_prs_per_repo": "10",
                "task_provider": "",
                "task_api_token": "",
            },
        )
        assert response.status_code == 200
        saved = yaml.safe_load(config_path.read_text())
        assert saved.get("task_provider") is None
        assert saved.get("task_provider_config") == {}


class TestLinkedStoriesInDb:
    def test_prs_table_has_linked_stories_column(self, tmp_db):
        import sqlite3

        conn = sqlite3.connect(tmp_db)
        cursor = conn.execute("PRAGMA table_info(prs)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()
        assert "linked_stories" in columns

    def test_get_prs_parses_linked_stories(self, tmp_db):
        import sqlite3

        conn = sqlite3.connect(tmp_db)
        stories = json.dumps([{"id": "1234", "name": "Story", "state": "Done"}])
        conn.execute(
            """INSERT INTO prs
            (repo, number, title, author, url, is_review_request, linked_stories)
            VALUES (?, ?, ?, ?, ?, 0, ?)""",
            ("org/repo", 1, "PR title", "dev", "https://github.com/org/repo/pull/1", stories),
        )
        conn.commit()
        conn.close()

        from prview.web.server import get_prs_from_db

        data = get_prs_from_db()
        prs = data["my_prs"].get("org/repo", [])
        assert len(prs) == 1
        assert len(prs[0]["linked_stories"]) == 1
        assert prs[0]["linked_stories"][0]["id"] == "1234"

    def test_get_prs_handles_empty_linked_stories(self, tmp_db):
        import sqlite3

        conn = sqlite3.connect(tmp_db)
        conn.execute(
            """INSERT INTO prs
            (repo, number, title, author, url, is_review_request)
            VALUES (?, ?, ?, ?, ?, 0)""",
            ("org/repo", 1, "PR title", "dev", "https://github.com/org/repo/pull/1"),
        )
        conn.commit()
        conn.close()

        from prview.web.server import get_prs_from_db

        data = get_prs_from_db()
        prs = data["my_prs"].get("org/repo", [])
        assert len(prs) == 1
        assert prs[0]["linked_stories"] == []


class TestFetchLinkedStories:
    def test_no_provider_configured(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.yaml"
        monkeypatch.setattr("prview.web.server.CONFIG_PATH", config_path)

        from prview.web.server import fetch_linked_stories

        assert fetch_linked_stories("org/repo", 1) == []

    def test_provider_returns_stories(self, tmp_path, monkeypatch):
        mock_provider = MagicMock()
        mock_provider.get_stories_for_pr.return_value = [
            {"id": "1", "name": "Story", "state": "Done"}
        ]

        with patch("prview.web.server.get_task_provider", return_value=mock_provider):
            from prview.web.server import fetch_linked_stories

            stories = fetch_linked_stories("org/repo", 42)
            assert len(stories) == 1

    def test_provider_error_returns_empty(self, tmp_path, monkeypatch):
        mock_provider = MagicMock()
        mock_provider.get_stories_for_pr.side_effect = RuntimeError("API down")

        with patch("prview.web.server.get_task_provider", return_value=mock_provider):
            from prview.web.server import fetch_linked_stories

            stories = fetch_linked_stories("org/repo", 42)
            assert stories == []
