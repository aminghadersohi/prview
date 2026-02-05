"""Shortcut.com task management provider."""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from prview.web.task_providers import TaskProvider, register_provider

logger = logging.getLogger(__name__)

SHORTCUT_API_BASE = "https://api.app.shortcut.com/api/v3"


@register_provider
class ShortcutProvider(TaskProvider):
    """Shortcut.com integration via REST API v3."""

    id = "shortcut"
    name = "Shortcut"

    def _headers(self) -> dict:
        return {
            "Shortcut-Token": self._config.get("api_token", ""),
            "Content-Type": "application/json",
        }

    def validate(self) -> dict:
        """Test connection by fetching the current member."""
        token = self._config.get("api_token", "")
        if not token:
            return {"valid": False, "error": "API token is required"}

        try:
            resp = httpx.get(
                f"{SHORTCUT_API_BASE}/member",
                headers=self._headers(),
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                name = data.get("profile", {}).get("name", "Unknown")
                return {"valid": True, "error": None, "member": name}
            if resp.status_code == 401:
                return {"valid": False, "error": "Invalid API token"}
            return {"valid": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        except httpx.TimeoutException:
            return {"valid": False, "error": "Connection timed out"}
        except httpx.ConnectError:
            return {"valid": False, "error": "Could not connect to Shortcut API"}

    def get_stories_for_pr(self, pr_number: int, repo: str) -> list[dict]:
        """Search for stories linked to a PR by searching for the PR URL pattern."""
        token = self._config.get("api_token", "")
        if not token:
            return []

        # Search for stories mentioning this PR
        query = f"pr:{pr_number} repo:{repo}"
        try:
            resp = httpx.get(
                f"{SHORTCUT_API_BASE}/search/stories",
                params={"query": query, "page_size": 10},
                headers=self._headers(),
                timeout=15,
            )
            if resp.status_code != 200:
                logger.warning("Shortcut search failed: %s", resp.status_code)
                return []

            data = resp.json()
            stories = []
            for story in data.get("data", []):
                stories.append(self._format_story(story))
            return stories
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            logger.warning("Shortcut search error: %s", e)
            return []

    def link_pr_to_story(self, pr_number: int, repo: str, story_id: str) -> bool:
        """Add PR URL as external link on a story."""
        token = self._config.get("api_token", "")
        if not token:
            return False

        pr_url = f"https://github.com/{repo}/pull/{pr_number}"
        try:
            # First check if already linked
            story = self.get_story(story_id)
            if story and pr_url in [
                link.get("url", "") for link in story.get("external_links", [])
            ]:
                return True

            resp = httpx.put(
                f"{SHORTCUT_API_BASE}/stories/{story_id}",
                headers=self._headers(),
                json={
                    "external_links": [pr_url],
                },
                timeout=10,
            )
            return resp.status_code == 200
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            logger.warning("Shortcut link error: %s", e)
            return False

    def get_story(self, story_id: str) -> Optional[dict]:
        """Get story details by ID."""
        token = self._config.get("api_token", "")
        if not token:
            return None

        try:
            resp = httpx.get(
                f"{SHORTCUT_API_BASE}/stories/{story_id}",
                headers=self._headers(),
                timeout=10,
            )
            if resp.status_code != 200:
                return None

            return self._format_story(resp.json())
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            logger.warning("Shortcut get_story error: %s", e)
            return None

    def _format_story(self, raw: dict) -> dict:
        """Format a raw Shortcut story into a standard dict."""
        workflow_state = (
            raw.get("workflow_state", {}) if isinstance(raw.get("workflow_state"), dict) else {}
        )
        return {
            "id": str(raw.get("id", "")),
            "name": raw.get("name", ""),
            "state": workflow_state.get("name", raw.get("workflow_state_id", "unknown")),
            "type": raw.get("story_type", "feature"),
            "url": raw.get("app_url", ""),
            "epic": raw.get("epic_id"),
            "external_links": raw.get("external_links", []),
        }
