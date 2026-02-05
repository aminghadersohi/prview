"""
Task management provider abstraction and registry.

Supports pluggable task management integrations (Shortcut, Jira, etc.).
New providers register by adding to the PROVIDERS dict.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class TaskProvider(ABC):
    """Base class for task management integrations."""

    id: str = ""
    name: str = ""

    def __init__(self):
        self._config: dict = {}

    def configure(self, config: dict) -> None:
        """Set provider configuration (API token, workspace, etc.)."""
        self._config = config

    @abstractmethod
    def validate(self) -> dict:
        """Check connection to the provider.

        Returns:
            dict with keys: valid (bool), error (str | None)
        """

    @abstractmethod
    def get_stories_for_pr(self, pr_number: int, repo: str) -> list[dict]:
        """Find linked stories/tasks for a PR.

        Returns:
            List of story dicts with keys: id, name, state, type, url
        """

    @abstractmethod
    def link_pr_to_story(self, pr_number: int, repo: str, story_id: str) -> bool:
        """Link a PR to a story/task. Returns True on success."""

    @abstractmethod
    def get_story(self, story_id: str) -> Optional[dict]:
        """Get story details by ID.

        Returns:
            Story dict with keys: id, name, state, type, url, epic
            or None if not found.
        """


# Registry: maps provider ID to provider class
PROVIDERS: dict[str, type[TaskProvider]] = {}


def register_provider(cls: type[TaskProvider]) -> type[TaskProvider]:
    """Decorator to register a task provider."""
    PROVIDERS[cls.id] = cls
    return cls


def get_provider(provider_id: str) -> Optional[TaskProvider]:
    """Instantiate a registered provider by ID."""
    cls = PROVIDERS.get(provider_id)
    if cls is None:
        return None
    return cls()


def list_providers() -> list[dict]:
    """List all registered providers with their metadata."""
    return [{"id": cls.id, "name": cls.name} for cls in PROVIDERS.values()]
