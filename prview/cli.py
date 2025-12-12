#!/usr/bin/env python3
"""
prview: A beautiful TUI and web dashboard for GitHub PRs and CI status.

Features:
- View your open PRs across multiple repos
- See PRs that need your review
- CI status and review status at a glance
- Keyboard navigation
- Watch mode with auto-refresh
- Web UI with real-time updates
- Configurable via YAML
"""

import json
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich import box
from rich.align import Align

try:
    import yaml
except ImportError:
    print("Installing required package: pyyaml")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml", "-q"])
    import yaml

console = Console()

# Default config path
CONFIG_PATH = Path.home() / ".config" / "prview" / "config.yaml"

# Color scheme
COLORS = {
    "header": "bold bright_cyan",
    "subheader": "bold cyan",
    "success": "bright_green",
    "failure": "bright_red",
    "warning": "bright_yellow",
    "pending": "yellow",
    "muted": "dim white",
    "highlight": "bold white on blue",
    "author": "magenta",
    "repo": "cyan",
    "number": "bright_blue",
    "draft": "dim italic",
    "title": "white",
    "border": "bright_black",
}


@dataclass
class Config:
    """Dashboard configuration."""

    github_user: str = ""
    github_email: str = ""
    refresh_interval: int = 60  # seconds
    include_orgs: list[str] = field(default_factory=list)
    exclude_orgs: list[str] = field(default_factory=list)
    include_repos: list[str] = field(default_factory=list)
    exclude_repos: list[str] = field(default_factory=list)
    show_drafts: bool = True
    max_prs_per_repo: int = 10

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Config":
        """Load config from YAML file."""
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            return cls(
                github_user=data.get("github_user", ""),
                github_email=data.get("github_email", ""),
                refresh_interval=data.get("refresh_interval", 60),
                include_orgs=data.get("include_orgs", []),
                exclude_orgs=data.get("exclude_orgs", []),
                include_repos=data.get("include_repos", []),
                exclude_repos=data.get("exclude_repos", []),
                show_drafts=data.get("show_drafts", True),
                max_prs_per_repo=data.get("max_prs_per_repo", 10),
            )
        return cls()

    def save(self, path: Path = CONFIG_PATH):
        """Save config to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "github_user": self.github_user,
            "github_email": self.github_email,
            "refresh_interval": self.refresh_interval,
            "include_orgs": self.include_orgs,
            "exclude_orgs": self.exclude_orgs,
            "include_repos": self.include_repos,
            "exclude_repos": self.exclude_repos,
            "show_drafts": self.show_drafts,
            "max_prs_per_repo": self.max_prs_per_repo,
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


@dataclass
class PR:
    """Pull request data."""

    repo: str
    number: int
    title: str
    author: str
    state: str
    draft: bool
    ci_status: str  # success, failure, pending, none
    review_status: str  # approved, changes_requested, review_required, pending, none
    url: str
    updated: str
    additions: int = 0
    deletions: int = 0
    comments: int = 0
    review_requested: bool = False  # True if current user's review is requested


@dataclass
class DashboardState:
    """Current state of the dashboard."""

    my_prs: dict[str, list[PR]] = field(default_factory=dict)  # repo -> PRs
    review_requests: list[PR] = field(default_factory=list)
    selected_index: int = 0
    selected_section: str = "my_prs"  # "my_prs" or "review_requests"
    all_prs_flat: list[PR] = field(default_factory=list)
    last_refresh: Optional[datetime] = None
    is_loading: bool = False
    error: Optional[str] = None


def run_gh(args: list[str], timeout: int = 30) -> Optional[str]:
    """Run gh CLI command and return output."""
    try:
        result = subprocess.run(["gh"] + args, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return result.stdout
        return None
    except Exception:
        return None


def get_github_user() -> str:
    """Get the current GitHub username."""
    output = run_gh(["api", "user", "--jq", ".login"])
    return output.strip() if output else ""


def get_pr_details(repo: str, pr_number: int) -> dict:
    """Get detailed PR info including CI and review status."""
    output = run_gh(
        [
            "pr",
            "view",
            str(pr_number),
            "--repo",
            repo,
            "--json",
            "reviewDecision,additions,deletions,comments,reviewRequests,statusCheckRollup",
        ]
    )
    if not output:
        return {}
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {}


def get_ci_status_from_checks(checks: list) -> str:
    """Determine CI status from status check rollup."""
    if not checks:
        return "none"

    states = []
    for check in checks:
        state = check.get("state", "").upper()
        conclusion = check.get("conclusion", "").upper()
        status = check.get("status", "").upper()

        # Real failures
        if conclusion in ("FAILURE", "ERROR", "TIMED_OUT"):
            states.append("FAILURE")
        # Success states (CANCELLED and SKIPPED are not failures)
        elif conclusion in ("SUCCESS", "SKIPPED", "CANCELLED", "NEUTRAL"):
            states.append("SUCCESS")
        # Still running
        elif status in ("IN_PROGRESS", "QUEUED", "PENDING", "WAITING") or state == "PENDING":
            states.append("PENDING")
        elif conclusion == "":
            states.append("PENDING")
        else:
            states.append("SUCCESS")

    if "FAILURE" in states:
        return "failure"
    if "PENDING" in states:
        return "pending"
    if all(s == "SUCCESS" for s in states):
        return "success"
    return "pending"


def get_review_status(decision: str) -> str:
    """Convert review decision to status."""
    mapping = {
        "APPROVED": "approved",
        "CHANGES_REQUESTED": "changes_requested",
        "REVIEW_REQUIRED": "review_required",
    }
    return mapping.get(decision, "pending")


def fetch_my_prs(config: Config) -> dict[str, list[PR]]:
    """Fetch all open PRs authored by the current user using search (fast)."""
    # Use gh search which is much faster than iterating repos
    output = run_gh(
        [
            "search",
            "prs",
            "--author",
            "@me",
            "--state",
            "open",
            "--json",
            "number,title,author,repository,isDraft,url,updatedAt",
            "--limit",
            "100",
        ],
        timeout=60,
    )

    if not output:
        return {}

    try:
        prs_data = json.loads(output)
    except json.JSONDecodeError:
        return {}

    result: dict[str, list[PR]] = {}

    for pr_data in prs_data:
        repo = pr_data["repository"]["nameWithOwner"]

        # Apply org/repo filters
        org = repo.split("/")[0] if "/" in repo else ""

        # Check exclusions
        if repo in config.exclude_repos:
            continue
        if org in config.exclude_orgs:
            continue

        # Check inclusions (if specified)
        if config.include_repos or config.include_orgs:
            in_included_repo = repo in config.include_repos
            in_included_org = org in config.include_orgs
            if not (in_included_repo or in_included_org):
                continue

        # Filter drafts if needed
        if not config.show_drafts and pr_data["isDraft"]:
            continue

        pr_number = pr_data["number"]

        # Fetch detailed info (CI status, review status, etc.)
        details = get_pr_details(repo, pr_number)

        ci_status = get_ci_status_from_checks(details.get("statusCheckRollup", []))
        review_status = get_review_status(details.get("reviewDecision", ""))

        pr = PR(
            repo=repo,
            number=pr_number,
            title=pr_data["title"],
            author=pr_data["author"]["login"],
            state="OPEN",
            draft=pr_data["isDraft"],
            ci_status=ci_status,
            review_status=review_status,
            url=pr_data["url"],
            updated=pr_data["updatedAt"][:10],
            additions=details.get("additions", 0),
            deletions=details.get("deletions", 0),
            comments=len(details.get("comments", [])),
        )

        if repo not in result:
            result[repo] = []
        result[repo].append(pr)

    # Sort repos and limit PRs per repo
    sorted_result: dict[str, list[PR]] = {}
    for repo in sorted(result.keys()):
        sorted_result[repo] = result[repo][: config.max_prs_per_repo]

    return sorted_result


def fetch_review_requests(config: Config) -> list[PR]:
    """Fetch PRs where the current user's review is requested."""
    prs = []

    # Search for PRs requesting review from current user
    output = run_gh(
        [
            "search",
            "prs",
            "--review-requested",
            "@me",
            "--state",
            "open",
            "--json",
            "number,title,author,repository,isDraft,url,updatedAt",
            "--limit",
            "50",
        ]
    )

    if not output:
        return prs

    try:
        prs_data = json.loads(output)
    except json.JSONDecodeError:
        return prs

    for pr_data in prs_data:
        repo = pr_data["repository"]["nameWithOwner"]

        # Check org/repo exclusions
        if repo in config.exclude_repos:
            continue

        org = repo.split("/")[0] if "/" in repo else ""
        if org in config.exclude_orgs:
            continue

        # If we have include filters, check them
        if config.include_repos or config.include_orgs:
            in_included_repo = repo in config.include_repos
            in_included_org = org in config.include_orgs
            if not (in_included_repo or in_included_org):
                continue

        if not config.show_drafts and pr_data["isDraft"]:
            continue

        pr_number = pr_data["number"]
        details = get_pr_details(repo, pr_number)

        ci_status = get_ci_status_from_checks(details.get("statusCheckRollup", []))
        review_status = get_review_status(details.get("reviewDecision", ""))

        pr = PR(
            repo=repo,
            number=pr_number,
            title=pr_data["title"],
            author=pr_data["author"]["login"],
            state="OPEN",
            draft=pr_data["isDraft"],
            ci_status=ci_status,
            review_status=review_status,
            url=pr_data["url"],
            updated=pr_data["updatedAt"][:10],
            additions=details.get("additions", 0),
            deletions=details.get("deletions", 0),
            comments=len(details.get("comments", [])),
            review_requested=True,
        )
        prs.append(pr)

    return prs


def ci_icon(status: str) -> Text:
    """Return colored icon for CI status."""
    icons = {
        "success": ("✓", COLORS["success"]),
        "failure": ("✗", COLORS["failure"]),
        "pending": ("◐", COLORS["pending"]),
        "none": ("○", COLORS["muted"]),
    }
    icon, color = icons.get(status, ("?", COLORS["muted"]))
    return Text(icon, style=color)


def review_icon(status: str) -> Text:
    """Return colored icon for review status."""
    icons = {
        "approved": ("✓", COLORS["success"]),
        "changes_requested": ("✗", COLORS["failure"]),
        "review_required": ("●", COLORS["warning"]),
        "pending": ("○", COLORS["muted"]),
        "none": ("○", COLORS["muted"]),
    }
    icon, color = icons.get(status, ("?", COLORS["muted"]))
    return Text(icon, style=color)


def format_diff_stats(additions: int, deletions: int) -> Text:
    """Format diff stats with colors."""
    text = Text()
    if additions:
        text.append(f"+{additions}", style=COLORS["success"])
    if additions and deletions:
        text.append(" ")
    if deletions:
        text.append(f"-{deletions}", style=COLORS["failure"])
    if not additions and not deletions:
        text.append("—", style=COLORS["muted"])
    return text


def render_pr_table(
    title: str,
    prs: list[PR],
    selected_index: int,
    is_active_section: bool,
    show_repo: bool = False,
    start_index: int = 0,
) -> Table:
    """Render a table of PRs."""
    table = Table(
        title=f"[{COLORS['header']}]{title}[/]",
        box=box.ROUNDED,
        show_header=True,
        header_style=COLORS["subheader"],
        title_justify="left",
        expand=True,
        border_style=COLORS["border"] if not is_active_section else COLORS["repo"],
        padding=(0, 1),
    )

    table.add_column("", width=2)  # Selection indicator
    table.add_column("#", style=COLORS["number"], width=6, justify="right")
    if show_repo:
        table.add_column("Repo", style=COLORS["repo"], max_width=25)
    table.add_column("Title", style=COLORS["title"], min_width=30, max_width=50)
    table.add_column("Author", style=COLORS["author"], width=15)
    table.add_column("CI", justify="center", width=3)
    table.add_column("Review", justify="center", width=3)
    table.add_column("Changes", justify="right", width=12)
    table.add_column("Updated", style=COLORS["muted"], width=10)

    if not prs:
        cols = 9 if show_repo else 8
        table.add_row(*[""] * (cols - 1), f"[{COLORS['muted']}]No PRs[/]")
    else:
        for i, pr in enumerate(prs):
            actual_index = start_index + i
            is_selected = is_active_section and actual_index == selected_index

            selector = "▶" if is_selected else " "
            selector_style = COLORS["highlight"] if is_selected else ""

            draft_prefix = f"[{COLORS['draft']}](draft)[/] " if pr.draft else ""
            title_text = f"{draft_prefix}{pr.title[:47]}{'...' if len(pr.title) > 47 else ''}"

            row_style = "reverse" if is_selected else ""

            row = [
                Text(selector, style=selector_style),
                str(pr.number),
            ]
            if show_repo:
                short_repo = pr.repo.split("/")[-1] if "/" in pr.repo else pr.repo
                row.append(short_repo)
            row.extend(
                [
                    title_text,
                    pr.author,
                    ci_icon(pr.ci_status),
                    review_icon(pr.review_status),
                    format_diff_stats(pr.additions, pr.deletions),
                    pr.updated,
                ]
            )

            table.add_row(*row, style=row_style)

    return table


def render_my_prs_section(state: DashboardState, is_active: bool) -> Group:
    """Render the My PRs section with repo grouping."""
    tables = []
    current_index = 0

    for repo, prs in state.my_prs.items():
        table = Table(
            title=f"[{COLORS['repo']}]{repo}[/]",
            box=box.SIMPLE,
            show_header=current_index == 0,
            header_style=COLORS["subheader"],
            title_justify="left",
            expand=True,
            padding=(0, 1),
        )

        table.add_column("", width=2)
        table.add_column("#", style=COLORS["number"], width=6, justify="right")
        table.add_column("Title", style=COLORS["title"], min_width=30, max_width=55)
        table.add_column("CI", justify="center", width=3)
        table.add_column("Review", justify="center", width=3)
        table.add_column("Changes", justify="right", width=12)
        table.add_column("Updated", style=COLORS["muted"], width=10)

        for i, pr in enumerate(prs):
            actual_index = current_index + i
            is_selected = (
                is_active
                and state.selected_section == "my_prs"
                and actual_index == state.selected_index
            )

            selector = "▶" if is_selected else " "
            draft_prefix = f"[{COLORS['draft']}](draft)[/] " if pr.draft else ""
            title_text = f"{draft_prefix}{pr.title[:52]}{'...' if len(pr.title) > 52 else ''}"

            row_style = "reverse" if is_selected else ""

            table.add_row(
                Text(selector, style=COLORS["highlight"] if is_selected else ""),
                str(pr.number),
                title_text,
                ci_icon(pr.ci_status),
                review_icon(pr.review_status),
                format_diff_stats(pr.additions, pr.deletions),
                pr.updated,
                style=row_style,
            )

        current_index += len(prs)
        tables.append(table)

    if not tables:
        empty = Table(box=None, expand=True)
        empty.add_row(f"[{COLORS['muted']}]No open PRs[/]")
        tables.append(empty)

    return Group(*tables)


def render_header(state: DashboardState, config: Config) -> Panel:
    """Render the dashboard header."""
    title = Text()
    title.append("  prview - GitHub PR Dashboard  ", style="bold bright_white on blue")

    status = Text()
    if state.is_loading:
        status.append("  ◐ Loading...", style=COLORS["pending"])
    elif state.last_refresh:
        status.append(
            f"  Last refresh: {state.last_refresh.strftime('%H:%M:%S')}",
            style=COLORS["muted"],
        )

    if state.error:
        status.append(f"  ⚠ {state.error}", style=COLORS["failure"])

    header_content = Text()
    header_content.append_text(title)
    header_content.append_text(status)

    return Panel(
        Align.center(header_content),
        box=box.DOUBLE,
        style=COLORS["border"],
        padding=(0, 1),
    )


def render_legend() -> Panel:
    """Render the legend and keyboard shortcuts."""
    legend = Text()
    legend.append("CI: ", style="bold")
    legend.append("✓", style=COLORS["success"])
    legend.append(" pass  ")
    legend.append("✗", style=COLORS["failure"])
    legend.append(" fail  ")
    legend.append("◐", style=COLORS["pending"])
    legend.append(" running  ")
    legend.append("○", style=COLORS["muted"])
    legend.append(" none")

    legend.append("   │   ", style=COLORS["muted"])

    legend.append("Review: ", style="bold")
    legend.append("✓", style=COLORS["success"])
    legend.append(" approved  ")
    legend.append("✗", style=COLORS["failure"])
    legend.append(" changes  ")
    legend.append("●", style=COLORS["warning"])
    legend.append(" required  ")
    legend.append("○", style=COLORS["muted"])
    legend.append(" pending")

    keys = Text()
    keys.append("Keys: ", style="bold")
    keys.append("↑↓", style=COLORS["number"])
    keys.append(" navigate  ")
    keys.append("Tab", style=COLORS["number"])
    keys.append(" switch section  ")
    keys.append("Enter", style=COLORS["number"])
    keys.append(" open in browser  ")
    keys.append("r", style=COLORS["number"])
    keys.append(" refresh  ")
    keys.append("q", style=COLORS["number"])
    keys.append(" quit")

    content = Group(legend, keys)
    return Panel(content, box=box.ROUNDED, style=COLORS["muted"], padding=(0, 1))


def render_dashboard(state: DashboardState, config: Config) -> Group:
    """Render the complete dashboard."""
    # Build flat list of all PRs for navigation
    state.all_prs_flat = []
    for repo, prs in state.my_prs.items():
        state.all_prs_flat.extend(prs)

    components = [
        render_header(state, config),
        render_legend(),
        Text(),  # Spacer
    ]

    # My PRs section
    my_prs_title = f"  My Open PRs ({len(state.all_prs_flat)})  "
    is_my_prs_active = state.selected_section == "my_prs"

    my_prs_panel = Panel(
        render_my_prs_section(state, is_my_prs_active),
        title=f"[{'bold bright_white on blue' if is_my_prs_active else COLORS['header']}]{my_prs_title}[/]",
        box=box.ROUNDED,
        border_style=COLORS["repo"] if is_my_prs_active else COLORS["border"],
        padding=(0, 0),
    )
    components.append(my_prs_panel)
    components.append(Text())  # Spacer

    # Review Requests section
    review_title = f"  Needs My Review ({len(state.review_requests)})  "
    is_review_active = state.selected_section == "review_requests"

    review_table = render_pr_table(
        "",
        state.review_requests,
        state.selected_index,
        is_review_active,
        show_repo=True,
        start_index=0,
    )

    review_panel = Panel(
        review_table,
        title=f"[{'bold bright_white on magenta' if is_review_active else COLORS['header']}]{review_title}[/]",
        box=box.ROUNDED,
        border_style="magenta" if is_review_active else COLORS["border"],
        padding=(0, 0),
    )
    components.append(review_panel)

    return Group(*components)


def get_selected_pr(state: DashboardState) -> Optional[PR]:
    """Get the currently selected PR."""
    if state.selected_section == "my_prs":
        if 0 <= state.selected_index < len(state.all_prs_flat):
            return state.all_prs_flat[state.selected_index]
    else:
        if 0 <= state.selected_index < len(state.review_requests):
            return state.review_requests[state.selected_index]
    return None


def refresh_data(state: DashboardState, config: Config, show_progress: bool = False):
    """Refresh all PR data."""
    state.is_loading = True
    state.error = None

    try:
        if show_progress:
            with console.status("[cyan]Searching for your PRs...[/]"):
                state.my_prs = fetch_my_prs(config)
            with console.status("[cyan]Finding PRs needing your review...[/]"):
                state.review_requests = fetch_review_requests(config)
        else:
            state.my_prs = fetch_my_prs(config)
            state.review_requests = fetch_review_requests(config)
        state.last_refresh = datetime.now()
    except Exception as e:
        state.error = str(e)
    finally:
        state.is_loading = False


def run_interactive(config: Config):
    """Run the dashboard in interactive mode with keyboard navigation."""
    try:
        from pynput import keyboard

        has_pynput = True
    except ImportError:
        has_pynput = False

    state = DashboardState()

    # Initial data fetch
    refresh_data(state, config, show_progress=True)

    if not has_pynput:
        # Fallback: just display once without interactivity
        console.clear()
        console.print(render_dashboard(state, config))
        console.print()
        console.print(
            f"[{COLORS['muted']}]Install pynput for keyboard navigation: pip install pynput[/]"
        )
        console.print(f"[{COLORS['muted']}]Press Ctrl+C to exit[/]")

        try:
            while True:
                time.sleep(config.refresh_interval)
                refresh_data(state, config)
                console.clear()
                console.print(render_dashboard(state, config))
        except KeyboardInterrupt:
            pass
        return

    # Interactive mode with pynput
    running = True
    needs_refresh = False
    needs_redraw = True

    def on_press(key):
        nonlocal running, needs_refresh, needs_redraw, state

        try:
            char = key.char
        except AttributeError:
            char = None

        if char == "q":
            running = False
        elif char == "r":
            needs_refresh = True
        elif key == keyboard.Key.tab:
            # Switch sections
            if state.selected_section == "my_prs":
                state.selected_section = "review_requests"
                state.selected_index = 0
            else:
                state.selected_section = "my_prs"
                state.selected_index = 0
            needs_redraw = True
        elif key == keyboard.Key.up:
            state.selected_index = max(0, state.selected_index - 1)
            needs_redraw = True
        elif key == keyboard.Key.down:
            max_index = (
                len(state.all_prs_flat) - 1
                if state.selected_section == "my_prs"
                else len(state.review_requests) - 1
            )
            state.selected_index = min(max_index, state.selected_index + 1)
            needs_redraw = True
        elif key == keyboard.Key.enter:
            pr = get_selected_pr(state)
            if pr:
                webbrowser.open(pr.url)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    last_refresh_time = time.time()

    try:
        while running:
            if needs_refresh or (time.time() - last_refresh_time > config.refresh_interval):
                refresh_data(state, config)
                last_refresh_time = time.time()
                needs_refresh = False
                needs_redraw = True

            if needs_redraw:
                console.clear()
                console.print(render_dashboard(state, config))
                needs_redraw = False

            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()


def run_once(config: Config):
    """Run the dashboard once and exit."""
    state = DashboardState()
    refresh_data(state, config, show_progress=True)
    console.print(render_dashboard(state, config))


def create_default_config():
    """Create a default config file."""
    config = Config(
        github_user="",
        github_email="",
        refresh_interval=60,
        include_orgs=[],
        exclude_orgs=[],
        include_repos=[],
        exclude_repos=[],
        show_drafts=True,
        max_prs_per_repo=10,
    )
    config.save()
    return config


def run_server(host: str = "127.0.0.1", port: int = 8420):
    """Run the web server."""
    try:
        import uvicorn
    except ImportError:
        console.print(f"[{COLORS['failure']}]Installing web dependencies...[/]")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "fastapi",
                "uvicorn",
                "jinja2",
                "-q",
            ]
        )
        import uvicorn

    from prview.web.server import app, init_db

    init_db()
    console.print(f"[{COLORS['success']}]Starting prview web server...[/]")
    console.print(f"[{COLORS['repo']}]Open http://{host}:{port} in your browser[/]")
    console.print(f"[{COLORS['muted']}]Press Ctrl+C to stop[/]")
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Beautiful TUI and web dashboard for GitHub PRs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  prview                     # Run TUI once
  prview --watch             # TUI with auto-refresh
  prview serve               # Start web UI server
  prview serve --port 3000   # Web UI on custom port
  prview --init              # Create default config
        """,
    )

    subparsers = parser.add_subparsers(dest="command")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start web UI server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    serve_parser.add_argument("--port", "-p", type=int, default=8420, help="Port to run on")

    # Main parser args
    parser.add_argument("--watch", "-w", action="store_true", help="Watch mode with auto-refresh")
    parser.add_argument("--init", action="store_true", help="Create default config file")
    parser.add_argument("--config", "-c", type=Path, default=CONFIG_PATH, help="Config file path")
    parser.add_argument("--repos", "-r", nargs="+", help="Override repos to check")
    parser.add_argument("--orgs", "-o", nargs="+", help="Override orgs to check")
    parser.add_argument(
        "--interval", "-i", type=int, default=60, help="Refresh interval in seconds"
    )
    parser.add_argument("--version", "-V", action="version", version="%(prog)s 0.1.0")

    args = parser.parse_args()

    # Handle serve command
    if args.command == "serve":
        run_server(host=args.host, port=args.port)
        return

    if args.init:
        config = create_default_config()
        console.print(f"[{COLORS['success']}]Created config at {CONFIG_PATH}[/]")
        console.print(f"[{COLORS['muted']}]Edit to customize your dashboard[/]")
        return

    # Load or create config
    if args.config.exists():
        config = Config.load(args.config)
    else:
        config = create_default_config()
        console.print(f"[{COLORS['muted']}]Created default config at {CONFIG_PATH}[/]")

    # Override with command line args
    if args.repos:
        config.include_repos = args.repos
        config.include_orgs = []
    if args.orgs:
        config.include_orgs = args.orgs
    if args.interval:
        config.refresh_interval = args.interval

    if args.watch:
        run_interactive(config)
    else:
        run_once(config)


if __name__ == "__main__":
    main()
