"""
prview web server with real-time updates via SSE.
Beautiful dark theme UI inspired by VS Code/Atom.
"""

import asyncio
import json
import logging
import os
import re
import sqlite3
import subprocess
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx
import yaml
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import prview.web.task_providers.shortcut  # noqa: F401
from prview.web.claude_cli import (
    CLAUDE_MODELS,
    DEFAULT_MODEL,
    check_claude_available,
    stream_claude_response,
)
from prview.web.task_providers import get_provider, list_providers
from prview.web.worktree import (
    commit_and_push,
    ensure_worktree,
    get_worktree_diff,
    get_worktree_path,
    get_worktree_status,
    remove_worktree,
    sync_worktree,
)

# Paths
CONFIG_PATH = Path.home() / ".config" / "prview" / "config.yaml"
DB_PATH = Path.home() / ".config" / "prview" / "prview.db"

# GitHub environment variables that affect gh CLI
GH_ENV_VARS = [
    ("GH_TOKEN", "Authentication token for github.com (takes precedence over stored credentials)"),
    ("GITHUB_TOKEN", "Alternative authentication token for github.com"),
    ("GH_ENTERPRISE_TOKEN", "Authentication token for GitHub Enterprise Server"),
    ("GITHUB_ENTERPRISE_TOKEN", "Alternative token for GitHub Enterprise Server"),
    ("GH_HOST", "Default GitHub hostname when not specified"),
    ("GH_REPO", "Default repository in [HOST/]OWNER/REPO format"),
]

# Ensure config directory exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Lock to prevent concurrent sync operations
_sync_lock = threading.Lock()


def get_db_connection(timeout: float = 30.0) -> sqlite3.Connection:
    """Get a database connection with proper settings for concurrency."""
    conn = sqlite3.connect(DB_PATH, timeout=timeout)
    # Enable WAL mode for better concurrent access
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")  # 30 second busy timeout
    return conn


def load_config() -> dict:
    """Load config from YAML file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def should_include_repo(repo: str, config: dict) -> bool:
    """Check if a repo should be included based on config."""
    org = repo.split("/")[0] if "/" in repo else ""

    # Check exclusions first
    if repo in config.get("exclude_repos", []):
        return False
    if org in config.get("exclude_orgs", []):
        return False

    # Check inclusions
    include_repos = config.get("include_repos", [])
    include_orgs = config.get("include_orgs", [])

    # If no includes specified, include everything (that's not excluded)
    if not include_repos and not include_orgs:
        return True

    # Check if repo matches includes
    if repo in include_repos:
        return True
    if org in include_orgs:
        return True

    return False


def init_db():
    """Initialize SQLite database."""
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prs (
            id INTEGER PRIMARY KEY,
            repo TEXT NOT NULL,
            number INTEGER NOT NULL,
            title TEXT NOT NULL,
            author TEXT NOT NULL,
            draft INTEGER DEFAULT 0,
            ci_status TEXT DEFAULT 'none',
            review_status TEXT DEFAULT 'pending',
            url TEXT NOT NULL,
            updated_at TEXT,
            additions INTEGER DEFAULT 0,
            deletions INTEGER DEFAULT 0,
            is_review_request INTEGER DEFAULT 0,
            last_synced TEXT,
            linked_stories TEXT DEFAULT '[]',
            UNIQUE(repo, number)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sync_status (
            id INTEGER PRIMARY KEY,
            last_sync TEXT,
            is_syncing INTEGER DEFAULT 0,
            error TEXT
        )
    """)
    # Migrate: add linked_stories column if missing
    try:
        conn.execute("SELECT linked_stories FROM prs LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE prs ADD COLUMN linked_stories TEXT DEFAULT '[]'")

    # --- Agent / worktree tables ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS repos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL UNIQUE,
            bare_path TEXT NOT NULL,
            clone_url TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_fetched TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS worktrees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo_full_name TEXT NOT NULL,
            pr_number INTEGER NOT NULL,
            worktree_path TEXT NOT NULL,
            branch_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_used TEXT,
            UNIQUE(repo_full_name, pr_number)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pr_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo_full_name TEXT NOT NULL,
            pr_number INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            model TEXT,
            created_at TEXT NOT NULL,
            last_used TEXT NOT NULL,
            UNIQUE(repo_full_name, pr_number)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pr_chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo_full_name TEXT NOT NULL,
            pr_number INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            model TEXT,
            session_id TEXT,
            cost_usd REAL,
            duration_ms INTEGER,
            input_tokens INTEGER,
            output_tokens INTEGER,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_chat_messages_pr
        ON pr_chat_messages(repo_full_name, pr_number, created_at)
    """)

    # Initialize sync status if not exists
    conn.execute(
        "INSERT OR IGNORE INTO sync_status (id, last_sync, is_syncing) VALUES (1, NULL, 0)"
    )
    conn.commit()
    conn.close()


def run_gh(args: list[str], timeout: int = 60) -> Optional[str]:
    """Run gh CLI command and return output."""
    try:
        result = subprocess.run(["gh"] + args, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return result.stdout
        return None
    except Exception:
        return None


def get_pr_details(repo: str, pr_number: int) -> dict:
    """Get detailed PR info."""
    output = run_gh(
        [
            "pr",
            "view",
            str(pr_number),
            "--repo",
            repo,
            "--json",
            "reviewDecision,additions,deletions,statusCheckRollup",
        ]
    )
    if not output:
        return {}
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {}


def get_ci_status(checks: list) -> str:
    """Determine CI status from checks."""
    if not checks:
        return "none"

    states = []
    for check in checks:
        conclusion = check.get("conclusion", "").upper()
        status = check.get("status", "").upper()

        # Real failures
        if conclusion in ("FAILURE", "ERROR", "TIMED_OUT"):
            states.append("FAILURE")
        # Success states (CANCELLED and SKIPPED are not failures)
        elif conclusion in ("SUCCESS", "SKIPPED", "CANCELLED", "NEUTRAL"):
            states.append("SUCCESS")
        # Still running
        elif status in ("IN_PROGRESS", "QUEUED", "PENDING", "WAITING"):
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


def sync_prs():
    """Sync PRs from GitHub to database."""
    # Prevent concurrent syncs
    if not _sync_lock.acquire(blocking=False):
        return  # Another sync is already in progress

    conn = None
    try:
        config = load_config()
        conn = get_db_connection()

        # Mark as syncing
        conn.execute("UPDATE sync_status SET is_syncing = 1, error = NULL WHERE id = 1")
        conn.commit()

        now = datetime.now().isoformat()

        # Clear old data (we'll re-add current ones)
        conn.execute("DELETE FROM prs")

        # Fetch my PRs
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

        if output:
            prs_data = json.loads(output)
            for pr_data in prs_data:
                repo = pr_data["repository"]["nameWithOwner"]

                # Filter by config
                if not should_include_repo(repo, config):
                    continue

                pr_number = pr_data["number"]

                # Get details
                details = get_pr_details(repo, pr_number)
                ci_status = get_ci_status(details.get("statusCheckRollup", []))
                review_status = get_review_status(details.get("reviewDecision", ""))

                # Fetch linked stories from task provider
                stories = fetch_linked_stories(repo, pr_number)
                stories_json = json.dumps(stories)

                conn.execute(
                    """
                    INSERT OR REPLACE INTO prs
                    (repo, number, title, author, draft, ci_status,
                     review_status, url, updated_at, additions,
                     deletions, is_review_request, last_synced,
                     linked_stories)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                    (
                        repo,
                        pr_number,
                        pr_data["title"],
                        pr_data["author"]["login"],
                        1 if pr_data["isDraft"] else 0,
                        ci_status,
                        review_status,
                        pr_data["url"],
                        pr_data["updatedAt"][:10],
                        details.get("additions", 0),
                        details.get("deletions", 0),
                        now,
                        stories_json,
                    ),
                )

        # Fetch review requests
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

        if output:
            prs_data = json.loads(output)
            for pr_data in prs_data:
                repo = pr_data["repository"]["nameWithOwner"]

                # Filter by config
                if not should_include_repo(repo, config):
                    continue

                pr_number = pr_data["number"]

                details = get_pr_details(repo, pr_number)
                ci_status = get_ci_status(details.get("statusCheckRollup", []))
                review_status = get_review_status(details.get("reviewDecision", ""))

                # Fetch linked stories from task provider
                stories = fetch_linked_stories(repo, pr_number)
                stories_json = json.dumps(stories)

                conn.execute(
                    """
                    INSERT OR REPLACE INTO prs
                    (repo, number, title, author, draft, ci_status,
                     review_status, url, updated_at, additions,
                     deletions, is_review_request, last_synced,
                     linked_stories)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                    (
                        repo,
                        pr_number,
                        pr_data["title"],
                        pr_data["author"]["login"],
                        1 if pr_data["isDraft"] else 0,
                        ci_status,
                        review_status,
                        pr_data["url"],
                        pr_data["updatedAt"][:10],
                        details.get("additions", 0),
                        details.get("deletions", 0),
                        now,
                        stories_json,
                    ),
                )

        conn.execute("UPDATE sync_status SET last_sync = ?, is_syncing = 0 WHERE id = 1", (now,))
        conn.commit()

    except Exception as e:
        if conn:
            conn.execute("UPDATE sync_status SET is_syncing = 0, error = ? WHERE id = 1", (str(e),))
            conn.commit()
    finally:
        if conn:
            conn.close()
        _sync_lock.release()


def get_prs_from_db():
    """Get all PRs from database."""
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row

    my_prs = conn.execute("""
        SELECT * FROM prs WHERE is_review_request = 0 ORDER BY repo, updated_at DESC
    """).fetchall()

    review_requests = conn.execute("""
        SELECT * FROM prs WHERE is_review_request = 1 ORDER BY updated_at DESC
    """).fetchall()

    sync_status = conn.execute("SELECT * FROM sync_status WHERE id = 1").fetchone()

    conn.close()

    def _parse_pr(pr):
        d = dict(pr)
        # Parse linked_stories JSON
        try:
            d["linked_stories"] = json.loads(d.get("linked_stories", "[]") or "[]")
        except (json.JSONDecodeError, TypeError):
            d["linked_stories"] = []
        return d

    # Group my PRs by repo
    my_prs_grouped = {}
    for pr in my_prs:
        repo = pr["repo"]
        if repo not in my_prs_grouped:
            my_prs_grouped[repo] = []
        my_prs_grouped[repo].append(_parse_pr(pr))

    return {
        "my_prs": my_prs_grouped,
        "review_requests": [_parse_pr(pr) for pr in review_requests],
        "sync_status": dict(sync_status) if sync_status else {},
    }


# Background sync task
sync_event = asyncio.Event()
clients: list[asyncio.Queue] = []


async def background_sync_loop():
    """Background task that syncs periodically."""
    # Wait a bit before first sync to let server start
    await asyncio.sleep(2)
    while True:
        try:
            # Run sync in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, sync_prs)
            sync_event.set()
        except Exception as e:
            print(f"Sync error: {e}")
        await asyncio.sleep(60)  # Sync every 60 seconds


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    init_db()
    # Start background sync (non-blocking)
    task = asyncio.create_task(background_sync_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)

# Templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main dashboard page."""
    data = get_prs_from_db()
    return templates.TemplateResponse("index.html", {"request": request, **data})


@app.get("/partials/prs", response_class=HTMLResponse)
async def partials_prs(request: Request):
    """HTMX partial for PR list."""
    data = get_prs_from_db()
    return templates.TemplateResponse("partials/prs.html", {"request": request, **data})


@app.get("/partials/status", response_class=HTMLResponse)
async def partials_status(request: Request):
    """HTMX partial for sync status."""
    data = get_prs_from_db()
    return templates.TemplateResponse(
        "partials/status.html", {"request": request, "sync_status": data["sync_status"]}
    )


@app.post("/sync")
async def trigger_sync(background_tasks: BackgroundTasks):
    """Trigger a manual sync."""
    background_tasks.add_task(sync_prs)
    return {"status": "syncing"}


@app.get("/events")
async def sse_events():
    """Server-Sent Events endpoint for real-time updates."""

    async def event_generator() -> AsyncGenerator[str, None]:
        last_data = None
        while True:
            data = get_prs_from_db()
            data_str = json.dumps(data, default=str)

            if data_str != last_data:
                yield f"event: update\ndata: {json.dumps({'updated': True})}\n\n"
                last_data = data_str

            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/prs")
async def api_prs():
    """JSON API for PR data."""
    return get_prs_from_db()


def get_gh_auth_status() -> dict:
    """Get GitHub CLI authentication status."""
    result = {
        "authenticated": False,
        "hostname": None,
        "username": None,
        "token_source": None,
        "scopes": [],
        "error": None,
    }

    try:
        # Check if gh is installed
        version_output = subprocess.run(
            ["gh", "--version"], capture_output=True, text=True, timeout=5
        )
        if version_output.returncode != 0:
            result["error"] = "gh CLI not installed"
            return result

        # Get auth status
        auth_output = subprocess.run(
            ["gh", "auth", "status"], capture_output=True, text=True, timeout=10
        )

        output = auth_output.stdout + auth_output.stderr

        if "Logged in to" in output:
            result["authenticated"] = True
            # Parse the output
            for line in output.split("\n"):
                line = line.strip()
                if line.startswith("✓ Logged in to"):
                    # Extract hostname and username
                    parts = line.replace("✓ Logged in to", "").strip().split(" account ")
                    if len(parts) >= 1:
                        result["hostname"] = parts[0].strip()
                    if len(parts) >= 2:
                        result["username"] = parts[1].split()[0].strip()
                elif "Token:" in line:
                    result["token_source"] = "configured"
                elif "Token scopes:" in line:
                    scopes_str = line.replace("Token scopes:", "").strip().strip("'")
                    result["scopes"] = [s.strip().strip("'") for s in scopes_str.split(",")]
        else:
            result["error"] = "Not authenticated. Run 'gh auth login' to authenticate."

    except subprocess.TimeoutExpired:
        result["error"] = "Timeout checking gh auth status"
    except FileNotFoundError:
        result["error"] = "gh CLI not found. Install from https://cli.github.com/"
    except Exception as e:
        result["error"] = str(e)

    return result


def get_env_vars_status() -> list[dict]:
    """Get status of GitHub-related environment variables."""
    env_status = []
    for var_name, description in GH_ENV_VARS:
        value = os.environ.get(var_name)
        env_status.append(
            {
                "name": var_name,
                "description": description,
                "is_set": value is not None,
                "value_preview": f"{value[:8]}..." if value and len(value) > 8 else value,
            }
        )
    return env_status


def save_config(config_data: dict):
    """Save config to YAML file."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)


def get_task_provider():
    """Get the configured task provider instance, or None."""
    config = load_config()
    provider_id = config.get("task_provider")
    if not provider_id:
        return None
    provider = get_provider(provider_id)
    if provider is None:
        return None
    provider.configure(config.get("task_provider_config", {}))
    return provider


def fetch_linked_stories(repo: str, pr_number: int) -> list[dict]:
    """Fetch linked stories for a PR from the configured task provider."""
    provider = get_task_provider()
    if provider is None:
        return []
    try:
        return provider.get_stories_for_pr(pr_number, repo)
    except Exception as e:
        print(f"Error fetching stories for {repo}#{pr_number}: {e}")
        return []


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings page with GitHub connection status and config."""
    auth_status = get_gh_auth_status()
    env_vars = get_env_vars_status()
    config = load_config()

    claude_status = check_claude_available()

    task_providers = list_providers()
    task_provider_id = config.get("task_provider", "")
    task_provider_config = config.get("task_provider_config", {})

    ci_config = {
        "jenkins_url": config.get("jenkins_url", "https://jenkins.devops.preset.zone"),
        "jenkins_user": config.get("jenkins_user", ""),
        "jenkins_token": config.get("jenkins_token", ""),
        "datadog_api_key": config.get("datadog_api_key", ""),
        "datadog_app_key": config.get("datadog_app_key", ""),
    }

    return templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "auth_status": auth_status,
            "env_vars": env_vars,
            "config": config,
            "config_path": str(CONFIG_PATH),
            "claude_status": claude_status,
            "claude_models": CLAUDE_MODELS,
            "task_providers": task_providers,
            "task_provider_id": task_provider_id,
            "task_provider_config": task_provider_config,
            "ci_config": ci_config,
        },
    )


@app.post("/settings/save")
async def save_settings(request: Request):
    """Save settings from form."""
    form_data = await request.form()

    # Parse form data
    include_orgs = [
        org.strip() for org in form_data.get("include_orgs", "").split("\n") if org.strip()
    ]
    exclude_orgs = [
        org.strip() for org in form_data.get("exclude_orgs", "").split("\n") if org.strip()
    ]
    include_repos = [
        repo.strip() for repo in form_data.get("include_repos", "").split("\n") if repo.strip()
    ]
    exclude_repos = [
        repo.strip() for repo in form_data.get("exclude_repos", "").split("\n") if repo.strip()
    ]

    claude_model = form_data.get("claude_model", "").strip()

    # Task provider config
    task_provider = form_data.get("task_provider", "").strip()
    task_api_token = form_data.get("task_api_token", "").strip()
    task_provider_config = {}
    if task_api_token:
        task_provider_config["api_token"] = task_api_token

    # CI config (Jenkins + Datadog)
    jenkins_url = form_data.get("jenkins_url", "").strip()
    jenkins_user = form_data.get("jenkins_user", "").strip()
    jenkins_token = form_data.get("jenkins_token", "").strip()
    datadog_api_key = form_data.get("datadog_api_key", "").strip()
    datadog_app_key = form_data.get("datadog_app_key", "").strip()

    config_data = {
        "include_orgs": include_orgs,
        "exclude_orgs": exclude_orgs,
        "include_repos": include_repos,
        "exclude_repos": exclude_repos,
        "refresh_interval": int(form_data.get("refresh_interval", 60)),
        "show_drafts": form_data.get("show_drafts") == "on",
        "max_prs_per_repo": int(form_data.get("max_prs_per_repo", 10)),
        "claude_model": claude_model if claude_model else DEFAULT_MODEL,
        "task_provider": task_provider if task_provider else None,
        "task_provider_config": task_provider_config if task_provider else {},
        "jenkins_url": jenkins_url or "https://jenkins.devops.preset.zone",
        "jenkins_user": jenkins_user,
        "jenkins_token": jenkins_token,
        "datadog_api_key": datadog_api_key,
        "datadog_app_key": datadog_app_key,
    }

    save_config(config_data)

    return RedirectResponse(url="/settings?saved=1", status_code=303)


@app.get("/api/auth-status")
async def api_auth_status():
    """JSON API for auth status."""
    return get_gh_auth_status()


# --- Task provider routes (Issue #6) ---


@app.get("/api/task-providers")
async def api_task_providers():
    """List available task management providers."""
    return {"providers": list_providers()}


@app.post("/api/task-provider/validate")
async def api_validate_task_provider(request: Request):
    """Test connection with a task provider."""
    body = await request.json()
    provider_id = body.get("provider_id", "")
    config = body.get("config", {})

    provider = get_provider(provider_id)
    if provider is None:
        return JSONResponse(
            {"valid": False, "error": f"Unknown provider: {provider_id}"}, status_code=400
        )

    provider.configure(config)
    result = provider.validate()
    return result


@app.get("/api/pr/{repo:path}/{number:int}/stories")
async def api_pr_stories(repo: str, number: int):
    """Get linked stories for a PR."""
    stories = fetch_linked_stories(repo, number)
    return {"stories": stories}


# --- CI analysis routes (Issue #6) ---

logger = logging.getLogger(__name__)


def get_ci_config() -> dict:
    """Get Jenkins and Datadog config from settings."""
    config = load_config()
    return {
        "jenkins_url": config.get("jenkins_url", "https://jenkins.devops.preset.zone"),
        "jenkins_user": config.get("jenkins_user", ""),
        "jenkins_token": config.get("jenkins_token", ""),
        "datadog_api_key": config.get("datadog_api_key", ""),
        "datadog_app_key": config.get("datadog_app_key", ""),
    }


def jenkins_request(path: str, ci_config: Optional[dict] = None) -> Optional[str]:
    """Make an authenticated request to Jenkins API."""
    if ci_config is None:
        ci_config = get_ci_config()
    user = ci_config.get("jenkins_user", "")
    token = ci_config.get("jenkins_token", "")
    base_url = ci_config.get("jenkins_url", "https://jenkins.devops.preset.zone")
    if not user or not token:
        return None
    url = f"{base_url}{path}"
    try:
        resp = httpx.get(url, auth=(user, token), timeout=30)
        if resp.status_code == 200:
            return resp.text
        return None
    except (httpx.TimeoutException, httpx.ConnectError):
        return None


def get_jenkins_job_path(repo: str) -> str:
    """Derive Jenkins job path from GitHub repo (e.g. preset-io/superset-shell)."""
    parts = repo.split("/")
    if len(parts) == 2:
        return f"/job/{parts[0]}/job/{parts[1]}"
    return f"/job/{repo}"


def get_jenkins_build_info(repo: str, pr_number: int, ci_config: Optional[dict] = None) -> dict:
    """Get latest build info for a PR from Jenkins."""
    job_path = get_jenkins_job_path(repo)
    # Get PR job info (latest build number)
    raw = jenkins_request(f"{job_path}/job/PR-{pr_number}/api/json", ci_config)
    if not raw:
        return {"error": "Could not reach Jenkins or PR job not found"}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON from Jenkins"}

    last_build = data.get("lastBuild", {}) or {}
    build_number = last_build.get("number")
    if not build_number:
        return {"error": "No builds found for this PR"}

    result = {
        "build_number": build_number,
        "color": data.get("color", ""),
        "url": last_build.get("url", ""),
    }

    # Get build details
    build_raw = jenkins_request(
        f"{job_path}/job/PR-{pr_number}/{build_number}/api/json",
        ci_config,
    )
    if build_raw:
        try:
            build_data = json.loads(build_raw)
            result["result"] = build_data.get("result", "UNKNOWN")
            result["building"] = build_data.get("building", False)
            result["duration_min"] = round(build_data.get("duration", 0) / 1000 / 60, 1)
        except json.JSONDecodeError:
            pass

    return result


def get_jenkins_stages(
    repo: str,
    pr_number: int,
    build_number: int,
    ci_config: Optional[dict] = None,
) -> list[dict]:
    """Get pipeline stage breakdown for a build."""
    job_path = get_jenkins_job_path(repo)
    raw = jenkins_request(
        f"{job_path}/job/PR-{pr_number}/{build_number}/wfapi/describe",
        ci_config,
    )
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    stages = []
    for stage in data.get("stages", []):
        stages.append(
            {
                "name": stage.get("name", "Unknown"),
                "status": stage.get("status", "UNKNOWN"),
                "duration_min": round(stage.get("durationMillis", 0) / 1000 / 60, 1),
            }
        )
    return stages


def get_jenkins_console_log(
    repo: str,
    pr_number: int,
    build_number: int,
    ci_config: Optional[dict] = None,
    tail_lines: int = 500,
) -> str:
    """Get console log for a build, optionally limited to last N lines."""
    job_path = get_jenkins_job_path(repo)
    raw = jenkins_request(
        f"{job_path}/job/PR-{pr_number}/{build_number}/consoleText",
        ci_config,
    )
    if not raw:
        return ""
    lines = raw.split("\n")
    if len(lines) > tail_lines:
        return "\n".join(lines[-tail_lines:])
    return raw


def extract_error_context(console_log: str) -> str:
    """Extract error-relevant sections from a Jenkins console log."""
    lines = console_log.split("\n")
    error_sections = []
    error_patterns = re.compile(
        r"ERROR|FAILED|FAILURE|Exception|exit code [1-9]|"
        r"AssertionError|ImportError|SyntaxError|"
        r"ruff.*check|pytest.*ERRORS",
        re.IGNORECASE,
    )

    for i, line in enumerate(lines):
        if error_patterns.search(line):
            start = max(0, i - 5)
            end = min(len(lines), i + 10)
            section = "\n".join(lines[start:end])
            error_sections.append(section)

    if not error_sections:
        # Return last 100 lines as fallback
        return "\n".join(lines[-100:])

    # Deduplicate overlapping sections, keep up to ~4000 chars
    combined = "\n---\n".join(error_sections)
    if len(combined) > 4000:
        combined = combined[:4000] + "\n... [truncated]"
    return combined


def search_datadog_logs(
    query: str,
    time_range: str = "now-2h",
    limit: int = 20,
    ci_config: Optional[dict] = None,
) -> list[dict]:
    """Search Datadog logs using the API v2."""
    if ci_config is None:
        ci_config = get_ci_config()
    api_key = ci_config.get("datadog_api_key", "")
    app_key = ci_config.get("datadog_app_key", "")
    if not api_key or not app_key:
        return []

    try:
        resp = httpx.post(
            "https://api.datadoghq.com/api/v2/logs/events/search",
            headers={
                "DD-API-KEY": api_key,
                "DD-APPLICATION-KEY": app_key,
                "Content-Type": "application/json",
            },
            json={
                "filter": {
                    "query": query,
                    "from": time_range,
                    "to": "now",
                },
                "sort": "desc",
                "page": {"limit": limit},
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        logs = []
        for entry in data.get("data", []):
            attrs = entry.get("attributes", {})
            logs.append(
                {
                    "message": attrs.get("message", ""),
                    "timestamp": attrs.get("timestamp", ""),
                    "status": attrs.get("status", ""),
                    "exc_info": attrs.get("exc_info", ""),
                    "error_kind": attrs.get("error", {}).get("kind", "")
                    if isinstance(attrs.get("error"), dict)
                    else "",
                }
            )
        return logs
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        logger.warning("Datadog search error: %s", e)
        return []


def find_deploy_slug_in_log(console_log: str) -> Optional[str]:
    """Try to find a deployment slug (ws--XXXX--main) in Jenkins log."""
    match = re.search(r"ws--([a-f0-9]+)--\w+", console_log)
    if match:
        return match.group(0)
    return None


def build_ci_analysis_context(repo: str, pr_number: int, mode: str = "explain") -> dict:
    """Gather all CI failure context for analysis.

    Returns a dict with build_info, stages, error_context,
    datadog_logs, and a formatted prompt.
    """
    ci_config = get_ci_config()
    result = {
        "build_info": {},
        "stages": [],
        "error_context": "",
        "datadog_logs": [],
        "prompt": "",
    }

    # 1. Get build info
    build_info = get_jenkins_build_info(repo, pr_number, ci_config)
    result["build_info"] = build_info
    if "error" in build_info:
        result["prompt"] = f"Could not retrieve CI build information: {build_info['error']}"
        return result

    build_number = build_info["build_number"]

    # 2. Get stages
    stages = get_jenkins_stages(repo, pr_number, build_number, ci_config)
    result["stages"] = stages
    failed_stages = [s for s in stages if s["status"] == "FAILED"]

    # 3. Get console log and extract errors
    console_log = get_jenkins_console_log(repo, pr_number, build_number, ci_config)
    error_context = extract_error_context(console_log)
    result["error_context"] = error_context

    # 4. Check for deploy slug and query Datadog if found
    deploy_slug = find_deploy_slug_in_log(console_log)
    if deploy_slug:
        dd_logs = search_datadog_logs(
            f"@release:{deploy_slug} status:error",
            time_range="now-2h",
            limit=10,
            ci_config=ci_config,
        )
        result["datadog_logs"] = dd_logs

    # 5. Build prompt
    parts = [
        f"## CI Failure Analysis for {repo} PR #{pr_number}",
        f"Build #{build_number} — Result: {build_info.get('result', 'UNKNOWN')}",
        "",
    ]

    if stages:
        parts.append("### Pipeline Stages")
        for s in stages:
            icon = "FAIL" if s["status"] == "FAILED" else "OK"
            parts.append(f"  [{icon}] {s['name']}: {s['status']} ({s['duration_min']} min)")
        parts.append("")

    if failed_stages:
        parts.append(f"### Failed stage(s): {', '.join(s['name'] for s in failed_stages)}")
        parts.append("")

    if error_context:
        parts.append("### Error details from console log")
        parts.append("```")
        parts.append(error_context)
        parts.append("```")
        parts.append("")

    if result["datadog_logs"]:
        parts.append(f"### Datadog error logs (deploy: {deploy_slug})")
        for log in result["datadog_logs"][:5]:
            parts.append(f"- [{log['timestamp']}] {log['message']}")
            if log.get("exc_info"):
                parts.append(f"  ```\n  {log['exc_info'][:500]}\n  ```")
        parts.append("")

    if mode == "explain":
        parts.append(
            "Analyze this CI failure. Explain what went wrong, "
            "which stage failed and why. Be specific about the root cause. "
            "Format your response in markdown."
        )
    else:
        parts.append(
            "Analyze this CI failure and suggest how to fix it. "
            "Explain what went wrong, then provide specific code changes "
            "or configuration fixes needed. "
            "Format your response in markdown with code blocks for changes."
        )

    result["prompt"] = "\n".join(parts)
    return result


@app.get(
    "/pr/{repo:path}/{number:int}/ci-analysis",
    response_class=HTMLResponse,
)
async def ci_analysis_page(request: Request, repo: str, number: int):
    """CI analysis page for a PR."""
    mode = request.query_params.get("mode", "explain")
    pr_info = get_pr_basic_info(repo, number)
    config = load_config()
    selected_model = config.get("claude_model", DEFAULT_MODEL)

    return templates.TemplateResponse(
        "ci_analysis.html",
        {
            "request": request,
            "repo": repo,
            "number": number,
            "mode": mode,
            "pr_title": pr_info.get("title", f"PR #{number}"),
            "pr_url": pr_info.get("url", f"https://github.com/{repo}/pull/{number}"),
            "claude_models": CLAUDE_MODELS,
            "selected_model": selected_model,
        },
    )


@app.post("/pr/{repo:path}/{number:int}/ci-analysis/stream")
async def stream_ci_analysis(request: Request, repo: str, number: int):
    """SSE endpoint to stream CI analysis."""
    body_data = await request.json()
    mode = body_data.get("mode", "explain")
    model = body_data.get("model", DEFAULT_MODEL)

    # Gather CI context
    context = build_ci_analysis_context(repo, number, mode)

    if not context["prompt"]:
        return JSONResponse(
            {"error": "Could not build CI analysis context"},
            status_code=500,
        )

    async def event_generator() -> AsyncGenerator[str, None]:
        # Send build info first
        info_data = json.dumps(
            {
                "type": "build_info",
                "build_info": context["build_info"],
                "stages": context["stages"],
            }
        )
        yield f"data: {info_data}\n\n"

        # Stream AI analysis
        async for chunk in stream_claude_response(context["prompt"], model=model):
            if chunk.get("type") in ("text", "text_start"):
                data = json.dumps(
                    {
                        "type": "text",
                        "content": chunk.get("content", ""),
                    }
                )
                yield f"data: {data}\n\n"
            elif chunk.get("type") == "error":
                data = json.dumps(
                    {
                        "type": "error",
                        "content": chunk.get("content", ""),
                    }
                )
                yield f"data: {data}\n\n"
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/ci-config/validate")
async def validate_ci_config(request: Request):
    """Validate Jenkins and/or Datadog configuration."""
    body = await request.json()
    service = body.get("service", "")
    config = body.get("config", {})
    result = {"valid": False, "error": None}

    if service == "jenkins":
        user = config.get("jenkins_user", "")
        token = config.get("jenkins_token", "")
        url = config.get("jenkins_url", "https://jenkins.devops.preset.zone")
        if not user or not token:
            result["error"] = "User and token are required"
            return result
        try:
            resp = httpx.get(
                f"{url}/api/json",
                auth=(user, token),
                timeout=10,
            )
            if resp.status_code == 200:
                result["valid"] = True
            elif resp.status_code == 401:
                result["error"] = "Invalid credentials"
            else:
                result["error"] = f"HTTP {resp.status_code}"
        except httpx.TimeoutException:
            result["error"] = "Connection timed out"
        except httpx.ConnectError:
            result["error"] = "Could not connect to Jenkins"

    elif service == "datadog":
        api_key = config.get("datadog_api_key", "")
        app_key = config.get("datadog_app_key", "")
        if not api_key or not app_key:
            result["error"] = "API key and app key are required"
            return result
        try:
            resp = httpx.get(
                "https://api.datadoghq.com/api/v1/validate",
                headers={
                    "DD-API-KEY": api_key,
                    "DD-APPLICATION-KEY": app_key,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                result["valid"] = True
            elif resp.status_code in (401, 403):
                result["error"] = "Invalid API keys"
            else:
                result["error"] = f"HTTP {resp.status_code}"
        except httpx.TimeoutException:
            result["error"] = "Connection timed out"
        except httpx.ConnectError:
            result["error"] = "Could not connect to Datadog"
    else:
        result["error"] = f"Unknown service: {service}"

    return result


# --- Chat routes ---


def build_pr_context_prompt(pr_data: dict) -> str:
    """Build a system prompt with PR context from the database."""
    lines = [
        "You are a helpful AI assistant integrated into prview, a GitHub PR dashboard.",
        "You have context about the user's current pull requests.",
        "",
    ]

    my_prs = pr_data.get("my_prs", {})
    review_requests = pr_data.get("review_requests", [])

    if my_prs:
        lines.append("## User's Open Pull Requests")
        count = 0
        for repo, prs in my_prs.items():
            for pr in prs:
                if count >= 50:
                    break
                draft_label = " [DRAFT]" if pr.get("draft") else ""
                lines.append(
                    f"- {repo}#{pr['number']}: {pr['title']}{draft_label} "
                    f"(CI: {pr.get('ci_status', 'unknown')}, "
                    f"Review: {pr.get('review_status', 'unknown')}, "
                    f"+{pr.get('additions', 0)}/-{pr.get('deletions', 0)})"
                )
                count += 1

    if review_requests:
        lines.append("")
        lines.append("## Review Requests (PRs requesting user's review)")
        for pr in review_requests[:50]:
            lines.append(
                f"- {pr['repo']}#{pr['number']}: {pr['title']} by {pr['author']} "
                f"(CI: {pr.get('ci_status', 'unknown')})"
            )

    return "\n".join(lines)


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Chat page with Claude AI assistant."""
    claude_status = check_claude_available()
    config = load_config()
    selected_model = config.get("claude_model", DEFAULT_MODEL)

    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "claude_status": claude_status,
            "claude_models": CLAUDE_MODELS,
            "selected_model": selected_model,
        },
    )


@app.post("/chat/send")
async def chat_send(request: Request):
    """SSE endpoint that streams Claude CLI responses."""
    form_data = await request.form()
    message = form_data.get("message", "").strip()
    model = form_data.get("model", DEFAULT_MODEL)

    if not message:
        return JSONResponse({"error": "Message is required"}, status_code=400)

    # Build PR context system prompt
    pr_data = get_prs_from_db()
    system_prompt = build_pr_context_prompt(pr_data)

    async def event_generator() -> AsyncGenerator[str, None]:
        async for chunk in stream_claude_response(
            message, model=model, system_prompt=system_prompt
        ):
            data = json.dumps(chunk)
            yield f"data: {data}\n\n"
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/claude-status")
async def api_claude_status():
    """JSON endpoint for Claude CLI availability."""
    return check_claude_available()


# --- Comments routes (Issue #3) ---


def _run_gh_graphql(args: list[str], timeout: int = 30) -> Optional[str]:
    """Run gh api graphql and return stdout even on non-zero exit.

    gh api graphql returns exit code 1 on GraphQL errors but may still
    include valid data in stdout. This helper captures stdout regardless.
    """
    try:
        result = subprocess.run(["gh"] + args, capture_output=True, text=True, timeout=timeout)
        # Return stdout if it looks like JSON, regardless of exit code
        if result.stdout and result.stdout.strip().startswith("{"):
            return result.stdout
        return None
    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        return None


def _graphql_review_threads(repo: str, number: int) -> Optional[list[dict]]:
    """Fetch review threads via GraphQL. Returns None on failure."""
    owner, name = repo.split("/", 1) if "/" in repo else (repo, repo)

    query = (
        "query($owner: String!, $name: String!, $number: Int!, $cursor: String) {"
        " repository(owner: $owner, name: $name) {"
        " pullRequest(number: $number) {"
        " reviewThreads(first: 100, after: $cursor) {"
        " pageInfo { hasNextPage endCursor }"
        " nodes {"
        " id isResolved isOutdated"
        " resolvedBy { login }"
        " path line diffSide"
        " comments(first: 50) {"
        " nodes { id databaseId body author { login } createdAt diffHunk }"
        " } } } } } }"
    )

    threads = []
    cursor = None

    for _ in range(10):  # max 10 pages
        variables: dict = {
            "owner": owner,
            "name": name,
            "number": number,
        }
        if cursor:
            variables["cursor"] = cursor

        args = [
            "api",
            "graphql",
            "-f",
            f"query={query}",
        ]
        for k, v in variables.items():
            if isinstance(v, int):
                args.extend(["-F", f"{k}={v}"])
            else:
                args.extend(["-f", f"{k}={v}"])

        output = _run_gh_graphql(args, timeout=30)
        if not output:
            return None

        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return None

        pr_data = data.get("data", {}).get("repository", {}).get("pullRequest")
        if not pr_data:
            return None

        thread_data = pr_data.get("reviewThreads", {})
        for node in thread_data.get("nodes", []):
            comment_nodes = node.get("comments", {}).get("nodes", [])
            if not comment_nodes:
                continue

            first_comment = comment_nodes[0]
            thread = {
                "thread_id": node.get("id", ""),
                "isResolved": node.get("isResolved", False),
                "isOutdated": node.get("isOutdated", False),
                "resolvedBy": ((node.get("resolvedBy") or {}).get("login")),
                "path": node.get("path", ""),
                "line": node.get("line"),
                "diffSide": node.get("diffSide", ""),
                "diff_hunk": first_comment.get("diffHunk", ""),
                "comments": [],
            }

            for c in comment_nodes:
                thread["comments"].append(
                    {
                        "id": c.get("databaseId", c.get("id")),
                        "body": c.get("body", ""),
                        "user": (c.get("author") or {}).get("login", "unknown"),
                        "created_at": c.get("createdAt", ""),
                    }
                )

            threads.append(thread)

        page_info = thread_data.get("pageInfo", {})
        if page_info.get("hasNextPage"):
            cursor = page_info.get("endCursor")
        else:
            break

    return threads


def _rest_comments_as_threads(repo: str, number: int) -> list[dict]:
    """Fallback: fetch review comments via REST and group into threads."""
    output = run_gh(
        [
            "api",
            f"/repos/{repo}/pulls/{number}/comments",
            "--paginate",
            "--jq",
            ".[] | {id, path, line: (.line // .original_line),"
            " diff_hunk, body, user: .user.login,"
            " created_at: .created_at, in_reply_to_id}",
        ],
        timeout=30,
    )
    if not output:
        return []

    comments = []
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            comments.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # Group by top-level comment (thread root)
    top_level = {}
    replies_by_parent: dict[int, list] = {}
    for c in comments:
        parent_id = c.get("in_reply_to_id")
        if parent_id:
            replies_by_parent.setdefault(parent_id, []).append(c)
        else:
            top_level[c["id"]] = c

    threads = []
    for cid, c in top_level.items():
        thread = {
            "thread_id": str(cid),
            "isResolved": False,
            "isOutdated": False,
            "resolvedBy": None,
            "path": c.get("path", ""),
            "line": c.get("line"),
            "diffSide": "",
            "diff_hunk": c.get("diff_hunk", ""),
            "comments": [
                {
                    "id": c["id"],
                    "body": c.get("body", ""),
                    "user": c.get("user", "unknown"),
                    "created_at": c.get("created_at", ""),
                }
            ],
        }
        for reply in replies_by_parent.get(cid, []):
            thread["comments"].append(
                {
                    "id": reply["id"],
                    "body": reply.get("body", ""),
                    "user": reply.get("user", "unknown"),
                    "created_at": reply.get("created_at", ""),
                }
            )
        threads.append(thread)

    return threads


def get_pr_review_threads(repo: str, number: int) -> list[dict]:
    """Fetch review comment threads for a PR.

    Tries GraphQL first (includes resolved/outdated status).
    Falls back to REST API if GraphQL fails.
    """
    # Try GraphQL first
    threads = _graphql_review_threads(repo, number)
    if threads is not None:
        return threads

    # Fallback to REST (no resolved status, but at least shows comments)
    return _rest_comments_as_threads(repo, number)


def get_pr_review_comments(repo: str, number: int) -> list[dict]:
    """Fetch review comments (flat, top-level only) via gh REST API.

    Kept for backward compatibility. Prefer get_pr_review_threads().
    """
    output = run_gh(
        [
            "api",
            f"/repos/{repo}/pulls/{number}/comments",
            "--paginate",
            "--jq",
            ".[] | {id, path, line: (.line // .original_line), diff_hunk,"
            " body, user: .user.login, created_at: .created_at,"
            " in_reply_to_id}",
        ],
        timeout=30,
    )
    if not output:
        return []

    comments = []
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            comments.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    top_level = [c for c in comments if not c.get("in_reply_to_id")]
    return top_level


def post_comment_reply(repo: str, number: int, comment_id: int, body: str) -> bool:
    """Post a reply to a review comment via gh api."""
    try:
        result = subprocess.run(
            [
                "gh",
                "api",
                "-X",
                "POST",
                f"/repos/{repo}/pulls/{number}/comments",
                "-f",
                f"body={body}",
                "-F",
                f"in_reply_to={comment_id}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except FileNotFoundError:
        return False


def get_pr_basic_info(repo: str, number: int) -> dict:
    """Get basic PR info (title, url)."""
    output = run_gh(
        [
            "pr",
            "view",
            str(number),
            "--repo",
            repo,
            "--json",
            "title,url",
        ]
    )
    if not output:
        return {"title": f"PR #{number}", "url": f"https://github.com/{repo}/pull/{number}"}
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {"title": f"PR #{number}", "url": f"https://github.com/{repo}/pull/{number}"}


@app.get("/pr/{repo:path}/{number:int}/comments", response_class=HTMLResponse)
async def comments_page(request: Request, repo: str, number: int):
    """Address reviewer comments page."""
    pr_info = get_pr_basic_info(repo, number)
    return templates.TemplateResponse(
        "comments.html",
        {
            "request": request,
            "repo": repo,
            "number": number,
            "pr_title": pr_info.get("title", f"PR #{number}"),
            "pr_url": pr_info.get("url", f"https://github.com/{repo}/pull/{number}"),
            "claude_models": CLAUDE_MODELS,
            "selected_model": DEFAULT_MODEL,
        },
    )


@app.get("/api/pr/{repo:path}/{number:int}/comments")
async def api_pr_comments(repo: str, number: int):
    """JSON API to fetch review comment threads."""
    threads = get_pr_review_threads(repo, number)
    return {"threads": threads}


@app.post("/pr/{repo:path}/{number:int}/comments/{comment_id:int}/draft")
async def draft_comment_reply(request: Request, repo: str, number: int, comment_id: int):
    """SSE endpoint to stream an AI-drafted reply to a review comment."""
    body_data = await request.json()

    comment_context = (
        f"You are helping a developer reply to a code review comment on GitHub.\n\n"
        f"File: {body_data.get('path', 'unknown')}\n\n"
        f"Diff context:\n```\n{body_data.get('diff_hunk', '')}\n```\n\n"
        f"Reviewer ({body_data.get('user', 'reviewer')}) wrote:\n{body_data.get('body', '')}\n\n"
        f"Draft a concise, professional reply addressing the reviewer's comment. "
        f"If the reviewer is requesting a change, acknowledge it and describe what you'll do. "
        f"If it's a question, answer it directly. Keep it brief (1-3 sentences)."
    )

    config = load_config()
    model = config.get("claude_model", DEFAULT_MODEL)

    async def event_generator() -> AsyncGenerator[str, None]:
        async for chunk in stream_claude_response(comment_context, model=model):
            if chunk.get("type") in ("text", "text_start"):
                data = json.dumps({"type": "text", "content": chunk.get("content", "")})
                yield f"data: {data}\n\n"
            elif chunk.get("type") == "error":
                data = json.dumps({"type": "error", "content": chunk.get("content", "")})
                yield f"data: {data}\n\n"
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/pr/{repo:path}/{number:int}/comments/{comment_id:int}/reply")
async def reply_to_comment(request: Request, repo: str, number: int, comment_id: int):
    """Submit a reply to a review comment on GitHub."""
    body_data = await request.json()
    body = body_data.get("body", "").strip()

    if not body:
        return JSONResponse({"error": "Reply body is required"}, status_code=400)

    success = post_comment_reply(repo, number, comment_id, body)
    if success:
        return {"status": "ok"}
    return JSONResponse({"error": "Failed to post reply to GitHub"}, status_code=500)


# --- Review routes (Issue #4) ---


def get_pr_full_data(repo: str, number: int) -> dict:
    """Get full PR data for review: title, body, author, files, diff."""
    output = run_gh(
        [
            "pr",
            "view",
            str(number),
            "--repo",
            repo,
            "--json",
            "title,body,author,files,additions,deletions,url,reviews",
        ]
    )
    pr_data = {
        "title": f"PR #{number}",
        "body": "",
        "author": "",
        "files": [],
        "additions": 0,
        "deletions": 0,
        "url": f"https://github.com/{repo}/pull/{number}",
        "diff": "",
    }

    if output:
        try:
            parsed = json.loads(output)
            pr_data["title"] = parsed.get("title", pr_data["title"])
            pr_data["body"] = parsed.get("body", "")
            pr_data["author"] = parsed.get("author", {}).get("login", "unknown")
            pr_data["additions"] = parsed.get("additions", 0)
            pr_data["deletions"] = parsed.get("deletions", 0)
            pr_data["url"] = parsed.get("url", pr_data["url"])

            files = parsed.get("files", [])
            pr_data["files"] = [
                {
                    "path": f.get("path", ""),
                    "additions": f.get("additions", 0),
                    "deletions": f.get("deletions", 0),
                    "status": f.get("status", "modified"),
                }
                for f in files[:50]
            ]
        except json.JSONDecodeError:
            pass

    # Get diff (sampled)
    diff_output = run_gh(
        ["pr", "diff", str(number), "--repo", repo],
        timeout=30,
    )
    if diff_output:
        # Sample diff: limit to ~8000 chars to stay within token limits
        if len(diff_output) > 8000:
            pr_data["diff"] = diff_output[:8000] + "\n... [diff truncated]"
        else:
            pr_data["diff"] = diff_output

    return pr_data


def build_review_prompt(pr_data: dict, conversation: list[dict]) -> str:
    """Build a prompt for AI-assisted PR review."""
    parts = [
        "You are an expert code reviewer. Review the following pull request thoroughly.",
        "Focus on: bugs, security issues, performance, code quality, and design.",
        "Be constructive and specific. Reference file paths and line numbers when possible.",
        "Format your review in markdown with sections.",
        "",
        f"## PR: {pr_data.get('title', 'Unknown')}",
        f"Author: {pr_data.get('author', 'unknown')}",
        f"Files: {len(pr_data.get('files', []))}"
        f" | +{pr_data.get('additions', 0)}/-{pr_data.get('deletions', 0)}",
        "",
    ]

    if pr_data.get("body"):
        parts.append("### PR Description")
        parts.append(pr_data["body"][:2000])
        parts.append("")

    if pr_data.get("diff"):
        parts.append("### Diff")
        parts.append("```diff")
        parts.append(pr_data["diff"])
        parts.append("```")
        parts.append("")

    # Add conversation context
    if conversation:
        parts.append("### Previous conversation")
        for msg in conversation:
            role = "User" if msg["role"] == "user" else "AI"
            parts.append(f"**{role}**: {msg['content'][:1000]}")
        parts.append("")
        parts.append("Generate an updated review incorporating the user's feedback.")

    return "\n".join(parts)


def submit_pr_review(repo: str, number: int, body: str, event: str) -> bool:
    """Submit a review to GitHub via gh pr review."""
    flag_map = {
        "COMMENT": "--comment",
        "APPROVE": "--approve",
        "REQUEST_CHANGES": "--request-changes",
    }
    flag = flag_map.get(event, "--comment")

    try:
        result = subprocess.run(
            [
                "gh",
                "pr",
                "review",
                str(number),
                "--repo",
                repo,
                flag,
                "--body",
                body,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except FileNotFoundError:
        return False


@app.get("/pr/{repo:path}/{number:int}/review", response_class=HTMLResponse)
async def review_page(request: Request, repo: str, number: int):
    """AI-assisted PR review page."""
    pr_data = get_pr_full_data(repo, number)
    config = load_config()
    selected_model = config.get("claude_model", DEFAULT_MODEL)

    return templates.TemplateResponse(
        "review.html",
        {
            "request": request,
            "repo": repo,
            "number": number,
            "pr_data": pr_data,
            "claude_models": CLAUDE_MODELS,
            "selected_model": selected_model,
        },
    )


@app.post("/pr/{repo:path}/{number:int}/review/generate")
async def generate_review(request: Request, repo: str, number: int):
    """SSE endpoint to stream AI-generated review."""
    body_data = await request.json()
    conversation = body_data.get("conversation", [])
    model = body_data.get("model", DEFAULT_MODEL)

    pr_data = get_pr_full_data(repo, number)
    prompt = build_review_prompt(pr_data, conversation)

    async def event_generator() -> AsyncGenerator[str, None]:
        async for chunk in stream_claude_response(prompt, model=model):
            data = json.dumps(chunk)
            yield f"data: {data}\n\n"
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/pr/{repo:path}/{number:int}/review/submit")
async def submit_review(request: Request, repo: str, number: int):
    """Submit a review to GitHub."""
    body_data = await request.json()
    body = body_data.get("body", "").strip()
    event = body_data.get("event", "COMMENT")

    if not body:
        return JSONResponse({"error": "Review body is required"}, status_code=400)

    if event not in ("COMMENT", "APPROVE", "REQUEST_CHANGES"):
        return JSONResponse({"error": "Invalid review event type"}, status_code=400)

    success = submit_pr_review(repo, number, body, event)
    if success:
        return {"status": "ok"}
    return JSONResponse({"error": "Failed to submit review to GitHub"}, status_code=500)


# --- PR Agent Chat routes ---


def get_pr_chat_messages(repo: str, pr_number: int) -> list[dict]:
    """Fetch all chat messages for a PR from the database."""
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, role, content, model, session_id, cost_usd,
               duration_ms, input_tokens, output_tokens, created_at
        FROM pr_chat_messages
        WHERE repo_full_name = ? AND pr_number = ?
        ORDER BY created_at ASC
        """,
        (repo, pr_number),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_chat_message(
    repo: str,
    pr_number: int,
    role: str,
    content: str,
    model: Optional[str] = None,
    session_id: Optional[str] = None,
    cost_usd: Optional[float] = None,
    duration_ms: Optional[int] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
) -> int:
    """Save a chat message to the database. Returns the message ID."""
    conn = get_db_connection()
    now = datetime.now().isoformat()
    cursor = conn.execute(
        """
        INSERT INTO pr_chat_messages
        (repo_full_name, pr_number, role, content, model, session_id,
         cost_usd, duration_ms, input_tokens, output_tokens, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            repo,
            pr_number,
            role,
            content,
            model,
            session_id,
            cost_usd,
            duration_ms,
            input_tokens,
            output_tokens,
            now,
        ),
    )
    msg_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return msg_id


def get_pr_session(repo: str, pr_number: int) -> Optional[dict]:
    """Get existing session for a PR, or None."""
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT session_id, model, created_at, last_used
        FROM pr_sessions
        WHERE repo_full_name = ? AND pr_number = ?
        """,
        (repo, pr_number),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def save_pr_session(repo: str, pr_number: int, session_id: str, model: Optional[str] = None):
    """Upsert a session for a PR."""
    conn = get_db_connection()
    now = datetime.now().isoformat()
    conn.execute(
        """
        INSERT INTO pr_sessions
        (repo_full_name, pr_number, session_id, model, created_at, last_used)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(repo_full_name, pr_number) DO UPDATE SET
            session_id = excluded.session_id,
            model = excluded.model,
            last_used = excluded.last_used
        """,
        (repo, pr_number, session_id, model, now, now),
    )
    conn.commit()
    conn.close()


def get_pr_branch_name(repo: str, pr_number: int) -> Optional[str]:
    """Get the head branch name for a PR via gh CLI."""
    output = run_gh(
        ["pr", "view", str(pr_number), "--repo", repo, "--json", "headRefName"],
    )
    if not output:
        return None
    try:
        data = json.loads(output)
        return data.get("headRefName")
    except json.JSONDecodeError:
        return None


@app.get("/pr/{repo:path}/{number:int}/chat", response_class=HTMLResponse)
async def pr_chat_page(request: Request, repo: str, number: int):
    """Per-PR persistent chat page."""
    pr_info = get_pr_basic_info(repo, number)
    messages = get_pr_chat_messages(repo, number)
    config = load_config()
    selected_model = config.get("claude_model", DEFAULT_MODEL)
    prompt_prefill = request.query_params.get("prompt", "")

    return templates.TemplateResponse(
        "pr_chat.html",
        {
            "request": request,
            "repo": repo,
            "number": number,
            "pr_title": pr_info.get("title", f"PR #{number}"),
            "pr_url": pr_info.get("url", f"https://github.com/{repo}/pull/{number}"),
            "messages": messages,
            "claude_models": CLAUDE_MODELS,
            "selected_model": selected_model,
            "prompt_prefill": prompt_prefill,
        },
    )


@app.post("/pr/{repo:path}/{number:int}/chat/send")
async def pr_chat_send(request: Request, repo: str, number: int):
    """SSE streaming chat with session persistence for a PR."""
    body_data = await request.json()
    message = body_data.get("message", "").strip()
    model = body_data.get("model", DEFAULT_MODEL)

    if not message:
        return JSONResponse({"error": "Message is required"}, status_code=400)

    # Save user message
    save_chat_message(repo, number, "user", message, model=model)

    # Look up existing session
    existing_session = get_pr_session(repo, number)
    resume = False
    if existing_session and existing_session.get("session_id"):
        session_id = existing_session["session_id"]
        resume = True
    else:
        session_id = str(uuid.uuid4())

    # Get worktree path as cwd (if it exists)
    wt_path = get_worktree_path(repo, number)
    cwd = str(wt_path) if wt_path.exists() else None

    # Build system prompt for first message only
    system_prompt = None
    if not resume:
        pr_info = get_pr_basic_info(repo, number)
        system_prompt = (
            "You are an AI coding assistant integrated into prview, a GitHub PR dashboard. "
            f"You are working on PR #{number} in {repo}. "
            f"PR title: {pr_info.get('title', 'Unknown')}. "
            "You have full access to the repository files in the current working directory. "
            "Make code changes directly when asked. Be concise and focus on the code."
        )

    async def event_generator() -> AsyncGenerator[str, None]:
        accumulated_text = ""
        result_session_id = session_id
        usage_data = {}

        async for chunk in stream_claude_response(
            message,
            model=model,
            system_prompt=system_prompt,
            session_id=session_id,
            resume_session=resume,
            cwd=cwd,
            timeout_seconds=300,
        ):
            data = json.dumps(chunk)
            yield f"data: {data}\n\n"

            if chunk.get("type") in ("text", "text_start"):
                accumulated_text += chunk.get("content", "")
            elif chunk.get("type") == "usage":
                usage_content = chunk.get("content", {})
                if isinstance(usage_content, dict):
                    usage_data = usage_content
                    if usage_content.get("session_id"):
                        result_session_id = usage_content["session_id"]

        # Save assistant message
        if accumulated_text:
            save_chat_message(
                repo,
                number,
                "assistant",
                accumulated_text,
                model=model,
                session_id=result_session_id,
                cost_usd=usage_data.get("cost_usd"),
                duration_ms=usage_data.get("duration_ms"),
                input_tokens=usage_data.get("input_tokens"),
                output_tokens=usage_data.get("output_tokens"),
            )

        # Save session
        save_pr_session(repo, number, result_session_id, model)

        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/pr/{repo:path}/{number:int}/chat/history")
async def api_pr_chat_history(repo: str, number: int):
    """JSON chat history for a PR."""
    messages = get_pr_chat_messages(repo, number)
    return {"messages": messages}


@app.post("/api/pr/{repo:path}/{number:int}/chat/clear")
async def api_pr_chat_clear(repo: str, number: int):
    """Clear chat history and session for a PR."""
    conn = get_db_connection()
    conn.execute(
        "DELETE FROM pr_chat_messages WHERE repo_full_name = ? AND pr_number = ?",
        (repo, number),
    )
    conn.execute(
        "DELETE FROM pr_sessions WHERE repo_full_name = ? AND pr_number = ?",
        (repo, number),
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


# --- Worktree API routes ---


@app.post("/api/pr/{repo:path}/{number:int}/worktree/setup")
async def api_worktree_setup(request: Request, repo: str, number: int):
    """Clone repo and create worktree for a PR."""
    pr_branch = get_pr_branch_name(repo, number)
    if not pr_branch:
        return JSONResponse(
            {"error": f"Could not determine branch for PR #{number}"},
            status_code=400,
        )

    try:
        wt_path = await ensure_worktree(repo, number, pr_branch)

        # Save worktree info to DB
        conn = get_db_connection()
        now = datetime.now().isoformat()
        conn.execute(
            """
            INSERT INTO worktrees
            (repo_full_name, pr_number, worktree_path, branch_name,
             created_at, last_used)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(repo_full_name, pr_number) DO UPDATE SET
                worktree_path = excluded.worktree_path,
                branch_name = excluded.branch_name,
                last_used = excluded.last_used
            """,
            (repo, number, str(wt_path), pr_branch, now, now),
        )
        conn.commit()
        conn.close()

        return {"status": "ok", "path": str(wt_path), "branch": pr_branch}
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/pr/{repo:path}/{number:int}/worktree/status")
async def api_worktree_status(repo: str, number: int):
    """Get git status of the PR worktree."""
    status = await get_worktree_status(repo, number)
    return status


@app.get("/api/pr/{repo:path}/{number:int}/worktree/diff")
async def api_worktree_diff(repo: str, number: int):
    """Get full git diff for the PR worktree."""
    diff = await get_worktree_diff(repo, number)
    return {"diff": diff}


@app.post("/api/pr/{repo:path}/{number:int}/worktree/push")
async def api_worktree_push(request: Request, repo: str, number: int):
    """Commit and push changes from the PR worktree."""
    body = await request.json()
    commit_message = body.get("message", "").strip()
    if not commit_message:
        return JSONResponse({"error": "Commit message is required"}, status_code=400)

    result = await commit_and_push(repo, number, commit_message)
    if not result.get("ok"):
        return JSONResponse({"error": result.get("error", "Unknown error")}, status_code=500)
    return result


@app.post("/api/pr/{repo:path}/{number:int}/worktree/sync")
async def api_worktree_sync(repo: str, number: int):
    """Pull latest remote changes into the worktree."""
    msg = await sync_worktree(repo, number)
    return {"status": "ok", "message": msg}


@app.post("/api/pr/{repo:path}/{number:int}/worktree/remove")
async def api_worktree_remove(repo: str, number: int):
    """Remove a PR worktree."""
    msg = await remove_worktree(repo, number)
    return {"status": "ok", "message": msg}


def run_server(host: str = "127.0.0.1", port: int = 8420):
    """Run the web server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
