"""
prview web server with real-time updates via SSE.
Beautiful dark theme UI inspired by VS Code/Atom.
"""

import asyncio
import json
import os
import subprocess
import sqlite3
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, AsyncGenerator

import yaml
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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
    conn = sqlite3.connect(DB_PATH)
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
    config = load_config()
    conn = sqlite3.connect(DB_PATH)

    # Mark as syncing
    conn.execute("UPDATE sync_status SET is_syncing = 1, error = NULL WHERE id = 1")
    conn.commit()

    try:
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

                conn.execute(
                    """
                    INSERT OR REPLACE INTO prs
                    (repo, number, title, author, draft, ci_status, review_status, url, updated_at, additions, deletions, is_review_request, last_synced)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
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

                conn.execute(
                    """
                    INSERT OR REPLACE INTO prs
                    (repo, number, title, author, draft, ci_status, review_status, url, updated_at, additions, deletions, is_review_request, last_synced)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
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
                    ),
                )

        conn.execute("UPDATE sync_status SET last_sync = ?, is_syncing = 0 WHERE id = 1", (now,))
        conn.commit()

    except Exception as e:
        conn.execute("UPDATE sync_status SET is_syncing = 0, error = ? WHERE id = 1", (str(e),))
        conn.commit()
    finally:
        conn.close()


def get_prs_from_db():
    """Get all PRs from database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    my_prs = conn.execute("""
        SELECT * FROM prs WHERE is_review_request = 0 ORDER BY repo, updated_at DESC
    """).fetchall()

    review_requests = conn.execute("""
        SELECT * FROM prs WHERE is_review_request = 1 ORDER BY updated_at DESC
    """).fetchall()

    sync_status = conn.execute("SELECT * FROM sync_status WHERE id = 1").fetchone()

    conn.close()

    # Group my PRs by repo
    my_prs_grouped = {}
    for pr in my_prs:
        repo = pr["repo"]
        if repo not in my_prs_grouped:
            my_prs_grouped[repo] = []
        my_prs_grouped[repo].append(dict(pr))

    return {
        "my_prs": my_prs_grouped,
        "review_requests": [dict(pr) for pr in review_requests],
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


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings page with GitHub connection status and config."""
    auth_status = get_gh_auth_status()
    env_vars = get_env_vars_status()
    config = load_config()

    return templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "auth_status": auth_status,
            "env_vars": env_vars,
            "config": config,
            "config_path": str(CONFIG_PATH),
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

    config_data = {
        "include_orgs": include_orgs,
        "exclude_orgs": exclude_orgs,
        "include_repos": include_repos,
        "exclude_repos": exclude_repos,
        "refresh_interval": int(form_data.get("refresh_interval", 60)),
        "show_drafts": form_data.get("show_drafts") == "on",
        "max_prs_per_repo": int(form_data.get("max_prs_per_repo", 10)),
    }

    save_config(config_data)

    return RedirectResponse(url="/settings?saved=1", status_code=303)


@app.get("/api/auth-status")
async def api_auth_status():
    """JSON API for auth status."""
    return get_gh_auth_status()


def run_server(host: str = "127.0.0.1", port: int = 8420):
    """Run the web server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
