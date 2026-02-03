"""
prview web server with real-time updates via SSE.
Beautiful dark theme UI inspired by VS Code/Atom.
"""

import asyncio
import json
import os
import subprocess
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, AsyncGenerator

import yaml
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from prview.web.claude_cli import (
    CLAUDE_MODELS,
    DEFAULT_MODEL,
    check_claude_available,
    stream_claude_response,
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

    claude_status = check_claude_available()

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

    config_data = {
        "include_orgs": include_orgs,
        "exclude_orgs": exclude_orgs,
        "include_repos": include_repos,
        "exclude_repos": exclude_repos,
        "refresh_interval": int(form_data.get("refresh_interval", 60)),
        "show_drafts": form_data.get("show_drafts") == "on",
        "max_prs_per_repo": int(form_data.get("max_prs_per_repo", 10)),
        "claude_model": claude_model if claude_model else DEFAULT_MODEL,
    }

    save_config(config_data)

    return RedirectResponse(url="/settings?saved=1", status_code=303)


@app.get("/api/auth-status")
async def api_auth_status():
    """JSON API for auth status."""
    return get_gh_auth_status()


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


def get_pr_review_comments(repo: str, number: int) -> list[dict]:
    """Fetch review comments for a PR via gh api."""
    output = run_gh(
        [
            "api",
            f"/repos/{repo}/pulls/{number}/comments",
            "--paginate",
            "--jq",
            ".[] | {id, path, line: (.line // .original_line), diff_hunk, body, user: .user.login, created_at: .created_at, in_reply_to_id}",
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

    # Filter to only top-level comments (not replies)
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
        },
    )


@app.get("/api/pr/{repo:path}/{number:int}/comments")
async def api_pr_comments(repo: str, number: int):
    """JSON API to fetch review comments."""
    comments = get_pr_review_comments(repo, number)
    return {"comments": comments}


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
        f"Files: {len(pr_data.get('files', []))} | +{pr_data.get('additions', 0)}/-{pr_data.get('deletions', 0)}",
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


def run_server(host: str = "127.0.0.1", port: int = 8420):
    """Run the web server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
