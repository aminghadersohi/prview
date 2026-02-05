"""
Git worktree lifecycle manager for prview PR agent.

Manages bare clones and per-PR worktrees so each PR gets an isolated checkout.
Uses `gh repo clone` for auth consistency with the rest of prview.
"""

import asyncio
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Base directories
_BASE_DIR = Path.home() / ".config" / "prview"
_REPOS_DIR = _BASE_DIR / "repos"
_WORKTREES_DIR = _BASE_DIR / "worktrees"


async def _run_cmd(
    args: list[str],
    cwd: Optional[str] = None,
    timeout: int = 120,
) -> tuple[int, str, str]:
    """Run a command asynchronously. Returns (returncode, stdout, stderr)."""
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    try:
        async with asyncio.timeout(timeout):
            stdout, stderr = await process.communicate()
    except TimeoutError:
        process.kill()
        await process.wait()
        return -1, "", f"Command timed out after {timeout}s"

    return (
        process.returncode or 0,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


async def _run_git(
    args: list[str],
    cwd: Optional[str] = None,
    timeout: int = 120,
) -> tuple[int, str, str]:
    """Run a git command asynchronously."""
    return await _run_cmd(["git"] + args, cwd=cwd, timeout=timeout)


def get_bare_repo_path(repo_full_name: str) -> Path:
    """Get the path for a bare clone: ~/.config/prview/repos/{org}/{repo}.git"""
    parts = repo_full_name.split("/")
    if len(parts) == 2:
        return _REPOS_DIR / parts[0] / f"{parts[1]}.git"
    return _REPOS_DIR / f"{repo_full_name}.git"


def get_worktree_path(repo_full_name: str, pr_number: int) -> Path:
    """Get the worktree path: ~/.config/prview/worktrees/{org}/{repo}/{number}"""
    parts = repo_full_name.split("/")
    if len(parts) == 2:
        return _WORKTREES_DIR / parts[0] / parts[1] / str(pr_number)
    return _WORKTREES_DIR / repo_full_name / str(pr_number)


async def ensure_bare_clone(repo_full_name: str) -> Path:
    """Ensure a bare clone exists for the repo. Returns the bare repo path.

    Clones with `gh repo clone` if missing, fetches if it already exists.
    """
    bare_path = get_bare_repo_path(repo_full_name)

    if bare_path.exists():
        # Fetch latest
        logger.info("Fetching updates for bare repo %s", repo_full_name)
        rc, _out, err = await _run_git(
            ["fetch", "--all", "--prune"],
            cwd=str(bare_path),
            timeout=120,
        )
        if rc != 0:
            logger.warning("git fetch failed for %s: %s", repo_full_name, err)
        return bare_path

    # Clone as bare
    bare_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Bare-cloning %s to %s", repo_full_name, bare_path)

    # Find gh CLI
    gh_path = shutil.which("gh")
    if not gh_path:
        raise FileNotFoundError("gh CLI not found")

    rc, out, err = await _run_cmd(
        [gh_path, "repo", "clone", repo_full_name, str(bare_path), "--", "--bare"],
        timeout=300,
    )
    if rc != 0:
        raise RuntimeError(f"Failed to clone {repo_full_name}: {err}")

    return bare_path


async def ensure_worktree(
    repo_full_name: str,
    pr_number: int,
    pr_branch: str,
) -> Path:
    """Ensure a worktree exists for the PR. Returns the worktree path.

    1. Ensures bare clone exists
    2. Fetches the PR ref
    3. Creates or updates the worktree
    """
    bare_path = await ensure_bare_clone(repo_full_name)
    wt_path = get_worktree_path(repo_full_name, pr_number)

    # Fetch the PR head ref
    rc, _out, err = await _run_git(
        ["fetch", "origin", f"pull/{pr_number}/head:{pr_branch}"],
        cwd=str(bare_path),
        timeout=120,
    )
    if rc != 0:
        logger.warning("Failed to fetch PR ref: %s", err)

    if wt_path.exists():
        # Worktree already exists — pull latest
        rc, _out, err = await _run_git(
            ["checkout", pr_branch],
            cwd=str(wt_path),
            timeout=30,
        )
        if rc != 0:
            logger.warning("Checkout failed in worktree: %s", err)
        rc, _out, err = await _run_git(
            ["pull", "--ff-only", "origin", pr_branch],
            cwd=str(wt_path),
            timeout=60,
        )
        if rc != 0:
            logger.warning("Pull failed in worktree (may need manual resolution): %s", err)
        return wt_path

    # Create new worktree
    wt_path.parent.mkdir(parents=True, exist_ok=True)
    rc, _out, err = await _run_git(
        ["worktree", "add", str(wt_path), pr_branch],
        cwd=str(bare_path),
        timeout=60,
    )
    if rc != 0:
        raise RuntimeError(f"Failed to create worktree: {err}")

    # Set up remote tracking so push works
    await _run_git(
        ["branch", f"--set-upstream-to=origin/{pr_branch}", pr_branch],
        cwd=str(wt_path),
        timeout=10,
    )

    return wt_path


async def sync_worktree(repo_full_name: str, pr_number: int) -> str:
    """Fetch and pull latest changes for a worktree. Returns status message."""
    bare_path = get_bare_repo_path(repo_full_name)
    wt_path = get_worktree_path(repo_full_name, pr_number)

    if not wt_path.exists():
        return "Worktree does not exist"

    # Fetch in bare repo
    await _run_git(["fetch", "--all", "--prune"], cwd=str(bare_path), timeout=120)

    # Pull in worktree
    rc, out, err = await _run_git(
        ["pull", "--ff-only"],
        cwd=str(wt_path),
        timeout=60,
    )
    if rc != 0:
        return f"Pull failed: {err.strip()}"

    return "Synced successfully"


async def commit_and_push(
    repo_full_name: str,
    pr_number: int,
    message: str,
) -> dict:
    """Stage all changes, commit, and push. Returns status dict."""
    wt_path = get_worktree_path(repo_full_name, pr_number)
    if not wt_path.exists():
        return {"ok": False, "error": "Worktree does not exist"}

    # Stage all
    rc, _out, err = await _run_git(["add", "-A"], cwd=str(wt_path), timeout=30)
    if rc != 0:
        return {"ok": False, "error": f"git add failed: {err}"}

    # Check if there's anything to commit
    rc, status_out, _ = await _run_git(
        ["status", "--porcelain"],
        cwd=str(wt_path),
        timeout=10,
    )
    if not status_out.strip():
        return {"ok": False, "error": "No changes to commit"}

    # Commit
    rc, _out, err = await _run_git(
        ["commit", "-m", message],
        cwd=str(wt_path),
        timeout=30,
    )
    if rc != 0:
        return {"ok": False, "error": f"git commit failed: {err}"}

    # Push
    rc, _out, err = await _run_git(
        ["push", "origin", "HEAD"],
        cwd=str(wt_path),
        timeout=120,
    )
    if rc != 0:
        return {"ok": False, "error": f"git push failed: {err}"}

    return {"ok": True, "message": "Changes committed and pushed"}


async def get_worktree_status(repo_full_name: str, pr_number: int) -> dict:
    """Get status of a worktree: exists, path, branch, has_changes, changed_files."""
    wt_path = get_worktree_path(repo_full_name, pr_number)

    if not wt_path.exists():
        return {
            "exists": False,
            "path": str(wt_path),
            "branch": None,
            "has_changes": False,
            "changed_files": [],
        }

    # Get current branch
    rc, branch_out, _ = await _run_git(
        ["rev-parse", "--abbrev-ref", "HEAD"],
        cwd=str(wt_path),
        timeout=10,
    )
    branch = branch_out.strip() if rc == 0 else "unknown"

    # Get status
    rc, status_out, _ = await _run_git(
        ["status", "--porcelain"],
        cwd=str(wt_path),
        timeout=10,
    )
    changed_files = [line[3:] for line in status_out.strip().split("\n") if line.strip()]

    return {
        "exists": True,
        "path": str(wt_path),
        "branch": branch,
        "has_changes": len(changed_files) > 0,
        "changed_files": changed_files,
    }


async def get_worktree_diff(repo_full_name: str, pr_number: int) -> str:
    """Get full git diff for a worktree (staged + unstaged)."""
    wt_path = get_worktree_path(repo_full_name, pr_number)
    if not wt_path.exists():
        return ""

    # Get both staged and unstaged changes
    rc, diff_out, _ = await _run_git(["diff", "HEAD"], cwd=str(wt_path), timeout=30)
    if rc != 0 or not diff_out.strip():
        # Try just unstaged
        rc, diff_out, _ = await _run_git(["diff"], cwd=str(wt_path), timeout=30)

    return diff_out


async def remove_worktree(repo_full_name: str, pr_number: int) -> str:
    """Remove a worktree and clean up."""
    bare_path = get_bare_repo_path(repo_full_name)
    wt_path = get_worktree_path(repo_full_name, pr_number)

    if not wt_path.exists():
        return "Worktree does not exist"

    rc, _out, err = await _run_git(
        ["worktree", "remove", str(wt_path), "--force"],
        cwd=str(bare_path),
        timeout=30,
    )
    if rc != 0:
        # Fallback: just delete the directory
        import shutil as _shutil

        _shutil.rmtree(wt_path, ignore_errors=True)

    # Prune worktree list
    await _run_git(["worktree", "prune"], cwd=str(bare_path), timeout=10)

    return "Worktree removed"


def get_all_worktree_info() -> dict:
    """Get info about all managed worktrees (sync, for status display)."""
    result = {}
    if not _WORKTREES_DIR.exists():
        return result

    for org_dir in _WORKTREES_DIR.iterdir():
        if not org_dir.is_dir():
            continue
        for repo_dir in org_dir.iterdir():
            if not repo_dir.is_dir():
                continue
            repo_full_name = f"{org_dir.name}/{repo_dir.name}"
            for pr_dir in repo_dir.iterdir():
                if not pr_dir.is_dir():
                    continue
                try:
                    pr_number = int(pr_dir.name)
                except ValueError:
                    continue
                result[(repo_full_name, pr_number)] = {
                    "path": str(pr_dir),
                    "created": datetime.fromtimestamp(pr_dir.stat().st_ctime).isoformat(),
                }

    return result
