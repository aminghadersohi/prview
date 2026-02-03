"""
Claude CLI wrapper for prview chat integration.
Uses the Claude Code CLI (`claude -p`) for responses — no API key needed.
"""

import asyncio
import json
import os
import shutil
from typing import AsyncGenerator, Optional

# Available Claude models
CLAUDE_MODELS = [
    {
        "id": "claude-sonnet-4-20250514",
        "name": "Sonnet 4",
        "description": "Balanced speed and intelligence (default)",
    },
    {
        "id": "claude-opus-4-20250514",
        "name": "Opus 4",
        "description": "Advanced reasoning and analysis",
    },
    {
        "id": "claude-3-5-haiku-20241022",
        "name": "Haiku 3.5",
        "description": "Fast responses, lower cost",
    },
]

DEFAULT_MODEL = "claude-sonnet-4-20250514"

# Common install paths to check beyond PATH
_COMMON_PATHS = [
    "/usr/local/bin/claude",
    "/opt/homebrew/bin/claude",
    os.path.expanduser("~/.local/bin/claude"),
    os.path.expanduser("~/.npm/bin/claude"),
    os.path.expanduser("~/.claude/local/claude"),
]


def find_claude_cli() -> Optional[str]:
    """Locate the claude binary."""
    # Try shutil.which first (searches PATH)
    path = shutil.which("claude")
    if path:
        return path

    # Check common install locations
    for p in _COMMON_PATHS:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p

    return None


def check_claude_available() -> dict:
    """Check if the Claude CLI is installed and accessible.

    Returns dict with keys: available, version, path, error
    """
    import subprocess

    result = {"available": False, "version": None, "path": None, "error": None}

    cli_path = find_claude_cli()
    if not cli_path:
        result["error"] = (
            "Claude CLI not found. Install it with: npm install -g @anthropic-ai/claude-code"
        )
        return result

    result["path"] = cli_path

    try:
        proc = subprocess.run(
            [cli_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            result["available"] = True
            result["version"] = proc.stdout.strip()
        else:
            result["error"] = f"Claude CLI returned error: {proc.stderr.strip()}"
    except subprocess.TimeoutExpired:
        result["error"] = "Timeout checking Claude CLI version"
    except FileNotFoundError:
        result["error"] = "Claude CLI binary not found at expected path"
    except OSError as e:
        result["error"] = str(e)

    return result


async def stream_claude_response(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system_prompt: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """Stream a response from the Claude CLI.

    Yields structured dicts:
        {"type": "text"|"thinking"|"tool_call"|"tool_result"|"usage"|"system"|"error", "content": ...}
    """
    cli_path = find_claude_cli()
    if not cli_path:
        yield {
            "type": "error",
            "content": "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code",
        }
        return

    cmd = [
        cli_path,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
        "--model",
        model,
        "--",
        prompt,
    ]

    if system_prompt:
        # Insert system prompt args before the -- separator
        separator_idx = cmd.index("--")
        cmd[separator_idx:separator_idx] = ["--system-prompt", system_prompt]

    # Clear ANTHROPIC_API_KEY to force subscription/CLI auth mode
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except FileNotFoundError:
        yield {"type": "error", "content": "Claude CLI binary not found"}
        return
    except OSError as e:
        yield {"type": "error", "content": f"Failed to start Claude CLI: {e}"}
        return

    try:
        async with asyncio.timeout(120):
            while True:
                line = await process.stdout.readline()
                if not line:
                    break

                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue

                try:
                    event = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                # Parse stream-json event types
                event_type = event.get("type", "")

                if event_type == "assistant" and "message" in event:
                    # Start of assistant message — skip
                    continue

                elif event_type == "content_block_start":
                    block = event.get("content_block", {})
                    if block.get("type") == "thinking":
                        yield {"type": "thinking_start", "content": ""}
                    elif block.get("type") == "text":
                        yield {"type": "text_start", "content": ""}

                elif event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    delta_type = delta.get("type", "")
                    if delta_type == "thinking_delta":
                        yield {"type": "thinking", "content": delta.get("thinking", "")}
                    elif delta_type == "text_delta":
                        yield {"type": "text", "content": delta.get("text", "")}

                elif event_type == "content_block_stop":
                    yield {"type": "content_block_stop", "content": ""}

                elif event_type == "result":
                    # Final result with usage info
                    usage = event.get("usage", {})
                    cost = event.get("cost_usd")
                    duration = event.get("duration_ms")
                    session_id = event.get("session_id")
                    yield {
                        "type": "usage",
                        "content": {
                            "input_tokens": usage.get("input_tokens", 0),
                            "output_tokens": usage.get("output_tokens", 0),
                            "cost_usd": cost,
                            "duration_ms": duration,
                            "session_id": session_id,
                        },
                    }

                elif event_type == "system":
                    # System messages from Claude CLI
                    yield {
                        "type": "system",
                        "content": event.get("message", event.get("subtype", "")),
                    }

    except TimeoutError:
        process.kill()
        yield {"type": "error", "content": "Request timed out after 120 seconds"}
    except asyncio.CancelledError:
        process.kill()
        raise

    # Wait for process to finish
    await process.wait()

    # Check for errors on stderr
    if process.returncode and process.returncode != 0:
        stderr_data = await process.stderr.read()
        stderr_str = stderr_data.decode("utf-8", errors="replace").strip()
        if stderr_str:
            yield {"type": "error", "content": f"Claude CLI error: {stderr_str}"}
