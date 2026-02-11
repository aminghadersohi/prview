# prview

A powerful PR dashboard with AI-powered code agent, CI analysis, and review tools — all from your browser or terminal.

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
[![PyPI](https://img.shields.io/pypi/v/prview.svg)](https://pypi.org/project/prview/)

<img width="1737" height="894" alt="prview dashboard" src="https://github.com/user-attachments/assets/67776c53-12c2-4b60-b6bf-e44240fb1aec" />
<img width="1746" height="977" alt="Screenshot 2026-02-11 at 2 53 46 PM" src="https://github.com/user-attachments/assets/11002dc2-9289-47f1-9dba-eaec0893e503" />



## Features

### Dashboard
- **Your Open PRs** - View all your open PRs across multiple repos/orgs
- **Needs Your Review** - PRs where your review has been requested
- **CI Status** - At-a-glance CI status (pass/fail/running)
- **Review Status** - Approval status (approved/changes requested/required)
- **Review Thread Counts** - See unresolved comment threads per PR
- **Diff Stats** - Lines added/removed per PR
- **Row & Card Views** - Toggle between compact rows and visual cards
- **Real-time Updates** - Server-Sent Events push changes as they happen

### AI Agent (Claude Code CLI)
- **Per-PR Chat** - Persistent chat sessions scoped to each pull request
- **Git Worktrees** - Isolated checkouts so the agent can read and modify code
- **Session Persistence** - Conversations survive page refreshes and carry context across messages
- **Streaming Responses** - Real-time SSE streaming with thinking/reasoning display
- **Model Selection** - Switch between Claude Sonnet 4, Opus 4, and Haiku 3.5
- **No API Key Needed** - Uses your Claude Code CLI subscription

### CI Analysis
- **Explain Failures** - AI-powered explanation of CI build failures
- **Suggest Fixes** - Get actionable fix suggestions for broken builds
- **Jenkins Integration** - Direct log fetching from Jenkins CI
- **Datadog Logs** - Search runtime logs for deployment failures

### Review Tools
- **AI-Assisted Review** - Get an AI review of PR changes
- **Comment Addressing** - View and address reviewer comments with AI help
- **Apply Changes** - Pre-fill the agent chat with review comment context

### Task Management
- **Shortcut Integration** - See linked Shortcut stories on PR rows
- **Story Badges** - Story ID and workflow state displayed inline

### Modes
- **Web UI** - VS Code-inspired dark theme dashboard
- **TUI Mode** - Beautiful terminal UI with keyboard navigation
- **Configurable** - YAML config for orgs, repos, includes/excludes

## Installation

### From PyPI

```bash
pip install prview

# With keyboard navigation support (TUI)
pip install prview[keyboard]

# With web UI support
pip install prview[web]

# Everything
pip install prview[all]
```

### From source

```bash
git clone https://github.com/aminghadersohi/prview.git
cd prview
pip install -e ".[all]"
```

## Requirements

- Python 3.9+
- `gh` CLI installed and authenticated (`gh auth login`)
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (optional, for AI features)

## Usage

### Web UI Mode

```bash
prview serve               # Start web server at http://localhost:8420
prview serve --port 3000   # Custom port
prview serve --host 0.0.0.0  # Listen on all interfaces
```

### TUI Mode

```bash
prview                     # Run once and display
prview --watch             # Watch mode with auto-refresh & keyboard navigation
prview --init              # Create default config file
prview --help              # Show all options
```

### CLI Options

```bash
# Override config via CLI
prview --repos owner/repo1 owner/repo2
prview --orgs my-org another-org
prview --interval 30       # Refresh every 30 seconds
```

## Web UI Pages

| Page | Path | Description |
|------|------|-------------|
| Dashboard | `/` | PR list with CI, review, and comment status |
| PR Agent Chat | `/pr/{repo}/{number}/chat` | Per-PR AI chat with worktree support |
| General Chat | `/chat` | General-purpose AI assistant chat |
| AI Review | `/pr/{repo}/{number}/review` | AI-powered code review |
| Comments | `/pr/{repo}/{number}/comments` | Address reviewer comments |
| CI Analysis | `/pr/{repo}/{number}/ci-analysis` | Explain or fix CI failures |
| Settings | `/settings` | Configure repos, model, integrations |

## Keyboard Shortcuts (TUI Watch Mode)

| Key     | Action                          |
|---------|---------------------------------|
| `Up/Down`   | Navigate PRs                    |
| `Tab`   | Switch between sections         |
| `Enter` | Open selected PR in browser     |
| `r`     | Refresh data                    |
| `q`     | Quit                            |

## Configuration

Config file location: `~/.config/prview/config.yaml`

```yaml
# Organizations to include
include_orgs:
  - my-org
  - another-org

# Specific repos to include
include_repos:
  - owner/specific-repo

# Organizations to exclude
exclude_orgs:
  - archived-org

# Specific repos to exclude
exclude_repos:
  - owner/archived-repo

# Auto-refresh interval in seconds
refresh_interval: 60

# Show draft PRs
show_drafts: true

# Max PRs to show per repository
max_prs_per_repo: 10

# Default Claude model for AI features
claude_model: claude-sonnet-4-20250514
```

## Status Icons

### CI Status
- `✓` (green) - All checks passed
- `✗` (red) - Checks failed
- `◐` (yellow) - Checks running/pending
- `○` (dim) - No checks

### Review Status
- `✓` (green) - Approved
- `✗` (red) - Changes requested
- `●` (yellow) - Review required
- `○` (dim) - Pending/No reviews

## Architecture

- **Web UI**: FastAPI + Jinja2 + HTMX for reactive server-rendered UI
- **TUI**: Python + [Rich](https://github.com/Textualize/rich) for beautiful terminal rendering
- **AI**: Claude Code CLI (`claude -p --output-format stream-json`) for agent features
- **Worktrees**: Git worktrees for isolated per-PR code checkouts
- **Data**: SQLite for caching, Server-Sent Events for real-time updates
- **GitHub**: Uses `gh` CLI for authentication and API access

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.
