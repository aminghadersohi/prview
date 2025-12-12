# prview

A beautiful TUI and web dashboard for GitHub PRs and CI status.

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **Your Open PRs** - View all your open PRs across multiple repos/orgs
- **Needs Your Review** - PRs where your review has been requested
- **CI Status** - At-a-glance CI status (pass/fail/running)
- **Review Status** - Approval status (approved/changes requested/required)
- **Diff Stats** - Lines added/removed per PR
- **TUI Mode** - Beautiful terminal UI with keyboard navigation
- **Web UI** - VS Code-inspired dark theme dashboard with real-time updates
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

## Usage

### TUI Mode

```bash
prview                     # Run once and display
prview --watch             # Watch mode with auto-refresh & keyboard navigation
prview --init              # Create default config file
prview --help              # Show all options
```

### Web UI Mode

```bash
prview serve               # Start web server at http://localhost:8420
prview serve --port 3000   # Custom port
prview serve --host 0.0.0.0  # Listen on all interfaces
```

### CLI Options

```bash
# Override config via CLI
prview --repos owner/repo1 owner/repo2
prview --orgs my-org another-org
prview --interval 30       # Refresh every 30 seconds
```

## Keyboard Shortcuts (TUI Watch Mode)

| Key     | Action                          |
|---------|---------------------------------|
| `↑/↓`   | Navigate PRs                    |
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

- **TUI**: Python + [Rich](https://github.com/Textualize/rich) for beautiful terminal rendering
- **Web UI**: FastAPI + Jinja2 + HTMX for reactive server-rendered UI
- **Data**: SQLite for caching, Server-Sent Events for real-time updates
- **GitHub**: Uses `gh` CLI for authentication and API access

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.
