"""Tests for CI analysis features (Jenkins + Datadog integration)."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prview.web.server import (
    app,
    build_ci_analysis_context,
    extract_error_context,
    find_deploy_slug_in_log,
    get_jenkins_build_info,
    get_jenkins_job_path,
    get_jenkins_stages,
    init_db,
    jenkins_request,
    search_datadog_logs,
)


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """Use a temporary database for each test."""
    db_path = tmp_path / "prview.db"
    monkeypatch.setattr("prview.web.server.DB_PATH", db_path)
    init_db()
    return db_path


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# --- Unit tests for helper functions ---


class TestGetJenkinsJobPath:
    def test_standard_repo(self):
        assert (
            get_jenkins_job_path("preset-io/superset-shell") == "/job/preset-io/job/superset-shell"
        )

    def test_other_repo(self):
        assert get_jenkins_job_path("myorg/myrepo") == "/job/myorg/job/myrepo"

    def test_no_slash(self):
        assert get_jenkins_job_path("repo") == "/job/repo"


class TestExtractErrorContext:
    def test_finds_error_lines(self):
        log = "line 1\nline 2\nERROR: something failed\nline 4\nline 5"
        result = extract_error_context(log)
        assert "ERROR: something failed" in result

    def test_finds_failure_lines(self):
        log = "ok\nok\nFAILED: test_foo\nok"
        result = extract_error_context(log)
        assert "FAILED: test_foo" in result

    def test_finds_exit_code(self):
        log = "running\nscript returned exit code 1\ndone"
        result = extract_error_context(log)
        assert "exit code 1" in result

    def test_fallback_to_last_lines(self):
        lines = [f"normal line {i}" for i in range(200)]
        log = "\n".join(lines)
        result = extract_error_context(log)
        # Should return last 100 lines as fallback
        assert "normal line 199" in result

    def test_truncates_long_output(self):
        lines = [f"ERROR: very long error message number {i} " * 10 for i in range(100)]
        log = "\n".join(lines)
        result = extract_error_context(log)
        assert len(result) <= 4100  # 4000 + truncation message


class TestFindDeploySlugInLog:
    def test_finds_slug(self):
        log = "deploying to ws--abc123def--main environment"
        assert find_deploy_slug_in_log(log) == "ws--abc123def--main"

    def test_no_slug(self):
        log = "normal build output"
        assert find_deploy_slug_in_log(log) is None

    def test_finds_first_slug(self):
        log = "ws--aaa111--staging and ws--bbb222--main"
        result = find_deploy_slug_in_log(log)
        assert result == "ws--aaa111--staging"


class TestJenkinsRequest:
    def test_no_credentials(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.yaml"
        monkeypatch.setattr("prview.web.server.CONFIG_PATH", config_path)
        assert jenkins_request("/api/json") is None

    def test_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"mode": "NORMAL"}'

        ci_config = {
            "jenkins_url": "https://jenkins.example.com",
            "jenkins_user": "user",
            "jenkins_token": "token",
        }
        with patch("prview.web.server.httpx.get", return_value=mock_resp):
            result = jenkins_request("/api/json", ci_config)
            assert result == '{"mode": "NORMAL"}'

    def test_auth_failure(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 401

        ci_config = {
            "jenkins_url": "https://jenkins.example.com",
            "jenkins_user": "user",
            "jenkins_token": "bad",
        }
        with patch("prview.web.server.httpx.get", return_value=mock_resp):
            assert jenkins_request("/api/json", ci_config) is None

    def test_timeout(self):
        import httpx

        ci_config = {
            "jenkins_url": "https://jenkins.example.com",
            "jenkins_user": "user",
            "jenkins_token": "token",
        }
        with patch(
            "prview.web.server.httpx.get",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            assert jenkins_request("/api/json", ci_config) is None


class TestGetJenkinsBuildInfo:
    def test_no_connection(self):
        ci_config = {
            "jenkins_url": "https://jenkins.example.com",
            "jenkins_user": "",
            "jenkins_token": "",
        }
        result = get_jenkins_build_info("org/repo", 1, ci_config)
        assert "error" in result

    def test_success(self):
        job_json = json.dumps(
            {
                "lastBuild": {"number": 5, "url": "http://jenkins/job/5/"},
                "color": "red",
            }
        )
        build_json = json.dumps(
            {
                "result": "FAILURE",
                "building": False,
                "duration": 300000,
            }
        )

        ci_config = {
            "jenkins_url": "https://jenkins.example.com",
            "jenkins_user": "user",
            "jenkins_token": "token",
        }

        def mock_request(path, config=None):
            if "api/json" in path and "/5/" not in path:
                return job_json
            if "/5/api/json" in path:
                return build_json
            return None

        with patch("prview.web.server.jenkins_request", side_effect=mock_request):
            result = get_jenkins_build_info("org/repo", 1, ci_config)
            assert result["build_number"] == 5
            assert result["result"] == "FAILURE"
            assert result["duration_min"] == 5.0


class TestGetJenkinsStages:
    def test_success(self):
        stages_json = json.dumps(
            {
                "stages": [
                    {"name": "Init", "status": "SUCCESS", "durationMillis": 60000},
                    {"name": "Build", "status": "FAILED", "durationMillis": 120000},
                ]
            }
        )

        ci_config = {
            "jenkins_url": "https://jenkins.example.com",
            "jenkins_user": "user",
            "jenkins_token": "token",
        }
        with patch("prview.web.server.jenkins_request", return_value=stages_json):
            stages = get_jenkins_stages("org/repo", 1, 5, ci_config)
            assert len(stages) == 2
            assert stages[0]["name"] == "Init"
            assert stages[0]["status"] == "SUCCESS"
            assert stages[1]["name"] == "Build"
            assert stages[1]["status"] == "FAILED"

    def test_no_connection(self):
        ci_config = {
            "jenkins_url": "https://jenkins.example.com",
            "jenkins_user": "",
            "jenkins_token": "",
        }
        stages = get_jenkins_stages("org/repo", 1, 5, ci_config)
        assert stages == []


class TestSearchDatadogLogs:
    def test_no_credentials(self):
        ci_config = {"datadog_api_key": "", "datadog_app_key": ""}
        assert search_datadog_logs("query", ci_config=ci_config) == []

    def test_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {
                    "attributes": {
                        "message": "Error occurred",
                        "timestamp": "2024-01-01T00:00:00Z",
                        "status": "error",
                        "exc_info": "Traceback...",
                        "error": {"kind": "RuntimeError"},
                    }
                }
            ]
        }

        ci_config = {"datadog_api_key": "key", "datadog_app_key": "app"}
        with patch("prview.web.server.httpx.post", return_value=mock_resp):
            logs = search_datadog_logs("query", ci_config=ci_config)
            assert len(logs) == 1
            assert logs[0]["message"] == "Error occurred"
            assert logs[0]["exc_info"] == "Traceback..."

    def test_api_error(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 403

        ci_config = {"datadog_api_key": "key", "datadog_app_key": "app"}
        with patch("prview.web.server.httpx.post", return_value=mock_resp):
            assert search_datadog_logs("query", ci_config=ci_config) == []


class TestBuildCiAnalysisContext:
    def test_no_jenkins_config(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.yaml"
        monkeypatch.setattr("prview.web.server.CONFIG_PATH", config_path)
        result = build_ci_analysis_context("org/repo", 1, "explain")
        assert "error" in result["build_info"] or result["prompt"]

    def test_builds_prompt_with_data(self):
        mock_build_info = {
            "build_number": 3,
            "color": "red",
            "url": "http://jenkins/3/",
            "result": "FAILURE",
            "building": False,
            "duration_min": 5.0,
        }
        mock_stages = [
            {"name": "Init", "status": "SUCCESS", "duration_min": 1.0},
            {"name": "Build", "status": "FAILED", "duration_min": 4.0},
        ]
        mock_console = "line1\nERROR: pip install failed\nexit code 1"

        with (
            patch(
                "prview.web.server.get_jenkins_build_info",
                return_value=mock_build_info,
            ),
            patch(
                "prview.web.server.get_jenkins_stages",
                return_value=mock_stages,
            ),
            patch(
                "prview.web.server.get_jenkins_console_log",
                return_value=mock_console,
            ),
            patch(
                "prview.web.server.search_datadog_logs",
                return_value=[],
            ),
        ):
            result = build_ci_analysis_context("org/repo", 1, "explain")
            assert "FAILURE" in result["prompt"]
            assert "Build" in result["prompt"]
            assert "pip install failed" in result["prompt"]
            assert "Analyze this CI failure" in result["prompt"]

    def test_fix_mode_prompt(self):
        mock_build_info = {
            "build_number": 1,
            "color": "red",
            "url": "",
            "result": "FAILURE",
        }

        with (
            patch(
                "prview.web.server.get_jenkins_build_info",
                return_value=mock_build_info,
            ),
            patch(
                "prview.web.server.get_jenkins_stages",
                return_value=[],
            ),
            patch(
                "prview.web.server.get_jenkins_console_log",
                return_value="ERROR: test",
            ),
            patch(
                "prview.web.server.search_datadog_logs",
                return_value=[],
            ),
        ):
            result = build_ci_analysis_context("org/repo", 1, "fix")
            assert "suggest how to fix" in result["prompt"]


# --- Route tests ---


class TestCiAnalysisRoutes:
    def test_ci_analysis_page(self, client):
        with patch(
            "prview.web.server.get_pr_basic_info",
            return_value={
                "title": "Test PR",
                "url": "https://github.com/org/repo/pull/1",
            },
        ):
            response = client.get("/pr/org/repo/1/ci-analysis?mode=explain")
            assert response.status_code == 200
            assert "CI" in response.text or "Analyze" in response.text

    def test_ci_analysis_page_fix_mode(self, client):
        with patch(
            "prview.web.server.get_pr_basic_info",
            return_value={
                "title": "Test PR",
                "url": "https://github.com/org/repo/pull/1",
            },
        ):
            response = client.get("/pr/org/repo/1/ci-analysis?mode=fix")
            assert response.status_code == 200
            assert "Fix" in response.text or "Suggest" in response.text


class TestCiConfigValidateRoute:
    def test_validate_jenkins_no_creds(self, client):
        response = client.post(
            "/api/ci-config/validate",
            json={"service": "jenkins", "config": {}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert "required" in data["error"].lower()

    def test_validate_jenkins_success(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("prview.web.server.httpx.get", return_value=mock_resp):
            response = client.post(
                "/api/ci-config/validate",
                json={
                    "service": "jenkins",
                    "config": {
                        "jenkins_url": "https://jenkins.example.com",
                        "jenkins_user": "user",
                        "jenkins_token": "token",
                    },
                },
            )
            assert response.status_code == 200
            assert response.json()["valid"] is True

    def test_validate_jenkins_auth_failure(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 401

        with patch("prview.web.server.httpx.get", return_value=mock_resp):
            response = client.post(
                "/api/ci-config/validate",
                json={
                    "service": "jenkins",
                    "config": {
                        "jenkins_url": "https://jenkins.example.com",
                        "jenkins_user": "user",
                        "jenkins_token": "bad",
                    },
                },
            )
            assert response.status_code == 200
            assert response.json()["valid"] is False

    def test_validate_datadog_no_keys(self, client):
        response = client.post(
            "/api/ci-config/validate",
            json={"service": "datadog", "config": {}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False

    def test_validate_datadog_success(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("prview.web.server.httpx.get", return_value=mock_resp):
            response = client.post(
                "/api/ci-config/validate",
                json={
                    "service": "datadog",
                    "config": {
                        "datadog_api_key": "key",
                        "datadog_app_key": "app",
                    },
                },
            )
            assert response.status_code == 200
            assert response.json()["valid"] is True

    def test_validate_unknown_service(self, client):
        response = client.post(
            "/api/ci-config/validate",
            json={"service": "unknown", "config": {}},
        )
        assert response.status_code == 200
        assert response.json()["valid"] is False


class TestSettingsCiConfig:
    def test_settings_page_shows_jenkins(self, client):
        with patch(
            "prview.web.server.check_claude_available",
            return_value={"available": False},
        ):
            response = client.get("/settings")
            assert response.status_code == 200
            assert "Jenkins" in response.text

    def test_settings_page_shows_datadog(self, client):
        with patch(
            "prview.web.server.check_claude_available",
            return_value={"available": False},
        ):
            response = client.get("/settings")
            assert response.status_code == 200
            assert "Datadog" in response.text

    def test_save_settings_with_ci_config(self, client, tmp_path, monkeypatch):
        import yaml

        config_path = tmp_path / "config.yaml"
        monkeypatch.setattr("prview.web.server.CONFIG_PATH", config_path)
        response = client.post(
            "/settings/save",
            data={
                "include_orgs": "",
                "exclude_orgs": "",
                "include_repos": "",
                "exclude_repos": "",
                "refresh_interval": "60",
                "max_prs_per_repo": "10",
                "task_provider": "",
                "task_api_token": "",
                "jenkins_url": "https://jenkins.mycompany.com",
                "jenkins_user": "admin",
                "jenkins_token": "mytoken",
                "datadog_api_key": "ddkey",
                "datadog_app_key": "ddapp",
            },
        )
        assert response.status_code == 200
        saved = yaml.safe_load(config_path.read_text())
        assert saved["jenkins_url"] == "https://jenkins.mycompany.com"
        assert saved["jenkins_user"] == "admin"
        assert saved["jenkins_token"] == "mytoken"
        assert saved["datadog_api_key"] == "ddkey"
        assert saved["datadog_app_key"] == "ddapp"
