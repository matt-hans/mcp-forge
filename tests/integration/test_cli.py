"""Integration tests for CLI commands."""

from __future__ import annotations

from click.testing import CliRunner
import pytest

from mcp_forge.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    """Create CLI test runner."""
    return CliRunner()


class TestCLICommands:
    """Tests for CLI command invocations."""

    def test_cli_help(self, runner: CliRunner) -> None:
        """Test --help displays usage."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Fine-tune" in result.output

    def test_cli_version(self, runner: CliRunner) -> None:
        """Test --version displays version."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_status_no_session(self, runner: CliRunner) -> None:
        """Test status command with no active session."""
        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "No active session" in result.output

    def test_tools_inspect_missing_server(self, runner: CliRunner) -> None:
        """Test tools inspect requires --server flag."""
        result = runner.invoke(cli, ["tools", "inspect"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_qa_missing_data(self, runner: CliRunner) -> None:
        """Test qa command requires --data flag."""
        result = runner.invoke(cli, ["qa"])
        assert result.exit_code != 0

    def test_run_requires_server_or_tools(self, runner: CliRunner) -> None:
        """Test run command requires --server or --tools-file."""
        result = runner.invoke(cli, ["run", "--output", "./test"])
        assert result.exit_code != 0
        assert "required" in result.output.lower() or "Error" in result.output


class TestDoctorCommand:
    """Tests for doctor command."""

    @pytest.mark.slow
    def test_doctor_runs(self, runner: CliRunner) -> None:
        """Test doctor command executes without error."""
        result = runner.invoke(cli, ["doctor"])
        # Doctor should run even if some checks fail
        assert result.exit_code == 0
        assert "Environment Check" in result.output
        assert "Python" in result.output


class TestToolsCommand:
    """Tests for tools subcommands."""

    def test_tools_help(self, runner: CliRunner) -> None:
        """Test tools --help displays subcommands."""
        result = runner.invoke(cli, ["tools", "--help"])
        assert result.exit_code == 0
        assert "inspect" in result.output
        assert "import" in result.output

    def test_tools_import_not_implemented(self, runner: CliRunner) -> None:
        """Test tools import shows not implemented message."""
        # Create a temp file for the --from argument
        with runner.isolated_filesystem():
            with open("tools.json", "w") as f:
                f.write("[]")
            result = runner.invoke(cli, ["tools", "import", "--from", "tools.json"])
            assert result.exit_code == 0
            assert "v1.2" in result.output


class TestQACommand:
    """Tests for qa command with threshold options."""

    def test_qa_help_shows_options(self, runner: CliRunner) -> None:
        """Test qa --help displays threshold options."""
        result = runner.invoke(cli, ["qa", "--help"])
        assert result.exit_code == 0
        assert "--threshold" in result.output
        assert "--min-samples" in result.output
        assert "--no-dedup" in result.output
        assert "--no-auto-repair" in result.output
        assert "--strict" in result.output

    def test_qa_with_custom_threshold(
        self,
        runner: CliRunner,
        sample_data_path,
        sample_tools_path,
    ) -> None:
        """Test qa command with custom threshold."""
        result = runner.invoke(cli, [
            "qa",
            "--data", str(sample_data_path),
            "--tools", str(sample_tools_path),
            "--threshold", "0.50",
            "--min-samples", "1",
        ])
        # Should pass with low thresholds
        assert "Quality Report" in result.output

    def test_qa_with_strict_mode(
        self,
        runner: CliRunner,
        sample_data_path,
        sample_tools_path,
    ) -> None:
        """Test qa command with --strict mode."""
        result = runner.invoke(cli, [
            "qa",
            "--data", str(sample_data_path),
            "--tools", str(sample_tools_path),
            "--strict",
            "--min-samples", "1",
        ])
        # Output should include strict mode handling
        assert "Quality Report" in result.output or "failed" in result.output

    def test_qa_with_no_dedup(
        self,
        runner: CliRunner,
        sample_data_path,
        sample_tools_path,
    ) -> None:
        """Test qa command with --no-dedup flag."""
        result = runner.invoke(cli, [
            "qa",
            "--data", str(sample_data_path),
            "--tools", str(sample_tools_path),
            "--no-dedup",
            "--min-samples", "1",
        ])
        assert "Quality Report" in result.output


class TestDataCommands:
    """Tests for data-related commands."""

    def test_generate_requires_data(self, runner: CliRunner) -> None:
        """Test generate command requires --data flag."""
        result = runner.invoke(cli, ["generate"])
        assert result.exit_code != 0

    def test_train_requires_data(self, runner: CliRunner) -> None:
        """Test train command requires --data flag."""
        result = runner.invoke(cli, ["train"])
        assert result.exit_code != 0

    def test_validate_requires_model(self, runner: CliRunner) -> None:
        """Test validate command requires --model flag."""
        result = runner.invoke(cli, ["validate"])
        assert result.exit_code != 0

    def test_benchmark_requires_model(self, runner: CliRunner) -> None:
        """Test benchmark command requires --model flag."""
        result = runner.invoke(cli, ["benchmark"])
        assert result.exit_code != 0


class TestExportCommands:
    """Tests for export and packaging commands."""

    def test_export_requires_model(self, runner: CliRunner) -> None:
        """Test export command requires --model flag."""
        result = runner.invoke(cli, ["export"])
        assert result.exit_code != 0

    def test_pack_requires_model(self, runner: CliRunner) -> None:
        """Test pack command requires --model flag."""
        result = runner.invoke(cli, ["pack"])
        assert result.exit_code != 0

    def test_verify_bundle_requires_path(self, runner: CliRunner) -> None:
        """Test verify-bundle command requires path argument."""
        result = runner.invoke(cli, ["verify-bundle"])
        assert result.exit_code != 0
