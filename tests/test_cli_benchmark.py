from typer.testing import CliRunner
from transcribe.cli import app
from transcribe.models import MODEL_CATALOG

runner = CliRunner()


def test_model_catalog_structure():
    assert len(MODEL_CATALOG) >= 15
    model_names = [m.name for m in MODEL_CATALOG]
    assert "tiny" in model_names
    assert "base" in model_names
    assert "small" in model_names
    assert "medium" in model_names
    assert "large-v3" in model_names
    assert "turbo" in model_names


def test_cli_models_command():
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "tiny" in result.stdout
    assert "base" in result.stdout
    assert "Whisper" in result.stdout


def test_cli_benchmark_command():
    result = runner.invoke(
        app,
        [
            "benchmark",
            "data/sample/proklamasi.wav",
            "--models",
            "tiny",
            "--models",
            "base",
            "--language",
            "id",
        ],
    )
    assert result.exit_code == 0
    assert "Sequential Benchmark Results" in result.stdout
    assert "tiny" in result.stdout
    assert "base" in result.stdout
    assert "✔ Passed" in result.stdout


def test_cli_test_all_retry_failed(tmp_path):
    # 1. Run tiny in tmp_path
    res1 = runner.invoke(
        app,
        [
            "test-all",
            "--audio",
            "data/sample/proklamasi.wav",
            "--output-dir",
            str(tmp_path),
            "--models",
            "tiny",
            "--language",
            "id",
        ],
    )
    assert res1.exit_code == 0
    assert (tmp_path / "tiny" / "metrics.json").exists()

    # 2. Retry failed when all passed
    res2 = runner.invoke(
        app,
        [
            "test-all",
            "--output-dir",
            str(tmp_path),
            "--retry-failed",
        ],
    )
    assert res2.exit_code == 0
    assert "No failed models found" in res2.stdout

