"""The CLI fails in one line, never a traceback: missing or unreadable files, and `--only` naming no sample."""

import pymupdf
import pytest
from typer.testing import CliRunner

from idp.cli import app
from idp.config import SAMPLES_DIR

runner = CliRunner()


def flat(output: str) -> str:
    """Output with rich's wrapping undone, so a long message can be matched whole."""
    return " ".join(output.split())


def test_missing_input_file_is_a_usage_error(tmp_path):
    result = runner.invoke(app, ["run", str(tmp_path / "missing.pdf")])
    assert result.exit_code == 2
    assert "missing.pdf" in flat(result.output).replace(" ", "") and "Traceback" not in result.output


def test_unsupported_file_type_is_one_line(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# notes\n", encoding="utf-8")
    result = runner.invoke(app, ["ocr", str(readme)])
    assert result.exit_code == 1
    assert "Unsupported file type" in flat(result.output) and "Traceback" not in result.output
    assert not isinstance(result.exception, ValueError)


def test_password_protected_pdf_is_one_line(tmp_path):
    path = tmp_path / "locked.pdf"
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "secret")
    pdf.save(str(path), encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    result = runner.invoke(app, ["ocr", str(path)])
    assert result.exit_code == 1
    assert "password-protected" in flat(result.output) and "Traceback" not in result.output


@pytest.mark.skipif(not (SAMPLES_DIR / "scanned" / "invoice-01.png").exists(), reason="run `idp generate` first")
def test_ocr_model_download_failure_is_one_line(monkeypatch):
    import rapidocr
    from rapidocr.utils.download_file import DownloadFileException

    from idp import ocr

    def no_download(*args, **kwargs):
        raise DownloadFileException("Failed to download https://modelscope.cn/models/RapidAI/RapidOCR/...")

    ocr._engine.cache_clear()
    monkeypatch.setattr(rapidocr, "RapidOCR", no_download)
    try:
        result = runner.invoke(app, ["ocr", str(SAMPLES_DIR / "scanned" / "invoice-01.png")])
    finally:
        ocr._engine.cache_clear()
    assert result.exit_code == 1
    assert "OCR model download failed" in flat(result.output) and "Traceback" not in result.output


@pytest.mark.skipif(not (SAMPLES_DIR / "manifest.json").exists(), reason="run `idp generate` first")
def test_only_naming_no_sample_is_a_usage_error(monkeypatch):
    def no_clients(**kwargs):
        raise AssertionError("no model client may be built when --only names nothing")

    monkeypatch.setattr("idp.evaluate.Pipeline", no_clients)
    result = runner.invoke(app, ["eval", "--only", "invoices"])
    assert result.exit_code == 2
    assert "no sample matches" in flat(result.output).lower() and "Traceback" not in result.output


def test_bad_source_and_mode_are_usage_errors(monkeypatch):
    monkeypatch.setattr("idp.evaluate.Pipeline", lambda **kwargs: pytest.fail("no client may be built"))
    assert runner.invoke(app, ["eval", "--source", "photos"]).exit_code == 2
    assert runner.invoke(app, ["eval", "--mode", "fast"]).exit_code == 2
    assert runner.invoke(app, ["eval", "--workers", "0"]).exit_code == 2
