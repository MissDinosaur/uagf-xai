from __future__ import annotations

import builtins
import sys
import types

import pytest

from report.pdf_exporter import PDFExportError, export_html_to_pdf


def test_pdf_exporter_rejects_missing_html(tmp_path):
    with pytest.raises(PDFExportError, match="HTML report not found"):
        export_html_to_pdf(tmp_path / "missing.html", tmp_path / "report.pdf")


def test_pdf_exporter_has_clear_missing_dependency_guidance(tmp_path, monkeypatch):
    html = tmp_path / "report.html"
    html.write_text("<html><body>Test</body></html>", encoding="utf-8")
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("playwright"):
            raise ImportError("Playwright blocked for unit test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(PDFExportError) as error:
        export_html_to_pdf(html, tmp_path / "report.pdf")

    assert "pip install playwright" in str(error.value)
    assert "python -m playwright install chromium" in str(error.value)


def test_pdf_exporter_opens_details_and_writes_pdf_with_mocked_playwright(
    tmp_path, monkeypatch
):
    html = tmp_path / "report.html"
    pdf = tmp_path / "report.pdf"
    html.write_text("<html><body><details>Raw</details></body></html>", encoding="utf-8")
    calls = {"evaluations": []}

    class FakePage:
        def goto(self, uri, wait_until):
            calls["uri"] = uri
            calls["wait_until"] = wait_until

        def evaluate(self, script):
            calls["evaluations"].append(script)
            return [] if "naturalWidth" in script else None

        def wait_for_function(self, script, timeout):
            calls["wait"] = (script, timeout)

        def emulate_media(self, media):
            calls["media"] = media

        def pdf(self, path, **kwargs):
            calls["pdf_options"] = kwargs
            with open(path, "wb") as handle:
                handle.write(b"%PDF-mocked")

    class FakeBrowser:
        def new_page(self):
            return FakePage()

        def close(self):
            calls["closed"] = True

    class FakeChromium:
        def launch(self, headless):
            calls["headless"] = headless
            return FakeBrowser()

    class FakeManager:
        def __enter__(self):
            return types.SimpleNamespace(chromium=FakeChromium())

        def __exit__(self, exc_type, exc, traceback):
            return False

    package = types.ModuleType("playwright")
    package.__path__ = []
    sync_api = types.ModuleType("playwright.sync_api")
    sync_api.Error = RuntimeError
    sync_api.sync_playwright = lambda: FakeManager()
    monkeypatch.setitem(sys.modules, "playwright", package)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)

    result = export_html_to_pdf(html, pdf)

    assert result == pdf.resolve()
    assert pdf.read_bytes() == b"%PDF-mocked"
    assert calls["uri"] == html.resolve().as_uri()
    assert any("details" in script for script in calls["evaluations"])
    assert calls["media"] == "print"
    assert calls["pdf_options"]["format"] == "A4"
    assert calls["pdf_options"]["landscape"] is True
    assert calls["pdf_options"]["print_background"] is True
    assert calls["pdf_options"]["prefer_css_page_size"] is True
    assert calls["closed"] is True


@pytest.mark.integration
def test_real_playwright_can_convert_tiny_html_when_runtime_is_available(tmp_path):
    pytest.importorskip("playwright.sync_api")
    html = tmp_path / "report.html"
    pdf = tmp_path / "report.pdf"
    html.write_text("<html><body><h1>PDF smoke test</h1></body></html>", encoding="utf-8")

    try:
        result = export_html_to_pdf(html, pdf)
    except PDFExportError as exc:
        pytest.skip(f"Local Chromium runtime is unavailable: {exc}")

    assert result == pdf.resolve()
    assert pdf.read_bytes().startswith(b"%PDF-")
