"""Export a generated local HTML audit report to PDF with Playwright."""

from __future__ import annotations

from pathlib import Path


INSTALL_GUIDANCE = (
    "PDF generation requires Playwright and Chromium. Install with:\n"
    "pip install playwright\n"
    "python -m playwright install chromium"
)


class PDFExportError(RuntimeError):
    """Raised when the optional PDF export step cannot be completed."""


def export_html_to_pdf(html_path: str | Path, pdf_path: str | Path) -> Path:
    """Render a local HTML report as an A4 PDF and return its resolved path."""
    html_file = Path(html_path).resolve()
    pdf_file = Path(pdf_path).resolve()
    if not html_file.is_file():
        raise PDFExportError(f"HTML report not found: {html_file}")

    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise PDFExportError(INSTALL_GUIDANCE) from exc

    pdf_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(html_file.as_uri(), wait_until="networkidle")
                page.evaluate(
                    "document.querySelectorAll('details').forEach(item => item.open = true)"
                )
                page.wait_for_function(
                    "Array.from(document.images).every(image => image.complete)",
                    timeout=30_000,
                )
                broken_images = page.evaluate(
                    "Array.from(document.images)"
                    ".filter(image => image.naturalWidth === 0)"
                    ".map(image => image.src)"
                )
                if broken_images:
                    raise PDFExportError(
                        "PDF export could not load report image(s): "
                        + ", ".join(broken_images)
                    )
                page.emulate_media(media="print")
                page.pdf(
                    path=str(pdf_file),
                    format="A4",
                    print_background=True,
                    prefer_css_page_size=True,
                )
            finally:
                browser.close()
    except PDFExportError:
        raise
    except PlaywrightError as exc:
        raise PDFExportError(f"{INSTALL_GUIDANCE}\nPlaywright error: {exc}") from exc
    except Exception as exc:
        raise PDFExportError(f"PDF generation failed: {type(exc).__name__}: {exc}") from exc

    if not pdf_file.is_file() or pdf_file.stat().st_size == 0:
        raise PDFExportError(f"Playwright did not create a valid PDF file: {pdf_file}")
    return pdf_file
