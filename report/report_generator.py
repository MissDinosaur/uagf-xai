import base64
import json
from jinja2 import Environment, FileSystemLoader, select_autoescape
import os
from pathlib import Path
from output_naming import build_output_path
from report.report_builder import build_report_model
from report.pdf_exporter import PDFExportError, export_html_to_pdf
from report.method_titles import REPORT_METHOD_TITLES, report_method_title
from report.method_ordering import ordered_cbep_trace, ordered_result_keys
from report.runtime_metadata import (
    collect_reproducibility_metadata,
    public_report_reproducibility_metadata,
)


def _to_json_pretty(value):
    """Jinja2 filter: convert any value to indented JSON string."""
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _image_data_uri(artifact_path):
    """Embed a local PNG or JPEG artifact without replacing its source path."""
    path_text = str(artifact_path or "")
    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(Path(path_text).suffix.lower())
    if not mime_type:
        return None

    image_path = Path(path_text.removeprefix("file://"))
    if not image_path.is_absolute():
        image_path = Path.cwd() / image_path
    if not image_path.is_file():
        return None

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _section_label(key: str) -> str:
    if key == "_cbep_trace":
        return "CBEP planning trace"
    if key == "_reproducibility":
        return "Reproducibility metadata"
    if key in REPORT_METHOD_TITLES:
        return report_method_title(key)

    pretty = key.replace("_", " ").strip()
    if not pretty:
        return key

    return pretty[:1].upper() + pretty[1:]


def _ordered_report_sections(results):
    return [
        {
            "key": key,
            "label": _section_label(key),
            "value": (
                ordered_cbep_trace(results[key])
                if key == "_cbep_trace" and isinstance(results[key], dict)
                else results[key]
            ),
        }
        for key in ordered_result_keys(results)
        if key == "_cbep_trace"
        or not (
            isinstance(results[key], dict)
            and str(results[key].get("status", "")).lower() == "not_applicable"
        )
    ]


def generate_report(
    results,
    risk_level,
    provider_name=None,
    output_namespace="audit",
    audit_context=None,
    governance_context=None,
    resource_context=None,
    generate_pdf=True,
    runtime_context=None,
):

    env = Environment(
        loader=FileSystemLoader("report/templates"),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.filters["tojson_pretty"] = _to_json_pretty
    template = env.get_template("report_template.html")

    os.makedirs("outputs/report", exist_ok=True)

    case_name = provider_name or output_namespace or "audit"
    output_path = build_output_path(
        "report",
        provider_name,
        "audit_report",
        ".html",
        fallback=output_namespace,
    )
    pdf_path = str(Path(output_path).with_suffix(".pdf"))

    supplied_runtime_context = runtime_context or {}
    supplied_runtime_context.setdefault(
        "cbep_version",
        (results.get("_cbep_trace") or {}).get("cbep_version", "unavailable"),
    )
    internal_runtime_context = collect_reproducibility_metadata(
        audit_context=audit_context,
        governance_context=governance_context,
        runtime_context=supplied_runtime_context,
    )
    report_runtime_context = public_report_reproducibility_metadata(
        internal_runtime_context
    )
    report_runtime_context.update(
        {
            key: value
            for key, value in supplied_runtime_context.items()
            if key not in {"run_timestamp", "run_id"}
        }
    )
    report_runtime_context.update(
        {
            "html_report_path": output_path,
            "pdf_report_path": pdf_path,
            "pdf_generated": bool(generate_pdf),
        }
    )

    report_sections = _ordered_report_sections(results)
    report_sections.append(
        {
            "key": "_reproducibility",
            "label": "Reproducibility metadata",
            "value": report_runtime_context["reproducibility"],
        }
    )

    def render_html():
        report_model = build_report_model(
            results=results,
            audit_context=audit_context,
            governance_context=governance_context,
            resource_context=resource_context,
            provider_name=case_name,
            runtime_context=report_runtime_context,
        )
        for finding in report_model.get("evidence_findings", []):
            finding["embedded_plot"] = _image_data_uri(finding.get("plot"))
        return template.render(
            sections=report_sections,
            risk_level=risk_level,
            timestamp=report_runtime_context["run_timestamp"],
            provider_name=case_name,
            report=report_model,
        )

    def write_html():
        with open(output_path, "w", encoding="utf-8") as report_file:
            report_file.write(render_html())

    write_html()

    print("HTML report generated at:", output_path)
    if generate_pdf:
        try:
            resolved_pdf_path = export_html_to_pdf(output_path, pdf_path)
            print("PDF report generated at:", resolved_pdf_path)
        except PDFExportError as exc:
            # Correct the HTML artifact inventory when PDF export was unavailable.
            report_runtime_context["pdf_generated"] = False
            write_html()
            print(f"[ReportGenerator] PDF generation warning:\n{exc}")
    else:
        print("[ReportGenerator] PDF generation skipped by --no-pdf.")

    return output_path
