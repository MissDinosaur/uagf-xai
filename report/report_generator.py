import json
from jinja2 import Environment, FileSystemLoader, select_autoescape
import os
from datetime import datetime
from output_naming import build_output_path
from report.report_builder import build_report_model
from layers.llm.evidence_methods import (
    LLM_METHOD_ORDER,
    llm_method_display_name,
)


def _to_json_pretty(value):
    """Jinja2 filter: convert any value to indented JSON string."""
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _section_label(key: str) -> str:
    if key == "_cbep_trace":
        return "CBEP planning trace"
    if key in LLM_METHOD_ORDER:
        return llm_method_display_name(key)

    pretty = key.replace("_", " ").strip()
    if not pretty:
        return key

    return pretty[:1].upper() + pretty[1:]


def _ordered_report_sections(results):
    preferred_order = [
        "_cbep_trace",
        "explainability",
        "lime",
        "fairness",
        "uncertainty",
        "drift",
        "counterfactual",
        *LLM_METHOD_ORDER,
    ]

    ordered = []
    seen = set()

    for key in preferred_order:
        if key in results:
            ordered.append({
                "key": key,
                "label": _section_label(key),
                "value": results[key],
            })
            seen.add(key)

    for key, value in results.items():
        if key in seen:
            continue
        ordered.append({
            "key": key,
            "label": _section_label(key),
            "value": value,
        })

    return ordered


def generate_report(
    results,
    risk_level,
    provider_name=None,
    output_namespace="audit",
    audit_context=None,
    governance_context=None,
    resource_context=None,
):

    env = Environment(
        loader=FileSystemLoader("report/templates"),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.filters["tojson_pretty"] = _to_json_pretty
    template = env.get_template("report_template.html")

    os.makedirs("outputs/report", exist_ok=True)

    case_name = provider_name or output_namespace or "audit"
    report_model = build_report_model(
        results=results,
        audit_context=audit_context,
        governance_context=governance_context,
        resource_context=resource_context,
        provider_name=case_name,
    )

    html_content = template.render(
        sections=_ordered_report_sections(results),
        risk_level=risk_level,
        timestamp=str(datetime.now()),
        provider_name=case_name,
        report=report_model,
    )

    output_path = build_output_path(
        "report",
        provider_name,
        "audit_report",
        ".html",
        fallback=output_namespace,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path

"""
pip install pdfkit
import pdfkit
pdfkit.from_file("outputs/report/<case>_audit_report.html", "outputs/report/<case>_audit_report.pdf")
"""
