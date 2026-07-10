import json
from jinja2 import Environment, FileSystemLoader
import os
from datetime import datetime
from output_naming import build_output_path


def _to_json_pretty(value):
    """Jinja2 filter: convert any value to indented JSON string."""
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _section_label(key: str) -> str:
    if key == "_cbep_trace":
        return "CBEP planning trace"

    pretty = key.replace("_", " ").strip()
    if not pretty:
        return key

    return pretty[:1].upper() + pretty[1:]


def _ordered_report_sections(results):
    preferred_order = [
        "_cbep_trace",
        "explainability",
        "fairness",
        "uncertainty",
        "drift",
        "counterfactual",
        "llm_explainability",
        "llm_fairness",
        "llm_uncertainty",
        "llm_drift",
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


def _extract_cbep_final_plan(results):
    trace = results.get("_cbep_trace")
    if isinstance(trace, dict):
        final_plan = trace.get("final_plan")
        if isinstance(final_plan, list):
            return final_plan
    return []


def generate_report(results, risk_level, provider_name=None, output_namespace="audit"):

    env = Environment(loader=FileSystemLoader("report/templates"))
    env.filters["tojson_pretty"] = _to_json_pretty
    template = env.get_template("report_template.html")

    os.makedirs("outputs/report", exist_ok=True)

    case_name = provider_name or output_namespace or "audit"
    cbep_final_tools = _extract_cbep_final_plan(results)

    html_content = template.render(
        sections=_ordered_report_sections(results),
        risk_level=risk_level,
        timestamp=str(datetime.now()),
        provider_name=case_name,
        cbep_final_tools=cbep_final_tools,
    )

    output_path = build_output_path(
        "report",
        provider_name,
        "audit_report",
        ".html",
        fallback=output_namespace,
    )

    with open(output_path, "w") as f:
        f.write(html_content)

    return output_path

"""
pip install pdfkit
import pdfkit
pdfkit.from_file("outputs/report/<case>_audit_report.html", "outputs/report/<case>_audit_report.pdf")
"""
