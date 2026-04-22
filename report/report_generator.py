from jinja2 import Environment, FileSystemLoader
import os
from datetime import datetime

def generate_report(results, risk_level):

    env = Environment(loader=FileSystemLoader("report/templates"))
    template = env.get_template("report_template.html")

    os.makedirs("outputs/report", exist_ok=True)

    html_content = template.render(
        results=results,
        risk_level=risk_level,
        timestamp=str(datetime.now())
    )

    output_path = "outputs/report/audit_report.html"

    with open(output_path, "w") as f:
        f.write(html_content)

    return output_path

"""
pip install pdfkit
import pdfkit
pdfkit.from_file("audit_report.html", "audit_report.pdf")
"""