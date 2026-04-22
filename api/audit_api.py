from planner.cbep import select_methods
from pipeline.executor import execute
from report.report_generator import generate_report


def audit(model, X, y, risk_level, system_type="traditional"):

    methods = select_methods(risk_level, system_type=system_type)

    results = execute(model, X, y, methods, system_type=system_type)

    report_path = generate_report(results, risk_level)

    print("Report generated at:", report_path)

    return results
