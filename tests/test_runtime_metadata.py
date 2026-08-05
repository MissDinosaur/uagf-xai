from __future__ import annotations

import re

from report.method_titles import report_method_title
from report.runtime_metadata import collect_runtime_environment, infer_record_count


def test_runtime_environment_contains_python_and_platform():
    metadata = collect_runtime_environment("2026-07-27T12:00:00+02:00")

    assert metadata["run_timestamp"] == "2026-07-27T12:00:00+02:00"
    assert metadata["python_version"]
    assert metadata["platform"]
    assert "package_versions" not in metadata


def test_runtime_environment_formats_timestamp_and_python_concisely():
    metadata = collect_runtime_environment("2026-07-27T12:00:00.987654+02:00")

    assert metadata["run_timestamp"] == "2026-07-27T12:00:00+02:00"
    assert re.fullmatch(r"\d+\.\d+\.\d+ \[[^\]]+\]", metadata["python_version"])


def test_record_count_supports_frames_lists_and_nested_golden_sets():
    class FrameLike:
        shape = (17, 4)

    assert infer_record_count(FrameLike()) == 17
    assert infer_record_count([1, 2, 3]) == 3
    assert infer_record_count({"metadata": {}, "test_cases": [1, 2, 3, 4]}) == 4
    assert infer_record_count({"metadata": {}}) is None


def test_report_titles_are_polished_without_changing_tokens():
    assert report_method_title("llm_grounding") == "LLM-E1 — Grounding Score"
    assert (
        report_method_title("shap")
        == "Explainability Evidence — SHAP Feature Attribution"
    )
    assert (
        report_method_title("dice")
        == "Explainability Evidence — DiCE Counterfactual Explanation"
    )
    assert report_method_title("llm_grounding") != "llm_grounding"
