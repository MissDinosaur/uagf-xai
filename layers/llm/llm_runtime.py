"""
Shared helpers for LLM / Agentic evidence execution.

These utilities keep the LLM runners small and consistent when the loaded
artifact is metadata-only and cannot be executed locally.
"""

from __future__ import annotations

from typing import Any
import re

import numpy as np


DEFAULT_SKIP_REASON = (
    "LLM inference was skipped because the provided model artifact is "
    "metadata-only and contains no loadable weights."
)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _artifact_status(generator, resource_context=None) -> dict[str, Any]:
    context = _as_dict(resource_context)
    model_metadata = _as_dict(context.get("model_metadata"))

    status = context.get("model_status")
    is_loadable = context.get("model_is_loadable")
    reason = context.get("model_reason")

    if is_loadable is None and hasattr(generator, "is_loadable"):
        is_loadable = bool(getattr(generator, "is_loadable"))

    if status is None and hasattr(generator, "status"):
        status = getattr(generator, "status")

    if reason is None and hasattr(generator, "reason"):
        reason = getattr(generator, "reason")

    if is_loadable is None:
        is_loadable = True

    if status is None:
        status = "loaded" if is_loadable else "metadata_only"

    return {
        "status": status,
        "is_loadable": bool(is_loadable),
        "reason": reason or model_metadata.get("reason"),
        "resolved_path": context.get("resolved_path")
        or getattr(generator, "resolved_path", None)
        or model_metadata.get("resolved_artifact_path"),
        "model_type": context.get("model_type")
        or getattr(generator, "model_type", None)
        or model_metadata.get("model_type"),
        "model_framework": context.get("model_framework")
        or getattr(generator, "model_framework", None)
        or model_metadata.get("model_framework"),
        "model_entrypoint": context.get("model_entrypoint")
        or getattr(generator, "model_entrypoint", None)
        or model_metadata.get("model_entrypoint"),
        "loaded_metadata_files": context.get("loaded_metadata_files")
        or getattr(generator, "loaded_metadata_files", None)
        or model_metadata.get("loaded_metadata_files", []),
        "golden_set_uri": context.get("golden_set_uri"),
        "has_golden_set": context.get("golden_set") is not None,
        "has_model_metadata": bool(model_metadata),
    }


def llm_execution_is_loadable(generator, resource_context=None) -> bool:
    return bool(_artifact_status(generator, resource_context)["is_loadable"])


def build_llm_skip_result(
    method: str,
    generator=None,
    resource_context=None,
    reason: str | None = None,
) -> dict[str, Any]:
    artifact_status = _artifact_status(generator, resource_context)
    final_reason = reason or artifact_status["reason"] or DEFAULT_SKIP_REASON

    available_resources: list[str] = []
    if artifact_status["has_golden_set"]:
        available_resources.append("golden_set")
    if artifact_status["has_model_metadata"]:
        available_resources.append("model_metadata")

    return {
        "type": "llm_evidence",
        "method": method,
        "status": "skipped",
        "reason": final_reason,
        "available_resources": available_resources,
        "resource_details": {
            "golden_set_uri": artifact_status["golden_set_uri"],
            "model_status": artifact_status["status"],
            "model_is_loadable": artifact_status["is_loadable"],
            "resolved_path": artifact_status["resolved_path"],
            "model_type": artifact_status["model_type"],
            "model_framework": artifact_status["model_framework"],
            "model_entrypoint": artifact_status["model_entrypoint"],
            "loaded_metadata_files": artifact_status["loaded_metadata_files"],
        },
    }


def evaluation_embedding_model(resource_context=None):
    context = _as_dict(resource_context)
    model = context.get("evaluation_embedding_model")
    if model is None or not hasattr(model, "encode"):
        raise ValueError(
            "Execution-level LLM evidence requires a loaded S6 evaluation "
            "embedding model with an encode() interface."
        )
    return model


def generation_record(generator, prompt: str, **configuration) -> dict[str, Any]:
    """Generate one continuation and expose the shared audit contract."""
    if hasattr(generator, "generate_response_with_metadata"):
        record = dict(
            generator.generate_response_with_metadata(prompt, **configuration)
        )
    elif hasattr(generator, "generate_response"):
        record = {
            "text": generator.generate_response(prompt, **configuration),
            "continuation_only": True,
            "input_token_count": None,
            "generated_token_count": None,
            **configuration,
        }
    else:
        outputs = generator(
            prompt,
            num_return_sequences=1,
            return_full_text=False,
            **configuration,
        )
        if not outputs or "generated_text" not in outputs[0]:
            raise RuntimeError("The generation pipeline returned no generated_text.")
        record = {
            "text": outputs[0]["generated_text"],
            "continuation_only": True,
            "input_token_count": None,
            "generated_token_count": None,
            **configuration,
        }
    text = str(record.get("text") or "").strip()
    if not text:
        raise RuntimeError("The local generation model returned empty generated text.")
    if record.get("continuation_only") is not True:
        raise RuntimeError("The generation boundary did not guarantee continuation-only text.")
    record["text"] = text
    for key in ("max_new_tokens", "do_sample", "seed", "temperature", "top_p"):
        record.setdefault(key, configuration.get(key))
    return record


def generation_text(generator, prompt: str, **configuration) -> str:
    """Generate plain continuation text through the shared audit contract."""
    return generation_record(generator, prompt, **configuration)["text"]


def lexical_echo_ratio(generated: str, source: str) -> float:
    """Return the share of generated lexical tokens also present in the source."""
    generated_tokens = re.findall(r"\b[\w'-]+\b", generated.lower())
    source_tokens = set(re.findall(r"\b[\w'-]+\b", source.lower()))
    if not generated_tokens:
        return 0.0
    return sum(token in source_tokens for token in generated_tokens) / len(generated_tokens)


def cosine_similarity(left, right) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
