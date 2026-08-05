"""
LLM drift evidence runner.
Computes vocabulary divergence between reference and current prompt sets.
"""

from collections import Counter
import math

from .llm_runtime import build_llm_skip_result, llm_execution_is_loadable
from .llm_evidence_methods import LLM_E3_SEMANTIC_DRIFT_INDEX


def _token_distribution(texts):
    token_counts = Counter()
    total = 0
    for text in texts:
        for token in text.lower().split():
            normalized = token.strip(".,!?;:\"")
            if normalized:
                token_counts[normalized] += 1
                total += 1

    if total == 0:
        return {}

    return {token: count / total for token, count in token_counts.items()}


def _js_divergence(p_dist, q_dist):
    all_tokens = set(p_dist.keys()) | set(q_dist.keys())
    if not all_tokens:
        return 0.0

    def _kl(a, b):
        value = 0.0
        for token in all_tokens:
            pa = a.get(token, 1e-12)
            pb = b.get(token, 1e-12)
            value += pa * math.log(pa / pb)
        return value

    m = {
        token: 0.5 * (p_dist.get(token, 0.0) + q_dist.get(token, 0.0))
        for token in all_tokens
    }
    return 0.5 * _kl(p_dist, m) + 0.5 * _kl(q_dist, m)


def run_llm_drift(generator, llm_payload, resource_context=None):
    if not llm_execution_is_loadable(generator, resource_context):
        return build_llm_skip_result(
            LLM_E3_SEMANTIC_DRIFT_INDEX,
            generator=generator,
            resource_context=resource_context,
        )

    reference_prompts = llm_payload.get("reference_prompts", [])
    current_prompts = llm_payload.get("current_prompts", [])

    ref_dist = _token_distribution(reference_prompts)
    cur_dist = _token_distribution(current_prompts)
    js_score = _js_divergence(ref_dist, cur_dist)

    return {
        "type": "llm_drift",
        "method": LLM_E3_SEMANTIC_DRIFT_INDEX,
        "status": "completed",
        "reference_prompt_count": len(reference_prompts),
        "current_prompt_count": len(current_prompts),
        "js_divergence": js_score,
    }
