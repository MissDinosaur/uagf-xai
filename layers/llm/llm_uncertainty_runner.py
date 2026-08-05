"""
LLM uncertainty evidence runner.
Uses multi-sample generation consistency as an uncertainty proxy.
"""

from .llm_runtime import build_llm_skip_result, llm_execution_is_loadable
from .llm_evidence_methods import LLM_E2_SELF_CONSISTENCY_SCORE


def run_llm_uncertainty(generator, llm_payload, resource_context=None):
    if not llm_execution_is_loadable(generator, resource_context):
        return build_llm_skip_result(
            LLM_E2_SELF_CONSISTENCY_SCORE,
            generator=generator,
            resource_context=resource_context,
        )

    prompts = llm_payload.get("current_prompts", [])
    max_prompts = min(3, len(prompts))
    per_prompt = []

    for prompt in prompts[:max_prompts]:
        outputs = generator(
            prompt,
            max_new_tokens=30,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            num_return_sequences=3,
        )

        generated = []
        for output in outputs:
            text = output["generated_text"]
            if text.startswith(prompt):
                trimmed = text[len(prompt):].strip()
            else:
                trimmed = text
            generated.append(trimmed)

        unique_ratio = len(set(generated)) / max(1, len(generated))
        per_prompt.append(
            {
                "prompt": prompt,
                "sample_count": len(generated),
                "unique_ratio": unique_ratio,
            }
        )

    avg_unique_ratio = sum(
        item["unique_ratio"] for item in per_prompt
    ) / max(1, len(per_prompt))

    return {
        "type": "llm_uncertainty",
        "method": LLM_E2_SELF_CONSISTENCY_SCORE,
        "status": "completed",
        "average_unique_ratio": avg_unique_ratio,
        "details": per_prompt,
    }
