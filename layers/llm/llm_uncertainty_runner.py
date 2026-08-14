"""LLM-E2 semantic self-consistency under reproducible stochastic sampling."""

from itertools import combinations
import statistics

from .llm_runtime import (
    build_llm_skip_result,
    cosine_similarity,
    evaluation_embedding_model,
    generation_record,
    llm_execution_is_loadable,
)
from .llm_evidence_methods import LLM_E2_SELF_CONSISTENCY_SCORE


def run_llm_uncertainty(generator, llm_payload, resource_context=None):
    if not llm_execution_is_loadable(generator, resource_context):
        return build_llm_skip_result(
            LLM_E2_SELF_CONSISTENCY_SCORE,
            generator=generator,
            resource_context=resource_context,
        )

    prompts = list(
        llm_payload.get("self_consistency_prompts")
        or llm_payload.get("current_prompts", [])[:1]
    )
    per_prompt = []
    seeds = [42, 43, 44, 45, 46]
    configuration = {
        "max_new_tokens": 24,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.9,
    }
    embedding_model = evaluation_embedding_model(resource_context)

    for prompt in prompts:
        generation_prompt = f"Prompt: {prompt}\nResponse:"
        generation_records = [
            generation_record(
                generator, generation_prompt, seed=seed, **configuration
            )
            for seed in seeds
        ]
        outputs = [record["text"] for record in generation_records]
        embeddings = embedding_model.encode(outputs)
        similarities = [
            cosine_similarity(embeddings[left], embeddings[right])
            for left, right in combinations(range(len(outputs)), 2)
        ]
        per_prompt.append(
            {
                "prompt": prompt,
                "generation_prompt": generation_prompt,
                "outputs": outputs,
                "generation_metadata": generation_records,
                "pairwise_similarities": similarities,
                "pairwise_comparison_count": len(similarities),
                "consistency": statistics.fmean(similarities),
            }
        )

    if not per_prompt:
        raise ValueError("LLM-E2 requires at least one selected prompt.")
    all_similarities = [
        score for item in per_prompt for score in item["pairwise_similarities"]
    ]
    aggregate = statistics.fmean(all_similarities)

    return {
        "type": "llm_evidence",
        "method": LLM_E2_SELF_CONSISTENCY_SCORE,
        "status": "completed",
        "continuation_only": True,
        "prompt_count": len(per_prompt),
        "outputs_per_prompt": 5,
        "seeds": seeds,
        "temperature": 0.8,
        "top_p": 0.9,
        "generation_configuration": configuration,
        "details": per_prompt,
        "aggregate_consistency": aggregate,
        "pairwise_similarity_standard_deviation": (
            statistics.pstdev(all_similarities)
        ),
        "evaluation_embedding_model": embedding_model.model_name,
        "key_findings": [
            f"Mean self-consistency = {aggregate:.4f} across {len(per_prompt)} "
            "prompts; lower values indicate less stable semantic outputs under "
            "stochastic sampling."
        ],
        "limitations": [
            "This metric measures semantic output stability, not calibrated predictive uncertainty."
        ],
    }
