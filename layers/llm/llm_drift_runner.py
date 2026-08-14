"""LLM-E3 controlled semantic-drift execution validation."""

import numpy as np

from .llm_evidence_methods import LLM_E3_SEMANTIC_DRIFT_INDEX
from .llm_runtime import (
    build_llm_skip_result,
    cosine_similarity,
    evaluation_embedding_model,
    generation_record,
    llm_execution_is_loadable,
)


def run_llm_drift(generator, llm_payload, resource_context=None):
    if not llm_execution_is_loadable(generator, resource_context):
        return build_llm_skip_result(
            LLM_E3_SEMANTIC_DRIFT_INDEX,
            generator=generator,
            resource_context=resource_context,
        )

    context = resource_context or {}
    dataset = context.get("semantic_drift_dataset") or {}
    baseline_prompts = list(dataset.get("baseline_prompts") or [])
    current_prompts = list(dataset.get("current_prompts") or [])
    if not baseline_prompts or not current_prompts:
        raise ValueError(
            "LLM-E3 requires explicit baseline_prompts and current_prompts."
        )
    if len(baseline_prompts) != len(current_prompts):
        raise ValueError(
            "LLM-E3 controlled validation requires equal paired baseline and "
            "current prompt counts."
        )

    provenance = dataset.get("provenance")
    if provenance != "controlled_semantic_drift_validation":
        raise ValueError(
            "LLM-E3 local fallback data must declare controlled validation provenance."
        )

    configuration = {"max_new_tokens": 24, "do_sample": False}
    baseline_generations = [
        generation_record(
            generator, f"Prompt: {prompt}\nResponse:", **configuration
        )
        for prompt in baseline_prompts
    ]
    current_generations = [
        generation_record(
            generator, f"Prompt: {prompt}\nResponse:", **configuration
        )
        for prompt in current_prompts
    ]
    baseline_outputs = [record["text"] for record in baseline_generations]
    current_outputs = [record["text"] for record in current_generations]
    embedding_model = evaluation_embedding_model(resource_context)
    baseline_vectors = embedding_model.encode(baseline_outputs)
    current_vectors = embedding_model.encode(current_outputs)
    baseline_centroid = np.mean(baseline_vectors, axis=0)
    current_centroid = np.mean(current_vectors, axis=0)
    centroid_similarity = cosine_similarity(baseline_centroid, current_centroid)
    drift_index = 1.0 - centroid_similarity
    paired_distances = [
        1.0 - cosine_similarity(baseline_vectors[index], current_vectors[index])
        for index in range(len(baseline_prompts))
    ]
    mean_paired_distance = float(np.mean(paired_distances))

    return {
        "type": "llm_evidence",
        "method": LLM_E3_SEMANTIC_DRIFT_INDEX,
        "status": "completed",
        "continuation_only": True,
        "validation_label": "controlled semantic-drift validation",
        "controlled_validation_provenance": provenance,
        "baseline_sample_count": len(baseline_prompts),
        "current_sample_count": len(current_prompts),
        "baseline_prompts": baseline_prompts,
        "current_prompts": current_prompts,
        "baseline_outputs": baseline_outputs,
        "current_outputs": current_outputs,
        "baseline_generation_metadata": baseline_generations,
        "current_generation_metadata": current_generations,
        "per_pair_semantic_distance": [
            {
                "pair_index": index + 1,
                "baseline_prompt": baseline_prompts[index],
                "current_prompt": current_prompts[index],
                "semantic_distance": distance,
            }
            for index, distance in enumerate(paired_distances)
        ],
        "mean_pairwise_semantic_distance": mean_paired_distance,
        "centroid_similarity": centroid_similarity,
        "centroid_distance": drift_index,
        "semantic_drift_index": drift_index,
        "centroid_semantic_drift_index": drift_index,
        "evaluation_embedding_model": embedding_model.model_name,
        "generation_configuration": configuration,
        "key_findings": [
            f"Controlled semantic drift index = {drift_index:.4f}; this is a "
            "synthetic validation result, not observed production drift."
        ],
        "limitations": [
            "This is controlled semantic-drift validation, not observed production drift.",
            "No legal-compliance threshold is inferred from the semantic drift index.",
        ],
    }
