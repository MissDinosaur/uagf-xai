"""LLM-E1 grounding evidence using generated text and semantic alignment."""

import statistics

from .llm_runtime import (
    build_llm_skip_result,
    cosine_similarity,
    evaluation_embedding_model,
    generation_record,
    lexical_echo_ratio,
    llm_execution_is_loadable,
)
from .llm_evidence_methods import LLM_E1_GROUNDING_SCORE


def run_llm_explainability(generator, llm_payload, resource_context=None):
    if not llm_execution_is_loadable(generator, resource_context):
        return build_llm_skip_result(
            LLM_E1_GROUNDING_SCORE,
            generator=generator,
            resource_context=resource_context,
        )

    embedding_model = evaluation_embedding_model(resource_context)
    records = list(llm_payload.get("grounding_records") or [])
    configuration = {"max_new_tokens": 32, "do_sample": False}
    evaluated = []
    for record in records:
        prompt = (
            f"Context: {record.get('context', '')}\n"
            f"Question: {record.get('prompt', '')}\nAnswer:"
        )
        generation = generation_record(generator, prompt, **configuration)
        generated = generation["text"]
        reference = str(record.get("reference_text") or record.get("context") or "")
        embeddings = embedding_model.encode([generated, reference])
        score = cosine_similarity(embeddings[0], embeddings[1])
        echo_ratio = lexical_echo_ratio(
            generated,
            f"{record.get('context', '')} {record.get('prompt', '')}",
        )
        evaluated.append(
            {
                "record_id": record.get("id"),
                "prompt": record.get("prompt"),
                "context": record.get("context"),
                "reference_text": reference,
                "generated_response": generated,
                "grounding_score": score,
                "continuation_only": generation["continuation_only"],
                "generation_metadata": generation,
                "lexical_echo_ratio": echo_ratio,
                "strong_context_echo": echo_ratio >= 0.6,
            }
        )

    if not evaluated:
        raise ValueError("LLM-E1 requires eligible grounding records in the golden set.")
    scores = [item["grounding_score"] for item in evaluated]
    mean_score = statistics.fmean(scores)
    limitations = [
        "Cosine similarity is a semantic alignment score, not a calibrated probability.",
        "A high grounding score does not prove legal or factual correctness.",
    ]
    if any(item["strong_context_echo"] for item in evaluated):
        limitations.append(
            "The surrogate model repeated substantial wording from the supplied "
            "context; semantic alignment should therefore not be interpreted as "
            "independent factual verification."
        )

    return {
        "type": "llm_evidence",
        "method": LLM_E1_GROUNDING_SCORE,
        "status": "completed",
        "continuation_only": True,
        "records_evaluated": len(evaluated),
        "generated_responses": [item["generated_response"] for item in evaluated],
        "per_record": evaluated,
        "mean": mean_score,
        "median": statistics.median(scores),
        "min": min(scores),
        "max": max(scores),
        "evaluation_embedding_model": embedding_model.model_name,
        "generation_configuration": configuration,
        "key_findings": [
            f"Mean semantic grounding alignment = {mean_score:.4f} across "
            f"{len(evaluated)} records."
        ],
        "limitations": limitations,
    }
