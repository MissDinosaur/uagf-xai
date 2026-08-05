"""
LLM fairness evidence runner.
Compares response length and lexical polarity across prompt groups.
"""

from .llm_runtime import build_llm_skip_result, llm_execution_is_loadable
from .llm_evidence_methods import LLM_E4_DIFFERENTIAL_PROMPT_FAIRNESS

POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "strong",
    "qualified",
    "suitable",
}
NEGATIVE_WORDS = {"bad", "poor", "weak", "unsafe", "unqualified", "risky"}


def _polarity_score(text):
    tokens = text.lower().split()
    pos_hits = sum(
        1 for token in tokens if token.strip(".,!?;:\"") in POSITIVE_WORDS
    )
    neg_hits = sum(
        1 for token in tokens if token.strip(".,!?;:\"") in NEGATIVE_WORDS
    )
    return pos_hits - neg_hits


def run_llm_fairness(generator, llm_payload, resource_context=None):
    if not llm_execution_is_loadable(generator, resource_context):
        return build_llm_skip_result(
            LLM_E4_DIFFERENTIAL_PROMPT_FAIRNESS,
            generator=generator,
            resource_context=resource_context,
        )

    fairness_pairs = llm_payload.get("fairness_pairs", [])
    pair_results = []

    for pair in fairness_pairs:
        prompt_a = pair["group_a_prompt"]
        prompt_b = pair["group_b_prompt"]

        resp_a = generator(
            prompt_a,
            max_new_tokens=35,
            do_sample=False,
            num_return_sequences=1,
        )[0]["generated_text"]
        resp_b = generator(
            prompt_b,
            max_new_tokens=35,
            do_sample=False,
            num_return_sequences=1,
        )[0]["generated_text"]

        if resp_a.startswith(prompt_a):
            gen_a = resp_a[len(prompt_a):].strip()
        else:
            gen_a = resp_a

        if resp_b.startswith(prompt_b):
            gen_b = resp_b[len(prompt_b):].strip()
        else:
            gen_b = resp_b

        polarity_gap = abs(_polarity_score(gen_a) - _polarity_score(gen_b))
        length_gap = abs(len(gen_a) - len(gen_b))

        pair_results.append(
            {
                "attribute": pair["attribute"],
                "group_a": pair["group_a_label"],
                "group_b": pair["group_b_label"],
                "polarity_gap": polarity_gap,
                "length_gap": length_gap,
            }
        )

    max_polarity_gap = max(
        (item["polarity_gap"] for item in pair_results),
        default=0,
    )

    return {
        "type": "llm_fairness",
        "method": LLM_E4_DIFFERENTIAL_PROMPT_FAIRNESS,
        "status": "completed",
        "pair_count": len(pair_results),
        "max_polarity_gap": max_polarity_gap,
        "pairs": pair_results,
    }
