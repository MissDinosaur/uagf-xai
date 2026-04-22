"""
LLM explainability evidence runner.
Builds prompt-response traces and lightweight token statistics.
"""


def run_llm_explainability(generator, llm_payload):
    prompts = llm_payload.get("current_prompts", [])
    max_prompts = min(3, len(prompts))

    traces = []
    for prompt in prompts[:max_prompts]:
        response = generator(
            prompt,
            max_new_tokens=40,
            do_sample=False,
            num_return_sequences=1,
        )[0]["generated_text"]

        if response.startswith(prompt):
            generated_part = response[len(prompt):].strip()
        else:
            generated_part = response
        traces.append(
            {
                "prompt": prompt,
                "generated_text": generated_part,
                "prompt_characters": len(prompt),
                "generated_characters": len(generated_part),
            }
        )

    return {
        "type": "llm_explainability",
        "method": "Prompt-Response Trace",
        "sample_count": len(traces),
        "traces": traces,
    }
