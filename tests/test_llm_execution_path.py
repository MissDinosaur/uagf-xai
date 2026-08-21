from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from layers.llm.llm_drift_runner import run_llm_drift
from layers.llm.llm_explainability_runner import run_llm_explainability
from layers.llm.llm_fairness_runner import run_llm_fairness
from layers.llm.llm_uncertainty_runner import run_llm_uncertainty
from resources.llm_model_loader import LLMModelLoader, LocalCausalLMArtifact
from resources.llm_evaluation_loader import LLMEvaluationModelLoader
from resources.local_llm_validation_config import LocalLLMValidationConfig


class _FakeEmbeddingModel:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            value = sum(ord(char) for char in str(text))
            vectors.append([1.0, float(value % 97 + 1), float(len(str(text)) + 1)])
        values = np.asarray(vectors, dtype=float)
        return values / np.linalg.norm(values, axis=1, keepdims=True)


class _FakeGenerator:
    status = "loaded"
    is_loadable = True

    def __init__(self):
        self.calls = []

    def generate_response_with_metadata(self, prompt, **configuration):
        self.calls.append((prompt, dict(configuration)))
        seed = configuration.get("seed", "deterministic")
        return {
            "text": f"continuation {seed}",
            "continuation_only": True,
            "input_token_count": 7,
            "generated_token_count": 2,
            "max_new_tokens": configuration.get("max_new_tokens"),
            "do_sample": configuration.get("do_sample"),
            "seed": configuration.get("seed"),
            "temperature": configuration.get("temperature"),
            "top_p": configuration.get("top_p"),
        }

    def generate_response(self, prompt, **configuration):
        raise AssertionError("Runners must use the shared structured generation contract.")


@pytest.fixture
def embedding_context():
    return {"evaluation_embedding_model": _FakeEmbeddingModel()}


def _fairness_context(embedding_context, pair):
    return {
        **embedding_context,
        "fairness_prompt_pairs": {
            "provenance": "synthetic_matched_prompt_pairs",
            "pairs": [pair],
        },
    }


def test_s6_local_config_and_cli_mode_are_separate_from_s5(monkeypatch):
    import main

    path = Path(
        "data/04b_local_distilgpt2_llm/s6_local_llm_validation_config.json"
    )
    local_case = LocalLLMValidationConfig.load(path)
    context = local_case.audit_context
    evidence_config = local_case.llm_evidence_config

    assert context.system_type == "llm"
    assert context.task_type == "llm_generation"
    assert context.application_domain == "justice"
    assert evidence_config.evaluation_embedding_model.endswith("all-MiniLM-L6-v2")
    assert evidence_config.semantic_drift_dataset_uri.endswith(
        "semantic_drift_validation.json"
    )
    assert evidence_config.fairness_prompt_pairs_uri.endswith(
        "fairness_prompt_pairs.json"
    )
    monkeypatch.setattr("sys.argv", ["main.py"])
    assert main.parse_args().mode == "s5"

    monkeypatch.setattr("sys.argv", ["main.py", "--mode", "local"])
    args = main.parse_args()
    assert args.mode == "local"
    assert args.local_config.endswith("s6_local_llm_validation_config.json")


def test_local_model_and_embedding_loaders_use_offline_transformers(
    monkeypatch,
    tmp_path,
):
    generator_path = tmp_path / "distilgpt2"
    embedding_path = tmp_path / "all-MiniLM-L6-v2"
    generator_path.mkdir()
    embedding_path.mkdir()
    (generator_path / "model.safetensors").write_bytes(b"weights")
    calls = []

    class FakeTokenizer:
        pad_token_id = None
        eos_token_id = 7
        eos_token = "<eos>"

    class Config:
        pad_token_id = None

    class FakeModel:
        config = Config()
        generation_config = Config()

        def to(self, device):
            return self

        def eval(self):
            self.eval_called = True
            return self

    def load_tokenizer(path, **kwargs):
        calls.append(("tokenizer", kwargs))
        return FakeTokenizer()

    def load_model(path, **kwargs):
        calls.append(("model", kwargs))
        return FakeModel()

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", load_tokenizer)
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", load_model
    )
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", load_model)

    artifact = LLMModelLoader.load(
        generator_path,
        model_format="huggingface_pretrained",
        model_framework="huggingface_transformers",
        model_type="distilgpt2_local_surrogate",
        model_entrypoint="distilgpt2",
    )
    context = SimpleNamespace(
        evaluation_embedding_model_uri=f"file://{embedding_path}",
        evaluation_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    instrument = LLMEvaluationModelLoader.load(context)

    assert isinstance(artifact, LocalCausalLMArtifact)
    assert artifact.is_loadable is True and artifact.status == "loaded"
    assert artifact.tokenizer.pad_token_id == artifact.tokenizer.eos_token_id
    assert artifact.model.eval_called is True
    assert instrument.model_name.endswith("all-MiniLM-L6-v2")
    assert all(kwargs["local_files_only"] is True for _, kwargs in calls)


def test_generation_wrapper_delegates_to_original_causal_model_generate():
    import torch

    class Tokenizer:
        pad_token_id = 0

        def __init__(self):
            self.decoded_ids = None

        def __call__(self, prompt, **kwargs):
            return {
                "input_ids": torch.tensor([[10, 20, 30]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

        def decode(self, values, **kwargs):
            self.decoded_ids = values.tolist()
            return "continuation D E" if self.decoded_ids == [40, 50] else "bad"

    class Model:
        device = torch.device("cpu")

        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return torch.tensor([[10, 20, 30, 40, 50]])

    model = Model()
    tokenizer = Tokenizer()
    artifact = LocalCausalLMArtifact(
        model=model,
        tokenizer=tokenizer,
        resolved_path="local/model",
    )

    prompt = "UNIQUE_INPUT_MARKER_8F91A"
    record = artifact.generate_response_with_metadata(prompt, do_sample=False)
    output = artifact.generate_response(prompt, do_sample=False)

    assert output == "continuation D E"
    assert record["text"] == "continuation D E"
    assert "UNIQUE_INPUT_MARKER_8F91A" not in output
    assert tokenizer.decoded_ids == [40, 50]
    assert record["continuation_only"] is True
    assert record["input_token_count"] == 3
    assert record["generated_token_count"] == 2
    assert len(model.calls) == 2
    assert model.calls[0]["do_sample"] is False


def test_e1_generates_text_and_returns_numeric_grounding_scores(embedding_context):
    generator = _FakeGenerator()
    payload = {
        "grounding_records": [
            {
                "id": "one",
                "prompt": "What is oversight?",
                "context": "People monitor the system.",
                "reference_text": "Humans monitor and can intervene.",
            }
        ]
    }

    result = run_llm_explainability(generator, payload, embedding_context)

    assert result["status"] == "completed"
    assert result["records_evaluated"] == 1
    assert result["generated_responses"][0]
    assert result["continuation_only"] is True
    assert result["per_record"][0]["generation_metadata"]["continuation_only"]
    assert embedding_context["evaluation_embedding_model"].calls[0][0] == (
        "continuation deterministic"
    )
    assert "What is oversight?" not in embedding_context[
        "evaluation_embedding_model"
    ].calls[0][0]
    assert isinstance(result["mean"], float)
    assert generator.calls[0][1]["do_sample"] is False


def test_e2_uses_all_prompts_with_five_generations_and_ten_pairs(embedding_context):
    generator = _FakeGenerator()
    result = run_llm_uncertainty(
        generator,
        {
            "self_consistency_prompts": [
                "Explain oversight.",
                "Explain transparency.",
                "Explain risk management.",
            ]
        },
        embedding_context,
    )

    assert result["status"] == "completed"
    assert result["prompt_count"] == 3
    assert result["outputs_per_prompt"] == 5
    assert result["seeds"] == [42, 43, 44, 45, 46]
    assert len(generator.calls) == 15
    assert all(len({text for text in item["outputs"]}) == 5 for item in result["details"])
    assert all(item["pairwise_comparison_count"] == 10 for item in result["details"])
    assert all(len(item["pairwise_similarities"]) == 10 for item in result["details"])


def test_e3_requires_and_reports_controlled_baseline_current_sets(embedding_context):
    context = {
        **embedding_context,
        "semantic_drift_dataset": {
            "provenance": "controlled_semantic_drift_validation",
            "baseline_prompts": ["baseline one", "baseline two"],
            "current_prompts": ["current one", "current two"],
        },
    }
    result = run_llm_drift(_FakeGenerator(), {}, context)

    assert result["status"] == "completed"
    assert result["validation_label"] == "controlled semantic-drift validation"
    assert result["baseline_sample_count"] == 2
    assert result["current_sample_count"] == 2
    assert isinstance(result["semantic_drift_index"], float)
    assert result["centroid_semantic_drift_index"] == result["semantic_drift_index"]
    assert len(result["per_pair_semantic_distance"]) == 2
    assert isinstance(result["mean_pairwise_semantic_distance"], float)
    assert all(result["baseline_outputs"])
    assert all(result["current_outputs"])

    invalid_context = {
        **embedding_context,
        "semantic_drift_dataset": {
            "provenance": "controlled_semantic_drift_validation",
            "baseline_prompts": ["baseline one", "baseline two"],
            "current_prompts": ["current one"],
        },
    }

    with pytest.raises(ValueError, match="equal paired baseline"):
        run_llm_drift(_FakeGenerator(), {}, invalid_context)


def test_e4_only_uses_explicit_matched_pairs_and_reports_numeric_difference(
    embedding_context,
):
    pair = {
        "pair_id": "p1",
        "attribute": "explicit_attribute",
        "group_a_label": "A",
        "group_b_label": "B",
        "group_a_prompt": "Person A requests neutral review.",
        "group_b_prompt": "Person B requests neutral review.",
    }
    context = _fairness_context(embedding_context, pair)
    result = run_llm_fairness(
        _FakeGenerator(),
        {"arbitrary_column_name": "must not be inferred"},
        context,
    )

    assert result["status"] == "completed"
    assert result["controlled_attributes"] == ["explicit_attribute"]
    assert result["prompt_pair_count"] == 1
    assert isinstance(result["pairs"][0]["output_difference"], float)
    assert result["continuation_only"] is True
    assert isinstance(result["aggregate_output_difference"], float)
    assert result["metric_direction"] == "lower_means_less_output_sensitivity"
    embedded = embedding_context["evaluation_embedding_model"].calls[0]
    assert embedded == ["continuation deterministic", "continuation deterministic"]
    assert all("Person" not in output for output in embedded)
    assert "unlawful discrimination" in result["limitations"][0]

    invalid_pair = {
        "pair_id": "invalid",
        "attribute": "gender",
        "group_a_label": "woman",
        "group_b_label": "man",
        "group_a_prompt": "A woman requests a neutral review.",
        "group_b_prompt": "A man requests an expedited review.",
    }
    with pytest.raises(ValueError, match="only by the declared controlled"):
        run_llm_fairness(
            _FakeGenerator(),
            {},
            _fairness_context(embedding_context, invalid_pair),
        )

    class EchoGenerator:
        is_loadable = True

        def generate_response_with_metadata(self, prompt, **configuration):
            text = prompt.removeprefix("Prompt: ").removesuffix("\nResponse:")
            return {
                "text": text,
                "continuation_only": True,
                "input_token_count": 10,
                "generated_token_count": 10,
                **configuration,
            }

    echo_pair = {
        **invalid_pair,
        "pair_id": "echo",
        "group_b_prompt": "A man requests a neutral review.",
    }
    echo_result = run_llm_fairness(
        EchoGenerator(),
        {},
        _fairness_context(embedding_context, echo_pair),
    )

    assert echo_result["status"] == "completed"
    assert echo_result["evidence_quality"] == "limited_by_degenerate_generation"
    assert "should not be interpreted as substantive fairness evidence" in (
        echo_result["key_findings"][0]
    )
    assert "degenerate generation behavior" in echo_result["limitations"][1]


def test_local_fallback_log_does_not_claim_real_s5_resources(monkeypatch, capsys):
    import api.audit_api as audit_api

    context = SimpleNamespace(
        system_type="llm",
        task_type="llm_generation",
        sensitive_feature_columns=[],
    )
    bundle = SimpleNamespace(model=object())
    monkeypatch.setattr(
        audit_api,
        "_load_s5_resources",
        lambda value, evidence_config=None: (bundle, {}),
    )
    monkeypatch.setattr(audit_api, "audit_with_detailed_data", lambda **kwargs: {})

    audit_api.audit(context, generate_pdf=False, runtime_mode="local")

    output = capsys.readouterr().out
    assert "Mode: S6 local execution fallback" in output
    assert "Mode: real S5 resources" not in output
