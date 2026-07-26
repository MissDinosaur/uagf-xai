Given the governance context, audit requirements, trained model, and evaluation dataset, determine the minimum sufficient evidence set and execute the corresponding evidence generation pipeline.

For the LegalMindD LLM/agentic validation case, the S5 artifact provided in the repository contains model configuration and adapter metadata but no loadable local model weights. Therefore, UAGF-XAI validates the LLM golden-set resource contract and metadata loading pathway, while execution-level LLM evidence generation is reported as skipped. A complete local HuggingFace model artifact would be required to run full LLM-E1 to LLM-E4 evidence generation.

# uagf-xai
UAGF-XAI is the S6 evidence generation component of the UAGF platform. It provides a CBEP-driven toolkit that selects and generates minimum sufficient audit evidence for both traditional ML and LLM/agentic systems under selected EU AI Act requirements.

UAGF-XAI is an automated AI audit evidence generator that integrates explainability, fairness, uncertainty and drift detection tools into a unified pipeline guided by a constraint-based evidence planner.

The proposed system architecture integrates multiple AI auditing techniques into a unified pipeline. The system receives an AI model and its associated dataset as input. A Constraint-Based Evidence Planner (CBEP) determines which analytical methods should be executed depending on the AI system risk level.

The architecture is composed of four analytical layers: explainability, fairness, uncertainty estimation, and drift detection. Each layer applies specialized techniques using established Python libraries. The outputs of these layers are aggregated into structured evidence reports that can support AI auditing and regulatory transparency.

This project now supports two system families:
- Traditional ML systems: SHAP + Fairlearn + MAPIE + Evidently (+ DiCE for high risk).
- LLM or Agentic systems: dedicated LLM evidence tools for explainability, fairness, uncertainty, and drift.

Validation domains now include a 5th domain for locally runnable open-source LLMs via HuggingFace (default GPT-2).

Removed from active design:
- Aequitas (Fairlearn is retained as the fairness toolkit).
- MC Dropout (MAPIE is retained for uncertainty).

## Project Architecture
```text
uagf-xai
│
├── api/
│   └── audit_api.py
│
├── planner/
│   └── cbep.py
│
├── layers/
│   ├── explainability/
│   │   ├── shap_runner.py
│   │   ├── lime_runner.py
│   │   └── dice_runner.py
│   ├── fairness/
│   │   └── fairlearn_runner.py
│   ├── uncertainty/
│   │   └── mapie_runner.py
│   ├── drift/
│   │   └── evidently_runner.py
│   └── llm/
│       ├── llm_explainability_runner.py
│       ├── llm_fairness_runner.py
│       ├── llm_uncertainty_runner.py
│       └── llm_drift_runner.py
│
├── models/
│   └── llm_validation.py
│
├── pipeline/
│   └── executor.py
│
├── schema/
│   └── evidence_schema.py
│
├── report/
│   ├── templates/
│   └── report_generator.py
│
├── data/
│
├── tests/
│
├── main.py
├── README.md
└── requirements.txt
```

### Quick Execution Flow
```text
main.py -> audit_api -> planner (CBEP) -> executor -> layers -> report
```

### System Design
- unified pipeline
- method registry
- evidence schema
- report generator
- CBEP planner

### Project Workflow
```text
                        ┌────────────────────────────────────────┐
                        │ User / API Input                       │
                        │ model + dataset + risk + system_type   │
                        └───────────────────┬────────────────────┘
                                            │
                                            ▼
                        ┌────────────────────────────────────────┐
                        │ UAGF-XAI API                           │
                        │ audit(model, X, y, risk_level,         │
                        │      system_type)                      │
                        └───────────────────┬────────────────────┘
                                            │
                                            ▼
                        ┌────────────────────────────────────────┐
                        │ CBEP Evidence Planner                  │
                        │ selects method set by:                 │
                        │ - risk_level                           │
                        │ - system_type                          │
                        └───────────────────┬────────────────────┘
                                            │
                                            ▼
                        ┌────────────────────────────────────────┐
                        │ Execution Pipeline (executor.py)       │
                        └───────────────────┬────────────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     │                                             │
                     ▼                                             ▼
      ┌───────────────────────────────────┐       ┌───────────────────────────────────┐
      │ if system_type in {llm, agentic}  │       │ else traditional ML system        │
      ├───────────────────────────────────┤       ├───────────────────────────────────┤
      │ LLM Layer 1: Explainability       │       │ Layer 1: Explainability (SHAP,    │
      │ LLM Layer 2: Fairness             │       │          LIME, DiCE)              │
      │ LLM Layer 3: Uncertainty          │       │ Layer 2: Fairness (Fairlearn)     │
      │ LLM Layer 4: Drift                │       │ Layer 3: Uncertainty (MAPIE)      │
      │                                   │       │ Layer 4: Drift (Evidently)        │
      └─────────────────────┬─────────────┘       └─────────────────────┬─────────────┘
                            │                                           │
                            └─────────────────────┬─────────────────────┘
                                                  │
                                                  ▼
                               ┌─────────────────────────────────┐
                               │ Unified Evidence Schema / Dict  │
                               └─────────────────┬───────────────┘
                                                 │
                                                 ▼
                               ┌─────────────────────────────────┐
                               │ Report Generator                │
                               │ HTML report output              │
                               └─────────────────┬───────────────┘
                                                 │
                                                 ▼
                               ┌─────────────────────────────────┐
                               │ AI Audit Report                 │
                               └─────────────────────────────────┘
```

### Dataset
| Case          | Domain         | System         |
| ------------- | -------------- | -------------- |
| German Credit | Finance        | Traditional ML |
| Energy        | Energy         | Traditional ML |
| M5            | Retail         | Traditional ML |
| IMDb          | NLP Classifier | Traditional ML |
| GPT2/Mistral  | LLM            | LLM            |


### Planner Workflow
```text
                 ┌──────────────────────────┐
                 │      Input to CBEP       │
                 │                          │
                 │  AI Model                │
                 │  Dataset                 │
                 │  system_type             │
                 │  Risk Tier (Minimal /    │
                 │             Limited /    │
                 │             High Risk)   │
                 └──────────────┬───────────┘
                                │
                                ▼
                ┌────────────────────────────────┐
                │  EU AI Act Constraint Rules    │
                │                                │
                │ Example rules:                 │
                │                                │
                │ IF risk = minimal              │
                │    require explainability      │
                │                                │
                │ IF risk = limited              │
                │    require explainability      │
                │    require fairness            │
                │                                │
                │ IF risk = high                 │
                │    require explainability      │
                │    require fairness            │
                │    require uncertainty         │
                │    require drift detection     │
                │                                │
                │ IF system_type in {llm,agentic}│
                │    choose llm_* method family  │
                │ ELSE                           │
                │    choose traditional methods  │
                └───────────────┬────────────────┘
                                │
                                ▼
             ┌────────────────────────────────────┐
             │  CSP Problem Construction          │
             │                                    │
             │ Variables:                         │
             │                                    │
             │ traditional variables              │
             │   shap, lime, dice, fairlearn,     │
             │   mapie, drift                     │
             │ llm variables                      │
             │   llm_explainability, llm_fairness,│
             │   llm_uncertainty, llm_drift       │
             │                                    │
             │ Constraints:                       │
             │ must satisfy EU AI Act evidence    │
             └───────────────┬────────────────────┘
                             │
                             ▼
                ┌─────────────────────────────┐
                │     CSP Solver              │
                │ (python-constraint)         │
                │                             │
                │ Goal:                       │
                │ Select MINIMUM set of       │
                │ evidence methods satisfying │
                │ all constraints             │
                └───────────────┬─────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │     Selected Evidence Set     │
                │                               │
                │ Example Outputs               │
                │                               │
                │ Traditional + Minimal →       │
                │   [shap]                      │
                │                               │
                │ Traditional + Limited →       │
                │   [shap, fairness]            │
                │                               │
                │ Traditional + High →          │
                │   [shap, fairness, uncertainty│
                │    drift, dice]               │
                │                               │
                │ LLM/Agentic + High →          │
                │   [llm_explainability,        │
                │    llm_fairness,              │
                │    llm_uncertainty, llm_drift]│
                └───────────────┬───────────────┘
                                │
                                ▼
                  ┌────────────────────────────┐
                  │   Evidence Execution       │
                  │                            │
                  │ Pipeline runs selected     │
                  │ XAI tools                  │
                  └───────────────┬────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   AI Audit Evidence     │
                    │                         │
                    │ Explainability          │
                    │ Fairness                │
                    │ Uncertainty             │
                    │ Drift                   │
                    └─────────────────────────┘
```

### Validation Domains
- Domain 1: German Credit (classification)
- Domain 2: Energy (regression)
- Domain 3: M5-style forecasting (time series)
- Domain 4: IMDB-style text baseline
- Domain 5: Local open-source LLM via HuggingFace (GPT-2 by default, Mistral model IDs are optional if hardware allows)

### CLI Usage
- Traditional ML run:
        python main.py --system-type traditional --risk-level high
- LLM run (default model GPT-2):
        python main.py --system-type llm --risk-level high --llm-model gpt2
- Agentic run:
        python main.py --system-type agentic --risk-level limited --llm-model gpt2

