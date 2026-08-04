# UAGF-XAI

UAGF-XAI is the S6 audit-evidence component of the Unified AI Governance
Framework (UAGF). It consumes governance context from S4 and audit/resource
context from S5, selects a minimum sufficient evidence set, executes compatible
evidence methods, and produces structured HTML/PDF audit reports.

## Implemented scope

The canonical method catalogue contains six traditional ML methods and four
LLM/agentic methods.

| Pathway | Layer | Implemented method |
| --- | --- | --- |
| Traditional ML | Explainability | SHAP feature attribution |
| Traditional ML | Explainability | LIME local explanation |
| Traditional ML | Explainability | DiCE counterfactual explanation |
| Traditional ML | Fairness | Fairlearn group fairness metrics |
| Traditional ML | Uncertainty | MAPIE conformal prediction |
| Traditional ML | Drift | Evidently plus feature-level drift tests |
| LLM / Agentic | LLM-E1 | Grounding Score |
| LLM / Agentic | LLM-E2 | Self-Consistency Score |
| LLM / Agentic | LLM-E3 | Semantic Drift Index |
| LLM / Agentic | LLM-E4 | Differential Prompt Fairness |

Aequitas, MC Dropout, automatic protected-attribute inference, text LIME,
attention visualization, and the obsolete “16 XAI methods” claim are not part
of the implemented scope.

## CBEP

The Constraint-Based Evidence Planner (CBEP) is inspired by constraint
satisfaction problem (CSP) theory and operationalizes evidence selection as a
deterministic, constraint-informed planning procedure. It applies explicit
regulatory, governance, system-type, and task-compatibility constraints to
select a minimum sufficient evidence set for each audit context.

CBEP does not use or claim to use a generic `python-constraint` solver. The
`csp_satisfied` field is an S5 audit-status input; when it is false, CBEP broadens
the evidence sweep. Every plan includes a structured trace covering:

1. the traditional or LLM/agentic branch;
2. the risk-tier base plan;
3. EU AI Act article mappings;
4. governance-priority adjustments;
5. the upstream CSP/status signal;
6. task-method compatibility screening;
7. the final method plan and per-method decisions.

Method metadata is centralized in `schema/method_catalog.py`. The planner,
evidence normalizer, ordering helpers, and report view model derive their method
identities, layers, article mappings, compatibility, and requirements from this
catalogue.

## Evidence semantics and boundaries

- Fairness evidence is generated only for sensitive features explicitly
  supplied by S5. UAGF-XAI does not infer legally sensitive attributes. Missing
  sensitive-feature metadata is reported as an evidence gap.
- MAPIE uses the exact fitted S5 estimator in prefit mode. Classification,
  regression, and forecasting use disjoint conformal-calibration and measurement
  subsets of the evaluation data; the audited model is not retrained.
- Drift evidence measures current data and feature-distribution drift. It does
  not currently claim label-based concept-drift detection.
- Anomaly detection is currently supported by SHAP and drift evidence. Methods
  requiring classification semantics or probability interfaces are reported as
  task-incompatible, not as planning failures.
- The LegalMindD fixture is a metadata-only HuggingFace artifact. Its golden-set
  resource contract is validated, while execution-level LLM-E1 to LLM-E4 results
  are reported as skipped because no local weights are available. UAGF-XAI does
  not download or fabricate a model.

## Architecture

```text
S4 JSON -> GovernanceAdapter -> GovernanceContext -----+
                                                       |
S5 JSON -> AuditAdapter -> AuditContext -> ResourceLoader
                                      |                |
                                      +-------> ResourceBundle
                                                       |
                                                       v
                     canonical method catalogue -> CBEP planner
                                                       |
                                                       v
                                               evidence executor
                                                       |
                                                       v
                                      unified EvidenceResult schema
                                                       |
                                                       v
                                             HTML/PDF audit report
```

Evidence layers receive resolved models and data from the execution boundary;
they do not parse S4/S5 JSON or load raw artifact URIs directly.

## Resource contracts

Traditional ML cases use a model file, sklearn model bundle, or model directory,
plus training/evaluation dataset URIs and a target column. LLM/agentic cases use
a model directory and golden-set URI, with optional prompt, RAG-manifest, and
guardrail resources. LLM cases do not require training/evaluation CSV files.

The normalized S5/S6 vocabulary is:

- `system_type`: `traditional_ml`, `llm`, or `agentic`;
- `modality`: `tabular`, `time_series`, `text`, `image`, `audio`,
  `multimodal`, or `unknown`.

The S5 `is_llm_or_agentic` flag is authoritative for selecting the system
family. When true, descriptive model metadata distinguishes `llm` from
`agentic`; when false, the adapter returns `traditional_ml`.

## Validation commands

Activate `venv310`, then run:

```bash
python main.py --s4-json data/01_finclear_gmbh/s4_finclear-creditguard-001.json --s5-json data/01_finclear_gmbh/s5_finclear_gmbh_audit_state.json
python main.py --s4-json data/02_retailiq_ag/s4_retailiq-demandpulse-001.json --s5-json data/02_retailiq_ag/s5_retailiq_ag_audit_state.json
python main.py --s4-json data/03_harbourlogistik_gmbh/s4_harbourlogistik-harboursense-001.json --s5-json data/03_harbourlogistik_gmbh/s5_harbourlogistik_gmbh_audit_state.json
python main.py --s4-json data/04_legalmindd_ai_ltd/s4_legalmindd-lexai-001.json --s5-json data/04_legalmindd_ai_ltd/s5_legalmindd_ai_ltd_audit_state.json
```

Add `--no-pdf` for faster HTML-only validation.

## Reports

HTML and PDF reports are written under `outputs/report/`. PDF export uses
Playwright Chromium. Install the browser once if required:

```bash
python -m playwright install chromium
```

The report distinguishes selected/executed, selected/skipped,
not-applicable, and compatibility-filtered methods through the Executive
Summary, CBEP planning table, Coverage Matrix, findings, runtime table, and raw
evidence appendix.

## Tests

Run the offline unit suite with:

```bash
python -m pytest -q
```

The real browser PDF smoke test is marked as integration coverage:

```bash
python -m pytest -m integration
```
