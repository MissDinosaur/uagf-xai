# uagf-xai


## Project Architecture
```text
uagf-xai
│
├── api/
│   └── audit_api.py 
│
├── planner/
│   └── cbep.py
|
├── layers/
│   ├── explainability/
│   │   ├── shap_runner.py
│   │   ├── lime_runner.py
│   │   └── dice_runner.py
│   ├── fairness/
│   │   ├── fairlearn_runner.py
│   │   └── aequitas_runner.py
│   ├── uncertainty/
│   │   ├── mapie_runner.py
│   │   └── mapie_runner.py
│   └── drift/
│       └── evidently_runner.py
|
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
|
├── tests/
│
├── main.py
├── README.md
└── requirements.txt
```