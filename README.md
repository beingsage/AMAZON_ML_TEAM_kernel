# Amazon ML Challenge 2026 — Project Workspace

This workspace contains the challenge PDF and the provided dataset. The goal is to build a business entity resolution pipeline that matches records from Source 2 and Source 3 back to Source 1, using the training data and a validation split to tune the model before generating final test-set outputs.

## Repository layout

- `amazon_ml_challenge_2026.pdf` — official challenge brief
- `student_resource/` — extracted challenge starter package, including dataset and validator
- `docs/` — planning and execution notes for the solution

## Critical challenge facts

- Source 1 is the deduplicated reference source.
- Source 2 and Source 3 are noisy candidate sources.
- A Source 1 entity can have zero, one, or many matches from S2/S3.
- Output must be two TSV files in `output/`:
  - `matching_results.tsv`
  - `candidate_pairs.tsv`
- The validation script is:

```bash
cd student_resource
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

- Evaluation metric is F_0.5 macro over Source 1 entities.
- Precision matters more than recall. False merges are costly.

## Immediate next steps

1. Read the official PDF and challenge README.
2. Explore dataset structure and label formats.
3. Create a validation split from training data.
4. Build a candidate-generation baseline.
5. Train a matching model using name + address features.
6. Optimize the decision threshold on validation data.
7. Generate final predictions for test data.
8. Validate outputs and package the final submission.

## Recommended working structure

```text
AMAZON_ML/
├── README.md
├── docs/
│   ├── 01_problem_brief.md
│   ├── 02_execution_plan.md
│   ├── 03_modeling_strategy.md
│   └── 04_submission_checklist.md
├── student_resource/
│   ├── dataset/
│   ├── utils/
│   ├── README.md
│   └── Documentation_template.md
├── output/
└── code/
```

This README is the launch point. The planning docs in `docs/` explain the exact work flow and success criteria.
