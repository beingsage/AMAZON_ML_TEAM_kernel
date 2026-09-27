# Amazon ML Challenge 2026 — Project Workspace

## Kaggle one-command run

After cloning this repository in a Kaggle notebook, run only:

```bash
python run.py
```

On Kaggle this invokes the three stages below automatically, in one process,
using the same output directory, cache, and settings. The runner stops if a stage
fails and records completed stages in `output/kaggle_run_summary.json`.
The equivalent direct command is `python kaggle_runner.py --full-kaggle-run`.

The runner can discover the Kaggle input mount automatically, or you can pass
`--data-root` (or both `--train-dir` and `--test-dir`). A recommended three-step
run measures blocking against the full target corpus, saves calibration, then
does final inference without repeating OOF:

```bash
python kaggle_runner.py \
  --retrieval-only \
  --sample-train-rows 20000 \
  --max-block-frequency 3000 \
  --candidate-channel-limit-multiplier 0.5 \
  --name-ngram-limit 4 \
  --max-test-candidates 25

python kaggle_runner.py \
  --calibration-only \
  --sample-train-rows 8000 \
  --cv-folds 3 \
  --max-block-frequency 3000 \
  --candidate-channel-limit-multiplier 0.5 \
  --name-ngram-limit 4 \
  --max-test-candidates 25

python kaggle_runner.py \
  --skip-oof \
  --final-train-rows 40000 \
  --max-block-frequency 3000 \
  --candidate-channel-limit-multiplier 0.5 \
  --name-ngram-limit 4 \
  --max-test-candidates 25 \
  --test-batch-size 50000
```

`--retrieval-only` writes `candidate_retrieval_diagnostics.csv` and exits;
`--calibration-only` saves `calibration.pkl` and exits; `--skip-oof` loads that
artifact, trains the final pair model once, writes and validates both TSVs, and
creates `AMAZON_ML_TEAM_submission.zip`. Keep the output directory between these
steps so the calibration artifact and versioned block-index cache are reused. The
blocking, graph, semantic retrieval, and candidate-cap settings must match across
calibration and final inference. The ZIP is also copied to `/kaggle/working/` when
available.

`python run.py --pipeline ...` forwards workflow options to `kaggle_runner.py`;
`python run.py --mode full` also starts the three-stage pipeline. For a standalone
stage, use `kaggle_runner.py --retrieval-only`, `--calibration-only`, or `--skip-oof`.
`run.py --mode test` and `run.py --mode sweep` remain legacy retrieval-only utilities.

The runner defaults to an 8,000-row OOF sample, 40,000 final training rows,
3 folds, block frequency 3,000, channel multiplier 0.5, name n-gram limit 4,
top-25 candidates, 50,000-row test batches, and a disabled target graph. Override
these with CLI options; common settings can also be supplied through `ER_*` environment variables. It
checks dependencies, discovers the train/test folders, and writes the outputs and
run summary beneath `output/` by default. Full-corpus runtime still needs to be
measured on the target Kaggle session.

Git LFS stores the provided 1.09 GB challenge archive. The runner extracts its seven
challenge TSVs automatically (about 2.4 GB unpacked), so Git does not store a second
copy of the extracted data. Public GitHub LFS downloads count against the repository
owner's monthly bandwidth allowance.

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


<!-- RUNS_METRICS_START -->

![runs metrics](runs/metrics.png)


| run_id | timestamp | candidate_recall | candidate_mean | total_seconds |
|---|---:|---:|---:|---:|
| 20260926T092300Z-65afb09d | 2026-09-26T09:23:54.400085+00:00 | 0.0000 | 1.2 | 0.1 |

<!-- RUNS_METRICS_END -->
