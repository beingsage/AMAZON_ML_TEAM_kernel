# ML Challenge 2026: Business Entity Resolution

- **Team Name:** [Enter team name]
- **Team Members:** [Enter team members]
- **Submission Date:** [Enter submission date]

## 1. Executive Summary

This solution uses multi-channel blocking to retrieve plausible Source-2 and Source-3 records, then scores each candidate with a supervised pair model. Model selection, calibration, thresholds, and optional entity-level decisions are evaluated with entity-stratified out-of-fold validation; final inference writes both required TSV files.

## 2. Methodology

### 2.1 Problem Analysis

Source records may differ in punctuation, abbreviations, legal suffixes, aliases, transliteration, address order, postal-code format, and missing components. A Source-1 entity may have no matches, one match, or multiple matches across the two target sources. The test set includes countries not present in training, so country handling remains open-set and uses global retrieval keys as a fallback.

### 2.2 Solution Strategy

**Approach Type:** Hybrid blocking and pair classification, with optional entity-level gate and cardinality prediction.

**Core Innovation:** Country-aware retrieval combines name, address, postal, house-number, and locality signals, while the pair model also receives direct S2↔S3 corroboration features. The pipeline keeps retrieval recall separate from final match recall.

The processing stages are:

1. Normalize names and addresses, including accent folding, optional transliteration, business-name aliases, legal suffixes, and address abbreviations.
2. Build a target index using exact names, token and token-pair keys, character n-grams, address tokens, postal codes, house numbers, and city tokens. Apply per-channel frequency ceilings and candidate limits, then rank the union of candidates.
3. Build pair features for name, address, geography, contradictions, retrieval context, and direct target-source graph evidence.
4. Train Logistic Regression or LightGBM pair models using a deterministic mixture of hard, medium-hard, structurally hard, and random negatives.
5. Select the model, calibrator, and pair threshold using entity-level OOF folds. Fold 0 is held out for independent diagnostics; threshold selection uses cross-fitted scores from non-holdout folds, and the pooled calibrator is used for fold 0 and test inference.
6. Evaluate optional gate and cardinality layers on separate fit, tune, and evaluation folds. The optional neural reranker only scores the base ranker’s top-K candidates.
7. Write one row per test Source-1 entity to `matching_results.tsv` and the final inference candidate set to `candidate_pairs.tsv`.

## 3. Candidate Generation (Blocking)

Blocking uses eight channel families: exact name, adjacent name-token pairs, character n-grams, name tokens, address tokens, postal codes, house numbers, and city tokens. Each key has country-specific and global forms. Frequency ceilings reduce common postings; channel-specific top lists are unioned and ranked using IDF-weighted retrieval scores plus address and name context.

- **Candidate pairs generated:** Read per-fold and final counts from the pipeline diagnostics; these depend on the run configuration and input data.
- **Recall measurement:** `oof_model_comparison.csv` distinguishes blocking recall, base-model top-K recall, and thresholded match recall. `retrieval_ablation.csv` reports recall-at-K and candidate-count distributions when retrieval ablations are enabled.
- **How true matches are protected:** Retrieval favors recall, retains several independent channels, includes global keys for country fallback, and reports misses separately from pair-model false negatives. The candidate list in the submission is the set passed to the final pair decision stage; with the neural reranker enabled, that is its top-K input.

## 4. Matching Model

**Features used:**

- **Name:** normalized exact/core-name matches, token and character similarity, aliases, and rare-token overlap.
- **Address:** token and character similarity, address components, postal codes, house numbers, locality/state signals, and contradiction features.
- **Other:** country compatibility, source identity, candidate rank and retrieval-channel metadata, and direct S2↔S3 support.

**Model type:** Logistic Regression and LightGBM are supported. LightGBM is the default; `--model compare` selects between the pair-model families using non-holdout entity F0.5. The optional MLP reranker is considered only when enabled and when its OOF entity score is competitive.

**Threshold selection:** Pair probabilities are Platt-calibrated. The pair threshold and within-entity margin are selected against the challenge’s entity-level F0.5 on cross-fitted non-holdout scores. Fold 0 remains outside threshold and model selection. Calibration and score-distribution diagnostics are written to `calibration_drift.csv`.

## 5. Results & Error Analysis

No full-data score is asserted in this document. After running the pipeline on the challenge training data, copy the reserved-fold entity, singleton, multi-match, and source-specific metrics from `oof_model_comparison.csv` into this section. Use `oof_validation_errors.tsv` and `oof_error_buckets.csv` to summarize observed false positives, false negatives, and retrieval misses. Do not substitute synthetic-test results for challenge validation results.

## 6. Conclusion

The pipeline is designed to make retrieval limits, pair-model errors, and entity-level decisions separately measurable. Final performance and resource use should be reported from a full challenge-data run with the chosen configuration and seed.

## Appendix

### A. Code Artefacts

The runnable code is under `code/business_entity_resolution/`. Install the pinned direct dependencies with:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r code/business_entity_resolution/requirements.txt
```

From the project root, reproduce the outputs with:

```bash
python code/business_entity_resolution/src/entity_resolution_pipeline.py \
  --train-dir student_resource/dataset/train \
  --test-dir student_resource/dataset/test \
  --output-dir output
```

The default seed is 42. Add `--validate-submission` to run the challenge validator when it is available. The test suite is under `code/business_entity_resolution/tests/`; run it with `python -m unittest discover -s code/business_entity_resolution/tests -p 'test_*.py'` in the pinned environment.

### B. Additional Results

Attach generated retrieval, calibration, model-comparison, and error-bucket reports after the full challenge run. The integration tests use synthetic records only to check pipeline behavior and do not estimate challenge performance.
