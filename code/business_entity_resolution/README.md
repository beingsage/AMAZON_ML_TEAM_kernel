# Business Entity Resolution Pipeline

This folder contains the business entity resolution pipeline for the Amazon ML Challenge 2026.

## Pipeline

- Normalizes names and addresses, including legal suffixes, DBA aliases, abbreviations, accents, and transliterated text.
- Treats numeric house-number bases such as `701` and `701A` as compatible, records conflicting letter suffixes separately, and recognizes unit numbers only after explicit unit markers.
- Builds country-aware exact-name, name-token, name-pair, character n-gram, gated Soundex, address, postal, house-number, and city blocks. Soundex runs only when symbolic retrieval finds fewer than five candidates. Each block channel has its own frequency ceiling, IDF weight, and candidate limit; the channels are then unioned and ranked with name and address context added once per candidate.
- Optionally adds a character-TF-IDF nearest-name fallback when symbolic and Soundex retrieval return fewer than five candidates. It handles orthographic variation and is not a language-model embedding. The sparse target index is opt-in and capped to control memory.
- Learns country-specific postal-format and component-order address signatures from the indexed Source-2/3 records, without external address data.
- Optionally builds conservative direct S2↔S3 links from exact name and strong address evidence, then adds direct-edge counts and score summaries to S1 pair features. It does not infer transitive links. `--disable-target-graph` skips this memory-intensive stage and leaves those features at zero.
- Measures candidate recall, candidate counts, block-level recall, and retrieval recall at several candidate limits.
- Trains Logistic Regression, LightGBM, or both. LightGBM is the default; `--model compare` selects from entity-level OOF results. `--ensemble` adds an OOF-validated mean-probability ensemble of the two models. `--compare-class-weight` adds unweighted variants, while `--compare-source-specific` adds separate S1→S2 and S1→S3 models.
- Performs stratified entity-level OOF validation. Pair thresholds and margins are selected from cross-fitted calibrated scores on folds 1–4, with fold 0 excluded. A pooled calibrator fit on folds 1–4 is used for fold 0 and test inference. The entity gate and cardinality layers use inner held-out meta-scores for calibration, tune their policies on the final non-holdout fold, then refit on every non-holdout fold.
- Jointly tunes cardinality confidence cutoffs for P(0/1/2+), the singleton top-1 threshold, and the multi-match threshold/margin against entity F0.5 on the tuning fold.
- Adds contradiction evidence and top-candidate raw-field evidence to the pair and entity decision features.
- Uses a mixed negative pool: 40% hardest, 30% medium-hard, 20% structurally hard, and 10% random, with deterministic fallback when a pool is too small. Each OOF fold mines only from its training entities and refits before scoring held-out entities; two bounded final-model mining rounds use the same mix. `calibration_drift.csv` records OOF and final-model score distributions.
- Writes OOF error examples and per-bucket precision, recall, entity F0.5, and repair impact; ranks evaluated pair models by OOF entity F0.5 with fold spread, fold-0 diagnostics, and runtime; writes normalization collision summaries and final TSVs. Candidate recall reporting distinguishes raw blocking recall, post-top-K recall, and thresholded match recall. The `other_candidate` bucket is printed and placed first in its report.

The optional MLP reranker (`--neural-reranker`) scores only the first-stage model's top-K pairs per entity. Its OOF scores are evaluated alongside the selected pair families, so it is used only if its mean OOF entity F0.5 is higher.

## Run

The pinned environment is CPython 3.12.3 with the versions in `requirements.txt` (validated in the repository `.venv`). From the project root, create it with:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r code/business_entity_resolution/requirements.txt
```

The input layout is `student_resource/dataset/{train,test}` with the challenge TSV filenames shown below. From the project root, run:

```bash
python code/business_entity_resolution/src/entity_resolution_pipeline.py \
  --train-dir student_resource/dataset/train \
  --test-dir student_resource/dataset/test \
  --output-dir student_resource/output
```

The default uses all labeled Source-1 rows, five stratified entity folds, LightGBM, 20 sampled negatives per positive, and five OOF hard negatives per entity. Fold index 0 is reserved for the independent evaluation; indices 1–4 are used for pair threshold selection and model comparison. Final training can add up to five mixed hard negatives per entity in each of two model-driven rounds before test inference.

Set `--seed` to reproduce sampling and model initialization (default `42`). Full-data runs keep the target index and generated training examples in memory; the ten-model experiment matrix and retrieval ablations add substantial runtime and memory use. `--sample-train-rows N` caps the OOF model-selection queries, and `--final-train-rows N` independently caps the labeled Source-1 queries used to build final pair-training examples. Both default to all labeled rows in the pipeline. These limits do not sample the target index or test inference: all records in the provided target files remain eligible, and every test Source-1 row is scored. No fixed RAM or runtime figure is claimed here; record those from the target machine and full challenge run.

Block indexes are cached separately for the training and test target files under `OUTPUT_DIR/block_index_cache` by default. The cache key includes the target file paths, sizes, modification times, and all index-building settings. Use `--block-index-cache-dir PATH` to place caches on persistent storage, and `--rebuild-block-index` to force a rebuild. Keep the cache directory between runs to avoid rebuilding the multi-million-row index.

To measure blocking on a larger query sample against the complete training target files without computing pair features or running OOF, use `--retrieval-only`:

```bash
python code/business_entity_resolution/src/entity_resolution_pipeline.py \
  --train-dir /kaggle/input/.../train \
  --test-dir /kaggle/input/.../test \
  --output-dir /kaggle/working/output \
  --sample-train-rows 20000 \
  --max-block-frequency 3000 \
  --candidate-channel-limit-multiplier 0.5 \
  --name-ngram-limit 4 \
  --max-test-candidates 25 \
  --retrieval-only
```

This reports raw blocking recall, retained top-K recall, candidate-count percentiles, and query throughput in `candidate_retrieval_diagnostics.csv`. It also builds and caches the training target index for later calibration.

For a two-stage Kaggle run, first calibrate on a small sample and stop before final training or test inference:

```bash
python code/business_entity_resolution/src/entity_resolution_pipeline.py \
  --train-dir /kaggle/input/.../train \
  --test-dir /kaggle/input/.../test \
  --output-dir /kaggle/working/output \
  --sample-train-rows 8000 \
  --cv-folds 3 \
  --model lightgbm \
  --max-block-frequency 3000 \
  --candidate-channel-limit-multiplier 0.5 \
  --name-ngram-limit 4 \
  --negatives-per-positive 6 \
  --adversarial-negatives-per-entity 2 \
  --calibration-only \
  --seed 42
```

This writes `calibration.pkl` with the selected matcher, threshold, score margin, probability calibrator, and entity decision layer. It also saves the training block-index cache. Preserve both artifacts between Kaggle sessions. Then train once on a moderate labeled sample and run chunked inference without OOF:

```bash
python code/business_entity_resolution/src/entity_resolution_pipeline.py \
  --train-dir /kaggle/input/.../train \
  --test-dir /kaggle/input/.../test \
  --output-dir /kaggle/working/output \
  --calibration-artifact /kaggle/working/output/calibration.pkl \
  --skip-oof \
  --final-train-rows 40000 \
  --max-block-frequency 3000 \
  --candidate-channel-limit-multiplier 0.5 \
  --name-ngram-limit 4 \
  --max-test-candidates 25 \
  --test-batch-size 50000 \
  --negatives-per-positive 6 \
  --seed 42
```

The inference pass reads Source-1 test rows in chunks and writes both TSVs incrementally. The default `--max-test-candidates 25` caps each entity's ranked candidate list before pair scoring during calibration, final training, and inference; set it to `0` to keep all retrieved candidates. Calibration and final-run blocking, candidate-cap, graph, and semantic-retrieval settings must match; the pipeline checks these when it loads the calibration artifact.

Use `--sample-train-rows N` to reduce OOF selection work and `--final-train-rows N` to limit final pair-training rows independently; `0` means all rows. Set `--cv-folds` to change the fold count; at least three linked and three singleton entities are needed per fold. Set `--model logistic`, `--model lightgbm`, `--model compare`, or `--model ensemble`. LightGBM must be installed for the default, `compare`, and `ensemble` modes.

Candidate blocking omits keys above a per-channel ceiling derived from `--max-block-frequency` (10,000 by default), and then keeps a bounded top list from each retrieval channel. Set the ceiling to `0` to retain all postings. Check candidate recall and candidate-count diagnostics before changing the ceiling because blocking sets the maximum recoverable match recall.

Add `--neural-reranker` to compare the optional top-K MLP pipeline during OOF selection; `--neural-top-k` controls its per-entity cap. Add `--run-ablations` to fit feature, negative-sampling, normalization, and gate ablations. Ablations rebuild examples and indexes and can add substantial runtime and memory use.

Set `--name-ngram-limit` and `--candidate-channel-limit-multiplier` to change the retrieval profile. Set `--name-ngram-limit 0` to use all name n-grams. `--retrieval-context-mode` defaults to `once`; `per_channel` is available for comparison. Add `--run-retrieval-ablation` to evaluate n-gram caps 6/10/16/24/32/all, channel cap multipliers 0.75/1/1.5/2, and both context modes on the OOF Source-1 sample. It writes recall, recall-at-K, candidate-count percentiles, and runtime to `retrieval_ablation.csv` without fitting pair models.

Add `--run-country-stress-test` to measure candidate recall after masking France, US/India, or all country values on a bounded Source-1 sample. Set `--country-stress-rows` to cap each scenario (default 1,000; `0` uses all eligible rows). Results go to `country_open_set_stress.csv`; these are retrieval stress results, not a replacement for country-held-out model evaluation.

Add `--semantic-retrieval` to enable the character-TF-IDF fallback. `--semantic-index-max-targets` defaults to 100,000 target records and fails fast if exceeded; `--semantic-top-k` defaults to 50 and `--semantic-min-similarity` defaults to 0.12. Tune these against labeled recall and candidate-list size before enabling the option on larger target corpora.

Add `--run-experiment-matrix` to compare ten shared, source-specific, weighted, unweighted, and MLP variants in the same OOF run. Model selection uses mean entity F0.5 across folds 1–4 only; `oof_model_comparison.csv` includes mean non-holdout and fold-0 singleton/multi/source-specific metrics, separate blocking/top-K/match recall, pair diagnostics, and per-model OOF runtime. This matrix requires LightGBM and also runs both MLP variants.

Add `--normalization-audit` to write a reproducible sample to `normalization_audit.tsv` and a full-source collision summary to `normalization_collisions.tsv`. Set `--normalization-audit-rows 0` to include all Source-1 rows in the detailed audit. Collision summaries cover normalized names, core names, addresses, and country aliases across the training sources. Submission validation is opt-in with `--validate-submission`; the validator is discovered from the checkout or workspace and missing validator files do not fail inference. Add `--check-submission-ids` to run validation with target ID existence checks at higher memory cost. `--skip-submission-validation` remains as a deprecated compatibility option.

## Output artifacts

- `matching_results.tsv` — selected S2/S3 matches for each test S1 entity.
- `candidate_pairs.tsv` — candidates passed to the final pair decision stage. The test inference cap defaults to 25 before pair scoring; with the MLP reranker enabled, the reranker can narrow this further.
- `oof_validation_errors.tsv` — false positives, false negatives, and candidate-retrieval misses from the reserved fold.
- `oof_error_buckets.csv` — per-bucket counts, precision, recall, bucket entity F0.5, and overall F0.5 change if those errors were fixed.
- `oof_model_comparison.csv` — pair matcher ranking by non-holdout OOF entity F0.5, with mean, min/max/std fold spread, fold-0 diagnostics, and runtime. It separates `blocking_candidate_recall`, `base_model_topk_recall`, and thresholded `match_recall` (`reranker_match_recall` on fold 0).
- `normalization_collisions.tsv` and `normalization_audit.tsv` — written when normalization audit is enabled.
- `ablation_results.csv` — written when ablations are enabled.
- `retrieval_ablation.csv` — written when retrieval profile comparisons are enabled.
- `country_open_set_stress.csv` — written when country-masking stress tests are enabled.
- `calibration_drift.csv` — fold-0 held-out pooled-calibrator summaries, cross-fitted non-holdout threshold-score summaries, and final-model raw/calibrated score distributions. Final training scores are in-sample diagnostics, not a replacement for held-out calibration.
- `calibration.pkl` — selected model family, OOF threshold/margin, pair-probability calibrator, and entity decision layer for a later `--skip-oof` run.
- `block_index_cache/` — versioned train/test target indexes reused when input file metadata and index settings match.
- `candidate_retrieval_diagnostics.csv` — full-target raw and retained candidate recall plus retrieval throughput when `--retrieval-only` is enabled.

The focused invariant tests are in `tests/test_v6_invariants.py`, `tests/test_v7_invariants.py`, `tests/test_v8_invariants.py`, and `tests/test_review_hardening.py`; they cover threshold search, cardinality behavior, calibration isolation, fold-local mining, phonetic and character-TF-IDF retrieval, country stress reporting, direct graph/miner behavior, fold spread, and mixed-negative sampling. `tests/test_pipeline_integration.py` covers inference branches, blocking and normalization cases, TSV validation, validator discovery, and a small train-to-output CLI run. The challenge validator checks are skipped if its development-only script is absent from the package.

Error buckets are feature-based diagnostics and can overlap; `f0_5_impact_if_fixed` estimates the overall reserved-fold change if each bucket's false positives and false negatives were corrected.

The required output TSVs are written under `--output-dir` and `output/` at the project root. The candidate list contains at most `--max-test-candidates` ranked retrieval results per entity (25 by default). With the MLP reranker, it contains only the candidates that reach the final scoring stage.

## Limits

The S2/S3 graph is deliberately conservative and heuristic. Country address profiles are learned from the challenge's indexed target records and currently summarize postal-code lengths and broad comma-component layout; they do not parse every country's address grammar. Candidate recall, OOF quality, and runtime depend on the provided data and must be read from the generated diagnostics. The optional character-TF-IDF channel is approximate lexical retrieval, not general semantic understanding.
