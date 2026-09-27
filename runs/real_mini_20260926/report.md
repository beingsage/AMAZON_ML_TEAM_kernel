# Real-data mini run — 2026-09-26

## Sample

- Labeled Source-1 rows: 200 for training (180 linked, 20 singletons) and 60 separate held-out rows (48 linked, 12 singletons).
- Indexed target rows: 30,107 sampled Source-2/Source-3 records. All labeled target IDs for the initial query sample were retained, with deterministic background rows sampled across the training files; mini blocking recall is therefore conditional on this reduced target universe.
- Model: Logistic Regression; 3 entity folds; semantic character-TF-IDF retrieval enabled; additional hard-negative rounds disabled for this diagnostic run.
- Both structural submission validation and target-ID existence validation passed.

## Results

| Metric | Result |
|---|---:|
| OOF entity F0.5, folds 1–2 | 0.9613 |
| OOF fold range / std | 0.9581–0.9644 / 0.0032 |
| Reserved fold entity F0.5 | 0.9522 |
| Reserved fold blocking recall | 1.0000 |
| Held-out mini blocking pair recall | 1.0000 (175/175) |
| Held-out pair precision / recall / micro F0.5 | 0.9822 / 0.9486 / 0.9753 |
| Held-out entity macro F0.5 | 0.9707 |
| Held-out pairs | 166 TP, 3 FP, 9 FN |
| Mean / p95 candidates per held-out entity | 306.5 / 445.3 |

## Retrieval fallback observation

On the 60 full-record held-out queries, symbolic plus Soundex blocking already recalled all labeled pairs. On a 40-query name-only slice with addresses hidden, baseline recall was 96.2% (102/106). Character-TF-IDF produced the same recall: its sparse-candidate gate did not activate on these rows because existing name channels returned at least five candidates. This run therefore validates the integrated pipeline and the fallback code path through unit fixtures, but it shows no incremental recall benefit on this particular real-data slice.

The optional character-TF-IDF matrix has a default 100,000-target safety cap. These mini-run results do not establish full-corpus runtime or quality; the full target corpus exceeds that cap, so the option must remain disabled there until a scalable index is evaluated.

## Reproduce

From the repository root, run:

```bash
cd runs/real_mini_20260926
../../.venv/bin/python ../../code/business_entity_resolution/src/entity_resolution_pipeline.py \
  --train-dir data/train --test-dir data/test --output-dir replay_output \
  --model logistic --cv-folds 3 --sample-train-rows 180 \
  --negatives-per-positive 3 --random-negatives 1 \
  --adversarial-negatives-per-entity 0 --semantic-retrieval \
  --semantic-index-max-targets 100000 --validate-submission
```

This keeps any repeated run's secondary `output/` files under this run directory.

## Artifacts

- `real_mini_end_to_end_metrics.json`
- `real_mini_retrieval_metrics.json`
- `real_mini_name_only_metrics.json`
- `pipeline_output/oof_model_comparison.csv`
- `pipeline_output/matching_results.tsv`
- `pipeline_output/candidate_pairs.tsv`
