# Business Entity Resolution Pipeline

This folder contains the business entity resolution pipeline for the Amazon ML Challenge 2026.

## What it does

- reads the training and test TSV files
- creates a Source-1-level train/validation split from the labeled training set
- normalizes legal suffixes, DBA/trade-name markers, address abbreviations, accents, and transliterated text
- unions IDF-weighted token retrieval with exact-name, character n-gram, postal-code, address-number, and locality blocks
- drops very common block keys and keeps every candidate from the remaining blocks
- extracts name, address, locality, landmark, numeric, postal-code, state, country, character, rare-token, and corpus-IDF comparison features
- mines deterministic hard negatives alongside sampled negatives
- trains a logistic-regression matcher or optional LightGBM matcher
- offers `--model compare` to choose between them on the entity-level validation sample
- tunes the probability threshold and within-entity score margin against entity-macro F_0.5, including correctly predicted singletons
- reports candidate-pair recall, per-block size percentiles/hit rates, and diagnostic recall-at-K efficiency curves on train and validation samples
- learns a separate linked-versus-singleton gate from entity-level validation predictions, tunes it on one held-out partition, and keeps it only when it improves on a second held-out partition
- reports pair precision/recall, singleton and multi-match F_0.5, and Source-2/Source-3 F_0.5 in optional ablations
- can export a sampled normalization and collision audit
- optionally trains a small MLP over the top-K pairs from the first-stage matcher; it is retained for inference only when its validation score is higher
- writes final outputs for the test set:
  - `matching_results.tsv`
  - `candidate_pairs.tsv`

## Run

From the project root:

```bash
python code/business_entity_resolution/src/entity_resolution_pipeline.py \
  --train-dir student_resource/dataset/train \
  --test-dir student_resource/dataset/test \
  --output-dir student_resource/output
```

The default run uses a stratified 50,000-entity training sample and a 20,000-entity
validation sample. Sampling preserves the proportions of linked versus singleton
entities and repeated versus less common business names. Pass `--sample-train-rows 0` or
`--sample-validation-rows 0` to use the full respective split. Blocking omits keys
with more than 10,000 target records by default; adjust this with
`--max-block-frequency` after checking the reported candidate recall. Set it to `0`
to retain every posting list when recall is the priority and memory allows. Test inference
always runs the trained matcher for every Source-1 entity.

The default matcher remains logistic regression. Use `--model compare` to train
logistic regression and LightGBM on the same sampled pairs, score both with the same
entity-level validation split, and select the higher-scoring model for inference.

To try the optional neural reranker, add `--neural-reranker`; `--neural-top-k 20`
sets its per-entity candidate limit. This is a small MLP trained from scratch on the
engineered pair features, not a pretrained text model. The full retrieved candidate
list is still scored by the first-stage matcher and written to `candidate_pairs.tsv`.
The MLP only reranks its top-K pairs. It is used for final matching results only if
it improves validation entity-macro F_0.5 over the first-stage matcher.

Run `--run-ablations` to fit feature-group ablations, a random-negative-only
control, and normalization controls with legal-suffix stripping, DBA splitting, or
address abbreviation normalization disabled. Those controls rebuild the block index
and candidate examples so candidate recall is measured for each setting. The
`all_features` rows also compare the singleton gate on a separate held-out partition.
The report is written to `student_resource/output/ablation_results.csv` and includes
pair precision/recall, candidate recall, entity F_0.5, singleton and multi-match
scores, source-specific scores, elapsed time, and sampled peak resident memory for
each experiment. Ablations add
several full feature builds and model fits, so they take substantially longer than a
normal run. Run with
`--normalization-audit` to write a reproducible 5,000-row sample to
`student_resource/output/normalization_audit.tsv`; set
`--normalization-audit-rows 0` to audit every Source-1 row. The normal run also
prints how often phone-like fragments occur in addresses.

## Expected artifacts

- student_resource/output/matching_results.tsv
- student_resource/output/candidate_pairs.tsv

## Notes

The pipeline uses an inverted index rather than comparing every Source-1 row with every Source-2/3 row. City and state extraction is heuristic because source addresses are unstructured; postal codes, house numbers, units, and explicit locality components provide separate evidence. The full target corpus is indexed in memory, so challenge-scale runs still require substantial RAM. Per-block retrieval-at-K figures are diagnostics only: normal inference still sends every retrieved candidate to the pair matcher. The token-order name feature now measures original order; a separate sorted-token feature preserves the order-invariant signal. No dataset run or model comparison has been run yet; use the validation output before choosing non-default options.
