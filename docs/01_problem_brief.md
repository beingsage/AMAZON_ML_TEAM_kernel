# 01. Problem Brief

## Business entity resolution problem

This challenge is an entity resolution task over business records from three sources: Source 1, Source 2, and Source 3.

- Source 1 is the reference source and acts as the anchor set.
- Source 2 and Source 3 contain noisy, partial, or inconsistent business records.
- The goal is to identify all Source 2 / Source 3 records that refer to the same real business as each Source 1 entity.

## Data format

All files are TSV and must be read with `sep='\t'`.

Fields in each source file:

- `entity_id`
- `business_name`
- `business_address`
- `country`

The ground truth file has:

- `source1_entity_id`
- `matched_entity_ids`

The `matched_entity_ids` field is a comma-separated list of S2/S3 entity IDs. Empty string means no match.

## Important constraints

- Country is not fixed to only US and India; the test set can include France.
- The pipeline must not hard-code a limited country set.
- Every Source 1 entity in the test set must appear once in the final output.
- Every matched ID must be from Source 2 or Source 3 and must exist in the test set.
- Duplicates are forbidden in ID lists.
- Final outputs are scored using F_0.5 macro, which heavily rewards precision.

## Evaluation metric

The score is computed as:

```text
F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
```

This is first computed per Source 1 entity and then averaged across all Source 1 entities.

Interpretation:

- A false merge is very expensive.
- Singletons matter; correctly predicting an empty match set gives full credit.
- Good precision is more important than recall.

## Submission requirements

The final package requires:

- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`
- `code/business_entity_resolution/` with source code, README, requirements
- `Documentation_template.md` filled with the methodology

The validator is included in the challenge starter package and should be used before submission:

```bash
cd student_resource
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Technical challenge summary

The biggest practical difficulty is not the raw model architecture but the pipeline design:

1. Candidate generation must be powerful enough to avoid missing true matches.
2. Matching must be precise enough to avoid false merges.
3. Validation must be robust and threshold-aware.
4. Final files must strictly follow the required TSV schema.

This is a strong blocking + matching problem with noisy text fields.
