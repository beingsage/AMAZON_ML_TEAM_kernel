# 02. Execution Plan

## Phase 1 — Environment and data inspection

Tasks:

- Confirm the dataset files and schema.
- Verify the exact column names and file format.
- Inspect the training labels and the distribution of match counts.
- Check whether there are empty match sets, multi-match groups, and label cardinality issues.

Outputs:

- Data profiling notes
- Summary of edge cases
- Candidate feature hypotheses

## Phase 2 — Validation strategy

Tasks:

- Split the training data into train/validation folds.
- Keep the validation split representative of the true distribution.
- Measure performance using the macro F_0.5 score.
- Tune threshold and candidate selection based on validation outcomes.

Important notes:

- The test set has a third country not seen in training.
- Thresholds should not be tightly coupled to one country label.
- Validation should emphasize false-positive control.

## Phase 3 — Candidate generation (blocking)

Tasks:

- Create candidate pairs between each Source 1 entity and plausible Source 2 / Source 3 matches.
- Use text normalization plus token/phonetic blocking.
- Add parallel candidate keys to reduce recall loss.
- Ensure the final candidate list is the last stage before the model.

Likely blocking strategies to test:

- Normalized name similarity
- Address-token overlap
- Country compatibility
- Token-level Jaccard / cosine similarity
- Hash or token-based indices
- Phonetic or edit-distance heuristics

Goal:

- High recall with manageable candidate count.
- Candidate set must include true matches but avoid exploding combinatorially.

## Phase 4 — Feature engineering

Build pair-level features for each S1-S2/S3 candidate pair.

Examples:

- Name exact match / normalized match
- Levenshtein distance
- Jaro-Winkler / token sort ratio
- Character n-gram similarity
- Address token overlap / set similarity
- Exact country match
- Shared city/state/pincode token features
- Abbreviation normalization features

Focus on features that still work when names and addresses are noisy or partially missing.

## Phase 5 — Matching model

A good baseline is a supervised binary classifier on pair-level features.

Possible architectures:

- Gradient-boosted trees on engineered features
- Logistic regression baseline
- Random forest or XGBoost
- A tuned hybrid model if needed

Model guidance:

- Use validation F_0.5 to choose threshold.
- Increase precision by choosing a conservative threshold.
- Use the candidate set as the inference pool; do not evaluate all cross-product pairs.

## Phase 6 — Final prediction generation

Tasks:

- For each Source 1 entity, gather all candidate matches scored by the model.
- Keep only those above the decision threshold.
- Export `matching_results.tsv` with one row per Source 1 entity.
- Export `candidate_pairs.tsv` as the exact candidate set used by the model.

## Phase 7 — Validation and packaging

Tasks:

- Run the challenge validator on the generated files.
- Inspect any warnings or formatting issues.
- Generate final output folder.
- Produce the challenge package structure:
  - `output/`
  - `code/business_entity_resolution/`
  - `Documentation_template.md`

## Deliverables

- Working baseline notebook or Python scripts
- Candidate generation code
- Trained model and feature pipeline
- Validation score report
- Final submission files
- Final methodology write-up
