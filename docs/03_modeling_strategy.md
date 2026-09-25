# 03. Modeling Strategy Blueprint

## Recommended approach

The strongest approach for this challenge is a two-stage entity resolution pipeline:

1. Candidate generation / blocking
2. Pair-level matching model with a precision-first threshold

This is preferable to a brute-force all-to-all setting because the data is large and noisy, and the scoring metric punishes false merges.

## Stage A — candidate generation

We should generate a candidate list for each Source 1 record by comparing it against relevant S2/S3 records from the same or nearby business context.

Good candidate-generation cues:

- normalized business name tokens
- common address tokens (city, state, road, street, pin code if present)
- country hints
- brand or abbreviation similarity
- phonetic matches

The candidate generator should favor recall over precision because the model can reject weak candidates later.

## Stage B — pair-level matching

Each candidate pair should be represented as a feature vector. Example features include:

- normalized name similarity
- token overlap on name
- edit-distance-like similarity metrics
- address token overlap and set similarity
- country match indicator
- exact shared address tokens or landmark tokens
- abbreviation-aware feature flags

The paired examples can then be classified as:

- match
- non-match

## Threshold tuning

Because the metric is precision-heavy, the threshold should be tuned to minimize false merges rather than maximize recall blindly.

Recommended procedure:

- Use a validation split from the training set.
- Score candidate pairs with the model.
- Sweep thresholds.
- Select the threshold with the highest macro F_0.5 on validation data.

This process should be repeated with a conservative threshold if the false-positive rate is too high.

## Baseline progression

1. Rule-based blocking baseline
2. Feature-engineered logistic regression or tree model
3. Improved tree model with tuned threshold
4. If needed, stronger feature combinations and more robust blocking

## Risk areas to watch

- Overly aggressive blocking that creates too-large candidate sets
- Under-blocking that misses true matches
- Name-only matching that ignores address context
- Country assumptions that fail on the test distribution
- Thresholds tuned only for recall, not for F_0.5

## Expected winning pattern

The winning pipeline usually combines:

- text normalization for names and addresses
- blocking by multiple keys to recover recall
- pair features with lexical similarity and token overlap
- precision-oriented thresholding
- strict schema compliance for final output files

This challenge is more about disciplined pipeline design than a single model trick.
