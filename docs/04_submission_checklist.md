# 04. Submission Checklist

## Before model training

- [ ] Confirm train/test file schema and entity ID prefixes
- [ ] Check whether all fields are read with tab-separated parsing
- [ ] Build an initial data profile for missing values and noise patterns
- [ ] Define a validation split from training data
- [ ] Decide candidate-generation strategy

## During pipeline development

- [ ] Implement normalization functions for names and addresses
- [ ] Build candidate pairs per Source 1 entity
- [ ] Compute pair-level features
- [ ] Train and validate the matching model
- [ ] Tune threshold using macro F_0.5
- [ ] Review common false positives and false negatives

## Before final export

- [ ] Ensure every Source 1 entity has exactly one row in `matching_results.tsv`
- [ ] Ensure `candidate_pairs.tsv` contains the exact model inference set
- [ ] Ensure IDs are from S2/S3 only
- [ ] Ensure no duplicates exist inside matched or candidate ID lists
- [ ] Ensure candidate IDs are subsets of actual test dataset IDs
- [ ] Run the challenge validator

## Final package structure

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

## Validation command

```bash
cd student_resource
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

The goal is a clean PASS with no validation issues before any leaderboard upload.

## Final execution sequence

1. Build candidate pairs from training data
2. Train the model on the training subset
3. Select threshold on validation data
4. Generate predictions for the test set
5. Validate the output files
6. Package submission zip
7. Upload leaderboard TSV
8. Submit final zip archive
