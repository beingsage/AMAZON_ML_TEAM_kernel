"""Focused regression coverage for the V7 calibration and diagnostics changes."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PIPELINE_SRC))

import entity_resolution_pipeline as er  # noqa: E402


class CapturingPairModel:
    def __init__(self) -> None:
        self.matrix: np.ndarray | None = None
        self.matrices: list[np.ndarray] = []

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        self.matrix = np.asarray(matrix, dtype=float).copy()
        self.matrices.append(self.matrix)
        core_name = matrix[:, er.FEATURE_NAMES.index("name_core_exact")]
        positive = np.where(core_name > 0, 0.9, 0.1)
        return np.column_stack((1.0 - positive, positive))


def make_calibration_result(holdout_score_shift: bool = False) -> tuple[er.OOFModelResult, dict[str, set[str]]]:
    entity_ids: list[str] = []
    folds: list[int] = []
    rows: list[dict[str, object]] = []
    truth: dict[str, set[str]] = {}
    for fold in range(5):
        for offset in range(4):
            entity_id = f"e{fold}-{offset}"
            linked = offset < 2
            entity_ids.append(entity_id)
            folds.append(fold)
            matched_id = f"S2-{entity_id}-positive"
            truth[entity_id] = {matched_id} if linked else set()
            positive_score = 0.9 if linked else 0.1
            negative_score = 0.1 if linked else 0.2
            if fold == 0 and holdout_score_shift:
                positive_score, negative_score = 0.01, 0.99
            rows.extend((
                {"entity_id": entity_id, "candidate_id": matched_id,
                 "label": int(linked), "probability": positive_score, "fold": fold},
                {"entity_id": entity_id, "candidate_id": f"S2-{entity_id}-negative",
                 "label": 0, "probability": negative_score, "fold": fold},
            ))
    result = er.OOFModelResult(
        model_type="logistic",
        pair_rows=pd.DataFrame(rows),
        entity_features=np.zeros((len(entity_ids), len(er.ENTITY_FEATURE_NAMES))),
        entity_ids=entity_ids,
        fold_by_entity=np.asarray(folds, dtype=int),
        mined_negatives=pd.DataFrame(columns=["entity_id", "candidate_id", "label", "features"]),
        candidate_recall_by_fold={fold: 1.0 for fold in range(5)},
    )
    return result, truth


class V7CalibrationTests(unittest.TestCase):
    def test_pooled_pair_calibration_and_threshold_exclude_fold_zero(self):
        baseline, truth = make_calibration_result()
        shifted_holdout, _ = make_calibration_result(holdout_score_shift=True)

        er.calibrate_and_compare_oof_models({"baseline": baseline}, truth, 5)
        er.calibrate_and_compare_oof_models({"shifted": shifted_holdout}, truth, 5)

        self.assertEqual(baseline.threshold, shifted_holdout.threshold)
        self.assertEqual(baseline.score_margin, shifted_holdout.score_margin)
        np.testing.assert_allclose(
            baseline.probability_calibrator.named_steps["logisticregression"].coef_,
            shifted_holdout.probability_calibrator.named_steps["logisticregression"].coef_,
        )

    def test_entity_gate_refits_after_tuning_on_fold_four(self):
        entity_ids: list[str] = []
        folds: list[int] = []
        rows: list[dict[str, object]] = []
        truth: dict[str, set[str]] = {}
        features = np.zeros((200, len(er.ENTITY_FEATURE_NAMES)), dtype=float)
        for fold in range(5):
            for offset in range(40):
                entity_id = f"gate-{fold}-{offset}"
                linked = offset < 20
                entity_ids.append(entity_id)
                folds.append(fold)
                features[len(entity_ids) - 1, 0] = float(linked)
                candidate_id = f"S2-{entity_id}"
                truth[entity_id] = {candidate_id} if linked else set()
                rows.append({
                    "entity_id": entity_id,
                    "candidate_id": candidate_id,
                    "label": int(linked),
                    "probability": 0.8,
                    "fold": fold,
                })
        result = er.OOFModelResult(
            model_type="logistic",
            pair_rows=pd.DataFrame(rows),
            entity_features=features,
            entity_ids=entity_ids,
            fold_by_entity=np.asarray(folds, dtype=int),
            mined_negatives=pd.DataFrame(),
            crossfit_probabilities=np.full(len(rows), 0.8),
        )

        decision, _, gate_score, _ = er.train_oof_entity_decision_layers(
            result, truth, 5, threshold=0.5, score_margin=None
        )

        self.assertIsNotNone(decision)
        self.assertIsNotNone(decision.classifier)
        self.assertGreater(gate_score, 0.9)
        self.assertEqual(int(np.max(decision.classifier.named_steps["standardscaler"].n_samples_seen_)), 160)

    def test_joint_cardinality_policy_returns_all_confidence_and_pair_cutoffs(self):
        ids = ["e0", "e1", "e2", "e3"]
        truth = {"e0": set(), "e1": {"m1"}, "e2": {"m2", "m3"}, "e3": set()}
        pair_rows = pd.DataFrame([
            ("e0", "m0", 0), ("e1", "m1", 1),
            ("e2", "m2", 1), ("e2", "m3", 1), ("e2", "m4", 0),
            ("e3", "m5", 0),
        ], columns=["entity_id", "candidate_id", "label"])
        pair_scores = np.asarray([0.1, 0.9, 0.9, 0.8, 0.2, 0.1])
        cardinality_scores = np.asarray([
            [0.90, 0.07, 0.03], [0.05, 0.90, 0.05],
            [0.03, 0.07, 0.90], [0.90, 0.07, 0.03],
        ])

        confidence, thresholds, margins, score = er.tune_joint_cardinality_policy(
            pair_rows, pair_scores, truth, ids, cardinality_scores,
            fallback_threshold=0.5, fallback_margin=None,
        )

        self.assertEqual(set(confidence), {0, 1, 2})
        self.assertIn(1, thresholds)
        self.assertIn(2, thresholds)
        self.assertIn(2, margins)
        self.assertGreaterEqual(score, 0.99)


class V7ReportingAndParityTests(unittest.TestCase):
    def test_model_comparison_report_ranks_oof_entity_score(self):
        def result(name: str, score: float) -> er.OOFModelResult:
            value = er.OOFModelResult(
                model_type=name,
                pair_rows=pd.DataFrame(columns=["entity_id", "candidate_id", "label", "fold"]),
                entity_features=np.empty((0, len(er.ENTITY_FEATURE_NAMES))),
                entity_ids=[], fold_by_entity=np.asarray([], dtype=int),
                mined_negatives=pd.DataFrame(),
            )
            value.cross_validation_f0_5 = score
            value.heldout_metrics = {"entity_f0_5": 0.4, "candidate_pair_recall": 0.8}
            return value

        with tempfile.TemporaryDirectory() as temp_dir:
            report = er.write_oof_model_comparison_report(
                Path(temp_dir) / "model_matrix.csv",
                {"lower": result("lower", 0.6), "higher": result("higher", 0.8)},
                "higher",
            )

        self.assertEqual(report["model_type"].tolist(), ["higher", "lower"])
        self.assertTrue(bool(report.iloc[0]["selected_by_oof_cv"]))

    def test_normalization_collision_feature_is_reported_as_a_bucket(self):
        buckets = er.diagnose_error_buckets({
            "name_core_exact": 1.0,
            "name_exact": 0.0,
            "common_name_tokens": 2.0,
            "address_token_jaccard": 0.9,
        }, expected=False, predicted=False)
        self.assertIn("normalization_collision", buckets)

    def test_retrieval_and_training_inference_features_match_for_both_context_modes(self):
        source_rows = []
        target_rows = []
        truth: dict[str, set[str]] = {}
        for position in range(24):
            entity_id = f"S1-{position}"
            name = f"Northwind Cafe Branch {position}"
            postal = f"021{position:02d}"
            address = f"{100 + position} Main Street, Boston, MA {postal}"
            source_rows.append({
                "entity_id": entity_id, "business_name": name,
                "business_address": address, "country": "USA",
            })
            target_rows.extend((
                {"entity_id": f"S2-{position}", "business_name": name,
                 "business_address": address.replace("Street", "St"), "country": "US"},
                {"entity_id": f"S3-{position}", "business_name": name,
                 "business_address": f"{300 + position} Main Road, Boston, MA {postal}",
                 "country": "US"},
            ))
            truth[entity_id] = {f"S2-{position}"}
        source1 = pd.DataFrame(source_rows)
        targets = pd.DataFrame(target_rows)
        lookup = er.build_source_lookup(targets)

        for context_mode in ("once", "per_channel"):
            index = er.build_block_index(
                targets, max_block_frequency=0, name_ngram_limit=0,
                channel_limit_multiplier=1.5, retrieval_context_mode=context_mode,
            )
            examples, _, _ = er.build_training_examples(
                source1, None, truth, training=False, source_lookup=lookup, index=index,
            )
            model = CapturingPairModel()
            predictions = list(er.predict_test_set(
                source1, None, model, threshold=0.5, source_lookup=lookup, index=index,
            ))

            self.assertEqual(len(model.matrices), len(predictions))
            training_features = {
                (str(row.entity_id), str(row.candidate_id)): er.features_to_vector(row.features)
                for row in examples.itertuples(index=False)
            }
            inference_features = {}
            for (entity_id, candidate_ids, _), matrix in zip(predictions, model.matrices):
                for position, candidate_id in enumerate(candidate_ids):
                    inference_features[(entity_id, candidate_id)] = matrix[position]
            self.assertEqual(set(training_features), set(inference_features))
            for candidate_id in training_features:
                np.testing.assert_allclose(
                    training_features[candidate_id], inference_features[candidate_id]
                )


if __name__ == "__main__":
    unittest.main()
