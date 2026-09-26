"""Focused invariants for threshold selection and train/inference parity."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PIPELINE_SRC))

import entity_resolution_pipeline as er  # noqa: E402


class FeatureBasedModel:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        name_exact = matrix[:, er.FEATURE_NAMES.index("name_core_exact")]
        positive = np.where(name_exact > 0, 0.9, 0.1)
        return np.column_stack((1.0 - positive, positive))


class FixedCardinalityModel:
    classes_ = np.asarray([0, 1, 2])

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        del matrix
        return np.asarray([
            [0.90, 0.08, 0.02],
            [0.02, 0.92, 0.06],
            [0.01, 0.15, 0.84],
            [0.20, 0.40, 0.40],
        ])


def brute_force_threshold_search(pair_df: pd.DataFrame,
                                 probabilities: np.ndarray,
                                 truth: dict[str, set[str]],
                                 entity_ids: list[str]):
    if pair_df.empty or not len(probabilities):
        return 1.000001, None, er.score_entity_predictions({}, truth, entity_ids)
    margins = (0.05, 0.1, 0.2, 0.35, 0.5, None)
    thresholds = [0.0]
    thresholds.extend(float(np.nextafter(value, np.inf))
                      for value in np.unique(probabilities))
    best_threshold, best_margin, best_score = 0.0, None, -1.0
    for margin in margins:
        for threshold in thresholds:
            predictions = er.predictions_from_pair_scores(
                pair_df, probabilities, threshold, margin, entity_ids
            )
            score = er.score_entity_predictions(predictions, truth, entity_ids)
            conservative_tie = (
                np.isclose(score, best_score)
                and (threshold > best_threshold or (
                    threshold == best_threshold
                    and margin is not None
                    and (best_margin is None or margin < best_margin)
                ))
            )
            score_is_better = score > best_score and not np.isclose(score, best_score)
            if score_is_better or conservative_tie:
                best_threshold, best_margin, best_score = threshold, margin, score
    return best_threshold, best_margin, best_score


class ThresholdOptimizerTests(unittest.TestCase):
    def test_event_sweep_matches_brute_force_on_edge_cases(self):
        fixtures = [
            # One entity, one positive and one negative.
            ([("e1", "m1", 1), ("e1", "m2", 0)], [0.9, 0.4], {"e1": {"m1"}}, ["e1"]),
            # Two entities, including a singleton and a two-match entity.
            ([
                ("e1", "m1", 1), ("e1", "m2", 1), ("e1", "m3", 0),
                ("e2", "m4", 0), ("e2", "m5", 0),
            ], [0.8, 0.7, 0.7, 0.5, 0.2],
             {"e1": {"m1", "m2"}, "e2": set()}, ["e1", "e2"]),
            # Ties and duplicate probabilities across entities.
            ([
                ("e1", "m1", 1), ("e1", "m2", 0),
                ("e2", "m3", 0), ("e2", "m4", 1),
            ], [0.5, 0.5, 0.5, 0.5],
             {"e1": {"m1"}, "e2": {"m4"}}, ["e1", "e2"]),
            # All-zero and all-one labels.
            ([("e1", "m1", 0), ("e2", "m2", 0)], [0.1, 0.3],
             {"e1": set(), "e2": set()}, ["e1", "e2"]),
            ([("e1", "m1", 1), ("e2", "m2", 1)], [0.2, 0.8],
             {"e1": {"m1"}, "e2": {"m2"}}, ["e1", "e2"]),
            # An entity with no candidate rows remains part of the macro score.
            ([("e1", "m1", 1)], [0.6],
             {"e1": {"m1"}, "e2": set()}, ["e1", "e2"]),
        ]
        for records, scores, truth, entity_ids in fixtures:
            with self.subTest(records=records):
                pair_df = pd.DataFrame(records, columns=["entity_id", "candidate_id", "label"])
                probabilities = np.asarray(scores, dtype=float)
                actual = er.threshold_search_from_scores(pair_df, probabilities, truth, entity_ids)
                expected = brute_force_threshold_search(pair_df, probabilities, truth, entity_ids)
                self.assertEqual(actual[0], expected[0])
                self.assertEqual(actual[1], expected[1])
                self.assertAlmostEqual(actual[2], expected[2])

        rng = np.random.default_rng(2026)
        for trial in range(25):
            entity_ids = [f"r{trial}-e{i}" for i in range(int(rng.integers(1, 5)))]
            records = []
            scores = []
            truth = {}
            for entity_id in entity_ids:
                candidate_count = int(rng.integers(0, 5))
                truth[entity_id] = set()
                for candidate_number in range(candidate_count):
                    candidate_id = f"{entity_id}-m{candidate_number}"
                    label = int(rng.integers(0, 2))
                    records.append((entity_id, candidate_id, label))
                    scores.append(float(rng.choice([0.1, 0.3, 0.5, 0.8])))
                    if label:
                        truth[entity_id].add(candidate_id)
            pair_df = pd.DataFrame(records, columns=["entity_id", "candidate_id", "label"])
            probabilities = np.asarray(scores, dtype=float)
            with self.subTest(random_trial=trial):
                actual = er.threshold_search_from_scores(pair_df, probabilities, truth, entity_ids)
                expected = brute_force_threshold_search(pair_df, probabilities, truth, entity_ids)
                self.assertEqual(actual[0], expected[0])
                self.assertEqual(actual[1], expected[1])
                self.assertAlmostEqual(actual[2], expected[2])

    def test_empty_candidate_frame_has_defined_result(self):
        pair_df = pd.DataFrame(columns=["entity_id", "candidate_id", "label"])
        actual = er.threshold_search_from_scores(
            pair_df, np.asarray([], dtype=float), {"e1": set()}, ["e1"]
        )
        self.assertEqual(actual, (1.000001, None, 1.0))


class PipelineInvariantTests(unittest.TestCase):
    def test_pair_and_entity_feature_vectors_match_declared_schemas(self):
        left = {
            "entity_id": "S1-1", "business_name": "Northwind Cafe",
            "business_address": "123 Main Street, Boston, MA 02110", "country": "USA",
        }
        right = {
            "entity_id": "S2-1", "business_name": "Northwind Cafe",
            "business_address": "123 Main St, Boston, MA 02110", "country": "US",
        }
        features = er.build_pair_features(left, right)
        self.assertEqual(set(er.FEATURE_NAMES) - set(features), set())
        self.assertEqual(len(er.features_to_vector(features)), len(er.FEATURE_NAMES))
        conflicting = dict(right, country="France")
        self.assertGreaterEqual(
            er.build_pair_features(left, conflicting)["contradiction_score"], 5.0
        )
        entity_features = er.entity_decision_feature_vector([features], np.asarray([0.9]), 0.5)
        self.assertEqual(len(entity_features), len(er.ENTITY_FEATURE_NAMES))

    def test_zero_ngram_limit_keeps_all_name_ngrams(self):
        name = "International Northwind Manufacturing Services"
        self.assertEqual(
            er.selected_name_ngrams(name, 0),
            er.selected_name_ngrams(name, 100_000),
        )

    def test_cardinality_probabilities_can_abstain_and_enforce_class_policy(self):
        labels, confidence = er.cardinality_labels_from_probabilities(
            FixedCardinalityModel(), np.zeros((4, 1)), confidence_threshold=0.5
        )
        np.testing.assert_array_equal(labels, [0, 1, 2, 0])
        np.testing.assert_allclose(confidence, [0.90, 0.92, 0.84, 0.40])

        pair_df = pd.DataFrame([
            ("e0", "m0", 1),
            ("e1", "m1", 1), ("e1", "m2", 0),
            ("e2", "m3", 1), ("e2", "m4", 1),
        ], columns=["entity_id", "candidate_id", "label"])
        predictions = er.cardinality_adjusted_predictions(
            pair_df, np.asarray([0.9, 0.8, 0.7, 0.8, 0.7]), 0.5, None,
            ["e0", "e1", "e2"], np.asarray([0, 1, 2]),
            {1: 0.5, 2: 0.5}, {1: None, 2: None},
        )
        self.assertEqual(predictions["e0"], set())
        self.assertLessEqual(len(predictions["e1"]), 1)
        self.assertEqual(len(predictions["e2"]), 2)

    def test_source_specific_model_uses_shared_fallback_for_one_class_source(self):
        matrix = np.zeros((4, len(er.FEATURE_NAMES)), dtype=float)
        matrix[:, er.FEATURE_NAMES.index("name_exact")] = [0.0, 1.0, 0.2, 0.9]
        matrix[:2, er.FEATURE_NAMES.index("candidate_source_s2")] = 1.0
        labels = np.asarray([0, 0, 0, 1], dtype=int)
        model = er.make_pair_estimator("logistic_source_specific", seed=7)
        model.fit(matrix, labels)
        self.assertIsNone(model.source2_model)
        np.testing.assert_allclose(
            model.predict_proba(matrix[:2]), model.shared_model.predict_proba(matrix[:2])
        )

    def test_prediction_output_preserves_candidate_and_match_invariants(self):
        source1 = pd.DataFrame([{
            "entity_id": "S1-1", "business_name": "Northwind Cafe",
            "business_address": "123 Main Street, Boston, MA 02110", "country": "USA",
        }])
        targets = pd.DataFrame([
            {"entity_id": "S2-1", "business_name": "Northwind Cafe",
             "business_address": "123 Main St, Boston, MA 02110", "country": "US"},
            {"entity_id": "S3-1", "business_name": "Northwind Cafe",
             "business_address": "123 Main Street, Boston, MA 02110", "country": "US"},
        ])
        lookup = er.build_source_lookup(targets)
        index = er.build_block_index(targets, max_block_frequency=0)
        predictions = list(er.predict_test_set(
            source1, None, FeatureBasedModel(), 0.5, source_lookup=lookup, index=index
        ))
        expected_candidates = er.generate_candidate_details(
            source1.iloc[0].to_dict(), index, lookup, collect_blocks=False
        )[0]
        self.assertEqual(len(predictions), 1)
        entity_id, candidates, matches = predictions[0]
        self.assertEqual(entity_id, "S1-1")
        self.assertEqual(candidates, expected_candidates)
        self.assertEqual(len(candidates), len(set(candidates)))
        self.assertTrue(all(value.startswith(("S2-", "S3-")) for value in candidates))
        self.assertTrue(set(matches).issubset(candidates))

    def test_empty_candidate_set_returns_one_empty_result(self):
        source1 = pd.DataFrame([{
            "entity_id": "S1-1", "business_name": "Unrelated Name",
            "business_address": "999 Unknown Road, Nowhere, ZZ 99999", "country": "France",
        }])
        targets = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
        lookup = er.build_source_lookup(targets)
        index = er.build_block_index(targets, max_block_frequency=0)
        predictions = list(er.predict_test_set(
            source1, None, None, 0.5, source_lookup=lookup, index=index
        ))
        self.assertEqual(predictions, [("S1-1", [], [])])


if __name__ == "__main__":
    unittest.main()
