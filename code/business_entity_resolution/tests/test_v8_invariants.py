"""Regression coverage for V8 threshold and mixed-negative fixes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PIPELINE_SRC))

import entity_resolution_pipeline as er  # noqa: E402


class V8ThresholdRegressionTests(unittest.TestCase):
    def test_gate_threshold_compares_normalized_scores(self):
        entity_ids = ["linked-1", "linked-2", "singleton-1", "singleton-2"]
        truth = {
            "linked-1": {"S2-match-1"},
            "linked-2": {"S2-match-2"},
            "singleton-1": set(),
            "singleton-2": set(),
        }
        pair_predictions = {
            "linked-1": {"S2-match-1"},
            "linked-2": {"S2-match-2"},
            "singleton-1": {"S2-false-1"},
            "singleton-2": {"S2-false-2"},
        }

        threshold = er.best_gate_threshold(
            np.asarray([0.9, 0.8, 0.1, 0.2]), pair_predictions, truth, entity_ids
        )

        self.assertGreater(threshold, 0.2)
        self.assertLess(threshold, 0.8)

    def test_top1_threshold_uses_the_stored_best_candidate_id(self):
        pair_rows = pd.DataFrame([
            ("linked", "S2-correct", 1),
            ("singleton", "S2-false", 0),
        ], columns=["entity_id", "candidate_id", "label"])
        truth = {"linked": {"S2-correct"}, "singleton": set()}

        threshold = er.best_top1_threshold(
            pair_rows, np.asarray([0.9, 0.8]), truth, ["linked", "singleton"]
        )

        self.assertGreater(threshold, 0.8)
        self.assertLess(threshold, 0.9)


class V8MixedNegativePoolTests(unittest.TestCase):
    def test_pool_counts_are_nonnegative_and_sum_to_requested_total(self):
        for total in range(201):
            with self.subTest(total=total):
                counts = er.mixed_negative_pool_counts(total)
                self.assertEqual(sum(counts), total)
                self.assertTrue(all(count >= 0 for count in counts))
                if total >= 4:
                    self.assertTrue(all(count > 0 for count in counts))

    def test_mixed_sampler_respects_limit_uniqueness_and_seed(self):
        ranked = [
            (float(100 - position), f"candidate-{position}",
             ("same_name_wrong_address", "same_address_wrong_name", "same_city", "other")[position % 4])
            for position in range(50)
        ]
        first = er.select_mixed_hard_negatives(ranked, 20, np.random.default_rng(73))
        second = er.select_mixed_hard_negatives(ranked, 20, np.random.default_rng(73))

        self.assertEqual(len(first), 20)
        self.assertEqual(len(set(first)), 20)
        self.assertEqual(first, second)
        self.assertEqual(er.select_mixed_hard_negatives(ranked, 0, np.random.default_rng(1)), [])
        self.assertEqual(len(er.select_mixed_hard_negatives(ranked, 100, np.random.default_rng(1))), 50)


if __name__ == "__main__":
    unittest.main()
