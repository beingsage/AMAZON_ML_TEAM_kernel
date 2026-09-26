"""Regression checks for retrieval, fold mining, and review-driven diagnostics."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

PIPELINE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PIPELINE_SRC))

import entity_resolution_pipeline as er  # noqa: E402


class ConstantProbabilityModel:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        return np.tile(np.asarray([[0.1, 0.9]]), (len(matrix), 1))


class ReviewHardeningTests(unittest.TestCase):
    def test_non_latin_name_is_transliterated_and_soundexed(self):
        self.assertIn("moskva", er.normalize_text("МОСКВА"))
        self.assertEqual(er.soundex_code("Александр"), er.soundex_code("Aleksandr"))

    def test_soundex_matches_standard_reference_cases(self):
        self.assertEqual(er.soundex_code("Robert"), "R163")
        self.assertEqual(er.soundex_code("Rupert"), "R163")
        self.assertEqual(er.soundex_code("Tymczak"), "T522")
        self.assertEqual(er.soundex_code("Ashcraft"), "A261")
        self.assertEqual(er.soundex_code("Ashcroft"), "A261")

    def test_phonetic_fallback_is_gated_and_feature_metadata_is_explicit(self):
        target_rows = [{
            "entity_id": "S2-phonetic", "business_name": "Bip",
            "business_address": "", "country": "US",
        }]
        targets = pd.DataFrame(target_rows)
        lookup = er.build_source_lookup(targets)
        index = er.build_block_index(
            targets, max_block_frequency=0, name_ngram_limit=0, build_graph=False,
        )
        query = {
            "entity_id": "S1-query", "business_name": "Bub",
            "business_address": "", "country": "USA",
        }

        candidates, blocks = er.generate_candidate_details(query, index, lookup)
        self.assertEqual(candidates, ["S2-phonetic"])
        self.assertIn("S2-phonetic", blocks["name_phonetic"])
        self.assertEqual(
            {channel for channel, candidate_ids in blocks.items()
             if "S2-phonetic" in candidate_ids},
            {"name_phonetic"},
        )

        raw_features = er.build_pair_features(query, lookup.loc["S2-phonetic"], index)
        self.assertEqual(raw_features["candidate_retrieval_channel_count"], 0.0)
        self.assertEqual(raw_features["candidate_block_name_phonetic"], 0.0)
        with self.assertRaisesRegex(ValueError, "add_candidate_block_memberships"):
            er.predict_feature_rows(ConstantProbabilityModel(), [raw_features])
        metadata = [dict()]
        er.add_candidate_block_memberships(metadata, candidates, blocks, index)
        raw_features.update(metadata[0])
        self.assertGreater(raw_features["candidate_retrieval_channel_count"], 0.0)
        self.assertEqual(raw_features["candidate_block_name_phonetic"], 1.0)
        self.assertEqual(
            len(er.features_to_vector(raw_features)), len(er.FEATURE_NAMES)
        )
        self.assertEqual(len(er.predict_feature_rows(ConstantProbabilityModel(), [raw_features])), 1)

        common_targets = [
            {"entity_id": f"S2-common-{position}", "business_name": f"Robert Store {position}",
             "business_address": "", "country": "US"}
            for position in range(er.PHONETIC_FALLBACK_MIN_SYMBOLIC_CANDIDATES)
        ] + target_rows
        common_df = pd.DataFrame(common_targets)
        common_lookup = er.build_source_lookup(common_df)
        common_index = er.BlockIndex()
        common_index.document_count = len(common_df)
        common_ids = {row["entity_id"] for row in common_targets[:-1]}
        common_index[("name_token", "us|robert")] = common_ids
        common_index[("name_token", "*|robert")] = common_ids
        common_index[("name_phonetic", "us|R163")] = {"S2-phonetic"}
        common_index[("name_phonetic", "*|R163")] = {"S2-phonetic"}
        common_candidates = er.generate_candidates({**query, "business_name": "Robert"},
                                                    common_index, common_lookup)
        self.assertNotIn("S2-phonetic", common_candidates)

    def test_character_tfidf_fallback_recovers_a_nonphonetic_typo_only_when_sparse(self):
        targets = pd.DataFrame([{
            "entity_id": "S2-fuzzy", "business_name": "Buc",
            "business_address": "", "country": "US",
        }])
        lookup = er.build_source_lookup(targets)
        query = {
            "entity_id": "S1-fuzzy", "business_name": "Bub",
            "business_address": "", "country": "US",
        }
        symbolic_index = er.build_block_index(
            targets, max_block_frequency=0, name_ngram_limit=0, build_graph=False,
        )
        semantic_index = er.build_block_index(
            targets, max_block_frequency=0, name_ngram_limit=0, build_graph=False,
            semantic_retrieval=True, semantic_top_k=3,
        )

        self.assertNotEqual(er.soundex_code("Bub"), er.soundex_code("Buc"))
        self.assertEqual(er.generate_candidates(query, symbolic_index, lookup), [])
        candidates, blocks = er.generate_candidate_details(query, semantic_index, lookup)
        self.assertEqual(candidates, ["S2-fuzzy"])
        self.assertEqual(blocks["name_tfidf"], {"S2-fuzzy"})

        feature = er.build_pair_features(query, lookup.loc["S2-fuzzy"], semantic_index)
        metadata = [{}]
        er.add_candidate_block_memberships(metadata, candidates, blocks, semantic_index)
        feature.update(metadata[0])
        self.assertEqual(feature["candidate_block_name_tfidf"], 1.0)
        self.assertEqual(len(er.features_to_vector(feature)), len(er.FEATURE_NAMES))

        crowded_targets = pd.DataFrame([
            {"entity_id": f"S2-common-{position}",
             "business_name": f"Robert Store {position}",
             "business_address": "", "country": "US"}
            for position in range(er.PHONETIC_FALLBACK_MIN_SYMBOLIC_CANDIDATES)
        ] + targets.to_dict("records"))
        crowded_lookup = er.build_source_lookup(crowded_targets)
        crowded_index = er.build_block_index(
            crowded_targets, max_block_frequency=0, name_ngram_limit=0,
            build_graph=False, semantic_retrieval=True,
        )
        crowded_query = {**query, "business_name": "Robert"}
        with patch.object(crowded_index.semantic_vectorizer, "transform",
                          side_effect=AssertionError("semantic fallback was not gated")):
            crowded_candidates, crowded_blocks = er.generate_candidate_details(
                crowded_query, crowded_index, crowded_lookup
            )
        self.assertGreaterEqual(len(crowded_candidates), er.PHONETIC_FALLBACK_MIN_SYMBOLIC_CANDIDATES)
        self.assertNotIn("name_tfidf", {key for key, ids in crowded_blocks.items() if ids})

    def test_character_tfidf_index_enforces_its_target_memory_cap(self):
        targets = pd.DataFrame([{
            "entity_id": "S2-one", "business_name": "One Name",
            "business_address": "", "country": "US",
        }])
        two_targets = pd.concat([targets, targets.assign(entity_id="S2-two")], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "capped at 1 target rows"):
            er.build_block_index(
                two_targets, semantic_retrieval=True, semantic_max_documents=1,
            )

    def test_cross_source_graph_builds_direct_verified_edges(self):
        target_rows = [
            {"entity_id": "S2-a", "business_name": "Acme Bakery",
             "business_address": "10 Main Street, Boston, MA 02110", "country": "US"},
            {"entity_id": "S3-a", "business_name": "Acme Bakery",
             "business_address": "10 Main Street, Boston, MA 02110", "country": "US"},
            {"entity_id": "S2-b", "business_name": "Acme Bakery",
             "business_address": "90 Oak Road, Boston, MA 02111", "country": "US"},
        ]
        index = er.BlockIndex()

        er.populate_cross_source_graph(pd.DataFrame(target_rows), index)

        self.assertEqual(index.target_graph_neighbors["S2-a"], {"S3-a"})
        self.assertEqual(index.target_graph_neighbors["S3-a"], {"S2-a"})
        self.assertNotIn("S2-b", index.target_graph_neighbors)
        self.assertEqual(index.target_graph_features["S2-a"]["direct_edge_count"], 1.0)

    def test_hard_negative_miner_returns_unseen_false_candidate(self):
        source1 = pd.DataFrame([{
            "entity_id": "S1-q", "business_name": "Robert",
            "business_address": "", "country": "US",
        }])
        targets = pd.DataFrame([{
            "entity_id": "S2-r", "business_name": "Rupert",
            "business_address": "", "country": "US",
        }])
        lookup = er.build_source_lookup(targets)
        index = er.build_block_index(
            targets, max_block_frequency=0, name_ngram_limit=0, build_graph=False,
        )
        existing = pd.DataFrame(columns=["entity_id", "candidate_id", "label", "features"])

        mined = er.mine_final_adversarial_negatives(
            source1, {"S1-q": set()}, lookup, index, ConstantProbabilityModel(),
            existing, per_entity=1, seed=7,
        )

        self.assertEqual(mined[["entity_id", "candidate_id", "label"]].to_dict("records"), [
            {"entity_id": "S1-q", "candidate_id": "S2-r", "label": 0}
        ])

    def test_oof_mining_receives_training_entities_only(self):
        source_rows = []
        target_rows = []
        truth = {}
        for position in range(6):
            entity_id = f"S1-{position}"
            if position < 3:
                name = f"Acme Branch {position}"
                truth[entity_id] = {f"S2-{position}"}
            else:
                name = f"Independent Outlet {position}"
                truth[entity_id] = set()
            source_rows.append({
                "entity_id": entity_id, "business_name": name,
                "business_address": f"{10 + position} Main Street, Boston, MA 021{position:02d}",
                "country": "US",
            })
            if position < 3:
                target_rows.append({
                    "entity_id": f"S2-{position}", "business_name": name,
                    "business_address": f"{10 + position} Main Street, Boston, MA 021{position:02d}",
                    "country": "US",
                })
        source1 = pd.DataFrame(source_rows)
        targets = pd.DataFrame(target_rows)
        lookup = er.build_source_lookup(targets)
        index = er.build_block_index(
            targets, max_block_frequency=0, name_ngram_limit=4,
        )
        received_training_ids: list[set[str]] = []

        def empty_fold_mining(fold_source1, *args, **kwargs):
            received_training_ids.append(set(fold_source1["entity_id"].astype(str)))
            return pd.DataFrame(columns=["entity_id", "candidate_id", "label", "features"])

        with patch.object(er, "mine_final_adversarial_negatives", side_effect=empty_fold_mining):
            er.run_entity_oof_validation(
                source1, truth, lookup, index, ["logistic"], folds=3, seed=23,
                negatives_per_positive=2, random_negatives=1,
                adversarial_negatives_per_entity=1,
            )

        all_ids = set(source1["entity_id"].astype(str))
        self.assertEqual(len(received_training_ids), 3)
        self.assertTrue(all(len(fold_ids) == 4 and fold_ids < all_ids
                            for fold_ids in received_training_ids))

    def test_ensemble_averages_positive_class_probabilities(self):
        class ProbabilityModel:
            def __init__(self, value: float) -> None:
                self.value = value

            def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
                positive = np.full(len(matrix), self.value)
                return np.column_stack((1.0 - positive, positive))

        ensemble = er.EnsemblePairEstimator([ProbabilityModel(0.2), ProbabilityModel(0.8)])
        probabilities = ensemble.predict_proba(np.zeros((3, 2)))[:, 1]
        np.testing.assert_allclose(probabilities, np.full(3, 0.5))

    def test_error_bucket_falls_back_to_visible_other_candidate(self):
        buckets = er.diagnose_error_buckets(
            {"has_name_both": 1.0, "has_address_both": 1.0},
            expected=False, predicted=False,
        )
        self.assertEqual(buckets, ["other_candidate"])

    def test_country_stress_report_measures_france_and_us_india_masking(self):
        source_rows = [
            {"entity_id": "S1-fr", "business_name": "Cafe Lyon",
             "business_address": "10 Rue Victor Hugo, 75001 Paris", "country": "France"},
            {"entity_id": "S1-us", "business_name": "Union Cafe",
             "business_address": "2 Main Street, Boston, MA 02110", "country": "US"},
            {"entity_id": "S1-in", "business_name": "Delhi Mart",
             "business_address": "12 Ring Road, Delhi 110001", "country": "India"},
        ]
        target_rows = [
            {"entity_id": f"S2-{row['entity_id'][3:]}",
             "business_name": row["business_name"],
             "business_address": row["business_address"], "country": row["country"]}
            for row in source_rows
        ]
        source1 = pd.DataFrame(source_rows)
        targets = pd.DataFrame(target_rows)
        lookup = er.build_source_lookup(targets)
        index = er.build_block_index(
            targets, max_block_frequency=0, name_ngram_limit=0,
        )
        truth = {
            row["entity_id"]: {f"S2-{row['entity_id'][3:]}"} for row in source_rows
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            report = er.write_country_open_set_stress_report(
                Path(temp_dir) / "country_stress.csv", source1, targets, lookup, index,
                truth, rows_per_scenario=10, seed=13, max_block_frequency=0,
                name_ngram_limit=0, channel_limit_multiplier=1.0,
                retrieval_context_mode="once",
            )

        self.assertIn("france_country_hidden", set(report["scenario"]))
        self.assertIn("us_india_country_hidden", set(report["scenario"]))
        self.assertIn("candidate_recall_delta", report.columns)

    def test_fold_spread_is_exported_in_oof_model_report(self):
        result = er.OOFModelResult(
            model_type="logistic", pair_rows=pd.DataFrame(
                columns=["entity_id", "candidate_id", "label", "fold"]
            ), entity_features=np.empty((0, len(er.ENTITY_FEATURE_NAMES)), dtype=float),
            entity_ids=[], fold_by_entity=np.asarray([], dtype=int),
            mined_negatives=pd.DataFrame(),
        )
        result.cross_validation_f0_5 = 0.5
        result.cross_validation_fold_scores = {1: 0.2, 2: 0.8}

        with tempfile.TemporaryDirectory() as temp_dir:
            report = er.write_oof_model_comparison_report(
                Path(temp_dir) / "model_matrix.csv", {"logistic": result}, "logistic",
            )

        self.assertAlmostEqual(report.loc[0, "oof_cv_f0_5_min"], 0.2)
        self.assertAlmostEqual(report.loc[0, "oof_cv_f0_5_max"], 0.8)
        self.assertGreater(report.loc[0, "oof_cv_f0_5_std"], 0.0)

    def test_gate_threshold_matches_a_sweep_over_exact_score_boundaries(self):
        rng = np.random.default_rng(19)
        for iteration in range(20):
            scores = rng.choice(np.linspace(0.0, 1.0, 11), size=12, replace=True)
            entity_ids = [f"e{position}" for position in range(len(scores))]
            truth = {entity_id: ({"match"} if position % 3 == 0 else set())
                     for position, entity_id in enumerate(entity_ids)}
            pair_predictions = {
                entity_id: ({"match"} if position % 2 == 0 else {"false"})
                for position, entity_id in enumerate(entity_ids)
            }
            threshold = er.best_gate_threshold(scores, pair_predictions, truth, entity_ids)
            candidate_thresholds = [0.0] + [
                float(np.nextafter(value, np.inf)) for value in np.unique(scores)
            ]
            sweep_scores = [er.score_entity_predictions(
                {entity_id: pair_predictions[entity_id] if score >= candidate else set()
                 for entity_id, score in zip(entity_ids, scores)}, truth, entity_ids,
            ) for candidate in candidate_thresholds]
            chosen_score = er.score_entity_predictions(
                {entity_id: pair_predictions[entity_id] if score >= threshold else set()
                 for entity_id, score in zip(entity_ids, scores)}, truth, entity_ids,
            )
            self.assertGreaterEqual(chosen_score + 1e-12, max(sweep_scores))


if __name__ == "__main__":
    unittest.main()
