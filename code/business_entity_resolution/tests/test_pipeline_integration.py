"""Integration coverage for inference, outputs, normalization, and blocking."""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parents[1]
PIPELINE_SRC = PIPELINE_DIR / "src"
REPO_ROOT = PIPELINE_DIR.parents[1]
VALIDATOR_PATH = REPO_ROOT / "student_resource" / "utils" / "validate_submission.py"
sys.path.insert(0, str(PIPELINE_SRC))

import entity_resolution_pipeline as er  # noqa: E402

submission_validator = None
if VALIDATOR_PATH.is_file():
    _VALIDATOR_SPEC = importlib.util.spec_from_file_location("submission_validator", VALIDATOR_PATH)
    if _VALIDATOR_SPEC is not None and _VALIDATOR_SPEC.loader is not None:
        submission_validator = importlib.util.module_from_spec(_VALIDATOR_SPEC)
        _VALIDATOR_SPEC.loader.exec_module(submission_validator)


class ConstantPairModel:
    classes_ = np.asarray([0, 1])

    def __init__(self, probability: float = 0.9) -> None:
        self.probability = probability

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        positive = np.full(len(features), self.probability, dtype=float)
        return np.column_stack((1.0 - positive, positive))


class FixedGate:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        positive = np.full(len(features), 0.05, dtype=float)
        return np.column_stack((1.0 - positive, positive))


class FixedCardinality:
    classes_ = np.asarray([0, 1, 2])

    def __init__(self, cardinality: int) -> None:
        self.cardinality = cardinality

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.zeros((len(features), 3), dtype=float)
        probabilities[:, self.cardinality] = 1.0
        return probabilities


def prediction_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, er.BlockIndex]:
    source1 = pd.DataFrame([{
        "entity_id": "S1-query",
        "business_name": "Northwind Cafe",
        "business_address": "10 Main Street, Boston, MA 02110",
        "country": "USA",
    }])
    targets = pd.DataFrame([
        {"entity_id": "S2-a", "business_name": "Northwind Cafe",
         "business_address": "10 Main St, Boston, MA 02110", "country": "US"},
        {"entity_id": "S2-b", "business_name": "Northwind Cafe",
         "business_address": "22 Main St, Boston, MA 02110", "country": "US"},
        {"entity_id": "S3-c", "business_name": "Northwind Cafe",
         "business_address": "31 Main St, Boston, MA 02110", "country": "US"},
    ])
    lookup = er.build_source_lookup(targets)
    index = er.build_block_index(targets, max_block_frequency=0, name_ngram_limit=0)
    return source1, targets, lookup, index


class InferenceIntegrationTests(unittest.TestCase):
    def test_reranker_outputs_only_the_candidates_it_scores(self):
        source1, _, lookup, index = prediction_fixture()
        model = er.NeuralReranker(
            base_model=ConstantPairModel(0.9),
            reranker=ConstantPairModel(0.9),
            top_k=1,
        )

        entity_id, candidates, matches = next(er.predict_test_set(
            source1, None, model, threshold=0.5,
            source_lookup=lookup, index=index,
        ))

        self.assertEqual(entity_id, "S1-query")
        self.assertEqual(len(candidates), 1)
        self.assertTrue(set(matches).issubset(candidates))

    def test_gate_can_reject_while_preserving_retrieved_candidates(self):
        source1, _, lookup, index = prediction_fixture()
        decision = er.EntityDecisionLayer(classifier=FixedGate(), threshold=0.5)

        _, candidates, matches = next(er.predict_test_set(
            source1, None, ConstantPairModel(), threshold=0.5,
            source_lookup=lookup, index=index, entity_decision=decision,
        ))

        self.assertGreater(len(candidates), 0)
        self.assertEqual(matches, [])

    def test_cardinality_one_and_multi_match_paths(self):
        source1, _, lookup, index = prediction_fixture()
        for cardinality, expected_count in ((1, 1), (2, 3)):
            with self.subTest(cardinality=cardinality):
                decision = er.EntityDecisionLayer(
                    cardinality_classifier=FixedCardinality(cardinality),
                    cardinality_thresholds={1: 0.5, 2: 0.5},
                    cardinality_margins={1: None, 2: None},
                )
                _, candidates, matches = next(er.predict_test_set(
                    source1, None, ConstantPairModel(), threshold=0.5,
                    source_lookup=lookup, index=index, entity_decision=decision,
                ))
                self.assertEqual(len(candidates), 3)
                self.assertEqual(len(matches), expected_count)
                self.assertTrue(set(matches).issubset(candidates))


class OutputIntegrationTests(unittest.TestCase):
    def test_writer_headers_order_empty_rows_and_validator_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            test_dir = root / "test"
            test_dir.mkdir()
            pd.DataFrame({"entity_id": ["S1-2", "S1-1"]}).to_csv(
                test_dir / "test_source1.tsv", sep="\t", index=False
            )
            output_dir = root / "submission" / "output"
            root_output = root / "output"
            er.write_prediction_outputs(
                output_dir,
                root_output,
                [
                    ("S1-2", ["S2-1", "S3-1"], ["S3-1"]),
                    ("S1-1", [], []),
                ],
            )

            with (output_dir / "matching_results.tsv").open(encoding="utf-8", newline="") as handle:
                matching = list(csv.DictReader(handle, delimiter="\t"))
            with (output_dir / "candidate_pairs.tsv").open(encoding="utf-8", newline="") as handle:
                candidates = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(list(matching[0]), ["source1_entity_id", "matched_entity_ids"])
            self.assertEqual(list(candidates[0]), ["source1_entity_id", "candidate_entity_ids"])
            self.assertEqual([row["source1_entity_id"] for row in matching], ["S1-2", "S1-1"])
            self.assertEqual(matching[1]["matched_entity_ids"], "")
            self.assertEqual(candidates[1]["candidate_entity_ids"], "")
            self.assertEqual(
                (root_output / "matching_results.tsv").read_bytes(),
                (output_dir / "matching_results.tsv").read_bytes(),
            )

            if submission_validator is not None:
                with contextlib.redirect_stdout(io.StringIO()):
                    errors, _ = submission_validator.validate(
                        str(output_dir / "matching_results.tsv"),
                        str(output_dir / "candidate_pairs.tsv"),
                        str(test_dir),
                    )
                self.assertEqual(errors, [])

    def test_validator_flags_duplicate_source1_rows(self):
        if submission_validator is None:
            self.skipTest("challenge validator is not included in this package layout")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            test_dir = root / "test"
            test_dir.mkdir()
            pd.DataFrame({"entity_id": ["S1-1"]}).to_csv(
                test_dir / "test_source1.tsv", sep="\t", index=False
            )
            matching = root / "matching_results.tsv"
            candidates = root / "candidate_pairs.tsv"
            matching.write_text(
                "source1_entity_id\tmatched_entity_ids\nS1-1\t\nS1-1\t\n",
                encoding="utf-8",
            )
            candidates.write_text(
                "source1_entity_id\tcandidate_entity_ids\nS1-1\t\n",
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                errors, _ = submission_validator.validate(
                    str(matching), str(candidates), str(test_dir)
                )
            self.assertTrue(any("duplicate" in error.lower() for error in errors))

    def test_validator_discovery_handles_checkout_and_submission_layouts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "code" / "business_entity_resolution" / "src" / "pipeline.py"
            test_dir = root / "data" / "test"
            validator = root / "student_resource" / "utils" / "validate_submission.py"
            validator.parent.mkdir(parents=True)
            validator.write_text("# validator\n", encoding="utf-8")
            found = er.find_submission_validator(script, root, test_dir)
            self.assertEqual(found, validator.resolve())

            validator.unlink()
            packaged_test_dir = root / "external_data" / "test"
            self.assertIsNone(er.find_submission_validator(script, root, packaged_test_dir))


class NormalizationAndBlockingTests(unittest.TestCase):
    def test_normalization_handles_alias_suffix_and_international_addresses(self):
        self.assertEqual(er.normalize_text("Café"), "cafe")
        self.assertEqual(
            er.name_variants("Acme Logistics DBA Northstar Express, LLC"),
            [["acme", "logistics"], ["northstar", "express"]],
        )
        self.assertEqual(er.core_name_tokens("Northwind Holdings Pvt. Ltd."),
                         ["northwind", "holdings"])
        self.assertEqual(
            er.address_city_tokens("12 Rue de Rivoli, 75001 Paris, France", "France"),
            ["paris"],
        )
        self.assertEqual(
            er.address_city_tokens("Alexanderplatz 1, 10178 Berlin, Germany", "Germany"),
            ["berlin"],
        )
        self.assertEqual(
            er.address_city_tokens("123 Main Rd, Boston, MA 02110", "US"), ["boston"]
        )

    def test_blocking_unions_name_and_postal_channels_and_keeps_global_fallback(self):
        query = {
            "entity_id": "S1-query", "business_name": "Acme Bakery",
            "business_address": "11 Main Street, Boston, MA 02110", "country": "France",
        }
        targets = pd.DataFrame([
            {"entity_id": "S2-name", "business_name": "Acme Bakery",
             "business_address": "999 Oak Road, Austin, TX 78701", "country": "US"},
            {"entity_id": "S2-postal", "business_name": "Other Cafe",
             "business_address": "22 Side Street, Boston, MA 02110", "country": "US"},
        ])
        lookup = er.build_source_lookup(targets)
        index = er.build_block_index(targets, max_block_frequency=0, name_ngram_limit=0)

        candidates, channels = er.generate_candidate_details(query, index, lookup)

        self.assertIn("S2-name", candidates)
        self.assertIn("S2-postal", candidates)
        self.assertIn("S2-name", channels["name_exact"])
        self.assertIn("S2-postal", channels["postal"])

    def test_adaptive_frequency_ceiling_drops_overcommon_exact_name_posting(self):
        targets = pd.DataFrame([
            {"entity_id": f"S2-{position}", "business_name": "Acme Bakery",
             "business_address": f"{100 + position} Oak Road, Austin, TX 78701",
             "country": "US"}
            for position in range(6)
        ])

        index = er.build_block_index(targets, max_block_frequency=1, name_ngram_limit=0)

        self.assertNotIn(("name_exact", "us|acme bakery"), index)
        self.assertIn(("name_exact", "*|acme bakery"), index.key_document_frequency)


class TinyEndToEndTests(unittest.TestCase):
    def test_cli_trains_predicts_writes_and_validates_synthetic_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_dir = root / "train"
            test_dir = root / "test"
            train_dir.mkdir()
            test_dir.mkdir()

            train_source1 = []
            train_source2 = []
            ground_truth = []
            for position in range(12):
                entity_id = f"S1-train-{position:02d}"
                target_id = f"S2-train-{position:02d}"
                address = f"{100 + position} Main Street, Boston, MA {21000 + position:05d}"
                train_source1.append({
                    "entity_id": entity_id, "business_name": "Northwind Cafe Group",
                    "business_address": address, "country": "US",
                })
                train_source2.append({
                    "entity_id": target_id, "business_name": "Northwind Cafe Group",
                    "business_address": address, "country": "US",
                })
                ground_truth.append({
                    "source1_entity_id": entity_id,
                    "matched_entity_ids": target_id if position < 6 else "",
                })
            pd.DataFrame(train_source1).to_csv(train_dir / "train_source1.tsv", sep="\t", index=False)
            pd.DataFrame(train_source2).to_csv(train_dir / "train_source2.tsv", sep="\t", index=False)
            pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"]).to_csv(
                train_dir / "train_source3.tsv", sep="\t", index=False
            )
            pd.DataFrame(ground_truth).to_csv(
                train_dir / "train_ground_truth.tsv", sep="\t", index=False
            )

            test_source1 = pd.DataFrame([
                {"entity_id": "S1-test-1", "business_name": "Northwind Cafe Group",
                 "business_address": "10 Main Street, Boston, MA 02110", "country": "US"},
                {"entity_id": "S1-test-2", "business_name": "Northwind Cafe Group",
                 "business_address": "20 Main Street, Boston, MA 02120", "country": "US"},
            ])
            test_source1.to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)
            pd.DataFrame([
                {"entity_id": "S2-test-1", "business_name": "Northwind Cafe Group",
                 "business_address": "10 Main Street, Boston, MA 02110", "country": "US"},
                {"entity_id": "S2-test-2", "business_name": "Northwind Cafe Group",
                 "business_address": "20 Main Street, Boston, MA 02120", "country": "US"},
            ]).to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)
            pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"]).to_csv(
                test_dir / "test_source3.tsv", sep="\t", index=False
            )

            script = PIPELINE_SRC / "entity_resolution_pipeline.py"
            command = [
                sys.executable, str(script),
                "--train-dir", str(train_dir),
                "--test-dir", str(test_dir),
                "--output-dir", str(root / "submission" / "output"),
                "--model", "logistic",
                "--cv-folds", "3",
                "--negatives-per-positive", "2",
                "--random-negatives", "0",
                "--adversarial-negatives-per-entity", "0",
                "--max-block-frequency", "0",
                "--name-ngram-limit", "0",
                "--validate-submission",
            ]
            completed = subprocess.run(
                command, cwd=root, capture_output=True, text=True, timeout=180, check=False
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + "\n" + completed.stderr)

            matching_path = root / "output" / "matching_results.tsv"
            candidate_path = root / "output" / "candidate_pairs.tsv"
            self.assertTrue(matching_path.is_file())
            self.assertTrue(candidate_path.is_file())
            if submission_validator is not None:
                with contextlib.redirect_stdout(io.StringIO()):
                    errors, _ = submission_validator.validate(
                        str(matching_path), str(candidate_path), str(test_dir)
                    )
                self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
