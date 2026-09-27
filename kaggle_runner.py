#!/usr/bin/env python3
"""One-command Kaggle runner for the Amazon ML Challenge 2026."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "code/business_entity_resolution/src/entity_resolution_pipeline.py"
REQUIREMENTS = ROOT / "code/business_entity_resolution/requirements.txt"
TEMPLATE = ROOT / "student_resource/Documentation_template.md"
DATA_ARCHIVE = ROOT / "6ab10eb3b23ba_student_resource.zip"

TRAIN_FILES = (
    "train_source1.tsv", "train_source2.tsv", "train_source3.tsv",
    "train_ground_truth.tsv",
)
TEST_FILES = ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv")
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def info(message: str) -> None:
    print(f"[kaggle-runner] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval diagnostics, calibration, or full Kaggle inference and package the submission."
    )
    parser.add_argument("--data-root", help="Optional directory containing train/ and test/.")
    parser.add_argument("--train-dir", help="Optional explicit directory containing train TSVs.")
    parser.add_argument("--test-dir", help="Optional explicit directory containing test TSVs.")
    parser.add_argument("--output-dir", default=os.environ.get("ER_OUTPUT_DIR", str(ROOT / "output")))
    parser.add_argument("--team-name", default=os.environ.get("ER_TEAM_NAME", "AMAZON_ML_TEAM"))
    workflow = parser.add_mutually_exclusive_group()
    workflow.add_argument("--retrieval-only", action="store_true",
                          help="Measure full-target blocking recall and stop without training or packaging.")
    workflow.add_argument("--calibration-only", action="store_true",
                          help="Run sampled OOF calibration and stop before final training or packaging.")
    workflow.add_argument("--skip-oof", action="store_true",
                          help="Load a prior calibration artifact and run final training plus inference.")
    workflow.add_argument("--full-kaggle-run", action="store_true",
                          help="Run retrieval diagnostics, calibration, and lean final inference in sequence.")
    parser.add_argument("--retrieval-sample-rows", type=int,
                        default=int(os.environ.get("ER_RETRIEVAL_SAMPLE_ROWS", "20000")),
                        help="Source-1 rows used by --full-kaggle-run retrieval diagnostics.")
    parser.add_argument("--sample-train-rows", type=int, default=None,
                        help="OOF/retrieval query rows; defaults to 20k for retrieval and 8k otherwise.")
    parser.add_argument("--final-train-rows", type=int, default=None,
                        help="Labeled Source-1 rows for final fit; defaults to 40k; 0 uses all rows.")
    parser.add_argument("--cv-folds", type=int, default=int(os.environ.get("ER_CV_FOLDS", "3")))
    parser.add_argument("--model", choices=("logistic", "lightgbm", "compare", "ensemble"),
                        default=os.environ.get("ER_MODEL", "lightgbm"))
    parser.add_argument("--negatives-per-positive", type=int,
                        default=int(os.environ.get("ER_NEGATIVES_PER_POSITIVE", "6")))
    parser.add_argument("--random-negatives", type=int,
                        default=int(os.environ.get("ER_RANDOM_NEGATIVES", "2")))
    parser.add_argument("--adversarial-negatives-per-entity", type=int,
                        default=int(os.environ.get("ER_HARD_NEGATIVES", "2")))
    parser.add_argument("--max-block-frequency", type=int,
                        default=int(os.environ.get("ER_MAX_BLOCK_FREQUENCY", "3000")))
    parser.add_argument("--candidate-channel-limit-multiplier", type=float,
                        default=float(os.environ.get("ER_CHANNEL_LIMIT_MULTIPLIER", "0.5")))
    parser.add_argument("--name-ngram-limit", type=int,
                        default=int(os.environ.get("ER_NAME_NGRAM_LIMIT", "4")))
    parser.add_argument("--retrieval-context-mode", choices=("once", "per_channel"),
                        default=os.environ.get("ER_RETRIEVAL_CONTEXT_MODE", "once"))
    parser.add_argument("--max-test-candidates", type=int,
                        default=int(os.environ.get("ER_MAX_TEST_CANDIDATES", "25")))
    parser.add_argument("--test-batch-size", type=int,
                        default=int(os.environ.get("ER_TEST_BATCH_SIZE", "50000")))
    parser.add_argument("--block-index-cache-dir", default=os.environ.get("ER_BLOCK_INDEX_CACHE_DIR"),
                        help="Persistent cache directory; defaults to OUTPUT_DIR/block_index_cache.")
    parser.add_argument("--rebuild-block-index", action="store_true",
                        help="Rebuild indexes even when a compatible cache is present.")
    parser.add_argument("--calibration-artifact", default=os.environ.get("ER_CALIBRATION_ARTIFACT"),
                        help="Calibration pickle; defaults to OUTPUT_DIR/calibration.pkl.")
    graph_options = parser.add_mutually_exclusive_group()
    graph_options.add_argument("--disable-target-graph", dest="disable_target_graph",
                                action="store_true", help="Skip the memory-intensive S2↔S3 graph.")
    graph_options.add_argument("--enable-target-graph", dest="disable_target_graph",
                                action="store_false", help="Build the optional S2↔S3 graph.")
    default_disable_graph = os.environ.get("ER_DISABLE_TARGET_GRAPH", "1").strip().lower() not in {
        "0", "false", "no"
    }
    parser.set_defaults(disable_target_graph=default_disable_graph)
    parser.add_argument("--semantic-retrieval", action="store_true")
    parser.add_argument("--semantic-index-max-targets", type=int, default=100000)
    parser.add_argument("--semantic-top-k", type=int, default=50)
    parser.add_argument("--semantic-min-similarity", type=float, default=0.12)
    parser.add_argument("--no-install-dependencies", action="store_true",
                        help="Skip checking/installing the pinned project requirements.")
    args = parser.parse_args()
    if not any((args.retrieval_only, args.calibration_only, args.skip_oof, args.full_kaggle_run)):
        args.full_kaggle_run = True
    if args.sample_train_rows is None:
        sample_default = os.environ.get("ER_SAMPLE_TRAIN_ROWS")
        args.sample_train_rows = int(sample_default) if sample_default is not None else (
            20000 if args.retrieval_only else 8000
        )
    if args.final_train_rows is None:
        args.final_train_rows = int(os.environ.get("ER_FINAL_TRAIN_ROWS", "40000"))
    return args


def is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(128).startswith(LFS_POINTER_PREFIX)
    except OSError:
        return False


def install_git_lfs_if_needed() -> None:
    if not shutil.which("git-lfs"):
        if not shutil.which("apt-get") or getattr(os, "geteuid", lambda: 1)() != 0:
            raise RuntimeError(
                "Git LFS pointer files were found, but git-lfs is unavailable. "
                "Install Git LFS and run `git lfs pull` in the cloned repository."
            )
        info("Git LFS is missing; installing it with apt so the tracked dataset can be checked out.")
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "git-lfs"], check=True)


def hydrate_lfs_dataset() -> None:
    dataset_dir = ROOT / "student_resource/dataset"
    pointers = [path for path in dataset_dir.rglob("*.tsv") if is_lfs_pointer(path)]
    archive_is_pointer = is_lfs_pointer(DATA_ARCHIVE)
    if archive_is_pointer:
        pointers.append(DATA_ARCHIVE)
    if pointers:
        install_git_lfs_if_needed()
        info("Hydrating the Git LFS challenge archive (about 1.09 GB).")
        subprocess.run(["git", "lfs", "install", "--local"], cwd=ROOT, check=True)
        subprocess.run(
            ["git", "lfs", "pull", "--include=" + DATA_ARCHIVE.name],
            cwd=ROOT,
            check=True,
        )
        still_pointers = [path for path in pointers if is_lfs_pointer(path)]
        if still_pointers:
            raise RuntimeError(f"Git LFS did not hydrate dataset file: {still_pointers[0]}")

    if has_files(dataset_dir / "train", TRAIN_FILES) and has_files(dataset_dir / "test", TEST_FILES):
        return
    if not DATA_ARCHIVE.is_file():
        raise RuntimeError(f"Dataset TSVs are missing and the Git LFS archive is unavailable: {DATA_ARCHIVE}")

    info("Extracting the challenge TSVs from the repository archive.")
    expected_paths = {
        *(f"student_resource/dataset/train/{name}" for name in TRAIN_FILES),
        *(f"student_resource/dataset/test/{name}" for name in TEST_FILES),
    }
    found: set[str] = set()
    with zipfile.ZipFile(DATA_ARCHIVE) as archive:
        for member in archive.infolist():
            if member.filename not in expected_paths:
                continue
            destination = ROOT / member.filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            found.add(member.filename)
    missing = expected_paths - found
    if missing:
        raise RuntimeError(f"Challenge archive is missing required TSV: {sorted(missing)[0]}")


def has_files(directory: Path, names: tuple[str, ...]) -> bool:
    return directory.is_dir() and all(
        (directory / name).is_file() and not is_lfs_pointer(directory / name)
        for name in names
    )


def count_tsv_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def validate_full_inference_outputs(test_dir: Path,
                                    output_dir: Path,
                                    expected_rows: int,
                                    max_candidates: int) -> None:
    """Check complete ordered test coverage and the configured per-row candidate cap."""
    test_path = test_dir / "test_source1.tsv"
    matching_path = output_dir / "matching_results.tsv"
    candidates_path = output_dir / "candidate_pairs.tsv"
    for path in (test_path, matching_path, candidates_path):
        if not path.is_file():
            raise RuntimeError(f"Full inference output is missing: {path}")

    with test_path.open(encoding="utf-8", newline="") as test_handle, \
            matching_path.open(encoding="utf-8", newline="") as matching_handle, \
            candidates_path.open(encoding="utf-8", newline="") as candidates_handle:
        test_rows = csv.DictReader(test_handle, delimiter="\t")
        matching_rows = csv.DictReader(matching_handle, delimiter="\t")
        candidate_rows = csv.DictReader(candidates_handle, delimiter="\t")
        if "entity_id" not in (test_rows.fieldnames or []):
            raise RuntimeError(f"Missing entity_id column in {test_path}")
        expected_headers = {
            str(matching_path): ["source1_entity_id", "matched_entity_ids"],
            str(candidates_path): ["source1_entity_id", "candidate_entity_ids"],
        }
        if matching_rows.fieldnames != expected_headers[str(matching_path)]:
            raise RuntimeError(f"Unexpected submission header in {matching_path}")
        if candidate_rows.fieldnames != expected_headers[str(candidates_path)]:
            raise RuntimeError(f"Unexpected submission header in {candidates_path}")

        count = 0
        while True:
            expected = next(test_rows, None)
            matching = next(matching_rows, None)
            candidates = next(candidate_rows, None)
            if expected is None and matching is None and candidates is None:
                break
            count += 1
            if expected is None or matching is None or candidates is None:
                raise RuntimeError(
                    "Submission row coverage differs from test Source-1; "
                    f"the mismatch starts at data row {count}."
                )
            entity_id = expected["entity_id"]
            if (matching["source1_entity_id"] != entity_id
                    or candidates["source1_entity_id"] != entity_id):
                raise RuntimeError(
                    f"Submission row {count} does not match test Source-1 ID {entity_id!r}."
                )
            candidate_ids = [value for value in candidates["candidate_entity_ids"].split(",") if value]
            matched_ids = [value for value in matching["matched_entity_ids"].split(",") if value]
            if max_candidates > 0 and len(candidate_ids) > max_candidates:
                raise RuntimeError(
                    f"Candidate cap exceeded for {entity_id}: {len(candidate_ids)} > {max_candidates}."
                )
            if not set(matched_ids).issubset(candidate_ids):
                raise RuntimeError(f"Matched IDs are not present in candidate list for {entity_id}.")

    if count != expected_rows:
        raise RuntimeError(
            f"Submission covers {count:,} Source-1 rows; expected {expected_rows:,}."
        )
    info(f"Full-output check passed: {count:,} test entities; candidate cap="
         f"{max_candidates if max_candidates else 'unlimited'}.")


def discover_data(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.train_dir or args.test_dir:
        if not args.train_dir or not args.test_dir:
            raise RuntimeError("Pass both --train-dir and --test-dir when overriding data paths.")
        train_dir, test_dir = Path(args.train_dir).expanduser(), Path(args.test_dir).expanduser()
        if not has_files(train_dir, TRAIN_FILES) or not has_files(test_dir, TEST_FILES):
            raise RuntimeError("The explicit train/test directories do not contain all challenge TSVs.")
        return train_dir.resolve(), test_dir.resolve()

    roots: list[Path] = []
    if args.data_root:
        roots.append(Path(args.data_root).expanduser())
    for value in (os.environ.get("ER_DATASET_DIR"), os.environ.get("DATASET_DIR")):
        if value:
            roots.append(Path(value).expanduser())
    roots.extend((ROOT / "student_resource/dataset", ROOT / "dataset"))
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        roots.append(kaggle_input)

    seen: set[Path] = set()
    for base in roots:
        base = base.resolve()
        if base in seen or not base.exists():
            continue
        seen.add(base)
        if has_files(base / "train", TRAIN_FILES) and has_files(base / "test", TEST_FILES):
            return base / "train", base / "test"
        train_candidates = sorted(base.rglob("train_source1.tsv"))
        test_candidates = sorted(base.rglob("test_source1.tsv"))
        for train_source1 in train_candidates:
            train_dir = train_source1.parent
            for test_source1 in test_candidates:
                test_dir = test_source1.parent
                if has_files(train_dir, TRAIN_FILES) and has_files(test_dir, TEST_FILES):
                    return train_dir, test_dir

    raise RuntimeError(
        "Could not locate all challenge data files. Expected student_resource/dataset/{train,test}, "
        "dataset/{train,test}, a Kaggle input mount, or ER_DATASET_DIR."
    )


def pinned_requirements() -> list[tuple[str, str]]:
    parsed = []
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        parsed.append((name.strip(), version.strip()))
    return parsed


def ensure_dependencies(skip_install: bool) -> None:
    if skip_install:
        return
    expected = pinned_requirements()
    missing_or_mismatched = []
    for name, version in expected:
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            missing_or_mismatched.append((name, installed, version))
    if not missing_or_mismatched:
        info("Pinned Python dependencies are already installed.")
        return
    info("Installing the repository's pinned Python dependencies (first run may take a few minutes).")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(REQUIREMENTS)],
        cwd=ROOT,
        check=True,
    )


def safe_team_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    if not result:
        raise RuntimeError("Team name must contain at least one letter or number.")
    return result


def selected_oof_metrics(report: Path) -> dict[str, str]:
    if not report.is_file():
        return {}
    with report.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    selected = next((row for row in rows if row.get("selected_by_oof_cv", "").lower() == "true"), rows[0])
    return selected


def filled_methodology(team: str, output_dir: Path, run_config: dict[str, object]) -> str:
    if not TEMPLATE.is_file():
        raise RuntimeError(f"Challenge methodology template not found: {TEMPLATE}")
    content = TEMPLATE.read_text(encoding="utf-8")
    content = content.replace("[Enter team name]", team)
    content = content.replace("[Enter team members]", os.environ.get("ER_TEAM_MEMBERS", "Not specified"))
    content = content.replace("[Enter submission date]", dt.date.today().isoformat())
    if not run_config["cross_source_target_graph"]:
        content = content.replace(
            "Country-aware retrieval combines name, address, postal, house-number, and locality signals, while the pair model also receives direct S2↔S3 corroboration features.",
            "Country-aware retrieval combines name, address, postal, house-number, and locality signals. "
            "The memory-bounded Kaggle run disables the optional direct S2↔S3 target graph; graph features are zero.",
        )
        content = content.replace(
            "3. Build pair features for name, address, geography, contradictions, retrieval context, and direct target-source graph evidence.",
            "3. Build pair features for name, address, geography, contradictions, and retrieval context. "
            "The target-source graph is disabled in this run to reduce peak memory.",
        )
        content = content.replace(
            "- **Other:** country compatibility, source identity, candidate rank and retrieval-channel metadata, and direct S2↔S3 support.",
            "- **Other:** country compatibility, source identity, candidate rank and retrieval-channel metadata. "
            "Direct S2↔S3 graph support is disabled in this run.",
        )
    report = selected_oof_metrics(output_dir / "oof_model_comparison.csv")
    placeholder = (
        "No full-data score is asserted in this document. After running the pipeline on the challenge training data, "
        "copy the reserved-fold entity, singleton, multi-match, and source-specific metrics from `oof_model_comparison.csv` "
        "into this section. Use `oof_validation_errors.tsv` and `oof_error_buckets.csv` to summarize observed false positives, "
        "false negatives, and retrieval misses. Do not substitute synthetic-test results for challenge validation results."
    )
    if report:
        def metric(name: str, digits: int = 4) -> str:
            try:
                return f"{float(report[name]):.{digits}f}"
            except (KeyError, TypeError, ValueError):
                return "not reported"

        result_section = (
            "Validation metrics below are from this run's sampled, entity-level OOF evaluation; they are not a "
            "test-set leaderboard score. The OOF query sample contains "
            f"{run_config['oof_source1_rows']:,} of {run_config['train_source1_rows']:,} labeled Source-1 rows, "
            f"and the final pair model is fit on {run_config['final_train_rows']:,} labeled Source-1 rows. "
            "All test Source-1 rows and all test Source-2/3 records are used for final inference.\n\n"
            f"- Selected pair model: `{report.get('model_type', 'not reported')}`.\n"
            f"- OOF entity macro F0.5: {metric('oof_cv_entity_f0_5')} "
            f"(fold range {metric('oof_cv_f0_5_min')}–{metric('oof_cv_f0_5_max')}; "
            f"standard deviation {metric('oof_cv_f0_5_std')}).\n"
            f"- Reserved fold entity F0.5: {metric('fold0_entity_f0_5')}.\n"
            f"- Reserved fold candidate blocking recall: {metric('fold0_blocking_candidate_recall')}; "
            f"final match recall: {metric('oof_match_recall')}.\n\n"
            "These validation values describe the labeled training sample only. The output TSVs cover every test "
            "Source-1 entity; test labels are unavailable, so no test F0.5 is reported. Full diagnostics are written "
            "under `output/` by the runner."
        )
        content = content.replace(placeholder, result_section)
    reproduce_command = (
        "python code/business_entity_resolution/src/entity_resolution_pipeline.py \\\n"
        "  --train-dir dataset/train \\\n"
        "  --test-dir dataset/test \\\n"
        "  --output-dir output \\\n"
        f"  --sample-train-rows {run_config['oof_source1_rows']} \\\n"
        f"  --final-train-rows {run_config['final_train_rows']} \\\n"
        f"  --cv-folds {run_config['cv_folds']} \\\n"
        f"  --model {run_config['model']} \\\n"
        f"  --negatives-per-positive {run_config['negatives_per_positive']} \\\n"
        f"  --random-negatives {run_config['random_negatives']} \\\n"
        f"  --adversarial-negatives-per-entity {run_config['hard_negatives']} \\\n"
        f"  --max-block-frequency {run_config['max_block_frequency']} \\\n"
        f"  --candidate-channel-limit-multiplier {run_config['channel_limit_multiplier']} \\\n"
        f"  --name-ngram-limit {run_config['name_ngram_limit']} \\\n"
        f"  --retrieval-context-mode {run_config['retrieval_context_mode']} \\\n"
        f"  --max-test-candidates {run_config['max_test_candidates']} \\\n"
        f"  --test-batch-size {run_config['test_batch_size']} \\\n"
        "  --seed 42 --validate-submission"
    )
    if not run_config["cross_source_target_graph"]:
        reproduce_command += " \\\n  --disable-target-graph"
    if run_config["semantic_retrieval"]:
        reproduce_command += (
            " \\\n  --semantic-retrieval"
            f" --semantic-index-max-targets {run_config['semantic_index_max_targets']}"
            f" --semantic-top-k {run_config['semantic_top_k']}"
            f" --semantic-min-similarity {run_config['semantic_min_similarity']}"
        )
    content = re.sub(
        r"python code/business_entity_resolution/src/entity_resolution_pipeline\.py \\\n"
        r"\s+--train-dir .*?--output-dir output",
        lambda _match: reproduce_command,
        content,
        count=1,
        flags=re.DOTALL,
    )
    return content


def create_submission_zip(team: str, output_dir: Path, run_config: dict[str, object]) -> Path:
    required_outputs = ("matching_results.tsv", "candidate_pairs.tsv")
    for name in required_outputs:
        path = output_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Required submission output is missing or empty: {path}")

    code_dir = ROOT / "code/business_entity_resolution"
    source_files = sorted((code_dir / "src").rglob("*.py"))
    if not source_files:
        raise RuntimeError(f"No source files found under {code_dir / 'src'}")
    archive = ROOT / f"{team}_submission.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="submission_", suffix=".md", delete=False) as handle:
        temporary_methodology = Path(handle.name)
    try:
        temporary_methodology.write_text(
            filled_methodology(team, output_dir, run_config), encoding="utf-8"
        )
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
            for name in required_outputs:
                bundle.write(output_dir / name, f"output/{name}")
            for path in source_files:
                bundle.write(path, path.relative_to(ROOT).as_posix())
            bundle.write(code_dir / "README.md", "code/business_entity_resolution/README.md")
            bundle.write(REQUIREMENTS, "code/business_entity_resolution/requirements.txt")
            bundle.write(temporary_methodology, "Documentation_template.md")
    finally:
        temporary_methodology.unlink(missing_ok=True)
    return archive


def main() -> None:
    args = parse_args()
    if (args.sample_train_rows < 0 or args.retrieval_sample_rows < 0
            or args.final_train_rows < 0 or args.cv_folds < 3
            or args.negatives_per_positive < 0 or args.random_negatives < 0
            or args.adversarial_negatives_per_entity < 0):
        raise SystemExit("Row counts and negative counts cannot be negative; --cv-folds must be at least 3.")
    if args.max_block_frequency < 0 or args.max_test_candidates < 0:
        raise SystemExit("Block frequency and candidate caps cannot be negative.")
    if args.name_ngram_limit < 0 or args.candidate_channel_limit_multiplier <= 0:
        raise SystemExit("Name n-gram limit cannot be negative and channel multiplier must be positive.")
    if args.test_batch_size < 1:
        raise SystemExit("--test-batch-size must be positive.")
    if args.semantic_index_max_targets < 1 or args.semantic_top_k < 1:
        raise SystemExit("Semantic index size and top-K must be positive.")
    if not 0.0 <= args.semantic_min_similarity <= 1.0:
        raise SystemExit("--semantic-min-similarity must be between 0 and 1.")
    team = safe_team_name(args.team_name)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prefer the Kaggle input mount (or explicit paths). Hydrate the repository's
    # optional local archive only when no usable mounted dataset can be found.
    try:
        train_dir, test_dir = discover_data(args)
    except RuntimeError:
        if args.train_dir or args.test_dir or args.data_root:
            raise
        hydrate_lfs_dataset()
        train_dir, test_dir = discover_data(args)
    info(f"Training files: {train_dir}")
    info(f"Test files: {test_dir}")
    train_source1_rows = count_tsv_rows(train_dir / "train_source1.tsv")
    test_source1_rows = count_tsv_rows(test_dir / "test_source1.tsv")
    ensure_dependencies(args.no_install_dependencies)

    thread_count = max(1, int(os.environ.get("ER_THREADS", "4")))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, str(thread_count))

    workflow_name = (
        "retrieval-only" if args.retrieval_only else
        "calibration-only" if args.calibration_only else
        "skip-oof" if args.skip_oof else
        "full-kaggle-run" if args.full_kaggle_run else "full"
    )
    run_config: dict[str, object] = {
        "workflow": workflow_name,
        "team_name": team,
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "output_dir": str(output_dir),
        "train_source1_rows": train_source1_rows,
        "test_source1_rows": test_source1_rows,
        "oof_source1_rows": (train_source1_rows if args.sample_train_rows == 0
                              else min(args.sample_train_rows, train_source1_rows)),
        "retrieval_source1_rows": min(args.retrieval_sample_rows, train_source1_rows),
        "final_train_rows": (train_source1_rows if args.final_train_rows == 0
                             else min(args.final_train_rows, train_source1_rows)),
        "cv_folds": args.cv_folds,
        "model": args.model,
        "negatives_per_positive": args.negatives_per_positive,
        "random_negatives": args.random_negatives,
        "hard_negatives": args.adversarial_negatives_per_entity,
        "max_block_frequency": args.max_block_frequency,
        "channel_limit_multiplier": args.candidate_channel_limit_multiplier,
        "name_ngram_limit": args.name_ngram_limit,
        "retrieval_context_mode": args.retrieval_context_mode,
        "max_test_candidates": args.max_test_candidates,
        "test_batch_size": args.test_batch_size,
        "cross_source_target_graph": not args.disable_target_graph,
        "semantic_retrieval": args.semantic_retrieval,
        "semantic_index_max_targets": args.semantic_index_max_targets,
        "semantic_top_k": args.semantic_top_k,
        "semantic_min_similarity": args.semantic_min_similarity,
        "block_index_cache_dir": args.block_index_cache_dir,
        "calibration_artifact": args.calibration_artifact,
        "all_test_source1_rows_scored": workflow_name in {"full", "skip-oof", "full-kaggle-run"},
        "seed": 42,
        "status": "running",
        "completed_stages": [],
        "stage_runtime_seconds": {},
    }
    def pipeline_command(stage: str, sample_rows: int, rebuild_index: bool) -> list[str]:
        command = [
            sys.executable, "-u", str(PIPELINE),
            "--train-dir", str(train_dir), "--test-dir", str(test_dir),
            "--output-dir", str(output_dir),
            "--sample-train-rows", str(sample_rows),
            "--final-train-rows", str(args.final_train_rows),
            "--cv-folds", str(args.cv_folds),
            "--model", args.model,
            "--negatives-per-positive", str(args.negatives_per_positive),
            "--random-negatives", str(args.random_negatives),
            "--adversarial-negatives-per-entity", str(args.adversarial_negatives_per_entity),
            "--max-block-frequency", str(args.max_block_frequency),
            "--candidate-channel-limit-multiplier", str(args.candidate_channel_limit_multiplier),
            "--name-ngram-limit", str(args.name_ngram_limit),
            "--retrieval-context-mode", args.retrieval_context_mode,
            "--max-test-candidates", str(args.max_test_candidates),
            "--test-batch-size", str(args.test_batch_size),
            "--seed", "42",
        ]
        if args.block_index_cache_dir:
            command.extend(("--block-index-cache-dir",
                            str(Path(args.block_index_cache_dir).expanduser().resolve())))
        if rebuild_index:
            command.append("--rebuild-block-index")
        if args.calibration_artifact:
            command.extend(("--calibration-artifact",
                            str(Path(args.calibration_artifact).expanduser().resolve())))
        if stage == "retrieval-only":
            command.append("--retrieval-only")
        elif stage == "calibration-only":
            command.append("--calibration-only")
        elif stage == "skip-oof":
            command.extend(("--skip-oof", "--validate-submission"))
        else:
            command.append("--validate-submission")
        if args.disable_target_graph:
            command.append("--disable-target-graph")
        if args.semantic_retrieval:
            command.extend((
                "--semantic-retrieval",
                "--semantic-index-max-targets", str(args.semantic_index_max_targets),
                "--semantic-top-k", str(args.semantic_top_k),
                "--semantic-min-similarity", str(args.semantic_min_similarity),
            ))
        return command

    if args.full_kaggle_run:
        stages = [
            ("retrieval-only", args.retrieval_sample_rows),
            ("calibration-only", args.sample_train_rows),
            ("skip-oof", args.sample_train_rows),
        ]
    elif args.retrieval_only:
        stages = [("retrieval-only", args.sample_train_rows)]
    elif args.calibration_only:
        stages = [("calibration-only", args.sample_train_rows)]
    elif args.skip_oof:
        stages = [("skip-oof", args.sample_train_rows)]
    else:
        stages = [("full", args.sample_train_rows)]

    summary_path = output_dir / "kaggle_run_summary.json"
    summary_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    info(
        f"Starting {workflow_name}: OOF/retrieval sample={run_config['oof_source1_rows']:,}, "
        f"retrieval sample={run_config['retrieval_source1_rows']:,}, "
        f"final training sample={run_config['final_train_rows']:,}, folds={args.cv_folds}; "
        f"test S1 rows={test_source1_rows:,}."
    )
    # The pipeline also writes a convenience `output/` relative to its CWD. Keep
    # that secondary path beside the requested output directory instead of
    # overwriting an unrelated repository-level output during custom runs.
    for stage_index, (stage, sample_rows) in enumerate(stages):
        info(f"Stage {stage_index + 1}/{len(stages)}: {stage}; sample={sample_rows:,}.")
        stage_started = time.monotonic()
        try:
            subprocess.run(
                pipeline_command(stage, sample_rows,
                                 rebuild_index=args.rebuild_block_index and stage_index == 0),
                cwd=output_dir.parent,
                check=True,
            )
        except subprocess.CalledProcessError:
            run_config["stage_runtime_seconds"][stage] = time.monotonic() - stage_started
            run_config["status"] = "failed"
            run_config["failed_stage"] = stage
            summary_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
            raise
        stage_seconds = time.monotonic() - stage_started
        run_config["stage_runtime_seconds"][stage] = stage_seconds
        info(f"Completed {stage} in {stage_seconds / 3600:.2f} hours.")
        run_config["completed_stages"].append(stage)
        summary_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    if args.retrieval_only:
        run_config["status"] = "completed"
        summary_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
        info(f"Retrieval diagnostics: {output_dir / 'candidate_retrieval_diagnostics.csv'}")
        info(f"Run summary: {summary_path}")
        return
    if args.calibration_only:
        run_config["status"] = "completed"
        summary_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
        calibration_path = (Path(args.calibration_artifact).expanduser().resolve()
                            if args.calibration_artifact else output_dir / "calibration.pkl")
        info(f"Calibration artifact: {calibration_path}")
        info(f"Run summary: {summary_path}")
        return

    try:
        validate_full_inference_outputs(
            test_dir, output_dir, test_source1_rows, args.max_test_candidates
        )
        archive = create_submission_zip(team, output_dir, run_config)
    except Exception:
        run_config["status"] = "failed"
        run_config["failed_stage"] = "output-validation-or-packaging"
        summary_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
        raise
    run_config["status"] = "completed"
    summary_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    info(f"Challenge outputs: {output_dir / 'matching_results.tsv'} and {output_dir / 'candidate_pairs.tsv'}")
    info(f"Final package: {archive}")

    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir() and kaggle_working.resolve() != ROOT.resolve():
        shutil.copy2(archive, kaggle_working / archive.name)
        info(f"Kaggle-downloadable copy: {kaggle_working / archive.name}")


if __name__ == "__main__":
    main()
