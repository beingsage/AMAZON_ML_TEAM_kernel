#!/usr/bin/env python3
"""One-command Kaggle runner for the Amazon ML Challenge 2026."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
import tempfile
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
        description="Install dependencies, run full test inference, validate, and build the submission ZIP."
    )
    parser.add_argument("--data-root", help="Optional directory containing train/ and test/.")
    parser.add_argument("--train-dir", help="Optional explicit directory containing train TSVs.")
    parser.add_argument("--test-dir", help="Optional explicit directory containing test TSVs.")
    parser.add_argument("--output-dir", default=os.environ.get("ER_OUTPUT_DIR", str(ROOT / "output")))
    parser.add_argument("--team-name", default=os.environ.get("ER_TEAM_NAME", "AMAZON_ML_TEAM"))
    parser.add_argument("--sample-train-rows", type=int,
                        default=int(os.environ.get("ER_SAMPLE_TRAIN_ROWS", "5000")),
                        help="OOF model-selection rows; final test inference always covers every test S1 row.")
    parser.add_argument("--final-train-rows", type=int,
                        default=int(os.environ.get("ER_FINAL_TRAIN_ROWS", "25000")),
                        help="Labeled Source-1 rows used for the final pair-model fit; 0 uses all rows.")
    parser.add_argument("--cv-folds", type=int, default=int(os.environ.get("ER_CV_FOLDS", "3")))
    parser.add_argument("--model", choices=("logistic", "lightgbm", "compare", "ensemble"),
                        default=os.environ.get("ER_MODEL", "lightgbm"))
    parser.add_argument("--no-install-dependencies", action="store_true",
                        help="Skip checking/installing the pinned project requirements.")
    return parser.parse_args()


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
    return directory.is_dir() and all((directory / name).is_file() for name in names)


def count_tsv_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


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
        "  --seed 42 --validate-submission"
    )
    if not run_config["cross_source_target_graph"]:
        reproduce_command += " \\\n  --disable-target-graph"
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
    if args.sample_train_rows < 0 or args.final_train_rows < 0 or args.cv_folds < 3:
        raise SystemExit("Training row counts cannot be negative, and --cv-folds must be at least 3.")
    team = safe_team_name(args.team_name)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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

    negatives_per_positive = int(os.environ.get("ER_NEGATIVES_PER_POSITIVE", "5"))
    random_negatives = int(os.environ.get("ER_RANDOM_NEGATIVES", "2"))
    hard_negatives = int(os.environ.get("ER_HARD_NEGATIVES", "1"))
    max_block_frequency = int(os.environ.get("ER_MAX_BLOCK_FREQUENCY", "5000"))
    channel_limit_multiplier = float(os.environ.get("ER_CHANNEL_LIMIT_MULTIPLIER", "0.1"))
    disable_target_graph = os.environ.get("ER_DISABLE_TARGET_GRAPH", "1").strip().lower() not in {
        "0", "false", "no"
    }
    run_config: dict[str, object] = {
        "team_name": team,
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "output_dir": str(output_dir),
        "train_source1_rows": train_source1_rows,
        "test_source1_rows": test_source1_rows,
        "oof_source1_rows": (train_source1_rows if args.sample_train_rows == 0
                              else min(args.sample_train_rows, train_source1_rows)),
        "final_train_rows": (train_source1_rows if args.final_train_rows == 0
                             else min(args.final_train_rows, train_source1_rows)),
        "cv_folds": args.cv_folds,
        "model": args.model,
        "negatives_per_positive": negatives_per_positive,
        "random_negatives": random_negatives,
        "hard_negatives": hard_negatives,
        "max_block_frequency": max_block_frequency,
        "channel_limit_multiplier": channel_limit_multiplier,
        "cross_source_target_graph": not disable_target_graph,
        "all_test_source1_rows_scored": True,
        "seed": 42,
    }
    command = [
        sys.executable, "-u", str(PIPELINE),
        "--train-dir", str(train_dir), "--test-dir", str(test_dir),
        "--output-dir", str(output_dir),
        "--sample-train-rows", str(args.sample_train_rows),
        "--final-train-rows", str(args.final_train_rows),
        "--cv-folds", str(args.cv_folds),
        "--model", args.model,
        "--negatives-per-positive", str(negatives_per_positive),
        "--random-negatives", str(random_negatives),
        "--adversarial-negatives-per-entity", str(hard_negatives),
        "--max-block-frequency", str(max_block_frequency),
        "--candidate-channel-limit-multiplier", str(channel_limit_multiplier),
        "--seed", "42", "--validate-submission",
    ]
    if disable_target_graph:
        command.append("--disable-target-graph")
    info(
        f"Starting pipeline: OOF sample={run_config['oof_source1_rows']:,}, final training sample="
        f"{run_config['final_train_rows']:,}, folds={args.cv_folds}; "
        f"test inference covers all {test_source1_rows:,} test Source-1 rows."
    )
    # The pipeline also writes a convenience `output/` relative to its CWD. Keep
    # that secondary path beside the requested output directory instead of
    # overwriting an unrelated repository-level output during custom runs.
    subprocess.run(command, cwd=output_dir.parent, check=True)

    archive = create_submission_zip(team, output_dir, run_config)
    summary_path = output_dir / "kaggle_run_summary.json"
    import json
    summary_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    info(f"Challenge outputs: {output_dir / 'matching_results.tsv'} and {output_dir / 'candidate_pairs.tsv'}")
    info(f"Final package: {archive}")

    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir() and kaggle_working.resolve() != ROOT.resolve():
        shutil.copy2(archive, kaggle_working / archive.name)
        info(f"Kaggle-downloadable copy: {kaggle_working / archive.name}")


if __name__ == "__main__":
    main()
