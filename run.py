#!/usr/bin/env python3
"""Kaggle pipeline alias with the older retrieval-ablation utility retained."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


RUNS_DIR = Path("runs")


PIPELINE_SWITCHES = {
    "--retrieval-only", "--calibration-only", "--skip-oof", "--train-dir",
    "--full-kaggle-run", "--test-dir", "--calibration-artifact", "--max-test-candidates",
    "--test-batch-size", "--block-index-cache-dir", "--rebuild-block-index",
    "--retrieval-sample-rows",
    "--data-root", "--output-dir", "--team-name", "--sample-train-rows",
    "--final-train-rows", "--cv-folds", "--model", "--name-ngram-limit",
    "--candidate-channel-limit-multiplier", "--retrieval-context-mode",
    "--disable-target-graph", "--enable-target-graph", "--semantic-retrieval",
    "--negatives-per-positive", "--random-negatives",
    "--adversarial-negatives-per-entity",
    "--no-install-dependencies", "--semantic-index-max-targets",
    "--semantic-top-k", "--semantic-min-similarity",
}


def load_legacy_dependencies() -> None:
    """Load plotting and retrieval dependencies only for the old ablation mode."""
    import pandas as pd

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    sys.path.insert(0, str(Path(__file__).with_name("code")))
    from business_entity_resolution.src.entity_resolution_pipeline import (
        load_source,
        parse_truth,
        build_source_lookup,
        build_block_index,
        run_retrieval_ablation,
        generate_candidates,
    )

    globals().update({
        "plt": plt,
        "pd": pd,
        "load_source": load_source,
        "parse_truth": parse_truth,
        "build_source_lookup": build_source_lookup,
        "build_block_index": build_block_index,
        "run_retrieval_ablation": run_retrieval_ablation,
        "generate_candidates": generate_candidates,
    })


def make_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def save_run_artifacts(run_id: str, args: argparse.Namespace, df: pd.DataFrame) -> Dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # save ablation table
    csv_path = run_dir / "ablation.csv"
    df.to_csv(csv_path, index=False)
    # summary metrics: pick row with max candidate_recall
    best_row = df.loc[df["candidate_recall"].idxmax()].to_dict() if not df.empty else {}
    summary = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "args": vars(args),
        "best_metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in best_row.items()},
        "ablation_csv": str(csv_path),
    }
    with open(run_dir / "meta.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def aggregate_runs() -> pd.DataFrame:
    rows = []
    for meta_file in sorted(RUNS_DIR.glob("*/meta.json")):
        try:
            with open(meta_file, "r") as fh:
                meta = json.load(fh)
            bm = meta.get("best_metrics", {})
            rows.append({
                "run_id": meta.get("run_id"),
                "timestamp": meta.get("timestamp"),
                "candidate_recall": float(bm.get("candidate_recall", 0.0)),
                "candidate_mean": float(bm.get("candidate_mean", 0.0)),
                "total_seconds": float(bm.get("total_seconds", 0.0)),
            })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["run_id", "timestamp", "candidate_recall", "candidate_mean", "total_seconds"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    return df


def plot_trajectory(agg: pd.DataFrame, out: Path) -> None:
    if agg.empty or plt is None:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(agg["timestamp"], agg["candidate_recall"], marker="o", label="candidate_recall")
    plt.plot(agg["timestamp"], agg["candidate_mean"], marker="x", label="candidate_mean")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.title("Metrics Trajectory Across Runs")
    plt.legend()
    plt.grid(True)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def run_single_retrieval(s1: pd.DataFrame,
                         truth: dict,
                         s2: pd.DataFrame,
                         lookup: pd.DataFrame,
                         max_block_frequency: int,
                         current_ngram_limit: int,
                         current_channel_multiplier: float,
                         current_context_mode: str,
                         output_path: Path,
                         limits=(50, 100, 200, 500, 1000)) -> pd.DataFrame:
    """Run a single retrieval configuration (faster, for testing)."""
    import time
    from collections import Counter

    started = time.perf_counter()
    index = build_block_index(
        s2,
        max_block_frequency=max_block_frequency,
        name_ngram_limit=current_ngram_limit,
        channel_limit_multiplier=current_channel_multiplier,
        retrieval_context_mode=current_context_mode,
        build_graph=False,
    )
    index_build_seconds = time.perf_counter() - started
    query_started = time.perf_counter()
    size_histogram: Counter[int] = Counter()
    recall_hits: Counter[int] = Counter()
    total_truth_pairs = 0
    found_truth_pairs = 0
    query_count = 0
    for row in s1.itertuples(index=False):
        values = row._asdict()
        entity_id = str(values.get("entity_id", ""))
        candidates = generate_candidates(values, index, lookup) if 'generate_candidates' in globals() else []
        true_ids = truth.get(entity_id, set())
        total_truth_pairs += len(true_ids)
        found_truth_pairs += len(set(candidates) & true_ids)
        size_histogram[len(candidates)] += 1
        for limit in limits:
            recall_hits[limit] += len(set(candidates[:limit]) & true_ids)
        query_count += 1
    query_seconds = time.perf_counter() - query_started
    row = {
        "name_ngram_limit": "single",
        "channel_limit_multiplier": float(current_channel_multiplier),
        "retrieval_context_mode": current_context_mode,
        "entity_count": float(query_count),
        "candidate_mean": (
            sum(size * count for size, count in size_histogram.items()) / query_count
            if query_count else 0.0
        ),
        "candidate_p50": float(pd.Series(list(size_histogram.elements())).quantile(0.50)) if size_histogram else 0.0,
        "candidate_p95": float(pd.Series(list(size_histogram.elements())).quantile(0.95)) if size_histogram else 0.0,
        "candidate_p99": float(pd.Series(list(size_histogram.elements())).quantile(0.99)) if size_histogram else 0.0,
        "candidate_max": float(max(size_histogram, default=0)),
        "candidate_recall": found_truth_pairs / total_truth_pairs if total_truth_pairs else 1.0,
        "index_build_seconds": index_build_seconds,
        "query_seconds": query_seconds,
        "total_seconds": index_build_seconds + query_seconds,
    }
    for limit in limits:
        row[f"recall_at_{limit}"] = recall_hits[limit] / total_truth_pairs if total_truth_pairs else 1.0
    df = pd.DataFrame([row])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def update_readme(agg: pd.DataFrame, image_path: Path) -> None:
    readme = Path("README.md")
    marker_start = "<!-- RUNS_METRICS_START -->"
    marker_end = "<!-- RUNS_METRICS_END -->"
    table_md = "\n"
    if not agg.empty:
        table_md += "| run_id | timestamp | candidate_recall | candidate_mean | total_seconds |\n"
        table_md += "|---|---:|---:|---:|---:|\n"
        for _, row in agg.iterrows():
            table_md += f"| {row['run_id']} | {row['timestamp'].isoformat()} | {row['candidate_recall']:.4f} | {row['candidate_mean']:.1f} | {row['total_seconds']:.1f} |\n"
    content = f"\n{marker_start}\n\n![runs metrics]({image_path})\n\n{table_md}\n{marker_end}\n"
    if readme.exists():
        text = readme.read_text()
        if marker_start in text and marker_end in text:
            pre, rest = text.split(marker_start, 1)
            _, post = rest.split(marker_end, 1)
            new = pre + content + post
            readme.write_text(new)
            return
        # append at end
        with open(readme, "a") as fh:
            fh.write("\n" + content)
    else:
        readme.write_text("# Runs\n" + content)


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    kaggle_default = not raw_args and Path("/kaggle/input").is_dir()
    full_mode = False
    if "--mode" in raw_args:
        mode_position = raw_args.index("--mode")
        full_mode = mode_position + 1 < len(raw_args) and raw_args[mode_position + 1] == "full"
    if ("--pipeline" in raw_args or kaggle_default or full_mode
            or any(flag in raw_args for flag in PIPELINE_SWITCHES)):
        forwarded = [arg for arg in raw_args if arg != "--pipeline"]
        if full_mode:
            mode_position = forwarded.index("--mode")
            del forwarded[mode_position:mode_position + 2]
        explicit_stage = any(flag in forwarded for flag in (
            "--retrieval-only", "--calibration-only", "--skip-oof", "--full-kaggle-run"
        ))
        if not explicit_stage and (kaggle_default or full_mode or "--pipeline" in raw_args):
            forwarded.append("--full-kaggle-run")
        runner = Path(__file__).with_name("kaggle_runner.py")
        return subprocess.run([sys.executable, str(runner), *forwarded], check=False).returncode

    load_legacy_dependencies()
    RUNS_DIR.mkdir(exist_ok=True)
    parser = argparse.ArgumentParser(
        description="Use --pipeline to forward Kaggle workflow options to kaggle_runner.py."
    )
    parser.add_argument("--pipeline", action="store_true",
                        help="Forward the remaining arguments to kaggle_runner.py.")
    parser.add_argument("--source1", default="student_resource/dataset/train/train_source1.tsv")
    parser.add_argument("--source2_3", default="student_resource/dataset/train/train_source2.tsv")
    parser.add_argument("--truth", default="student_resource/dataset/train/train_ground_truth.tsv")
    parser.add_argument("--max-block-frequency", type=int, default=10000)
    parser.add_argument("--ngram-limit", type=int, default=6)
    parser.add_argument("--channel-multiplier", type=float, default=1.0)
    parser.add_argument("--context-mode", choices=["once", "per_channel"], default="once")
    parser.add_argument("--mode", choices=["test", "sweep"], default=None,
                        help="'test' runs one small retrieval profile; 'sweep' runs the legacy profile sweep.")
    parser.add_argument("--sample-size", type=int, default=100,
                        help="When in test mode, number of rows to sample from source1 and source2.")
    args = parser.parse_args(argv)

    run_id = make_run_id()
    print(f"Run id: {run_id}")

    mode = args.mode
    if mode is None:
        try:
            choice = input("Choose run mode ('test' for a quick sample, 'sweep' for legacy retrieval ablations): ").strip().lower()
        except Exception:
            choice = "test"
        if choice not in {"test", "sweep"}:
            print("Invalid choice, defaulting to 'test'.")
            choice = "test"
        mode = choice

    s1 = load_source(args.source1)
    s2 = load_source(args.source2_3)
    truth = parse_truth(args.truth)
    lookup = build_source_lookup(s2)

    output_path = RUNS_DIR / run_id / "ablation.csv"
    if mode == "test":
        sample_size = max(1, int(args.sample_size))
        s1_small = s1.sample(n=min(sample_size, len(s1)), random_state=0) if len(s1) > sample_size else s1.copy()
        s2_small = s2.sample(n=min(sample_size, len(s2)), random_state=0) if len(s2) > sample_size else s2.copy()
        df = run_single_retrieval(
            s1_small,
            truth,
            s2_small,
            lookup.loc[s2_small["entity_id"].values] if hasattr(lookup, "loc") else lookup,
            max_block_frequency=args.max_block_frequency,
            current_ngram_limit=args.ngram_limit,
            current_channel_multiplier=args.channel_multiplier,
            current_context_mode=args.context_mode,
            output_path=output_path,
        )
    else:
        df = run_retrieval_ablation(
            s1,
            truth,
            s2,
            lookup,
            max_block_frequency=args.max_block_frequency,
            current_ngram_limit=args.ngram_limit,
            current_channel_multiplier=args.channel_multiplier,
            current_context_mode=args.context_mode,
            output_path=output_path,
        )

    summary = save_run_artifacts(run_id, args, df)

    agg = aggregate_runs()
    image_path = RUNS_DIR / "metrics.png"
    plot_trajectory(agg, image_path)
    update_readme(agg, image_path)

    print("Run complete. Artifacts:")
    print(f"  runs/{run_id}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
