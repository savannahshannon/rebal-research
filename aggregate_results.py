"""Aggregate per-seed REBAL metrics (results/seed_*.json) into a
mean +/- std summary -- the Sprint 1 deliverable ("A summary with mean and
standard deviation per metric").

Usage:
    python aggregate_results.py
"""

import glob
import json

import numpy as np

METRICS = ["acc", "bacc", "head", "mid", "tail", "worst"]
METRIC_LABELS = {
    "acc":   "Top-1 accuracy",
    "bacc":  "Balanced accuracy",
    "head":  "Macro F1 (head)",
    "mid":   "Macro F1 (mid)",
    "tail":  "Macro F1 (tail)",
    "worst": "Worst-class F1",
}
CONFIGS = ["baseline", "rebal", "rebal_crt"]
CONFIG_LABELS = {
    "baseline":  "Baseline (imbalanced, no REBAL modules)",
    "rebal":     "Full REBAL",
    "rebal_crt": "Full REBAL + cRT",
}


def load_runs(pattern="results/seed_*.json"):
    files = sorted(glob.glob(pattern))
    runs = []
    for f in files:
        with open(f) as fh:
            runs.append(json.load(fh))
    return runs


def summarize(runs):
    lines = []
    seeds = [r["seed"] for r in runs]
    lines.append(f"# Multi-seed baseline summary (n = {len(runs)} seeds: {seeds})\n")

    if "runtime_sec" in runs[0]:
        times = np.array([r["runtime_sec"] for r in runs]) / 60.0
        lines.append(f"Mean runtime per seed: {times.mean():.1f} min "
                    f"(min {times.min():.1f}, max {times.max():.1f})\n")

    for cfg in CONFIGS:
        present = [r[cfg] for r in runs if cfg in r]
        if not present:
            continue
        lines.append(f"## {CONFIG_LABELS[cfg]}  (n = {len(present)})\n")
        lines.append("| Metric | Mean | Std |")
        lines.append("|---|---|---|")
        for m in METRICS:
            vals = np.array([r[m] for r in present], dtype="float64")
            mean = vals.mean()
            std = vals.std(ddof=1) if len(vals) > 1 else 0.0
            lines.append(f"| {METRIC_LABELS[m]} | {mean:.4f} | {std:.4f} |")
        lines.append("")

    lines.append("## Per-seed raw values\n")
    for cfg in CONFIGS:
        present = [r for r in runs if cfg in r]
        if not present:
            continue
        lines.append(f"### {CONFIG_LABELS[cfg]}\n")
        header = "| Seed | " + " | ".join(METRIC_LABELS[m] for m in METRICS) + " |"
        sep = "|---" * (len(METRICS) + 1) + "|"
        lines.append(header)
        lines.append(sep)
        for r in present:
            row = [f"{r[cfg][m]:.4f}" for m in METRICS]
            lines.append(f"| {r['seed']} | " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)


def main():
    runs = load_runs()
    if not runs:
        print("No results found in results/seed_*.json -- "
              "run ./run_multiseed.sh first.")
        return

    summary = summarize(runs)
    print(summary)

    with open("results/summary.md", "w") as f:
        f.write(summary)
    print("\nWrote results/summary.md")


if __name__ == "__main__":
    main()
