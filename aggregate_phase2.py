"""Sprint 1 deliverable: aggregate experiments/phase2_multiseed/seed_*/metrics.json
into phase2_seeds_summary.csv (Table 3) and a short variance memo.

Usage:
    python aggregate_phase2.py
"""

import glob
import json
import os

import numpy as np

ROOT = "experiments/phase2_multiseed"

METRIC_KEYS = ["acc", "bacc", "head", "mid", "tail",
            "head_tail_gap", "worst_f1", "worst_acc"]
METRIC_NAMES = {
    "acc":           "top1_acc",
    "bacc":          "balanced_acc",
    "head":          "head_f1",
    "mid":           "mid_f1",
    "tail":          "tail_f1",
    "head_tail_gap": "head_tail_f1_gap",
    "worst_f1":      "worst_class_f1",
    "worst_acc":     "worst_class_acc",
}
CONFIGS = ["baseline", "rebal", "rebal_crt"]

# Risk threshold from the sprint brief: >3pp across-seed std weakens the
# equity claim and should be flagged in the memo.
VARIANCE_FLAG_THRESHOLD = 0.03


def load_runs():
    runs = []
    for f in sorted(glob.glob(os.path.join(ROOT, "seed_*", "metrics.json"))):
        with open(f) as fh:
            runs.append(json.load(fh))
    return sorted(runs, key=lambda r: r["seed"])


def write_summary_csv(runs, path):
    seeds = [r["seed"] for r in runs]
    header = (["config", "metric", "mean", "std", "median", "n_seeds"]
            + [f"seed_{s}" for s in seeds])
    rows = [header]

    for cfg in CONFIGS:
        present = [(r["seed"], r[cfg]) for r in runs if cfg in r]
        if not present:
            continue
        for key in METRIC_KEYS:
            vals = np.array([m[key] for _, m in present], dtype="float64")
            mean = vals.mean()
            std = vals.std(ddof=1) if len(vals) > 1 else 0.0
            median = np.median(vals)
            row = [cfg, METRIC_NAMES[key], f"{mean:.4f}", f"{std:.4f}", f"{median:.4f}",
                str(len(vals))]
            seed_vals = {s: m[key] for s, m in present}
            for s in seeds:
                row.append(f"{seed_vals[s]:.4f}" if s in seed_vals else "")
            rows.append(row)

    with open(path, "w") as f:
        for row in rows:
            f.write(",".join(str(c) for c in row) + "\n")
    print(f"Wrote {path}")


def write_memo(runs, path):
    seeds = [r["seed"] for r in runs]
    lines = []
    lines.append("# Sprint 1 -- Multi-Seed Phase II Baseline: Variance Memo")
    lines.append("")
    lines.append(f"Seeds run: {seeds} (n={len(runs)})")
    if runs and "runtime_sec" in runs[0]:
        times = np.array([r["runtime_sec"] for r in runs]) / 60.0
        lines.append(f"Runtime per seed: mean {times.mean():.1f} min, "
                    f"range [{times.min():.1f}, {times.max():.1f}] min")
    lines.append("")

    flagged = []
    for cfg in CONFIGS:
        present = [r[cfg] for r in runs if cfg in r]
        if not present:
            continue
        lines.append(f"## {cfg}")
        for key in METRIC_KEYS:
            vals = np.array([m[key] for m in present], dtype="float64")
            mean = vals.mean()
            std = vals.std(ddof=1) if len(vals) > 1 else 0.0
            median = np.median(vals)
            flag = ""
            if std > VARIANCE_FLAG_THRESHOLD:
                flag = "  <-- std > 3pp"
                flagged.append((cfg, METRIC_NAMES[key], std))
            lines.append(f"- {METRIC_NAMES[key]}: mean={mean:.4f} "
                        f"std={std:.4f} median={median:.4f}{flag}")
        lines.append("")

    lines.append("## Variance assessment")
    if flagged:
        lines.append("The following metrics exceed the 3pp across-seed std "
                    "threshold from the sprint brief, which would weaken "
                    "the equity claim if it holds up under inspection:")
        for cfg, name, std in flagged:
            lines.append(f"- {cfg}.{name}: std={std:.4f}")
        lines.append("")
        lines.append("Suggested follow-ups per the sprint brief's mitigation:")
        lines.append("- Compare per-seed run.log cGAN g/d loss curves for "
                    "outlier seeds -- large divergence would point to "
                    "cGAN seed-dependence rather than the classifier itself.")
        lines.append("- Re-run the highest-variance seed with a tuned "
                    "gradient-clipping schedule (e.g. warmup clipnorm) and "
                    "compare against its original run.")
        lines.append("- Report the median (above) alongside the mean when "
                    "presenting Table 3, as medians are less sensitive to "
                    "any single outlier seed.")
    else:
        lines.append("All metrics are within the 3pp across-seed std "
                    "threshold from the sprint brief -- no evidence yet of "
                    "batch-size sensitivity or cGAN seed-dependence large "
                    "enough to threaten the equity claim.")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {path}")


def main():
    runs = load_runs()
    if not runs:
        print(f"No metrics.json found under {ROOT}/seed_*/ -- "
            "run ./run_multiseed.sh first.")
        return

    write_summary_csv(runs, os.path.join(ROOT, "phase2_seeds_summary.csv"))
    write_memo(runs, os.path.join(ROOT, "memo.md"))


if __name__ == "__main__":
    main()
