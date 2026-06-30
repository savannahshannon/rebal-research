"""Sprint 2.5 aggregator for the rho=100 module ablation grid.

Reads experiments/sprint2_ablation/{config}/seed_*/metrics.json and produces
a single comparison table across all 6 configs, to identify which REBAL
module is responsible for the in-loop degradation at extreme imbalance.

Usage:
    python aggregate_ablation.py
"""

import glob
import json
import os

import numpy as np

ABLATION_ROOT = "experiments/sprint2_ablation"

METRIC_KEYS = ["acc", "bacc", "head", "mid", "tail",
               "head_tail_gap", "worst_f1", "worst_acc"]
METRIC_LABELS = {
    "acc":           "top1_acc",
    "bacc":          "balanced_acc",
    "head":          "head_f1",
    "mid":           "mid_f1",
    "tail":          "tail_f1",
    "head_tail_gap": "head_tail_f1_gap",
    "worst_f1":      "worst_class_f1",
    "worst_acc":     "worst_class_acc",
}
# Order matters for the printed table — matches the spec's table.
CONFIG_ORDER = ["baseline", "no_cgan", "no_eq", "no_rw", "decoupled_only", "full_rebal"]


def bootstrap_ci(vals, n_boot=1000, seed=0):
    """Percentile bootstrap 95% CI on the mean. Returns (lo, hi)."""
    vals = np.array(vals, dtype="float64")
    if len(vals) < 2:
        v = float(vals[0]) if len(vals) else float("nan")
        return v, v
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(vals, size=len(vals), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def load_grid():
    """Return dict: grid[config_name][seed] = full metrics.json contents."""
    grid = {}
    pattern = os.path.join(ABLATION_ROOT, "*", "seed_*", "metrics.json")
    for mf in sorted(glob.glob(pattern)):
        parts = mf.replace("\\", "/").split("/")
        config_name, seed_tag = parts[-3], parts[-2]
        seed = int(seed_tag.split("_")[1])
        with open(mf) as f:
            data = json.load(f)
        grid.setdefault(config_name, {})[seed] = data
    return grid


def _result_for(config_name, data):
    """Pick the right evaluation point: 'baseline' config uses the vanilla
    feature-extractor eval; every ablation config uses REBAL(+cRT)."""
    if config_name == "baseline":
        return data.get("baseline")
    return data.get("rebal_crt") or data.get("rebal")


def write_comparison_table(grid, path):
    configs = [c for c in CONFIG_ORDER if c in grid]
    header = (["metric"]
              + [f"{c}_median" for c in configs]
              + [f"{c}_mean" for c in configs]
              + [f"{c}_ci_lo" for c in configs]
              + [f"{c}_ci_hi" for c in configs]
              + [f"{c}_n" for c in configs])
    rows = [header]

    for key in METRIC_KEYS:
        row = [METRIC_LABELS[key]]
        medians, means, ci_los, ci_his, ns = [], [], [], [], []
        for c in configs:
            seeds = sorted(grid[c].keys())
            vals = []
            for s in seeds:
                r = _result_for(c, grid[c][s])
                if r and key in r:
                    vals.append(r[key])
            if vals:
                medians.append(f"{np.median(vals):.4f}")
                means.append(f"{np.mean(vals):.4f}")
                lo, hi = bootstrap_ci(vals)
                ci_los.append(f"{lo:.4f}")
                ci_his.append(f"{hi:.4f}")
                ns.append(str(len(vals)))
            else:
                medians.append(""); means.append("")
                ci_los.append(""); ci_his.append(""); ns.append("0")
        row += medians + means + ci_los + ci_his + ns
        rows.append(row)

    with open(path, "w") as f:
        for row in rows:
            f.write(",".join(str(c) for c in row) + "\n")
    print(f"Wrote {path}")


def write_memo(grid, path):
    configs = [c for c in CONFIG_ORDER if c in grid]
    lines = [
        "# Sprint 2.5 — rho=100 Module Ablation Memo",
        "",
        f"Configs found: {configs}",
        "",
        "## Per-config summary (median across seeds)",
        "",
    ]

    summary = {}
    for c in configs:
        seeds = sorted(grid[c].keys())
        vals = {k: [] for k in METRIC_KEYS}
        for s in seeds:
            r = _result_for(c, grid[c][s])
            if r:
                for k in METRIC_KEYS:
                    if k in r:
                        vals[k].append(r[k])
        summary[c] = {k: (float(np.median(v)) if v else None) for k, v in vals.items()}
        n = len(seeds)
        m = summary[c]
        if m["acc"] is not None:
            lines.append(
                f"- {c} (n={n}): top1={m['acc']:.3f}  tail_f1={m['tail']:.3f}  "
                f"gap={m['head_tail_gap']:.3f}  worst_acc={m['worst_acc']:.3f}"
            )
        else:
            lines.append(f"- {c}: NO DATA")
    lines.append("")

    lines.append("## Diagnosis")
    lines.append("")
    if "baseline" in summary and "full_rebal" in summary and "no_cgan" in summary:
        base = summary["baseline"]["acc"]
        full = summary["full_rebal"]["acc"]
        no_cgan = summary["no_cgan"]["acc"]
        if base is not None and full is not None and no_cgan is not None:
            lines.append(f"baseline top1={base:.3f}, full_rebal top1={full:.3f}, "
                        f"no_cgan top1={no_cgan:.3f}")
            if no_cgan > full and no_cgan >= base - 0.02:
                lines.append("")
                lines.append("→ no_cgan outperforms full_rebal and is competitive with "
                            "baseline: cGAN samples are degrading the trunk at rho=100, "
                            "consistent with the gating hypothesis (tail classes have too "
                            "few real samples for the cGAN to learn a useful distribution).")
            elif full >= base:
                lines.append("")
                lines.append("→ full_rebal matches or beats baseline: the Sprint 2.5 "
                            "stability fixes (gated cGAN, deferred rebalancing, decoupled "
                            "seed, warmup) appear to have resolved the degradation.")
            else:
                lines.append("")
                lines.append("→ Inconclusive from top-1 alone — check tail_f1 and "
                            "head_tail_gap per-config above; REBAL may still win on "
                            "equity even if top-1 is roughly flat.")
    lines.append("")
    lines.append("## Next steps")
    lines.append("")
    lines.append("- If no_cgan wins: keep CGAN_MIN_REAL gating, consider raising the "
                "threshold further or dropping cGAN entirely at rho>=100.")
    lines.append("- If decoupled_only matches full_rebal: cRT is carrying most of the "
                "equity gain at extreme imbalance; in-loop terms add little here.")
    lines.append("- Re-run aggregate_sprint2.py on the full grid once the winning "
                "configuration is folded back into the default Sprint 2.5 preset.")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {path}")


def main():
    grid = load_grid()
    if not grid:
        print(f"No data found under {ABLATION_ROOT}/. Run run_ablation.py first.")
        return

    print(f"Found configs: {sorted(grid.keys())}")
    for c, seeds in grid.items():
        print(f"  {c}: seeds {sorted(seeds.keys())}")

    write_comparison_table(grid, os.path.join(ABLATION_ROOT, "ablation_comparison.csv"))
    write_memo(grid, os.path.join(ABLATION_ROOT, "ablation_memo.md"))


if __name__ == "__main__":
    main()
