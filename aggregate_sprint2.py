"""Sprint 2 aggregator.

Reads experiments/sprint2_grid/ and produces:
  table_rho{10,50,100}.csv  — per-rho metric tables (baseline / rebal / rebal_crt)
  sprint2_scaling.csv        — per-rho delta (REBAL gain over baseline)
  sprint2_memo.md            — scaling analysis template

Usage:
    python aggregate_sprint2.py
"""

import glob
import json
import os

import numpy as np

GRID_ROOT = "experiments/sprint2_grid"

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
VARIANCE_THRESHOLD = 0.03   # 3 pp — sprint brief risk threshold


# ── data loading ─────────────────────────────────────────────────────────────

def load_grid():
    """Return nested dict: grid[rho][seed][mode][config_key] = scalar_dict."""
    grid = {}
    pattern = os.path.join(GRID_ROOT, "rho_*", "seed_*", "*", "metrics.json")
    for mf in sorted(glob.glob(pattern)):
        parts = mf.replace("\\", "/").split("/")
        # …/rho_10/seed_0/baseline/metrics.json
        rho_tag, seed_tag, mode_tag = parts[-4], parts[-3], parts[-2]
        rho  = float(rho_tag.split("_")[1])
        seed = int(seed_tag.split("_")[1])
        with open(mf) as f:
            data = json.load(f)
        grid.setdefault(rho, {}).setdefault(seed, {})[mode_tag] = data
    return grid


# ── per-rho table ─────────────────────────────────────────────────────────────

def bootstrap_ci(vals, n_boot=1000, seed=0):
    """Percentile bootstrap 95% CI on the mean. Returns (lo, hi).

    More honest than mean±std for n=5 seeds: doesn't assume normality and
    isn't dominated by a single outlier seed the way std can be.
    """
    a = np.array(vals, dtype="float64")
    if len(a) < 2:
        v = float(a[0]) if len(a) else float("nan")
        return v, v
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(a, size=len(a), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def _ms(vals):
    """Return (mean, std, median, ci_lo, ci_hi) over a list of floats."""
    a = np.array(vals, dtype="float64")
    mean   = a.mean()
    std    = a.std(ddof=1) if len(a) > 1 else 0.0
    median = np.median(a)
    ci_lo, ci_hi = bootstrap_ci(vals)
    return mean, std, median, ci_lo, ci_hi


def write_rho_table(grid, rho, path):
    seeds = sorted(grid.get(rho, {}).keys())
    if not seeds:
        print(f"  no data for ρ={rho}, skipping.")
        return

    # median + bootstrap 95% CI are the primary reportable numbers (more
    # honest than mean±std at n=5 seeds); mean/std kept for completeness.
    header = (["config", "metric", "median", "ci_lo", "ci_hi",
               "mean", "std", "n_seeds"]
              + [f"seed_{s}" for s in seeds])
    rows = [header]

    # baseline config: from "baseline" mode run, key "baseline"
    # rebal config:    from "rebal" mode run,    key "rebal"
    # rebal_crt:       from "rebal" mode run,    key "rebal_crt"
    configs = [
        ("baseline", "baseline", "baseline"),   # (label, mode, json_key)
        ("rebal",    "rebal",    "rebal"),
        ("rebal_crt","rebal",    "rebal_crt"),
    ]
    for label, mode, jkey in configs:
        vals_by_seed = {}
        for s in seeds:
            entry = grid.get(rho, {}).get(s, {}).get(mode, {})
            if jkey in entry:
                vals_by_seed[s] = entry[jkey]
        if not vals_by_seed:
            continue
        for key in METRIC_KEYS:
            v_list = [vals_by_seed[s][key] for s in seeds if s in vals_by_seed]
            if not v_list:
                continue
            mean, std, median, ci_lo, ci_hi = _ms(v_list)
            row = [label, METRIC_LABELS[key],
                   f"{median:.4f}", f"{ci_lo:.4f}", f"{ci_hi:.4f}",
                   f"{mean:.4f}", f"{std:.4f}", str(len(v_list))]
            for s in seeds:
                row.append(f"{vals_by_seed[s][key]:.4f}" if s in vals_by_seed else "")
            rows.append(row)

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(",".join(str(c) for c in row) + "\n")
    print(f"Wrote {path}")


# ── scaling CSV ───────────────────────────────────────────────────────────────

def write_scaling_csv(grid, path):
    """Per-rho mean delta: REBAL_cRT (or REBAL) minus baseline."""
    rhos = sorted(grid.keys())
    header = (["rho", "n_seeds"]
              + [f"delta_{METRIC_LABELS[k]}" for k in METRIC_KEYS]
              + [f"delta_{METRIC_LABELS[k]}_std" for k in METRIC_KEYS])
    rows = [header]

    for rho in rhos:
        deltas = {k: [] for k in METRIC_KEYS}
        n = 0
        for seed, modes in grid[rho].items():
            base    = modes.get("baseline", {}).get("baseline")
            rebal   = modes.get("rebal", {})
            top_cfg = rebal.get("rebal_crt") or rebal.get("rebal")
            if base is None or top_cfg is None:
                continue
            for k in METRIC_KEYS:
                if k in base and k in top_cfg:
                    deltas[k].append(top_cfg[k] - base[k])
            n += 1
        if n == 0:
            continue
        row = [_rho_str(rho), str(n)]
        for k in METRIC_KEYS:
            row.append(f"{np.mean(deltas[k]):.4f}" if deltas[k] else "")
        for k in METRIC_KEYS:
            v = deltas[k]
            row.append(f"{np.std(v, ddof=1):.4f}" if len(v) > 1 else "0.0000")
        rows.append(row)

    with open(path, "w") as f:
        for row in rows:
            f.write(",".join(row) + "\n")
    print(f"Wrote {path}")


def _rho_str(rho):
    return str(int(rho)) if rho == int(rho) else str(rho)


# ── memo ──────────────────────────────────────────────────────────────────────

def write_memo(grid, scaling_path, path):
    rhos = sorted(grid.keys())
    lines = [
        "# Sprint 2 — Multi-Imbalance Grid: Scaling Memo",
        "",
        f"ρ values tested: {[_rho_str(r) for r in rhos]}",
        f"Seeds per cell:  {sorted(next(iter(grid.values())).keys())}",
        "",
        "## Core question",
        "",
        "The REBAL framework predicts that its equity gains should *grow* with ρ:",
        "more severe imbalance gives REBAL more room to improve tail coverage.",
        "Concretely, delta_tail_f1 and delta_worst_class_acc should increase as ρ rises.",
        "",
        "## Per-ρ summary",
        "",
    ]

    for rho in rhos:
        lines.append(f"### ρ = {_rho_str(rho)}")
        seeds = sorted(grid[rho].keys())
        for seed in seeds:
            modes = grid[rho][seed]
            base  = modes.get("baseline", {}).get("baseline")
            rebal = modes.get("rebal", {})
            top   = rebal.get("rebal_crt") or rebal.get("rebal")
            if base and top:
                lines.append(
                    f"  seed {seed}:  baseline top-1={base['acc']:.3f} "
                    f"tail={base['tail']:.3f} gap={base['head_tail_gap']:.3f} | "
                    f"REBAL+cRT top-1={top['acc']:.3f} "
                    f"tail={top['tail']:.3f} gap={top['head_tail_gap']:.3f}  "
                    f"(Δtail={top['tail']-base['tail']:+.3f} "
                    f"Δgap={top['head_tail_gap']-base['head_tail_gap']:+.3f})"
                )
            else:
                missing = []
                if not base: missing.append("baseline")
                if not top:  missing.append("rebal")
                lines.append(f"  seed {seed}: MISSING — {', '.join(missing)} run not found")
        lines.append("")

    lines += [
        "## Scaling assessment  (fill in after reviewing sprint2_scaling.csv)",
        "",
        "[ ] delta_tail_f1 increases with ρ        → REBAL equity gains scale with severity",
        "[ ] delta_tail_f1 flat or decreasing       → gains limited to moderate imbalance",
        "[ ] delta_worst_class_acc increases with ρ → worst-case protection is robust",
        "[ ] head_tail_gap reduction grows with ρ   → equity claim strengthens at high ρ",
        "",
        "## Variance flags  (std > 3 pp triggers a flag)",
        "",
    ]

    # Variance check per rho
    for rho in rhos:
        seeds = sorted(grid[rho].keys())
        flags = []
        for mode, jkey, label in [("rebal", "rebal_crt", "rebal_crt"),
                                   ("rebal", "rebal",     "rebal")]:
            vals_by_key = {k: [] for k in METRIC_KEYS}
            for seed in seeds:
                entry = grid[rho].get(seed, {}).get(mode, {}).get(jkey)
                if entry:
                    for k in METRIC_KEYS:
                        vals_by_key[k].append(entry[k])
            for k, vlist in vals_by_key.items():
                if len(vlist) > 1:
                    std = float(np.std(vlist, ddof=1))
                    if std > VARIANCE_THRESHOLD:
                        flags.append(f"{label}.{METRIC_LABELS[k]}: std={std:.4f}")
            if flags:
                break   # one config check is enough per rho
        if flags:
            lines.append(f"ρ={_rho_str(rho)}  FLAGGED (std > 3 pp):")
            for f in flags:
                lines.append(f"  {f}")
        else:
            lines.append(f"ρ={_rho_str(rho)}  all metrics within 3 pp std threshold")
        lines.append("")

    lines += [
        "## Next: Sprint 3",
        "",
        "Generator upgrade (improved cGAN / diffusion-based synthesis) to address",
        "the bimodal convergence identified in Sprint 1 and quantified across ρ here.",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    grid = load_grid()
    if not grid:
        print(f"No data found under {GRID_ROOT}/.")
        print("Run:  python run_multi_imbalance.py  (and/or --baseline) first.")
        return

    rhos = sorted(grid.keys())
    print(f"Found data for ρ = {[_rho_str(r) for r in rhos]}")
    print(f"Seeds per ρ: { {_rho_str(r): sorted(grid[r].keys()) for r in rhos} }")

    for rho in rhos:
        table_path = os.path.join(GRID_ROOT, f"table_rho{_rho_str(rho)}.csv")
        write_rho_table(grid, rho, table_path)

    scaling_path = os.path.join(GRID_ROOT, "sprint2_scaling.csv")
    write_scaling_csv(grid, scaling_path)
    write_memo(grid, scaling_path, os.path.join(GRID_ROOT, "sprint2_memo.md"))


if __name__ == "__main__":
    main()
