"""Sprint 2 multi-imbalance grid runner.

Iterates over (rho, seed) pairs and calls framework.py for each cell,
writing artifacts under experiments/sprint2_grid/rho_{rho}/seed_{seed}/{mode}/.

Usage:
    # Full 3x3 REBAL grid (9 runs)
    python run_multi_imbalance.py

    # Full 3x3 baseline grid (9 runs, REBAL disabled — plain cross-entropy)
    python run_multi_imbalance.py --baseline

    # Single calibration run at rho=100 to verify ~38-40% top-1
    python run_multi_imbalance.py --baseline --rhos 100 --seeds 0

    # Resume interrupted grid (skips cells with existing metrics.json)
    python run_multi_imbalance.py --skip-existing

    # CIFAR-10 grid
    python run_multi_imbalance.py --dataset cifar10

Ctrl-C interrupts the current cell cleanly; rerun with --skip-existing to continue.
"""

import argparse
import json
import os
import subprocess
import sys
import time

PYTHON = sys.executable
FRAMEWORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "framework.py")
GRID_ROOT = "experiments/sprint2_grid"

DEFAULT_RHOS  = [10, 50, 100]
DEFAULT_SEEDS = [0, 1, 2]


# ── helpers ──────────────────────────────────────────────────────────────────

def _rho_str(rho):
    return str(int(rho)) if rho == int(rho) else str(rho)


def cell_out_dir(rho, seed, baseline):
    mode = "baseline" if baseline else "rebal"
    return os.path.join(GRID_ROOT, f"rho_{_rho_str(rho)}", f"seed_{seed}", mode)


def cell_done(rho, seed, baseline):
    return os.path.exists(os.path.join(cell_out_dir(rho, seed, baseline), "metrics.json"))


# ── single-cell runner ────────────────────────────────────────────────────────

def run_cell(rho, seed, baseline, dataset):
    """Launch framework.py for one (rho, seed) cell; stream output live.
    Returns True on success, False on non-zero exit."""
    out_dir = cell_out_dir(rho, seed, baseline)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")

    env = os.environ.copy()
    env["REBAL_RHO"]     = str(rho)
    env["REBAL_ARCH"]    = "resnet32"
    env["REBAL_DATASET"] = dataset
    env["REBAL_OUT_DIR"] = out_dir
    if baseline:
        env["REBAL_BASELINE_ONLY"] = "1"
    else:
        env.pop("REBAL_BASELINE_ONLY", None)

    cmd = [PYTHON, "-u", FRAMEWORK, "--seed", str(seed)]

    mode_label = "baseline" if baseline else "REBAL"
    print(f"\n{'=' * 60}")
    print(f"  ρ={rho}  seed={seed}  mode={mode_label}  dataset={dataset}")
    print(f"  out → {out_dir}")
    print(f"{'=' * 60}", flush=True)

    t0 = time.time()
    try:
        with open(log_path, "w", buffering=1) as flog:
            proc = subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                print(line, end="", flush=True)
                flog.write(line)
            proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\n  [interrupted — cell partial, rerun with --skip-existing to resume]")
        return False

    elapsed = (time.time() - t0) / 60
    if proc.returncode != 0:
        print(f"\n  ERROR: exit code {proc.returncode} after {elapsed:.1f} min")
        return False
    print(f"\n  Done: {elapsed:.1f} min  →  {out_dir}")
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sprint 2: run the REBAL grid across imbalance ratios and seeds.")
    parser.add_argument("--rhos", type=float, nargs="+", default=DEFAULT_RHOS,
                        metavar="RHO",
                        help=f"imbalance ratios to sweep (default: {DEFAULT_RHOS})")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        metavar="SEED",
                        help=f"random seeds (default: {DEFAULT_SEEDS})")
    parser.add_argument("--baseline", action="store_true",
                        help="run baseline mode (REBAL disabled, plain cross-entropy)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip cells where metrics.json already exists")
    parser.add_argument("--dataset", default="cifar100",
                        choices=["cifar100", "cifar10"],
                        help="dataset (default: cifar100)")
    args = parser.parse_args()

    total    = len(args.rhos) * len(args.seeds)
    done = failed = skipped = 0

    print(f"Sprint 2 grid: ρ={args.rhos}  seeds={args.seeds}  "
          f"mode={'baseline' if args.baseline else 'rebal'}  "
          f"dataset={args.dataset}  total={total} cells")

    for rho in args.rhos:
        for seed in args.seeds:
            if args.skip_existing and cell_done(rho, seed, args.baseline):
                print(f"  skip ρ={rho} seed={seed} (metrics.json exists)")
                skipped += 1
                continue
            ok = run_cell(rho, seed, args.baseline, args.dataset)
            done += 1 if ok else 0
            failed += 0 if ok else 1

    print(f"\n{'=' * 60}")
    print(f"Grid complete: {done} succeeded, {failed} failed, {skipped} skipped "
          f"(of {total} total)")
    if done + skipped == total and failed == 0:
        print("All cells done. Run:  python aggregate_sprint2.py")


if __name__ == "__main__":
    main()
