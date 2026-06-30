"""Sprint 2.5: rho=100 module ablation grid.

Runs 6 configs x N seeds at rho=100 to identify which REBAL module is
responsible for the in-loop degradation observed in the Sprint 2 grid:

    baseline        vanilla CE, no REBAL at all
    no_cgan         REBAL without cGAN samples (SMOTE + RW + EQ only)
    no_eq           REBAL without the equalization regularizer
    no_rw           REBAL without effective-number reweighting
    decoupled_only  vanilla trunk + cRT step only (Kang et al. 2020 recipe)
    full_rebal      everything on

Each cell also applies the Sprint 2.5 stability preset (gated cGAN,
decoupled cGAN seed, deferred reweighting/equalization, LR warmup) via
framework.py's resnet32 ENV var preset, so this grid tests module
contribution on top of an already-stabilized baseline.

Usage:
    python run_ablation.py                  # all 6 configs, 5 seeds, rho=100
    python run_ablation.py --seeds 0 1 2     # fewer seeds
    python run_ablation.py --configs no_cgan full_rebal
    python run_ablation.py --skip-existing   # resume an interrupted grid
"""

import argparse
import os
import subprocess
import sys
import time

PYTHON = sys.executable
FRAMEWORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "framework.py")
ABLATION_ROOT = "experiments/sprint2_ablation"
RHO = 100
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

# (rebal_off, ablate) per config — see module docstring for what each means.
CONFIGS = {
    "baseline":       {"rebal_off": True,  "ablate": None},
    "no_cgan":        {"rebal_off": False, "ablate": "cgan"},
    "no_eq":          {"rebal_off": False, "ablate": "eq"},
    "no_rw":          {"rebal_off": False, "ablate": "rw"},
    "decoupled_only": {"rebal_off": False, "ablate": "decoupled"},
    "full_rebal":     {"rebal_off": False, "ablate": None},
}


def out_dir(config_name, seed):
    return os.path.join(ABLATION_ROOT, config_name, f"seed_{seed}")


def cell_done(config_name, seed):
    return os.path.exists(os.path.join(out_dir(config_name, seed), "metrics.json"))


def run_cell(config_name, cfg, seed, dataset):
    d = out_dir(config_name, seed)
    os.makedirs(d, exist_ok=True)
    log_path = os.path.join(d, "run.log")

    env = os.environ.copy()
    env["REBAL_RHO"]     = str(RHO)
    env["REBAL_ARCH"]    = "resnet32"   # also applies the Sprint 2.5 stability preset
    env["REBAL_DATASET"] = dataset
    env["REBAL_OUT_DIR"] = d
    if cfg["rebal_off"]:
        env["REBAL_BASELINE_ONLY"] = "1"
    else:
        env.pop("REBAL_BASELINE_ONLY", None)
    if cfg["ablate"]:
        env["REBAL_ABLATE"] = cfg["ablate"]
    else:
        env.pop("REBAL_ABLATE", None)

    cmd = [PYTHON, "-u", FRAMEWORK, "--seed", str(seed)]

    print(f"\n{'=' * 60}")
    print(f"  config={config_name}  seed={seed}  ρ={RHO}  dataset={dataset}")
    print(f"  out → {d}")
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
        print("\n  [interrupted — rerun with --skip-existing to resume]")
        return False

    elapsed = (time.time() - t0) / 60
    if proc.returncode != 0:
        print(f"\n  ERROR: exit code {proc.returncode} after {elapsed:.1f} min")
        return False
    print(f"\n  Done: {elapsed:.1f} min  →  {d}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Sprint 2.5: rho=100 module ablation grid.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        metavar="SEED")
    parser.add_argument("--configs", type=str, nargs="+",
                        default=list(CONFIGS.keys()), choices=list(CONFIGS.keys()))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100", "cifar10"])
    args = parser.parse_args()

    total = len(args.configs) * len(args.seeds)
    done = failed = skipped = 0

    print(f"Sprint 2.5 ablation grid: configs={args.configs}  seeds={args.seeds}  "
          f"ρ={RHO}  dataset={args.dataset}  total={total} cells")

    for cname in args.configs:
        cfg = CONFIGS[cname]
        for seed in args.seeds:
            if args.skip_existing and cell_done(cname, seed):
                print(f"  skip {cname} seed={seed} (metrics.json exists)")
                skipped += 1
                continue
            ok = run_cell(cname, cfg, seed, args.dataset)
            done += 1 if ok else 0
            failed += 0 if ok else 1

    print(f"\n{'=' * 60}")
    print(f"Ablation grid complete: {done} succeeded, {failed} failed, "
          f"{skipped} skipped (of {total} total)")
    if done + skipped == total and failed == 0:
        print("All cells done. Run:  python aggregate_ablation.py")


if __name__ == "__main__":
    main()
