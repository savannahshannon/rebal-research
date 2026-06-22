#!/bin/bash
# Sprint 1: multi-seed Phase II baseline.
#
# Runs the REBAL pipeline across multiple seeds, each in its own Python
# process (clean TF/Keras state, avoids memory creep across runs). Each
# seed writes to experiments/phase2_multiseed/seed_<N>/:
#   run.log           -- full training log
#   metrics.json       -- top-1/balanced acc/head-mid-tail F1/worst-class/gap
#   per_class_f1.csv   -- per-class F1 + accuracy for baseline/REBAL/cRT
#   checkpoints/        -- model weights (baseline, final, crt_head)
#
# Usage:
#   ./run_multiseed.sh                # seeds 0 1 2 3 4 (default)
#   ./run_multiseed.sh 0 1 2          # explicit seed list
#
# After all seeds finish, aggregate_phase2.py runs automatically to produce
# experiments/phase2_multiseed/phase2_seeds_summary.csv and memo.md.

set -e
cd "$(dirname "$0")"

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
    SEEDS=(0 1 2 3 4)
fi

for seed in "${SEEDS[@]}"; do
    exp_dir="experiments/phase2_multiseed/seed_${seed}"
    mkdir -p "$exp_dir"
    echo "=================================================="
    echo "  Seed $seed  ($(date))"
    echo "=================================================="
    python -u framework.py --seed "$seed" 2>&1 | tee "${exp_dir}/run.log"
done

echo
echo "All runs complete. Aggregating..."
python aggregate_phase2.py
