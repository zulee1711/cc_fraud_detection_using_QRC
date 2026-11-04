#!/bin/sh
# Runs every line of a jobs file (default jobs.txt), 4 at a time, from the repository root.
# Output goes to results/fraud_2/sweep.log, failures to results/fraud_2/errors.log.
# Usage: run_jobs.sh [jobs file, relative to experiments/fraud_2] [parallel jobs, default 4]
cd "$(dirname "$0")/../.."
mkdir -p results/fraud_2
xargs -P "${2:-4}" -L 1 sh -c '.venv/bin/python -m experiments.fraud_2.run "$@" >> results/fraud_2/sweep.log 2>&1 || echo "FAILED: $*" >> results/fraud_2/errors.log' _ < "experiments/fraud_2/${1:-jobs.txt}"
