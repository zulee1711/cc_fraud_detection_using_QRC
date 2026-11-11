#!/bin/sh
# Runs every line of jobs.txt, N at a time (default 4), from the repository root.
# Output goes to results/fraud_2_noreset/sweep.log, failures to results/fraud_2_noreset/errors.log.
# Usage: run_jobs.sh [parallel jobs, default 4]
cd "$(dirname "$0")/../.."
mkdir -p results/fraud_2_noreset
xargs -P "${1:-4}" -L 1 sh -c '.venv/bin/python -m experiments.fraud_2_noreset.run "$@" >> results/fraud_2_noreset/sweep.log 2>&1 || echo "FAILED: $*" >> results/fraud_2_noreset/errors.log' _ < experiments/fraud_2_noreset/jobs.txt
