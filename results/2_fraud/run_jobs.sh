#!/bin/sh
# Runs every line of a jobs file (default jobs.txt), 4 at a time, from the repository root. Failures go to errors.log.
# Usage: run_jobs.sh [jobs file, relative to results/2_fraud] [parallel jobs, default 4]
cd "$(dirname "$0")/../.."
xargs -P "${2:-4}" -L 1 sh -c 'PYTHONPATH=. .venv/bin/python results/2_fraud/fraud.py "$@" >> results/2_fraud/sweep.log 2>&1 || echo "FAILED: $*" >> results/2_fraud/errors.log' _ < "results/2_fraud/${1:-jobs.txt}"
