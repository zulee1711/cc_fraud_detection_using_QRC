"""Run many ``experiments.cli`` commands one after another, preparing each dataset only once.

The jobs file holds one command per line, as you would pass it to ``python -m experiments.cli``
(blank lines and ``#`` comments are skipped):

    qrc --data load --feature-set 7 --window-length 1 --n-mem-qubits 0 --seed 42
    qrc --data load --feature-set 7 --window-length 1 --n-mem-qubits 0 --seed 43
    baseline --data load --feature-set 7 --window-length 1

The jobs are grouped by their data settings (source, feature set, window length): the data is
prepared once per group, in memory, and every job of the group runs on it. A failing job is
reported and the sweep moves on to the next one. Results go to the jobs' ``--output-dir`` and
``summary.csv`` as usual.

Usage:
    python -m experiments.sweep jobs.txt
"""

import argparse
import json
import shlex
import sys
import traceback
from pathlib import Path

from experiments.cli import FEATURE_SETS, build_parser, data_source, prepare_data, print_data_summary, run


def read_jobs(path):
    """The argument lists in a jobs file, one per non-blank, non-comment line."""
    lines = Path(path).read_text().splitlines()
    return [shlex.split(line) for line in lines if line.strip() and not line.lstrip().startswith("#")]


def data_key(args):
    """Everything ``prepare_data`` depends on: jobs with the same key share the prepared data."""
    spec = {**data_source(args), "features": FEATURE_SETS[args.feature_set], "window_length": args.window_length}
    return json.dumps(spec, sort_keys=True, default=str)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jobs_file", type=Path, help="one experiments.cli command per line")
    args = parser.parse_args(argv)

    cli_parser = build_parser()
    groups = {}
    for job in read_jobs(args.jobs_file):
        job_args = cli_parser.parse_args(job)  # fails fast on a typo, before anything runs
        groups.setdefault(data_key(job_args), []).append((job, job_args))
    n_jobs = sum(len(jobs) for jobs in groups.values())
    print(f"{n_jobs} jobs on {len(groups)} datasets", flush=True)

    done, failed = 0, []
    for jobs in groups.values():
        data = prepare_data(jobs[0][1])
        print_data_summary(data)
        for job, job_args in jobs:
            done += 1
            print(f"\n[{done}/{n_jobs}] {shlex.join(job)}", flush=True)
            if job_args.command == "prepare":
                continue
            try:
                run(job_args, data)
            except Exception:
                traceback.print_exc()
                failed.append(shlex.join(job))
    if failed:
        print(f"\n{len(failed)} failed:\n" + "\n".join(failed), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
