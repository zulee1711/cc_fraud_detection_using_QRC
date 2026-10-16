"""The sweep runner, on the smallest simulation preset."""

import csv

from experiments import sweep
from experiments.sweep import main


def test_sweep_prepares_each_dataset_once_and_reports_failures(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    jobs = tmp_path / "jobs.txt"
    jobs.write_text(
        "# two readout settings on one dataset: prepared once, then shared\n"
        f"baseline --size small --window-length 1 --alpha 1 --output-dir {runs}\n"
        f"baseline --size small --window-length 1 --alpha 10 --output-dir {runs}\n"
        "\n"
        # parses, but fails at run time
        f"qrc --size small --window-length 1 --observables bogus --output-dir {runs}\n"
    )
    preparations = []
    prepare = sweep.prepare_data
    monkeypatch.setattr(sweep, "prepare_data", lambda args: preparations.append(args) or prepare(args))

    code = main([str(jobs)])

    assert code == 1
    assert len(preparations) == 1
    with open(runs / "summary.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert sorted(row["alpha"] for row in rows) == ["1.0", "10.0"]
