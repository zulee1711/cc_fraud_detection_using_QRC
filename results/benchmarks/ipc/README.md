# Information Processing Capacity (IPC) Benchmarks

These experiments measure the reservoir's capacity to retain and reproduce
delayed input signals. Run logs are organized by experiment and, where
applicable, Hamiltonian. Each `.txt` file is a run's console log; generated
logs are excluded from Git, while `jobs.txt` remains tracked.

## Experiment folders

| Folder | Comparison |
|---|---|
| `precision/` | Shot-based precision settings and exact-estimator results |
| `depth/` | Reservoir depths 1, 3, 8, and 15 |
| `window/` | Input window lengths; depth 4 for `random`, depth 2 for the other Hamiltonians |
| `max_delay/` | Maximum-delay comparisons |
| `observables_copy/<hamiltonian>/` | Observable-set logs grouped by Hamiltonian |
| `plots/` | Main IPC summary figures |

The job definitions are in [`jobs.txt`](jobs.txt). Run one study at a time from
the repository root, for example:

```bash
bash results/benchmarks/ipc/jobs.txt observables
bash results/benchmarks/ipc/jobs.txt spinpulse
```

Available studies are `precision`, `depth`, `window`, `max-delay`,
`observables`, and `spinpulse`. Logs are named with their backend and
experiment parameters; observable logs are stored under the matching
Hamiltonian directory. A successful existing log is skipped, while a failed
or incomplete log is rerun.

## Metrics

The plot scripts extract these values from each log's IPC JSON block:

- `degree_1`
- `degree_2`
- `degree_3`
- `total_capacity`

Plots report means across available seeds with standard-deviation error bars.
Logs without a valid IPC JSON block are skipped.

## Generate plots

Run from the repository root:

```bash
python results/benchmarks/ipc/precision/plot.py
python results/benchmarks/ipc/depth/plot.py
python results/benchmarks/ipc/window/plot.py
python results/benchmarks/ipc/plot_max_delay.py
python results/benchmarks/ipc/plot_obs.py
python results/benchmarks/ipc/plot_obs_sp.py
```

The scripts save figures under `results/benchmarks/ipc/plots/` and the
experiment-specific plot folders.
