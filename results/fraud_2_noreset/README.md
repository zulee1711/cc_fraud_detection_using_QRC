# fraud_2_noreset — the fraud_2 sweep without input reset

**Question.** In the fraud_2 setting, how fast does the sliding-window QRC degrade with window length when the
input qubit is not reset between encoding steps?

**Setup.** Identical to [fraud_2](../fraud_2/README.md) (data `data/sc3_c1000_x6`, feature set 1, 1 input + 3 memory
qubits, numpy exact, L = 1, 3, 5, 10, 15, 20, 30, seeds 42–51) except: no input reset, two reservoirs (random
circuit depth 4, TFIM at CLI defaults), XYZ-strings readout only (45 observables, 3 settings), exact and 10⁵ / 10⁴
shots. Reset QRCs, logistic, ESN-45 and LSTM are read from `../fraud_2/runs`. Details in
[`experiments/fraud_2_noreset/run.py`](../../experiments/fraud_2_noreset/run.py).

## Reproduce

```bash
.venv/bin/python -m experiments.fraud_2_noreset.run check     # readout variants == NumpyBackend(reset_inputs=False)
experiments/fraud_2_noreset/run_jobs.sh 6                     # the 140 jobs of jobs.txt, 6 in parallel (~70 min, M-series laptop)
.venv/bin/python results/fraud_2_noreset/aggregate.py         # runs.csv, tables.md, noreset_ap_vs_L.{pdf,png}
```

## Findings (test customer AP, exact, mean over 10 seeds; `tables.md`)

- **Up to L = 5 the reset barely matters** (L = 5: TFIM 0.231 vs 0.242, random 0.235 vs 0.252; both still above
  logistic 0.228).
- **From L = 10 the no-reset QRC collapses** while the reset one keeps improving: TFIM 0.166 / 0.094 / 0.072 / 0.051
  at L = 10 / 15 / 20 / 30 vs 0.351 / 0.429 / 0.481 / 0.538 with reset (paired Δ −0.19 to −0.49, p < 1e-8); random
  circuit 0.200 / 0.111 / 0.087 / 0.050 vs ~0.35. Every no-reset point from L = 10 is below logistic.
- **It loses the current transaction too:** amount AP (decided by the last transaction) falls from 0.98 at L = 5 to
  0.84 / 0.51 / 0.37 / 0.22 (TFIM) at L = 10–30; with reset it stays at 0.99. The unitary evolution scrambles the
  latest input into an ever larger history that the 45 observables cannot unmix.
- **Shot noise does not change the picture**: at 10⁵ and 10⁴ shots the gaps shrink only because the reset QRC also
  loses accuracy; no-reset customer AP is almost unchanged (it was already near the floor).
