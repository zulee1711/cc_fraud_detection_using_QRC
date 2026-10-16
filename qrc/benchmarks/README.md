`MemoryTask`, `NARMA10Task` and `ParityTask` are three tasks -- completely agnostic to our financial fraud context -- that can be used for benchmarking our QRC.

- `MemoryTask` is a short-term memory (STM) benchmark that checks the QRC retains some information from the last inputs it was given,
- `NARMA10Task` is a stress test that combines long-range dependencies, non-linear interactions and sensitivity to the consistency of state evolution over time, and
- `ParityTask` checks that the reservoir injects non-linearity.

They are the three tasks that are implemented in the QRC-Lab toolbox.
