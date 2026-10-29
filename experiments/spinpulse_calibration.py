"""
SpinPulse Hardware realistic QRC oeprating point.

For the SpinPulse settings of a ``qrc`` command, reports:

1. the native gates: pulse duration and average gate fidelity, without noise and with the
   configured noise (SpinPulse's own ``mean_fidelity``, as in its calibration example);
2. the pulse timeline of one window: encoding block, reservoir step, whole window, in time
   steps and, for ``--hardware realistic``, in us;
3. the window duration relative to T2*, and a rough window fidelity (product of the gate
   fidelities, idle time not included).

    python -m experiments.spinpulse_calibration qrc --backend spinpulse --hardware realistic \
        --hamiltonian tfim --observables Z+ZZ --window-length 15 --reset-inputs
"""

import argparse
import warnings

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, average_gate_fidelity

from experiments.cli import FEATURE_SETS, build_parser, build_protocol
from qrc.backends.spinpulse import SpinPulseBackend

try:
    from spin_pulse import ExperimentalEnvironment, PulseCircuit
    from spin_pulse.environment.noise import NoiseType
except ImportError:  # the backend itself raises a clear error
    pass

# name: (number of qubits, gate)
GATES = {
    "X (rx pi)": (1, lambda qc: qc.rx(np.pi, 0)),
    "Z (rz pi)": (1, lambda qc: qc.rz(np.pi, 0)),
    "RZZ(pi/2)": (2, lambda qc: qc.rzz(np.pi / 2, 0, 1)),
}


def gate_report(backend, samples=200):
    """``{gate: (duration in steps, fidelity without noise, fidelity with noise or None)}``."""
    c = backend.config
    report = {}
    for name, (n, add_gate) in GATES.items():
        qc = QuantumCircuit(n)
        add_gate(qc)
        specs = backend._hardware_specs(n)
        pulse = PulseCircuit.from_circuit(specs.gate_transpile(qc), hardware_specs=specs)
        ideal = average_gate_fidelity(Operator.from_circuit(pulse.to_circuit()), Operator(qc))
        noisy = None
        if c.noisy:
            # traces as long as the gate, so every sample regenerates them: independent samples
            duration = pulse.duration + pulse.duration % 2
            if c.noise_type == "white":
                segment = 1  # required by SpinPulse
            else:
                segment = c.segment_duration or duration
                segment += segment % 2  # pink noise needs an even segment
                if c.noise_type == "quasistatic" and duration % segment:
                    segment = duration  # as in the backend: one constant value over the gate
            env = ExperimentalEnvironment(
                hardware_specs=specs,
                noise_type=NoiseType[c.noise_type.upper()],
                T2S=c.T2S,
                TJS=c.TJS if n == 2 else None,
                duration=duration,
                only_idle=c.only_idle,
                segment_duration=segment,
                seed=c.seed,
            )
            with warnings.catch_warnings():  # "more samples than one trace": intended
                warnings.simplefilter("ignore")
                noisy = pulse.mean_fidelity(env, qc, num_samples=samples, progress_bar=False)
        report[name] = (pulse.duration, ideal, noisy)
    return report


def timeline_report(protocol, args):
    """Pulse timeline of one window of the protocol's reservoir."""
    backend = protocol.backend
    input_width = protocol.encoder.input_width(len(FEATURE_SETS[args.feature_set]))
    tl = backend._timeline_for(protocol.encoder, protocol.reservoir, input_width)
    ops = tl.specs.gate_transpile(protocol.reservoir.circuit()).count_ops()
    return dict(
        encoding=tl.enc_duration,
        reservoir=tl.res_duration,
        step=tl.step_duration,
        window=args.window_length * tl.step_duration,
        rzz_per_step=ops.get("rzz", 0),
        one_qubit_per_step=sum(v for k, v in ops.items() if k != "rzz") + tl.n_inputs,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--samples", type=int, default=200, help="noise samples per gate fidelity")
    own, qrc_argv = parser.parse_known_args(argv)
    args = build_parser().parse_args(qrc_argv)
    if args.command != "qrc" or args.backend != "spinpulse":
        parser.error("give a `qrc --backend spinpulse ...` command")

    protocol, _ = build_protocol(args)
    backend: SpinPulseBackend = protocol.backend
    c = backend.config
    step_ns = c.time_step_ns

    def fmt(steps):
        return f"{steps:>7,} steps" + ("" if step_ns is None else f" = {steps * step_ns / 1000:8.2f} us")

    print(f"hardware: {args.hardware}" + ("" if step_ns is None else f" (time step {step_ns:g} ns)"))
    print(f"fields: B = {c.B_field:.4g}, delta = {c.delta:.4g}, J = {c.J_coupling:.4g}, ramp = {c.ramp_duration}")
    print(f"noise: {c.noise_type}, T2* = {fmt(round(c.T2S)) if c.noisy else '-'}, "
          f"TJ* = {fmt(round(c.TJS)) if c.noisy and c.TJS else 'off'}")

    gates = gate_report(backend, own.samples)
    print("\ngate          duration                      F (no noise)   F (noise)")
    for name, (duration, ideal, noisy) in gates.items():
        print(f"{name:<12}{fmt(duration):<30}{ideal:>12.6f}" + ("" if noisy is None else f"{noisy:>12.5f}"))

    t = timeline_report(protocol, args)
    print(f"\nwindow of {args.window_length} transactions, {protocol.reservoir.num_qubits} qubits")
    print(f"  encoding block   {fmt(t['encoding'])}")
    print(f"  reservoir step   {fmt(t['reservoir'])}   ({t['rzz_per_step']} rzz + {t['one_qubit_per_step']} 1q gates)")
    print(f"  one transaction  {fmt(t['step'])}")
    print(f"  whole window     {fmt(t['window'])}")
    if c.noisy:
        print(f"  window / T2*     {t['window'] / c.T2S:.2f}")
        f1, f2 = gates["X (rx pi)"][2], gates["RZZ(pi/2)"][2]
        estimate = f1 ** (t["one_qubit_per_step"] * args.window_length) * f2 ** (t["rzz_per_step"] * args.window_length)
        print(f"  rough window fidelity (gates only) {estimate:.3f}")


if __name__ == "__main__":
    main()