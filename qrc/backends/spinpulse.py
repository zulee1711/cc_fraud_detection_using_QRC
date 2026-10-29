"""
SpinPulse backend: pulse-level noisy simulation of the QRC protocol.

Each window is the circuit :

enc_1 res_1 enc_2 res_2 ... enc_L res_L

laid out on one fixed pulse timeline. 

A noise realization is one stretch of SpinPulse noise
traces covering that timeline. With it fixed, every block is a definite unitary. 
The features are the exact expectation values averaged over noise_samples realizations,
optionally followed by finite-shot sampling ( for precision > 0).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.providers.fake_provider import GenericBackendV2     # for reproducible compilation target
from qiskit.transpiler import PassManager, generate_preset_pass_manager
from qiskit.transpiler.passes import (
    Collect2qBlocks,    # groups connected two-qubit operations into blocks
    ConsolidateBlocks,  # combines operations in those blocks into larger operations
    Optimize1qGatesDecomposition,   # optimizes single-qubit gate deocomposition
)
from qiskit.quantum_info import Operator, PauliList, SparsePauliOp

from qrc.backends.base import Backend
from qrc.encodings import Encoding
from qrc.reservoirs import Reservoir

try:
    from spin_pulse import (
        DynamicalDecoupling,
        ExperimentalEnvironment,
        HardwareSpecs,
        PulseCircuit,
        Shape,
    )
    from spin_pulse.environment.noise import NoiseType
    from spin_pulse.transpilation.instructions import IdleInstruction
    from spin_pulse.transpilation.passes.rzz_echo import RZZEchoPass
    from spin_pulse.transpilation.pulse_sequence import PulseSequence
    from spin_pulse.transpilation.utils import propagate

    SPINPULSE_AVAILABLE = True
except ImportError:
    SPINPULSE_AVAILABLE = False

# combines groups of quantum gates into larger blocks so the qrc unitary matrix can be calculated faster
_FUSE_BLOCKS = PassManager([Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)])

NOISE_TYPES = ("none", "white", "pink", "quasistatic")
PULSE_SHAPES = ("gaussian", "square")
DD_MODES = ("none", "spin-echo", "full-drive")
RESET_METHODS = ("auto", "exact", "sampled")
EXACT_RESET_MAX_QUBITS = 6
NOISE_CALL_STRIDE = 1_000_003

_AXIS = {"rx": "x", "ry": "y", "rz": "z"}


# Simulation Settings
@dataclass(frozen=True) # config cannot be changed after object is created during the simulation running
class SpinPulseConfig:
    """
    Hardware, noise and sampling settings of the SpinPulse backend.

    Hardware (passed to spin_pulse.HardwareSpecs):
        B_field: maximal x/y drive
        delta: z detuning
        J_coupling: exchange coupling
        pulse_shape: gaussian or square control pulses
        dynamical_decoupling: none, spin-echo or full-drive on idle qubits.
        ramp_duration: idle time steps before and after every rzz pulse
        time_step_ns: physical length of one time step (None = arbitrary units);
            set by SpinPulseConfig.realistic(), used to report times in us

    Noise (passed to spin_pulse.ExperimentalEnvironment):
        noise_type: none, white, pink or quasistatic
        T2S: characteristic dephasing time of the qubits (shorter = stronger noise)
        TJS: characteristic time of the exchange-coupling noise | "None" disables it
        only_idle: apply the frequency noise to idle qubits only
        segment_duration: segment length of the noise generator: 
            - for pink noise : the low-frequency cutoff of pink noise (pink noise correlated across the window)
            - for quasistatic noise : the hold time of quasistatic noise (quasistatic noise constant qcross one shot)
            - for white noise : Ignored
            - default : "None" | uses one whole window

    Sampling:
        noise_samples: number of noise realizations averaged per window (ignored without noise, unless reset_inputs is on)
        precision: 
            - 0 for exact expectation values
            - the target standard error of each estimate | N = (1 / precision**2) shots are sampled per window
        seed: seeds the noise traces, the reset outcomes and the shot sampling
        reset_method: how reset_inputs is simulated
            - exact : density matrices, Tr_inputs(rho) (x) |0><0| (memory grows as 4^n per window)
            - sampled : a sampled measurement per realization (quantum trajectories)
            - auto (default) : exact up to EXACT_RESET_MAX_QUBITS qubits, sampled above
    """

    noise_type: str = "pink"
    T2S: float = 100.0
    TJS: Optional[float] = None
    only_idle: bool = False
    segment_duration: Optional[int] = None  # controls how long a noise pattern lasts before the generator's pattern repeats
    B_field: float = 1.0
    delta: float = 1.0  # must be >> J_coupling: rzz relies on detuning
    J_coupling: float = 0.05
    pulse_shape: str = "gaussian"
    dynamical_decoupling: str = "none"
    ramp_duration: int = 1  # idle time steps before and after every rzz pulse
    time_step_ns : Optional[float] = None  # physical length of one time step. None = arbitrary units
    noise_samples: int = 20
    precision: float = 0.0
    seed: Optional[int] = None
    reset_method : str = "auto" # input reset : exact (density matrices), sampled, auto
    max_chunk_bytes : int = 256* 2**20

    @classmethod
    def realistic(cls, step_ns=10.0, t2_us=10.0, tj_us=5.0, **overrides):
        """
        Silicon spin-qubit chip of SpinPulse's example ``ParametrizingfromQPUSpecs``.

        Gate times: X/Y 1 us, Z 0.1 us, RZZ(pi/2) 1 us (about 3 us with its X echoes).
        Pink noise with T2* = 10 us and TJ* = 5 us, which SpinPulse calibrates to a
        single-qubit fidelity above 0.9995 and a two-qubit fidelity of about 0.995.

        Args:
            step_ns: simulation time step in ns. The fields are rescaled so the gate times
                stay the same; 10 ns reproduces the 1 ns fidelities within 6e-4 at about a
                tenth of the cost (5 to 10 ns is safe, 20 ns distorts the 100 ns Z pulses).
            t2_us, tj_us: qubit and coupling dephasing times in us; ``tj_us=None`` turns
                the coupling noise off. Larger values mean a better chip.
            overrides: any other field, e.g. ``noise_type`` or ``noise_samples``.
        """
        coeff_duration = 5  # SpinPulse's default: a Gaussian pulse lasts 5 widths

        def width(gate_ns):  # Gaussian width, in time steps, of a gate lasting gate_ns
            return gate_ns / step_ns / coeff_duration

        settings = dict(
            # field strengths that give an R(pi) X/Y pulse of 1 us, R(pi) Z of 0.1 us and RZZ(pi/2) of 1 us
            B_field=np.pi / (np.sqrt(2 * np.pi) * width(1000)),
            delta=np.pi / (np.sqrt(2 * np.pi) * width(100)),
            J_coupling=np.pi / (2 * np.sqrt(2 * np.pi) * width(1000)),
            ramp_duration=max(1, round(5 / step_ns)),  # 5 ns ramps
            pulse_shape="gaussian",
            noise_type="pink",
            T2S=t2_us * 1000 / step_ns,
            TJS=None if tj_us is None else tj_us * 1000 / step_ns,
            # 1 ms pink-noise segment as in the example: keeps drifts slower than one window
            segment_duration=2 * round(1e6 / step_ns / 2),
            time_step_ns=step_ns,
        )
        settings.update(overrides)
        return cls(**settings)   

    # safety check to reject bad values immediately
    def __post_init__(self):
        for name, value, allowed in (
            ("noise_type", self.noise_type, NOISE_TYPES),
            ("pulse_shape", self.pulse_shape, PULSE_SHAPES),
            ("dynamical_decoupling", self.dynamical_decoupling, DD_MODES),
            ("reset_method", self.reset_method, RESET_METHODS),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        if self.noise_samples < 1:
            raise ValueError(f"noise_samples must be at least 1, got {self.noise_samples}")
        if self.max_chunk_bytes < 1:
            raise ValueError(f"max_chunk_bytes must be positive, got {self.max_chunk_bytes}")
        if self.precision < 0:
            raise ValueError("precision must be non-negative")

        if self.ramp_duration < 1:
            raise ValueError(f"ramp_duration must be at least 1, got {self.ramp_duration}")
        if self.time_step_ns is not None and self.time_step_ns <= 0:
            raise ValueError(f"time_step_ns must be positive, got {self.time_step_ns}")
    
    # check if HW noise is present
    @property
    def noisy(self) -> bool:
        return self.noise_type != "none"

# Plan of one window : Same for :
#   - evry window
#   - eve4ry noise realization
# Computed once per run
# encoding duration + reservoir duration

# Timeline of how simulation will run in time
@dataclass  # stores the results of preparing the encoder and reservoir for the HW
class _Timeline:
    """
    Everything that depends on the circuit but not on the data or the noise
    """

    specs: "HardwareSpecs"
    n_qubits: int
    n_inputs: int
    physical: list  # hardware qubit holding logical qubit q when the reservoir starts
    slots: list  # per encoding slot: (start, duration, {axis: unit-angle pulse shape})
    gates: list  # per input qubit: (slot, axis, (feature index, slope) or None, offset)
    enc_duration: int
    reservoir: "PulseCircuit"
    res_duration: int

    # encoding + reservoir time
    @property
    def step_duration(self) -> int:
        return self.enc_duration + self.res_duration


class SpinPulseBackend(Backend):
    """
    Pulse-level, hardware-noise backend built on SpinPulse.

    Per run the reservoir is compiled to pulses once. For each of noise_samples
    realizations, the noisy reservoir unitary of every timestep is built once and applied
    to all windows together.
    Encoding pulses (data dependent) are propagated per window.
    """

    def __init__(self, config: SpinPulseConfig = SpinPulseConfig(), reset_inputs: bool = False):
        if not SPINPULSE_AVAILABLE:
            raise ImportError("spin_pulse package is not installed")
        self.config = config
        self.reset_inputs = reset_inputs
        self._timeline_key = None
        self._timeline = None
        self.last_stderr = None     # stores uncertainty estimates from different noise realizations
        self._calls = 0 # each run_batch call draws its own noise
        self._noise_seed = config.seed if config.seed is not None else int(np.random.SeedSequence().generate_state(1)[0])

    def _hardware_specs(self, n_qubits: int) -> "HardwareSpecs":
        c = self.config
        dd = {
            "none": None,
            "spin-echo": DynamicalDecoupling.SPIN_ECHO,
            "full-drive": DynamicalDecoupling.FULL_DRIVE,
        }[c.dynamical_decoupling]
        specs = HardwareSpecs(
            num_qubits=n_qubits,
            B_field=c.B_field,
            delta=c.delta,
            J_coupling=c.J_coupling,
            rotation_shape=Shape(c.pulse_shape),
            dynamical_decoupling=dd,
            ramp_duration = c.ramp_duration,
        )
        # replaace random compiler with seeded one so that the same reservoir always becomes saame pulses
        # HardwareSpecs compiles against a GenericBackendV2 with random gate errors and an
        # unseeded router, so the same reservoir compiles differently on every call (other
        # swaps, other 1q decompositions, other pulse durations). Rebuild both passes seeded.
        seed = c.seed or 0  # config object with random seed, noise options and sampling settings
        # target hardware description that Qiskit's compiler needs
        target = GenericBackendV2(
            num_qubits=n_qubits,
            coupling_map=specs.coupling_map,
            basis_gates=specs.basis_gates,
            seed=seed,
        ).target
        # take original circ and prrepare it for the HW described by target
        specs.first_pass = generate_preset_pass_manager(
            target=target, optimization_level=0, seed_transpiler=seed
        )
        # rebuild SpinPulse's second stage (rzz echo + single qubit emrging) on the seeded target
        specs.second_pass = PassManager(
            [RZZEchoPass(), Optimize1qGatesDecomposition(target=target)]
        )
        return specs

    # creates Spinpulse's noise generator, configured for one window
    # noise generator for one complete input window
    def _environment(self, specs, window_duration: int, seed = None):
        """
        Noise environment whose traces cover exactly one window (rounded up to even).

        SpinPulse builds a pink trace by repeating one ``segment_duration`` pattern, so two
        slices of the same trace are shifted copies, not independent realizations. Every
        realization therefore regenerates the traces (``generate_time_traces``)."""
        c = self.config
        if not c.noisy:
            return None
        duration = window_duration + window_duration % 2
        if c.noise_type == "white":
            segment = 1  # required by SpinPulse
        else:
            segment = c.segment_duration or duration
            segment += segment % 2  # pink noise needs an even segment
            if c.noise_type == "quasistatic" and duration % segment:
                segment = duration  # SpinPulse needs whole segments; a longer hold is constant over the window anyway
        # returns object that generates noise traces : one list of frequency offsets dω(t) per qubit
        return ExperimentalEnvironment(
            hardware_specs=specs,
            noise_type=NoiseType[c.noise_type.upper()],
            T2S=c.T2S,
            TJS=c.TJS,
            duration=duration,
            only_idle=c.only_idle,
            segment_duration=segment,
            seed=seed,
        )

    # fixed plan of a inwdow : encoding pulse pla, angle rules, compiled reservoir
    def _build_timeline(self, encoder: Encoding, reservoir: Reservoir, input_width: int) -> _Timeline:
        n, n_in = reservoir.num_qubits, reservoir.num_input_qubits
        specs = self._hardware_specs(n)

        # Encoding: every rotation gets a fixed-length pulse, long enough for |angle| = pi
        # on its axis, so the timeline (and the reservoir's noise slices) is the same for
        # every window. Angles are wrapped to [-pi, pi]: R(a + 2 pi) = -R(a), a global phase.
        theta = ParameterVector("x", input_width)
        circuit = encoder.encode(theta)
        per_qubit = [[] for _ in range(n_in)]
        for instruction in circuit.data:
            name = instruction.operation.name
            if name not in _AXIS:
                raise ValueError(
                    f"SpinPulse encoding supports rx/ry/rz gates only, got {name!r}"
                )
            (qubit,) = instruction.qubits
            angle = instruction.operation.params[0]
            slope, offset = _linear_coefficients(angle, theta)
            per_qubit[circuit.find_bit(qubit).index].append((_AXIS[name], slope, offset))

        slots, gates, start = [], [[] for _ in range(n_in)], 0
        for k in range(max(len(g) for g in per_qubit)):
            axes = {g[k][0] for g in per_qubit if len(g) > k}
            shapes = {axis: _unit_pulse(axis, specs) for axis in axes}
            duration = max(len(s) for s in shapes.values())
            for q, g in enumerate(per_qubit):
                if len(g) > k:
                    axis, slope, offset = g[k]
                    gates[q].append((k, axis, slope, offset))
            slots.append((start, duration, shapes))
            start += duration

        # Reservoir: compiled to pulses once; noise is attached per realization and step.
        res_isa = specs.gate_transpile(reservoir.circuit())
        res_pulse = PulseCircuit.from_circuit(res_isa, hardware_specs=specs, exp_env=None)
        layout = res_isa.layout
        physical = list(layout.initial_index_layout()[:n]) if layout is not None else list(range(n))
        return _Timeline(specs, n, n_in, physical, slots, gates, start, res_pulse, res_pulse.duration)

    def _timeline_for(self, encoder, reservoir, input_width) -> _Timeline:
        key = (id(encoder), id(reservoir), input_width)
        if key != self._timeline_key:
            self._timeline = self._build_timeline(encoder, reservoir, input_width)
            self._timeline_key = key
        return self._timeline


    def run_batch(
        self,
        windows: np.ndarray,
        encoder: Encoding,
        reservoir: Reservoir,
        observables: PauliList,
    ) -> np.ndarray:
        windows = np.asarray(windows, dtype=float)
        if windows.ndim != 3:
            raise ValueError("windows must have shape (samples, window_length, features)")
        samples, window_length, num_features = windows.shape
        if samples == 0:
            return np.empty((0, len(observables)))
        if observables.num_qubits != reservoir.num_qubits:
            raise ValueError("observables must act on the reservoir's qubits")
        if self.config.precision > 0:
            _require_diagonal(observables)

        input_width = encoder.input_width(num_features)
        values = windows[..., np.arange(input_width) % num_features]
        tl = self._timeline_for(encoder, reservoir, input_width)
        c = self.config
        window_duration = window_length * tl.step_duration
        env = self._environment(tl.specs, window_duration, self._noise_seed)
        rng = np.random.default_rng(self._noise_seed + NOISE_CALL_STRIDE * self._calls)
        self._calls += 1
        # With input reset, small registers are propagated as density matrices, so the reset is
        # exact; larger ones sample the reset outcome per realization (quantum trajectories).
        exact_reset = self.reset_inputs and self._exact_reset(tl.n_qubits)
        sampled_reset = self.reset_inputs and not exact_reset
        # several realizations when there is noise to average or reset outcomes to sample
        n_realizations = c.noise_samples if (env is not None or sampled_reset) else 1
        obs_matrices = [SparsePauliOp(p).to_matrix(sparse=True) for p in observables]
        if exact_reset:
            obs_matrices = [op.toarray() for op in obs_matrices]

        dim = 2**tl.n_qubits
        # windows are processed in chunks whose states fit in max_chunk_bytes
        per_window = 16 * dim * (dim if exact_reset else 1) * 3  # complex128 state + temporaries
        chunk = max(1, c.max_chunk_bytes // per_window)
        total = np.zeros((len(observables), samples))
        total_sq = np.zeros_like(total)
        probabilities = np.zeros((dim, samples)) if c.precision > 0 else None
        for m in range(n_realizations):
            if env is not None and m > 0:
                env.generate_time_traces()  # fresh, independent noise for this realization
            # this realization's reservoir unitaries, shared by every window and chunk
            reservoir_unitaries = [
                self._reservoir_unitary(tl, env, t * tl.step_duration + tl.enc_duration)
                for t in range(window_length)
            ]
            for start in range(0, samples, chunk):
                part = slice(start, min(samples, start + chunk))
                expectations, populations = self._propagate(
                    values[part], tl, env, reservoir_unitaries, obs_matrices,
                    exact_reset, sampled_reset, rng,
                )
                total[:, part] += expectations
                total_sq[:, part] += expectations**2
                if probabilities is not None:
                    probabilities[:, part] += populations

        # window_duration = window_length * tl.step_duration
        # env = self._environment(tl.specs, window_duration)
        # rng = np.random.default_rng(c.seed)
        # n_realizations = c.noise_samples if (env is not None or self.reset_inputs) else 1
        # obs_matrices = [SparsePauliOp(p).to_matrix(sparse=True) for p in observables]

        # dim = 2**tl.n_qubits
        # total = np.zeros((len(observables), samples))
        # total_sq = np.zeros_like(total)
        # probabilities = np.zeros((dim, samples)) if c.precision > 0 else None
        # # multiple realizations when noise or input resets are enabled, otherwise one realization is enough
        # for m in range(n_realizations):
        #     if env is not None and m > 0:
        #         env.generate_time_traces()  # fresh, independent noise for this realization
        #     state = np.zeros((dim, samples), dtype=complex)
        #     state[0] = 1.0
        #     for t in range(window_length):
        #         t_enc = t * tl.step_duration
        #         if self.reset_inputs and t > 0:
        #             state = _reset_qubits(state, range(tl.n_inputs), tl.n_qubits, rng)
        #         state = self._apply_encoding(state, values[:, t], tl, env, t_enc)
        #         state = self._reservoir_unitary(tl, env, t_enc + tl.enc_duration) @ state

        #     expectations = np.stack(
        #         [np.real(np.sum(state.conj() * (op @ state), axis=0)) for op in obs_matrices]
        #     )
        #     total += expectations
        #     total_sq += expectations**2
        #     if probabilities is not None:
        #         probabilities += np.abs(state) ** 2

        mean = total / n_realizations
        if n_realizations > 1:
            var = np.maximum(total_sq / n_realizations - mean**2, 0) / (n_realizations - 1)
            self.last_stderr = np.sqrt(var).T
        else:
            self.last_stderr = np.zeros_like(mean.T)

        if probabilities is None:
            return mean.T
        shots = int(np.ceil(1 / c.precision**2))
        return _sample_diagonal_expectations(
            probabilities / n_realizations, observables, shots, rng
        )

    def _propagate(self, values, tl: _Timeline, env, reservoir_unitaries, obs_matrices,
                   exact_reset, sampled_reset, rng):
        """One noise realization for a chunk of windows: (expectations (K, n), populations (2^n, n)).

        ``values`` has shape (n, window_length, input_width); ``reservoir_unitaries`` holds the
        realization's reservoir unitary of every timestep."""
        samples = len(values)
        dim = 2**tl.n_qubits
        if exact_reset:
            rho = np.zeros((samples, dim, dim), dtype=complex)
            rho[:, 0, 0] = 1.0
            for t, U in enumerate(reservoir_unitaries):
                if t > 0:
                    rho = _reset_inputs_exact(rho, tl.n_inputs)
                for q, unitaries in self._encoding_unitaries(values[:, t], tl, env, t * tl.step_duration, samples):
                    rho = _apply_single_qubit_dm(rho, unitaries, q, tl.n_qubits)
                rho = U @ rho @ U.conj().T
            expectations = np.stack([np.einsum("ij,nji->n", op, rho).real for op in obs_matrices])
            return expectations, np.real(np.diagonal(rho, axis1=1, axis2=2)).T

        state = np.zeros((dim, samples), dtype=complex)
        state[0] = 1.0
        for t, U in enumerate(reservoir_unitaries):
            if sampled_reset and t > 0:
                state = _reset_qubits(state, range(tl.n_inputs), tl.n_qubits, rng)
            state = self._apply_encoding(state, values[:, t], tl, env, t * tl.step_duration)
            state = U @ state
        expectations = np.stack(
            [np.real(np.sum(state.conj() * (op @ state), axis=0)) for op in obs_matrices]
        )
        return expectations, np.abs(state) ** 2

    def _reservoir_unitary(self, tl: _Timeline, env, t_start: int) -> np.ndarray:
        """
        Noisy reservoir unitary for the noise slice starting at t_start. Can return the unitary
        for the reservoir associated with a particular timeline or a particular segment of the noise trace.
        """
        if env is not None:
            tl.reservoir.t_lab = t_start
            tl.reservoir.attach_time_traces(env)    # attach noise data to reservoir circuit
        elif hasattr(self, "_ideal_reservoir") and self._ideal_reservoir[0] is tl:
            return self._ideal_reservoir[1] # resuse the ideal unitary when noise is absent
        matrix = _circuit_unitary(tl.reservoir.to_circuit())
        if env is None:
            self._ideal_reservoir = (tl, matrix)
        return matrix

    # def _apply_encoding(self, state, step_values, tl: _Timeline, env, t_start: int):
    #     """
    #     Apply the (noisy) encoding pulses of one timestep to every window.
    #         state : current batch of quantum states, shape : (2^n, S)
    #         step_values : input values for the current timestep, one value per input qubit
    #         tl : timeline which specifies reservoir structure
    #         env : noise env
    #         t_start : start index of the current timestep's ecnoding block in the noise traces
    #     """
    #     samples = state.shape[1]
    #     for q in range(tl.n_qubits):
    #         noise = _trace_slice(env, tl.physical[q], t_start, tl.enc_duration)
    #         if q < tl.n_inputs: # idenitfy input qubits -> input-dependent rotation
    #             unitaries = _propagate_rotations(   # encoing operation
    #                 tl.gates[q], tl.slots, step_values, noise, tl.enc_duration,
    #                 only_idle=env is not None and env.only_idle,
    #             )
    #         else:   # memory qubits _. idle unitary evolution
    #             unitaries = np.broadcast_to(
    #                 _idle_unitary(noise, tl.enc_duration, tl.specs), (samples, 2, 2)
    #             )
    #         state = _apply_single_qubit(state, unitaries, q, tl.n_qubits)
    #     return state

    def _exact_reset(self, n_qubits: int) -> bool:
        """Whether the input reset is done exactly, with density matrices (see ``reset_method``)."""
        method = self.config.reset_method
        return method == "exact" or (method == "auto" and n_qubits <= EXACT_RESET_MAX_QUBITS)

    def _apply_encoding(self, state, step_values, tl: _Timeline, env, t_start: int):
        """
        Apply the (noisy) encoding pulses of one timestep to every window.
            state : current batch of quantum states, shape : (2^n, S)
            step_values : input values for the current timestep, one value per input qubit
            tl : timeline which specifies reservoir structure
            env : noise env
            t_start : start index of the current timestep's encoding block in the noise traces
        """
        for q, unitaries in self._encoding_unitaries(step_values, tl, env, t_start, state.shape[1]):
            state = _apply_single_qubit(state, unitaries, q, tl.n_qubits)
        return state

    def _encoding_unitaries(self, step_values, tl: _Timeline, env, t_start: int, samples: int):
        """Per qubit ``(q, unitaries of shape (samples, 2, 2))`` of one timestep's encoding block."""
        for q in range(tl.n_qubits):
            noise = _trace_slice(env, tl.physical[q], t_start, tl.enc_duration)
            if q < tl.n_inputs:  # input qubits: data-dependent rotation
                unitaries = _propagate_rotations(
                    tl.gates[q], tl.slots, step_values, noise, tl.enc_duration,
                    only_idle=env is not None and env.only_idle,
                )
            else:  # memory qubits: idle evolution
                unitaries = np.broadcast_to(
                    _idle_unitary(noise, tl.enc_duration, tl.specs), (samples, 2, 2)
                )
            yield q, unitaries


def _apply_single_qubit_dm(rho, unitaries, qubit: int, n_qubits: int):
    """U rho U^dagger with a per-window 2x2 unitary on ``qubit``; rho has shape (samples, 2^n, 2^n)."""
    samples, dim, _ = rho.shape

    def left(m):  # U m, acting on the row index of every window's matrix
        view = m.reshape(samples, 2 ** (n_qubits - 1 - qubit), 2, 2**qubit, dim)
        return np.einsum("sab,sibjk->siajk", unitaries, view).reshape(m.shape)

    rho = left(rho)  # U rho
    return left(rho.conj().transpose(0, 2, 1)).conj().transpose(0, 2, 1)  # (U (U rho)^dag)^dag = U rho U^dag


def _reset_inputs_exact(rho, n_inputs: int):
    """Reset the input qubits (the lowest bits) to |0>: rho -> Tr_inputs(rho) (x) |0><0|."""
    samples, dim, _ = rho.shape
    d_in = 2**n_inputs
    blocks = rho.reshape(samples, dim // d_in, d_in, dim // d_in, d_in)
    out = np.zeros_like(rho)
    out.reshape(samples, dim // d_in, d_in, dim // d_in, d_in)[:, :, 0, :, 0] = np.einsum(
        "saibi->sab", blocks
    )
    return out


def _circuit_unitary(circuit: QuantumCircuit) -> np.ndarray:
    """
    Convert a pulse-level quantum circuit into its overall unitary, presrving logical qubit ordering.
    """
    fused = _FUSE_BLOCKS.run(circuit)   # combines one-qubit and two-qubit pulse operations into larger blocks before calculating total unitary
    fused._layout = circuit.layout  # keep the transpiler layout so it is undone below
    return Operator.from_circuit(fused).data

def _linear_coefficients(angle, theta):
    """
    Return (slope over theta, offset) for an angle that is affine in one parameter.
    """
    if not hasattr(angle, "parameters"):
        return None, float(angle)
    (param,) = angle.parameters
    index = list(theta).index(param)
    offset = float(angle.bind({param: 0.0}))
    slope = float(angle.bind({param: 1.0})) - offset
    return (index, slope), offset

def _unit_pulse(axis: str, specs) -> np.ndarray:
    """
    Pulse envelope (per time step) producing a rotation of exactly 1 rad on axis,
    with the duration SpinPulse needs for an angle of pi.
    """
    instruction = specs.rotation_generator.from_angle(axis, [None], np.pi, specs)
    pulse = np.real(instruction.to_pulse())
    return pulse / pulse.sum()


def _trace_slice(env, qubit: int, start: int, length: int) -> np.ndarray:
    """
    Extract a selected time window of a qubit's noise trace.
    """
    if env is None:
        return np.zeros(length)
    return np.asarray(env.time_traces[qubit].values[start : start + length], dtype=float)


def _propagate_rotations(gates, slots, step_values, noise, duration, only_idle):
    """Exact time-step propagation of one input qubit's encoding pulses, for every window.

    Per time step the Hamiltonian is ``0.5 * (drive * sigma_axis + noise * Z)``, the same
    model SpinPulse uses; the step unitary has a closed form, vectorised over windows.
    """
    samples = step_values.shape[0]
    drive = np.zeros((3, samples, duration))  # x, y, z drive coefficients
    active = np.zeros(duration, dtype=bool)
    for k, axis, coefficients, offset in gates:
        start, length, shapes = slots[k]
        angle = np.full(samples, offset)
        if coefficients is not None:
            index, slope = coefficients
            angle = angle + slope * step_values[:, index]
        angle = (angle + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi)
        shape = shapes[axis]
        row = "xyz".index(axis)
        drive[row, :, start : start + len(shape)] += angle[:, None] * shape[None, :]
        active[start : start + len(shape)] = True
    noise = np.where(active, 0.0, noise) if only_idle else noise
    drive[2] += noise[None, :]

    unitary = np.broadcast_to(np.eye(2, dtype=complex), (samples, 2, 2)).copy()
    for tau in range(duration):
        vx, vy, vz = drive[0, :, tau], drive[1, :, tau], drive[2, :, tau]
        norm = np.sqrt(vx**2 + vy**2 + vz**2)
        cos, sinc = np.cos(norm / 2), np.sinc(norm / (2 * np.pi)) / 2  # sin(n/2)/n
        step = np.empty((samples, 2, 2), dtype=complex)
        step[:, 0, 0] = cos - 1j * sinc * vz
        step[:, 1, 1] = cos + 1j * sinc * vz
        step[:, 0, 1] = -1j * sinc * vx - sinc * vy
        step[:, 1, 0] = -1j * sinc * vx + sinc * vy
        unitary = step @ unitary
    return unitary


def _idle_unitary(noise, duration, specs) -> np.ndarray:
    """
    Constructs a time-evolution operator for an idle memory qubit.
    """
    qubit = QuantumCircuit(1).qubits[0]
    sequence = PulseSequence([IdleInstruction([qubit], duration=duration)]) # leave the specified qubit idle for 'duration'
    if specs is not None and specs.dynamical_decoupling is not None:
        sequence = sequence.to_dynamical_decoupling(specs)  # sequence to dynamically decoupled sequence
    sequence.attach_time_trace(noise, only_idle=False)      # attach time-dependent noise description to the pulse sequence
    H, coeff = sequence.to_hamiltonian()
    return propagate(H, coeff)


def _apply_single_qubit(state, unitaries, qubit: int, n_qubits: int):
    """
    Apply a per-window 2x2 unitary (shape (samples, 2, 2)) to each sample in a batch of quantum states.
    Tensor-manipulation routine that upplies the unitary to the correct qubit without explicitly
    constructing a full 2^n x 2^n operator.
    """
    samples = state.shape[1]
    view = state.reshape(2 ** (n_qubits - 1 - qubit), 2, 2**qubit, samples)
    return np.einsum("sab,ibjs->iajs", unitaries, view).reshape(state.shape)


def _reset_qubits(state, qubits, n_qubits: int, rng):
    """
    Reset qubits selected qubits to |0> by sampling a measurement per window (quantum trajectory approach).
    """
    samples = state.shape[1]    # how many states the function needs to reset
    for q in qubits:
        view = state.reshape(2 ** (n_qubits - 1 - q), 2, 2**q, samples).copy()
        p1 = np.sum(np.abs(view[:, 1]) ** 2, axis=(0, 1))
        # simulate randomness of quantum measurement
        outcome = rng.random(samples) < p1
        kept = np.where(outcome[None, None, :], view[:, 1], view[:, 0])
        norm = np.sqrt(np.where(outcome, p1, 1 - p1))
        view[:, 0] = kept / np.where(norm > 0, norm, 1)[None, None, :]
        view[:, 1] = 0
        # selected qubits are back to |0>, the other retain the conditional state
        state = view.reshape(state.shape)
    return state


def _require_diagonal(observables: PauliList):
    """
    Check that the requested observables contain only I and Z operators.
    """
    if np.any(observables.x):
        bad = [p.to_label() for p, x in zip(observables, observables.x) if x.any()]
        raise NotImplementedError(
            f"Finite-shot SpinPulse runs support only I/Z observables, got {bad}. "
            "Use precision=0 for exact expectation values of any Pauli."
        )


def _sample_diagonal_expectations(probabilities, observables: PauliList, shots: int, rng):
    """
    Sample shots bitstrings per window and estimate I/Z observables from them.
    """
    dim, samples = probabilities.shape
    probabilities = probabilities / probabilities.sum(axis=0, keepdims=True)
    counts = np.stack([rng.multinomial(shots, probabilities[:, s]) for s in range(samples)])
    return counts @ _diagonal_signs(observables, dim).T / shots


def _diagonal_signs(observables: PauliList, dim: int) -> np.ndarray:
    """
    Precalculate the +1/-1 result of every I/Z observable for every possible computational basis state.
    """
    bits = (np.arange(dim)[:, None] >> np.arange(observables.num_qubits)[None, :]) & 1
    return np.stack([(-1.0) ** (bits @ z.astype(int)) for z in observables.z])
