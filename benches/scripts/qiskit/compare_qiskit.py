#!/usr/bin/env python3
"""Performance comparison: qcrypto (Rust) vs Qiskit Aer (Python/C++).

Methodological honesty
----------------------
A simulator comparison is only publishable if it benchmarks *equivalent tasks*
under declared conditions.  This script measures three workloads:

1. Build + evolve an N-qubit GHZ state (state vector).
2. Sample M shots from a fixed circuit.
3. Apply individual gates (X, CX, CCX) to an N-qubit state.

What this script deliberately does **not** do:

- Does **not** benchmark noisy density-matrix simulation: qcrypto and Aer
  handle noise at different abstraction levels (Aer injects errors inline
  during circuit execution; qcrypto applies channels as separate operations),
  so there is no single equivalent task to compare.
- Does **not** compare QKD protocols (BB84, etc.): Aer does not implement them.
- Does **not** hide cases where qcrypto is slower.

Methodological decisions for the paper
---------------------------------------
- ``optimization_level=0`` for **all** Aer benchmarks: disables transpilation
  passes (gate fusion, pattern simplification).  Without this, recognisable
  circuits such as GHZ may be simplified before simulation, artificially
  favouring Aer.
- ``bench_gates`` uses ``set_statevector`` to load a pre-computed state before
  timing, mirroring ``iter_batched`` in criterion: only the gate cost is
  measured, not state preparation.  Declare this in the paper.
- ``bench_sampling`` measures full n-qubit bitstring sampling (2^N outcomes).
  qcrypto's ``Sampler`` targets qubit 0 only (2 outcomes); the qubit-count
  axis reflects probability-distribution computation cost (O(2^N)), not the
  number of possible outcomes.  State this distinction in the paper.
- qcrypto is measured with criterion (rigorous, subtracts overhead).
  Aer is measured with timeit.  Declare both methods in the paper.

Usage
-----
::

    uv run compare_qiskit.py --qubits 2 3 4 5 6 8 10 12 --repeats 20
"""

from __future__ import annotations

import argparse
import csv
import platform
import statistics
import sys
import timeit
from dataclasses import dataclass

try:
    import numpy as np
    import qiskit
    import qiskit_aer
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
except ImportError:
    sys.exit("Missing dependencies. Add with:\n  uv add qiskit qiskit-aer numpy")


@dataclass
class Measurement:
    task: str
    qubits: int
    shots: int
    median_s: float
    mean_s: float
    lower_s: float
    upper_s: float
    stdev_s: float
    repeats: int


# ---------------------------------------------------------------------------
# Statistics utility
# ---------------------------------------------------------------------------


def compute_confidence_interval(
    samples: list[float], iterations: int = 1000, conf: float = 0.95
) -> tuple[float, float]:
    """Compute confidence interval for the median using bootstrapping."""
    if len(samples) < 2:
        return samples[0], samples[0]

    boot_medians = []
    for _ in range(iterations):
        resample = np.random.choice(samples, size=len(samples), replace=True)
        boot_medians.append(np.median(resample))

    lower = np.percentile(boot_medians, (1 - conf) / 2 * 100)
    upper = np.percentile(boot_medians, (1 + conf) / 2 * 100)
    return float(lower), float(upper)


# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------


def build_ghz_circuit(n: int) -> QuantumCircuit:
    """Return an n-qubit GHZ circuit: H on q0, chain of CNOTs."""
    qc = QuantumCircuit(n)
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    return qc


def build_superposition_circuit(n: int) -> QuantumCircuit:
    """Return an n-qubit circuit with H on all qubits."""
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    return qc


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------


def time_callable(fn, repeats: int) -> tuple[float, float, float, float, float]:
    """Run ``fn()`` *repeats* times; return (median, mean, lower, upper, stdev) in seconds."""
    samples = [timeit.timeit(fn, number=1) for _ in range(repeats)]
    median = statistics.median(samples)
    mean = statistics.fmean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    lower, upper = compute_confidence_interval(samples)
    return median, mean, lower, upper, stdev


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def bench_statevector(qubits: list[int], repeats: int) -> list[Measurement]:
    """GHZ state vector benchmark."""
    sim = AerSimulator(method="statevector")
    out: list[Measurement] = []
    for n in qubits:
        qc = build_ghz_circuit(n)
        qc.save_statevector()  # type: ignore[attr-defined]
        qc_t = transpile(qc, sim, optimization_level=0)

        def task(qc=qc_t):
            sim.run(qc).result()

        task()  # warmup
        med, mean, low, up, sd = time_callable(task, repeats)
        out.append(
            Measurement("statevector_ghz", n, 0, med, mean, low, up, sd, repeats)
        )
        print(f"  [SV-GHZ ] {n:2d} qubits: median {med * 1e3:8.3f} ms")
    return out


def bench_sampling(
    qubits: list[int], shots_list: list[int], repeats: int
) -> list[Measurement]:
    """Sampling benchmark with ``optimization_level=0``."""
    sim = AerSimulator(method="statevector")
    out: list[Measurement] = []
    for n in qubits:
        qc = build_superposition_circuit(n)
        qc.measure_all()
        qc_t = transpile(qc, sim, optimization_level=0)
        for shots in shots_list:

            def task(qc=qc_t, s=shots):
                sim.run(qc, shots=s).result().get_counts()

            task()  # warmup
            med, mean, low, up, sd = time_callable(task, repeats)
            out.append(
                Measurement("sampling", n, shots, med, mean, low, up, sd, repeats)
            )
            print(
                f"  [SMP    ] {n:2d} qubits, {shots:>9d} shots: median {med * 1e3:8.3f} ms"
            )
    return out


def bench_gates(qubits: list[int], repeats: int) -> list[Measurement]:
    """Apply X, CX, CCX to a pre-prepared N-qubit state.

    Uses ``set_statevector`` to load a pre-computed state (H on all qubits)
    before timing, so only the gate cost is measured — equivalent to
    ``iter_batched`` in criterion.  ``optimization_level=0`` prevents Aer from
    fusing or eliminating the target gate.
    """
    sim = AerSimulator(method="statevector")
    out: list[Measurement] = []

    # Pre-compute the initial state (H on all qubits) for each size.
    print("  Preparing initial states...")
    initial_svs: dict[int, object] = {}
    for n in qubits:
        prep = QuantumCircuit(n)
        for q in range(n):
            prep.h(q)
        prep.save_statevector()  # type: ignore[attr-defined]
        initial_svs[n] = sim.run(prep).result().get_statevector()

    gates_spec = [
        ("X", 1, lambda qc: qc.x(0)),
        ("CX", 2, lambda qc: qc.cx(0, 1)),
        ("CCX", 3, lambda qc: qc.ccx(0, 1, 2)),
    ]

    for gate_name, min_q, apply_fn in gates_spec:
        for n in [q for q in qubits if q >= min_q]:
            sv = initial_svs[n]
            qc = QuantumCircuit(n)
            qc.set_statevector(sv)  # type: ignore[attr-defined]
            apply_fn(qc)
            qc.save_statevector()  # type: ignore[attr-defined]
            qc_t = transpile(qc, sim, optimization_level=0)

            def task(qc=qc_t):
                sim.run(qc).result()

            task()  # warmup
            med, mean, low, up, sd = time_callable(task, repeats)
            out.append(
                Measurement(f"gate_{gate_name}", n, 0, med, mean, low, up, sd, repeats)
            )
            print(f"  [GATE-{gate_name:3s}] {n:2d} qubits: median {med * 1e6:8.3f} µs")
    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_csv(path: str, rows: list[Measurement]) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "simulator",
                "task",
                "qubits",
                "shots",
                "median_s",
                "mean_s",
                "lower_s",
                "upper_s",
                "stdev_s",
                "repeats",
            ]
        )
        for m in rows:
            w.writerow(
                [
                    "qiskit_aer",
                    m.task,
                    m.qubits,
                    m.shots,
                    f"{m.median_s:.9f}",
                    f"{m.mean_s:.9f}",
                    f"{m.lower_s:.9f}",
                    f"{m.upper_s:.9f}",
                    f"{m.stdev_s:.9f}",
                    m.repeats,
                ]
            )


def write_environment(path: str) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        qiskit_v = qiskit.__version__
        aer_v = qiskit_aer.__version__
    except Exception:
        qiskit_v = "unknown"
        aer_v = "unknown"
    with open(path, "w") as f:
        f.write("# Qiskit comparison environment\n")
        f.write(f"qiskit_version = {qiskit_v}\n")
        f.write(f"qiskit_aer_version = {aer_v}\n")
        f.write(f"python_version = {platform.python_version()}\n")
        f.write(f"platform = {platform.platform()}\n")
        f.write(f"processor = {platform.processor()}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--qubits",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 8, 10, 12, 14],
        help="Qubit counts for state-vector and gate benchmarks.",
    )
    ap.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=[100_000, 1_000_000],
    )
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--out", default="../data/qiskit/aer_results.csv")
    ap.add_argument("--env-out", default="../data/qiskit/aer_environment.txt")
    ap.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["ghz", "sampling", "gates"],
        help="Skip one or more benchmarks.",
    )
    args = ap.parse_args()
    skip = set(args.skip)

    print("=" * 60)
    print("Qiskit Aer comparison benchmark  (optimization_level=0)")
    print(f"  state-vector / gates qubits: {args.qubits}")
    print(f"  sampling shots: {args.shots}")
    print(f"  repeats per point: {args.repeats}")
    if skip:
        print(f"  skipping: {skip}")
    print("=" * 60)

    rows: list[Measurement] = []

    if "ghz" not in skip:
        print("\n>> State vector (GHZ)")
        rows += bench_statevector(args.qubits, args.repeats)

    if "sampling" not in skip:
        print("\n>> Sampling")
        rows += bench_sampling(args.qubits, args.shots, args.repeats)

    if "gates" not in skip:
        print("\n>> Gates: X / CX / CCX by state size")
        rows += bench_gates(args.qubits, args.repeats)

    write_csv(args.out, rows)
    write_environment(args.env_out)
    print(f"\nResults written to {args.out}; environment info in {args.env_out}")


if __name__ == "__main__":
    main()
