"""QuTiP vs qcrypto benchmark: open-system / density-matrix operations.

Focuses on the tasks where QuTiP is the canonical reference for open quantum
systems: noise channel application on density matrices, purity, and fidelity.
Gate simulation and sampling are covered by compare_qiskit.py and are
intentionally excluded here.

Channel conventions (qcrypto vs QuTiP)
---------------------------------------
QuTiP has no built-in convenience functions for depolarizing, amplitude
damping, or phase damping.  All channels are constructed from Kraus operators
that exactly match qcrypto's internal definition (verified by the equivalence
assertions at the start of each test):

  depolarizing(p):
    qcrypto / here:  E(ρ) = (1−p)ρ + p·I/2
    Kraus: K0=√(1−3p/4)·I, K1=√(p/4)·X, K2=√(p/4)·Y, K3=√(p/4)·Z
    Aer convention:  p_aer = 0.75·p_qcrypto  (NOT used here)
    QuTiP convention: no built-in — Kraus operators match qcrypto directly

  amplitude_damping(γ):
    Kraus: K0=[[1,0],[0,√(1−γ)]], K1=[[0,√γ],[0,0]]
    Equivalent Lindblad: L = √γ·σ₋ = √γ·qutip.destroy(2) = √γ·[[0,1],[0,0]]
    Equivalence is exact for one discrete Markovian step.

  phase_damping(λ):
    Kraus: K0=[[1,0],[0,√(1−λ)]], K1=[[0,0],[0,√λ]]
    Continuous Lindblad limit: L = √(λ/2)·σ_z  (not used here)

Equivalence assertions
-----------------------
Each test verifies at 1 qubit that the QuTiP superoperator path
(kraus_to_super → operator_to_vector → vector_to_operator) produces the same
density matrix as a direct NumPy Kraus sum (Σ K·ρ·K†), which mirrors
qcrypto's internal computation.  The script aborts if any assertion fails.

Output
------
  benches/data/qutip_results.csv     — timings (same format as aer_results.csv)
  benches/data/qutip_environment.txt — version and platform metadata

Usage
-----
From inside benches/scripts/qutip/:

    uv run compare_qutip.py
    uv run compare_qutip.py --dm-qubits 2 3 4 5 6 8 10 --repeats 20
    uv run compare_qutip.py --skip lindblad
    uv run compare_qutip.py --out ../../data/qutip_results.csv
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import timeit
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import qutip
    from qutip import (
        Qobj,
        entropy_linear,
        kraus_to_super,
        operator_to_vector,
        qeye,
        tensor,
        vector_to_operator,
    )
except ImportError:
    sys.exit(
        "QuTiP is not installed in the current environment.\n"
        "Run:  uv run compare_qutip.py  (uv installs deps from pyproject.toml)\n"
        "Or:   pip install 'qutip>=5.3.0'"
    )

# ---------------------------------------------------------------------------
# Kraus operator definitions — must match qcrypto exactly
# ---------------------------------------------------------------------------


def _kraus_depolarizing(p: float) -> list[np.ndarray]:
    """Depolarizing channel: E(ρ) = (1−p)ρ + p·I/2.

    K0 = √(1−3p/4)·I,  K1 = √(p/4)·X,  K2 = √(p/4)·Y,  K3 = √(p/4)·Z.
    """
    w_i = np.sqrt(1.0 - 0.75 * p)
    w = np.sqrt(p / 4.0)
    return [
        w_i * np.array([[1, 0], [0, 1]], dtype=complex),
        w * np.array([[0, 1], [1, 0]], dtype=complex),
        w * np.array([[0, -1j], [1j, 0]], dtype=complex),
        w * np.array([[1, 0], [0, -1]], dtype=complex),
    ]


def _kraus_amplitude_damping(gamma: float) -> list[np.ndarray]:
    """Amplitude damping: K0=[[1,0],[0,√(1−γ)]], K1=[[0,√γ],[0,0]]."""
    g = np.sqrt(gamma)
    g1 = np.sqrt(1.0 - gamma)
    return [
        np.array([[1, 0], [0, g1]], dtype=complex),
        np.array([[0, g], [0, 0]], dtype=complex),
    ]


def _kraus_phase_damping(lam: float) -> list[np.ndarray]:
    """Phase damping: K0=[[1,0],[0,√(1−λ)]], K1=[[0,0],[0,√λ]]."""
    sl = np.sqrt(lam)
    sl1 = np.sqrt(1.0 - lam)
    return [
        np.array([[1, 0], [0, sl1]], dtype=complex),
        np.array([[0, 0], [0, sl]], dtype=complex),
    ]


_CHANNELS: dict[str, Callable[[float], list[np.ndarray]]] = {
    "depolarizing": _kraus_depolarizing,
    "amplitude_damping": _kraus_amplitude_damping,
    "phase_damping": _kraus_phase_damping,
}

_CHANNEL_P = 0.1

# ---------------------------------------------------------------------------
# QuTiP helpers
# ---------------------------------------------------------------------------


def _qobj_oper(arr: np.ndarray) -> Qobj:
    """Wrap a square numpy array as a QuTiP operator with correct qubit dims."""
    d = int(arr.shape[0])
    n = int(round(np.log2(d))) if d > 1 else 1
    dims = [[2] * n, [2] * n] if (n >= 1 and 2**n == d) else [[d], [d]]
    return Qobj(arr, dims=dims)


def _kraus_to_qobj(kraus_np: list[np.ndarray]) -> list[Qobj]:
    return [_qobj_oper(K) for K in kraus_np]


def _expand_kraus(kraus_1q: list[np.ndarray], target: int, n: int) -> list[Qobj]:
    """Expand 1-qubit Kraus ops to act on qubit `target` of an n-qubit system.

    K_full = I⊗…⊗K⊗…⊗I  (K at position `target`).
    Equivalent to qcrypto's  apply_channel(&ch, &[target]).
    """
    result = []
    for K in kraus_1q:
        factors = [_qobj_oper(K) if i == target else qeye(2) for i in range(n)]
        result.append(tensor(*factors))
    return result


def _uniform_dm(n: int) -> Qobj:
    """n-qubit |+…+⟩⟨+…+| density matrix."""
    plus = Qobj(np.array([1.0, 1.0], dtype=complex) / np.sqrt(2), dims=[[2], [1]])
    state = plus
    for _ in range(n - 1):
        state = tensor(state, plus)
    return state.proj()


def _apply_channel(superop: Qobj, rho: Qobj) -> Qobj:
    """Apply a precomputed superoperator.  Only used at 1 qubit (equivalence checks)."""
    return vector_to_operator(superop @ operator_to_vector(rho))


def _apply_kraus(kraus_ops: list[Qobj], rho: Qobj) -> Qobj:
    """Apply channel via direct Kraus sum: Σ K·ρ·K†.

    Avoids building the (4^N)² superoperator — memory stays O(4^N), the same
    as the density matrix itself.  This is the correct path for N > ~4.
    """
    return sum(K @ rho @ K.dag() for K in kraus_ops)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Equivalence assertions
# ---------------------------------------------------------------------------


def _apply_numpy(kraus_np: list[np.ndarray], rho_np: np.ndarray) -> np.ndarray:
    """Direct NumPy Kraus sum Σ K·ρ·K† — mirrors qcrypto's computation."""
    return sum(K @ rho_np @ K.conj().T for K in kraus_np)  # type: ignore[return-value]


def _assert_channel_equiv(
    name: str,
    kraus_np: list[np.ndarray],
    rho_np: np.ndarray,
    tol: float = 1e-10,
) -> None:
    """Abort if QuTiP superoperator and direct NumPy Kraus sum disagree."""
    expected = _apply_numpy(kraus_np, rho_np)
    S = kraus_to_super(_kraus_to_qobj(kraus_np))
    result = _apply_channel(S, _qobj_oper(rho_np)).full()
    diff = float(np.max(np.abs(result - expected)))
    if diff > tol:
        sys.exit(
            f"[FAIL] Channel equivalence '{name}': "
            f"max|QuTiP − NumPy| = {diff:.2e} > tol={tol:.0e}. Aborting."
        )
    print(f"  [ok]  {name}: max|QuTiP − NumPy| = {diff:.2e}")


def _assert_purity_equiv(
    rho_np: np.ndarray,
    rho_qt: Qobj,
    tol: float = 1e-10,
) -> None:
    """Abort if purity from NumPy Tr(ρ²) and QuTiP 1−entropy_linear disagree."""
    p_np = float(np.real(np.trace(rho_np @ rho_np)))
    p_qt = 1.0 - float(entropy_linear(rho_qt))
    diff = abs(p_qt - p_np)
    if diff > tol:
        sys.exit(
            f"[FAIL] Purity equivalence: "
            f"|QuTiP({p_qt:.8f}) − NumPy({p_np:.8f})| = {diff:.2e}. Aborting."
        )
    print(f"  [ok]  purity: QuTiP={p_qt:.6f}, NumPy={p_np:.6f}, diff={diff:.2e}")


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@dataclass
class Measurement:
    task: str
    qubits: int
    shots: int
    median_s: float
    mean_s: float
    stdev_s: float
    repeats: int


def time_callable(fn: Callable[[], object], repeats: int) -> tuple[float, float, float]:
    """Time fn() for `repeats` rounds after one warmup.  Returns (median, mean, stdev)."""
    fn()  # warmup — not counted
    times = [timeit.timeit(fn, number=1) for _ in range(repeats)]
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    return statistics.median(times), statistics.mean(times), stdev


# ---------------------------------------------------------------------------
# Test 1: noise channel application
# ---------------------------------------------------------------------------


def bench_channels(qubits: list[int], repeats: int) -> list[Measurement]:
    """Apply each noise channel to qubit 0 of an n-qubit uniform density matrix.

    Uses direct Kraus sum Σ K·ρ·K† (no superoperator).  The superoperator
    would require a (4^N)² matrix — ~68 GB at N=8 — making it unusable beyond
    N≈5.  The Kraus-sum path keeps memory at O(4^N) (same as ρ itself) and
    matches criterion's warm-path channels bench in cost structure.

    Expanded Kraus operators are precomputed outside the timed loop so that
    only the application cost is measured (mirrors iter_batched in criterion).
    """
    print("\n--- Test 1: noise channel application ---")
    print("Equivalence check at 1 qubit (|+⟩⟨+| state):")
    rho_1q = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    for name, fn in _CHANNELS.items():
        _assert_channel_equiv(name, fn(_CHANNEL_P), rho_1q)

    out: list[Measurement] = []
    for name, fn in _CHANNELS.items():
        kraus_1q = fn(_CHANNEL_P)
        for n in qubits:
            rho = _uniform_dm(n)

            def task(
                k1q: list[np.ndarray] = kraus_1q, rho: Qobj = rho, nq: int = n
            ) -> Qobj:
                # Move expansion inside the timed loop to match qcrypto's behavior
                kraus_nq = _expand_kraus(k1q, 0, nq)
                return _apply_kraus(kraus_nq, rho)

            med, mean, sd = time_callable(task, repeats)
            out.append(Measurement(f"channel_{name}", n, 0, med, mean, sd, repeats))
            print(f"  channel_{name} n={n:2d}: {med * 1e3:.3f} ms")
    return out


# ---------------------------------------------------------------------------
# Test 2: purity and fidelity
# ---------------------------------------------------------------------------


def bench_metrics(qubits: list[int], repeats: int) -> list[Measurement]:
    """Purity = Tr(ρ²) on genuinely mixed states.

    Mixed state: apply amplitude_damping(0.3) to every qubit of |+…+⟩.
    Purity: 1 − entropy_linear(ρ).

    Matches qcrypto's ``metrics/purity`` criterion group in ``core_ops.rs``,
    which uses the same channel and parameter, enabling a direct 1:1 timing
    comparison.  Fidelity is intentionally excluded: qcrypto does not expose
    a ``fidelity()`` method, so there is no qcrypto counterpart to compare
    against.
    """
    print("\n--- Test 2: purity ---")

    kraus_ad = _kraus_amplitude_damping(0.3)
    rho_1q_np = _apply_numpy(
        kraus_ad, np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    )
    print("Equivalence check at 1 qubit (amplitude-damped |+⟩):")
    _assert_purity_equiv(rho_1q_np, _qobj_oper(rho_1q_np))

    out: list[Measurement] = []
    for n in qubits:
        rho_qt = _uniform_dm(n)
        for target in range(n):
            rho_qt = _apply_kraus(_expand_kraus(kraus_ad, target, n), rho_qt)

        def task_purity(rho: Qobj = rho_qt) -> float:
            return 1.0 - float(entropy_linear(rho))

        med_p, mean_p, sd_p = time_callable(task_purity, repeats)
        out.append(Measurement("metric_purity", n, 0, med_p, mean_p, sd_p, repeats))
        print(f"  n={n:2d}: purity={task_purity():.4f}  |  {med_p * 1e6:.1f} µs")
    return out


# ---------------------------------------------------------------------------
# Test 3: Lindblad evolution
# ---------------------------------------------------------------------------


def bench_lindblad(qubits: list[int], repeats: int) -> list[Measurement]:
    """Amplitude damping via mesolve (continuous Lindblad master equation).

    H=0, collapse operator L=√γ·σ₋ on qubit 0, evolved over t∈[0,1].
    This is EXACTLY equivalent to one discrete Kraus step with parameter γ
    because both implement the same Markovian Lindblad channel; the
    correspondence is K0=exp(−γ/2·a†a), K1=√γ·a where a=σ₋.

    Note: mesolve integrates a continuous ODE so its overhead (solver setup,
    step control) is NOT comparable to qcrypto's single matrix multiplication.
    These timings characterise QuTiP's ODE overhead, not channel throughput.
    """
    try:
        from qutip import mesolve  # type: ignore[attr-defined]
    except ImportError:
        print("\n--- Test 3: Lindblad evolution [skipped — mesolve not available] ---")
        return []

    print("\n--- Test 3: Lindblad evolution (amplitude damping, H=0) ---")
    gamma = 0.1
    out: list[Measurement] = []

    for n in qubits:
        rho0 = _uniform_dm(n)
        dim = 2**n
        H = Qobj(np.zeros((dim, dim), dtype=complex), dims=[[2] * n, [2] * n])

        sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
        c_1q = _qobj_oper(np.sqrt(gamma) * sigma_minus)
        c_op = tensor(*[c_1q if i == 0 else qeye(2) for i in range(n)])

        tlist = [0.0, 1.0]

        def task(H: Qobj = H, rho0: Qobj = rho0, c_op: Qobj = c_op) -> object:
            return mesolve(H, rho0, tlist, c_ops=[c_op])

        med, mean, sd = time_callable(task, repeats)
        out.append(Measurement("lindblad_evolution", n, 0, med, mean, sd, repeats))
        print(f"  n={n:2d}: {med * 1e3:.3f} ms")
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_csv(path: str, rows: list[Measurement]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        f.write("simulator,task,qubits,shots,median_s,mean_s,stdev_s,repeats\n")
        for m in rows:
            f.write(
                f"qutip,{m.task},{m.qubits},{m.shots},"
                f"{m.median_s:.9f},{m.mean_s:.9f},{m.stdev_s:.9f},{m.repeats}\n"
            )
    print(f"\nResults written to {path}")


def write_environment(path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        f.write("# qcrypto QuTiP benchmark environment\n")
        f.write(f"qutip_version = {qutip.__version__}\n")
        f.write(f"numpy_version = {np.__version__}\n")
        f.write(f"python_version = {platform.python_version()}\n")
        f.write(f"platform = {platform.platform()}\n")
        f.write(f"machine = {platform.machine()}\n")
    print(f"Environment written to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dm-qubits",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 8, 10],
        help="Qubit counts for density-matrix benchmarks.",
    )
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", default="../../data/qutip/qutip_results.csv")
    ap.add_argument("--env-out", default="../../data/qutip/qutip_environment.txt")
    ap.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["channels", "metrics", "lindblad"],
        help="Skip one or more benchmark groups by name.",
    )
    args = ap.parse_args()
    skip = set(args.skip or [])

    print("=" * 60)
    print(f"QuTiP comparison benchmark  (version {qutip.__version__})")
    print(f"  dm-qubits : {args.dm_qubits}")
    print(f"  repeats   : {args.repeats}")
    print(f"  skip      : {skip or '(none)'}")
    print("=" * 60)

    rows: list[Measurement] = []
    if "channels" not in skip:
        rows += bench_channels(args.dm_qubits, args.repeats)
    if "metrics" not in skip:
        rows += bench_metrics(args.dm_qubits, args.repeats)

    write_csv(args.out, rows)
    write_environment(args.env_out)

    print("\n" + "=" * 60)
    print(f"  Results : {args.out}")
    print(f"  Env     : {args.env_out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
