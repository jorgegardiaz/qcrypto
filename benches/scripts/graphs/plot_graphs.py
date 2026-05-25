"""Analysis and figure generation for qcrypto benchmarks.

Data sources
------------
1. criterion JSON (``target/criterion/.../new/estimates.json``): timing +
   95 % CI for all qcrypto Rust benchmarks.
2. ``benches/data/qcrypto/``: auxiliary CSVs from the Rust bench binaries
   (theoretical memory, protocol correctness).
3. ``benches/data/qiskit/aer_results.csv``: Qiskit Aer timings from
   ``compare_qiskit.py``.
4. ``benches/data/qutip/qutip_results.csv``: QuTiP timings from
   ``compare_qutip.py``.

Figure output
-------------
Each figure is saved in three formats under ``benches/figures/<category>/``:

    figures/
    ├── qcrypto/
    │   ├── pdf/   ← vector, for inclusion in LaTeX
    │   ├── png/   ← raster preview
    │   └── eps/   ← alternative vector format
    ├── qiskit/    ← qcrypto vs Qiskit Aer comparisons
    │   ├── pdf/ png/ eps/
    └── qutip/     ← qcrypto vs QuTiP comparisons
        ├── pdf/ png/ eps/

Usage
-----
After running ``cargo bench`` and the Python comparison scripts::

    uv run plot_graphs.py
    uv run plot_graphs.py --criterion-root ../../../target/criterion \\
                      --data ../../data --out ../../figures \\
                      --aer ../../data/qiskit/aer_results.csv \\
                      --qutip ../../data/qutip/qutip_results.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

# --- Publication aesthetics -------------------------------------------------

COLOR_QCRYPTO = "tab:orange"
COLOR_QISKIT = "tab:blue"
COLOR_QUTIP = "tab:green"

# Internal qcrypto formalism colors (Option 1: Rust-themed palette)
COLOR_SV = "tab:orange"
COLOR_DM = "saddlebrown"
plt.rcParams.update(
    {
        "figure.figsize": (6.0, 4.0),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
    }
)


# ---------------------------------------------------------------------------
# DataFrame helpers
#
# pandas-stubs types DataFrame.__getitem__ as ``DataFrame | Series`` (or
# ``DataFrame | Series | Unknown`` in some versions).  These helpers
# centralise the necessary ``cast`` calls so the rest of the module stays
# annotation-clean.
# ---------------------------------------------------------------------------


def _col(df: pd.DataFrame, name: str) -> pd.Series:  # type: ignore[type-arg]
    """Return a single column as a Series."""
    return cast(pd.Series, df[name])  # type: ignore[return-value]


def _drop_na(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Drop rows where *col* is NaN."""
    return cast(pd.DataFrame, df[_col(df, col).notna()])


def _where(df: pd.DataFrame, col: str, value: object) -> pd.DataFrame:
    """Return rows where ``col == value``."""
    return cast(pd.DataFrame, df[_col(df, col) == value])


def _where_not(df: pd.DataFrame, col: str, value: object) -> pd.DataFrame:
    """Return rows where ``col != value``."""
    return cast(pd.DataFrame, df[_col(df, col) != value])


# ---------------------------------------------------------------------------
# criterion data loading
# ---------------------------------------------------------------------------


def read_criterion_estimates(root: Path) -> pd.DataFrame:
    """Walk *root* for ``estimates.json`` files inside ``new/`` directories.

    Returns a DataFrame with columns: ``group``, ``function``, ``value``
    (benchmark parameter), ``point_ns``, ``lower_ns``, ``upper_ns``.

    Criterion collapses all ``/`` in group and function names to ``_``, so the
    on-disk layout is always one of::

        <group>/<value>/new/estimates.json        (BenchmarkId::from_parameter)
        <group>/<function>/<value>/new/estimates.json  (BenchmarkId::new)

    where both ``<group>`` and ``<function>`` are single flat directory
    components (any original slashes replaced with underscores by criterion).
    """
    records = []
    for est_path in root.rglob("new/estimates.json"):
        rel = est_path.relative_to(root)
        parts = list(rel.parts[:-2])  # drop 'new' and 'estimates.json'
        if len(parts) == 2:
            group, function, value = parts[0], "", parts[1]
        elif len(parts) == 3:
            group, function, value = parts[0], parts[1], parts[2]
        else:
            continue  # unexpected depth; skip
        try:
            data = json.loads(est_path.read_text())
            est = data.get("median") or data.get("mean")
            point = est["point_estimate"]
            lower = est["confidence_interval"]["lower_bound"]
            upper = est["confidence_interval"]["upper_bound"]
        except (KeyError, json.JSONDecodeError):
            continue

        records.append(
            {
                "group": group,
                "function": function,
                "value": value,
                "point_ns": point,
                "lower_ns": lower,
                "upper_ns": upper,
            }
        )
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _norm(name: str) -> str:
    """Normalise a criterion group/function name (``/`` → ``_``, lowercase)."""
    return re.sub(r"[/ ]+", "_", str(name)).strip("_").lower()


def _by_group(crit: pd.DataFrame, group: str) -> pd.DataFrame:
    """Filter *crit* to rows matching *group*, tolerating ``/`` → ``_``."""
    target = _norm(group)
    mask = _col(crit, "group").map(_norm) == target
    return cast(pd.DataFrame, crit[mask].copy())


def _numeric(s: str) -> int | None:
    """Extract the first integer from *s*."""
    m = re.search(r"\d+", str(s))
    return int(m.group()) if m else None


def _save(fig: Figure, out: Path, name: str, category: str = "qcrypto") -> None:
    """Save *fig* as PDF, PNG and EPS under ``out/<category>/{pdf,png,eps}/``."""
    for fmt in ("pdf", "png", "eps"):
        subdir = out / category / fmt
        subdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(subdir / f"{name}.{fmt}", bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {name}  ({category}/{{pdf,png,eps}})")


# ---------------------------------------------------------------------------
# qcrypto figures
# ---------------------------------------------------------------------------


def fig_scaling_time(crit: pd.DataFrame, out: Path) -> None:
    sub = _by_group(crit, "scaling/build_ghz")
    if sub.empty:
        print("  [skip] scaling/build_ghz not found in criterion data")
        return
    sub["qubits"] = _col(sub, "value").map(_numeric)
    sub = _drop_na(sub, "qubits").sort_values("qubits")

    fig, ax = plt.subplots()
    for fn, grp in sub.groupby("function"):
        grp = grp.sort_values("qubits")
        # Use Rust-themed colors for high contrast
        color = COLOR_SV if fn == "StateVector" else COLOR_DM
        lstyle = "-" if fn == "StateVector" else "--"
        marker = "o" if fn == "StateVector" else "s"
        ax.errorbar(
            grp["qubits"],
            grp["point_ns"] / 1e6,
            yerr=[
                (grp["point_ns"] - grp["lower_ns"]) / 1e6,
                (grp["upper_ns"] - grp["point_ns"]) / 1e6,
            ],
            marker=marker,
            linestyle=lstyle,
            capsize=3,
            color=color,
            label=str(fn) or "state",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("GHZ build time (ms, log scale)")
    ax.set_title("Time scaling: StateVector vs DensityMatrix")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, out, "fig_scaling_time")


def fig_scaling_memory(data_qcrypto: Path, out: Path) -> None:
    path = data_qcrypto / "scaling_memory_theoretical.csv"
    if not path.exists():
        print("  [skip] scaling_memory_theoretical.csv not found")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots()
    for st, grp in df.groupby("state_type"):
        grp = grp.sort_values("qubits")
        color = COLOR_SV if st == "StateVector" else COLOR_DM
        lstyle = "-" if st == "StateVector" else "--"
        marker = "o" if st == "StateVector" else "s"
        ax.plot(
            grp["qubits"],
            grp["bytes"] / 1e6,
            marker=marker,
            linestyle=lstyle,
            color=color,
            label=str(st),
        )
    ax.set_yscale("log")
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Theoretical memory (MB, log scale)")
    ax.set_title(r"Memory footprint: $O(2^N)$ vs $O(4^N)$")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, out, "fig_scaling_memory")


def fig_channels(crit: pd.DataFrame, out: Path) -> None:
    sub = _by_group(crit, "channels/apply_single_qubit")
    if sub.empty:
        print("  [skip] channels/apply_single_qubit not found in criterion data")
        return
    sub["qubits"] = _col(sub, "value").map(_numeric)

    def _split_regime(fn: str) -> tuple[str, str]:
        token = str(fn).replace("/", "_")
        head, _sep, tail = token.rpartition("_")
        if tail in ("cold", "warm") and head:
            return head, tail
        return token, ""

    split = _col(sub, "function").map(_split_regime)
    sub["channel"] = split.map(lambda t: t[0])
    sub["regime"] = split.map(lambda t: t[1])
    sub = _drop_na(sub, "qubits")

    channels = sorted(sub["channel"].unique())
    cmap = plt.get_cmap("tab10")
    channel_colors = {ch: cmap(i) for i, ch in enumerate(channels)}

    fig, (ax_warm, ax_cold) = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=True)

    for regime, ax in zip(["warm", "cold"], [ax_warm, ax_cold]):
        reg_sub = _where(sub, "regime", regime)
        for ch, grp in reg_sub.groupby("channel"):
            grp = grp.sort_values("qubits")
            ax.plot(
                grp["qubits"],
                grp["point_ns"] / 1e3,
                marker="o",
                color=channel_colors[ch],
            )
        ax.set_yscale("log")
        ax.set_xlabel("Number of qubits")
        ax.set_title(f"Regime: {regime.upper()}")
        if ax is ax_warm:
            ax.set_ylabel("Channel application time (µs, log scale)")

    # Shared legend using proxies
    channel_proxies = [
        Line2D([0], [0], color=channel_colors[ch], lw=2, label=ch.replace("_", " "))
        for ch in channels
    ]
    ax_warm.legend(
        handles=channel_proxies,
        loc="upper left",
        ncol=2,
        fontsize=8,
        title="Channels",
    )

    fig.suptitle("Noise channel cost: Warm (DM only) vs Cold (SV→DM) path")
    fig.tight_layout()
    _save(fig, out, "fig_channels")


def fig_protocols(crit: pd.DataFrame, out: Path) -> None:
    sub = _by_group(crit, "protocols/qkd_key_length_scaling")
    if sub.empty:
        print("  [skip] protocols/qkd_key_length_scaling not found in criterion data")
        return
    sub["length"] = _col(sub, "value").map(_numeric)
    sub = _drop_na(sub, "length").sort_values("length")

    fig, ax = plt.subplots()
    for proto, grp in sub.groupby("function"):
        grp = grp.sort_values("length")
        ax.errorbar(
            grp["length"],
            grp["point_ns"] / 1e6,
            yerr=[
                (grp["point_ns"] - grp["lower_ns"]) / 1e6,
                (grp["upper_ns"] - grp["point_ns"]) / 1e6,
            ],
            marker="o",
            capsize=3,
            label=str(proto),
        )
    ax.set_xlabel("Key length (qubits / pairs)")
    ax.set_ylabel("Excution time (ms)")
    ax.set_title("QKD protocol scaling (BB84, B92, BBM92, E91, SixState, SARG04)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, out, "fig_protocols")


def fig_qber(data_qcrypto: Path, out: Path) -> None:
    path = data_qcrypto / "protocols_qkd_correctness.csv"
    if not path.exists():
        print("  [skip] protocols_qkd_correctness.csv not found")
        return
    df = pd.read_csv(path)
    by_channel = _where_not(df, "channel", "bit_flip_0.01")
    if by_channel.empty:
        by_channel = df
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    pivot = by_channel.pivot_table(
        index="channel", columns="protocol", values="qber", aggfunc="mean"
    )
    pivot.plot(kind="bar", ax=ax)
    ax.legend(loc="upper left")
    ax.set_xlabel("Channel")
    ax.set_ylabel("QBER")
    ax.set_title("QBER by channel (correctness verification)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, out, "fig_qber")


def fig_gates_sv_dm(crit: pd.DataFrame, out: Path) -> None:
    """Gate application time: SV vs DM, three gate types, fill_between CI."""
    gate_groups = [
        ("gates/single_qubit_X", "X (1 qubit)", "-", "o"),
        ("gates/two_qubit_CNOT", "CNOT (2 qubits)", "--", "x"),
        ("gates/three_qubit_Toffoli", "Toffoli (3 qubits)", ":", "s"),
    ]
    colors = {"StateVector": COLOR_SV, "DensityMatrix": COLOR_DM}

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    found = False
    for group_name, gate_label, lstyle, marker in gate_groups:
        sub = _by_group(crit, group_name)
        if sub.empty:
            continue
        found = True
        sub["qubits"] = _col(sub, "value").map(_numeric)
        sub = _drop_na(sub, "qubits")
        for fn, grp in sub.groupby("function"):
            fn = str(fn)
            color = colors.get(fn, "gray")
            grp = grp.sort_values("qubits")
            ax.fill_between(
                grp["qubits"],
                grp["lower_ns"] / 1e6,
                grp["upper_ns"] / 1e6,
                alpha=0.12,
                color=color,
            )
            ax.plot(
                grp["qubits"],
                grp["point_ns"] / 1e6,
                linestyle=lstyle,
                marker=marker,
                color=color,
            )
    if not found:
        print("  [skip] gates/* not found in criterion data")
        plt.close(fig)
        return

    # Consolidate legend
    formalism_proxies = [
        Line2D([0], [0], color=COLOR_SV, lw=2, label="StateVector"),
        Line2D([0], [0], color=COLOR_DM, lw=2, label="DensityMatrix"),
    ]
    gate_proxies = [
        Line2D(
            [0], [0], color="gray", marker="o", linestyle="-", label="X gate (1 qubit)"
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            marker="x",
            linestyle="--",
            label="CNOT gate (2 qubits)",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            marker="s",
            linestyle=":",
            label="Toffoli gate (3 qubits)",
        ),
    ]

    ax.set_yscale("log")
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Gate application time (ms, log scale)")
    ax.set_title("Gate Application Time: StateVector vs DensityMatrix")
    ax.legend(
        handles=formalism_proxies + gate_proxies, loc="upper left", ncol=2, fontsize=8
    )
    fig.tight_layout()
    _save(fig, out, "fig_gates_sv_dm")


def fig_conversion(crit: pd.DataFrame, out: Path) -> None:
    """Benchmark SV -> DM conversion cost (identity channel)."""
    sub = _by_group(crit, "conversion/sv_to_dm")
    if sub.empty:
        print("  [skip] conversion/sv_to_dm not found")
        return
    sub["qubits"] = _col(sub, "value").map(_numeric)
    sub = _drop_na(sub, "qubits").sort_values("qubits")

    fig, ax = plt.subplots()
    ax.plot(
        sub["qubits"],
        sub["point_ns"] / 1e6,
        marker="o",
        color=COLOR_DM,
        label="SV → DM promotion",
    )
    ax.fill_between(
        sub["qubits"],
        sub["lower_ns"] / 1e6,
        sub["upper_ns"] / 1e6,
        color=COLOR_DM,
        alpha=0.15,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Promotion time (ms, log scale)")
    ax.set_title("Dual-State Architecture: StateVector → DensityMatrix cost")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, out, "fig_conversion")


def fig_measurement(crit: pd.DataFrame, out: Path) -> None:
    """Benchmark Z-basis measurement on GHZ states."""
    sub = _by_group(crit, "measurement/z_basis")
    if sub.empty:
        print("  [skip] measurement/z_basis not found")
        return
    sub["qubits"] = _col(sub, "value").map(_numeric)
    sub = _drop_na(sub, "qubits")

    fig, ax = plt.subplots()
    for fn, grp in sub.groupby("function"):
        grp = grp.sort_values("qubits")
        is_all = "all" in str(fn)
        color = COLOR_DM if is_all else COLOR_SV
        lstyle = "-" if not is_all else "--"
        marker = "s" if is_all else "o"
        label = "Measure all qubits (seq)" if is_all else "Measure 1 qubit"

        ax.plot(
            grp["qubits"],
            grp["point_ns"] / 1e6,
            marker=marker,
            linestyle=lstyle,
            color=color,
            label=label,
        )
        ax.fill_between(
            grp["qubits"],
            grp["lower_ns"] / 1e6,
            grp["upper_ns"] / 1e6,
            color=color,
            alpha=0.1,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Measurement time (ms, log scale)")
    ax.set_title("Measurement performance: Single vs Sequential")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, out, "fig_measurement")


def fig_sampling_qcrypto(crit: pd.DataFrame, out: Path) -> None:
    """Compare qcrypto sampling times for 1M shots."""
    fig, ax = plt.subplots()

    sub = _by_group(crit, "sampling/1M")
    if sub.empty:
        print("  [skip] sampling/1M not found in criterion data")
        plt.close(fig)
        return

    sub["qubits"] = _col(sub, "value").map(_numeric)
    sub = _drop_na(sub, "qubits").sort_values("qubits")

    # Use a standard line plot with markers to avoid messy error intervals
    ax.plot(
        sub["qubits"],
        sub["point_ns"] / 1e6,
        marker="o",
        color=COLOR_QCRYPTO,
        label="qcrypto (1M shots)",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Sampling time (ms, log scale)")
    ax.set_title("qcrypto Sampling Performance: 1,000,000 shots")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, out, "fig_sampling_qcrypto", category="qcrypto")


# ---------------------------------------------------------------------------
# qiskit figures
# ---------------------------------------------------------------------------


def fig_vs_qiskit(crit: pd.DataFrame, aer_csv: Path, out: Path) -> None:
    """GHZ state-vector build time: qcrypto vs Qiskit Aer."""
    if not aer_csv.exists():
        print("  [skip] aer_results.csv not found; skipping Qiskit comparison")
        return
    aer = pd.read_csv(aer_csv)
    aer_sv = _where(aer, "task", "statevector_ghz").copy()
    aer_sv["qubits"] = _col(aer_sv, "qubits").astype(int)

    qc = _by_group(crit, "scaling/build_ghz")
    qc = _where(qc, "function", "StateVector")
    qc["qubits"] = _col(qc, "value").map(_numeric)
    qc = _drop_na(qc, "qubits").sort_values("qubits")

    if qc.empty or aer_sv.empty:
        print("  [skip] insufficient data for Qiskit comparison")
        return

    fig, ax = plt.subplots()
    ax.plot(
        qc["qubits"],
        qc["point_ns"] / 1e6,
        marker="o",
        color=COLOR_QCRYPTO,
        label="qcrypto (Rust)",
    )
    ax.fill_between(
        qc["qubits"],
        qc["lower_ns"] / 1e6,
        qc["upper_ns"] / 1e6,
        color=COLOR_QCRYPTO,
        alpha=0.15,
    )
    ax.plot(
        aer_sv["qubits"],
        aer_sv["median_s"] * 1e3,
        marker="s",
        linestyle="--",
        color=COLOR_QISKIT,
        label="Qiskit Aer",
    )
    if "lower_s" in aer_sv.columns and "upper_s" in aer_sv.columns:
        ax.fill_between(
            aer_sv["qubits"],
            aer_sv["lower_s"] * 1e3,
            aer_sv["upper_s"] * 1e3,
            color=COLOR_QISKIT,
            alpha=0.15,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("GHZ build time (ms, log scale)")
    ax.set_title("qcrypto vs Qiskit Aer (state vector, GHZ)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, out, "fig_ghz_vs_qiskit", category="qiskit")


def fig_gates_vs_qiskit(crit: pd.DataFrame, aer_csv: Path, out: Path) -> None:
    """X, CNOT, Toffoli gate times: qcrypto (SV) vs Qiskit Aer."""
    if not aer_csv.exists():
        print("  [skip] aer_results.csv not found; skipping gate comparison")
        return
    aer = pd.read_csv(aer_csv)

    gate_map = [
        ("gates/single_qubit_X", "X", "gate_X", "o", "-"),
        ("gates/two_qubit_CNOT", "CNOT", "gate_CX", "x", "--"),
        ("gates/three_qubit_Toffoli", "Toffoli", "gate_CCX", "s", ":"),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    found = False
    for group_name, label, aer_task, marker, lstyle in gate_map:
        sub = _by_group(crit, group_name)
        sub = _where(sub, "function", "StateVector") if not sub.empty else sub
        if not sub.empty:
            sub["qubits"] = _col(sub, "value").map(_numeric)
            sub = _drop_na(sub, "qubits").sort_values("qubits")
            ax.plot(
                sub["qubits"],
                sub["point_ns"] / 1e6,
                marker=marker,
                linestyle=lstyle,
                color=COLOR_QCRYPTO,
            )
            ax.fill_between(
                sub["qubits"],
                sub["lower_ns"] / 1e6,
                sub["upper_ns"] / 1e6,
                color=COLOR_QCRYPTO,
                alpha=0.1,
            )
            found = True
        aer_sub = _where(aer, "task", aer_task).copy()
        if not aer_sub.empty:
            aer_sub = aer_sub.sort_values("qubits")
            ax.plot(
                aer_sub["qubits"],
                aer_sub["median_s"] * 1e3,
                marker=marker,
                linestyle=lstyle,
                color=COLOR_QISKIT,
            )
            if "lower_s" in aer_sub.columns and "upper_s" in aer_sub.columns:
                ax.fill_between(
                    aer_sub["qubits"],
                    aer_sub["lower_s"] * 1e3,
                    aer_sub["upper_s"] * 1e3,
                    color=COLOR_QISKIT,
                    alpha=0.1,
                )
            found = True
    if not found:
        print("  [skip] no data for gate comparison")
        plt.close(fig)
        return

    # Consolidate legend
    sim_proxies = [
        Line2D([0], [0], color=COLOR_QCRYPTO, lw=2, label="qcrypto (Rust)"),
        Line2D([0], [0], color=COLOR_QISKIT, lw=2, label="Qiskit Aer"),
    ]
    gate_proxies = [
        Line2D(
            [0], [0], color="gray", marker="o", linestyle="-", label="X gate (1 qubit)"
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            marker="x",
            linestyle="--",
            label="CNOT gate (2 qubits)",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            marker="s",
            linestyle=":",
            label="Toffoli gate (3 qubits)",
        ),
    ]

    ax.set_yscale("log")
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Gate application time (ms, log scale)")
    ax.set_title("qcrypto vs Qiskit Aer: X, CNOT, Toffoli")
    ax.legend(handles=sim_proxies + gate_proxies, loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    _save(fig, out, "fig_gates_vs_qiskit", category="qiskit")


# ---------------------------------------------------------------------------
# qutip figures
# ---------------------------------------------------------------------------


def fig_channels_vs_qutip(crit: pd.DataFrame, qutip_csv: Path, out: Path) -> None:
    """Noise channel application: qcrypto (Rust, warm DM) vs QuTiP.

    Compares the three channels benchmarked by both simulators on the same
    axis (time vs qubits, log scale).  qcrypto data come from criterion's
    channels/apply_single_qubit warm path; QuTiP data from qutip_results.csv.
    """
    if not qutip_csv.exists():
        print("  [skip] qutip_results.csv not found; skipping QuTiP channel comparison")
        return

    qutip_df = pd.read_csv(qutip_csv)

    channel_names = ["depolarizing", "amplitude_damping", "phase_damping"]

    # Pull qcrypto warm-path data from criterion
    crit_ch = _by_group(crit, "channels/apply_single_qubit")
    crit_ch["qubits"] = _col(crit_ch, "value").map(_numeric)

    def _split_regime(fn: str) -> tuple[str, str]:
        token = str(fn).replace("/", "_")
        head, _sep, tail = token.rpartition("_")
        return (head, tail) if tail in ("cold", "warm") and head else (token, "")

    split = _col(crit_ch, "function").map(_split_regime)
    crit_ch["channel"] = split.map(lambda t: t[0])
    crit_ch["regime"] = split.map(lambda t: t[1])
    crit_ch = _drop_na(crit_ch, "qubits")

    fig, axes = plt.subplots(1, len(channel_names), figsize=(14.0, 4.5), sharey=True)
    found_any = False

    for ax, ch_name in zip(axes, channel_names):
        # qcrypto warm path
        qc_sub = cast(
            pd.DataFrame,
            crit_ch[
                (_col(crit_ch, "channel") == ch_name)
                & (_col(crit_ch, "regime") == "warm")
            ],
        ).sort_values("qubits")

        if not qc_sub.empty:
            ax.errorbar(
                qc_sub["qubits"],
                qc_sub["point_ns"] / 1e6,
                yerr=[
                    (qc_sub["point_ns"] - qc_sub["lower_ns"]) / 1e6,
                    (qc_sub["upper_ns"] - qc_sub["point_ns"]) / 1e6,
                ],
                marker="o",
                capsize=3,
                color=COLOR_QCRYPTO,
                label="qcrypto",
            )
            found_any = True

        # QuTiP
        qt_sub = _where(qutip_df, "task", f"channel_{ch_name}").copy()
        qt_sub = qt_sub.sort_values("qubits")
        if not qt_sub.empty:
            if "lower_s" in qt_sub.columns and "upper_s" in qt_sub.columns:
                ax.errorbar(
                    qt_sub["qubits"],
                    qt_sub["median_s"] * 1e3,
                    yerr=[
                        (qt_sub["median_s"] - qt_sub["lower_s"]) * 1e3,
                        (qt_sub["upper_s"] - qt_sub["median_s"]) * 1e3,
                    ],
                    marker="s",
                    linestyle="--",
                    capsize=3,
                    color=COLOR_QUTIP,
                    label="QuTiP",
                )
            else:
                ax.plot(
                    qt_sub["qubits"],
                    qt_sub["median_s"] * 1e3,
                    marker="s",
                    linestyle="--",
                    color=COLOR_QUTIP,
                    label="QuTiP",
                )
            found_any = True

        ax.set_yscale("log")
        ax.set_xlabel("Number of qubits")
        ax.set_title(ch_name.replace("_", " "))
        if ax is axes[0]:
            ax.set_ylabel("Channel application time (ms, log scale)")
        ax.legend(loc="upper left", fontsize=8)

    if not found_any:
        print("  [skip] no data for QuTiP channel comparison")
        plt.close(fig)
        return

    fig.suptitle("Noise channel application: qcrypto (Rust) vs QuTiP (Python)")
    fig.tight_layout()
    _save(fig, out, "fig_channels_vs_qutip", category="qutip")


def fig_purity_vs_qutip(crit: pd.DataFrame, qutip_csv: Path, out: Path) -> None:
    """Purity Tr(ρ²): qcrypto (Rust, criterion) vs QuTiP.

    qcrypto data from criterion's metrics/purity group.
    QuTiP data from qutip_results.csv task metric_purity.
    Both operate on the same type of mixed state (amplitude-damped |+…+⟩).
    """
    if not qutip_csv.exists():
        print("  [skip] qutip_results.csv not found; skipping purity comparison")
        return

    qutip_df = pd.read_csv(qutip_csv)
    qt_sub = _where(qutip_df, "task", "metric_purity").copy().sort_values("qubits")

    qc_sub = _by_group(crit, "metrics/purity")
    qc_sub["qubits"] = _col(qc_sub, "value").map(_numeric)
    qc_sub = _drop_na(qc_sub, "qubits").sort_values("qubits")

    if qc_sub.empty and qt_sub.empty:
        print("  [skip] no purity data for QuTiP comparison")
        return

    fig, ax = plt.subplots()
    if not qc_sub.empty:
        ax.errorbar(
            qc_sub["qubits"],
            qc_sub["point_ns"] / 1e6,
            yerr=[
                (qc_sub["point_ns"] - qc_sub["lower_ns"]) / 1e6,
                (qc_sub["upper_ns"] - qc_sub["point_ns"]) / 1e6,
            ],
            marker="o",
            capsize=3,
            color=COLOR_QCRYPTO,
            label="qcrypto (Rust)",
        )
    if not qt_sub.empty:
        if "lower_s" in qt_sub.columns and "upper_s" in qt_sub.columns:
            ax.errorbar(
                qt_sub["qubits"],
                qt_sub["median_s"] * 1e3,
                yerr=[
                    (qt_sub["median_s"] - qt_sub["lower_s"]) * 1e3,
                    (qt_sub["upper_s"] - qt_sub["median_s"]) * 1e3,
                ],
                marker="s",
                linestyle="--",
                capsize=3,
                color=COLOR_QUTIP,
                label="QuTiP (Python)",
            )
        else:
            ax.plot(
                _col(qt_sub, "qubits"),
                _col(qt_sub, "median_s") * 1e3,
                marker="s",
                linestyle="--",
                color=COLOR_QUTIP,
                label="QuTiP (Python)",
            )
    ax.set_yscale("log")
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Purity Tr(ρ²) computation time (ms, log scale)")
    ax.set_title("Purity computation: qcrypto vs QuTiP")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, out, "fig_purity_vs_qutip", category="qutip")


def fig_sampling_vs_qiskit(crit: pd.DataFrame, aer_csv: Path, out: Path) -> None:
    """Multi-shot sampling throughput: qcrypto Sampler vs Qiskit Aer.

    Two subplots, one per shot count (100 k and 1 M), matching Qiskit's
    ``--shots`` defaults.

    Methodological note: qcrypto's ``Sampler`` measures a single qubit (2
    outcomes); Qiskit samples the full n-qubit bitstring (2^N outcomes).
    The qubit-count axis on both sides reflects the O(2^N) cost of computing
    the probability distribution from the amplitude vector, not the number of
    distinct outcomes.  Declare this difference in the paper.
    """
    if not aer_csv.exists():
        print("  [skip] aer_results.csv not found; skipping sampling comparison")
        return
    aer = pd.read_csv(aer_csv)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5), sharey=True)
    found = False

    for ax, (shots_label, shots_val) in zip(
        axes, [("100k", 100_000), ("1M", 1_000_000)]
    ):
        qc_sub = _by_group(crit, f"sampling/{shots_label}")
        if not qc_sub.empty:
            qc_sub["qubits"] = _col(qc_sub, "value").map(_numeric)
            qc_sub = _drop_na(qc_sub, "qubits").sort_values("qubits")
            ax.errorbar(
                qc_sub["qubits"],
                qc_sub["point_ns"] / 1e6,
                yerr=[
                    (qc_sub["point_ns"] - qc_sub["lower_ns"]) / 1e6,
                    (qc_sub["upper_ns"] - qc_sub["point_ns"]) / 1e6,
                ],
                marker="o",
                capsize=3,
                color=COLOR_QCRYPTO,
                label="qcrypto Sampler",
            )
            found = True

        aer_sub = cast(
            pd.DataFrame,
            aer[(_col(aer, "task") == "sampling") & (_col(aer, "shots") == shots_val)],
        ).copy()
        if not aer_sub.empty:
            aer_sub = aer_sub.sort_values("qubits")
            if "lower_s" in aer_sub.columns and "upper_s" in aer_sub.columns:
                ax.errorbar(
                    aer_sub["qubits"],
                    aer_sub["median_s"] * 1e3,
                    yerr=[
                        (aer_sub["median_s"] - aer_sub["lower_s"]) * 1e3,
                        (aer_sub["upper_s"] - aer_sub["median_s"]) * 1e3,
                    ],
                    marker="s",
                    linestyle="--",
                    capsize=3,
                    color=COLOR_QISKIT,
                    label="Qiskit Aer",
                )
            else:
                ax.plot(
                    _col(aer_sub, "qubits"),
                    _col(aer_sub, "median_s") * 1e3,
                    marker="s",
                    linestyle="--",
                    color=COLOR_QISKIT,
                    label="Qiskit Aer",
                )
            found = True

        ax.set_yscale("log")
        ax.set_xlabel("Number of qubits")
        if ax is axes[0]:
            ax.set_ylabel("Sampling time (ms, log scale)")
        ax.set_title(f"{shots_label} shots")
        ax.legend(loc="upper left", fontsize=8)

    if not found:
        print("  [skip] no data for sampling comparison")
        plt.close(fig)
        return

    fig.suptitle("Multi-shot sampling: qcrypto Sampler (Rust) vs Qiskit Aer")
    fig.tight_layout()
    _save(fig, out, "fig_sampling_vs_qiskit", category="qiskit")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--criterion-root", default="../../../target/criterion")
    ap.add_argument(
        "--data",
        default="../../data",
        help="Root of benches/data/ (subdirs qcrypto/, qiskit/, qutip/).",
    )
    ap.add_argument("--aer", default="../../data/qiskit/aer_results.csv")
    ap.add_argument("--qutip", default="../../data/qutip/qutip_results.csv")
    ap.add_argument("--out", default="../../figures")
    args = ap.parse_args()

    root = Path(args.criterion_root)
    data = Path(args.data)
    data_qcrypto = data / "qcrypto"
    out = Path(args.out)
    aer_csv = Path(args.aer)
    qutip_csv = Path(args.qutip)

    if root.exists():
        crit = read_criterion_estimates(root)
        print(f"Loaded {len(crit)} criterion estimates from {root}")
    else:
        crit = pd.DataFrame(
            columns=["group", "function", "value", "point_ns", "lower_ns", "upper_ns"]
        )
        print(f"[warning] {root} not found — run 'cargo bench' first")

    print("\nGenerating figures:")

    # qcrypto
    fig_scaling_time(crit, out)
    fig_scaling_memory(data_qcrypto, out)
    fig_channels(crit, out)
    fig_protocols(crit, out)
    fig_qber(data_qcrypto, out)
    fig_gates_sv_dm(crit, out)
    fig_conversion(crit, out)
    fig_measurement(crit, out)
    fig_sampling_qcrypto(crit, out)

    # qiskit
    fig_vs_qiskit(crit, aer_csv, out)
    fig_gates_vs_qiskit(crit, aer_csv, out)
    fig_sampling_vs_qiskit(crit, aer_csv, out)

    # qutip
    fig_channels_vs_qutip(crit, qutip_csv, out)
    fig_purity_vs_qutip(crit, qutip_csv, out)

    print(f"\nFigures written to {out}/")
    print(f"  qcrypto : {out}/qcrypto/{{pdf,png,eps}}/")
    print(f"  qiskit  : {out}/qiskit/{{pdf,png,eps}}/")
    print(f"  qutip   : {out}/qutip/{{pdf,png,eps}}/")


if __name__ == "__main__":
    main()
