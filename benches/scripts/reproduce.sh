#!/usr/bin/env bash
# Run the full qcrypto benchmark pipeline:
#   1. Rust criterion benchmarks (all four bench binaries)
#   2. Qiskit Aer comparison (optional — skipped if qiskit-aer is not installed)
#   3. Figure generation (plot_graphs.py)
#
# Run from anywhere:
#   ./benches/scripts/reproduce.sh
#   cd benches/scripts && ./reproduce.sh
#
# Options:
#   --skip-qiskit   Skip the Qiskit Aer comparison step.
#   --skip-rust     Skip the Rust criterion step (use existing target/criterion data).
#   --filter <pat>  Pass a filter pattern to cargo bench (e.g. --filter scaling).

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCHES_DIR="$(cd "$SCRIPTS_DIR/.." && pwd)"
ROOT_DIR="$(cd "$BENCHES_DIR/.." && pwd)"
QISKIT_DIR="$SCRIPTS_DIR/qiskit"
QUTIP_DIR="$SCRIPTS_DIR/qutip"
GRAPHS_DIR="$SCRIPTS_DIR/graphs"
DATA_DIR="$BENCHES_DIR/data"
FIGURES_DIR="$BENCHES_DIR/figures"
CRITERION_DIR="$ROOT_DIR/target/criterion"

SKIP_QISKIT=0
SKIP_QUTIP=0
SKIP_RUST=0
FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-qiskit) SKIP_QISKIT=1 ;;
        --skip-qutip)  SKIP_QUTIP=1 ;;
        --skip-rust)   SKIP_RUST=1 ;;
        --filter)      FILTER="$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

echo "============================================"
echo "  qcrypto benchmark pipeline"
echo "============================================"

# ---------------------------------------------------------------------------
# Step 1: Rust criterion benchmarks
# ---------------------------------------------------------------------------
if [[ $SKIP_RUST -eq 0 ]]; then
    echo
    echo "[1/4] Running Rust benchmarks..."
    echo "      Output: target/criterion/  +  benches/data/"
    echo

    cd "$ROOT_DIR"
    if [[ -n "$FILTER" ]]; then
        cargo bench -- "$FILTER"
    else
        cargo bench --bench scaling
        cargo bench --bench core_ops
        cargo bench --bench channels
        cargo bench --bench protocols
    fi
    echo
    echo "      Rust benchmarks complete."
else
    echo
    echo "[1/4] Skipping Rust benchmarks (--skip-rust)."
fi

# ---------------------------------------------------------------------------
# Step 2: Qiskit Aer comparison
# ---------------------------------------------------------------------------
if [[ $SKIP_QISKIT -eq 0 ]]; then
    echo
    echo "[2/4] Running Qiskit Aer comparison..."
    echo "      Output: $DATA_DIR/qiskit/aer_results.csv"
    echo

    cd "$QISKIT_DIR"
    uv run compare_qiskit.py \
        --repeats 20 \
        --out "$DATA_DIR/qiskit/aer_results.csv" \
        --env-out "$DATA_DIR/qiskit/aer_environment.txt"

    echo
    echo "      Qiskit comparison complete."
else
    echo
    echo "[2/4] Skipping Qiskit Aer comparison (--skip-qiskit)."
fi

# ---------------------------------------------------------------------------
# Step 3: QuTiP comparison
# ---------------------------------------------------------------------------
if [[ $SKIP_QUTIP -eq 0 ]]; then
    echo
    echo "[3/4] Running QuTiP comparison..."
    echo "      Output: $DATA_DIR/qutip/qutip_results.csv"
    echo

    cd "$QUTIP_DIR"
    uv run compare_qutip.py \
        --repeats 50 \
        --out "$DATA_DIR/qutip/qutip_results.csv" \
        --env-out "$DATA_DIR/qutip/qutip_environment.txt"

    echo
    echo "      QuTiP comparison complete."
else
    echo
    echo "[3/4] Skipping QuTiP comparison (--skip-qutip)."
fi

# ---------------------------------------------------------------------------
# Step 4: Figure generation
# ---------------------------------------------------------------------------
echo
echo "[4/4] Generating figures..."
echo "      Output: $FIGURES_DIR/"
echo

cd "$GRAPHS_DIR"
uv run plot_graphs.py \
    --criterion-root "$CRITERION_DIR" \
    --data "$DATA_DIR" \
    --aer "$DATA_DIR/qiskit/aer_results.csv" \
    --qutip "$DATA_DIR/qutip/qutip_results.csv" \
    --out "$FIGURES_DIR"

echo
echo "============================================"
echo "  Done."
echo "  Criterion HTML : $ROOT_DIR/target/criterion/report/index.html"
echo "  Raw CSV data   : $DATA_DIR/"
echo "  Figures        : $FIGURES_DIR/"
echo "============================================"
