#!/usr/bin/env bash
# Sets up the panecho conda environment on macOS (Apple Silicon or Intel).
# Usage: bash setup_mac.sh

set -euo pipefail

echo "=== PanEchoTest macOS Setup ==="

# ── Detect architecture ──────────────────────────────────────────────────────
ARCH=$(uname -m)
echo "Architecture: $ARCH"
if [[ "$ARCH" == "arm64" ]]; then
    echo "Apple Silicon detected — MPS acceleration will be available."
else
    echo "Intel Mac detected — CPU-only inference."
fi

# ── Prefer conda, fall back to venv ─────────────────────────────────────────
if command -v conda &>/dev/null; then
    echo ""
    echo "Creating conda environment 'panecho' from environment_mac.yml ..."
    conda env remove -n panecho --yes 2>/dev/null || true
    conda env create -f environment_mac.yml
    echo ""
    echo "Done. Activate with:  conda activate panecho"
    echo "Then run:             python run_panecho.py --help"
else
    echo ""
    echo "conda not found — creating a Python venv instead."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo ""
    echo "Done. Activate with:  source .venv/bin/activate"
    echo "Then run:             python run_panecho.py --help"
fi

echo ""
echo "=== Setup complete ==="
