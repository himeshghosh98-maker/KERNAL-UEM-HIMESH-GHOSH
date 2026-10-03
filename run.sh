#!/usr/bin/env bash
# GapDetect — one-command setup + run
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt

echo "== Running tests =="
pytest -q || true

echo "== Evaluation =="
python evaluate.py

echo "== Launching Streamlit on http://localhost:8501 =="
streamlit run app.py --server.headless true
