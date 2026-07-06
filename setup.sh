#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
pip install -r requirements-equity.txt
printf "[OK] Environment ready.
Activate with: source .venv/bin/activate
"
