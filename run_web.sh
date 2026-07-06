#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python3 run_web_app.py --host 0.0.0.0 --port 8001 --no-reload
