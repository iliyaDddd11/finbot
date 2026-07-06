#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python3 finrobot_equity/core/src/generate_financial_analysis.py   --company-ticker "${1:-NVDA}"   --company-name "${2:-NVIDIA Corporation}"   --config-file finrobot_equity/core/config/config.ini   --peer-tickers AMD INTC   --generate-text-sections
