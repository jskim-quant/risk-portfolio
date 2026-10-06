#!/bin/bash
cd "$(dirname "$0")"
pip install -q pandas yfinance requests
for s in 01_etf.py 03_ecb.py 04_mof.py; do echo "===== $s"; python3 $s; done
echo "===== DONE"
