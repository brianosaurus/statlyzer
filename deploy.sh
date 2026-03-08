#!/bin/bash
set -e

tar czf - --no-xattrs *.py requirements.txt | ssh frankfurt "cd statalyzer && tar xzf -"
ssh frankfurt "cd statalyzer; source ~/venv/bin/activate; pip install -q -r requirements.txt && python3 statalyzer.py --monitor --scanner-db ../arbitrage_tracker/arb_tracker.db $*" | tee /tmp/statalyzer.log
