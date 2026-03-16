#!/bin/bash
# Launch paper trading experiments targeting profitability.
#
# Phase 1 (AA-DD): Diagnostic — sweep slippage to find break-even point.
#   All identical except slippage: 0/5/15/25 bps.
#
# Phase 2 (EE-HH): Optimized — best-case realistic configurations.
#   Lower slippage (10-15bps, realistic for LSTs on Jupiter),
#   larger positions, faster candles, higher rate limits.
#
# All use: both directions, no paper errors, bundled Lunar Lander fees.

cd ~/statalyzer
source ~/venv/bin/activate

SCANNER="--scanner-db ../arbitrage_tracker/arb_tracker.db"
BASE="--monitor $SCANNER --no-paper-errors --direction both"

# Kill old experiments (excluding P which is live)
echo "Killing old experiments..."
for db in exp_r.db exp_s.db exp_t.db exp_u.db exp_v.db exp_w.db exp_x.db exp_y.db \
          exp_aa.db exp_bb.db exp_cc.db exp_dd.db exp_ee.db exp_ff.db exp_gg.db exp_hh.db; do
    pid=$(ps auxww | grep "statalyzer.py" | grep "$db" | grep -v grep | awk '{print $2}')
    if [ -n "$pid" ]; then
        echo "  Killing $db (PID $pid)"
        kill $pid
    fi
done
sleep 2

# Remove old DBs to start fresh
echo "Removing old experiment DBs..."
rm -f exp_r.db exp_s.db exp_t.db exp_u.db exp_v.db exp_w.db exp_x.db exp_y.db
rm -f exp_aa.db exp_bb.db exp_cc.db exp_dd.db exp_ee.db exp_ff.db exp_gg.db exp_hh.db
mkdir -p logs

echo ""
echo "=== PHASE 1: Diagnostic (slippage sweep) ==="
echo ""

# Shared Phase 1 params: $5k, 5% fraction, z2.0-4.0, exit0.3, stop5.0, 40/hr, 30s candles
P1="$BASE --capital 5000 --fixed-fraction 0.05 --entry-z 2.0 --max-entry-z 4.0 \
    --exit-z 0.3 --stop-z 5.0 --max-positions 40 --max-per-token 10 --max-exposure 5.0 \
    --max-per-hour 40 --candle-interval 30 --max-hl 3600"

# AA: Zero slippage — theoretical maximum from pure z-score edge
nohup python3 statalyzer.py $P1 --slippage-bps 0 --db exp_aa.db \
    > logs/exp_aa.log 2>&1 &
echo "  AA: 0bps slippage (theoretical max)"

# BB: 5bps slippage — near-zero
nohup python3 statalyzer.py $P1 --slippage-bps 5 --db exp_bb.db \
    > logs/exp_bb.log 2>&1 &
echo "  BB: 5bps slippage"

# CC: 15bps slippage — low but realistic for LSTs
nohup python3 statalyzer.py $P1 --slippage-bps 15 --db exp_cc.db \
    > logs/exp_cc.log 2>&1 &
echo "  CC: 15bps slippage"

# DD: 25bps slippage — moderate
nohup python3 statalyzer.py $P1 --slippage-bps 25 --db exp_dd.db \
    > logs/exp_dd.log 2>&1 &
echo "  DD: 25bps slippage"

echo ""
echo "=== PHASE 2: Optimized for profit ==="
echo ""

# EE: Max volume — large capital, aggressive sizing, low z entry, very tight exit
nohup python3 statalyzer.py $BASE --capital 20000 --fixed-fraction 0.10 \
    --entry-z 1.5 --max-entry-z 4.0 --exit-z 0.1 --stop-z 6.0 \
    --max-positions 60 --max-per-token 15 --max-exposure 5.0 \
    --max-per-hour 100 --candle-interval 30 --slippage-bps 10 --max-hl 3600 \
    --db exp_ee.db > logs/exp_ee.log 2>&1 &
echo "  EE: Max volume \$20k (z1.5, exit0.1, 10bps, 100/hr)"

# FF: High-z only, no stop loss — let mean reversion work
nohup python3 statalyzer.py $BASE --capital 10000 --fixed-fraction 0.07 \
    --entry-z 2.5 --max-entry-z 4.0 --exit-z 0.2 --stop-z 999 \
    --max-positions 40 --max-per-token 10 --max-exposure 5.0 \
    --max-per-hour 60 --candle-interval 30 --slippage-bps 10 --max-hl 3600 \
    --db exp_ff.db > logs/exp_ff.log 2>&1 &
echo "  FF: High-z no-stop \$10k (z2.5, exit0.2, 10bps, no stop)"

# GG: Ultra-fast candles — match 14s avg half-life
nohup python3 statalyzer.py $BASE --capital 10000 --fixed-fraction 0.10 \
    --entry-z 2.0 --max-entry-z 4.0 --exit-z 0.2 --stop-z 5.0 \
    --max-positions 40 --max-per-token 10 --max-exposure 5.0 \
    --max-per-hour 60 --candle-interval 15 --slippage-bps 15 --max-hl 3600 \
    --db exp_gg.db > logs/exp_gg.log 2>&1 &
echo "  GG: Ultra-fast 15s candles \$10k (z2.0, exit0.2, 15bps)"

# HH: P-optimized — P's z-range but wider max-entry-z, lower slippage
nohup python3 statalyzer.py $BASE --capital 10000 --fixed-fraction 0.07 \
    --entry-z 2.7 --max-entry-z 3.5 --exit-z 0.3 --stop-z 4.0 \
    --max-positions 40 --max-per-token 10 --max-exposure 4.0 \
    --max-per-hour 40 --candle-interval 30 --slippage-bps 10 --max-hl 3600 \
    --db exp_hh.db > logs/exp_hh.log 2>&1 &
echo "  HH: P-optimized \$10k (z2.7-3.5, exit0.3, 10bps)"

echo ""
echo "=== PHASE 3: Whitelist (real slippage, low-slippage tokens only) ==="
echo ""

# Low-slippage tokens (measured via Jupiter Quote API round-trip):
#   bSOL ~1bps, jitoSOL ~0.2bps, mSOL ~0.5bps, jupSOL ~0.5bps, stSOL ~2bps,
#   FARTCOIN ~8bps, BONK ~8bps
# Average per-leg slippage ~2bps → paper setting 24bps (conservative)
LOWSLIP="bSOL,FARTCOIN,BONK,mSOL,jitoSOL,jupSOL,stSOL"

# II: Realistic slippage, low-slippage tokens only — proof of concept
nohup python3 statalyzer.py $BASE --capital 10000 --fixed-fraction 0.07 \
    --entry-z 2.0 --max-entry-z 4.0 --exit-z 0.2 --stop-z 5.0 \
    --max-positions 40 --max-per-token 10 --max-exposure 5.0 \
    --max-per-hour 60 --candle-interval 30 --slippage-bps 24 --max-hl 7200 \
    --token-whitelist $LOWSLIP \
    --db exp_ii.db > logs/exp_ii.log 2>&1 &
echo "  II: Whitelist \$10k realistic 24bps (avg real slippage)"

# JJ: $1k/day target — large capital, aggressive sizing, low-slippage tokens
nohup python3 statalyzer.py $BASE --capital 50000 --fixed-fraction 0.10 \
    --entry-z 1.5 --max-entry-z 4.0 --exit-z 0.1 --stop-z 6.0 \
    --max-positions 60 --max-per-token 15 --max-exposure 5.0 \
    --max-per-hour 100 --candle-interval 30 --slippage-bps 24 --max-hl 7200 \
    --token-whitelist $LOWSLIP \
    --db exp_jj.db > logs/exp_jj.log 2>&1 &
echo "  JJ: \$1k/day target \$50k whitelist (z1.5, exit0.1, 24bps, 100/hr)"

# KK: Same as JJ but with 10bps paper slippage (optimistic/bSOL-heavy pairs)
nohup python3 statalyzer.py $BASE --capital 50000 --fixed-fraction 0.10 \
    --entry-z 1.5 --max-entry-z 4.0 --exit-z 0.1 --stop-z 6.0 \
    --max-positions 60 --max-per-token 15 --max-exposure 5.0 \
    --max-per-hour 100 --candle-interval 30 --slippage-bps 10 --max-hl 7200 \
    --token-whitelist $LOWSLIP \
    --db exp_kk.db > logs/exp_kk.log 2>&1 &
echo "  KK: \$1k/day optimistic \$50k whitelist (z1.5, exit0.1, 10bps, 100/hr)"

sleep 2
echo ""
echo "Running experiments:"
ps auxww | grep statalyzer | grep -v grep | grep -oP 'exp_\w+\.db' | sort
echo ""
echo "Monitor with: python3 compare_experiments.py"
