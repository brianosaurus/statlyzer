#!/usr/bin/env bash
# Experiment bot launch commands for frankfurt server
# Usage: ssh frankfurt 'cd statalyzer && source ~/venv/bin/activate && bash experiments.sh'

LSTS="SOL,bSOL,jitoSOL,mSOL,jupSOL,stSOL,JUP,FARTCOIN,ETH"
SCANNER_DB="../arbitrage_tracker/arb_tracker.db"
COMMON="--monitor --scanner-db $SCANNER_DB --no-paper-errors --direction both --slippage-bps 3 --min-spread-bps 15 --max-basket-size 2 --entry-z 1.0 --max-entry-z 6.0 --exit-z 0.1 --stop-z 4.0 --max-positions 40 --max-per-token 10 --max-per-hour 60 --candle-interval 30 --max-hl 7200 --token-whitelist $LSTS"

mkdir -p logs

# exp_ll: 10k capital, 7% sizing, no RL
nohup python3 statalyzer.py $COMMON \
  --no-lunar-lander \
  --capital 10000 --fixed-fraction 0.07 \
  --max-exposure 5.0 \
  --db exp_ll.db >> logs/exp_ll.log 2>&1 &

# exp_5k: 5k capital, 12% sizing, RL
nohup python3 statalyzer.py $COMMON \
  --lunar-lander \
  --capital 5000 --fixed-fraction 0.12 \
  --max-exposure 5.0 \
  --db exp_5k.db >> logs/exp_5k.log 2>&1 &

# exp_5x: 10k capital, 18% sizing, RL
nohup python3 statalyzer.py $COMMON \
  --lunar-lander \
  --capital 10000 --fixed-fraction 0.18 \
  --max-exposure 10.0 \
  --db exp_5x.db >> logs/exp_5x.log 2>&1 &

echo "Started all experiments. PIDs:"
ps aux | grep statalyzer | grep python3 | grep -v grep | awk '{print $2, $NF}'
