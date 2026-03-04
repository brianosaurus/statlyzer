# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Statalyzer — statistical arbitrage bot for Solana. Monitors cointegrated token pairs from the upstream cointegration_scanner, generates z-score-based mean reversion signals, and executes pair trades (paper mode first, live later).

**Stack:** Python 3 · gRPC (Geyser) · Protocol Buffers · SQLite (WAL mode) · asyncio · numpy

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Paper trading (default)
python statalyzer.py --monitor --scanner-db ../cointegration_scanner/arb_tracker.db

# Custom capital
python statalyzer.py --monitor --capital 5000

# Live trading (requires both flags)
python statalyzer.py --live --confirm-live

# Show portfolio status
python statalyzer.py --status

# Override thresholds
python statalyzer.py --monitor --entry-z 2.5 --exit-z 0.3 --stop-z 5.0

# Debug logging
python statalyzer.py --monitor --verbose
```

No test suite yet. No linter configured.

## Architecture

**Data flow:** gRPC block stream → swap detection → price extraction → z-score signals → risk checks → position sizing → execution (paper/live) → portfolio tracking → SQLite

Key modules:
- **statalyzer.py** — CLI entry point. `--monitor` (default), `--status`, `--live --confirm-live`. Orchestrates all components in async main loop.
- **signal.py** — Z-score signal generator. Loads cointegrated pairs from scanner DB, maintains circular price buffers, emits ENTRY_LONG/ENTRY_SHORT/EXIT/STOP_LOSS signals.
- **executor.py** — Paper mode simulates fills with slippage model. Live mode stubbed for Jupiter integration.
- **portfolio.py** — Position tracking, mark-to-market, P&L, crash recovery from DB.
- **position.py** — Position sizing: fixed fraction (default 2%) or Kelly criterion.
- **risk.py** — Pre-entry checks: kill switch, max positions, max exposure, drawdown, staleness, half-life, rate limiting.
- **db.py** — SQLite persistence. Tables: signals, positions, execution_log, portfolio_snapshots, config_state. Read-only access to scanner's cointegration DB.
- **display.py** — Console output: banner, signals, positions, z-score dashboard, session summary.
- **block_fetcher.py** — gRPC streaming via Yellowstone Geyser. Auto-reconnect with backoff.
- **swap_detector.py** — Identifies swaps across Raydium, Orca, Meteora, PumpSwap DEXes.
- **config.py** — Environment-based config via .env. Signal thresholds, sizing, risk limits, execution params.
- **constants.py** — DEX program IDs, token mints, known bot wallets, Jito tip accounts.
- **grpc_utils.py** — Signer extraction, bot filtering, Jito detection.

**Key design decisions:**
- Numpy circular buffers for O(1) price append, no pandas overhead
- Scanner DB is read-only input (opened with `?mode=ro`)
- Paper/live use identical code paths, only executor internals differ
- Crash recovery: every position change persisted to SQLite immediately
- Pair key canonical form: `min(mint_a,mint_b):max(mint_a,mint_b)`
- Kill switch auto-engages on drawdown but does NOT auto-close positions

## Upstream Dependency

Requires cointegration_scanner's DB (default: `../cointegration_scanner/arb_tracker.db`) with `cointegration_results` table containing cointegrated pairs, hedge ratios, and half-lives.
