#!/usr/bin/env python3
"""
Statalyzer — Statistical Arbitrage Bot for Solana

Usage:
    python statalyzer.py --monitor                           # Paper trade (default)
    python statalyzer.py --monitor --scanner-db PATH         # Custom scanner DB
    python statalyzer.py --live --confirm-live               # Live trading
    python statalyzer.py --status                            # Show portfolio
    python statalyzer.py --monitor --capital 5000            # Custom capital
    python statalyzer.py --monitor --no-scanner              # Discovery only (no scanner DB)
"""

import argparse
import asyncio
import logging
import sys
import time

from config import Config
from price_feed import JupiterPriceFeed
from signals import SignalGenerator, SignalType
from position import PositionSizer
from portfolio import PortfolioManager
from risk import RiskManager
from executor import PaperExecutor, QuoteExecutor, LiveExecutor
from db import Database
from display import (
    print_banner, print_signal, print_entry, print_exit,
    print_positions, print_zscore_dashboard, print_progress, print_summary,
    print_discovery_status,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Statalyzer — statistical arbitrage on Solana',
    )
    parser.add_argument('--monitor', action='store_true', default=True,
                        help='Stream blocks, generate signals, trade (default)')
    parser.add_argument('--quote', action='store_true',
                        help='Quote mode: use real Jupiter quotes for pricing (no execution)')
    parser.add_argument('--live', action='store_true',
                        help='Enable live execution (requires --confirm-live)')
    parser.add_argument('--confirm-live', action='store_true',
                        help='Confirm live trading (safety flag)')
    parser.add_argument('--status', action='store_true',
                        help='Show current portfolio and exit')
    parser.add_argument('--capital', type=float, default=None,
                        help='Initial capital for paper trading')
    parser.add_argument('--scanner-db', type=str, default=None,
                        help='Path to scanner cointegration DB')
    parser.add_argument('--db', type=str, default='statalyzer.db',
                        help='Path to statalyzer DB (default: statalyzer.db)')
    parser.add_argument('--entry-z', type=float, default=None,
                        help='Entry z-score threshold')
    parser.add_argument('--exit-z', type=float, default=None,
                        help='Exit z-score threshold')
    parser.add_argument('--stop-z', type=float, default=None,
                        help='Stop-loss z-score threshold')
    parser.add_argument('--max-positions', type=int, default=None,
                        help='Maximum number of open positions')
    parser.add_argument('--duration', type=float, default=None,
                        help='Run for this many minutes then stop')
    parser.add_argument('--no-scanner', action='store_true',
                        help='Disable scanner DB, use inline cointegration discovery only')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable debug logging')
    return parser.parse_args()


def apply_overrides(config: Config, args):
    """Apply CLI overrides to config."""
    if args.capital is not None:
        config.initial_capital = args.capital
    if args.scanner_db is not None:
        config.scanner_db_path = args.scanner_db
    if args.entry_z is not None:
        config.entry_zscore = args.entry_z
    if args.exit_z is not None:
        config.exit_zscore = args.exit_z
    if args.stop_z is not None:
        config.stop_loss_zscore = args.stop_z
    if args.max_positions is not None:
        config.max_positions = args.max_positions
    if args.no_scanner:
        config.use_scanner_db = False
    if args.live and args.confirm_live:
        config.paper_trade = False


def show_status(config: Config, db_path: str):
    """Show current portfolio state and exit."""
    db = Database(db_path)
    stats = db.get_stats()

    print(f"\n{'=' * 72}")
    print(f"  STATALYZER — Portfolio Status")
    print(f"{'=' * 72}")
    print(f"  DB:              {db_path}")
    print(f"  Open positions:  {stats['open_positions']}")
    print(f"  Closed trades:   {stats['closed_positions']}")
    print(f"  Total signals:   {stats['total_signals']}")
    print(f"  Realized P&L:    ${stats['total_realized_pnl']:+,.2f}")

    # Show open positions
    rows = db.get_open_positions()
    if rows:
        columns = db.get_position_columns()
        print(f"\n  Open Positions:")
        print(f"  {'Pair':<20} {'Dir':<6} {'Entry Z':>8} {'Entry $A':>10} {'Entry $B':>10}")
        print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*10} {'-'*10}")
        for row in rows:
            data = dict(zip(columns, row))
            pair_short = data['pair_key'][:8] + '..' + data['pair_key'][-6:]
            print(f"  {pair_short:<20} {data['direction']:<6} {data['entry_zscore']:>+8.3f} "
                  f"{data['entry_value_a']:>10.2f} {data['entry_value_b']:>10.2f}")

    print(f"{'=' * 72}")
    db.close()


async def run_monitor(config: Config, args):
    """Main monitoring and trading loop."""
    # Initialize components
    db = Database(args.db)
    scanner_path = config.scanner_db_path if config.use_scanner_db else None
    signal_gen = SignalGenerator(config, scanner_path, db=db)
    portfolio = PortfolioManager(config, db)
    risk_mgr = RiskManager(config, portfolio)
    sizer = PositionSizer(config)

    if args.quote:
        executor = QuoteExecutor(config, db)
        mode = "QUOTE MODE (Jupiter prices, fee-inclusive P&L)"
    elif config.paper_trade:
        executor = PaperExecutor(config, db)
        mode = "PAPER TRADING"
    else:
        executor = LiveExecutor(config, db)
        mode = "LIVE TRADING"

    # Load cointegrated pairs
    num_pairs = signal_gen.load_pairs()

    # Try loading cached discovered pairs from DB
    if num_pairs == 0:
        cached = db.load_discovered_pairs()
        if cached:
            # Add eg_is_cointegrated flag (all saved pairs were cointegrated)
            for d in cached:
                d['eg_is_cointegrated'] = True
            loaded = signal_gen.load_discovered_pairs(
                [type('R', (), d)() for d in cached]
            )
            num_pairs = loaded
            print(f"  Loaded {loaded} cached discovered pairs from DB")

    if num_pairs == 0:
        print("  No pairs yet — waiting for inline discovery (warmup ~60 min)...")

    # Sync position state: mark pairs that have open positions
    for pair_key in portfolio.positions:
        pair_state = signal_gen.get_pair_states().get(pair_key)
        if pair_state:
            pair_state.in_position = True

    # Persist initial capital
    portfolio.save_capital()

    # Initialize Jupiter price feed
    feed = JupiterPriceFeed(config, signal_gen.all_mints())

    # Print startup banner
    print_banner(config, num_pairs, mode)

    stats = {
        'polls': 0,
        'signals': 0,
        'trades': 0,
    }
    last_discovery_save = 0
    start_time = time.time()
    deadline = start_time + args.duration * 60 if args.duration else None
    last_dashboard = 0
    last_snapshot = 0
    snapshot_interval = 300  # 5 minutes
    dashboard_interval = 60  # 1 minute

    try:
        async for prices in feed.poll():
            now_ts = time.time()

            # Process prices through signal generator
            signals = signal_gen.process_prices(prices, now_ts)

            # Sync SOL/USD price to executor for fee estimation
            if hasattr(executor, 'sol_usd_price') and signal_gen.sol_usd_price > 0:
                executor.sol_usd_price = signal_gen.sol_usd_price

            # Update price feed mints if discovery added new pairs
            feed.update_mints(signal_gen.all_mints())

            for sig in signals:
                stats['signals'] += 1

                if sig.signal_type in (SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT):
                    # Entry signal
                    pair_state = signal_gen.get_pair_states().get(sig.pair_key)
                    if not pair_state:
                        continue

                    risk_check = risk_mgr.check_entry(sig, pair_state)
                    if risk_check.allowed or not risk_check.reason.startswith(("Max positions", "Max exposure")):
                        print_signal(sig, risk_check)
                    db.save_signal(sig, acted_on=risk_check.allowed, reason=risk_check.reason)

                    if not risk_check.allowed:
                        continue

                    # Get current prices
                    price_a = signal_gen.token_prices.get(sig.token_a_mint, 0)
                    price_b = signal_gen.token_prices.get(sig.token_b_mint, 0)
                    if price_a <= 0 or price_b <= 0:
                        continue

                    # Size the position
                    exposure = portfolio.get_total_exposure()
                    portfolio_value = portfolio.get_total_value()
                    size = sizer.compute_size(sig, portfolio_value, exposure, price_a, price_b)
                    if size is None:
                        continue

                    # Execute
                    execution = executor.execute_entry(sig, size, price_a, price_b)
                    if execution is None:
                        continue

                    position = portfolio.open_position(
                        sig, size, is_paper=config.paper_trade,
                        price_a=execution.fill_a.price, price_b=execution.fill_b.price,
                        fees_usd=execution.estimated_fees_usd,
                    )

                    executor.log_execution(position.id, execution, 0)
                    risk_mgr.record_entry()
                    stats['trades'] += 1

                    pair_state.in_position = True
                    pair_state.position_entry_time = time.time()

                    print_entry(sig, position, execution)

                elif sig.signal_type in (SignalType.EXIT, SignalType.STOP_LOSS):
                    if not portfolio.has_position(sig.pair_key):
                        continue

                    risk_check = risk_mgr.check_exit(sig)
                    db.save_signal(sig, acted_on=risk_check.allowed, reason=risk_check.reason)

                    if not risk_check.allowed:
                        continue

                    position = portfolio.positions[sig.pair_key]
                    price_a = signal_gen.token_prices.get(sig.token_a_mint, position.current_price_a)
                    price_b = signal_gen.token_prices.get(sig.token_b_mint, position.current_price_b)

                    execution = executor.execute_exit(position, price_a, price_b)
                    position.current_price_a = execution.fill_a.price
                    position.current_price_b = execution.fill_b.price

                    reason = "stop_loss" if sig.signal_type == SignalType.STOP_LOSS else "mean_reversion"
                    closed = portfolio.close_position(sig.pair_key, sig.zscore, 0, reason,
                                                      exit_fees_usd=execution.estimated_fees_usd)
                    if closed:
                        executor.log_execution(closed.id, execution, 0)
                        pair_state = signal_gen.get_pair_states().get(sig.pair_key)
                        if pair_state:
                            pair_state.in_position = False
                        print_exit(closed, reason)

            # Mark-to-market using Jupiter prices directly
            portfolio.mark_to_market(signal_gen.token_prices)
            for pk, pos in portfolio.positions.items():
                ps = signal_gen.get_pair_states().get(pk)
                if ps:
                    pos.current_zscore = ps.current_zscore

            # Dollar-based stop loss
            for pair_key in list(portfolio.positions.keys()):
                pos = portfolio.positions[pair_key]
                entry_value = pos.entry_value_a + pos.entry_value_b
                if entry_value > 0 and pos.unrealized_pnl < -entry_value * config.max_position_loss_pct:
                    price_a = signal_gen.token_prices.get(pos.token_a_mint, pos.current_price_a)
                    price_b = signal_gen.token_prices.get(pos.token_b_mint, pos.current_price_b)
                    execution = executor.execute_exit(pos, price_a, price_b)
                    pos.current_price_a = execution.fill_a.price
                    pos.current_price_b = execution.fill_b.price
                    closed = portfolio.close_position(pair_key, 0.0, 0, "dollar_stop",
                                                      exit_fees_usd=execution.estimated_fees_usd)
                    if closed:
                        executor.log_execution(closed.id, execution, 0)
                        pair_state = signal_gen.get_pair_states().get(pair_key)
                        if pair_state:
                            pair_state.in_position = False
                        print_exit(closed, f"dollar_stop (lost >{config.max_position_loss_pct:.0%})")

            # Time-based exit: close positions older than N half-lives
            now = int(time.time())
            for pair_key in list(portfolio.positions.keys()):
                if pair_key not in portfolio.positions:
                    continue
                pos = portfolio.positions[pair_key]
                pair_state = signal_gen.get_pair_states().get(pair_key)
                if not pair_state or pair_state.half_life <= 0 or pair_state.half_life == float('inf'):
                    continue
                hl_seconds = pair_state.half_life / 2.5
                max_age = hl_seconds * config.max_position_age_half_lives
                age = now - pos.entry_time
                if age > max_age:
                    price_a = signal_gen.token_prices.get(pos.token_a_mint, pos.current_price_a)
                    price_b = signal_gen.token_prices.get(pos.token_b_mint, pos.current_price_b)
                    execution = executor.execute_exit(pos, price_a, price_b)
                    pos.current_price_a = execution.fill_a.price
                    pos.current_price_b = execution.fill_b.price
                    closed = portfolio.close_position(pair_key, pos.current_zscore, 0, "time_exit",
                                                      exit_fees_usd=execution.estimated_fees_usd)
                    if closed:
                        executor.log_execution(closed.id, execution, 0)
                        if pair_state:
                            pair_state.in_position = False
                        print_exit(closed, f"time_exit ({age:.0f}s > {max_age:.0f}s = {config.max_position_age_half_lives}x HL)")

            # Orphan cleanup
            for pair_key in list(portfolio.positions.keys()):
                if pair_key not in portfolio.positions:
                    continue
                if pair_key in signal_gen.get_pair_states():
                    continue
                pos = portfolio.positions[pair_key]
                price_a = signal_gen.token_prices.get(pos.token_a_mint, pos.current_price_a)
                price_b = signal_gen.token_prices.get(pos.token_b_mint, pos.current_price_b)
                if price_a <= 0:
                    price_a = pos.entry_price_a
                if price_b <= 0:
                    price_b = pos.entry_price_b
                execution = executor.execute_exit(pos, price_a, price_b)
                pos.current_price_a = execution.fill_a.price
                pos.current_price_b = execution.fill_b.price
                closed = portfolio.close_position(pair_key, 0.0, 0, "orphan_cleanup",
                                                  exit_fees_usd=execution.estimated_fees_usd)
                if closed:
                    executor.log_execution(closed.id, execution, 0)
                    print_exit(closed, "orphan_cleanup (pair no longer monitored)")

            # Progress display
            stats['polls'] += 1
            num_prices = len(signal_gen.token_prices)
            active_pairs = sum(1 for p in signal_gen.get_pair_states().values() if p.prices_a.count > 0)
            print_progress(stats['polls'], num_prices, stats['signals'], stats['trades'], start_time)

            if stats['polls'] % 10 == 0:
                pairs_with_min = sum(1 for p in signal_gen.get_pair_states().values()
                                     if p.prices_a.count >= 10)
                logger.info(f"Prices tracked: {num_prices} tokens | "
                            f"Active pairs: {active_pairs}/{len(signal_gen.get_pair_states())} | "
                            f"Pairs with min obs: {pairs_with_min}")

            # Periodic z-score dashboard
            now_f = time.time()
            if now_f - last_dashboard > dashboard_interval:
                print_zscore_dashboard(signal_gen.get_pair_states())
                print_positions(portfolio.positions, portfolio.get_total_value())
                print_discovery_status(signal_gen.discovery)
                last_dashboard = now_f

            # Periodic snapshot
            if now_f - last_snapshot > snapshot_interval:
                portfolio.take_snapshot()
                last_snapshot = now_f

            # Persist discovered pairs to DB for crash recovery
            if now_f - last_discovery_save > 60:
                for result in signal_gen.discovery.discovered_pairs.values():
                    db.save_discovered_pair(result)
                last_discovery_save = now_f

            # Duration limit
            if deadline and now_f >= deadline:
                print(f"\nDuration limit reached ({args.duration} minutes).")
                break

    except KeyboardInterrupt:
        pass
    finally:
        portfolio.take_snapshot()
        elapsed = time.time() - start_time
        print_summary(portfolio, db, elapsed)
        db.close()


def main():
    args = parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    config = Config()
    apply_overrides(config, args)

    if args.status:
        show_status(config, args.db)
        return

    if args.live and not args.confirm_live:
        print("Error: --live requires --confirm-live safety flag")
        sys.exit(1)

    asyncio.run(run_monitor(config, args))


if __name__ == '__main__':
    main()
