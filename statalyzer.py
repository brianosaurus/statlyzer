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
import json
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
    parser.add_argument('--max-per-token', type=int, default=None,
                        help='Max positions per token (concentration limit)')
    parser.add_argument('--max-exposure', type=float, default=None,
                        help='Max exposure ratio (e.g. 2.0 = 200%% of capital)')
    parser.add_argument('--fixed-fraction', type=float, default=None,
                        help='Fixed fraction sizing (e.g. 0.05 = 5%%)')
    parser.add_argument('--max-position-usd', type=float, default=None,
                        help='Max USD per position')
    parser.add_argument('--sizing', type=str, default=None, choices=['fixed_fraction', 'kelly'],
                        help='Sizing method: fixed_fraction or kelly')
    parser.add_argument('--max-entry-z', type=float, default=None,
                        help='Max entry z-score (reject entries above this)')
    parser.add_argument('--direction', type=str, default=None, choices=['both', 'long', 'short'],
                        help='Allowed trade direction: both, long, or short')
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
    if args.max_per_token is not None:
        config.max_positions_per_token = args.max_per_token
    if args.max_exposure is not None:
        config.max_exposure_ratio = args.max_exposure
    if args.fixed_fraction is not None:
        config.fixed_fraction = args.fixed_fraction
    if args.max_position_usd is not None:
        config.max_position_usd = args.max_position_usd
    if args.sizing is not None:
        config.sizing_method = args.sizing
    if args.max_entry_z is not None:
        config.max_entry_zscore = args.max_entry_z
    if args.direction is not None:
        config.allowed_direction = args.direction
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
        print(f"  {'Basket':<24} {'Dir':<6} {'Entry Z':>8} {'Exposure':>10}")
        print(f"  {'-'*24} {'-'*6} {'-'*8} {'-'*10}")
        for row in rows:
            data = dict(zip(columns, row))
            pair_short = data['pair_key'][:8] + '..' + data['pair_key'][-6:]
            entry_values = json.loads(data['entry_values_json'])
            total_exposure = sum(entry_values)
            print(f"  {pair_short:<24} {data['direction']:<6} {data['entry_zscore']:>+8.3f} "
                  f"{total_exposure:>10.2f}")

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

    # Load cointegrated baskets
    num_pairs = signal_gen.load_baskets()

    # Try loading cached discovered pairs from DB
    if num_pairs == 0:
        cached = db.load_discovered_pairs()
        if cached:
            for d in cached:
                d['eg_is_cointegrated'] = True
            loaded = signal_gen.load_discovered_pairs(
                [type('R', (), d)() for d in cached]
            )
            num_pairs = loaded
            print(f"  Loaded {loaded} cached discovered pairs from DB")

    if num_pairs == 0:
        print("  No baskets yet — waiting for inline discovery (warmup ~60 min)...")

    # Sync position state: mark baskets that have open positions
    for pair_key in portfolio.positions:
        basket_state = signal_gen.get_basket_states().get(pair_key)
        if basket_state:
            basket_state.in_position = True

    # Force-close positions that don't match the allowed direction
    if config.allowed_direction in ('long', 'short'):
        unwanted = 'long' if config.allowed_direction == 'short' else 'short'
        to_close = [pk for pk, pos in portfolio.positions.items()
                    if pos.direction == unwanted]
        if to_close:
            print(f"  Closing {len(to_close)} {unwanted} positions (direction filter)...")
            for pair_key in to_close:
                pos = portfolio.positions[pair_key]
                # Use entry prices as fallback (no live prices yet)
                exit_prices = list(pos.entry_prices)
                execution = executor.execute_exit(pos, exit_prices)
                for i, fill in enumerate(execution.fills):
                    pos.current_prices[i] = fill.price
                closed = portfolio.close_position(pair_key, 0.0, 0, "direction_filter",
                                                  exit_fees_usd=execution.estimated_fees_usd)
                if closed:
                    executor.log_execution(closed.id, execution, 0)
                    basket_state = signal_gen.get_basket_states().get(pair_key)
                    if basket_state:
                        basket_state.in_position = False
                    print(f"    Closed {unwanted} {pair_key[:16]}.. P&L: ${closed.realized_pnl:+.2f}")

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

            # Update price feed mints if discovery added new baskets
            feed.update_mints(signal_gen.all_mints())

            for sig in signals:
                stats['signals'] += 1

                if sig.signal_type in (SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT):
                    # Direction filter
                    if config.allowed_direction == 'short' and sig.signal_type == SignalType.ENTRY_LONG:
                        db.save_signal(sig, acted_on=False, reason="Direction filter: shorts only")
                        continue
                    if config.allowed_direction == 'long' and sig.signal_type == SignalType.ENTRY_SHORT:
                        db.save_signal(sig, acted_on=False, reason="Direction filter: longs only")
                        continue

                    # Entry signal
                    basket_state = signal_gen.get_basket_states().get(sig.pair_key)
                    if not basket_state:
                        continue

                    risk_check = risk_mgr.check_entry(sig, basket_state)
                    if risk_check.allowed or not risk_check.reason.startswith(("Max positions", "Max exposure")):
                        print_signal(sig, risk_check)
                    db.save_signal(sig, acted_on=risk_check.allowed, reason=risk_check.reason)

                    if not risk_check.allowed:
                        continue

                    # Get current prices for all mints in basket
                    entry_prices = [signal_gen.token_prices.get(m, 0) for m in sig.mints]
                    if any(p <= 0 for p in entry_prices):
                        continue

                    # Size the position
                    exposure = portfolio.get_total_exposure()
                    portfolio_value = portfolio.get_total_value()
                    size = sizer.compute_size(sig, portfolio_value, exposure, entry_prices)
                    if size is None:
                        continue

                    # Execute
                    execution = executor.execute_entry(sig, size, entry_prices)
                    if execution is None:
                        continue

                    # Fill prices from execution
                    fill_prices = [f.price for f in execution.fills]

                    position = portfolio.open_position(
                        sig, size, is_paper=config.paper_trade,
                        prices=fill_prices,
                        fees_usd=execution.estimated_fees_usd,
                    )

                    executor.log_execution(position.id, execution, 0)
                    risk_mgr.record_entry()
                    stats['trades'] += 1

                    basket_state.in_position = True
                    basket_state.position_entry_time = time.time()

                    print_entry(sig, position, execution)

                elif sig.signal_type in (SignalType.EXIT, SignalType.STOP_LOSS):
                    if not portfolio.has_position(sig.pair_key):
                        continue

                    risk_check = risk_mgr.check_exit(sig)
                    db.save_signal(sig, acted_on=risk_check.allowed, reason=risk_check.reason)

                    if not risk_check.allowed:
                        continue

                    position = portfolio.positions[sig.pair_key]

                    # Min P&L filter: don't exit mean-reversion at a loss unless
                    # z has fully crossed zero (strong reversion) or stop loss
                    if sig.signal_type == SignalType.EXIT and position.unrealized_pnl < 0:
                        # Allow exit if z crossed zero (overshot mean), block if
                        # z is just drifting near the exit band with negative P&L
                        z = sig.zscore
                        crossed_zero = (position.direction == "long" and z > 0) or \
                                       (position.direction == "short" and z < 0)
                        if not crossed_zero:
                            logger.debug(f"Skipping exit {sig.pair_key[:16]}.. — "
                                         f"unrealized ${position.unrealized_pnl:+.2f}, "
                                         f"z={z:+.2f}, waiting for better reversion")
                            continue

                    exit_prices = [signal_gen.token_prices.get(m, position.current_prices[i])
                                   for i, m in enumerate(position.mints)]

                    execution = executor.execute_exit(position, exit_prices)
                    # Update current prices from fills
                    for i, fill in enumerate(execution.fills):
                        position.current_prices[i] = fill.price

                    reason = "stop_loss" if sig.signal_type == SignalType.STOP_LOSS else "mean_reversion"
                    closed = portfolio.close_position(sig.pair_key, sig.zscore, 0, reason,
                                                      exit_fees_usd=execution.estimated_fees_usd)
                    if closed:
                        executor.log_execution(closed.id, execution, 0)
                        basket_state = signal_gen.get_basket_states().get(sig.pair_key)
                        if basket_state:
                            basket_state.in_position = False
                        print_exit(closed, reason)

            # Mark-to-market using Jupiter prices directly
            portfolio.mark_to_market(signal_gen.token_prices)
            for pk, pos in portfolio.positions.items():
                bs = signal_gen.get_basket_states().get(pk)
                if bs:
                    pos.current_zscore = bs.current_zscore

            # Dollar-based stop loss
            for pair_key in list(portfolio.positions.keys()):
                pos = portfolio.positions[pair_key]
                entry_value = sum(pos.entry_values)
                if entry_value > 0 and pos.unrealized_pnl < -entry_value * config.max_position_loss_pct:
                    exit_prices = [signal_gen.token_prices.get(m, pos.current_prices[i])
                                   for i, m in enumerate(pos.mints)]
                    execution = executor.execute_exit(pos, exit_prices)
                    for i, fill in enumerate(execution.fills):
                        pos.current_prices[i] = fill.price
                    closed = portfolio.close_position(pair_key, 0.0, 0, "dollar_stop",
                                                      exit_fees_usd=execution.estimated_fees_usd)
                    if closed:
                        executor.log_execution(closed.id, execution, 0)
                        basket_state = signal_gen.get_basket_states().get(pair_key)
                        if basket_state:
                            basket_state.in_position = False
                        print_exit(closed, f"dollar_stop (lost >{config.max_position_loss_pct:.0%})")

            # Time-based exit: close positions older than N half-lives
            now = int(time.time())
            for pair_key in list(portfolio.positions.keys()):
                if pair_key not in portfolio.positions:
                    continue
                pos = portfolio.positions[pair_key]
                basket_state = signal_gen.get_basket_states().get(pair_key)
                if not basket_state or basket_state.half_life <= 0 or basket_state.half_life == float('inf'):
                    continue
                hl_seconds = basket_state.half_life / 2.5
                max_age = hl_seconds * config.max_position_age_half_lives
                age = now - pos.entry_time
                if age > max_age:
                    exit_prices = [signal_gen.token_prices.get(m, pos.current_prices[i])
                                   for i, m in enumerate(pos.mints)]
                    execution = executor.execute_exit(pos, exit_prices)
                    for i, fill in enumerate(execution.fills):
                        pos.current_prices[i] = fill.price
                    closed = portfolio.close_position(pair_key, pos.current_zscore, 0, "time_exit",
                                                      exit_fees_usd=execution.estimated_fees_usd)
                    if closed:
                        executor.log_execution(closed.id, execution, 0)
                        if basket_state:
                            basket_state.in_position = False
                        print_exit(closed, f"time_exit ({age:.0f}s > {max_age:.0f}s = {config.max_position_age_half_lives}x HL)")

            # Orphan cleanup
            for pair_key in list(portfolio.positions.keys()):
                if pair_key not in portfolio.positions:
                    continue
                if pair_key in signal_gen.get_basket_states():
                    continue
                pos = portfolio.positions[pair_key]
                exit_prices = []
                for i, m in enumerate(pos.mints):
                    p = signal_gen.token_prices.get(m, pos.current_prices[i])
                    if p <= 0:
                        p = pos.entry_prices[i]
                    exit_prices.append(p)
                execution = executor.execute_exit(pos, exit_prices)
                for i, fill in enumerate(execution.fills):
                    pos.current_prices[i] = fill.price
                closed = portfolio.close_position(pair_key, 0.0, 0, "orphan_cleanup",
                                                  exit_fees_usd=execution.estimated_fees_usd)
                if closed:
                    executor.log_execution(closed.id, execution, 0)
                    print_exit(closed, "orphan_cleanup (basket no longer monitored)")

            # Progress display
            stats['polls'] += 1
            num_prices = len(signal_gen.token_prices)
            active_baskets = sum(1 for b in signal_gen.get_basket_states().values()
                                 if b.price_buffers[0].count > 0)
            print_progress(stats['polls'], num_prices, stats['signals'], stats['trades'], start_time)

            if stats['polls'] % 10 == 0:
                baskets_with_min = sum(1 for b in signal_gen.get_basket_states().values()
                                       if b.price_buffers[0].count >= 10)
                logger.info(f"Prices tracked: {num_prices} tokens | "
                            f"Active baskets: {active_baskets}/{len(signal_gen.get_basket_states())} | "
                            f"Baskets with min obs: {baskets_with_min}")

            # Periodic z-score dashboard
            now_f = time.time()
            if now_f - last_dashboard > dashboard_interval:
                print_zscore_dashboard(signal_gen.get_basket_states())
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
