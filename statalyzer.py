#!/usr/bin/env python3
"""
Statalyzer — Statistical Arbitrage Bot for Solana

Usage:
    python statalyzer.py --monitor                           # Paper trade (default)
    python statalyzer.py --monitor --scanner-db PATH         # Custom scanner DB
    python statalyzer.py --live --confirm-live               # Live trading
    python statalyzer.py --status                            # Show portfolio
    python statalyzer.py --monitor --capital 5000            # Custom capital
"""

import argparse
import asyncio
import logging
import sys
import time

from config import Config
from block_fetcher import BlockFetcher
from signals import SignalGenerator, SignalType
from position import PositionSizer
from portfolio import PortfolioManager
from risk import RiskManager
from executor import PaperExecutor, LiveExecutor
from db import Database
from display import (
    print_banner, print_signal, print_entry, print_exit,
    print_positions, print_zscore_dashboard, print_progress, print_summary,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Statalyzer — statistical arbitrage on Solana',
    )
    parser.add_argument('--monitor', action='store_true', default=True,
                        help='Stream blocks, generate signals, trade (default)')
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
    fetcher = BlockFetcher(config.grpc_endpoint, config.grpc_token)
    signal_gen = SignalGenerator(config, config.scanner_db_path)
    portfolio = PortfolioManager(config, db)
    risk_mgr = RiskManager(config, portfolio)
    sizer = PositionSizer(config)

    if config.paper_trade:
        executor = PaperExecutor(config, db)
        mode = "PAPER TRADING"
    else:
        executor = LiveExecutor(config, db)
        mode = "LIVE TRADING"

    # Load cointegrated pairs
    num_pairs = signal_gen.load_pairs()
    if num_pairs == 0:
        print("No cointegrated pairs found. Check scanner DB path.")
        db.close()
        return

    # Persist initial capital
    portfolio.save_capital()

    # Print startup banner
    print_banner(config, num_pairs, mode)

    stats = {
        'blocks': 0,
        'signals': 0,
        'trades': 0,
    }
    start_time = time.time()
    deadline = start_time + args.duration * 60 if args.duration else None
    last_dashboard = 0
    last_snapshot = 0
    snapshot_interval = 300  # 5 minutes
    dashboard_interval = 60  # 1 minute

    try:
        async for slot, block in fetcher.follow_confirmed():
            block_time = block.block_time.timestamp if hasattr(block, 'block_time') and block.block_time else 0

            # Process block through signal generator
            signals = signal_gen.process_block(slot, block, block_time)

            for sig in signals:
                stats['signals'] += 1

                if sig.signal_type in (SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT):
                    # Entry signal
                    pair_state = signal_gen.get_pair_states().get(sig.pair_key)
                    if not pair_state:
                        continue

                    risk_check = risk_mgr.check_entry(sig, pair_state)
                    print_signal(sig, risk_check)
                    db.save_signal(sig, acted_on=risk_check.allowed, reason=risk_check.reason)

                    if not risk_check.allowed:
                        continue

                    # Get current prices from signal generator cache
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

                    # Open position with fill prices
                    position = portfolio.open_position(
                        sig, size, is_paper=config.paper_trade,
                        price_a=execution.fill_a.price, price_b=execution.fill_b.price,
                    )

                    # Log execution and record entry for rate limiting
                    executor.log_execution(position.id, execution, slot)
                    risk_mgr.record_entry()
                    stats['trades'] += 1

                    print_entry(sig, position, execution)

                elif sig.signal_type in (SignalType.EXIT, SignalType.STOP_LOSS):
                    # Exit signal
                    if not portfolio.has_position(sig.pair_key):
                        continue

                    risk_check = risk_mgr.check_exit(sig)
                    db.save_signal(sig, acted_on=risk_check.allowed, reason=risk_check.reason)

                    if not risk_check.allowed:
                        continue

                    position = portfolio.positions[sig.pair_key]

                    # Get current prices
                    price_a = signal_gen.token_prices.get(sig.token_a_mint, position.current_price_a)
                    price_b = signal_gen.token_prices.get(sig.token_b_mint, position.current_price_b)

                    # Execute exit
                    execution = executor.execute_exit(position, price_a, price_b)

                    # Update position prices before closing
                    position.current_price_a = execution.fill_a.price
                    position.current_price_b = execution.fill_b.price

                    reason = "stop_loss" if sig.signal_type == SignalType.STOP_LOSS else "mean_reversion"
                    closed = portfolio.close_position(sig.pair_key, sig.zscore, slot, reason)
                    if closed:
                        executor.log_execution(closed.id, execution, slot)
                        print_exit(closed, reason)

            # Mark-to-market all open positions
            portfolio.mark_to_market(signal_gen.token_prices)

            # Progress display
            stats['blocks'] += 1
            if stats['blocks'] % 10 == 0:
                print_progress(slot, stats['blocks'], stats['signals'], stats['trades'], start_time)

            # Periodic z-score dashboard
            now = time.time()
            if now - last_dashboard > dashboard_interval:
                print_zscore_dashboard(signal_gen.get_pair_states())
                if portfolio.positions:
                    print_positions(portfolio.positions, portfolio.get_total_value())
                last_dashboard = now

            # Periodic snapshot
            if now - last_snapshot > snapshot_interval:
                portfolio.take_snapshot()
                last_snapshot = now

            # Duration limit
            if deadline and now >= deadline:
                print(f"\nDuration limit reached ({args.duration} minutes).")
                break

    except KeyboardInterrupt:
        pass
    finally:
        # Final snapshot and summary
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
