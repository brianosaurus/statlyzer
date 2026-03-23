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
    parser.add_argument('--min-spread-bps', type=float, default=None,
                        help='Min absolute spread deviation in bps to enter (e.g. 10)')
    parser.add_argument('--max-hl', type=float, default=None,
                        help='Max half-life in seconds (reject slow mean-reverters)')
    parser.add_argument('--direction', type=str, default=None, choices=['both', 'long', 'short'],
                        help='Allowed trade direction: both, long, or short')
    parser.add_argument('--candle-interval', type=float, default=None,
                        help='Candle resample interval in seconds (default: 300)')
    parser.add_argument('--slippage-bps', type=int, default=None,
                        help='Slippage in basis points (default: from .env)')
    parser.add_argument('--max-per-hour', type=int, default=None,
                        help='Max new positions per hour (default: 5)')
    parser.add_argument('--no-paper-errors', action='store_true',
                        help='Disable paper leg failures and latency rejections')
    parser.add_argument('--lunar-lander', action='store_true',
                        help='Enable Lunar Lander fee model (embedded tip per bundle)')
    parser.add_argument('--no-lunar-lander', action='store_true',
                        help='Disable Lunar Lander fee model (SwQOS mode)')
    parser.add_argument('--token-whitelist', type=str, default=None,
                        help='Only trade baskets where ALL tokens are in this comma-separated list '
                             '(e.g. bSOL,mSOL,jitoSOL,jupSOL,BONK)')
    parser.add_argument('--rl', action='store_true',
                        help='Enable RL agent for entry/exit decisions (learns to maximize $/day)')
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
    if args.min_spread_bps is not None:
        config.min_spread_bps = args.min_spread_bps
    if args.max_hl is not None:
        config.max_half_life_secs = args.max_hl
    if args.direction is not None:
        config.allowed_direction = args.direction
    if args.candle_interval is not None:
        config.signal_resample_secs = args.candle_interval
    if args.slippage_bps is not None:
        config.slippage_bps = args.slippage_bps
    if args.max_per_hour is not None:
        config.max_positions_per_hour = args.max_per_hour
    if args.no_paper_errors:
        config.paper_leg_failure_pct = 0.0
        config.paper_latency_mean_s = 0.0
        config.paper_latency_std_s = 0.0
    if args.lunar_lander:
        config.use_lunar_lander = True
    if args.no_lunar_lander:
        config.use_lunar_lander = False
    if args.token_whitelist:
        from constants import WELL_KNOWN_TOKENS
        symbol_to_mint = {v['symbol']: k for k, v in WELL_KNOWN_TOKENS.items()}
        whitelist_symbols = [s.strip() for s in args.token_whitelist.split(',')]
        whitelist_mints = set()
        for sym in whitelist_symbols:
            mint = symbol_to_mint.get(sym)
            if mint:
                whitelist_mints.add(mint)
            else:
                logger.warning(f"Unknown token symbol in whitelist: {sym}")
        config.token_whitelist_mints = whitelist_mints
        logger.info(f"Token whitelist: {', '.join(whitelist_symbols)} ({len(whitelist_mints)} mints)")
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


def _get_sol_before(config, executor) -> tuple:
    """Get SOL balance and slot before a live exit. Returns (0.0, 0) for paper mode."""
    if config.paper_trade:
        return 0.0, 0
    try:
        return executor.get_sol_balance_and_slot_sync()
    except Exception as e:
        logger.warning(f"Could not get SOL balance before exit: {e}")
        return 0.0, 0


def _record_exit_recon(config, executor, closed, execution,
                       sol_before: float, slot_before: int):
    """Record exit reconciliation for live trades."""
    if config.paper_trade or sol_before <= 0 or closed is None:
        return
    if not hasattr(executor, 'record_exit_reconciliation'):
        return
    executor.record_exit_reconciliation(
        position_id=closed.id,
        pair_key=closed.pair_key,
        sol_before=sol_before,
        slot_before=slot_before,
        expected_pnl=closed.realized_pnl,
        execution=execution,
    )


async def run_monitor(config: Config, args):
    """Main monitoring and trading loop."""
    # Initialize executor first (live mode needs wallet balance before portfolio)
    db = Database(args.db)

    if args.quote:
        executor = QuoteExecutor(config, db)
        mode = "QUOTE MODE (Jupiter prices, fee-inclusive P&L)"
    elif config.paper_trade:
        executor = PaperExecutor(config, db)
        mode = "PAPER TRADING"
    else:
        executor = LiveExecutor(config, db)
        mode = "LIVE TRADING"

        # In live mode, set working capital from wallet SOL balance
        # unless --capital was explicitly provided
        if args.capital is None:
            from constants import USDC_MINT
            sol_balance = executor.get_sol_balance_sync()
            sol_quote = executor._get_quote(
                "So11111111111111111111111111111111111111112",
                USDC_MINT, 1_000_000_000)
            if sol_quote:
                sol_price = int(sol_quote["outAmount"]) / 1e6
                wallet_usd = sol_balance * sol_price
                config.initial_capital = wallet_usd
                executor.sol_usd_price = sol_price
                # Persist to DB so PortfolioManager picks it up
                db.set_state('initial_capital', str(wallet_usd))
                db.set_state('peak_value', str(wallet_usd))
                print(f"  Wallet: {sol_balance:.6f} SOL (${wallet_usd:.2f} @ ${sol_price:.2f}/SOL)")
            else:
                print(f"  WARNING: Could not get SOL price, using --capital or default")

    # Initialize remaining components (after capital is set)
    scanner_path = config.scanner_db_path if config.use_scanner_db else None
    signal_gen = SignalGenerator(config, scanner_path, db=db)
    portfolio = PortfolioManager(config, db)

    # Slippage monitor (optional — skip if token whitelist is set)
    slippage_mon = None
    if not config.token_whitelist_mints:
        from slippage import SlippageMonitor
        from constants import WELL_KNOWN_TOKENS, STABLECOIN_MINTS
        slippage_mon = SlippageMonitor(
            config, WELL_KNOWN_TOKENS, STABLECOIN_MINTS,
            poll_interval=300,
            db=db,
        )
        slippage_mon.start()

    # Wire slippage monitor into paper executor for dynamic per-token slippage
    if slippage_mon and hasattr(executor, 'slippage_monitor'):
        executor.slippage_monitor = slippage_mon

    # RL agent (optional)
    rl = None
    if args.rl:
        from rl_agent import RLDecisionMaker
        rl = RLDecisionMaker(config, db)
        logger.info(f"RL agent enabled: {rl.status_str()}")

    risk_mgr = RiskManager(config, portfolio, slippage_monitor=slippage_mon,
                            rl_enabled=(rl is not None))
    sizer = PositionSizer(config)

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

    # Sync position state: mark baskets that have open positions.
    # If a position's basket is no longer in the scanner DB (filtered out by
    # half-life, staleness, etc.), create a synthetic basket so exit/stop
    # signals can still fire.
    from signals import BasketState, token_symbol
    for pair_key, pos in portfolio.positions.items():
        basket_state = signal_gen.get_basket_states().get(pair_key)
        if basket_state:
            basket_state.in_position = True
            basket_state.position_entry_time = pos.entry_time
            basket_state.position_entry_zscore = pos.entry_zscore
        else:
            # Create synthetic basket for orphaned position
            symbols = [token_symbol(m) for m in pos.mints]
            synthetic = BasketState(
                pair_key=pair_key,
                basket_size=pos.basket_size,
                mints=pos.mints,
                symbols=symbols,
                hedge_ratios=pos.hedge_ratios,
                half_life=500,  # default; only needed for timeout calc
                eg_p_value=0.05,
                cointegration_analyzed_at=pos.entry_time,
            )
            synthetic.init_buffers(config.lookback_window)
            synthetic.in_position = True
            synthetic.position_entry_time = pos.entry_time
            synthetic.position_entry_zscore = pos.entry_zscore
            signal_gen.baskets[pair_key] = synthetic
            for m in pos.mints:
                signal_gen.monitored_mints.add(m)
            logger.info(f"Created synthetic basket for orphaned position {pair_key[:16]}.. "
                        f"({'/'.join(symbols)})")

    # Force-close positions that don't match the allowed direction
    if config.allowed_direction in ('long', 'short'):
        unwanted = 'long' if config.allowed_direction == 'short' else 'short'
        to_close = [pk for pk, pos in portfolio.positions.items()
                    if pos.direction == unwanted]
        if to_close:
            # Fetch live prices before force-closing (don't use stale entry prices)
            _prefetch_feed = JupiterPriceFeed(config, signal_gen.all_mints())
            _prefetch_prices = _prefetch_feed._fetch_prices()

            print(f"  Closing {len(to_close)} {unwanted} positions (direction filter)...")
            for pair_key in to_close:
                pos = portfolio.positions[pair_key]
                exit_prices = [_prefetch_prices.get(m, pos.current_prices[i])
                               for i, m in enumerate(pos.mints)]
                other_mints = {m for pk, p in portfolio.positions.items() if pk != pair_key for m in p.mints}
                sol_before, slot_before = _get_sol_before(config, executor)
                try:
                    execution = await executor.execute_exit(pos, exit_prices, other_position_mints=other_mints)
                except Exception as e:
                    logger.error(f"direction_filter exit failed for {pair_key}: {e} — position kept open")
                    continue
                for i, fill in enumerate(execution.fills):
                    pos.current_prices[i] = fill.price
                closed = portfolio.close_position(pair_key, 0.0, 0, "direction_filter",
                                                  exit_fees_usd=execution.estimated_fees_usd)
                if closed:
                    executor.log_execution(closed.id, execution, 0)
                    _record_exit_recon(config, executor, closed, execution, sol_before, slot_before)
                    basket_state = signal_gen.get_basket_states().get(pair_key)
                    if basket_state:
                        basket_state.in_position = False
                    print(f"    Closed {unwanted} {pair_key[:16]}.. P&L: ${closed.realized_pnl:+.2f}")


    # Recover pending RL decisions for open positions
    if rl and portfolio.positions:
        rl.load_pending_from_db(set(portfolio.positions.keys()))

    # Persist initial capital
    portfolio.save_capital()

    # Initialize Jupiter price feed
    feed = JupiterPriceFeed(config, signal_gen.all_mints())

    # Live mode: also track wallet token mints so we can price them
    if not config.paper_trade and hasattr(executor, 'get_all_token_balances_sync'):
        try:
            wallet_tokens = executor.get_all_token_balances_sync()
            wallet_mints = {mint for mint, _ in wallet_tokens}
            feed.update_mints(wallet_mints)
        except Exception:
            pass

    # Live mode: sweep orphaned tokens not belonging to any open position
    if not config.paper_trade and hasattr(executor, 'get_all_token_balances_sync'):
        try:
            wallet_tokens = executor.get_all_token_balances_sync()
            # Collect mints that belong to open positions
            position_mints = set()
            for pos in portfolio.positions.values():
                position_mints.update(pos.mints)
            # Identify orphans
            orphans = [(mint, amt) for mint, amt in wallet_tokens if mint not in position_mints]
            if orphans:
                from constants import WELL_KNOWN_TOKENS
                print(f"  Sweeping {len(orphans)} orphaned token(s) back to SOL...")
                # Get actual on-chain balances for all orphans
                sweep_list = []
                for mint, amt in orphans:
                    try:
                        bal_raw, _ = executor._get_token_balance_sync(mint)
                        if bal_raw > 0:
                            sweep_list.append((mint, bal_raw))
                    except Exception as e:
                        sym = WELL_KNOWN_TOKENS.get(mint, {}).get('symbol', mint[:8] + '..')
                        print(f"    Balance check failed for {sym}: {e}")

                if sweep_list and hasattr(executor, '_sweep_tokens_batch_sync'):
                    # Batch sweep: all swaps + 1 tip in a single request (up to 15 tokens)
                    results = executor._sweep_tokens_batch_sync(sweep_list)
                    for mint, _ in sweep_list:
                        sym = WELL_KNOWN_TOKENS.get(mint, {}).get('symbol', mint[:8] + '..')
                        amt = next((a for m, a in orphans if m == mint), 0)
                        if results.get(mint):
                            print(f"    Swept {sym} ({amt:.6g}) → SOL ✓")
                        else:
                            print(f"    Failed to sweep {sym} ({amt:.6g})")
                elif sweep_list:
                    # Fallback: sweep one at a time
                    for mint, bal_raw in sweep_list:
                        sym = WELL_KNOWN_TOKENS.get(mint, {}).get('symbol', mint[:8] + '..')
                        amt = next((a for m, a in orphans if m == mint), 0)
                        ok = executor._sweep_token_sync(mint, bal_raw)
                        if ok:
                            print(f"    Swept {sym} ({amt:.6g}) → SOL ✓")
                        else:
                            print(f"    Failed to sweep {sym} ({amt:.6g})")
                # Capital will be re-synced in the main loop once token prices are available
        except Exception as e:
            logger.warning(f"Startup orphan sweep failed: {e}")

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
    last_rl_train = 0
    snapshot_interval = 300  # 5 minutes
    dashboard_interval = 60  # 1 minute
    rl_train_interval = 600  # 10 minutes

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

                    # RL entry decision (if enabled)
                    rl_size_mult = 1.0
                    if rl is not None:
                        from rl_agent import ACTION_PASS
                        rl_action = rl.decide_entry(sig, basket_state, portfolio, risk_mgr,
                                                     slippage_monitor=slippage_mon)
                        if rl_action == ACTION_PASS:
                            rl.on_entry_skipped(sig, basket_state, portfolio, risk_mgr,
                                                slippage_monitor=slippage_mon)
                            db.save_signal(sig, acted_on=False, reason="RL: PASS")
                            continue
                        from rl_agent import SIZE_MULTIPLIERS
                        rl_size_mult = SIZE_MULTIPLIERS.get(rl_action, 1.0)

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

                    # Block baskets where worst token's round-trip slippage exceeds budget.
                    # Scaling down doesn't help — slippage is proportional to size.
                    if slippage_mon is not None:
                        from constants import SOL_MINT, STABLECOIN_MINTS
                        slippage_budget_bps = 20
                        worst_rt = 0
                        worst_sym = ""
                        for mint in sig.mints:
                            if mint in STABLECOIN_MINTS or mint == SOL_MINT:
                                continue
                            rt = slippage_mon.get_slippage_at_size(
                                mint, size.total_exposure_usd / max(sig.basket_size, 1))
                            if rt is not None and rt > worst_rt:
                                worst_rt = rt
                                worst_sym = mint[:8]
                        if worst_rt > slippage_budget_bps:
                            db.save_signal(sig, acted_on=False,
                                           reason="Slippage %s %.0fbps > %dbps budget" % (
                                               worst_sym, worst_rt, slippage_budget_bps))
                            continue

                    # Apply RL size multiplier
                    if rl is not None and rl_size_mult != 1.0:
                        size.amounts = [a * rl_size_mult for a in size.amounts]
                        size.amounts_raw = [max(1, int(r * rl_size_mult)) for r in size.amounts_raw]
                        size.dollar_amounts = [d * rl_size_mult for d in size.dollar_amounts]
                        size.total_exposure_usd *= rl_size_mult

                    # Paper realism: simulate execution latency + z-score decay
                    if hasattr(executor, 'simulate_latency'):
                        reject_reason = executor.simulate_latency(sig)
                        if reject_reason:
                            logger.info(f"Paper latency reject: {reject_reason}")
                            db.save_signal(sig, acted_on=False, reason=reject_reason)
                            continue

                    # Capture SOL balance before entry (live only)
                    entry_sol_before, _ = _get_sol_before(config, executor)

                    # Execute
                    try:
                        execution = await executor.execute_entry(sig, size, entry_prices)
                    except Exception as e:
                        logger.error(f"execute_entry failed for {sig.pair_key}: {e}")
                        continue
                    if execution is None:
                        continue

                    # Fill prices from execution
                    fill_prices = [f.price for f in execution.fills]

                    # Update position sizes with actual fill quantities
                    # so exits sell exactly what was received, not the estimate
                    for i, fill in enumerate(execution.fills):
                        size.amounts[i] = fill.quantity
                        size.amounts_raw[i] = fill.quantity_raw

                    position = portfolio.open_position(
                        sig, size, is_paper=config.paper_trade,
                        prices=fill_prices,
                        fees_usd=execution.estimated_fees_usd,
                    )

                    executor.log_execution(position.id, execution, 0)

                    # Record entry SOL balance for reconciliation
                    if entry_sol_before > 0 and hasattr(executor, 'get_sol_balance_and_slot_sync'):
                        try:
                            entry_sol_after, _ = executor.get_sol_balance_and_slot_sync()
                            db.update_position_entry_sol(position.id, entry_sol_before, entry_sol_after)
                        except Exception:
                            pass
                    risk_mgr.record_entry()
                    stats['trades'] += 1

                    basket_state.in_position = True
                    basket_state.position_entry_time = time.time()
                    basket_state.position_entry_zscore = sig.zscore

                    print_entry(sig, position, execution)

                elif sig.signal_type in (SignalType.EXIT, SignalType.STOP_LOSS):
                    if not portfolio.has_position(sig.pair_key):
                        continue

                    risk_check = risk_mgr.check_exit(sig)
                    db.save_signal(sig, acted_on=risk_check.allowed, reason=risk_check.reason)

                    if not risk_check.allowed:
                        continue

                    position = portfolio.positions[sig.pair_key]

                    # RL exit decision (only for EXIT signals, not STOP_LOSS)
                    if rl is not None and sig.signal_type == SignalType.EXIT:
                        exit_basket_state = signal_gen.get_basket_states().get(sig.pair_key)
                        should_exit = rl.decide_exit(
                            sig, exit_basket_state, portfolio, risk_mgr, position,
                            slippage_monitor=slippage_mon)
                        if not should_exit:
                            db.save_signal(sig, acted_on=False, reason="RL: HOLD")
                            continue

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

                    other_mints = {m for pk, p in portfolio.positions.items() if pk != sig.pair_key for m in p.mints}

                    sol_before, slot_before = _get_sol_before(config, executor)
                    try:
                        execution = await executor.execute_exit(position, exit_prices, other_position_mints=other_mints)
                    except Exception as e:
                        logger.error(f"execute_exit failed for {sig.pair_key}: {e} — position kept open")
                        continue
                    # Update current prices from fills
                    for i, fill in enumerate(execution.fills):
                        position.current_prices[i] = fill.price

                    reason = "stop_loss" if sig.signal_type == SignalType.STOP_LOSS else "mean_reversion"
                    closed = portfolio.close_position(sig.pair_key, sig.zscore, 0, reason,
                                                      exit_fees_usd=execution.estimated_fees_usd)
                    if closed:
                        executor.log_execution(closed.id, execution, 0)
                        _record_exit_recon(config, executor, closed, execution, sol_before, slot_before)

                        # RL reward feedback
                        if rl is not None:
                            entry_val = sum(closed.entry_values) if closed.entry_values else 1.0
                            dur_hrs = max((closed.exit_time - closed.entry_time) / 3600, 0.01)
                            rl.on_position_closed(sig.pair_key, closed.realized_pnl,
                                                  entry_val, dur_hrs,
                                                  is_live=not config.paper_trade,
                                                  position_id=closed.id)
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

            # Live mode: sync capital from actual wallet value (SOL + tokens)
            if not config.paper_trade and hasattr(executor, 'get_sol_balance_sync') and signal_gen.sol_usd_price > 0:
                try:
                    loop = asyncio.get_event_loop()
                    sol_balance = await loop.run_in_executor(
                        None, executor.get_sol_balance_sync)
                    token_balances = await loop.run_in_executor(
                        None, executor.get_all_token_balances_sync)
                    # Ensure price feed tracks all wallet tokens
                    feed.update_mints({mint for mint, _ in token_balances})
                    portfolio.sync_wallet_capital(
                        sol_balance, signal_gen.sol_usd_price,
                        token_balances, signal_gen.token_prices)
                except Exception as e:
                    logger.debug(f"Wallet balance sync failed: {e}")

            # Dollar-based stop loss
            for pair_key in list(portfolio.positions.keys()):
                pos = portfolio.positions[pair_key]
                entry_value = sum(pos.entry_values)
                if entry_value > 0 and pos.unrealized_pnl < -entry_value * config.max_position_loss_pct:
                    exit_prices = [signal_gen.token_prices.get(m, pos.current_prices[i])
                                   for i, m in enumerate(pos.mints)]
                    other_mints = {m for pk, p in portfolio.positions.items() if pk != pair_key for m in p.mints}
                    sol_before, slot_before = _get_sol_before(config, executor)
                    try:
                        execution = await executor.execute_exit(pos, exit_prices, other_position_mints=other_mints)
                    except Exception as e:
                        logger.error(f"dollar_stop exit failed for {pair_key}: {e} — position kept open")
                        continue
                    for i, fill in enumerate(execution.fills):
                        pos.current_prices[i] = fill.price
                    closed = portfolio.close_position(pair_key, 0.0, 0, "dollar_stop",
                                                      exit_fees_usd=execution.estimated_fees_usd)
                    if closed:
                        executor.log_execution(closed.id, execution, 0)
                        _record_exit_recon(config, executor, closed, execution, sol_before, slot_before)
                        if rl is not None:
                            entry_val = sum(closed.entry_values) if closed.entry_values else 1.0
                            dur_hrs = max((closed.exit_time - closed.entry_time) / 3600, 0.01)
                            rl.on_position_closed(pair_key, closed.realized_pnl, entry_val, dur_hrs,
                                                  is_live=not config.paper_trade,
                                                  position_id=closed.id)
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
                    other_mints = {m for pk, p in portfolio.positions.items() if pk != pair_key for m in p.mints}
                    sol_before, slot_before = _get_sol_before(config, executor)
                    try:
                        execution = await executor.execute_exit(pos, exit_prices, other_position_mints=other_mints)
                    except Exception as e:
                        logger.error(f"time_exit failed for {pair_key}: {e} — position kept open")
                        continue
                    for i, fill in enumerate(execution.fills):
                        pos.current_prices[i] = fill.price
                    closed = portfolio.close_position(pair_key, pos.current_zscore, 0, "time_exit",
                                                      exit_fees_usd=execution.estimated_fees_usd)
                    if closed:
                        executor.log_execution(closed.id, execution, 0)
                        _record_exit_recon(config, executor, closed, execution, sol_before, slot_before)
                        if rl is not None:
                            entry_val = sum(closed.entry_values) if closed.entry_values else 1.0
                            dur_hrs = max((closed.exit_time - closed.entry_time) / 3600, 0.01)
                            rl.on_position_closed(pair_key, closed.realized_pnl, entry_val, dur_hrs,
                                                  is_live=not config.paper_trade,
                                                  position_id=closed.id)
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
                other_mints = {m for pk, p in portfolio.positions.items() if pk != pair_key for m in p.mints}
                sol_before, slot_before = _get_sol_before(config, executor)
                try:
                    execution = await executor.execute_exit(pos, exit_prices, other_position_mints=other_mints)
                except Exception as e:
                    logger.error(f"orphan_cleanup exit failed for {pair_key}: {e} — position kept open")
                    continue
                for i, fill in enumerate(execution.fills):
                    pos.current_prices[i] = fill.price
                closed = portfolio.close_position(pair_key, 0.0, 0, "orphan_cleanup",
                                                  exit_fees_usd=execution.estimated_fees_usd)
                if closed:
                    executor.log_execution(closed.id, execution, 0)
                    _record_exit_recon(config, executor, closed, execution, sol_before, slot_before)
                    if rl is not None:
                        entry_val = sum(closed.entry_values) if closed.entry_values else 1.0
                        dur_hrs = max((closed.exit_time - closed.entry_time) / 3600, 0.01)
                        rl.on_position_closed(pair_key, closed.realized_pnl, entry_val, dur_hrs,
                                                  is_live=not config.paper_trade,
                                                  position_id=closed.id)
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
                print_positions(portfolio.positions, portfolio.get_total_value(),
                                signal_gen.get_basket_states(), config.max_position_age_half_lives)
                print_discovery_status(signal_gen.discovery, signal_gen.get_basket_states())
                if slippage_mon is not None:
                    print(f"  {slippage_mon.status_str()}")
                if rl is not None:
                    print(f"  RL: {rl.status_str()}")
                last_dashboard = now_f

            # Periodic snapshot
            if now_f - last_snapshot > snapshot_interval:
                portfolio.take_snapshot()
                last_snapshot = now_f

            # RL: process finalized reconciliations (live trades)
            if rl is not None and not config.paper_trade:
                sol_price = signal_gen.sol_usd_price or 0
                if sol_price > 0:
                    n = rl.process_reconciled_exits(db, sol_price)
                    if n > 0:
                        logger.info(f"RL: processed {n} reconciled exit(s)")

            # RL periodic training
            if rl is not None and now_f - last_rl_train > rl_train_interval:
                if rl.maybe_train():
                    logger.info(f"RL trained: {rl.status_str()}")
                last_rl_train = now_f

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
        if slippage_mon is not None:
            slippage_mon.stop()
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
