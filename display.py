"""
Console display for statalyzer.
"""

import sys
import time
from typing import Dict, List


def token_label(symbol_a: str, symbol_b: str) -> str:
    return f"{symbol_a}/{symbol_b}"


def print_banner(config, num_pairs: int, mode: str):
    """Print startup banner."""
    print(f"\n{'=' * 72}")
    print(f"  STATALYZER — Statistical Arbitrage Bot")
    print(f"{'=' * 72}")
    print(f"  Mode:            {mode}")
    print(f"  gRPC:            {config.grpc_endpoint}")
    print(f"  Scanner DB:      {config.scanner_db_path}")
    print(f"  Pairs loaded:    {num_pairs}")
    print(f"  Initial capital: ${config.initial_capital:,.0f}")
    print(f"  Entry z-score:   {config.entry_zscore}")
    print(f"  Exit z-score:    {config.exit_zscore}")
    print(f"  Stop-loss z:     {config.stop_loss_zscore}")
    print(f"  Max positions:   {config.max_positions}")
    print(f"  Max exposure:    ${config.max_total_exposure_usd:,.0f}")
    print(f"  Sizing:          {config.sizing_method} ({config.fixed_fraction:.0%})")
    print(f"{'=' * 72}\n")


def print_signal(signal, risk_result):
    """Print a signal alert with risk check result."""
    z_str = f"z={signal.zscore:+.3f}"
    pair_label = token_label(signal.token_a_symbol, signal.token_b_symbol)

    if risk_result.allowed:
        status = "PASS"
    else:
        status = f"BLOCKED: {risk_result.reason}"

    sig_type = signal.signal_type.value.upper().replace('_', ' ')
    print(f"  [{sig_type}] {pair_label}  {z_str}  | {status}")


def print_entry(signal, position, execution):
    """Print when a new position is opened."""
    pair_label = token_label(signal.token_a_symbol, signal.token_b_symbol)
    direction = position.direction.upper()
    exposure = position.entry_value_a + position.entry_value_b
    slip_a = execution.fill_a.slippage_bps
    slip_b = execution.fill_b.slippage_bps
    print(f"  >>> OPEN {direction} {pair_label}  z={signal.zscore:+.3f}  "
          f"${exposure:.0f} exposure  slip={slip_a:.0f}/{slip_b:.0f}bps")


def print_exit(position, reason: str):
    """Print when a position is closed."""
    pair_label = f"{position.token_a_mint[:6]}../{position.token_b_mint[:6]}.."
    pnl = position.realized_pnl
    pnl_str = f"${pnl:+.2f}"
    duration = (position.exit_time - position.entry_time) if position.exit_time else 0
    print(f"  <<< CLOSE {position.direction.upper()} {pair_label}  "
          f"P&L {pnl_str}  ({reason}, {duration}s)")


def print_positions(positions: Dict, portfolio_value: float):
    """Print open positions table."""
    if not positions:
        print("  No open positions")
        return

    print(f"\n  {'Pair':<20} {'Dir':<6} {'Entry Z':>8} {'Curr Z':>8} {'Unrl P&L':>10} {'Exposure':>10}")
    print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

    for pair_key, pos in positions.items():
        pair_short = pair_key[:8] + '..' + pair_key[-6:]
        exposure = abs(pos.current_price_a * pos.quantity_a) + abs(pos.current_price_b * pos.quantity_b)
        print(f"  {pair_short:<20} {pos.direction:<6} {pos.entry_zscore:>+8.3f} "
              f"{pos.current_zscore:>+8.3f} {pos.unrealized_pnl:>+10.2f} {exposure:>10.0f}")

    print(f"\n  Portfolio value: ${portfolio_value:,.2f}")


def print_zscore_dashboard(pair_states: Dict, max_rows: int = 10):
    """Print z-score summary for monitored pairs (throttled)."""
    # Sort by abs z-score descending
    active = [(k, p) for k, p in pair_states.items() if p.prices_a.count > 0]
    active.sort(key=lambda x: abs(x[1].current_zscore), reverse=True)

    if not active:
        return

    print(f"\n  {'Pair':<24} {'Z-score':>8} {'Spread':>10} {'Obs':>5}")
    print(f"  {'-'*24} {'-'*8} {'-'*10} {'-'*5}")

    for pair_key, pair in active[:max_rows]:
        label = f"{pair.token_a_symbol}/{pair.token_b_symbol}"
        print(f"  {label:<24} {pair.current_zscore:>+8.3f} {pair.current_spread:>10.6f} {pair.prices_a.count:>5}")

    if len(active) > max_rows:
        print(f"  ... and {len(active) - max_rows} more pairs")


def print_progress(slot: int, blocks: int, signals: int, trades: int, start_time: float):
    """Print progress on a single line."""
    elapsed = time.time() - start_time
    rate = blocks / elapsed if elapsed > 0 else 0
    sys.stdout.write(
        f"\r  Slot {slot:,} | Blocks: {blocks:,} | Signals: {signals:,} | Trades: {trades:,} | {rate:.1f} blk/s"
    )
    sys.stdout.flush()


def print_summary(portfolio, db, elapsed: float):
    """Print session summary on exit."""
    stats = db.get_stats()
    total_value = portfolio.get_total_value()
    drawdown = portfolio.get_drawdown()
    unrealized = portfolio.get_total_unrealized_pnl()

    print(f"\n\n{'=' * 72}")
    print(f"  SESSION SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Duration:            {elapsed:.1f}s")
    print(f"  Total signals:       {stats['total_signals']:,}")
    print(f"  Total trades:        {stats['total_positions']:,}")
    print(f"  Open positions:      {stats['open_positions']:,}")
    print(f"  Closed positions:    {stats['closed_positions']:,}")
    print(f"  Realized P&L:        ${stats['total_realized_pnl']:+,.2f}")
    print(f"  Unrealized P&L:      ${unrealized:+,.2f}")
    print(f"  Portfolio value:     ${total_value:,.2f}")
    print(f"  Max drawdown:        {drawdown:.1%}")
    print(f"{'=' * 72}")
