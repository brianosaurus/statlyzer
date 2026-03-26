"""
Console display for statalyzer.
"""

import sys
import time
from typing import Dict, List

from constants import WELL_KNOWN_TOKENS

# mint -> symbol lookup
_MINT_TO_SYMBOL = {mint: info["symbol"] for mint, info in WELL_KNOWN_TOKENS.items()}


def mints_to_label(mints: List[str]) -> str:
    """Convert list of mints to human-readable label like 'mSOL/jitoSOL'."""
    return "/".join(_MINT_TO_SYMBOL.get(m, m[:6]) for m in mints)


def basket_label(symbols: List[str]) -> str:
    return "/".join(symbols)


# Backward compat alias
def token_label(symbol_a: str, symbol_b: str) -> str:
    return basket_label([symbol_a, symbol_b])


def print_banner(config, num_pairs: int, mode: str):
    """Print startup banner."""
    print(f"\n{'=' * 72}")
    print(f"  STATALYZER — Statistical Arbitrage Bot")
    print(f"{'=' * 72}")
    print(f"  Mode:            {mode}")
    print(f"  Price feed:      Jupiter API ({config.price_poll_interval:.0f}s poll)")
    if config.use_scanner_db:
        print(f"  Scanner DB:      {config.scanner_db_path}")
    else:
        print(f"  Discovery:       inline (warmup {config.coint_warmup_minutes:.0f}min)")
    print(f"  Baskets loaded:  {num_pairs}")
    print(f"  Initial capital: ${config.initial_capital:,.0f}")
    print(f"  Entry z-score:   {config.entry_zscore}")
    print(f"  Exit z-score:    {config.exit_zscore}")
    print(f"  Stop-loss z:     {config.stop_loss_zscore}")
    print(f"  Max positions:   {config.max_positions}")
    print(f"  Max exposure:    {config.max_exposure_ratio:.0%} of capital")
    print(f"  Sizing:          {config.sizing_method} ({config.fixed_fraction:.0%})")
    print(f"  Max half-life:   {config.max_half_life_secs:.0f}s ({config.max_half_life_secs/60:.0f}min)")
    print(f"  Resample:        {config.signal_resample_secs:.0f}s candles, {config.lookback_window} lookback")
    print(f"{'=' * 72}\n")


def print_signal(signal, risk_result):
    """Print a signal alert with risk check result."""
    z_str = f"z={signal.zscore:+.3f}"
    label = basket_label(signal.symbols)

    if risk_result.allowed:
        status = "PASS"
    else:
        status = f"BLOCKED: {risk_result.reason}"

    sig_type = signal.signal_type.value.upper().replace('_', ' ')
    sys.stdout.write('\n')
    print(f"  [{sig_type}] {label}  {z_str}  | {status}")


def print_entry(signal, position, execution):
    """Print when a new position is opened."""
    label = basket_label(signal.symbols)
    direction = position.direction.upper()
    exposure = sum(position.entry_values)
    slips = [f"{f.slippage_bps:.0f}" for f in execution.fills]
    slip_str = "/".join(slips) + "bps"
    print(f"  >>> OPEN {direction} {label}  z={signal.zscore:+.3f}  "
          f"${exposure:.0f} exposure  slip={slip_str}")


def print_exit(position, reason: str):
    """Print when a position is closed."""
    mints_short = "/".join(m[:6] + ".." for m in position.mints)
    pnl = position.realized_pnl
    pnl_str = f"${pnl:+.2f}"
    fees_str = f" (fees: ${position.fees_usd:.4f})" if position.fees_usd > 0 else ""
    duration = (position.exit_time - position.entry_time) if position.exit_time else 0
    print(f"  <<< CLOSE {position.direction.upper()} {mints_short}  "
          f"P&L {pnl_str}{fees_str}  ({reason}, {duration}s)")


def print_positions(positions: Dict, portfolio_value: float,
                    basket_states: Dict = None, max_age_half_lives: float = 5.0):
    """Print open positions table with time-to-timeout."""
    if positions:
        print(f"\n  {'Basket':<24} {'Dir':<6} {'Entry Z':>8} {'Curr Z':>8} {'Unrl P&L':>10} {'Exposure':>10} {'Timeout':>10}")
        print(f"  {'-'*24} {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

        now = int(time.time())
        for pair_key, pos in positions.items():
            pair_short = mints_to_label(pos.mints)
            exposure = sum(abs(pos.current_prices[i] * pos.quantities[i])
                          for i in range(pos.basket_size))
            # Compute time remaining until timeout
            timeout_str = "—"
            if basket_states:
                bsk = basket_states.get(pair_key)
                if bsk and bsk.half_life > 0 and bsk.half_life != float('inf'):
                    hl_seconds = bsk.half_life / 2.5
                    max_age = hl_seconds * max_age_half_lives
                    remaining = max_age - (now - pos.entry_time)
                    if remaining <= 0:
                        timeout_str = "NOW"
                    elif remaining < 3600:
                        timeout_str = f"{remaining / 60:.0f}m"
                    else:
                        timeout_str = f"{remaining / 3600:.1f}h"
            print(f"  {pair_short:<24} {pos.direction:<6} {pos.entry_zscore:>+8.3f} "
                  f"{pos.current_zscore:>+8.3f} {pos.unrealized_pnl:>+10.2f} {exposure:>10.0f} {timeout_str:>10}")

    print(f"\n  Portfolio value: ${portfolio_value:,.2f}")


def print_zscore_dashboard(basket_states: Dict, max_rows: int = 10):
    """Print z-score summary for monitored baskets (throttled)."""
    active = [(k, b) for k, b in basket_states.items() if b.price_buffers[0].count > 0]
    active.sort(key=lambda x: abs(x[1].current_zscore), reverse=True)

    if not active:
        return

    print(f"\n  {'Basket':<28} {'Z-score':>8} {'Edge bps':>9} {'Spread':>10} {'Obs':>5}")
    print(f"  {'-'*28} {'-'*8} {'-'*9} {'-'*10} {'-'*5}")

    for pair_key, bsk in active[:max_rows]:
        label = basket_label(bsk.symbols)
        edge_bps = abs(bsk.current_zscore) * bsk.spread_std * 10000 if bsk.spread_std > 0 else 0
        print(f"  {label:<28} {bsk.current_zscore:>+8.3f} {edge_bps:>8.1f} {bsk.current_spread:>10.6f} {bsk.price_buffers[0].count:>5}")

    if len(active) > max_rows:
        print(f"  ... and {len(active) - max_rows} more baskets")


def print_progress(polls: int, num_prices: int, signals: int, trades: int, start_time: float):
    """Print progress on a single line."""
    elapsed = time.time() - start_time
    sys.stdout.write(
        f"\r  Poll {polls:,} | Prices: {num_prices:,} | Signals: {signals:,} | Trades: {trades:,} | {elapsed:.0f}s"
    )
    sys.stdout.flush()


def print_discovery_status(discovery, baskets=None):
    """Print inline cointegration discovery status and active basket breakdown."""
    tracked = len(discovery.token_histories)
    sufficient = sum(1 for h in discovery.token_histories.values() if h.buffer.count >= 100)
    discovered = len(discovery.discovered_pairs)
    elapsed = time.time() - discovery.start_time
    warmup_remaining = max(0, discovery.config.coint_warmup_minutes * 60 - elapsed)

    if baskets:
        total = len(baskets)
        by_size = {}
        for b in baskets.values():
            by_size[b.basket_size] = by_size.get(b.basket_size, 0) + 1
        parts = []
        for sz in sorted(by_size):
            label = "pairs" if sz == 2 else f"{sz}-token"
            parts.append(f"{by_size[sz]} {label}")
        breakdown = ", ".join(parts)
        status = f"  Baskets: {total} active ({breakdown})"
    else:
        status = f"  Discovery: {tracked} tokens tracked, {sufficient} with data, {discovered} pairs found"

    if warmup_remaining > 0:
        status += f" | warmup {warmup_remaining/60:.0f}min left"
    print(status)


def print_regime_status(regime_detector):
    """Print current regime detection status."""
    if regime_detector is not None:
        print(f"  {regime_detector.status_str()}")


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
