#!/usr/bin/env python3
"""
Backtester for statalyzer.
Replays historical price candles through z-score signal logic and simulates trading.

Usage:
    # Single run
    python3 backtest.py --db exp_ll.db --entry-z 1.5 --exit-z 0.2 --min-spread-bps 10

    # Parameter sweep
    python3 backtest.py --db exp_ll.db --sweep

    # Pairs only
    python3 backtest.py --db exp_ll.db --max-basket-size 2 --entry-z 1.5

    # With whitelist
    python3 backtest.py --db exp_ll.db --token-whitelist SOL,bSOL,jitoSOL,mSOL,jupSOL
"""

import argparse
import json
import itertools
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Basket:
    """Cointegration parameters for one basket."""
    basket_key: str
    basket_size: int
    mints: List[str]
    symbols: List[str]
    hedge_ratios: List[float]
    spread_mean: float
    spread_std: float
    half_life: float


@dataclass
class BTPosition:
    """A simulated position."""
    basket_key: str
    basket_size: int
    direction: str          # "long" or "short"
    entry_z: float
    entry_spread: float
    entry_log_prices: List[float]
    entry_time: float
    size_usd: float         # total USD notional at entry


@dataclass
class ClosedTrade:
    """A completed round-trip trade."""
    basket_key: str
    basket_size: int
    direction: str
    entry_z: float
    exit_z: float
    entry_time: float
    exit_time: float
    pnl_usd: float          # net of slippage
    entry_spread: float
    exit_spread: float
    reason: str              # "exit", "stop_loss"


@dataclass
class BacktestParams:
    """All tunable parameters for one backtest run."""
    entry_z: float = 2.0
    exit_z: float = 0.3
    stop_z: float = 4.0
    max_entry_z: float = 6.0
    min_spread_bps: float = 0.0
    slippage_bps: float = 3.0
    max_positions: int = 10
    fixed_fraction: float = 0.05
    capital: float = 1000.0
    max_basket_size: int = 99
    lookback: int = 100


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Replays candles and simulates trading."""

    def __init__(self, params: BacktestParams, baskets: Dict[str, Basket],
                 token_whitelist: Optional[set] = None):
        self.params = params
        self.baskets = baskets
        self.token_whitelist = token_whitelist

        # State
        self.capital = params.capital
        self.positions: Dict[str, BTPosition] = {}
        self.closed_trades: List[ClosedTrade] = []

        # Per-basket spread history buffers
        self.spread_buffers: Dict[str, List[float]] = defaultdict(list)
        # Per-basket last log_prices (for exit P&L)
        self.last_log_prices: Dict[str, List[float]] = {}

    def run(self, candles: List[Tuple[str, float, List[float]]]) -> 'BacktestResult':
        """Run backtest over sorted candles: [(basket_key, timestamp, log_prices), ...]"""
        start_time = None
        end_time = None

        for basket_key, ts, log_prices in candles:
            if start_time is None:
                start_time = ts
            end_time = ts

            basket = self.baskets.get(basket_key)
            if basket is None:
                continue

            # Filter by basket size
            if basket.basket_size > self.params.max_basket_size:
                continue

            # Filter by whitelist (check symbols)
            if self.token_whitelist:
                if not all(s in self.token_whitelist for s in basket.symbols):
                    continue

            if len(log_prices) != basket.basket_size:
                continue

            self.last_log_prices[basket_key] = log_prices
            self._process_candle(basket_key, basket, ts, log_prices)

        # Force-close any remaining open positions at last known prices
        for bk in list(self.positions.keys()):
            self._force_close(bk, end_time or 0, "end_of_data")

        duration_hrs = (end_time - start_time) / 3600.0 if (start_time and end_time and end_time > start_time) else 1.0

        return BacktestResult(
            params=self.params,
            closed_trades=self.closed_trades,
            start_time=start_time or 0,
            end_time=end_time or 0,
            duration_hrs=duration_hrs,
            initial_capital=self.params.capital,
            final_capital=self.capital,
        )

    def _process_candle(self, basket_key: str, basket: Basket,
                        ts: float, log_prices: List[float]):
        """Process one candle: update spread buffer, check signals."""
        hr = np.array(basket.hedge_ratios)
        lp = np.array(log_prices)
        spread = float(lp @ hr)

        buf = self.spread_buffers[basket_key]
        buf.append(spread)
        # Trim to lookback
        if len(buf) > self.params.lookback:
            self.spread_buffers[basket_key] = buf[-self.params.lookback:]
            buf = self.spread_buffers[basket_key]

        if len(buf) < 30:
            return  # Not enough data

        arr = np.array(buf)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std < 1e-12:
            return

        z = (spread - mean) / std
        if abs(z) > 100:
            return  # Degenerate

        in_position = basket_key in self.positions

        # --- Exit / Stop-loss check ---
        if in_position:
            pos = self.positions[basket_key]
            # Stop loss
            if abs(z) > self.params.stop_z:
                self._close_position(basket_key, z, spread, ts, log_prices, "stop_loss")
                return
            # Exit: z reverted inside exit band
            if abs(z) < self.params.exit_z:
                self._close_position(basket_key, z, spread, ts, log_prices, "exit")
                return
            # Exit: z crossed through zero (overshot mean)
            if pos.direction == "long" and z > 0:
                self._close_position(basket_key, z, spread, ts, log_prices, "exit")
                return
            if pos.direction == "short" and z < 0:
                self._close_position(basket_key, z, spread, ts, log_prices, "exit")
                return

        # --- Entry check ---
        if not in_position:
            if len(self.positions) >= self.params.max_positions:
                return
            if abs(z) > self.params.max_entry_z:
                return  # Too extreme, likely broken cointegration

            # Min spread bps filter
            if self.params.min_spread_bps > 0:
                spread_dev_bps = abs(z) * std * 10000
                if spread_dev_bps < self.params.min_spread_bps:
                    return

            if z < -self.params.entry_z:
                self._open_position(basket_key, basket, "long", z, spread, ts, log_prices)
            elif z > self.params.entry_z:
                self._open_position(basket_key, basket, "short", z, spread, ts, log_prices)

    def _open_position(self, basket_key: str, basket: Basket, direction: str,
                       z: float, spread: float, ts: float, log_prices: List[float]):
        size_usd = self.params.capital * self.params.fixed_fraction  # fixed sizing, no compounding
        if size_usd <= 0:
            return

        # Deduct entry slippage (per leg, both legs)
        slippage_cost = size_usd * (self.params.slippage_bps / 10000.0) * basket.basket_size
        self.capital -= slippage_cost

        self.positions[basket_key] = BTPosition(
            basket_key=basket_key,
            basket_size=basket.basket_size,
            direction=direction,
            entry_z=z,
            entry_spread=spread,
            entry_log_prices=list(log_prices),
            entry_time=ts,
            size_usd=size_usd,
        )

    def _close_position(self, basket_key: str, z: float, spread: float,
                        ts: float, log_prices: List[float], reason: str):
        pos = self.positions.pop(basket_key)
        basket = self.baskets[basket_key]

        # P&L from spread change
        # Long spread: profit when spread goes up (z was negative, now closer to 0)
        # Short spread: profit when spread goes down (z was positive, now closer to 0)
        spread_change = spread - pos.entry_spread
        if pos.direction == "short":
            spread_change = -spread_change

        # Convert spread change to approximate USD P&L
        # spread is in log-price space. A spread change of X means ~X fraction of value.
        # Scale by position size.
        pnl = spread_change * pos.size_usd

        # Deduct exit slippage
        slippage_cost = pos.size_usd * (self.params.slippage_bps / 10000.0) * basket.basket_size
        pnl -= slippage_cost

        self.capital += pnl
        self.closed_trades.append(ClosedTrade(
            basket_key=basket_key,
            basket_size=pos.basket_size,
            direction=pos.direction,
            entry_z=pos.entry_z,
            exit_z=z,
            entry_time=pos.entry_time,
            exit_time=ts,
            pnl_usd=pnl,
            entry_spread=pos.entry_spread,
            exit_spread=spread,
            reason=reason,
        ))

    def _force_close(self, basket_key: str, ts: float, reason: str):
        """Close position at last known prices."""
        lp = self.last_log_prices.get(basket_key)
        if lp is None:
            # No data — close at zero P&L
            pos = self.positions.pop(basket_key)
            return
        basket = self.baskets.get(basket_key)
        if basket is None:
            self.positions.pop(basket_key)
            return
        hr = np.array(basket.hedge_ratios)
        spread = float(np.array(lp) @ hr)
        buf = self.spread_buffers.get(basket_key, [])
        if len(buf) < 2:
            self.positions.pop(basket_key)
            return
        arr = np.array(buf)
        std = float(np.std(arr))
        mean = float(np.mean(arr))
        z = (spread - mean) / std if std > 1e-12 else 0.0
        self._close_position(basket_key, z, spread, ts, lp, reason)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    params: BacktestParams
    closed_trades: List[ClosedTrade]
    start_time: float
    end_time: float
    duration_hrs: float
    initial_capital: float
    final_capital: float

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl_usd for t in self.closed_trades)

    @property
    def pnl_per_hr(self) -> float:
        return self.total_pnl / self.duration_hrs if self.duration_hrs > 0 else 0.0

    @property
    def num_trades(self) -> int:
        return len(self.closed_trades)

    @property
    def win_rate(self) -> float:
        if not self.closed_trades:
            return 0.0
        wins = sum(1 for t in self.closed_trades if t.pnl_usd > 0)
        return wins / len(self.closed_trades)

    @property
    def avg_pnl(self) -> float:
        if not self.closed_trades:
            return 0.0
        return self.total_pnl / len(self.closed_trades)

    @property
    def avg_win(self) -> float:
        wins = [t.pnl_usd for t in self.closed_trades if t.pnl_usd > 0]
        return np.mean(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl_usd for t in self.closed_trades if t.pnl_usd <= 0]
        return np.mean(losses) if losses else 0.0

    @property
    def max_drawdown(self) -> float:
        if not self.closed_trades:
            return 0.0
        equity = self.initial_capital
        peak = equity
        max_dd = 0.0
        for t in self.closed_trades:
            equity += t.pnl_usd
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def sharpe(self) -> float:
        """Annualized Sharpe from per-trade returns (assumes ~6 trades/day)."""
        if len(self.closed_trades) < 2:
            return 0.0
        returns = [t.pnl_usd / self.initial_capital for t in self.closed_trades]
        mu = np.mean(returns)
        sigma = np.std(returns)
        if sigma < 1e-12:
            return 0.0
        trades_per_day = len(self.closed_trades) / max(self.duration_hrs / 24, 1/24)
        return float(mu / sigma * np.sqrt(trades_per_day * 365))

    @property
    def stop_rate(self) -> float:
        if not self.closed_trades:
            return 0.0
        stops = sum(1 for t in self.closed_trades if t.reason == "stop_loss")
        return stops / len(self.closed_trades)

    def by_basket_size(self) -> Dict[int, 'BacktestResult']:
        """Group trades by basket size."""
        groups: Dict[int, List[ClosedTrade]] = defaultdict(list)
        for t in self.closed_trades:
            groups[t.basket_size].append(t)
        results = {}
        for size, trades in sorted(groups.items()):
            results[size] = BacktestResult(
                params=self.params, closed_trades=trades,
                start_time=self.start_time, end_time=self.end_time,
                duration_hrs=self.duration_hrs,
                initial_capital=self.initial_capital, final_capital=self.final_capital,
            )
        return results

    def by_z_bucket(self) -> Dict[str, 'BacktestResult']:
        """Group trades by entry z-score bucket."""
        buckets = [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 99.0)]
        groups: Dict[str, List[ClosedTrade]] = {}
        for lo, hi in buckets:
            label = f"|z| {lo:.1f}-{hi:.1f}"
            groups[label] = []
        for t in self.closed_trades:
            az = abs(t.entry_z)
            for lo, hi in buckets:
                if lo <= az < hi:
                    label = f"|z| {lo:.1f}-{hi:.1f}"
                    groups[label].append(t)
                    break
        results = {}
        for label, trades in groups.items():
            if trades:
                results[label] = BacktestResult(
                    params=self.params, closed_trades=trades,
                    start_time=self.start_time, end_time=self.end_time,
                    duration_hrs=self.duration_hrs,
                    initial_capital=self.initial_capital, final_capital=self.final_capital,
                )
        return results

    def by_time_period(self, period_hrs: float = 1.0) -> List[Tuple[str, 'BacktestResult']]:
        """Group trades by time period."""
        if not self.closed_trades:
            return []
        periods = []
        period_start = self.start_time
        while period_start < self.end_time:
            period_end = period_start + period_hrs * 3600
            trades = [t for t in self.closed_trades
                      if period_start <= t.exit_time < period_end]
            if trades:
                from datetime import datetime, timezone
                label = datetime.fromtimestamp(period_start, tz=timezone.utc).strftime('%m/%d %H:%M')
                periods.append((label, BacktestResult(
                    params=self.params, closed_trades=trades,
                    start_time=period_start, end_time=period_end,
                    duration_hrs=period_hrs,
                    initial_capital=self.initial_capital, final_capital=self.final_capital,
                )))
            period_start = period_end
        return periods


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_candles(db_path: str) -> List[Tuple[str, float, List[float]]]:
    """Load all price candles from a statalyzer DB, sorted by timestamp."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT basket_key, timestamp, log_prices_json "
        "FROM price_candles ORDER BY timestamp ASC"
    ).fetchall()
    conn.close()
    candles = []
    for basket_key, ts, lp_json in rows:
        log_prices = json.loads(lp_json)
        candles.append((basket_key, ts, log_prices))
    return candles


def load_baskets_from_scanner(scanner_db_path: str) -> Dict[str, Basket]:
    """Load cointegrated baskets from scanner DB."""
    try:
        conn = sqlite3.connect(f"file:{scanner_db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        print(f"ERROR: Cannot open scanner DB: {scanner_db_path}")
        sys.exit(1)

    try:
        rows = conn.execute("""
            SELECT basket_key, basket_size, mints_json, symbols_json,
                   hedge_ratios_json, spread_mean, spread_std, half_life
            FROM cointegration_results
            WHERE (eg_is_cointegrated = 1 OR johansen_is_cointegrated = 1)
        """).fetchall()
    except Exception as e:
        print(f"ERROR reading scanner DB: {e}")
        sys.exit(1)
    finally:
        conn.close()

    baskets = {}
    for row in rows:
        bk = row[0]
        baskets[bk] = Basket(
            basket_key=bk,
            basket_size=row[1],
            mints=json.loads(row[2]),
            symbols=json.loads(row[3]),
            hedge_ratios=json.loads(row[4]),
            spread_mean=row[5],
            spread_std=row[6],
            half_life=row[7],
        )
    return baskets


def load_baskets_from_candles_db(db_path: str) -> Dict[str, Basket]:
    """Fallback: infer baskets from the discovered_pairs table in the candles DB."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return {}

    baskets = {}
    try:
        rows = conn.execute("""
            SELECT pair_key, basket_size, mints_json, symbols_json,
                   hedge_ratios_json, spread_mean, spread_std, half_life
            FROM discovered_pairs
        """).fetchall()
        for row in rows:
            bk = row[0]
            baskets[bk] = Basket(
                basket_key=bk,
                basket_size=row[1],
                mints=json.loads(row[2]),
                symbols=json.loads(row[3]),
                hedge_ratios=json.loads(row[4]),
                spread_mean=row[5],
                spread_std=row[6],
                half_life=row[7],
            )
    except Exception:
        pass
    finally:
        conn.close()

    return baskets


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_result(result: BacktestResult, verbose: bool = False):
    """Print a single backtest result."""
    print("\n" + "=" * 70)
    print("  BACKTEST RESULTS")
    print("=" * 70)

    from datetime import datetime, timezone
    if result.start_time:
        start_str = datetime.fromtimestamp(result.start_time, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        end_str = datetime.fromtimestamp(result.end_time, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        print(f"  Period: {start_str} -> {end_str}  ({result.duration_hrs:.1f}h)")

    p = result.params
    print(f"  Params: entry_z={p.entry_z} exit_z={p.exit_z} stop_z={p.stop_z} "
          f"max_entry_z={p.max_entry_z}")
    print(f"          min_spread_bps={p.min_spread_bps} slippage_bps={p.slippage_bps} "
          f"max_positions={p.max_positions}")
    print(f"          capital=${p.capital:.0f} fixed_fraction={p.fixed_fraction} "
          f"max_basket_size={p.max_basket_size}")

    print(f"\n  {'Metric':<25} {'Value':>15}")
    print(f"  {'-'*25} {'-'*15}")
    print(f"  {'Total P&L':<25} {'${:+,.2f}'.format(result.total_pnl):>15}")
    print(f"  {'$/hr':<25} {'${:+,.4f}'.format(result.pnl_per_hr):>15}")
    print(f"  {'Num Trades':<25} {result.num_trades:>15}")
    print(f"  {'Win Rate':<25} {result.win_rate:>14.1%}")
    print(f"  {'Avg P&L/Trade':<25} {'${:+,.4f}'.format(result.avg_pnl):>15}")
    print(f"  {'Avg Win':<25} {'${:+,.4f}'.format(result.avg_win):>15}")
    print(f"  {'Avg Loss':<25} {'${:+,.4f}'.format(result.avg_loss):>15}")
    print(f"  {'Max Drawdown':<25} {result.max_drawdown:>14.1%}")
    print(f"  {'Sharpe (ann.)':<25} {result.sharpe:>15.2f}")
    print(f"  {'Stop Rate':<25} {result.stop_rate:>14.1%}")
    print(f"  {'Final Capital':<25} {'${:,.2f}'.format(result.final_capital):>15}")

    # By basket size
    by_size = result.by_basket_size()
    if len(by_size) > 1:
        print(f"\n  --- By Basket Size ---")
        print(f"  {'Size':<8} {'Trades':>8} {'P&L':>12} {'$/hr':>10} {'Win%':>8} {'StopR':>8}")
        for size, r in sorted(by_size.items()):
            print(f"  {size:<8} {r.num_trades:>8} {'${:+,.2f}'.format(r.total_pnl):>12} "
                  f"{'${:+,.4f}'.format(r.pnl_per_hr):>10} {r.win_rate:>7.1%} {r.stop_rate:>7.1%}")

    # By z-score bucket
    by_z = result.by_z_bucket()
    if by_z:
        print(f"\n  --- By Entry |z| ---")
        print(f"  {'Bucket':<16} {'Trades':>8} {'P&L':>12} {'Win%':>8} {'AvgPnL':>10}")
        for label, r in sorted(by_z.items()):
            print(f"  {label:<16} {r.num_trades:>8} {'${:+,.2f}'.format(r.total_pnl):>12} "
                  f"{r.win_rate:>7.1%} {'${:+,.4f}'.format(r.avg_pnl):>10}")

    # By time period (hourly, show top 10 best/worst)
    if verbose:
        by_time = result.by_time_period(period_hrs=1.0)
        if by_time:
            print(f"\n  --- By Hour (top 5 best / worst) ---")
            print(f"  {'Period':<16} {'Trades':>8} {'P&L':>12} {'Win%':>8}")
            sorted_periods = sorted(by_time, key=lambda x: x[1].total_pnl, reverse=True)
            for label, r in sorted_periods[:5]:
                print(f"  {label:<16} {r.num_trades:>8} {'${:+,.2f}'.format(r.total_pnl):>12} {r.win_rate:>7.1%}")
            if len(sorted_periods) > 5:
                print(f"  {'...':<16}")
                for label, r in sorted_periods[-5:]:
                    print(f"  {label:<16} {r.num_trades:>8} {'${:+,.2f}'.format(r.total_pnl):>12} {r.win_rate:>7.1%}")

    # Top baskets by P&L
    basket_pnl: Dict[str, float] = defaultdict(float)
    basket_count: Dict[str, int] = defaultdict(int)
    for t in result.closed_trades:
        basket_pnl[t.basket_key] += t.pnl_usd
        basket_count[t.basket_key] += 1
    if basket_pnl:
        sorted_baskets = sorted(basket_pnl.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  --- Top 10 Baskets by P&L ---")
        print(f"  {'Basket':<40} {'Trades':>8} {'P&L':>12}")
        for bk, pnl in sorted_baskets[:10]:
            short_key = bk[:37] + "..." if len(bk) > 40 else bk
            print(f"  {short_key:<40} {basket_count[bk]:>8} {'${:+,.4f}'.format(pnl):>12}")
        if len(sorted_baskets) > 10:
            print(f"  ... and {len(sorted_baskets) - 10} more baskets")

    print()


def print_sweep_results(results: List[BacktestResult], top_n: int = 20):
    """Print parameter sweep comparison table."""
    # Sort by $/hr descending
    results.sort(key=lambda r: r.pnl_per_hr, reverse=True)

    print("\n" + "=" * 120)
    print("  PARAMETER SWEEP RESULTS  (top {} of {})".format(min(top_n, len(results)), len(results)))
    print("=" * 120)
    print(f"  {'#':>3} {'entry_z':>8} {'exit_z':>7} {'stop_z':>7} {'min_bps':>8} {'max_bsk':>8} "
          f"{'Trades':>7} {'P&L':>11} {'$/hr':>10} {'Win%':>7} {'Sharpe':>8} {'MaxDD':>7} {'StopR':>7}")
    print(f"  {'-'*3} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*8} "
          f"{'-'*7} {'-'*11} {'-'*10} {'-'*7} {'-'*8} {'-'*7} {'-'*7}")

    for i, r in enumerate(results[:top_n]):
        p = r.params
        pnl_str = '${:+,.2f}'.format(r.total_pnl)
        phr_str = '${:+,.4f}'.format(r.pnl_per_hr)
        print(f"  {i+1:>3} {p.entry_z:>8.1f} {p.exit_z:>7.1f} {p.stop_z:>7.1f} "
              f"{p.min_spread_bps:>8.0f} {p.max_basket_size:>8} "
              f"{r.num_trades:>7} {pnl_str:>11} {phr_str:>10} "
              f"{r.win_rate:>6.1%} {r.sharpe:>8.2f} {r.max_drawdown:>6.1%} {r.stop_rate:>6.1%}")

    # Also show bottom 5
    if len(results) > top_n:
        print(f"\n  --- Bottom 5 ---")
        for r in results[-5:]:
            p = r.params
            pnl_str = '${:+,.2f}'.format(r.total_pnl)
            phr_str = '${:+,.4f}'.format(r.pnl_per_hr)
            print(f"      {p.entry_z:>8.1f} {p.exit_z:>7.1f} {p.stop_z:>7.1f} "
                  f"{p.min_spread_bps:>8.0f} {p.max_basket_size:>8} "
                  f"{r.num_trades:>7} {pnl_str:>11} {phr_str:>10} "
                  f"{r.win_rate:>6.1%} {r.sharpe:>8.2f} {r.max_drawdown:>6.1%} {r.stop_rate:>6.1%}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Statalyzer backtester: replay historical candles and simulate trading.")
    parser.add_argument('--db', required=True, help='Source DB with price_candles table')
    parser.add_argument('--scanner-db', default=None,
                        help='Scanner DB with cointegration_results (default: use discovered_pairs from --db)')
    parser.add_argument('--entry-z', type=float, default=2.0)
    parser.add_argument('--exit-z', type=float, default=0.3)
    parser.add_argument('--stop-z', type=float, default=4.0)
    parser.add_argument('--max-entry-z', type=float, default=6.0)
    parser.add_argument('--min-spread-bps', type=float, default=0.0)
    parser.add_argument('--slippage-bps', type=float, default=3.0,
                        help='Simulated slippage per leg in bps (default: 3)')
    parser.add_argument('--max-positions', type=int, default=10)
    parser.add_argument('--fixed-fraction', type=float, default=0.05)
    parser.add_argument('--capital', type=float, default=1000.0)
    parser.add_argument('--max-basket-size', type=int, default=99,
                        help='Limit to N-token baskets (e.g., 2 for pairs only)')
    parser.add_argument('--lookback', type=int, default=100,
                        help='Rolling lookback window for z-score (default: 100)')
    parser.add_argument('--token-whitelist', type=str, default=None,
                        help='Comma-separated symbols to restrict trading to')
    parser.add_argument('--sweep', action='store_true',
                        help='Test multiple parameter combinations')
    parser.add_argument('--verbose', '-v', action='store_true')
    return parser.parse_args()


def run_single(args, candles, baskets, token_whitelist):
    """Run a single backtest."""
    params = BacktestParams(
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        stop_z=args.stop_z,
        max_entry_z=args.max_entry_z,
        min_spread_bps=args.min_spread_bps,
        slippage_bps=args.slippage_bps,
        max_positions=args.max_positions,
        fixed_fraction=args.fixed_fraction,
        capital=args.capital,
        max_basket_size=args.max_basket_size,
        lookback=args.lookback,
    )
    engine = BacktestEngine(params, baskets, token_whitelist)
    result = engine.run(candles)
    print_result(result, verbose=args.verbose)
    return result


def run_sweep(args, candles, baskets, token_whitelist):
    """Run parameter sweep."""
    entry_z_vals = [1.0, 1.5, 2.0, 2.5]
    exit_z_vals = [0.1, 0.2, 0.3, 0.5]
    stop_z_vals = [args.stop_z]  # Keep stop fixed from CLI
    min_spread_vals = [5, 10, 15, 20]
    max_basket_vals = [2, 3, 4]

    combos = list(itertools.product(entry_z_vals, exit_z_vals, stop_z_vals,
                                     min_spread_vals, max_basket_vals))
    total = len(combos)
    print(f"\nRunning {total} parameter combinations...")

    results = []
    t0 = time.time()
    for i, (ez, xz, sz, msbps, mbs) in enumerate(combos):
        # Skip nonsensical combos where exit >= entry
        if xz >= ez:
            continue

        params = BacktestParams(
            entry_z=ez, exit_z=xz, stop_z=sz,
            max_entry_z=args.max_entry_z,
            min_spread_bps=msbps,
            slippage_bps=args.slippage_bps,
            max_positions=args.max_positions,
            fixed_fraction=args.fixed_fraction,
            capital=args.capital,
            max_basket_size=mbs,
            lookback=args.lookback,
        )
        engine = BacktestEngine(params, baskets, token_whitelist)
        result = engine.run(candles)
        results.append(result)

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{total}] {rate:.1f} runs/sec", end='\r')

    elapsed = time.time() - t0
    print(f"\n  Completed {len(results)} runs in {elapsed:.1f}s "
          f"({len(results)/elapsed:.1f} runs/sec)")

    # Filter out runs with zero trades
    results_with_trades = [r for r in results if r.num_trades > 0]
    if not results_with_trades:
        print("\n  No parameter combination produced any trades!")
        return

    print_sweep_results(results_with_trades)

    # Print the best result in detail
    best = max(results_with_trades, key=lambda r: r.pnl_per_hr)
    print("  --- Best Run (by $/hr) ---")
    print_result(best, verbose=args.verbose)


def main():
    args = parse_args()

    # Load candles
    print(f"Loading candles from {args.db}...")
    candles = load_candles(args.db)
    if not candles:
        print("ERROR: No candles found in DB")
        sys.exit(1)

    # Gather unique basket keys
    basket_keys = set(c[0] for c in candles)
    print(f"  {len(candles)} candles across {len(basket_keys)} baskets")

    # Load baskets (cointegration params)
    baskets = {}
    if args.scanner_db:
        print(f"Loading baskets from scanner DB: {args.scanner_db}...")
        baskets = load_baskets_from_scanner(args.scanner_db)
    if not baskets:
        print(f"Loading baskets from discovered_pairs in {args.db}...")
        baskets = load_baskets_from_candles_db(args.db)
    if not baskets:
        print("ERROR: No basket/cointegration data found. Provide --scanner-db or ensure "
              "discovered_pairs table exists in --db.")
        sys.exit(1)

    # Filter to baskets that have candles
    matched = set(baskets.keys()) & basket_keys
    print(f"  {len(baskets)} baskets loaded, {len(matched)} have candle data")

    if not matched:
        print("ERROR: No baskets match candle data. Check that basket keys align.")
        sys.exit(1)

    # Token whitelist
    token_whitelist = None
    if args.token_whitelist:
        token_whitelist = set(s.strip() for s in args.token_whitelist.split(','))
        print(f"  Token whitelist: {token_whitelist}")

    if args.sweep:
        run_sweep(args, candles, baskets, token_whitelist)
    else:
        run_single(args, candles, baskets, token_whitelist)


if __name__ == '__main__':
    main()
