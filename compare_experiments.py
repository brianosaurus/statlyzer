#!/usr/bin/env python3
"""
Compare statalyzer experiments running in parallel.
Reads each experiment's DB and computes performance metrics including
direction breakdown, z-score analysis, basket size, and signal rejections.

Usage:
    python3 compare_experiments.py [--since-restart] [--csv output.csv]

By default compares all trades. --since-restart only counts trades
opened after the most recent restart (detected by entry_time gap).
"""

import argparse
import json
import math
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


EXPERIMENTS = {
    "P": {"db": "exp_p.db", "label": "LIVE $88",
           "desc": "40pos/10tok/3.63x/6%/z2.7-3.0/$88/LIVE"},
    # Phase 1: Diagnostic — sweep slippage to find break-even
    "AA": {"db": "exp_aa.db", "label": "0bps $5k",
            "desc": "40pos/10tok/5x/5%/z2.0-4.0/ez0.3/0bps/30s/$5k"},
    "BB": {"db": "exp_bb.db", "label": "5bps $5k",
            "desc": "40pos/10tok/5x/5%/z2.0-4.0/ez0.3/5bps/30s/$5k"},
    "CC": {"db": "exp_cc.db", "label": "15bps $5k",
            "desc": "40pos/10tok/5x/5%/z2.0-4.0/ez0.3/15bps/30s/$5k"},
    "DD": {"db": "exp_dd.db", "label": "25bps $5k",
            "desc": "40pos/10tok/5x/5%/z2.0-4.0/ez0.3/25bps/30s/$5k"},
    # Phase 2: Optimized for profit
    "EE": {"db": "exp_ee.db", "label": "MaxVol $20k",
            "desc": "60pos/15tok/5x/10%/z1.5-4.0/ez0.1/10bps/30s/100hr/$20k"},
    "FF": {"db": "exp_ff.db", "label": "HiZ $10k",
            "desc": "40pos/10tok/5x/7%/z2.5-4.0/ez0.2/10bps/30s/noStop/$10k"},
    "GG": {"db": "exp_gg.db", "label": "Fast $10k",
            "desc": "40pos/10tok/5x/10%/z2.0-4.0/ez0.2/15bps/15s/60hr/$10k"},
    "HH": {"db": "exp_hh.db", "label": "P-opt $10k",
            "desc": "40pos/10tok/4x/7%/z2.7-3.5/ez0.3/10bps/30s/40hr/$10k"},
    # Phase 3: Whitelist (real slippage, low-slippage tokens only)
    "II": {"db": "exp_ii.db", "label": "WL $10k",
            "desc": "40pos/10tok/5x/7%/z2.0-4.0/ez0.2/24bps/30s/WL/$10k"},
    "JJ": {"db": "exp_jj.db", "label": "$1k/d $50k",
            "desc": "60pos/15tok/5x/10%/z1.5-4.0/ez0.1/24bps/30s/WL/100hr/$50k"},
    "KK": {"db": "exp_kk.db", "label": "$1k/d opt",
            "desc": "60pos/15tok/5x/10%/z1.5-4.0/ez0.1/10bps/30s/WL/100hr/$50k"},
    # Phase 4: LST-only (pure LST baskets, realistic 5bps)
    "LL": {"db": "exp_ll.db", "label": "LST $10k",
            "desc": "40pos/10tok/5x/7%/z2.0-4.0/ez0.2/5bps/30s/LST/$10k"},
    "MM": {"db": "exp_mm.db", "label": "LST $50k",
            "desc": "60pos/15tok/5x/10%/z1.5-4.0/ez0.1/5bps/30s/LST/100hr/$50k"},
}


@dataclass
class DirectionStats:
    direction: str
    count: int
    total_pnl: float
    avg_pnl: float
    win_rate: float


@dataclass
class ZBucketStats:
    bucket: str
    count: int
    total_pnl: float
    avg_pnl: float


@dataclass
class BasketSizeStats:
    size: int
    count: int
    total_pnl: float
    avg_pnl: float


@dataclass
class TradeStats:
    name: str
    label: str
    desc: str
    duration_min: float
    num_closed: int
    num_open: int
    total_pnl: float
    pnl_per_hour: float
    pnl_per_trade: float
    win_rate: float
    num_winners: int
    num_losers: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    max_win: float
    max_loss: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    avg_duration_s: float
    portfolio_value: float
    start_time: float = 0.0
    # Breakdowns
    direction_stats: List[DirectionStats] = field(default_factory=list)
    basket_size_stats: List[BasketSizeStats] = field(default_factory=list)
    zscore_stats: List[ZBucketStats] = field(default_factory=list)
    exit_type_stats: List[Tuple[str, int, float]] = field(default_factory=list)
    signal_rejections: List[Tuple[str, int]] = field(default_factory=list)
    total_signals: int = 0
    acted_signals: int = 0
    num_stopped_out: int = 0
    stopped_out_pnl: float = 0.0
    projected_daily: float = 0.0


def detect_restart_time(conn) -> float:
    """Find the most recent large gap in entry times (= restart)."""
    rows = conn.execute(
        "SELECT entry_time FROM positions ORDER BY entry_time"
    ).fetchall()
    if len(rows) < 2:
        return rows[0][0] if rows else time.time()
    last_gap_time = rows[0][0]
    for i in range(1, len(rows)):
        gap = rows[i][0] - rows[i - 1][0]
        if gap > 7200:
            last_gap_time = rows[i][0]
    return last_gap_time


def find_snapshot_time(all_dbs: dict) -> float:
    """Find the snapshot time by intersecting entry_times across all experiments."""
    entry_time_sets = []
    for info in all_dbs.values():
        try:
            conn = sqlite3.connect(f"file:{info['db']}?mode=ro", uri=True)
            rows = conn.execute("SELECT entry_time FROM positions ORDER BY entry_time").fetchall()
            entry_time_sets.append(set(r[0] for r in rows))
            conn.close()
        except Exception:
            pass
    if not entry_time_sets:
        return 0
    common = entry_time_sets[0]
    for s in entry_time_sets[1:]:
        common = common & s
    if not common:
        return 0
    smallest = min(len(s) for s in entry_time_sets)
    if len(common) < smallest * 0.5:
        return 0
    return max(common)


def analyze_experiment(name: str, info: dict, since_restart: bool,
                       snapshot_time: float = 0) -> Optional[TradeStats]:
    try:
        conn = sqlite3.connect(f"file:{info['db']}?mode=ro", uri=True)
    except Exception as e:
        print(f"  Cannot open {info['db']}: {e}", file=sys.stderr)
        return None

    now = time.time()

    if since_restart:
        start_time = detect_restart_time(conn)
    elif snapshot_time > 0:
        row = conn.execute(
            "SELECT MIN(entry_time) FROM positions WHERE entry_time > ?",
            (snapshot_time,)
        ).fetchone()
        start_time = row[0] if row and row[0] else snapshot_time
    else:
        row = conn.execute("SELECT MIN(entry_time) FROM positions").fetchone()
        start_time = row[0] if row and row[0] else now

    duration_min = (now - start_time) / 60
    if duration_min < 1:
        conn.close()
        return None

    # Closed trades since start_time
    closed = conn.execute("""
        SELECT realized_pnl, entry_time, exit_time, direction, basket_size,
               entry_zscore, status
        FROM positions
        WHERE status != 'open' AND entry_time >= ?
        ORDER BY entry_time
    """, (start_time,)).fetchall()

    open_pos = conn.execute(
        "SELECT COUNT(*) FROM positions WHERE status = 'open' AND entry_time >= ?",
        (start_time,)
    ).fetchone()[0]

    # Current capital
    cap_row = conn.execute(
        "SELECT value FROM config_state WHERE key = 'initial_capital'"
    ).fetchone()
    current_capital = float(cap_row[0]) if cap_row else 5000.0

    # --- Direction breakdown ---
    dir_rows = conn.execute("""
        SELECT direction, COUNT(*), ROUND(SUM(realized_pnl),2), ROUND(AVG(realized_pnl),2),
               ROUND(SUM(CASE WHEN realized_pnl > 0 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 1)
        FROM positions WHERE status != 'open' AND entry_time >= ?
        GROUP BY direction
    """, (start_time,)).fetchall()
    direction_stats = [DirectionStats(r[0], r[1], r[2], r[3], r[4]) for r in dir_rows]

    # --- Basket size breakdown ---
    bs_rows = conn.execute("""
        SELECT basket_size, COUNT(*), ROUND(SUM(realized_pnl),2), ROUND(AVG(realized_pnl),2)
        FROM positions WHERE status != 'open' AND entry_time >= ?
        GROUP BY basket_size
    """, (start_time,)).fetchall()
    basket_size_stats = [BasketSizeStats(r[0], r[1], r[2], r[3]) for r in bs_rows]

    # --- Z-score bucket breakdown ---
    z_rows = conn.execute("""
        SELECT ROUND(ABS(entry_zscore),1) as z_bucket, COUNT(*),
               ROUND(SUM(realized_pnl),2), ROUND(AVG(realized_pnl),2)
        FROM positions WHERE status != 'open' AND entry_time >= ?
        GROUP BY z_bucket ORDER BY z_bucket
    """, (start_time,)).fetchall()
    # Aggregate into wider buckets for readability
    z_buckets = {}
    for z_val, cnt, total, avg in z_rows:
        if z_val < 2.5:
            key = "<2.5"
        elif z_val < 2.7:
            key = "2.5-2.7"
        elif z_val < 2.9:
            key = "2.7-2.9"
        elif z_val < 3.1:
            key = "2.9-3.1"
        else:
            key = "3.1+"
        if key not in z_buckets:
            z_buckets[key] = [0, 0.0]
        z_buckets[key][0] += cnt
        z_buckets[key][1] += total
    zscore_stats = []
    for key in ["<2.5", "2.5-2.7", "2.7-2.9", "2.9-3.1", "3.1+"]:
        if key in z_buckets:
            cnt, total = z_buckets[key]
            zscore_stats.append(ZBucketStats(key, cnt, total, total / cnt if cnt else 0))

    # --- Exit type breakdown ---
    exit_rows = conn.execute("""
        SELECT status, COUNT(*), ROUND(SUM(realized_pnl),2)
        FROM positions WHERE status != 'open' AND entry_time >= ?
        GROUP BY status
    """, (start_time,)).fetchall()
    exit_type_stats = [(r[0], r[1], r[2]) for r in exit_rows]

    # Stop-out stats
    stopped = [r for r in exit_type_stats if r[0] == 'stopped_out']
    num_stopped_out = stopped[0][1] if stopped else 0
    stopped_out_pnl = stopped[0][2] if stopped else 0.0

    # --- Signal rejection reasons (last hour) ---
    sig_rows = conn.execute("""
        SELECT reason_not_acted, COUNT(*)
        FROM signals WHERE acted_on = 0
          AND reason_not_acted IS NOT NULL
          AND timestamp > (strftime('%%s','now') - 3600)
        GROUP BY reason_not_acted ORDER BY COUNT(*) DESC LIMIT 8
    """).fetchall()
    signal_rejections = [(r[0], r[1]) for r in sig_rows]

    # Total vs acted signals
    sig_totals = conn.execute("""
        SELECT COUNT(*), SUM(CASE WHEN acted_on = 1 THEN 1 ELSE 0 END)
        FROM signals WHERE timestamp >= ?
    """, (int(start_time),)).fetchone()
    total_signals = sig_totals[0] if sig_totals else 0
    acted_signals = sig_totals[1] if sig_totals and sig_totals[1] else 0

    conn.close()

    if not closed:
        return TradeStats(
            name=name, label=info["label"], desc=info["desc"],
            duration_min=duration_min, num_closed=0, num_open=open_pos,
            total_pnl=0, pnl_per_hour=0, pnl_per_trade=0,
            win_rate=0, num_winners=0, num_losers=0,
            avg_win=0, avg_loss=0, profit_factor=0, expectancy=0,
            max_win=0, max_loss=0, sharpe=0, sortino=0,
            max_drawdown_pct=0, avg_duration_s=0,
            portfolio_value=current_capital, start_time=start_time,
            total_signals=total_signals, acted_signals=acted_signals,
            signal_rejections=signal_rejections,
        )

    pnls = [r[0] for r in closed]
    durations = [(r[2] - r[1]) for r in closed if r[2]]

    total_pnl = sum(pnls)
    num_closed = len(pnls)
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]

    win_rate = len(winners) / num_closed if num_closed else 0
    avg_win = sum(winners) / len(winners) if winners else 0
    avg_loss = sum(losers) / len(losers) if losers else 0
    gross_profit = sum(winners)
    gross_loss = abs(sum(losers))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    expectancy = total_pnl / num_closed if num_closed else 0

    # Sharpe ratio
    if len(pnls) >= 2:
        mean_r = sum(pnls) / len(pnls)
        std_r = math.sqrt(sum((p - mean_r) ** 2 for p in pnls) / (len(pnls) - 1))
        sharpe = (mean_r / std_r) * math.sqrt(252) if std_r > 0 else 0
    else:
        sharpe = 0

    # Sortino ratio
    if len(pnls) >= 2:
        mean_r = sum(pnls) / len(pnls)
        downside = [min(0, p - mean_r) ** 2 for p in pnls]
        downside_dev = math.sqrt(sum(downside) / (len(downside) - 1))
        sortino = (mean_r / downside_dev) * math.sqrt(252) if downside_dev > 0 else 0
    else:
        sortino = 0

    # Max drawdown
    cumulative = 0
    peak = 0
    max_dd = 0
    for p in pnls:
        cumulative += p
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    max_dd_pct = max_dd / current_capital * 100 if max_dd > 0 else 0

    pnl_per_hour = total_pnl / (duration_min / 60) if duration_min > 0 else 0
    avg_dur = sum(durations) / len(durations) if durations else 0
    projected_daily = pnl_per_hour * 24

    return TradeStats(
        name=name, label=info["label"], desc=info["desc"],
        duration_min=duration_min, start_time=start_time,
        num_closed=num_closed, num_open=open_pos,
        total_pnl=total_pnl, pnl_per_hour=pnl_per_hour,
        pnl_per_trade=expectancy,
        win_rate=win_rate, num_winners=len(winners), num_losers=len(losers),
        avg_win=avg_win, avg_loss=avg_loss,
        profit_factor=profit_factor, expectancy=expectancy,
        max_win=max(pnls) if pnls else 0,
        max_loss=min(pnls) if pnls else 0,
        sharpe=sharpe, sortino=sortino,
        max_drawdown_pct=max_dd_pct, avg_duration_s=avg_dur,
        portfolio_value=current_capital,
        direction_stats=direction_stats,
        basket_size_stats=basket_size_stats,
        zscore_stats=zscore_stats,
        exit_type_stats=exit_type_stats,
        signal_rejections=signal_rejections,
        total_signals=total_signals,
        acted_signals=acted_signals,
        num_stopped_out=num_stopped_out,
        stopped_out_pnl=stopped_out_pnl,
        projected_daily=projected_daily,
    )


def print_comparison(results: List[TradeStats]):
    results.sort(key=lambda r: r.pnl_per_hour, reverse=True)

    W = 100
    print(f"\n{'=' * W}")
    print(f"  STATALYZER EXPERIMENT COMPARISON")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * W}")

    # Runtime
    print(f"\n  {'#':<3} {'Name':<14} {'Started':<20} {'Runtime':>8} {'Capital':>10}")
    print(f"  {'-'*3} {'-'*14} {'-'*20} {'-'*8} {'-'*10}")
    for r in results:
        started = time.strftime('%Y-%m-%d %H:%M', time.localtime(r.start_time))
        hrs = r.duration_min / 60
        runtime = f"{hrs:.1f}h" if hrs >= 1 else f"{r.duration_min:.0f}m"
        print(f"  {r.name:<3} {r.label:<14} {started:<20} {runtime:>8} ${r.portfolio_value:>9,.2f}")

    # ── SUMMARY ──
    print(f"\n  PERFORMANCE SUMMARY")
    print(f"  {'-' * (W - 4)}")
    hdr = f"  {'#':<3} {'Config':<28} {'Trades':>6} {'Open':>4} {'PnL':>9} {'$/hr':>8} {'$/day':>8} {'Win%':>6} {'PF':>6} {'Sharpe':>7}"
    print(hdr)
    print(f"  {'-'*3} {'-'*28} {'-'*6} {'-'*4} {'-'*9} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*7}")
    for r in results:
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor < 100 else "inf"
        print(f"  {r.name:<3} {r.desc:<28} {r.num_closed:>6} {r.num_open:>4} "
              f"${r.total_pnl:>+7.2f} ${r.pnl_per_hour:>6.2f} ${r.projected_daily:>6.0f} "
              f"{r.win_rate:>5.1%} {pf_str:>6} {r.sharpe:>+7.2f}")

    # ── DETAILED METRICS ──
    print(f"\n  DETAILED METRICS")
    print(f"  {'-' * (W - 4)}")
    hdr2 = f"  {'#':<3} {'AvgWin':>7} {'AvgLoss':>8} {'MaxWin':>7} {'MaxLoss':>8} {'MaxDD%':>7} {'Expect':>7} {'Sortino':>8} {'AvgDur':>7} {'StopOut':>12}"
    print(hdr2)
    print(f"  {'-'*3} {'-'*7} {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*12}")
    for r in results:
        dur_str = f"{r.avg_duration_s:.0f}s" if r.avg_duration_s < 3600 else f"{r.avg_duration_s/3600:.1f}h"
        stop_str = f"{r.num_stopped_out} (${r.stopped_out_pnl:+.0f})"
        print(f"  {r.name:<3} ${r.avg_win:>+5.2f} ${r.avg_loss:>+6.2f} ${r.max_win:>+5.2f} ${r.max_loss:>+6.2f} "
              f"{r.max_drawdown_pct:>6.2f}% ${r.expectancy:>+5.3f} {r.sortino:>+8.2f} {dur_str:>7} {stop_str:>12}")

    # ── DIRECTION BREAKDOWN ──
    print(f"\n  P&L BY DIRECTION")
    print(f"  {'-' * (W - 4)}")
    print(f"  {'#':<3} {'Direction':<8} {'Trades':>7} {'Total PnL':>10} {'Avg PnL':>9} {'Win%':>7}")
    print(f"  {'-'*3} {'-'*8} {'-'*7} {'-'*10} {'-'*9} {'-'*7}")
    for r in results:
        for d in r.direction_stats:
            print(f"  {r.name:<3} {d.direction:<8} {d.count:>7} ${d.total_pnl:>+8.2f} ${d.avg_pnl:>+7.2f} {d.win_rate:>6.1f}%")

    # ── BASKET SIZE BREAKDOWN ──
    print(f"\n  P&L BY BASKET SIZE")
    print(f"  {'-' * (W - 4)}")
    print(f"  {'#':<3} {'Tokens':>6} {'Trades':>7} {'Total PnL':>10} {'Avg PnL':>9}")
    print(f"  {'-'*3} {'-'*6} {'-'*7} {'-'*10} {'-'*9}")
    for r in results:
        for b in r.basket_size_stats:
            print(f"  {r.name:<3} {b.size:>6} {b.count:>7} ${b.total_pnl:>+8.2f} ${b.avg_pnl:>+7.2f}")

    # ── Z-SCORE BUCKET BREAKDOWN ──
    print(f"\n  P&L BY ENTRY Z-SCORE")
    print(f"  {'-' * (W - 4)}")
    print(f"  {'#':<3} {'|Z| Range':<10} {'Trades':>7} {'Total PnL':>10} {'Avg PnL':>9}")
    print(f"  {'-'*3} {'-'*10} {'-'*7} {'-'*10} {'-'*9}")
    for r in results:
        for z in r.zscore_stats:
            print(f"  {r.name:<3} {z.bucket:<10} {z.count:>7} ${z.total_pnl:>+8.2f} ${z.avg_pnl:>+7.2f}")
        if r != results[-1]:
            print()

    # ── SIGNAL CONVERSION ──
    print(f"\n  SIGNAL CONVERSION")
    print(f"  {'-' * (W - 4)}")
    print(f"  {'#':<3} {'Total Sigs':>11} {'Acted':>7} {'Rate':>7}")
    print(f"  {'-'*3} {'-'*11} {'-'*7} {'-'*7}")
    for r in results:
        rate = r.acted_signals / r.total_signals * 100 if r.total_signals else 0
        print(f"  {r.name:<3} {r.total_signals:>11,} {r.acted_signals:>7,} {rate:>6.1f}%")

    # ── SIGNAL REJECTIONS (last hour) ──
    print(f"\n  SIGNAL REJECTIONS (last hour)")
    print(f"  {'-' * (W - 4)}")
    for r in results:
        if r.signal_rejections:
            print(f"  [{r.name}]")
            for reason, count in r.signal_rejections:
                print(f"      {count:>5}x  {reason}")

    # ── RECOMMENDATION ──
    print(f"\n{'=' * W}")
    best = results[0]
    print(f"  BEST: Experiment {best.name} ({best.label})")
    print(f"  ${best.pnl_per_hour:.2f}/hr (${best.projected_daily:.0f}/day projected) | "
          f"{best.win_rate:.1%} win rate | PF {best.profit_factor:.2f} | Sharpe {best.sharpe:+.2f}")

    if best.num_closed < 30:
        print(f"  WARNING: Only {best.num_closed} trades — not yet statistically significant")
    if best.duration_min < 120:
        print(f"  WARNING: Only {best.duration_min:.0f} min of data — let it run longer")

    if len(results) >= 2 and results[0].num_closed >= 10 and results[1].num_closed >= 10:
        diff = results[0].pnl_per_hour - results[1].pnl_per_hour
        if diff < 2.0:
            print(f"  NOTE: Top 2 are close (${diff:.2f}/hr apart) — may not be significant")

    print(f"{'=' * W}\n")


def write_csv(results: List[TradeStats], path: str):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "label", "config", "duration_min", "closed_trades", "open_trades",
                     "total_pnl", "pnl_per_hour", "projected_daily", "win_rate",
                     "avg_win", "avg_loss", "profit_factor", "expectancy",
                     "sharpe", "sortino", "max_drawdown_pct", "avg_duration_s",
                     "portfolio_value", "total_signals", "acted_signals",
                     "stopped_out", "stopped_out_pnl"])
        for r in results:
            w.writerow([r.name, r.label, r.desc, f"{r.duration_min:.1f}", r.num_closed, r.num_open,
                         f"{r.total_pnl:.2f}", f"{r.pnl_per_hour:.2f}", f"{r.projected_daily:.0f}",
                         f"{r.win_rate:.3f}", f"{r.avg_win:.2f}", f"{r.avg_loss:.2f}",
                         f"{r.profit_factor:.2f}", f"{r.expectancy:.3f}",
                         f"{r.sharpe:.2f}", f"{r.sortino:.2f}",
                         f"{r.max_drawdown_pct:.2f}", f"{r.avg_duration_s:.0f}",
                         f"{r.portfolio_value:.2f}", r.total_signals, r.acted_signals,
                         r.num_stopped_out, f"{r.stopped_out_pnl:.2f}"])
    print(f"  CSV written to {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare statalyzer experiments")
    parser.add_argument("--since-restart", action="store_true",
                        help="Only count trades since the most recent restart")
    parser.add_argument("--csv", type=str, default=None,
                        help="Write results to CSV file")
    args = parser.parse_args()

    snapshot_time = 0

    results = []
    for name, info in sorted(EXPERIMENTS.items()):
        stats = analyze_experiment(name, info, args.since_restart, snapshot_time)
        if stats:
            results.append(stats)

    if not results:
        print("No experiment data found.")
        return

    print_comparison(results)

    if args.csv:
        write_csv(results, args.csv)


if __name__ == "__main__":
    main()
