#!/usr/bin/env python3
"""
Analyze statalyzer results from a remote or local DB.

Usage:
    python analyze.py                          # fetch from frankfurt, analyze
    python analyze.py --local statalyzer.db    # analyze a local DB
    python analyze.py --host frankfurt          # fetch from custom host
"""

import argparse
import os
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta


def fetch_db(host: str, remote_path: str, local_path: str) -> bool:
    """SCP the DB + WAL/SHM from remote host, then checkpoint WAL into main DB."""
    print(f"Fetching DB from {host}:{remote_path} ...")
    # Copy all DB files (main, WAL, SHM)
    files = [remote_path, remote_path + "-wal", remote_path + "-shm"]
    local_dir = os.path.dirname(local_path)
    local_base = os.path.basename(local_path)
    for f in files:
        suffix = f.replace(remote_path, "")
        dest = local_path + suffix
        result = subprocess.run(
            ["scp", f"{host}:{f}", dest],
            capture_output=True, text=True,
        )
        # WAL/SHM may not exist, that's OK
    if not os.path.exists(local_path):
        print(f"  Failed to download DB")
        return False
    # Checkpoint WAL into main DB so we get all data
    try:
        conn = sqlite3.connect(local_path)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
    except Exception:
        pass
    size = os.path.getsize(local_path)
    print(f"  Downloaded {size / 1024:.1f} KB")
    return True


def fmt_time(ts):
    """Unix timestamp to readable string."""
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def fmt_duration(seconds):
    """Seconds to human-readable duration."""
    if not seconds or seconds <= 0:
        return "—"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def analyze(db_path: str):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # ── Overview ──────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  STATALYZER — Performance Report")
    print(f"{'=' * 72}")
    print(f"  DB: {db_path}")

    # Time range
    row = conn.execute("SELECT MIN(entry_time), MAX(COALESCE(exit_time, entry_time)) FROM positions").fetchone()
    first_time, last_time = row[0], row[1]
    if first_time:
        duration = (last_time or first_time) - first_time
        print(f"  Period: {fmt_time(first_time)} → {fmt_time(last_time)}  ({fmt_duration(duration)})")
    else:
        print(f"  Period: no trades yet")

    # ── Signals ───────────────────────────────────────────────
    total_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    acted_signals = conn.execute("SELECT COUNT(*) FROM signals WHERE acted_on = 1").fetchone()[0]
    blocked_signals = total_signals - acted_signals

    # Signal type breakdown
    sig_types = conn.execute(
        "SELECT signal_type, COUNT(*) FROM signals GROUP BY signal_type ORDER BY COUNT(*) DESC"
    ).fetchall()

    # Block reasons
    block_reasons = conn.execute(
        "SELECT reason_not_acted, COUNT(*) FROM signals WHERE acted_on = 0 AND reason_not_acted != '' "
        "GROUP BY reason_not_acted ORDER BY COUNT(*) DESC LIMIT 10"
    ).fetchall()

    print(f"\n  ── Signals ──")
    print(f"  Total:     {total_signals:,}")
    print(f"  Acted on:  {acted_signals:,}")
    print(f"  Blocked:   {blocked_signals:,}")
    if sig_types:
        print(f"  By type:")
        for st in sig_types:
            print(f"    {st[0]:<20} {st[1]:>6,}")
    if block_reasons:
        print(f"  Top block reasons:")
        for br in block_reasons:
            print(f"    {br[0]:<50} {br[1]:>5,}")

    # ── Positions ─────────────────────────────────────────────
    total_pos = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    open_pos = conn.execute("SELECT COUNT(*) FROM positions WHERE status = 'open'").fetchone()[0]
    closed_pos = conn.execute("SELECT COUNT(*) FROM positions WHERE status != 'open'").fetchone()[0]

    print(f"\n  ── Positions ──")
    print(f"  Total:    {total_pos}")
    print(f"  Open:     {open_pos}")
    print(f"  Closed:   {closed_pos}")

    if closed_pos == 0 and open_pos == 0:
        print(f"\n  No trades to analyze.")
        conn.close()
        return

    # ── P&L ───────────────────────────────────────────────────
    closed_rows = conn.execute(
        "SELECT * FROM positions WHERE status != 'open' ORDER BY exit_time ASC"
    ).fetchall()

    if closed_rows:
        pnls = [r['realized_pnl'] for r in closed_rows]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]
        flat = [p for p in pnls if p == 0]

        total_pnl = sum(pnls)
        gross_profit = sum(winners) if winners else 0
        gross_loss = sum(losers) if losers else 0
        avg_win = gross_profit / len(winners) if winners else 0
        avg_loss = gross_loss / len(losers) if losers else 0
        win_rate = len(winners) / len(pnls) * 100 if pnls else 0
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        max_win = max(pnls) if pnls else 0
        max_loss = min(pnls) if pnls else 0

        # Cumulative P&L for max drawdown
        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cum_pnl += p
            if cum_pnl > peak:
                peak = cum_pnl
            dd = peak - cum_pnl
            if dd > max_dd:
                max_dd = dd

        print(f"\n  ── Realized P&L ──")
        print(f"  Net P&L:        ${total_pnl:+,.2f}")
        print(f"  Gross profit:   ${gross_profit:+,.2f}  ({len(winners)} wins)")
        print(f"  Gross loss:     ${gross_loss:+,.2f}  ({len(losers)} losses)")
        if flat:
            print(f"  Flat:           {len(flat)} trades")
        print(f"  Win rate:       {win_rate:.1f}%")
        print(f"  Profit factor:  {profit_factor:.2f}")
        print(f"  Avg win:        ${avg_win:+,.2f}")
        print(f"  Avg loss:       ${avg_loss:+,.2f}")
        print(f"  Best trade:     ${max_win:+,.2f}")
        print(f"  Worst trade:    ${max_loss:+,.2f}")
        print(f"  Max drawdown:   ${max_dd:,.2f}")

        # ── Exit reasons ──────────────────────────────────────
        exit_reasons = conn.execute(
            """SELECT
                CASE
                    WHEN status = 'stopped_out' THEN 'stop_loss'
                    ELSE 'mean_reversion'
                END as reason,
                COUNT(*),
                SUM(realized_pnl),
                AVG(realized_pnl)
            FROM positions WHERE status != 'open'
            GROUP BY reason ORDER BY COUNT(*) DESC"""
        ).fetchall()

        # Also check execution_log for more detailed exit reasons
        # The reason is stored in the close_position call, but the DB status
        # only has 'closed' vs 'stopped_out'. Let's also look at durations.
        print(f"\n  ── By Exit Type ──")
        print(f"  {'Reason':<20} {'Count':>6} {'Total P&L':>12} {'Avg P&L':>10}")
        print(f"  {'-'*20} {'-'*6} {'-'*12} {'-'*10}")
        for er in exit_reasons:
            print(f"  {er[0]:<20} {er[1]:>6} ${er[2]:>+11,.2f} ${er[3]:>+9,.2f}")

    # ── Trade Durations ───────────────────────────────────────
    if closed_rows:
        durations = [(r['exit_time'] - r['entry_time']) for r in closed_rows if r['exit_time']]
        if durations:
            avg_dur = sum(durations) / len(durations)
            min_dur = min(durations)
            max_dur = max(durations)

            # Duration by outcome
            win_durs = [(r['exit_time'] - r['entry_time']) for r in closed_rows
                        if r['exit_time'] and r['realized_pnl'] > 0]
            loss_durs = [(r['exit_time'] - r['entry_time']) for r in closed_rows
                         if r['exit_time'] and r['realized_pnl'] < 0]

            print(f"\n  ── Trade Durations ──")
            print(f"  Average:    {fmt_duration(avg_dur)}")
            print(f"  Shortest:   {fmt_duration(min_dur)}")
            print(f"  Longest:    {fmt_duration(max_dur)}")
            if win_durs:
                print(f"  Avg win:    {fmt_duration(sum(win_durs) / len(win_durs))}")
            if loss_durs:
                print(f"  Avg loss:   {fmt_duration(sum(loss_durs) / len(loss_durs))}")

    # ── Per-Pair Breakdown ────────────────────────────────────
    pair_stats = conn.execute(
        """SELECT pair_key, direction,
                  COUNT(*) as trades,
                  SUM(realized_pnl) as total_pnl,
                  AVG(realized_pnl) as avg_pnl,
                  SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                  AVG(entry_value_a + entry_value_b) as avg_exposure
           FROM positions WHERE status != 'open'
           GROUP BY pair_key, direction
           ORDER BY total_pnl DESC"""
    ).fetchall()

    if pair_stats:
        print(f"\n  ── Per-Pair Breakdown ──")
        print(f"  {'Pair':<40} {'Dir':<6} {'#':>4} {'W/L':>6} {'Total P&L':>12} {'Avg P&L':>10} {'Avg Exp':>10}")
        print(f"  {'-'*40} {'-'*6} {'-'*4} {'-'*6} {'-'*12} {'-'*10} {'-'*10}")
        for ps in pair_stats:
            pair_short = ps[0][:18] + '..' + ps[0][-18:] if len(ps[0]) > 38 else ps[0]
            wl = f"{ps[5]}/{ps[2] - ps[5]}"
            print(f"  {pair_short:<40} {ps[1]:<6} {ps[2]:>4} {wl:>6} ${ps[3]:>+11,.2f} ${ps[4]:>+9,.2f} ${ps[6]:>9,.0f}")

    # ── Open Positions ────────────────────────────────────────
    open_rows = conn.execute(
        "SELECT * FROM positions WHERE status = 'open' ORDER BY entry_time ASC"
    ).fetchall()

    if open_rows:
        print(f"\n  ── Open Positions ──")
        print(f"  {'Pair':<40} {'Dir':<6} {'Entry Z':>8} {'Entry $':>10} {'Age':>8}")
        print(f"  {'-'*40} {'-'*6} {'-'*8} {'-'*10} {'-'*8}")
        import time as _time
        now = _time.time()
        for r in open_rows:
            pair_short = r['pair_key'][:18] + '..' + r['pair_key'][-18:] if len(r['pair_key']) > 38 else r['pair_key']
            exposure = r['entry_value_a'] + r['entry_value_b']
            age = now - r['entry_time']
            print(f"  {pair_short:<40} {r['direction']:<6} {r['entry_zscore']:>+8.3f} ${exposure:>9,.0f} {fmt_duration(age):>8}")

    # ── Portfolio Snapshots ───────────────────────────────────
    snapshots = conn.execute(
        "SELECT * FROM portfolio_snapshots ORDER BY timestamp ASC"
    ).fetchall()

    if snapshots:
        first_snap = snapshots[0]
        last_snap = snapshots[-1]
        peak_val = max(s['total_value'] for s in snapshots)
        trough_val = min(s['total_value'] for s in snapshots)
        max_dd_pct = max(s['drawdown_pct'] for s in snapshots)

        print(f"\n  ── Portfolio ──")
        print(f"  Starting value: ${first_snap['total_value']:,.2f}")
        print(f"  Current value:  ${last_snap['total_value']:,.2f}")
        print(f"  Peak value:     ${peak_val:,.2f}")
        print(f"  Trough value:   ${trough_val:,.2f}")
        print(f"  Max drawdown:   {max_dd_pct:.2%}")
        print(f"  Snapshots:      {len(snapshots)}")

    print(f"\n{'=' * 72}\n")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze statalyzer results")
    parser.add_argument("--local", metavar="DB_PATH", help="Path to local DB file")
    parser.add_argument("--host", default="frankfurt", help="SSH host (default: frankfurt)")
    parser.add_argument("--remote-path", default="~/statalyzer/statalyzer.db",
                        help="Remote DB path (default: ~/statalyzer/statalyzer.db)")
    args = parser.parse_args()

    if args.local:
        if not os.path.exists(args.local):
            print(f"Error: {args.local} not found")
            sys.exit(1)
        analyze(args.local)
    else:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp_path = f.name
        try:
            if not fetch_db(args.host, args.remote_path, tmp_path):
                sys.exit(1)
            analyze(tmp_path)
        finally:
            for suffix in ["", "-wal", "-shm"]:
                try:
                    os.unlink(tmp_path + suffix)
                except FileNotFoundError:
                    pass


if __name__ == "__main__":
    main()
