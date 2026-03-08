"""
SQLite persistence for statalayer.
Own DB for positions/signals + read-only access to scanner's cointegration results.
"""

import json
import sqlite3
import logging
import time
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: str = 'statalayer.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                slot INTEGER NOT NULL,
                pair_key TEXT NOT NULL,
                token_a_mint TEXT NOT NULL,
                token_b_mint TEXT NOT NULL,
                zscore REAL NOT NULL,
                signal_type TEXT NOT NULL,
                hedge_ratio REAL,
                spread REAL,
                spread_mean REAL,
                spread_std REAL,
                acted_on INTEGER NOT NULL DEFAULT 0,
                reason_not_acted TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair_key TEXT NOT NULL,
                token_a_mint TEXT NOT NULL,
                token_b_mint TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_time INTEGER NOT NULL,
                entry_slot INTEGER NOT NULL,
                entry_zscore REAL NOT NULL,
                entry_price_a REAL NOT NULL,
                entry_price_b REAL NOT NULL,
                hedge_ratio REAL NOT NULL,
                quantity_a REAL NOT NULL,
                quantity_b REAL NOT NULL,
                quantity_a_raw INTEGER NOT NULL,
                quantity_b_raw INTEGER NOT NULL,
                entry_value_a REAL NOT NULL,
                entry_value_b REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                exit_time INTEGER,
                exit_slot INTEGER,
                exit_zscore REAL,
                exit_price_a REAL,
                exit_price_b REAL,
                realized_pnl REAL DEFAULT 0.0,
                is_paper INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS execution_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id INTEGER,
                leg TEXT NOT NULL,
                side TEXT NOT NULL,
                token_mint TEXT NOT NULL,
                amount_raw INTEGER NOT NULL,
                price REAL NOT NULL,
                dex TEXT,
                pool_address TEXT,
                signature TEXT,
                slot INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                slippage_bps REAL,
                fee_lamports INTEGER DEFAULT 0,
                is_paper INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY (position_id) REFERENCES positions(id)
            );

            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                total_value REAL NOT NULL,
                num_open_positions INTEGER NOT NULL,
                total_exposure REAL NOT NULL,
                total_realized_pnl REAL NOT NULL,
                total_unrealized_pnl REAL NOT NULL,
                drawdown_pct REAL NOT NULL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS config_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS discovered_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair_key TEXT NOT NULL UNIQUE,
                token_a_mint TEXT NOT NULL,
                token_b_mint TEXT NOT NULL,
                token_a_symbol TEXT NOT NULL,
                token_b_symbol TEXT NOT NULL,
                hedge_ratio REAL NOT NULL,
                half_life REAL NOT NULL,
                eg_p_value REAL NOT NULL,
                eg_test_statistic REAL NOT NULL,
                spread_mean REAL NOT NULL,
                spread_std REAL NOT NULL,
                num_observations INTEGER NOT NULL,
                analyzed_at REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_signals_pair ON signals(pair_key);
            CREATE INDEX IF NOT EXISTS idx_signals_time ON signals(timestamp);
            CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
            CREATE INDEX IF NOT EXISTS idx_positions_pair ON positions(pair_key);
            CREATE INDEX IF NOT EXISTS idx_exec_position ON execution_log(position_id);
        """)
        self.conn.commit()

    # --- Signal methods ---

    def save_signal(self, signal, acted_on: bool, reason: str = '') -> int:
        cursor = self.conn.execute(
            """INSERT INTO signals
               (timestamp, slot, pair_key, token_a_mint, token_b_mint,
                zscore, signal_type, hedge_ratio, spread, spread_mean, spread_std,
                acted_on, reason_not_acted)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (signal.timestamp, signal.slot, signal.pair_key,
             signal.token_a_mint, signal.token_b_mint,
             signal.zscore, signal.signal_type.value,
             signal.hedge_ratio, signal.spread, signal.spread_mean, signal.spread_std,
             int(acted_on), reason),
        )
        self.conn.commit()
        return cursor.lastrowid

    # --- Position methods ---

    def save_position(self, position) -> int:
        cursor = self.conn.execute(
            """INSERT INTO positions
               (pair_key, token_a_mint, token_b_mint, direction,
                entry_time, entry_slot, entry_zscore,
                entry_price_a, entry_price_b, hedge_ratio,
                quantity_a, quantity_b, quantity_a_raw, quantity_b_raw,
                entry_value_a, entry_value_b, status, is_paper)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (position.pair_key, position.token_a_mint, position.token_b_mint,
             position.direction,
             position.entry_time, position.entry_slot, position.entry_zscore,
             position.entry_price_a, position.entry_price_b, position.hedge_ratio,
             position.quantity_a, position.quantity_b,
             position.quantity_a_raw, position.quantity_b_raw,
             position.entry_value_a, position.entry_value_b,
             position.status.value, int(position.is_paper)),
        )
        self.conn.commit()
        position.id = cursor.lastrowid
        return cursor.lastrowid

    def update_position(self, position) -> None:
        self.conn.execute(
            """UPDATE positions SET
               status = ?, exit_time = ?, exit_slot = ?, exit_zscore = ?,
               exit_price_a = ?, exit_price_b = ?, realized_pnl = ?,
               updated_at = CURRENT_TIMESTAMP
               WHERE id = ?""",
            (position.status.value, position.exit_time, position.exit_slot,
             position.exit_zscore, position.exit_price_a, position.exit_price_b,
             position.realized_pnl, position.id),
        )
        self.conn.commit()

    def get_open_positions(self) -> list:
        return self.conn.execute(
            "SELECT * FROM positions WHERE status = 'open' ORDER BY entry_time ASC"
        ).fetchall()

    def get_position_columns(self) -> list:
        cursor = self.conn.execute("PRAGMA table_info(positions)")
        return [row[1] for row in cursor.fetchall()]

    # --- Execution log ---

    def save_execution(self, position_id: int, leg: str, side: str,
                       token_mint: str, amount_raw: int, price: float,
                       dex: str, pool_address: str, signature: str,
                       slot: int, timestamp: int, slippage_bps: float,
                       fee_lamports: int, is_paper: bool) -> None:
        self.conn.execute(
            """INSERT INTO execution_log
               (position_id, leg, side, token_mint, amount_raw, price,
                dex, pool_address, signature, slot, timestamp,
                slippage_bps, fee_lamports, is_paper)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (position_id, leg, side, token_mint, amount_raw, price,
             dex, pool_address, signature, slot, timestamp,
             slippage_bps, fee_lamports, int(is_paper)),
        )
        self.conn.commit()

    # --- Portfolio snapshots ---

    def save_snapshot(self, total_value: float, num_positions: int,
                      total_exposure: float, realized_pnl: float,
                      unrealized_pnl: float, drawdown_pct: float) -> None:
        self.conn.execute(
            """INSERT INTO portfolio_snapshots
               (timestamp, total_value, num_open_positions, total_exposure,
                total_realized_pnl, total_unrealized_pnl, drawdown_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (int(time.time()), total_value, num_positions, total_exposure,
             realized_pnl, unrealized_pnl, drawdown_pct),
        )
        self.conn.commit()

    # --- Config state (crash recovery) ---

    def set_state(self, key: str, value: str) -> None:
        self.conn.execute(
            """INSERT INTO config_state (key, value) VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value,
               updated_at = CURRENT_TIMESTAMP""",
            (key, value),
        )
        self.conn.commit()

    def get_state(self, key: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT value FROM config_state WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    # --- Scanner DB reader (read-only) ---

    @staticmethod
    def read_cointegrated_pairs(scanner_db_path: str) -> List[Dict]:
        """Read cointegrated pairs from scanner's DB (read-only)."""
        try:
            conn = sqlite3.connect(f"file:{scanner_db_path}?mode=ro", uri=True)
        except sqlite3.OperationalError:
            logger.error(f"Cannot open scanner DB: {scanner_db_path}")
            return []

        try:
            rows = conn.execute("""
                SELECT token_a_mint, token_b_mint, token_a_symbol, token_b_symbol,
                       hedge_ratio, spread_mean, spread_std, half_life,
                       eg_p_value, eg_is_cointegrated, johansen_is_cointegrated,
                       num_observations, analyzed_at
                FROM cointegration_results
                WHERE id IN (
                    SELECT MAX(id) FROM cointegration_results
                    GROUP BY token_a_mint, token_b_mint
                )
                AND (eg_is_cointegrated = 1 OR johansen_is_cointegrated = 1)
                ORDER BY eg_p_value ASC
            """).fetchall()

            pairs = []
            for row in rows:
                pairs.append({
                    'token_a_mint': row[0],
                    'token_b_mint': row[1],
                    'token_a_symbol': row[2],
                    'token_b_symbol': row[3],
                    'hedge_ratio': row[4],
                    'spread_mean': row[5],
                    'spread_std': row[6],
                    'half_life': row[7],
                    'eg_p_value': row[8],
                    'eg_is_cointegrated': bool(row[9]),
                    'johansen_is_cointegrated': bool(row[10]),
                    'num_observations': row[11],
                    'analyzed_at': row[12],
                })
            return pairs
        except Exception as e:
            logger.error(f"Error reading scanner DB: {e}")
            return []
        finally:
            conn.close()

    # --- Discovered pairs persistence ---

    def save_discovered_pair(self, result) -> None:
        """UPSERT a discovered cointegration result."""
        from signals import make_pair_key
        pair_key = make_pair_key(result.token_a_mint, result.token_b_mint)
        self.conn.execute("""
            INSERT INTO discovered_pairs
                (pair_key, token_a_mint, token_b_mint, token_a_symbol, token_b_symbol,
                 hedge_ratio, half_life, eg_p_value, eg_test_statistic,
                 spread_mean, spread_std, num_observations, analyzed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pair_key) DO UPDATE SET
                hedge_ratio=excluded.hedge_ratio, half_life=excluded.half_life,
                eg_p_value=excluded.eg_p_value, eg_test_statistic=excluded.eg_test_statistic,
                spread_mean=excluded.spread_mean, spread_std=excluded.spread_std,
                num_observations=excluded.num_observations, analyzed_at=excluded.analyzed_at
        """, (pair_key, result.token_a_mint, result.token_b_mint,
              result.token_a_symbol, result.token_b_symbol,
              result.hedge_ratio, result.half_life, result.eg_p_value,
              result.eg_test_statistic, result.spread_mean, result.spread_std,
              result.num_observations, result.analyzed_at))
        self.conn.commit()

    def load_discovered_pairs(self) -> List[Dict]:
        """Load all discovered pairs from DB (for crash recovery)."""
        rows = self.conn.execute("""
            SELECT pair_key, token_a_mint, token_b_mint, token_a_symbol, token_b_symbol,
                   hedge_ratio, half_life, eg_p_value, eg_test_statistic,
                   spread_mean, spread_std, num_observations, analyzed_at
            FROM discovered_pairs
        """).fetchall()
        pairs = []
        for row in rows:
            pairs.append({
                'pair_key': row[0],
                'token_a_mint': row[1],
                'token_b_mint': row[2],
                'token_a_symbol': row[3],
                'token_b_symbol': row[4],
                'hedge_ratio': row[5],
                'half_life': row[6],
                'eg_p_value': row[7],
                'eg_test_statistic': row[8],
                'spread_mean': row[9],
                'spread_std': row[10],
                'num_observations': row[11],
                'analyzed_at': row[12],
            })
        return pairs

    def remove_discovered_pair(self, pair_key: str) -> None:
        self.conn.execute("DELETE FROM discovered_pairs WHERE pair_key = ?", (pair_key,))
        self.conn.commit()

    def get_stats(self) -> dict:
        total = self.conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        open_pos = self.conn.execute("SELECT COUNT(*) FROM positions WHERE status = 'open'").fetchone()[0]
        closed = self.conn.execute("SELECT COUNT(*) FROM positions WHERE status != 'open'").fetchone()[0]
        total_pnl = self.conn.execute("SELECT COALESCE(SUM(realized_pnl), 0) FROM positions WHERE status != 'open'").fetchone()[0]
        signals = self.conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        return {
            'total_positions': total,
            'open_positions': open_pos,
            'closed_positions': closed,
            'total_realized_pnl': total_pnl,
            'total_signals': signals,
        }

    def close(self):
        self.conn.close()
