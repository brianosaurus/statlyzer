"""
Signal generator for statalayer.
Monitors cointegrated pairs, computes rolling z-scores from Jupiter prices,
and emits entry/exit signals.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

import numpy as np

from constants import WELL_KNOWN_TOKENS, STABLECOIN_MINTS, SOL_MINT

logger = logging.getLogger(__name__)


class SignalType(Enum):
    ENTRY_LONG = "entry_long"
    ENTRY_SHORT = "entry_short"
    EXIT = "exit"
    STOP_LOSS = "stop_loss"


@dataclass
class Signal:
    signal_type: SignalType
    pair_key: str
    token_a_mint: str
    token_b_mint: str
    token_a_symbol: str
    token_b_symbol: str
    zscore: float
    hedge_ratio: float
    spread: float
    spread_mean: float
    spread_std: float
    timestamp: int
    slot: int


class CircularBuffer:
    """Fixed-size numpy circular buffer for O(1) append."""

    def __init__(self, capacity: int):
        self.data = np.empty(capacity, dtype=np.float64)
        self.capacity = capacity
        self.write_idx = 0
        self.count = 0

    def append(self, value: float):
        self.data[self.write_idx] = value
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def get_array(self) -> np.ndarray:
        if self.count == 0:
            return np.array([], dtype=np.float64)
        if self.count < self.capacity:
            return self.data[:self.count].copy()
        return np.concatenate([
            self.data[self.write_idx:],
            self.data[:self.write_idx]
        ])

    @property
    def latest(self) -> float:
        if self.count == 0:
            return 0.0
        idx = (self.write_idx - 1) % self.capacity
        return self.data[idx]


@dataclass
class PairState:
    """Tracks rolling price window and cointegration params for one pair."""
    pair_key: str
    token_a_mint: str
    token_b_mint: str
    token_a_symbol: str
    token_b_symbol: str
    hedge_ratio: float
    half_life: float
    eg_p_value: float

    # Rolling price buffers
    prices_a: CircularBuffer = field(default=None, repr=False)
    prices_b: CircularBuffer = field(default=None, repr=False)

    # Cached z-score
    current_zscore: float = 0.0
    current_spread: float = 0.0
    spread_mean: float = 0.0
    spread_std: float = 0.0
    last_update_slot: int = 0
    cointegration_analyzed_at: int = 0
    in_position: bool = False
    position_entry_time: float = 0.0

    # Resampling state: accumulate latest price, flush to buffer at fixed intervals
    pending_price_a: float = 0.0
    pending_price_b: float = 0.0
    last_resample_time: float = 0.0

    def init_buffers(self, capacity: int):
        self.prices_a = CircularBuffer(capacity)
        self.prices_b = CircularBuffer(capacity)


def make_pair_key(mint_a: str, mint_b: str) -> str:
    """Canonical pair key — lexicographic order to avoid duplicates."""
    return f"{min(mint_a, mint_b)}:{max(mint_a, mint_b)}"


def token_symbol(mint: str) -> str:
    info = WELL_KNOWN_TOKENS.get(mint)
    return info['symbol'] if info else mint[:8] + '..'


class SignalGenerator:
    """Monitors cointegrated pairs and generates trading signals."""

    def __init__(self, config, scanner_db_path: str = None, db=None):
        self.config = config
        self.scanner_db_path = scanner_db_path
        self.db = db  # For candle persistence
        self.pairs: Dict[str, PairState] = {}
        self.token_prices: Dict[str, float] = {}  # USD-normalized prices
        self.monitored_mints: Set[str] = set()
        self.last_pair_reload = 0
        self.pair_reload_interval = 900  # 15 minutes
        self.sol_usd_price: float = 0.0  # cached SOL/USD
        self._candle_save_count = 0  # batch commit counter

        # Inline cointegration discovery
        from cointegration import CointegrationDiscovery
        self.discovery = CointegrationDiscovery(config, WELL_KNOWN_TOKENS, STABLECOIN_MINTS)

    def load_pairs(self) -> int:
        """Load cointegrated pairs from scanner DB."""
        if not self.scanner_db_path:
            return 0

        from db import Database

        pairs_data = Database.read_cointegrated_pairs(self.scanner_db_path)
        if not pairs_data:
            logger.warning("No cointegrated pairs found in scanner DB")
            return 0

        # Track which base tokens already have a stablecoin pair (dedup USDC/USDT/USDH)
        base_with_stable: Dict[str, str] = {}  # base_mint -> preferred quote mint

        loaded = 0
        skipped_dup = 0
        skipped_unknown = 0
        for p in pairs_data:
            # Half-life filter
            # Scanner half-lives are in 5-min resampling periods; convert to blocks
            # 1 period = 300s, 1 block ≈ 0.4s → 1 period ≈ 750 blocks
            hl_periods = p.get('half_life', float('inf'))
            if hl_periods == float('inf') or hl_periods <= 0:
                continue  # No mean reversion — skip
            hl = hl_periods * 750  # Convert to blocks
            if hl < self.config.min_half_life:
                continue
            # lookback_window is in candles; convert to blocks for half-life comparison
            lookback_blocks = self.config.lookback_window * self.config.signal_resample_secs / 0.4
            if hl > lookback_blocks * self.config.max_half_life_ratio:
                continue

            # Staleness filter
            analyzed_at = p.get('analyzed_at', 0)
            if isinstance(analyzed_at, str):
                try:
                    analyzed_at = int(analyzed_at)
                except ValueError:
                    analyzed_at = 0
            age_hours = (time.time() - analyzed_at) / 3600 if analyzed_at > 0 else float('inf')
            if age_hours > self.config.pair_staleness_hours:
                continue

            mint_a = p['token_a_mint']
            mint_b = p['token_b_mint']

            # Well-known token filter: require at least ONE token to be known
            if mint_a not in WELL_KNOWN_TOKENS and mint_b not in WELL_KNOWN_TOKENS:
                skipped_unknown += 1
                continue

            # Skip stablecoin pairs entirely — USDC/TOKEN is not real stat-arb,
            # it's just single-directional token trading with no mean-reverting spread
            stable_mints = STABLECOIN_MINTS
            if mint_a in stable_mints or mint_b in stable_mints:
                skipped_dup += 1
                continue

            pair_key = make_pair_key(mint_a, mint_b)

            if pair_key in self.pairs:
                # Update existing pair's params
                self.pairs[pair_key].hedge_ratio = p['hedge_ratio']
                self.pairs[pair_key].half_life = hl
                self.pairs[pair_key].cointegration_analyzed_at = analyzed_at
            else:
                state = PairState(
                    pair_key=pair_key,
                    token_a_mint=mint_a,
                    token_b_mint=mint_b,
                    token_a_symbol=p.get('token_a_symbol', token_symbol(mint_a)),
                    token_b_symbol=p.get('token_b_symbol', token_symbol(mint_b)),
                    hedge_ratio=p['hedge_ratio'],
                    half_life=hl,
                    eg_p_value=p.get('eg_p_value', 1.0),
                    cointegration_analyzed_at=analyzed_at,
                )
                state.init_buffers(self.config.lookback_window)
                self._restore_candles(state)
                self.pairs[pair_key] = state

            self.monitored_mints.add(mint_a)
            self.monitored_mints.add(mint_b)
            loaded += 1

        if skipped_dup or skipped_unknown:
            logger.info(f"Skipped {skipped_dup} duplicate stablecoin pairs, "
                        f"{skipped_unknown} pairs with no well-known token")

        self.last_pair_reload = time.time()
        logger.info(f"Loaded {loaded} cointegrated pairs, monitoring {len(self.monitored_mints)} tokens")
        return loaded

    def load_discovered_pairs(self, results) -> int:
        """Load pairs discovered by inline cointegration analysis."""
        loaded = 0
        for r in results:
            if not r.eg_is_cointegrated:
                continue
            # Skip stablecoin pairs — not real stat-arb
            if r.token_a_mint in STABLECOIN_MINTS or r.token_b_mint in STABLECOIN_MINTS:
                continue
            # Convert half-life from resampled periods to blocks
            hl_blocks = r.half_life * (self.config.coint_resample_secs / 0.4)
            if hl_blocks < self.config.min_half_life:
                continue
            lookback_blocks = self.config.lookback_window * self.config.signal_resample_secs / 0.4
            max_hl = lookback_blocks * self.config.max_half_life_ratio
            if hl_blocks > max_hl:
                continue

            pair_key = make_pair_key(r.token_a_mint, r.token_b_mint)
            if pair_key in self.pairs:
                # Update existing pair's params
                self.pairs[pair_key].hedge_ratio = r.hedge_ratio
                self.pairs[pair_key].half_life = hl_blocks
                self.pairs[pair_key].cointegration_analyzed_at = int(r.analyzed_at)
            else:
                state = PairState(
                    pair_key=pair_key,
                    token_a_mint=r.token_a_mint,
                    token_b_mint=r.token_b_mint,
                    token_a_symbol=r.token_a_symbol,
                    token_b_symbol=r.token_b_symbol,
                    hedge_ratio=r.hedge_ratio,
                    half_life=hl_blocks,
                    eg_p_value=r.eg_p_value,
                    cointegration_analyzed_at=int(r.analyzed_at),
                )
                state.init_buffers(self.config.lookback_window)
                self._restore_candles(state)
                self.pairs[pair_key] = state
                logger.info(f"Discovered pair: {r.token_a_symbol}/{r.token_b_symbol} "
                            f"p={r.eg_p_value:.4f} hl={hl_blocks:.0f}blk hr={r.hedge_ratio:.4f}")

            self.monitored_mints.add(r.token_a_mint)
            self.monitored_mints.add(r.token_b_mint)
            loaded += 1
        return loaded

    def process_prices(self, new_prices: Dict[str, float], timestamp: float) -> List[Signal]:
        """Process a batch of Jupiter prices and return any triggered signals."""
        # Periodic pair reload from scanner DB
        if self.scanner_db_path and time.time() - self.last_pair_reload > self.pair_reload_interval:
            self.load_pairs()

        if not new_prices:
            self.discovery.update_prices(self.token_prices, timestamp)
            new_pairs = self.discovery.maybe_run_scan()
            if new_pairs:
                self.load_discovered_pairs(new_pairs)
            return []

        # Update SOL/USD
        sol_price = new_prices.get(SOL_MINT)
        if sol_price and sol_price > 0:
            self.sol_usd_price = sol_price

        # Update token prices (Jupiter prices are clean, light outlier rejection)
        for mint, new_price in new_prices.items():
            old_price = self.token_prices.get(mint)
            if old_price and old_price > 0:
                change_pct = abs(new_price - old_price) / old_price
                if change_pct > 1.0:  # >100% jump: reject
                    logger.debug(f"Price outlier REJECTED for {mint[:8]}.. "
                                 f"${old_price:.6f} -> ${new_price:.6f} ({change_pct:.0%})")
                    continue
            self.token_prices[mint] = new_price

        # Feed prices to cointegration discovery
        self.discovery.update_prices(self.token_prices, timestamp)
        new_pairs = self.discovery.maybe_run_scan()
        if new_pairs:
            added = self.load_discovered_pairs(new_pairs)
            if added > 0:
                logger.info(f"Discovery added {added} pairs, now monitoring {len(self.pairs)} total")

        # Update each monitored pair and check for signals
        signals = []
        for pair in self.pairs.values():
            price_a = new_prices.get(pair.token_a_mint)
            price_b = new_prices.get(pair.token_b_mint)

            if price_a is None and price_b is None:
                continue

            # Use latest known price if only one updated
            if price_a is None:
                price_a = self.token_prices.get(pair.token_a_mint)
            if price_b is None:
                price_b = self.token_prices.get(pair.token_b_mint)

            if price_a is None or price_b is None:
                continue
            if price_a <= 0 or price_b <= 0:
                continue

            signal = self._update_pair(pair, price_a, price_b, 0, timestamp)
            if signal is not None:
                signals.append(signal)

        return signals

    def all_mints(self) -> Set[str]:
        """Return all mints that need pricing (monitored pairs + discovery tokens)."""
        mints = set(self.monitored_mints)
        # Include well-known tokens for cointegration discovery
        for mint in WELL_KNOWN_TOKENS:
            if mint not in STABLECOIN_MINTS:
                mints.add(mint)
        mints.add(SOL_MINT)
        return mints

    def _restore_candles(self, pair: PairState):
        """Restore saved candles from DB into the pair's circular buffers."""
        if not self.db:
            return
        candles = self.db.load_candles(pair.pair_key, self.config.lookback_window)
        if not candles:
            return
        for ts, log_a, log_b in candles:
            pair.prices_a.append(log_a)
            pair.prices_b.append(log_b)
        # Set last_resample_time so the next candle flushes at the right time
        pair.last_resample_time = candles[-1][0]
        logger.info(f"Restored {len(candles)} candles for {pair.pair_key[:8]}..{pair.pair_key[-6:]} "
                    f"(count={pair.prices_a.count})")

    def _update_pair(self, pair: PairState, price_a: float, price_b: float,
                     slot: int, block_time: int) -> Optional[Signal]:
        """Update pair with resampled candle-close prices.

        Instead of appending every raw swap price (irregular intervals), we:
        1. Cache the latest price as "pending"
        2. Every signal_resample_secs, flush the pending price to the buffer
        3. Compute z-score only from the resampled buffer

        This matches the scanner's 5-min resampling timescale and eliminates
        noise from irregular swap arrival rates.
        """
        now = block_time or time.time()

        # Always update pending (latest known price for this pair)
        pair.pending_price_a = price_a
        pair.pending_price_b = price_b
        pair.last_update_slot = slot

        # Initialize resample timer on first observation
        if pair.last_resample_time == 0:
            pair.last_resample_time = now
            return None

        # Check if it's time to flush a new candle
        elapsed = now - pair.last_resample_time
        if elapsed < self.config.signal_resample_secs:
            # Not time yet — but still compute live z-score for display
            # using buffered history + current pending price
            if pair.prices_a.count >= 2:
                self._compute_zscore_live(pair)
            return None

        # Time to resample: append the latest price as a candle close
        log_a = np.log(pair.pending_price_a)
        log_b = np.log(pair.pending_price_b)
        pair.prices_a.append(log_a)
        pair.prices_b.append(log_b)
        pair.last_resample_time = now

        # Persist candle to DB for warmup avoidance on restart
        if self.db:
            self.db.save_candle(pair.pair_key, now, float(log_a), float(log_b))

        # Need minimum observations for meaningful statistics
        if pair.prices_a.count < 30:
            return None

        # Compute spread and z-score from resampled buffer
        arr_a = pair.prices_a.get_array()
        arr_b = pair.prices_b.get_array()
        spread = arr_a - pair.hedge_ratio * arr_b

        pair.current_spread = float(spread[-1])
        pair.spread_mean = float(np.mean(spread))
        pair.spread_std = float(np.std(spread))

        if pair.spread_std < 1e-12:
            pair.current_zscore = 0.0
            return None

        pair.current_zscore = (pair.current_spread - pair.spread_mean) / pair.spread_std

        # Guard against degenerate z-scores from near-zero std
        if abs(pair.current_zscore) > 100:
            pair.current_zscore = 0.0
            return None

        # Check thresholds (only at resample boundaries)
        signal_type = self._check_signal(pair)
        if signal_type is None:
            return None

        return Signal(
            signal_type=signal_type,
            pair_key=pair.pair_key,
            token_a_mint=pair.token_a_mint,
            token_b_mint=pair.token_b_mint,
            token_a_symbol=pair.token_a_symbol,
            token_b_symbol=pair.token_b_symbol,
            zscore=pair.current_zscore,
            hedge_ratio=pair.hedge_ratio,
            spread=pair.current_spread,
            spread_mean=pair.spread_mean,
            spread_std=pair.spread_std,
            timestamp=block_time or int(time.time()),
            slot=slot,
        )

    def _compute_zscore_live(self, pair: PairState):
        """Compute a live z-score for display (using buffer + pending price).
        Does NOT trigger signals — just updates the dashboard display."""
        if pair.pending_price_a <= 0 or pair.pending_price_b <= 0:
            return
        arr_a = pair.prices_a.get_array()
        arr_b = pair.prices_b.get_array()
        if len(arr_a) < 2:
            return
        spread = arr_a - pair.hedge_ratio * arr_b
        mean = float(np.mean(spread))
        std = float(np.std(spread))
        if std < 1e-12:
            return
        # Current spread using pending (live) prices
        live_spread = np.log(pair.pending_price_a) - pair.hedge_ratio * np.log(pair.pending_price_b)
        z = (live_spread - mean) / std
        if abs(z) > 100:
            return  # degenerate — don't display garbage
        pair.current_zscore = z
        pair.current_spread = float(live_spread)

    def _check_signal(self, pair: PairState) -> Optional[SignalType]:
        z = pair.current_zscore

        if abs(z) > self.config.stop_loss_zscore:
            if pair.in_position:
                return SignalType.STOP_LOSS
            # z beyond stop-loss but no position — don't enter (too extreme)
            return None
        elif z < -self.config.entry_zscore:
            if not pair.in_position:
                return SignalType.ENTRY_LONG
        elif z > self.config.entry_zscore:
            if not pair.in_position:
                return SignalType.ENTRY_SHORT
        elif abs(z) < self.config.exit_zscore:
            if pair.in_position:
                # Enforce cooldown: don't exit too soon after entry
                # entry_cooldown_slots / 2.5 blk/s ≈ seconds
                cooldown_secs = self.config.entry_cooldown_slots / 2.5
                if (time.time() - pair.position_entry_time) >= cooldown_secs:
                    return SignalType.EXIT
        return None

    def get_pair_states(self) -> Dict[str, PairState]:
        return self.pairs
