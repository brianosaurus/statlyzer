"""
Signal generator for statalayer.
Monitors cointegrated pairs in real-time via gRPC stream,
computes rolling z-scores, and emits entry/exit signals.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

import base58
import numpy as np

from constants import WELL_KNOWN_TOKENS, QUOTE_PRIORITY
from swap_detector import SwapDetector

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

    def __init__(self, config, scanner_db_path: str):
        self.config = config
        self.scanner_db_path = scanner_db_path
        self.pairs: Dict[str, PairState] = {}
        self.token_prices: Dict[str, float] = {}
        self.monitored_mints: Set[str] = set()
        self.swap_detector = SwapDetector()
        self.last_pair_reload = 0
        self.pair_reload_interval = 900  # 15 minutes

    def load_pairs(self) -> int:
        """Load cointegrated pairs from scanner DB."""
        from db import Database

        pairs_data = Database.read_cointegrated_pairs(self.scanner_db_path)
        if not pairs_data:
            logger.warning("No cointegrated pairs found in scanner DB")
            return 0

        loaded = 0
        for p in pairs_data:
            # Half-life filter
            hl = p.get('half_life', float('inf'))
            if hl < self.config.min_half_life:
                continue
            if hl > self.config.lookback_window * self.config.max_half_life_ratio:
                if hl != float('inf'):
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

            pair_key = make_pair_key(p['token_a_mint'], p['token_b_mint'])

            if pair_key in self.pairs:
                # Update existing pair's params
                self.pairs[pair_key].hedge_ratio = p['hedge_ratio']
                self.pairs[pair_key].half_life = hl
                self.pairs[pair_key].cointegration_analyzed_at = analyzed_at
            else:
                state = PairState(
                    pair_key=pair_key,
                    token_a_mint=p['token_a_mint'],
                    token_b_mint=p['token_b_mint'],
                    token_a_symbol=p.get('token_a_symbol', token_symbol(p['token_a_mint'])),
                    token_b_symbol=p.get('token_b_symbol', token_symbol(p['token_b_mint'])),
                    hedge_ratio=p['hedge_ratio'],
                    half_life=hl,
                    eg_p_value=p.get('eg_p_value', 1.0),
                    cointegration_analyzed_at=analyzed_at,
                )
                state.init_buffers(self.config.lookback_window)
                self.pairs[pair_key] = state

            self.monitored_mints.add(p['token_a_mint'])
            self.monitored_mints.add(p['token_b_mint'])
            loaded += 1

        self.last_pair_reload = time.time()
        logger.info(f"Loaded {loaded} cointegrated pairs, monitoring {len(self.monitored_mints)} tokens")
        return loaded

    def process_block(self, slot: int, block, block_time: int = 0) -> List[Signal]:
        """Process a block and return any triggered signals."""
        # Periodic pair reload
        if time.time() - self.last_pair_reload > self.pair_reload_interval:
            self.load_pairs()

        if not self.pairs:
            return []

        # Extract prices from swaps in this block
        new_prices = self._extract_prices_from_block(slot, block)
        if not new_prices:
            return []

        # Update token prices
        self.token_prices.update(new_prices)

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

            signal = self._update_pair(pair, price_a, price_b, slot, block_time)
            if signal is not None:
                signals.append(signal)

        return signals

    def _extract_prices_from_block(self, slot: int, block) -> Dict[str, float]:
        """Extract median implied price per token from all swaps in a block."""
        if not hasattr(block, 'transactions'):
            return {}

        # Collect all observed prices per token
        token_obs: Dict[str, List[float]] = {}

        for tx in block.transactions:
            if not tx or not hasattr(tx, 'transaction'):
                continue
            if hasattr(tx, 'meta') and tx.meta and tx.meta.err:
                continue

            try:
                swaps = self.swap_detector.analyze_transaction(tx)
            except Exception:
                continue

            if not swaps:
                continue

            for swap in swaps:
                vaults = swap.get('vault_balance_changes', {})
                if len(vaults) < 2:
                    continue

                # Extract the two tokens involved
                vault_list = list(vaults.values())
                changes = []
                for v in vault_list:
                    mint = v.get('mint')
                    change = v.get('balance_change', 0)
                    decimals = v.get('decimals', 0)
                    if mint and change != 0 and decimals >= 0:
                        changes.append({
                            'mint': mint,
                            'change': change,
                            'decimals': decimals,
                        })

                if len(changes) < 2:
                    continue

                # Find the token pair (one increases, one decreases)
                token_in = None
                token_out = None
                for c in changes:
                    if c['change'] < 0:
                        token_in = c  # tokens leaving pool = user bought
                    elif c['change'] > 0:
                        token_out = c  # tokens entering pool = user sold

                if not token_in or not token_out:
                    continue

                # Only price against quote tokens
                quote_mint = self._pick_quote(token_in['mint'], token_out['mint'])
                if quote_mint is None:
                    continue

                amt_in = abs(token_in['change']) / (10 ** token_in['decimals'])
                amt_out = abs(token_out['change']) / (10 ** token_out['decimals'])

                if amt_in <= 0 or amt_out <= 0:
                    continue

                # Determine base token and its price in quote
                if quote_mint == token_out['mint']:
                    base_mint = token_in['mint']
                    price = amt_out / amt_in
                else:
                    base_mint = token_out['mint']
                    price = amt_in / amt_out

                if price <= 0 or not np.isfinite(price):
                    continue

                # Only track tokens we're monitoring
                if base_mint in self.monitored_mints:
                    token_obs.setdefault(base_mint, []).append(price)

        # Take median price per token
        result = {}
        for mint, prices in token_obs.items():
            result[mint] = float(np.median(prices))

        return result

    def _pick_quote(self, token_a: str, token_b: str) -> Optional[str]:
        for quote in QUOTE_PRIORITY:
            if token_a == quote:
                return token_a
            if token_b == quote:
                return token_b
        return None

    def _update_pair(self, pair: PairState, price_a: float, price_b: float,
                     slot: int, block_time: int) -> Optional[Signal]:
        """Update pair buffers and check for signal."""
        log_a = np.log(price_a)
        log_b = np.log(price_b)

        pair.prices_a.append(log_a)
        pair.prices_b.append(log_b)
        pair.last_update_slot = slot

        # Need minimum observations
        min_obs = max(10, self.config.lookback_window // 2)
        if pair.prices_a.count < min_obs:
            return None

        # Compute spread
        arr_a = pair.prices_a.get_array()
        arr_b = pair.prices_b.get_array()
        spread = arr_a - pair.hedge_ratio * arr_b

        pair.spread_mean = float(np.mean(spread))
        pair.spread_std = float(np.std(spread))
        pair.current_spread = float(spread[-1])

        if pair.spread_std <= 0:
            pair.current_zscore = 0.0
            return None

        pair.current_zscore = (pair.current_spread - pair.spread_mean) / pair.spread_std

        # Check thresholds
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

    def _check_signal(self, pair: PairState) -> Optional[SignalType]:
        z = pair.current_zscore

        if abs(z) > self.config.stop_loss_zscore:
            return SignalType.STOP_LOSS
        elif z < -self.config.entry_zscore:
            return SignalType.ENTRY_LONG
        elif z > self.config.entry_zscore:
            return SignalType.ENTRY_SHORT
        elif abs(z) < self.config.exit_zscore:
            return SignalType.EXIT
        return None

    def get_pair_states(self) -> Dict[str, PairState]:
        return self.pairs
