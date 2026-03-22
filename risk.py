"""
Risk controls for statalayer.
All checks must pass before a new position is opened.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RiskCheck:
    allowed: bool
    reason: str = ""


class RiskManager:
    """Evaluates risk before allowing new positions."""

    def __init__(self, config, portfolio, slippage_monitor=None):
        self.config = config
        self.portfolio = portfolio
        self.slippage_monitor = slippage_monitor
        self.entries_this_hour: list = []  # timestamps
        self.kill_switch: bool = False
        self.kill_switch_time: float = 0  # when kill switch was engaged
        self.kill_switch_cooldown: float = 600  # 10 minutes

    def check_entry(self, signal, pair_state) -> RiskCheck:
        """Run all risk checks before allowing a new entry."""
        # 1. Kill switch (auto-resets after cooldown if drawdown recovered)
        if self.kill_switch:
            elapsed = time.time() - self.kill_switch_time
            if elapsed > self.kill_switch_cooldown:
                dd = self.portfolio.get_drawdown()
                if dd <= self.config.max_drawdown_pct:
                    logger.info(f"Kill switch auto-reset after {elapsed:.0f}s cooldown "
                                f"(drawdown recovered to {dd:.1%})")
                    self.kill_switch = False
                else:
                    return RiskCheck(False, f"Kill switch engaged (dd={dd:.1%}, cooldown {elapsed:.0f}s)")
            else:
                return RiskCheck(False, "Kill switch engaged")

        # 1b. Entry z-score cap — don't enter too close to stop loss
        abs_z = abs(signal.zscore)
        if abs_z > self.config.max_entry_zscore:
            return RiskCheck(False, f"|z|={abs_z:.1f} > max entry {self.config.max_entry_zscore}")

        # 2. Already in this pair
        if self.portfolio.has_position(signal.pair_key):
            return RiskCheck(False, f"Already in {signal.pair_key}")

        # 3. Max positions
        num_open = len(self.portfolio.positions)
        if num_open >= self.config.max_positions:
            return RiskCheck(False, f"Max positions ({self.config.max_positions}) reached")

        # 3b. Concentration limit — max positions sharing a token
        # Per-token limits based on liquidity tier (deep liquidity = higher limit)
        from constants import STABLECOIN_MINTS, TOKEN_LIQUIDITY_TIER
        base_max = self.config.max_positions_per_token
        for mint in signal.mints:
            if mint in STABLECOIN_MINTS:
                continue
            tier_mult = TOKEN_LIQUIDITY_TIER.get(mint, 1.0)
            max_for_token = int(base_max * tier_mult)
            count = sum(
                1 for p in self.portfolio.positions.values()
                if mint in p.mints
            )
            if count >= max_for_token:
                return RiskCheck(False, f"Concentration limit ({max_for_token}) for {mint[:8]}.. reached")

        # 3b2. Slippage check — reject baskets where no profitable size exists
        if self.slippage_monitor is not None:
            max_size = self.slippage_monitor.get_basket_max_size(signal.mints)
            if max_size <= 0:
                return RiskCheck(False, "No profitable trade size for basket")

        # 3c. Rate limit — max entries per hour
        self._prune_entries()
        if len(self.entries_this_hour) >= self.config.max_positions_per_hour:
            return RiskCheck(False, f"Rate limit ({self.config.max_positions_per_hour}/hr) reached")

        # 4. Max exposure (scales with portfolio capital)
        exposure = self.portfolio.get_total_exposure()
        max_exposure = self.portfolio.initial_capital * self.config.max_exposure_ratio
        if exposure >= max_exposure:
            return RiskCheck(False, f"Max exposure (${max_exposure:.0f}) reached")

        # 5. Drawdown check
        if self.check_drawdown():
            return RiskCheck(False, f"Max drawdown ({self.config.max_drawdown_pct:.0%}) exceeded — kill switch engaged")

        # 6. Pair staleness
        analyzed_at = pair_state.cointegration_analyzed_at
        if analyzed_at > 0:
            age_hours = (time.time() - analyzed_at) / 3600
            if age_hours > self.config.pair_staleness_hours:
                return RiskCheck(False, f"Pair stale ({age_hours:.0f}h > {self.config.pair_staleness_hours}h)")

        # 7. Half-life filter
        hl = pair_state.half_life
        if hl < self.config.min_half_life:
            return RiskCheck(False, f"Half-life too short ({hl:.1f} < {self.config.min_half_life})")
        # lookback_window is in candles; convert to blocks for half-life comparison
        lookback_blocks = self.config.lookback_window * self.config.signal_resample_secs / 0.4
        max_hl = lookback_blocks * self.config.max_half_life_ratio
        if hl > max_hl and hl != float('inf'):
            return RiskCheck(False, f"Half-life too long ({hl:.1f} > {max_hl:.1f})")
        # Absolute max half-life in seconds (blocks * 0.4s/block)
        hl_secs = hl * 0.4
        if hl_secs > self.config.max_half_life_secs:
            return RiskCheck(False, f"Half-life {hl_secs:.0f}s > {self.config.max_half_life_secs:.0f}s max")

        return RiskCheck(True)

    def check_exit(self, signal) -> RiskCheck:
        """Check if we should exit a position."""
        if not self.portfolio.has_position(signal.pair_key):
            return RiskCheck(False, "No position to exit")
        return RiskCheck(True)

    def check_drawdown(self) -> bool:
        """If drawdown exceeds threshold, engage kill switch. Returns True if triggered."""
        dd = self.portfolio.get_drawdown()
        if dd > self.config.max_drawdown_pct:
            if not self.kill_switch:
                logger.warning(f"KILL SWITCH engaged: drawdown {dd:.1%} > {self.config.max_drawdown_pct:.1%}")
                self.kill_switch = True
                self.kill_switch_time = time.time()
            return True
        return False

    def record_entry(self):
        self.entries_this_hour.append(time.time())

    def _prune_entries(self):
        cutoff = time.time() - 3600
        self.entries_this_hour = [t for t in self.entries_this_hour if t > cutoff]

    def reset_kill_switch(self):
        logger.info("Kill switch manually reset")
        self.kill_switch = False
