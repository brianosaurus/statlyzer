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

    def __init__(self, config, portfolio):
        self.config = config
        self.portfolio = portfolio
        self.entries_this_hour: list = []  # timestamps
        self.kill_switch: bool = False

    def check_entry(self, signal, pair_state) -> RiskCheck:
        """Run all risk checks before allowing a new entry."""
        # 1. Kill switch
        if self.kill_switch:
            return RiskCheck(False, "Kill switch engaged")

        # 2. Already in this pair
        if self.portfolio.has_position(signal.pair_key):
            return RiskCheck(False, f"Already in {signal.pair_key}")

        # 3. Max positions
        num_open = len(self.portfolio.positions)
        if num_open >= self.config.max_positions:
            return RiskCheck(False, f"Max positions ({self.config.max_positions}) reached")

        # 4. Max exposure
        exposure = self.portfolio.get_total_exposure()
        if exposure >= self.config.max_total_exposure_usd:
            return RiskCheck(False, f"Max exposure (${self.config.max_total_exposure_usd:.0f}) reached")

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
        max_hl = self.config.lookback_window * self.config.max_half_life_ratio
        if hl > max_hl and hl != float('inf'):
            return RiskCheck(False, f"Half-life too long ({hl:.1f} > {max_hl:.1f})")

        # 8. Rate limit
        self._prune_entries()
        if len(self.entries_this_hour) >= self.config.max_positions_per_hour:
            return RiskCheck(False, f"Rate limit ({self.config.max_positions_per_hour}/hr) reached")

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
