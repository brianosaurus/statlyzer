"""
Position sizing for statalayer.
Computes dollar amounts and token quantities for basket trades (2-4 tokens).
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from constants import WELL_KNOWN_TOKENS
from signals import Signal

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    amounts: List[float]        # Human-readable amount per token
    amounts_raw: List[int]      # Raw lamports/units per token
    dollar_amounts: List[float] # USD value per leg
    total_exposure_usd: float


def get_decimals(mint: str) -> int:
    info = WELL_KNOWN_TOKENS.get(mint)
    return info['decimals'] if info else 9


class PositionSizer:
    """Computes position sizes for basket trades."""

    def __init__(self, config):
        self.config = config

    def compute_size(self, signal: Signal, portfolio_value: float,
                     current_exposure: float,
                     prices: List[float],
                     regime_multiplier: float = 1.0) -> Optional[PositionSize]:
        """
        Compute position size for a basket entry.

        Leg 0 (the reference leg) is sized by fixed fraction or Kelly,
        then scaled by conviction (z-score magnitude).
        Other legs are sized proportionally by |hedge_ratios[i] / hedge_ratios[0]|.

        Returns None if position would exceed limits.
        """
        if any(p <= 0 for p in prices):
            return None

        # Compute dollar size for the reference leg (index 0)
        if self.config.sizing_method == 'kelly':
            dollar_size = self._kelly_size(
                portfolio_value, signal.zscore,
                signal.spread_std, signal.hedge_ratios,
            )
        else:
            dollar_size = self._fixed_fraction_size(portfolio_value)

        # Conviction scaling: scale up for higher |z| beyond entry threshold
        # At entry_zscore: 1.0x, at 2x entry_zscore: 1.5x (linear ramp, capped at 1.5x)
        abs_z = abs(signal.zscore)
        entry_z = self.config.entry_zscore
        if abs_z > entry_z and entry_z > 0:
            conviction = 1.0 + 0.5 * min((abs_z - entry_z) / entry_z, 1.0)
            dollar_size *= conviction

        # Apply regime-based size scaling
        dollar_size *= regime_multiplier

        if dollar_size <= 0:
            return None

        # Compute relative weights from hedge ratios
        hr = signal.hedge_ratios
        abs_hr0 = abs(hr[0]) if hr[0] != 0 else 1.0
        weights = [abs(h) / abs_hr0 for h in hr]  # weight[0] = 1.0

        total_weight = sum(weights)

        # Cap total exposure at max_position_usd
        max_ref_leg = self.config.max_position_usd / total_weight
        dollar_size = min(dollar_size, max_ref_leg)

        # Check against remaining exposure budget
        max_exposure = portfolio_value * self.config.max_exposure_ratio
        remaining_budget = max_exposure - current_exposure
        if remaining_budget <= 0:
            return None
        dollar_size = min(dollar_size, remaining_budget / total_weight)

        # Compute per-leg dollar amounts and token quantities
        dollar_amounts = [dollar_size * w for w in weights]
        amounts = [d / p for d, p in zip(dollar_amounts, prices)]
        amounts_raw = []
        for i, mint in enumerate(signal.mints):
            decimals = get_decimals(mint)
            raw = int(amounts[i] * (10 ** decimals))
            amounts_raw.append(raw)

        if any(r <= 0 for r in amounts_raw):
            return None

        # Guard against SQLite INTEGER overflow (max 2^63-1)
        max_raw = 2**63 - 1
        if any(r > max_raw for r in amounts_raw):
            label = "/".join(signal.symbols)
            logger.warning(f"Raw quantity overflow for {label}, skipping")
            return None

        return PositionSize(
            amounts=amounts,
            amounts_raw=amounts_raw,
            dollar_amounts=dollar_amounts,
            total_exposure_usd=sum(dollar_amounts),
        )

    def _fixed_fraction_size(self, portfolio_value: float) -> float:
        return portfolio_value * self.config.fixed_fraction

    def _kelly_size(self, portfolio_value: float, zscore: float,
                    spread_std: float, hedge_ratios: list) -> float:
        """
        Kelly criterion sizing.

        Edge: expected profit from mean reversion = |z| * spread_std * decay
        Variance: spread_std^2
        Kelly fraction = edge / variance, capped at 2x fixed_fraction
        """
        if spread_std <= 0:
            return self._fixed_fraction_size(portfolio_value)

        z = abs(zscore)
        edge = z * spread_std * 0.5  # conservative decay estimate
        variance = spread_std ** 2

        if variance <= 0:
            return self._fixed_fraction_size(portfolio_value)

        kelly_frac = edge / variance
        max_frac = self.config.fixed_fraction * 2.0
        kelly_frac = min(kelly_frac, max_frac)
        kelly_frac = max(kelly_frac, 0.0)

        return portfolio_value * kelly_frac
