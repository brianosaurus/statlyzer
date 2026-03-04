"""
Position sizing for statalayer.
Computes dollar amounts and token quantities for pair trades.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from constants import WELL_KNOWN_TOKENS
from signal import Signal

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    token_a_amount: float       # Human-readable
    token_b_amount: float
    token_a_raw: int            # Raw lamports/units
    token_b_raw: int
    dollar_amount_a: float
    dollar_amount_b: float
    total_exposure_usd: float


def get_decimals(mint: str) -> int:
    info = WELL_KNOWN_TOKENS.get(mint)
    return info['decimals'] if info else 9


class PositionSizer:
    """Computes position sizes for pair trades."""

    def __init__(self, config):
        self.config = config

    def compute_size(self, signal: Signal, portfolio_value: float,
                     current_exposure: float,
                     price_a: float, price_b: float) -> Optional[PositionSize]:
        """
        Compute position size for a pair entry.

        For ENTRY_LONG (z < -threshold): buy A, sell B
        For ENTRY_SHORT (z > +threshold): sell A, buy B

        Returns None if position would exceed limits.
        """
        if price_a <= 0 or price_b <= 0:
            return None

        # Compute dollar size
        if self.config.sizing_method == 'kelly':
            dollar_size = self._kelly_size(
                portfolio_value, signal.zscore,
                signal.spread_std, signal.hedge_ratio,
            )
        else:
            dollar_size = self._fixed_fraction_size(portfolio_value)

        if dollar_size <= 0:
            return None

        # Check against max position
        dollar_size = min(dollar_size, self.config.max_position_usd)

        # Check against remaining exposure budget
        remaining_budget = self.config.max_total_exposure_usd - current_exposure
        if remaining_budget <= 0:
            return None
        dollar_size = min(dollar_size, remaining_budget / 2)  # /2 because it's two legs

        # Compute token quantities
        # Leg A gets dollar_size, leg B gets dollar_size * |hedge_ratio|
        hedge = abs(signal.hedge_ratio) if signal.hedge_ratio != 0 else 1.0
        dollar_a = dollar_size
        dollar_b = dollar_size * hedge

        amount_a = dollar_a / price_a
        amount_b = dollar_b / price_b

        decimals_a = get_decimals(signal.token_a_mint)
        decimals_b = get_decimals(signal.token_b_mint)

        raw_a = int(amount_a * (10 ** decimals_a))
        raw_b = int(amount_b * (10 ** decimals_b))

        if raw_a <= 0 or raw_b <= 0:
            return None

        return PositionSize(
            token_a_amount=amount_a,
            token_b_amount=amount_b,
            token_a_raw=raw_a,
            token_b_raw=raw_b,
            dollar_amount_a=dollar_a,
            dollar_amount_b=dollar_b,
            total_exposure_usd=dollar_a + dollar_b,
        )

    def _fixed_fraction_size(self, portfolio_value: float) -> float:
        return portfolio_value * self.config.fixed_fraction

    def _kelly_size(self, portfolio_value: float, zscore: float,
                    spread_std: float, hedge_ratio: float) -> float:
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
