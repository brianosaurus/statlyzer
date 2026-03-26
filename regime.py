"""
Regime detection for statalyzer.
Monitors market conditions and adjusts trading aggressiveness.

Regime score is an EMA of a composite signal:
  0.4 * variance_ratio + 0.3 * stop_rate + 0.2 * win_rate + 0.1 * vol_expansion

Regimes:
  "normal"  — score < caution_threshold  (full size, normal z thresholds)
  "caution" — score < danger_threshold   (half size, tighter entry z)
  "danger"  — score >= danger_threshold  (no new entries)
"""

import logging
import time
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Detects market regime from variance ratios, stop-loss frequency,
    win rate, and spread volatility expansion."""

    def __init__(self, config):
        self.config = config
        self.caution_threshold = config.regime_caution_threshold
        self.danger_threshold = config.regime_danger_threshold
        self.ema_alpha = config.regime_ema_alpha
        self.caution_size_mult = getattr(config, 'regime_caution_size_mult', 0.5)
        self.caution_entry_z_mult = getattr(config, 'regime_caution_entry_z_mult', 1.3)

        self.regime_score = 0.0
        self._stop_loss_times: deque = deque(maxlen=100)  # timestamps of recent stop-outs
        self._prev_spread_stds: dict = {}  # pair_key -> previous spread_std

        # Component signals (for display)
        self._vr_signal = 0.0
        self._stop_rate_signal = 0.0
        self._win_rate_signal = 0.0
        self._vol_expansion_signal = 0.0

    def update(self, basket_states: dict, portfolio):
        """Recompute regime score from current market state."""
        vr_signal = self._compute_variance_ratio_signal(basket_states)
        stop_signal = self._compute_stop_rate_signal()
        win_signal = self._compute_win_rate_signal(portfolio)
        vol_signal = self._compute_vol_expansion_signal(basket_states)

        self._vr_signal = vr_signal
        self._stop_rate_signal = stop_signal
        self._win_rate_signal = win_signal
        self._vol_expansion_signal = vol_signal

        composite = (0.4 * vr_signal +
                     0.3 * stop_signal +
                     0.2 * win_signal +
                     0.1 * vol_signal)

        # EMA update
        self.regime_score = (self.ema_alpha * composite +
                             (1.0 - self.ema_alpha) * self.regime_score)

    def get_regime(self) -> str:
        if self.regime_score >= self.danger_threshold:
            return "danger"
        if self.regime_score >= self.caution_threshold:
            return "caution"
        return "normal"

    def get_size_multiplier(self) -> float:
        regime = self.get_regime()
        if regime == "danger":
            return 0.0
        if regime == "caution":
            return self.caution_size_mult
        return 1.0

    def get_entry_z_multiplier(self) -> float:
        """Returns multiplier for entry z threshold.
        In caution mode, signals need higher z to pass (threshold is multiplied up)."""
        if self.get_regime() == "caution":
            return self.caution_entry_z_mult
        return 1.0

    def on_stop_loss(self):
        """Record a stop-loss event."""
        self._stop_loss_times.append(time.time())

    def status_str(self) -> str:
        regime = self.get_regime()
        return (f"Regime: {regime} (score={self.regime_score:.3f} | "
                f"VR={self._vr_signal:.2f} stop={self._stop_rate_signal:.2f} "
                f"win={self._win_rate_signal:.2f} vol={self._vol_expansion_signal:.2f})")

    # ------------------------------------------------------------------ #
    # Component signal computations (each returns 0.0 - 1.0)
    # ------------------------------------------------------------------ #

    def _compute_variance_ratio_signal(self, basket_states: dict) -> float:
        """Median Variance Ratio across all baskets with enough data.
        VR > 1 indicates trending (bad for mean reversion).
        Returns 0.0 (mean-reverting) to 1.0 (trending)."""
        vr_values = []
        lag = 5

        for bsk in basket_states.values():
            if bsk.price_buffers is None:
                continue
            if bsk.price_buffers[0].count < 25:
                continue

            # Build spread from price buffers and hedge ratios
            arrays = [buf.get_array() for buf in bsk.price_buffers]
            if len(arrays[0]) < 25:
                continue
            price_matrix = np.column_stack(arrays)
            hr = np.array(bsk.hedge_ratios)
            spread = price_matrix @ hr

            vr = self._variance_ratio(spread, lag)
            if vr is not None:
                vr_values.append(vr)

        if not vr_values:
            return 0.0

        median_vr = float(np.median(vr_values))
        # Map VR: 0.5 (strong MR) -> 0.0, 1.0 (random walk) -> 0.5, 1.5+ (trending) -> 1.0
        signal = max(0.0, min(1.0, (median_vr - 0.5) / 1.0))
        return signal

    @staticmethod
    def _variance_ratio(spread: np.ndarray, k: int = 5) -> float:
        """Compute Variance Ratio at lag k.
        VR = Var(spread[t] - spread[t-k]) / (k * Var(spread[t] - spread[t-1]))"""
        if len(spread) < k + 2:
            return None
        diff1 = spread[1:] - spread[:-1]
        diffk = spread[k:] - spread[:-k]
        var1 = np.var(diff1)
        vark = np.var(diffk)
        if var1 < 1e-20:
            return None
        return float(vark / (k * var1))

    def _compute_stop_rate_signal(self) -> float:
        """Fraction of recent stop-losses in the last hour.
        More stops = more trending / adverse conditions."""
        now = time.time()
        cutoff = now - 3600
        recent = sum(1 for t in self._stop_loss_times if t > cutoff)
        # 0 stops -> 0.0, 5+ stops -> 1.0
        return min(1.0, recent / 5.0)

    def _compute_win_rate_signal(self, portfolio) -> float:
        """Signal from recent win rate (last 20 closed positions).
        Low win rate -> high signal (bad conditions)."""
        closed = getattr(portfolio, 'closed_positions', [])
        recent = closed[-20:] if len(closed) >= 5 else closed
        if not recent:
            return 0.0
        wins = sum(1 for p in recent if p.realized_pnl > 0)
        win_rate = wins / len(recent)
        # win_rate 0.6+ -> 0.0 (healthy), 0.3- -> 1.0 (bad)
        signal = max(0.0, min(1.0, (0.6 - win_rate) / 0.3))
        return signal

    def _compute_vol_expansion_signal(self, basket_states: dict) -> float:
        """Detect spread volatility expansion across baskets.
        Compares current spread_std to previous measurement."""
        ratios = []
        new_stds = {}

        for pair_key, bsk in basket_states.items():
            if bsk.spread_std <= 0:
                continue
            prev = self._prev_spread_stds.get(pair_key)
            new_stds[pair_key] = bsk.spread_std
            if prev is not None and prev > 0:
                ratios.append(bsk.spread_std / prev)

        self._prev_spread_stds = new_stds

        if not ratios:
            return 0.0

        median_ratio = float(np.median(ratios))
        # ratio 1.0 (stable) -> 0.0, 1.5+ (expanding) -> 1.0
        signal = max(0.0, min(1.0, (median_ratio - 1.0) / 0.5))
        return signal
