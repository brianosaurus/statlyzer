"""
Dynamic slippage monitoring via Jupiter quote API.

Periodically measures real slippage for all known tokens at multiple trade
sizes, then computes the maximum profitable trade size per token/basket.
"""

import json
import logging
import math
import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

SOL_MINT = "So11111111111111111111111111111111111111112"

# Trade sizes to probe (USD per leg)
PROBE_SIZES = [50, 250, 1000, 5000]

# Number of measurements to average over (EMA-style)
SMOOTHING_ALPHA = 0.3  # weight of new measurement (0.3 = 30% new, 70% old)


@dataclass
class TokenSlippage:
    """Slippage curve for one token across multiple trade sizes."""
    mint: str
    symbol: str
    # List of (size_usd, round_trip_bps) sorted by size — smoothed EMA
    curve: List[Tuple[float, float]] = field(default_factory=list)
    measured_at: float = 0.0
    max_profitable_size: float = 0.0  # largest size where trade is profitable


class SlippageMonitor:
    """Background thread that measures Jupiter slippage at multiple sizes
    and computes max profitable trade size per token."""

    DEFAULT_EDGE_BPS = 10.6
    DEFAULT_FIXED_FEES_USD = 0.13

    def __init__(self, config, tokens: Dict[str, dict],
                 stablecoin_mints: Set[str],
                 poll_interval: float = 600,
                 edge_bps: float = DEFAULT_EDGE_BPS,
                 fixed_fees_usd: float = DEFAULT_FIXED_FEES_USD):
        self.config = config
        self.tokens = tokens
        self.stablecoin_mints = stablecoin_mints
        self.poll_interval = poll_interval
        self.edge_bps = edge_bps
        self.fixed_fees_usd = fixed_fees_usd

        self.jupiter_api_key = os.environ.get("JUPITER_API_KEY", "")
        self.jupiter_url = os.environ.get(
            "JUPITER_QUOTE_URL", "https://api.jup.ag"
        ).rstrip("/")

        self._lock = threading.Lock()
        self._token_slippage: Dict[str, TokenSlippage] = {}
        self._sol_price: float = 0.0
        self._last_poll: float = 0.0

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("Slippage monitor started (poll every %.0fs, %d probe sizes)"
                     % (self.poll_interval, len(PROBE_SIZES)))

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _poll_loop(self):
        # Random initial delay (0-120s) to stagger across experiments
        import random
        jitter = random.uniform(0, 120)
        logger.info("Slippage monitor: initial delay %.0fs to avoid rate limits" % jitter)
        self._stop.wait(jitter)
        if self._stop.is_set():
            return
        self._measure_all()
        while not self._stop.is_set():
            self._stop.wait(self.poll_interval)
            if not self._stop.is_set():
                self._measure_all()

    def _measure_all(self):
        try:
            mints = [m for m in self.tokens if m not in self.stablecoin_mints and m != SOL_MINT]
            prices = self._get_prices(mints + [SOL_MINT])
            sol_price = prices.get(SOL_MINT, 87.0)

            for mint in mints:
                info = self.tokens.get(mint, {})
                symbol = info.get("symbol", mint[:8])
                decimals = info.get("decimals", 6)
                mid_price = prices.get(mint)
                if not mid_price or mid_price <= 0:
                    continue

                new_curve = []
                for size_usd in PROBE_SIZES:
                    buy_bps = self._measure_direction(
                        SOL_MINT, mint, size_usd, sol_price, mid_price,
                        9, decimals, "buy"
                    )
                    sell_bps = self._measure_direction(
                        mint, SOL_MINT, size_usd, sol_price, mid_price,
                        decimals, 9, "sell"
                    )
                    if buy_bps is not None and sell_bps is not None:
                        rt = abs(buy_bps) + abs(sell_bps)
                        new_curve.append((size_usd, rt))
                    time.sleep(0.5)

                if new_curve:
                    # Blend with previous measurement via EMA
                    with self._lock:
                        prev = self._token_slippage.get(mint)
                    if prev and prev.curve:
                        prev_dict = {s: rt for s, rt in prev.curve}
                        smoothed = []
                        for size_usd, new_rt in new_curve:
                            old_rt = prev_dict.get(size_usd)
                            if old_rt is not None:
                                blended = SMOOTHING_ALPHA * new_rt + (1 - SMOOTHING_ALPHA) * old_rt
                            else:
                                blended = new_rt
                            smoothed.append((size_usd, blended))
                        curve = smoothed
                    else:
                        curve = new_curve

                    max_size = self._compute_max_profitable_size(curve)
                    ts = TokenSlippage(
                        mint=mint, symbol=symbol, curve=curve,
                        measured_at=time.time(),
                        max_profitable_size=max_size,
                    )
                    with self._lock:
                        self._token_slippage[mint] = ts

                time.sleep(0.5)

            with self._lock:
                self._sol_price = sol_price
                self._last_poll = time.time()

            # Log summary
            with self._lock:
                all_ts = self._token_slippage
            tradeable = [(ts.symbol, ts.max_profitable_size)
                         for ts in all_ts.values() if ts.max_profitable_size > 0]
            tradeable.sort(key=lambda x: -x[1])
            logger.info(
                "Slippage scan: %d/%d tokens tradeable | %s" % (
                    len(tradeable), len(all_ts),
                    ", ".join("%s($%.0f)" % (s, sz) for s, sz in tradeable)
                )
            )

        except Exception as e:
            logger.warning("Slippage measurement failed: %s" % e)

    def _compute_max_profitable_size(self, curve: List[Tuple[float, float]]) -> float:
        """Find the largest trade size where the trade is still profitable.

        profit = size × (edge_bps - slippage_bps) / 10000 - fixed_fees > 0
        """
        max_size = 0.0
        for size_usd, rt_bps in curve:
            net_edge_bps = self.edge_bps - rt_bps
            if net_edge_bps <= 0:
                continue
            profit = size_usd * net_edge_bps / 10000 - self.fixed_fees_usd
            if profit > 0:
                max_size = size_usd

        # Interpolate between probed sizes if the last profitable size
        # isn't the largest probe
        if max_size > 0 and max_size < PROBE_SIZES[-1]:
            idx = next(i for i, (s, _) in enumerate(curve) if s == max_size)
            if idx + 1 < len(curve):
                s1, rt1 = curve[idx]
                s2, rt2 = curve[idx + 1]
                # Linear interpolation: find size where profit = 0
                # profit(s) = s * (edge - lerp(rt1,rt2,t)) / 10000 - fees = 0
                # Approximate: use rt at the midpoint
                mid_rt = (rt1 + rt2) / 2
                net = self.edge_bps - mid_rt
                if net > 0:
                    breakeven = self.fixed_fees_usd / (net / 10000)
                    if s1 < breakeven < s2:
                        max_size = breakeven

        return max_size

    def _measure_direction(self, input_mint, output_mint, size_usd,
                           sol_price, token_price,
                           in_decimals, out_decimals, direction):
        try:
            if direction == "buy":
                amount_raw = int(size_usd / sol_price * (10 ** in_decimals))
            else:
                amount_raw = int(size_usd / token_price * (10 ** in_decimals))

            quote = self._get_quote(input_mint, output_mint, amount_raw)
            if not quote:
                return None

            out_raw = int(quote.get("outAmount", 0))
            actual_out = out_raw / (10 ** out_decimals)
            if actual_out <= 0:
                return None

            if direction == "buy":
                quote_price = size_usd / actual_out
                return (quote_price - token_price) / token_price * 10000
            else:
                usd_received = actual_out * sol_price
                quote_price = usd_received / (amount_raw / (10 ** in_decimals))
                return (token_price - quote_price) / token_price * 10000

        except Exception:
            return None

    def _get_prices(self, mints):
        ids = ",".join(mints)
        url = "https://api.jup.ag/price/v3?ids=%s" % ids
        headers = {"Accept": "application/json", "User-Agent": "statalyzer/1.0"}
        if self.jupiter_api_key:
            headers["x-api-key"] = self.jupiter_api_key
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        prices = {}
        for mint in mints:
            entry = data.get(mint)
            if entry and entry.get("usdPrice"):
                prices[mint] = float(entry["usdPrice"])
        return prices

    def _get_quote(self, input_mint, output_mint, amount_raw):
        params = (
            "?inputMint=%s&outputMint=%s&amount=%s&slippageBps=1000"
            % (input_mint, output_mint, amount_raw)
        )
        url = "%s/swap/v1/quote%s" % (self.jupiter_url, params)
        headers = {"Accept": "application/json"}
        if self.jupiter_api_key:
            headers["x-api-key"] = self.jupiter_api_key
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(5 * (attempt + 1))  # backoff: 5s, 10s, 15s
                    continue
                return None
            except Exception:
                return None
        return None

    # ── Public API ──────────────────────────────────────────────

    def get_basket_max_size(self, mints: List[str]) -> float:
        """Return the max profitable trade size for a basket.
        Limited by the worst (smallest max_size) token in the basket."""
        with self._lock:
            if not self._token_slippage:
                return 0.0
            max_sizes = []
            for mint in mints:
                if mint in self.stablecoin_mints or mint == SOL_MINT:
                    continue
                ts = self._token_slippage.get(mint)
                if ts is None:
                    return 0.0  # unknown token = can't trade
                if ts.max_profitable_size <= 0:
                    return 0.0  # token is never profitable
                max_sizes.append(ts.max_profitable_size)
            return min(max_sizes) if max_sizes else 0.0

    def get_slippage_at_size(self, mint: str, size_usd: float) -> Optional[float]:
        """Interpolate round-trip slippage for a token at a given trade size."""
        with self._lock:
            ts = self._token_slippage.get(mint)
            if not ts or not ts.curve:
                return None
            # Find bracketing probe sizes
            for i, (s, rt) in enumerate(ts.curve):
                if size_usd <= s:
                    if i == 0:
                        return rt
                    s0, rt0 = ts.curve[i - 1]
                    t = (size_usd - s0) / (s - s0)
                    return rt0 + t * (rt - rt0)
            return ts.curve[-1][1]  # extrapolate last

    def is_basket_tradeable(self, mints: List[str]) -> bool:
        """Check if a basket can be traded at any profitable size."""
        return self.get_basket_max_size(mints) > 0

    def get_tradeable_mints(self) -> Set[str]:
        with self._lock:
            return {mint for mint, ts in self._token_slippage.items()
                    if ts.max_profitable_size > 0}

    def status_str(self) -> str:
        with self._lock:
            if not self._token_slippage:
                return "Slippage: measuring..."

            tradeable = sorted(
                [(ts.symbol, ts.max_profitable_size)
                 for ts in self._token_slippage.values()
                 if ts.max_profitable_size > 0],
                key=lambda x: -x[1],
            )
            n_total = len(self._token_slippage)
            tokens_str = ", ".join(
                "%s($%s)" % (s, ("%.0f" % sz) if sz < 10000 else "10k+")
                for s, sz in tradeable
            )
            return "Slippage: %d/%d tradeable [%s]" % (
                len(tradeable), n_total, tokens_str
            )
