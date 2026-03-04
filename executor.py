"""
Execution layer for statalyzer.
Paper mode: simulates fills with slippage model.
Live mode: stub for future Jupiter/DEX integration.
"""

import logging
import random
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Fill:
    token_mint: str
    side: str           # "buy" or "sell"
    price: float        # fill price after slippage
    quantity: float     # human-readable amount
    quantity_raw: int   # raw lamports/units
    slippage_bps: float # actual slippage applied
    timestamp: int
    tx_signature: Optional[str] = None  # None for paper


@dataclass
class PairExecution:
    fill_a: Fill
    fill_b: Fill
    is_paper: bool = True


class PaperExecutor:
    """Simulates trade execution with a random slippage model."""

    def __init__(self, config, db):
        self.config = config
        self.db = db

    def execute_entry(self, signal, position_size, price_a: float, price_b: float) -> PairExecution:
        """Simulate entry fills for both legs."""
        from signals import SignalType

        now = int(time.time())

        # For ENTRY_LONG (z < -threshold): buy A, sell B
        # For ENTRY_SHORT (z > +threshold): sell A, buy B
        if signal.signal_type == SignalType.ENTRY_LONG:
            side_a, side_b = "buy", "sell"
        else:
            side_a, side_b = "sell", "buy"

        slip_a = self._random_slippage()
        slip_b = self._random_slippage()

        # Apply slippage: buying costs more, selling gets less
        fill_price_a = price_a * (1 + slip_a / 10000) if side_a == "buy" else price_a * (1 - slip_a / 10000)
        fill_price_b = price_b * (1 + slip_b / 10000) if side_b == "buy" else price_b * (1 - slip_b / 10000)

        fill_a = Fill(
            token_mint=signal.token_a_mint,
            side=side_a,
            price=fill_price_a,
            quantity=position_size.token_a_amount,
            quantity_raw=position_size.token_a_raw,
            slippage_bps=slip_a,
            timestamp=now,
        )
        fill_b = Fill(
            token_mint=signal.token_b_mint,
            side=side_b,
            price=fill_price_b,
            quantity=position_size.token_b_amount,
            quantity_raw=position_size.token_b_raw,
            slippage_bps=slip_b,
            timestamp=now,
        )

        return PairExecution(fill_a=fill_a, fill_b=fill_b, is_paper=True)

    def execute_exit(self, position, price_a: float, price_b: float) -> PairExecution:
        """Simulate exit fills for both legs (reverse of entry)."""
        now = int(time.time())

        # Reverse of entry: long exit = sell A, buy B
        if position.direction == "long":
            side_a, side_b = "sell", "buy"
        else:
            side_a, side_b = "buy", "sell"

        slip_a = self._random_slippage()
        slip_b = self._random_slippage()

        fill_price_a = price_a * (1 + slip_a / 10000) if side_a == "buy" else price_a * (1 - slip_a / 10000)
        fill_price_b = price_b * (1 + slip_b / 10000) if side_b == "buy" else price_b * (1 - slip_b / 10000)

        fill_a = Fill(
            token_mint=position.token_a_mint,
            side=side_a,
            price=fill_price_a,
            quantity=position.quantity_a,
            quantity_raw=position.quantity_a_raw,
            slippage_bps=slip_a,
            timestamp=now,
        )
        fill_b = Fill(
            token_mint=position.token_b_mint,
            side=side_b,
            price=fill_price_b,
            quantity=position.quantity_b,
            quantity_raw=position.quantity_b_raw,
            slippage_bps=slip_b,
            timestamp=now,
        )

        return PairExecution(fill_a=fill_a, fill_b=fill_b, is_paper=True)

    def log_execution(self, position_id: int, execution: PairExecution, slot: int):
        """Persist both fills to the execution_log table."""
        for leg, fill in [("A", execution.fill_a), ("B", execution.fill_b)]:
            self.db.save_execution(
                position_id=position_id,
                leg=leg,
                side=fill.side,
                token_mint=fill.token_mint,
                amount_raw=fill.quantity_raw,
                price=fill.price,
                dex="paper",
                pool_address="",
                signature=fill.tx_signature or "",
                slot=slot,
                timestamp=fill.timestamp,
                slippage_bps=fill.slippage_bps,
                fee_lamports=0,
                is_paper=True,
            )

    def _random_slippage(self) -> float:
        """Random slippage in bps, normal distribution centered at 0."""
        return abs(random.gauss(0, self.config.slippage_bps / 2))


class LiveExecutor:
    """Placeholder for live execution via Jupiter or direct DEX swaps."""

    def __init__(self, config, db):
        self.config = config
        self.db = db

    def execute_entry(self, signal, position_size, price_a: float, price_b: float) -> PairExecution:
        raise NotImplementedError(
            "Live execution not yet implemented. "
            "Future: Jupiter aggregator integration for optimal routing."
        )

    def execute_exit(self, position, price_a: float, price_b: float) -> PairExecution:
        raise NotImplementedError("Live execution not yet implemented.")

    def log_execution(self, position_id: int, execution: PairExecution, slot: int):
        raise NotImplementedError("Live execution not yet implemented.")
