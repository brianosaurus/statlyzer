"""
Portfolio manager for statalayer.
Tracks open positions, marks-to-market, computes P&L, persists state.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    STOPPED_OUT = "stopped_out"


@dataclass
class Position:
    pair_key: str
    token_a_mint: str
    token_b_mint: str
    direction: str  # "long" or "short" (refers to the spread)

    # Entry state
    entry_time: int
    entry_slot: int
    entry_zscore: float
    entry_price_a: float
    entry_price_b: float
    hedge_ratio: float

    # Quantities (always positive, direction encodes long/short)
    quantity_a: float
    quantity_b: float
    quantity_a_raw: int
    quantity_b_raw: int

    # Dollar values at entry
    entry_value_a: float
    entry_value_b: float

    # Current state
    current_price_a: float = 0.0
    current_price_b: float = 0.0
    current_zscore: float = 0.0
    unrealized_pnl: float = 0.0

    # Exit state
    exit_time: Optional[int] = None
    exit_slot: Optional[int] = None
    exit_zscore: Optional[float] = None
    exit_price_a: Optional[float] = None
    exit_price_b: Optional[float] = None
    realized_pnl: float = 0.0

    status: PositionStatus = PositionStatus.OPEN
    is_paper: bool = True
    id: Optional[int] = None


class PortfolioManager:
    """Tracks positions, P&L, and portfolio state."""

    def __init__(self, config, db):
        self.config = config
        self.db = db
        self.positions: Dict[str, Position] = {}  # pair_key -> Position
        self.closed_positions: List[Position] = []
        self.initial_capital: float = config.initial_capital
        self.total_realized_pnl: float = 0.0
        self.peak_value: float = config.initial_capital
        self._load_state()

    def _load_state(self):
        """Load state from DB for crash recovery."""
        # Load initial capital and peak from config_state
        saved_capital = self.db.get_state('initial_capital')
        if saved_capital:
            self.initial_capital = float(saved_capital)

        saved_peak = self.db.get_state('peak_value')
        if saved_peak:
            self.peak_value = float(saved_peak)

        saved_pnl = self.db.get_state('total_realized_pnl')
        if saved_pnl:
            self.total_realized_pnl = float(saved_pnl)

        # Load open positions
        rows = self.db.get_open_positions()
        columns = self.db.get_position_columns()

        for row in rows:
            data = dict(zip(columns, row))
            position = Position(
                pair_key=data['pair_key'],
                token_a_mint=data['token_a_mint'],
                token_b_mint=data['token_b_mint'],
                direction=data['direction'],
                entry_time=data['entry_time'],
                entry_slot=data['entry_slot'],
                entry_zscore=data['entry_zscore'],
                entry_price_a=data['entry_price_a'],
                entry_price_b=data['entry_price_b'],
                hedge_ratio=data['hedge_ratio'],
                quantity_a=data['quantity_a'],
                quantity_b=data['quantity_b'],
                quantity_a_raw=data['quantity_a_raw'],
                quantity_b_raw=data['quantity_b_raw'],
                entry_value_a=data['entry_value_a'],
                entry_value_b=data['entry_value_b'],
                status=PositionStatus.OPEN,
                is_paper=bool(data['is_paper']),
                id=data['id'],
            )
            self.positions[position.pair_key] = position

        if self.positions:
            logger.info(f"Recovered {len(self.positions)} open positions from DB")

    def open_position(self, signal, size, is_paper: bool = True,
                      price_a: float = 0.0, price_b: float = 0.0) -> Position:
        """Create a new position from a signal and size."""
        direction = "long" if signal.signal_type.value == "entry_long" else "short"

        position = Position(
            pair_key=signal.pair_key,
            token_a_mint=signal.token_a_mint,
            token_b_mint=signal.token_b_mint,
            direction=direction,
            entry_time=signal.timestamp,
            entry_slot=signal.slot,
            entry_zscore=signal.zscore,
            entry_price_a=price_a,
            entry_price_b=price_b,
            hedge_ratio=signal.hedge_ratio,
            quantity_a=size.token_a_amount,
            quantity_b=size.token_b_amount,
            quantity_a_raw=size.token_a_raw,
            quantity_b_raw=size.token_b_raw,
            entry_value_a=size.dollar_amount_a,
            entry_value_b=size.dollar_amount_b,
            is_paper=is_paper,
        )
        position.current_price_a = position.entry_price_a
        position.current_price_b = position.entry_price_b

        # Persist immediately
        self.db.save_position(position)
        self.positions[position.pair_key] = position

        logger.info(f"Opened {direction} position: {signal.pair_key} "
                     f"z={signal.zscore:+.2f} "
                     f"${size.total_exposure_usd:.0f} exposure")
        return position

    def close_position(self, pair_key: str, exit_zscore: float,
                       exit_slot: int, reason: str) -> Optional[Position]:
        """Close an existing position and compute realized P&L."""
        position = self.positions.get(pair_key)
        if not position:
            return None

        position.exit_time = int(time.time())
        position.exit_slot = exit_slot
        position.exit_zscore = exit_zscore
        position.exit_price_a = position.current_price_a
        position.exit_price_b = position.current_price_b

        # Compute realized P&L
        if position.direction == "long":
            # Long spread: bought A, sold B
            pnl_a = (position.current_price_a - position.entry_price_a) * position.quantity_a
            pnl_b = (position.entry_price_b - position.current_price_b) * position.quantity_b
        else:
            # Short spread: sold A, bought B
            pnl_a = (position.entry_price_a - position.current_price_a) * position.quantity_a
            pnl_b = (position.current_price_b - position.entry_price_b) * position.quantity_b

        position.realized_pnl = pnl_a + pnl_b
        position.status = PositionStatus.STOPPED_OUT if reason == "stop_loss" else PositionStatus.CLOSED

        # Update totals
        self.total_realized_pnl += position.realized_pnl
        self.db.set_state('total_realized_pnl', str(self.total_realized_pnl))

        # Persist
        self.db.update_position(position)
        del self.positions[pair_key]
        self.closed_positions.append(position)

        logger.info(f"Closed {position.direction} {pair_key} ({reason}): "
                     f"P&L ${position.realized_pnl:+.2f}")
        return position

    def mark_to_market(self, prices: Dict[str, float]):
        """Update all open positions with latest prices."""
        for position in self.positions.values():
            price_a = prices.get(position.token_a_mint)
            price_b = prices.get(position.token_b_mint)

            if price_a:
                position.current_price_a = price_a
            if price_b:
                position.current_price_b = price_b

            # Compute unrealized P&L
            if position.direction == "long":
                pnl_a = (position.current_price_a - position.entry_price_a) * position.quantity_a
                pnl_b = (position.entry_price_b - position.current_price_b) * position.quantity_b
            else:
                pnl_a = (position.entry_price_a - position.current_price_a) * position.quantity_a
                pnl_b = (position.current_price_b - position.entry_price_b) * position.quantity_b

            position.unrealized_pnl = pnl_a + pnl_b

        # Update peak value
        total = self.get_total_value()
        if total > self.peak_value:
            self.peak_value = total
            self.db.set_state('peak_value', str(self.peak_value))

    def get_total_exposure(self) -> float:
        total = 0.0
        for p in self.positions.values():
            total += abs(p.current_price_a * p.quantity_a)
            total += abs(p.current_price_b * p.quantity_b)
        return total

    def get_total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    def get_total_value(self) -> float:
        return self.initial_capital + self.total_realized_pnl + self.get_total_unrealized_pnl()

    def get_drawdown(self) -> float:
        if self.peak_value <= 0:
            return 0.0
        total = self.get_total_value()
        return (self.peak_value - total) / self.peak_value

    def has_position(self, pair_key: str) -> bool:
        return pair_key in self.positions

    def take_snapshot(self):
        """Save a portfolio snapshot to DB."""
        self.db.save_snapshot(
            total_value=self.get_total_value(),
            num_positions=len(self.positions),
            total_exposure=self.get_total_exposure(),
            realized_pnl=self.total_realized_pnl,
            unrealized_pnl=self.get_total_unrealized_pnl(),
            drawdown_pct=self.get_drawdown(),
        )

    def save_capital(self):
        """Persist initial capital to DB."""
        self.db.set_state('initial_capital', str(self.initial_capital))
