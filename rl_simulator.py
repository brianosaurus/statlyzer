#!/usr/bin/env python3
"""
RL Simulator — trains an RL agent using historical candle data as a live environment.

Instead of PPO pretraining with hindsight labels (which collapses to "always enter"),
this uses the backtester as a LIVE ENVIRONMENT where the RL agent makes decisions
and sees real outcomes. Proper online RL, not offline.

Usage:
    python3 rl_simulator.py \
        --db backtest_data.db \
        --scanner-db ../arbitrage_tracker/arb_tracker.db \
        --episodes 50 \
        --entry-z 1.0 --exit-z 0.1 --min-spread-bps 15 \
        --slippage-bps 3 --capital 5000 \
        --max-basket-size 2 \
        --token-whitelist SOL,bSOL,jitoSOL,mSOL,jupSOL,stSOL,JUP,FARTCOIN,ETH \
        --model-path rl_sim_model \
        --hidden 64
"""

import argparse
import json
import math
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_agent import (
    PolicyNetwork, RunningNormalizer, Transition,
    OBS_DIM, NUM_ACTIONS,
    ACTION_PASS, ACTION_ENTER_SMALL, ACTION_ENTER_NORMAL, ACTION_ENTER_LARGE, ACTION_EXIT,
    SIZE_MULTIPLIERS, ENTRY_ACTIONS, EXIT_ACTIONS,
)


# ---------------------------------------------------------------------------
# Data classes (simplified from backtest.py)
# ---------------------------------------------------------------------------

@dataclass
class Basket:
    basket_key: str
    basket_size: int
    mints: List[str]
    symbols: List[str]
    hedge_ratios: List[float]
    spread_mean: float
    spread_std: float
    half_life: float


@dataclass
class SimPosition:
    """A simulated position held by the RL agent."""
    basket_key: str
    basket_size: int
    direction: str           # "long" or "short"
    entry_z: float
    entry_spread: float
    entry_time: float
    size_usd: float
    size_multiplier: float   # from RL action


# ---------------------------------------------------------------------------
# Data loaders (copied/simplified from backtest.py)
# ---------------------------------------------------------------------------

def load_candles(db_path: str) -> List[Tuple[str, float, List[float]]]:
    """Load all price candles from DB, sorted by timestamp."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT basket_key, timestamp, log_prices_json "
        "FROM price_candles ORDER BY timestamp ASC"
    ).fetchall()
    conn.close()
    candles = []
    for basket_key, ts, lp_json in rows:
        log_prices = json.loads(lp_json)
        candles.append((basket_key, ts, log_prices))
    return candles


def load_baskets_from_scanner(scanner_db_path: str) -> Dict[str, Basket]:
    """Load cointegrated baskets from scanner DB."""
    conn = sqlite3.connect(f"file:{scanner_db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute("""
            SELECT basket_key, basket_size, mints_json, symbols_json,
                   hedge_ratios_json, spread_mean, spread_std, half_life
            FROM cointegration_results
            WHERE (eg_is_cointegrated = 1 OR johansen_is_cointegrated = 1)
        """).fetchall()
    finally:
        conn.close()
    baskets = {}
    for row in rows:
        bk = row[0]
        baskets[bk] = Basket(
            basket_key=bk, basket_size=row[1],
            mints=json.loads(row[2]), symbols=json.loads(row[3]),
            hedge_ratios=json.loads(row[4]),
            spread_mean=row[5], spread_std=row[6], half_life=row[7],
        )
    return baskets


def load_baskets_from_candles_db(db_path: str) -> Dict[str, Basket]:
    """Fallback: infer baskets from discovered_pairs table."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return {}
    baskets = {}
    try:
        rows = conn.execute("""
            SELECT pair_key, basket_size, mints_json, symbols_json,
                   hedge_ratios_json, spread_mean, spread_std, half_life
            FROM discovered_pairs
        """).fetchall()
        for row in rows:
            bk = row[0]
            baskets[bk] = Basket(
                basket_key=bk, basket_size=row[1],
                mints=json.loads(row[2]), symbols=json.loads(row[3]),
                hedge_ratios=json.loads(row[4]),
                spread_mean=row[5], spread_std=row[6], half_life=row[7],
            )
    except Exception:
        pass
    finally:
        conn.close()
    return baskets


# ---------------------------------------------------------------------------
# Simulator Environment
# ---------------------------------------------------------------------------

@dataclass
class SimParams:
    entry_z: float = 2.0
    exit_z: float = 0.3
    stop_z: float = 4.0
    max_entry_z: float = 6.0
    min_spread_bps: float = 0.0
    slippage_bps: float = 3.0
    max_positions: int = 10
    fixed_fraction: float = 0.05
    capital: float = 1000.0
    max_basket_size: int = 99
    lookback: int = 100
    # RL-specific
    max_half_life_secs: float = 3600.0
    max_position_age_half_lives: float = 5.0


class SimulatorEnv:
    """
    Gym-like environment that wraps candle replay for RL training.

    Each step advances one candle. When entry/exit thresholds are hit,
    the environment yields an observation and asks the RL agent for a decision.
    """

    def __init__(self, params: SimParams, baskets: Dict[str, Basket],
                 candles: List[Tuple[str, float, List[float]]],
                 token_whitelist: Optional[set] = None):
        self.params = params
        self.baskets = baskets
        self.candles = candles
        self.token_whitelist = token_whitelist

        # Pre-filter candles by basket validity + whitelist
        self.valid_candles = self._filter_candles()

        self.reset()

    def _filter_candles(self) -> List[Tuple[str, float, List[float]]]:
        valid = []
        for basket_key, ts, log_prices in self.candles:
            basket = self.baskets.get(basket_key)
            if basket is None:
                continue
            if basket.basket_size > self.params.max_basket_size:
                continue
            if self.token_whitelist:
                if not all(s in self.token_whitelist for s in basket.symbols):
                    continue
            if len(log_prices) != basket.basket_size:
                continue
            valid.append((basket_key, ts, log_prices))
        return valid

    def reset(self):
        """Reset environment to start of candle data."""
        self.capital = self.params.capital
        self.initial_capital = self.params.capital
        self.peak_capital = self.params.capital
        self.positions: Dict[str, SimPosition] = {}
        self.closed_trades: List[dict] = []
        self.spread_buffers: Dict[str, List[float]] = defaultdict(list)
        self.candle_idx = 0
        self.entries_this_episode = 0
        self.current_time = 0.0
        self.start_time = 0.0
        if self.valid_candles:
            self.start_time = self.valid_candles[0][1]

    def finished(self) -> bool:
        return self.candle_idx >= len(self.valid_candles)

    def step(self) -> Optional[Tuple[np.ndarray, str, str, dict]]:
        """
        Advance one candle. If a signal fires, return (obs, context, basket_key, info).
        context is "entry" or "exit".
        info contains z, spread, direction, basket, etc.
        Returns None if no decision point at this candle.
        """
        if self.finished():
            return None

        basket_key, ts, log_prices = self.valid_candles[self.candle_idx]
        self.candle_idx += 1
        self.current_time = ts

        basket = self.baskets[basket_key]
        hr = np.array(basket.hedge_ratios)
        lp = np.array(log_prices)
        spread = float(lp @ hr)

        # Update spread buffer
        buf = self.spread_buffers[basket_key]
        buf.append(spread)
        if len(buf) > self.params.lookback:
            self.spread_buffers[basket_key] = buf[-self.params.lookback:]
            buf = self.spread_buffers[basket_key]

        if len(buf) < 30:
            return None

        arr = np.array(buf)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std < 1e-12:
            return None

        z = (spread - mean) / std
        if abs(z) > 100:
            return None

        in_position = basket_key in self.positions

        # --- Exit / Stop-loss check ---
        if in_position:
            pos = self.positions[basket_key]

            # Stop loss — forced, no RL decision
            if abs(z) > self.params.stop_z:
                pnl = self._close_position(basket_key, z, spread, ts, "stop_loss")
                # Return an exit observation with done=True so agent sees the outcome
                obs = self._build_obs(z, std, basket, pos)
                return (obs, "stop_loss", basket_key, {
                    'z': z, 'spread': spread, 'pnl': pnl, 'reason': 'stop_loss',
                })

            # Exit signal: z reverted inside exit band or crossed zero
            should_present_exit = False
            if abs(z) < self.params.exit_z:
                should_present_exit = True
            elif pos.direction == "long" and z > 0:
                should_present_exit = True
            elif pos.direction == "short" and z < 0:
                should_present_exit = True

            if should_present_exit:
                obs = self._build_obs(z, std, basket, pos)
                return (obs, "exit", basket_key, {
                    'z': z, 'spread': spread, 'direction': pos.direction,
                    'ts': ts,
                })

        # --- Entry check ---
        if not in_position:
            if len(self.positions) >= self.params.max_positions:
                return None
            if abs(z) > self.params.max_entry_z:
                return None
            if self.params.min_spread_bps > 0:
                spread_dev_bps = abs(z) * std * 10000
                if spread_dev_bps < self.params.min_spread_bps:
                    return None

            direction = None
            if z < -self.params.entry_z:
                direction = "long"
            elif z > self.params.entry_z:
                direction = "short"

            if direction:
                obs = self._build_obs(z, std, basket, position=None)
                return (obs, "entry", basket_key, {
                    'z': z, 'spread': spread, 'direction': direction,
                    'basket': basket, 'ts': ts,
                })

        return None

    def apply_action(self, action: int, context: str, basket_key: str, info: dict) -> float:
        """
        Apply the RL agent's action. Returns immediate reward (0 for entries,
        realized PnL/duration for exits and stop losses).
        """
        if context == "stop_loss":
            # Already closed in step(), just return the reward
            pnl = info['pnl']
            pos_duration = info.get('duration_hours', 0.01)
            return pnl / max(pos_duration, 0.01)

        if context == "entry":
            if action == ACTION_PASS:
                return 0.0  # chose to skip

            basket = info['basket']
            direction = info['direction']
            z = info['z']
            spread = info['spread']
            ts = info['ts']
            multiplier = SIZE_MULTIPLIERS.get(action, 1.0)
            base_size = self.params.capital * self.params.fixed_fraction
            size_usd = base_size * multiplier

            if size_usd <= 0:
                return 0.0

            # Deduct entry slippage
            slippage_cost = size_usd * (self.params.slippage_bps / 10000.0) * basket.basket_size
            self.capital -= slippage_cost

            self.positions[basket_key] = SimPosition(
                basket_key=basket_key,
                basket_size=basket.basket_size,
                direction=direction,
                entry_z=z,
                entry_spread=spread,
                entry_time=ts,
                size_usd=size_usd,
                size_multiplier=multiplier,
            )
            self.entries_this_episode += 1
            return 0.0  # reward comes when position closes

        if context == "exit":
            if action == ACTION_PASS:
                return 0.0  # chose to hold

            if action == ACTION_EXIT:
                z = info['z']
                spread = info['spread']
                ts = info['ts']
                pnl = self._close_position(basket_key, z, spread, ts, "exit")
                pos_duration = self.closed_trades[-1]['duration_hours'] if self.closed_trades else 0.01
                return pnl / max(pos_duration, 0.01)

        return 0.0

    def _close_position(self, basket_key: str, z: float, spread: float,
                        ts: float, reason: str) -> float:
        """Close a position, return PnL."""
        pos = self.positions.pop(basket_key)
        basket = self.baskets[basket_key]

        spread_change = spread - pos.entry_spread
        if pos.direction == "short":
            spread_change = -spread_change

        pnl = spread_change * pos.size_usd

        # Exit slippage
        slippage_cost = pos.size_usd * (self.params.slippage_bps / 10000.0) * basket.basket_size
        pnl -= slippage_cost

        self.capital += pnl
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital

        duration_hours = max((ts - pos.entry_time) / 3600, 0.01)

        self.closed_trades.append({
            'basket_key': basket_key,
            'direction': pos.direction,
            'entry_z': pos.entry_z,
            'exit_z': z,
            'entry_time': pos.entry_time,
            'exit_time': ts,
            'pnl': pnl,
            'reason': reason,
            'duration_hours': duration_hours,
            'size_multiplier': pos.size_multiplier,
        })

        return pnl

    def force_close_all(self):
        """Force-close remaining positions at last known state (end of episode)."""
        for bk in list(self.positions.keys()):
            buf = self.spread_buffers.get(bk, [])
            if len(buf) < 2:
                self.positions.pop(bk)
                continue
            arr = np.array(buf)
            std = float(np.std(arr))
            mean = float(np.mean(arr))
            spread = buf[-1]
            z = (spread - mean) / std if std > 1e-12 else 0.0
            self._close_position(bk, z, spread, self.current_time, "end_of_data")

    def _build_obs(self, z: float, spread_std: float, basket: Basket,
                   position: Optional[SimPosition] = None) -> np.ndarray:
        """Build the 24-dim observation vector matching rl_agent.py."""
        # Signal features (7)
        half_life = basket.half_life
        hl_norm = half_life * 0.4 / max(self.params.max_half_life_secs, 1)
        buf = self.spread_buffers.get(basket.basket_key, [])
        buf_full = min(len(buf) / self.params.lookback, 1.0)
        # p_value not available in sim — use 0.05 (cointegrated)
        p_val = 0.05
        # time_since_resample not meaningful in sim — use 0
        time_since_norm = 0.0

        # Position features (5)
        in_pos = 0.0
        age_norm = 0.0
        upnl_norm = 0.0
        entry_z = 0.0
        z_dist = 0.0
        if position is not None:
            in_pos = 1.0
            age_hours = (self.current_time - position.entry_time) / 3600
            hl_hours = half_life * 0.4 / 3600
            max_age_hl = self.params.max_position_age_half_lives
            age_norm = age_hours / max(hl_hours * max_age_hl, 0.01)

            # Estimate unrealized PnL
            current_buf = self.spread_buffers.get(basket.basket_key, [])
            if current_buf:
                current_spread = current_buf[-1]
                spread_change = current_spread - position.entry_spread
                if position.direction == "short":
                    spread_change = -spread_change
                unrealized_pnl = spread_change * position.size_usd
            else:
                unrealized_pnl = 0.0
            upnl_norm = unrealized_pnl / max(position.size_usd, 0.01)
            entry_z = position.entry_z
            z_dist = abs(z - entry_z)

        # Portfolio features (5)
        max_pos = max(self.params.max_positions, 1)
        cap = max(self.initial_capital, 1.0)
        total_exposure = sum(p.size_usd for p in self.positions.values())
        max_exp = cap * 0.5  # 50% max exposure
        pos_ratio = len(self.positions) / max_pos
        exp_ratio = total_exposure / max(max_exp, 1.0)
        drawdown = (self.peak_capital - self.capital) / max(self.peak_capital, 1.0)
        port_return = (self.capital - self.initial_capital) / cap
        rate_usage = min(self.entries_this_episode / max(max_pos * 2, 1), 1.0)

        # Performance features (4)
        recent = self.closed_trades[-20:] if self.closed_trades else []
        if recent:
            wins = sum(1 for t in recent if t['pnl'] > 0)
            win_rate = wins / len(recent)
            pnls = [t['pnl'] for t in recent]
            avg_pnl = np.mean(pnls)
            avg_size = np.mean([t.get('size_multiplier', 1.0) * self.params.capital * self.params.fixed_fraction
                                for t in recent]) or 1.0
            avg_pnl_norm = avg_pnl / max(avg_size, 0.01)
            if len(pnls) >= 2:
                m = np.mean(pnls)
                s = np.std(pnls)
                sharpe = m / s if s > 0 else 0.0
            else:
                sharpe = 0.0
        else:
            win_rate = 0.5
            avg_pnl_norm = 0.0
            sharpe = 0.0

        # Time-of-day (use simulated time)
        hour = (self.current_time % 86400) / 3600
        tod_sin = math.sin(2 * math.pi * hour / 24)

        # Slippage features (3) — in sim we use configured slippage
        worst_rt_bps = self.params.slippage_bps / 100.0  # normalize
        avg_rt_bps = self.params.slippage_bps / 100.0
        basket_n = basket.basket_size

        obs = np.array([
            # Signal (7)
            z,
            abs(z),
            spread_std,
            hl_norm,
            buf_full,
            p_val,
            time_since_norm,
            # Position (5)
            in_pos,
            age_norm,
            upnl_norm,
            entry_z,
            z_dist,
            # Portfolio (5)
            pos_ratio,
            exp_ratio,
            drawdown,
            port_return,
            rate_usage,
            # Performance (4)
            win_rate,
            avg_pnl_norm,
            sharpe,
            tod_sin,
            # Slippage (3)
            worst_rt_bps,
            avg_rt_bps,
            basket_n / 4.0,
        ], dtype=np.float32)

        obs = np.clip(obs, -10.0, 10.0)
        return obs

    def get_episode_stats(self) -> dict:
        """Return stats for the completed episode."""
        trades = self.closed_trades
        total_pnl = sum(t['pnl'] for t in trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        return {
            'total_pnl': total_pnl,
            'num_trades': len(trades),
            'win_rate': wins / len(trades) if trades else 0.0,
            'final_capital': self.capital,
            'max_drawdown': (self.peak_capital - min(
                self.initial_capital,
                *(self._running_equity())
            )) / max(self.peak_capital, 1.0) if trades else 0.0,
            'stop_rate': sum(1 for t in trades if t['reason'] == 'stop_loss') / len(trades) if trades else 0.0,
        }

    def _running_equity(self) -> List[float]:
        """Compute running equity curve from closed trades."""
        equity = self.initial_capital
        curve = [equity]
        for t in self.closed_trades:
            equity += t['pnl']
            curve.append(equity)
        return curve


# ---------------------------------------------------------------------------
# PPO Trainer (standalone, no RLDecisionMaker dependency)
# ---------------------------------------------------------------------------

class PPOTrainer:
    """PPO trainer that interacts with SimulatorEnv."""

    # Hyperparameters
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPSILON = 0.2
    ENTROPY_COEF = 0.02       # slightly higher for exploration in sim
    VALUE_COEF = 0.5
    PPO_EPOCHS = 4
    BATCH_SIZE = 64
    MIN_TRANSITIONS = 32
    MAX_GRAD_NORM = 0.5

    def __init__(self, network: PolicyNetwork, lr: float = 3e-4):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.normalizer = RunningNormalizer(OBS_DIM)
        self.buffer: List[Transition] = []
        # Pending entries: basket_key -> Transition (reward assigned on close)
        self.pending_entries: Dict[str, Transition] = {}

    def select_action(self, obs: np.ndarray, valid_actions: List[int],
                      epsilon: float = 0.0) -> Tuple[int, float, float]:
        """Select action using policy network with optional epsilon-greedy exploration."""
        self.normalizer.update(obs)
        norm_obs = self.normalizer.normalize(obs)

        mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
        for a in valid_actions:
            mask[a] = True

        with torch.no_grad():
            x = torch.FloatTensor(norm_obs).unsqueeze(0)
            m = mask.unsqueeze(0)
            logits, value = self.network(x, action_mask=m)
            dist = torch.distributions.Categorical(logits=logits)

            if np.random.random() < epsilon:
                action = np.random.choice(valid_actions)
                action_t = torch.tensor(action)
                log_prob = dist.log_prob(action_t)
            else:
                action_t = dist.sample()
                log_prob = dist.log_prob(action_t)
                action = action_t.item()

        return action, log_prob.item(), value.item()

    def record_entry(self, basket_key: str, obs: np.ndarray,
                     action: int, log_prob: float, value: float):
        """Record an entry decision, waiting for position close to get reward."""
        if action == ACTION_PASS:
            # PASS is immediate with zero reward
            self.buffer.append(Transition(
                obs=obs, action=action, log_prob=log_prob, value=value,
                reward=0.0, done=True, context="entry",
            ))
        else:
            self.pending_entries[basket_key] = Transition(
                obs=obs, action=action, log_prob=log_prob, value=value,
                context="entry",
            )

    def record_exit(self, obs: np.ndarray, action: int, log_prob: float,
                    value: float, reward: float):
        """Record an exit decision with immediate reward."""
        self.buffer.append(Transition(
            obs=obs, action=action, log_prob=log_prob, value=value,
            reward=reward, done=True, context="exit",
        ))

    def assign_entry_reward(self, basket_key: str, reward: float):
        """Assign reward to a pending entry decision when position closes."""
        if basket_key in self.pending_entries:
            t = self.pending_entries.pop(basket_key)
            t.reward = reward
            t.done = True
            self.buffer.append(t)

    def flush_pending(self):
        """Flush any remaining pending entries at end of episode with zero reward."""
        for bk, t in self.pending_entries.items():
            t.reward = 0.0
            t.done = True
            self.buffer.append(t)
        self.pending_entries.clear()

    def train_step(self) -> Optional[dict]:
        """Run PPO update on buffer. Returns stats or None if not enough data."""
        done = [t for t in self.buffer if t.done and len(t.obs) == OBS_DIM]
        if len(done) < self.MIN_TRANSITIONS:
            return None

        obs_arr = np.array([t.obs for t in done])
        actions = torch.LongTensor([t.action for t in done])
        old_log_probs = torch.FloatTensor([t.log_prob for t in done])
        rewards = torch.FloatTensor([t.reward for t in done])
        old_values = torch.FloatTensor([t.value for t in done])

        # Update normalizer
        for o in obs_arr:
            self.normalizer.update(o)
        norm_obs = np.array([self.normalizer.normalize(o) for o in obs_arr])
        obs_t = torch.FloatTensor(norm_obs)

        # Action masks
        masks = []
        for t in done:
            mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
            valid = ENTRY_ACTIONS if t.context == "entry" else EXIT_ACTIONS
            for a in valid:
                mask[a] = True
            masks.append(mask)
        mask_t = torch.stack(masks)

        # Returns and advantages (all done=True, single-step)
        returns = rewards
        advantages = returns - old_values
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(done)
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.PPO_EPOCHS):
            indices = np.random.permutation(n)
            for start in range(0, n, self.BATCH_SIZE):
                end = min(start + self.BATCH_SIZE, n)
                idx = indices[start:end]

                b_obs = obs_t[idx]
                b_actions = actions[idx]
                b_old_lp = old_log_probs[idx]
                b_adv = advantages[idx]
                b_ret = returns[idx]
                b_mask = mask_t[idx]

                logits, values = self.network(b_obs, action_mask=b_mask)
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - b_old_lp)
                clipped = torch.clamp(ratio, 1 - self.CLIP_EPSILON,
                                      1 + self.CLIP_EPSILON)
                policy_loss = -torch.min(ratio * b_adv, clipped * b_adv).mean()
                value_loss = F.mse_loss(values.squeeze(-1), b_ret)
                loss = (policy_loss
                        + self.VALUE_COEF * value_loss
                        - self.ENTROPY_COEF * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.MAX_GRAD_NORM)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        # Clear buffer
        self.buffer = [t for t in self.buffer if not t.done]

        return {
            'loss': total_loss / max(num_updates, 1),
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'value_loss': total_value_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1),
            'n_transitions': n,
            'avg_reward': float(rewards.mean()),
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training(args):
    """Main training loop."""
    # Load data
    print(f"Loading candles from {args.db}...")
    candles = load_candles(args.db)
    if not candles:
        print("ERROR: No candles found")
        sys.exit(1)

    basket_keys = set(c[0] for c in candles)
    print(f"  {len(candles)} candles across {len(basket_keys)} baskets")

    # Load baskets
    baskets = {}
    if args.scanner_db:
        print(f"Loading baskets from scanner DB: {args.scanner_db}...")
        baskets = load_baskets_from_scanner(args.scanner_db)
    if not baskets:
        print(f"Loading baskets from discovered_pairs in {args.db}...")
        baskets = load_baskets_from_candles_db(args.db)
    if not baskets:
        print("ERROR: No basket data found")
        sys.exit(1)

    matched = set(baskets.keys()) & basket_keys
    print(f"  {len(baskets)} baskets loaded, {len(matched)} have candle data")
    if not matched:
        print("ERROR: No baskets match candle data")
        sys.exit(1)

    token_whitelist = None
    if args.token_whitelist:
        token_whitelist = set(s.strip() for s in args.token_whitelist.split(','))
        print(f"  Token whitelist: {token_whitelist}")

    # Create env params
    sim_params = SimParams(
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        stop_z=args.stop_z,
        max_entry_z=args.max_entry_z,
        min_spread_bps=args.min_spread_bps,
        slippage_bps=args.slippage_bps,
        max_positions=args.max_positions,
        fixed_fraction=args.fixed_fraction,
        capital=args.capital,
        max_basket_size=args.max_basket_size,
        lookback=args.lookback,
    )

    # Create network and trainer
    network = PolicyNetwork(OBS_DIM, NUM_ACTIONS, hidden=args.hidden)

    # Optionally load existing model
    if args.load_model:
        try:
            network.load_state_dict(
                torch.load(f'{args.load_model}.pt', weights_only=True, map_location='cpu'))
            print(f"  Loaded existing model from {args.load_model}.pt")
        except Exception as e:
            print(f"  Could not load model: {e}. Starting fresh.")

    trainer = PPOTrainer(network, lr=args.lr)

    # If loading normalizer from existing model
    if args.load_model:
        try:
            with open(f'{args.load_model}_meta.json') as f:
                meta = json.load(f)
            trainer.normalizer.load_state_dict(meta['normalizer'])
            print(f"  Loaded normalizer state")
        except Exception:
            pass

    env = SimulatorEnv(sim_params, baskets, candles, token_whitelist)
    print(f"  {len(env.valid_candles)} valid candles after filtering")

    if not env.valid_candles:
        print("ERROR: No valid candles after filtering")
        sys.exit(1)

    # Training loop
    print(f"\n{'='*70}")
    print(f"  RL SIMULATOR TRAINING")
    print(f"  Episodes: {args.episodes} | Hidden: {args.hidden} | LR: {args.lr}")
    print(f"  Entry Z: {args.entry_z} | Exit Z: {args.exit_z} | Slippage: {args.slippage_bps}bps")
    print(f"  Capital: ${args.capital} | Max positions: {args.max_positions}")
    print(f"{'='*70}\n")

    episode_results = []
    best_pnl_per_hr = float('-inf')
    best_episode = -1

    # Exploration schedule: start high, decay
    epsilon_start = 0.3
    epsilon_end = 0.02
    epsilon_decay = args.episodes

    for episode in range(args.episodes):
        env.reset()
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0, 1 - episode / epsilon_decay)

        episode_rewards = []
        entry_decisions = 0
        exit_decisions = 0
        pass_count = 0
        stop_count = 0

        while not env.finished():
            result = env.step()
            if result is None:
                continue

            obs, context, basket_key, info = result

            if context == "stop_loss":
                # Forced stop loss — record as observation
                pnl = info['pnl']
                duration_hours = env.closed_trades[-1]['duration_hours'] if env.closed_trades else 0.01
                reward = pnl / max(duration_hours, 0.01)

                # Get log_prob and value for the forced exit
                _, log_prob, value = trainer.select_action(obs, EXIT_ACTIONS, epsilon=0.0)
                trainer.record_exit(obs, ACTION_EXIT, log_prob, value, reward)

                # Also assign reward to the entry that led here
                trainer.assign_entry_reward(basket_key, reward)
                episode_rewards.append(reward)
                stop_count += 1
                continue

            if context == "entry":
                action, log_prob, value = trainer.select_action(obs, ENTRY_ACTIONS, epsilon=epsilon)
                trainer.record_entry(basket_key, obs, action, log_prob, value)

                # Apply action to environment
                env.apply_action(action, context, basket_key, info)
                entry_decisions += 1
                if action == ACTION_PASS:
                    pass_count += 1
                    episode_rewards.append(0.0)

            elif context == "exit":
                action, log_prob, value = trainer.select_action(obs, EXIT_ACTIONS, epsilon=epsilon)

                # Apply action to get reward
                reward = env.apply_action(action, context, basket_key, info)

                trainer.record_exit(obs, action, log_prob, value, reward)

                if action == ACTION_EXIT and reward != 0.0:
                    # Assign reward to the entry that opened this position
                    trainer.assign_entry_reward(basket_key, reward)
                    episode_rewards.append(reward)
                    exit_decisions += 1
                elif action == ACTION_PASS:
                    pass_count += 1

        # End of episode: force-close remaining positions
        for bk in list(env.positions.keys()):
            pos = env.positions[bk]
            buf = env.spread_buffers.get(bk, [])
            if len(buf) >= 2:
                arr = np.array(buf)
                std = float(np.std(arr))
                mean = float(np.mean(arr))
                spread = buf[-1]
                z = (spread - mean) / std if std > 1e-12 else 0.0

                obs = env._build_obs(z, std, baskets[bk], pos)
                pnl = env._close_position(bk, z, spread, env.current_time, "end_of_data")
                duration_hours = env.closed_trades[-1]['duration_hours'] if env.closed_trades else 0.01
                reward = pnl / max(duration_hours, 0.01)

                _, log_prob, value = trainer.select_action(obs, EXIT_ACTIONS, epsilon=0.0)
                trainer.record_exit(obs, ACTION_EXIT, log_prob, value, reward)
                trainer.assign_entry_reward(bk, reward)
                episode_rewards.append(reward)
            else:
                env.positions.pop(bk)

        # Flush any remaining pending entries
        trainer.flush_pending()

        # Run PPO update
        train_stats = trainer.train_step()

        # Episode stats
        stats = env.get_episode_stats()
        total_pnl = stats['total_pnl']
        num_trades = stats['num_trades']
        win_rate = stats['win_rate']
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0

        duration_hrs = (env.current_time - env.start_time) / 3600 if env.current_time > env.start_time else 1.0
        pnl_per_hr = total_pnl / max(duration_hrs, 0.01)

        episode_results.append({
            'episode': episode + 1,
            'total_pnl': total_pnl,
            'pnl_per_hr': pnl_per_hr,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'epsilon': epsilon,
            'entry_decisions': entry_decisions,
            'pass_count': pass_count,
            'stop_count': stop_count,
            'final_capital': stats['final_capital'],
            'train_loss': train_stats['loss'] if train_stats else None,
            'train_entropy': train_stats['entropy'] if train_stats else None,
        })

        # Track best
        if pnl_per_hr > best_pnl_per_hr and num_trades > 0:
            best_pnl_per_hr = pnl_per_hr
            best_episode = episode + 1
            _save_model(network, trainer.normalizer, args.model_path, episode + 1,
                        best_pnl_per_hr, args.hidden)

        # Print progress
        loss_str = f"loss={train_stats['loss']:.4f}" if train_stats else "no_update"
        ent_str = f"H={train_stats['entropy']:.3f}" if train_stats else ""
        pass_pct = pass_count / max(entry_decisions, 1) * 100
        print(f"  Ep {episode+1:>3}/{args.episodes} | "
              f"PnL=${total_pnl:+8.2f} (${pnl_per_hr:+.4f}/hr) | "
              f"trades={num_trades:>4} win={win_rate:5.1%} | "
              f"pass={pass_pct:4.1f}% stops={stop_count:>3} | "
              f"eps={epsilon:.3f} | {loss_str} {ent_str}")

    # ---------------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE — {args.episodes} episodes")
    print(f"{'='*70}")

    if not episode_results:
        print("  No episodes completed.")
        return

    # Episode 1 vs last
    first = episode_results[0]
    last = episode_results[-1]

    print(f"\n  {'Metric':<25} {'Episode 1':>15} {'Episode {}'.format(args.episodes):>15} {'Change':>12}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*12}")

    def _fmt(v, fmt='${:+,.2f}'):
        return fmt.format(v) if v is not None else 'n/a'

    print(f"  {'Total PnL':<25} {_fmt(first['total_pnl']):>15} {_fmt(last['total_pnl']):>15} "
          f"{_fmt(last['total_pnl'] - first['total_pnl']):>12}")
    print(f"  {'$/hr':<25} {_fmt(first['pnl_per_hr'], '${:+,.4f}'):>15} "
          f"{_fmt(last['pnl_per_hr'], '${:+,.4f}'):>15} "
          f"{_fmt(last['pnl_per_hr'] - first['pnl_per_hr'], '${:+,.4f}'):>12}")
    print(f"  {'Trades':<25} {first['num_trades']:>15} {last['num_trades']:>15} "
          f"{last['num_trades'] - first['num_trades']:>12}")
    print(f"  {'Win Rate':<25} {first['win_rate']:>14.1%} {last['win_rate']:>14.1%} "
          f"{last['win_rate'] - first['win_rate']:>+11.1%}")
    print(f"  {'Avg Reward':<25} {first['avg_reward']:>15.4f} {last['avg_reward']:>15.4f} "
          f"{last['avg_reward'] - first['avg_reward']:>+12.4f}")
    print(f"  {'Pass Rate':<25} "
          f"{first['pass_count']/max(first['entry_decisions'],1):>14.1%} "
          f"{last['pass_count']/max(last['entry_decisions'],1):>14.1%}")

    # Best episode
    print(f"\n  Best episode: #{best_episode} (${best_pnl_per_hr:+.4f}/hr)")
    print(f"  Model saved to: {args.model_path}.pt")

    # Running averages
    window = min(5, len(episode_results))
    if len(episode_results) >= window * 2:
        early = episode_results[:window]
        late = episode_results[-window:]
        early_pnl = np.mean([e['total_pnl'] for e in early])
        late_pnl = np.mean([e['total_pnl'] for e in late])
        early_wr = np.mean([e['win_rate'] for e in early])
        late_wr = np.mean([e['win_rate'] for e in late])
        print(f"\n  Avg first {window} episodes:  PnL=${early_pnl:+.2f}  WinRate={early_wr:.1%}")
        print(f"  Avg last  {window} episodes:  PnL=${late_pnl:+.2f}  WinRate={late_wr:.1%}")

    print()


def _save_model(network: PolicyNetwork, normalizer: RunningNormalizer,
                model_path: str, episode: int, best_pnl_per_hr: float,
                hidden: int):
    """Save model in the same format as rl_agent.py (compatible with live bot)."""
    torch.save(network.state_dict(), f'{model_path}.pt')
    meta = {
        'normalizer': normalizer.state_dict(),
        'total_decisions': 0,
        'train_updates': episode,
        'num_closed': 0,
        'sim_best_pnl_per_hr': best_pnl_per_hr,
        'sim_episode': episode,
        'hidden_dim': hidden,
    }
    with open(f'{model_path}_meta.json', 'w') as f:
        json.dump(meta, f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="RL Simulator: train an RL agent on historical candle data")
    parser.add_argument('--db', required=True, help='DB with price_candles table')
    parser.add_argument('--scanner-db', default=None,
                        help='Scanner DB with cointegration_results')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of training episodes (default: 50)')
    parser.add_argument('--entry-z', type=float, default=2.0)
    parser.add_argument('--exit-z', type=float, default=0.3)
    parser.add_argument('--stop-z', type=float, default=4.0)
    parser.add_argument('--max-entry-z', type=float, default=6.0)
    parser.add_argument('--min-spread-bps', type=float, default=0.0)
    parser.add_argument('--slippage-bps', type=float, default=3.0)
    parser.add_argument('--max-positions', type=int, default=10)
    parser.add_argument('--fixed-fraction', type=float, default=0.05)
    parser.add_argument('--capital', type=float, default=1000.0)
    parser.add_argument('--max-basket-size', type=int, default=99)
    parser.add_argument('--lookback', type=int, default=100)
    parser.add_argument('--token-whitelist', type=str, default=None,
                        help='Comma-separated symbols')
    parser.add_argument('--model-path', type=str, default='rl_sim_model',
                        help='Path prefix for saving model (default: rl_sim_model)')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path prefix to load existing model from')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Hidden layer size (default: 64)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_training(args)
