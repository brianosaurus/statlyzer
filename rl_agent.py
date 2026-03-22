"""
Reinforcement learning agent for statalyzer.
PPO-based policy that learns entry/exit decisions to maximize $/day.

Replaces fixed z-score thresholds with a learned policy that adapts
based on market state, portfolio context, and recent performance.
"""

import json
import logging
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Observation dimensions
OBS_DIM = 21

# Actions
ACTION_PASS = 0
ACTION_ENTER_SMALL = 1   # 0.5x base size
ACTION_ENTER_NORMAL = 2  # 1.0x base size
ACTION_ENTER_LARGE = 3   # 1.5x base size
ACTION_EXIT = 4

SIZE_MULTIPLIERS = {
    ACTION_PASS: 0.0,
    ACTION_ENTER_SMALL: 0.5,
    ACTION_ENTER_NORMAL: 1.0,
    ACTION_ENTER_LARGE: 1.5,
    ACTION_EXIT: 0.0,
}

# Valid actions per context
ENTRY_ACTIONS = [ACTION_PASS, ACTION_ENTER_SMALL, ACTION_ENTER_NORMAL, ACTION_ENTER_LARGE]
EXIT_ACTIONS = [ACTION_PASS, ACTION_EXIT]  # PASS = hold

NUM_ACTIONS = 5


@dataclass
class Transition:
    """One decision + outcome."""
    obs: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float = 0.0
    done: bool = False
    context: str = "entry"  # "entry" or "exit"


class PolicyNetwork(nn.Module):
    """Small actor-critic MLP shared across entry and exit decisions."""

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = NUM_ACTIONS,
                 hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x, action_mask=None):
        """
        Args:
            x: (batch, obs_dim) observation tensor
            action_mask: (batch, n_actions) bool tensor, True = valid action
        Returns:
            logits: (batch, n_actions) masked logits
            value: (batch, 1) state value estimate
        """
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h)

        if action_mask is not None:
            # Set invalid action logits to -inf
            logits = logits.masked_fill(~action_mask, float('-inf'))

        return logits, value


class RunningNormalizer:
    """Online running mean/std for observation normalization."""

    def __init__(self, dim: int):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray):
        self.count += 1
        if self.count == 1:
            self.mean = x.astype(np.float64).copy()
            self.var = np.zeros(len(x), dtype=np.float64)
        else:
            delta = x.astype(np.float64) - self.mean
            self.mean += delta / self.count
            self.var += delta * (x.astype(np.float64) - self.mean)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var / max(self.count - 1, 1)) + 1e-8
        return ((x.astype(np.float64) - self.mean) / std).astype(np.float32)

    def state_dict(self) -> dict:
        return {
            'mean': self.mean.tolist(),
            'var': self.var.tolist(),
            'count': self.count,
        }

    def load_state_dict(self, d: dict):
        self.mean = np.array(d['mean'], dtype=np.float64)
        self.var = np.array(d['var'], dtype=np.float64)
        self.count = d['count']


class RLDecisionMaker:
    """
    Wraps the PolicyNetwork with observation building, experience tracking,
    phased bootstrapping, and PPO training.

    Phases:
        1 (0 closed trades):    Rule-based passthrough, collecting experiences
        2 (1-100 closed trades): 80% rule-based, 20% exploration
        3 (100+ closed trades):  Full RL with 5% exploration
    """

    PHASE_1_MAX = 0
    PHASE_2_MAX = 100
    PHASE_2_EPSILON = 0.20
    PHASE_3_EPSILON = 0.05

    # PPO hyperparameters
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPSILON = 0.2
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.5
    PPO_EPOCHS = 4
    BATCH_SIZE = 32
    MIN_BUFFER_SIZE = 32
    LR = 3e-4
    MAX_GRAD_NORM = 0.5

    def __init__(self, config, db):
        self.config = config
        self.db = db

        model_path = getattr(config, 'rl_model_path', 'rl_model')
        self.model_path = model_path

        self.network = PolicyNetwork(OBS_DIM, NUM_ACTIONS,
                                     hidden=getattr(config, 'rl_hidden_dim', 64))
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=getattr(config, 'rl_learning_rate', self.LR),
        )
        self.normalizer = RunningNormalizer(OBS_DIM)

        # Experience buffer for PPO updates
        self.buffer: List[Transition] = []

        # Pending entry decisions (waiting for position close)
        self.pending_entries: Dict[str, Transition] = {}

        # Pending reconciliation (live trades waiting for on-chain finalization)
        # Keyed by position_id → (Transition, pair_key, duration_hours)
        self.pending_reconciliations: Dict[int, tuple] = {}

        # Stats
        self.total_decisions = 0
        self.train_updates = 0
        self.recent_rewards = deque(maxlen=100)

        # Determine phase from DB
        self.num_closed = self._count_closed_trades()

        # Try to load saved model
        self._try_load()

        phase = self._phase()
        logger.info(f"RL agent initialized: phase={phase}, "
                    f"closed_trades={self.num_closed}, "
                    f"model={'loaded' if os.path.exists(f'{model_path}.pt') else 'new'}")

    def _count_closed_trades(self) -> int:
        try:
            row = self.db.conn.execute(
                "SELECT COUNT(*) FROM positions WHERE status != 'open'"
            ).fetchone()
            return row[0] if row else 0
        except Exception:
            return 0

    def _phase(self) -> int:
        if self.num_closed <= self.PHASE_1_MAX:
            return 1
        elif self.num_closed <= self.PHASE_2_MAX:
            return 2
        return 3

    # ── Observation builder ──────────────────────────────────────────

    def build_obs(self, signal, basket_state, portfolio, risk_mgr,
                  position=None) -> np.ndarray:
        """Build the 21-dim observation vector."""
        z = signal.zscore if signal else (basket_state.current_zscore if basket_state else 0.0)
        spread_std = basket_state.spread_std if basket_state else 0.0
        half_life = basket_state.half_life if basket_state else 500.0
        hl_norm = half_life * 0.4 / max(self.config.max_half_life_secs, 1)  # normalized
        buf_full = (basket_state.price_buffers[0].count / self.config.lookback_window
                    if basket_state and basket_state.price_buffers else 0.0)
        p_val = basket_state.eg_p_value if basket_state else 1.0
        time_since = (time.time() - basket_state.last_resample_time
                      if basket_state and basket_state.last_resample_time > 0 else 0.0)
        time_since_norm = time_since / max(self.config.signal_resample_secs, 1)

        # Position features (zeros if no position)
        in_pos = 0.0
        age_norm = 0.0
        upnl_norm = 0.0
        entry_z = 0.0
        z_dist = 0.0
        if position is not None:
            in_pos = 1.0
            entry_value = sum(position.entry_values) if position.entry_values else 1.0
            age_hours = (time.time() - position.entry_time) / 3600
            hl_hours = half_life * 0.4 / 3600
            max_age_hl = self.config.max_position_age_half_lives
            age_norm = age_hours / max(hl_hours * max_age_hl, 0.01)
            upnl_norm = position.unrealized_pnl / max(entry_value, 0.01)
            entry_z = position.entry_zscore
            z_dist = abs(z - entry_z)

        # Portfolio features
        max_pos = max(self.config.max_positions, 1)
        cap = max(portfolio.initial_capital, 1.0)
        max_exp = cap * self.config.max_exposure_ratio
        pos_ratio = len(portfolio.positions) / max_pos
        exp_ratio = portfolio.get_total_exposure() / max(max_exp, 1.0)
        drawdown = portfolio.get_drawdown()
        port_return = (portfolio.get_total_value() - cap) / cap
        rate_usage = len(risk_mgr.entries_this_hour) / max(self.config.max_positions_per_hour, 1)

        # Recent performance
        closed = portfolio.closed_positions[-20:] if portfolio.closed_positions else []
        if closed:
            wins = sum(1 for p in closed if p.realized_pnl > 0)
            win_rate = wins / len(closed)
            pnls = [p.realized_pnl for p in closed]
            avg_pnl = np.mean(pnls)
            avg_entry = np.mean([sum(p.entry_values) for p in closed]) or 1.0
            avg_pnl_norm = avg_pnl / max(avg_entry, 0.01)
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

        hour = (time.time() % 86400) / 3600
        tod_sin = math.sin(2 * math.pi * hour / 24)

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
        ], dtype=np.float32)

        # Clamp extreme values
        obs = np.clip(obs, -10.0, 10.0)
        return obs

    # ── Decision methods ─────────────────────────────────────────────

    def decide_entry(self, signal, basket_state, portfolio, risk_mgr) -> int:
        """
        Decide entry action for a signal that passed risk checks.
        Returns action index (0=PASS, 1=SMALL, 2=NORMAL, 3=LARGE).
        """
        obs = self.build_obs(signal, basket_state, portfolio, risk_mgr)
        phase = self._phase()

        # Phase 1: always enter normal (rule-based passthrough)
        if phase == 1:
            action = ACTION_ENTER_NORMAL
            log_prob, value = self._get_log_prob_value(obs, action, ENTRY_ACTIONS)
            self._store_pending_entry(signal.pair_key, obs, action, log_prob, value)
            self.total_decisions += 1
            return action

        # Phase 2: mostly rule-based, some exploration
        epsilon = self.PHASE_2_EPSILON if phase == 2 else self.PHASE_3_EPSILON
        if random.random() < epsilon:
            action = random.choice(ENTRY_ACTIONS)
            log_prob, value = self._get_log_prob_value(obs, action, ENTRY_ACTIONS)
        else:
            if phase == 2 and random.random() < 0.8:
                # 80% rule-based in phase 2
                action = ACTION_ENTER_NORMAL
                log_prob, value = self._get_log_prob_value(obs, action, ENTRY_ACTIONS)
            else:
                # RL policy
                action, log_prob, value = self._sample_action(obs, ENTRY_ACTIONS)

        self._store_pending_entry(signal.pair_key, obs, action, log_prob, value)
        self.total_decisions += 1
        return action

    def decide_exit(self, signal, basket_state, portfolio, risk_mgr,
                    position) -> bool:
        """
        Decide whether to exit when an EXIT signal fires.
        Returns True to exit, False to hold.
        Stop-loss signals bypass this (always exit).
        """
        obs = self.build_obs(signal, basket_state, portfolio, risk_mgr,
                             position=position)
        phase = self._phase()

        # Phase 1: always exit (rule-based passthrough)
        if phase == 1:
            action = ACTION_EXIT
            log_prob, value = self._get_log_prob_value(obs, action, EXIT_ACTIONS)
            # Record immediately with estimated reward
            entry_value = sum(position.entry_values) if position.entry_values else 1.0
            duration_hours = max((time.time() - position.entry_time) / 3600, 0.01)
            reward = position.unrealized_pnl / max(duration_hours, 0.01)
            self.buffer.append(Transition(
                obs=obs, action=action, log_prob=log_prob, value=value,
                reward=reward, done=True, context="exit",
            ))
            return True

        # Phase 2/3: RL decision
        epsilon = self.PHASE_2_EPSILON if phase == 2 else self.PHASE_3_EPSILON
        if random.random() < epsilon:
            action = random.choice(EXIT_ACTIONS)
            log_prob, value = self._get_log_prob_value(obs, action, EXIT_ACTIONS)
        else:
            if phase == 2 and random.random() < 0.8:
                action = ACTION_EXIT
                log_prob, value = self._get_log_prob_value(obs, action, EXIT_ACTIONS)
            else:
                action, log_prob, value = self._sample_action(obs, EXIT_ACTIONS)

        should_exit = (action == ACTION_EXIT)

        # Record exit decision with immediate reward estimate
        if should_exit:
            entry_value = sum(position.entry_values) if position.entry_values else 1.0
            duration_hours = max((time.time() - position.entry_time) / 3600, 0.01)
            reward = position.unrealized_pnl / max(duration_hours, 0.01)
        else:
            reward = 0.0  # hold: reward comes later

        self.buffer.append(Transition(
            obs=obs, action=action, log_prob=log_prob, value=value,
            reward=reward, done=should_exit, context="exit",
        ))
        self.total_decisions += 1
        return should_exit

    # ── Reward feedback ──────────────────────────────────────────────

    def on_position_closed(self, pair_key: str, realized_pnl: float,
                           entry_value: float, duration_hours: float,
                           is_live: bool = False, position_id: int = 0):
        """Called when a position closes. Assigns reward to the entry decision.
        For live trades, defers reward until exit reconciliation completes."""
        self.num_closed += 1

        if is_live and position_id > 0:
            # Defer reward — actual PnL comes from on-chain reconciliation
            if pair_key in self.pending_entries:
                t = self.pending_entries.pop(pair_key)
                self.pending_reconciliations[position_id] = (t, pair_key, duration_hours)
                logger.info(f"RL: deferred reward for position {position_id} "
                            f"({pair_key[:16]}), waiting for reconciliation")
            self._save_pending_to_db()
            return

        # Paper mode: use theoretical PnL immediately
        reward = realized_pnl / max(duration_hours, 0.01)
        self.recent_rewards.append(reward)

        if pair_key in self.pending_entries:
            t = self.pending_entries.pop(pair_key)
            t.reward = reward
            t.done = True
            self.buffer.append(t)

        # Persist pending entry to DB for crash recovery
        self._save_pending_to_db()

    def on_entry_skipped(self, signal, basket_state, portfolio, risk_mgr):
        """Record a PASS decision with zero reward."""
        obs = self.build_obs(signal, basket_state, portfolio, risk_mgr)
        log_prob, value = self._get_log_prob_value(obs, ACTION_PASS, ENTRY_ACTIONS)
        self.buffer.append(Transition(
            obs=obs, action=ACTION_PASS, log_prob=log_prob, value=value,
            reward=0.0, done=True, context="entry",
        ))

    def process_reconciled_exits(self, db, sol_price: float) -> int:
        """Poll for finalized exit reconciliations and assign actual PnL rewards.
        Returns number of reconciliations processed."""
        if not self.pending_reconciliations and not db:
            return 0

        try:
            records = db.get_unprocessed_reconciliations()
        except Exception as e:
            logger.debug(f"RL reconciliation poll failed: {e}")
            return 0

        processed = 0
        for rec in records:
            position_id = rec['position_id']
            rec_id = rec['rec_id']

            # Compute actual PnL from SOL balance deltas
            entry_sol_before = rec.get('entry_sol_before')
            entry_sol_after = rec.get('entry_sol_after')
            exit_sol_before = rec['exit_sol_before']
            exit_sol_after = rec['exit_sol_after']

            if entry_sol_before is not None and entry_sol_after is not None:
                # Full round-trip: entry delta + exit delta
                entry_delta = entry_sol_after - entry_sol_before
                exit_delta = exit_sol_after - exit_sol_before
                actual_pnl_sol = entry_delta + exit_delta
                actual_pnl_usd = actual_pnl_sol * sol_price
            else:
                # No entry SOL data — fall back to exit-only correction
                # Adjust expected PnL by exit execution difference
                actual_pnl_usd = rec['expected_pnl']

            entry_time = rec.get('entry_time', 0)
            exit_time = rec.get('exit_time', 0)
            duration_hours = max((exit_time - entry_time) / 3600, 0.01) if exit_time and entry_time else 0.01

            reward = actual_pnl_usd / max(duration_hours, 0.01)
            self.recent_rewards.append(reward)

            # Assign reward to the pending transition
            if position_id in self.pending_reconciliations:
                t, pair_key, _ = self.pending_reconciliations.pop(position_id)
                t.reward = reward
                t.done = True
                self.buffer.append(t)
                logger.info(f"RL: reconciled pos {position_id} — "
                            f"actual=${actual_pnl_usd:+.4f} vs expected=${rec['expected_pnl']:+.4f} "
                            f"reward={reward:+.4f}$/hr")
            else:
                logger.info(f"RL: reconciled pos {position_id} (no pending transition) — "
                            f"actual=${actual_pnl_usd:+.4f}")

            db.mark_reconciliation_rl_processed(rec_id)
            processed += 1

        return processed

    # ── Training ─────────────────────────────────────────────────────

    def maybe_train(self) -> bool:
        """Run PPO update if buffer has enough transitions. Returns True if trained."""
        done_transitions = [t for t in self.buffer if t.done]
        if len(done_transitions) < self.MIN_BUFFER_SIZE:
            return False

        logger.info(f"RL training: {len(done_transitions)} transitions, "
                    f"phase={self._phase()}")

        self._ppo_update(done_transitions)
        # Keep only undone transitions (pending holds)
        self.buffer = [t for t in self.buffer if not t.done]
        self.train_updates += 1

        # Save model
        self.save()

        # Check for fallback
        if self._should_fallback():
            logger.warning("RL fallback triggered: performance below baseline. "
                           "Reverting to Phase 1.")
            self.num_closed = 0  # Reset to phase 1

        return True

    def _should_fallback(self) -> bool:
        """Check if RL performance has degraded enough to warrant fallback."""
        if len(self.recent_rewards) < 20:
            return False
        rewards = list(self.recent_rewards)
        mean_r = np.mean(rewards[-20:])
        # Fallback if average reward is significantly negative
        if len(rewards) >= 40:
            baseline = np.mean(rewards[:20])
            std = np.std(rewards[:20]) or 1.0
            if mean_r < baseline - 2 * std:
                return True
        return False

    def _ppo_update(self, transitions: List[Transition]):
        """Run PPO update on completed transitions."""
        obs_arr = np.array([t.obs for t in transitions])
        actions = torch.LongTensor([t.action for t in transitions])
        old_log_probs = torch.FloatTensor([t.log_prob for t in transitions])
        rewards = torch.FloatTensor([t.reward for t in transitions])
        old_values = torch.FloatTensor([t.value for t in transitions])

        # Normalize observations
        for o in obs_arr:
            self.normalizer.update(o)
        norm_obs = np.array([self.normalizer.normalize(o) for o in obs_arr])
        obs_t = torch.FloatTensor(norm_obs)

        # Build action masks
        masks = []
        for t in transitions:
            mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
            valid = ENTRY_ACTIONS if t.context == "entry" else EXIT_ACTIONS
            for a in valid:
                mask[a] = True
            masks.append(mask)
        mask_t = torch.stack(masks)

        # Compute returns and advantages (all done=True, so returns = rewards)
        returns = rewards
        advantages = returns - old_values
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        n = len(transitions)
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
                nn.utils.clip_grad_norm_(self.network.parameters(),
                                         self.MAX_GRAD_NORM)
                self.optimizer.step()

        logger.info(f"PPO update done: loss={loss.item():.4f}, "
                    f"avg_reward={rewards.mean():.4f}")

    # ── Internal helpers ─────────────────────────────────────────────

    def _sample_action(self, obs: np.ndarray,
                       valid_actions: list) -> tuple:
        """Sample action from policy network."""
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
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def _get_log_prob_value(self, obs: np.ndarray, action: int,
                            valid_actions: list) -> tuple:
        """Get log_prob and value for a specific action (used for rule-based)."""
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
            log_prob = dist.log_prob(torch.tensor(action))

        return log_prob.item(), value.item()

    def _store_pending_entry(self, pair_key: str, obs: np.ndarray,
                             action: int, log_prob: float, value: float):
        """Store an entry decision that's waiting for position close."""
        if action == ACTION_PASS:
            # PASS is immediate, no pending
            self.buffer.append(Transition(
                obs=obs, action=action, log_prob=log_prob, value=value,
                reward=0.0, done=True, context="entry",
            ))
        else:
            self.pending_entries[pair_key] = Transition(
                obs=obs, action=action, log_prob=log_prob, value=value,
                context="entry",
            )

    # ── Persistence ──────────────────────────────────────────────────

    def save(self):
        """Save model weights and normalizer state."""
        path = self.model_path
        torch.save(self.network.state_dict(), f'{path}.pt')
        torch.save(self.optimizer.state_dict(), f'{path}_optimizer.pt')
        meta = {
            'normalizer': self.normalizer.state_dict(),
            'total_decisions': self.total_decisions,
            'train_updates': self.train_updates,
            'num_closed': self.num_closed,
        }
        with open(f'{path}_meta.json', 'w') as f:
            json.dump(meta, f)
        logger.debug(f"RL model saved to {path}.pt")

    def _try_load(self):
        """Load model if available."""
        path = self.model_path
        if not os.path.exists(f'{path}.pt'):
            return
        try:
            self.network.load_state_dict(
                torch.load(f'{path}.pt', weights_only=True,
                           map_location='cpu'))
            if os.path.exists(f'{path}_optimizer.pt'):
                self.optimizer.load_state_dict(
                    torch.load(f'{path}_optimizer.pt', weights_only=True,
                               map_location='cpu'))
            if os.path.exists(f'{path}_meta.json'):
                with open(f'{path}_meta.json') as f:
                    meta = json.load(f)
                self.normalizer.load_state_dict(meta['normalizer'])
                self.total_decisions = meta.get('total_decisions', 0)
                self.train_updates = meta.get('train_updates', 0)
                # Use DB count, not saved count (more accurate after restarts)
            logger.info(f"RL model loaded from {path}.pt "
                        f"(updates={self.train_updates})")
        except Exception as e:
            logger.warning(f"Failed to load RL model: {e}. Starting fresh.")

    def _save_pending_to_db(self):
        """Persist pending entries for crash recovery."""
        try:
            for pair_key, t in self.pending_entries.items():
                self.db.save_rl_experience(
                    pair_key=pair_key,
                    observation_json=json.dumps(t.obs.tolist()),
                    action=t.action,
                    log_prob=t.log_prob,
                    value=t.value,
                    context=t.context,
                )
        except Exception as e:
            logger.debug(f"Failed to save pending RL experiences: {e}")

    def load_pending_from_db(self, open_pair_keys: set):
        """Restore pending entry decisions for open positions."""
        try:
            rows = self.db.load_rl_experiences()
            for row in rows:
                pair_key = row['pair_key']
                if pair_key in open_pair_keys:
                    self.pending_entries[pair_key] = Transition(
                        obs=np.array(json.loads(row['observation_json']),
                                     dtype=np.float32),
                        action=row['action'],
                        log_prob=row['log_prob'],
                        value=row['value'],
                        context=row['context'],
                    )
            if self.pending_entries:
                logger.info(f"Recovered {len(self.pending_entries)} pending "
                            f"RL entry decisions from DB")
        except Exception as e:
            logger.debug(f"Failed to load pending RL experiences: {e}")

    # ── Status ───────────────────────────────────────────────────────

    def status_str(self) -> str:
        phase = self._phase()
        avg_r = (f"{np.mean(list(self.recent_rewards)):+.3f}"
                 if self.recent_rewards else "n/a")
        pending = len(self.pending_entries)
        buf = len([t for t in self.buffer if t.done])
        return (f"RL phase={phase} | decisions={self.total_decisions} | "
                f"updates={self.train_updates} | pending={pending} | "
                f"buf={buf}/{self.MIN_BUFFER_SIZE} | avg_r={avg_r}")
