#!/usr/bin/env python3
"""
Pre-train the RL agent from historical closed positions.

Synthesizes 24-dim observation vectors from DB records and runs PPO updates
with hindsight-labeled actions and realized PnL rewards.

Usage:
    python3 pretrain_rl.py --db exp_ll.db \
        --scanner-db ../arbitrage_tracker/arb_tracker.db \
        --model-path rl_model_pretrained --epochs 10
"""

import argparse
import json
import logging
import math
import os
import sqlite3
import sys

import numpy as np
import torch

from rl_agent import (
    ACTION_ENTER_NORMAL,
    ACTION_PASS,
    ENTRY_ACTIONS,
    NUM_ACTIONS,
    OBS_DIM,
    PolicyNetwork,
    RunningNormalizer,
    Transition,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# PPO hyperparameters (match rl_agent.py defaults)
GAMMA = 0.99
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
PPO_EPOCHS = 4
BATCH_SIZE = 32
LR = 3e-4
MAX_GRAD_NORM = 0.5


def load_closed_positions(db_path: str) -> list:
    """Load all closed positions from the experiment DB."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, pair_key, basket_size, mints_json, direction,
               entry_time, exit_time, entry_zscore, exit_zscore,
               entry_values_json, realized_pnl, status
        FROM positions
        WHERE status IN ('closed', 'stopped_out')
        ORDER BY entry_time ASC
    """).fetchall()
    conn.close()
    logger.info(f"Loaded {len(rows)} closed positions from {db_path}")
    return [dict(r) for r in rows]


def load_scanner_data(scanner_db_path: str) -> dict:
    """Load cointegration results keyed by basket_key."""
    try:
        conn = sqlite3.connect(f"file:{scanner_db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        logger.warning(f"Cannot open scanner DB: {scanner_db_path}")
        return {}

    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT basket_key, spread_std, half_life, eg_p_value
            FROM cointegration_results
        """).fetchall()
    except Exception as e:
        logger.warning(f"Error reading scanner DB: {e}")
        return {}
    finally:
        conn.close()

    result = {}
    for r in rows:
        result[r['basket_key']] = {
            'spread_std': r['spread_std'] or 0.0,
            'half_life': r['half_life'] or 500.0,
            'eg_p_value': r['eg_p_value'] or 0.05,
        }
    logger.info(f"Loaded {len(result)} cointegration results from scanner DB")
    return result


def build_observation(pos: dict, idx: int, total: int,
                      cumulative_pnl: float, capital: float,
                      scanner_data: dict, recent_trades: list,
                      token_slippage: dict = None) -> np.ndarray:
    """Reconstruct a 24-dim observation vector from a closed position."""
    # Look up scanner data for this pair
    pair_key = pos['pair_key']
    scan = scanner_data.get(pair_key, {})
    spread_std = scan.get('spread_std', 0.01)
    half_life = scan.get('half_life', 500.0)
    p_value = scan.get('eg_p_value', 0.05)

    z = pos['entry_zscore'] or 0.0
    max_half_life_secs = 1800.0
    hl_norm = half_life * 0.4 / max_half_life_secs

    # Signal features (7)
    signal_feats = [
        z,                   # zscore
        abs(z),              # abs_zscore
        spread_std,          # spread_std
        hl_norm,             # half_life normalized
        1.0,                 # buffer_full (assume full for historical)
        p_value,             # p_value
        0.0,                 # time_since_resample (entry moment)
    ]

    # Position features (5) - all zeros at entry decision time
    position_feats = [0.0, 0.0, 0.0, 0.0, 0.0]

    # Portfolio features (5)
    max_positions = 10
    pos_ratio = min(idx, max_positions) / max_positions
    entry_values = json.loads(pos['entry_values_json']) if pos['entry_values_json'] else []
    total_entry = sum(entry_values) if entry_values else 0.0
    exp_ratio = min(total_entry / max(capital, 1.0), 1.0)
    port_return = cumulative_pnl / max(capital, 1.0)
    portfolio_feats = [
        pos_ratio,       # position ratio
        exp_ratio,       # exposure ratio
        0.0,             # drawdown (approximate as 0)
        port_return,     # portfolio return
        0.5,             # rate_usage (middle)
    ]

    # Performance features (4) - rolling stats from prior trades
    if recent_trades:
        wins = sum(1 for t in recent_trades if t['realized_pnl'] > 0)
        win_rate = wins / len(recent_trades)
        pnls = [t['realized_pnl'] for t in recent_trades]
        avg_pnl = np.mean(pnls)
        avg_entry_val = np.mean([
            sum(json.loads(t['entry_values_json'])) if t['entry_values_json'] else 1.0
            for t in recent_trades
        ]) or 1.0
        avg_pnl_norm = avg_pnl / max(avg_entry_val, 0.01)
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

    # Time-of-day sin (use entry_time)
    entry_time = pos['entry_time'] or 0
    hour = (entry_time % 86400) / 3600
    tod_sin = math.sin(2 * math.pi * hour / 24)

    performance_feats = [win_rate, avg_pnl_norm, sharpe, tod_sin]

    # Slippage features (3) — use per-token slippage if available
    basket_size = pos['basket_size'] or 2
    mints = json.loads(pos['mints_json']) if pos.get('mints_json') else []
    sol_mint = "So11111111111111111111111111111111111111112"
    if token_slippage and mints:
        rts = [token_slippage.get(m, 50.0) for m in mints
               if m != sol_mint]
        worst_rt = max(rts) / 100.0 if rts else 0.03
        avg_rt = (sum(rts) / len(rts) / 100.0) if rts else 0.02
    else:
        worst_rt = 0.03
        avg_rt = 0.02
    slippage_feats = [
        worst_rt,
        avg_rt,
        basket_size / 4.0,
    ]

    obs = np.array(
        signal_feats + position_feats + portfolio_feats
        + performance_feats + slippage_feats,
        dtype=np.float32,
    )
    obs = np.clip(obs, -10.0, 10.0)
    assert len(obs) == OBS_DIM, f"Expected {OBS_DIM} dims, got {len(obs)}"
    return obs


def load_token_slippage(db_path: str) -> dict:
    """Compute average round-trip slippage per token from execution_log."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("""
            SELECT token_mint, AVG(ABS(slippage_bps)) as avg_slip
            FROM execution_log
            WHERE slippage_bps IS NOT NULL
            GROUP BY token_mint
        """).fetchall()
    except Exception:
        return {}
    finally:
        conn.close()
    # Round-trip = 2x per-leg average
    return {mint: avg_slip * 2 for mint, avg_slip in rows}


def assign_action(pnl: float) -> int:
    """Hindsight labeling: good trades get ENTER_NORMAL, bad ones get PASS."""
    if pnl < -1.0:
        return ACTION_PASS
    return ACTION_ENTER_NORMAL


def compute_reward(pnl: float, entry_time: int, exit_time: int) -> float:
    """Reward = realized_pnl / duration_hours."""
    duration_secs = (exit_time or 0) - (entry_time or 0)
    duration_hours = max(duration_secs / 3600.0, 0.01)
    return pnl / duration_hours


def ppo_update(network, optimizer, normalizer, transitions):
    """Run one PPO update pass over transitions."""
    import torch.nn.functional as F

    obs_arr = np.array([t.obs for t in transitions])
    actions = torch.LongTensor([t.action for t in transitions])
    old_log_probs = torch.FloatTensor([t.log_prob for t in transitions])
    rewards = torch.FloatTensor([t.reward for t in transitions])
    old_values = torch.FloatTensor([t.value for t in transitions])

    # Normalize observations
    for o in obs_arr:
        normalizer.update(o)
    norm_obs = np.array([normalizer.normalize(o) for o in obs_arr])
    obs_t = torch.FloatTensor(norm_obs)

    # Build action masks (all entry context)
    mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
    for a in ENTRY_ACTIONS:
        mask[a] = True
    mask_t = mask.unsqueeze(0).expand(len(transitions), -1)

    # Returns = rewards (all done=True, single-step)
    returns = rewards
    advantages = returns - old_values
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(transitions)
    total_loss = 0.0
    num_batches = 0

    for _ in range(PPO_EPOCHS):
        indices = np.random.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            idx = indices[start:end]

            b_obs = obs_t[idx]
            b_actions = actions[idx]
            b_old_lp = old_log_probs[idx]
            b_adv = advantages[idx]
            b_ret = returns[idx]
            b_mask = mask_t[idx]

            logits, values = network(b_obs, action_mask=b_mask)
            dist = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(b_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_lp - b_old_lp)
            clipped = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
            policy_loss = -torch.min(ratio * b_adv, clipped * b_adv).mean()
            value_loss = F.mse_loss(values.squeeze(-1), b_ret)
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train RL agent from historical closed positions")
    parser.add_argument("--db", required=True, nargs="+",
                        help="Path(s) to experiment DB(s) with closed positions")
    parser.add_argument("--scanner-db",
                        default="../arbitrage_tracker/arb_tracker.db",
                        help="Path to scanner cointegration DB")
    parser.add_argument("--model-path", default="rl_model_pretrained",
                        help="Output model path prefix")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs over the dataset")
    parser.add_argument("--capital", type=float, default=1000.0,
                        help="Initial capital for portfolio feature approximation")
    parser.add_argument("--hidden", type=int, default=64,
                        help="Hidden layer size")
    parser.add_argument("--lr", type=float, default=LR,
                        help="Learning rate")
    args = parser.parse_args()

    # Load data from all DBs
    positions = []
    token_slippage = {}
    for db_path in args.db:
        positions.extend(load_closed_positions(db_path))
        ts = load_token_slippage(db_path)
        for mint, slip in ts.items():
            # Keep the max slippage seen across DBs
            token_slippage[mint] = max(token_slippage.get(mint, 0), slip)
    positions.sort(key=lambda p: p['entry_time'] or 0)

    if not positions:
        logger.error("No closed positions found. Nothing to train on.")
        sys.exit(1)

    logger.info(f"Total: {len(positions)} positions from {len(args.db)} DB(s), "
                f"slippage data for {len(token_slippage)} tokens")
    scanner_data = load_scanner_data(args.scanner_db)

    # Initialize network
    network = PolicyNetwork(OBS_DIM, NUM_ACTIONS, hidden=args.hidden)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    normalizer = RunningNormalizer(OBS_DIM)

    # Try to load existing model as starting point
    if os.path.exists(f"{args.model_path}.pt"):
        try:
            network.load_state_dict(
                torch.load(f"{args.model_path}.pt", weights_only=True,
                           map_location="cpu"))
            logger.info(f"Loaded existing model from {args.model_path}.pt")
        except Exception as e:
            logger.warning(f"Could not load existing model: {e}")

    # Build transitions from all closed positions
    logger.info("Building observation vectors...")
    transitions = []
    cumulative_pnl = 0.0

    for i, pos in enumerate(positions):
        pnl = pos['realized_pnl'] or 0.0

        # Rolling window of prior 20 trades for performance features
        start_idx = max(0, i - 20)
        recent_trades = positions[start_idx:i]

        obs = build_observation(
            pos, idx=i, total=len(positions),
            cumulative_pnl=cumulative_pnl, capital=args.capital,
            scanner_data=scanner_data, recent_trades=recent_trades,
            token_slippage=token_slippage,
        )

        action = assign_action(pnl)
        reward = compute_reward(pnl, pos['entry_time'], pos['exit_time'])

        # Get log_prob and value from current network
        normalizer.update(obs)
        norm_obs = normalizer.normalize(obs)
        mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
        for a in ENTRY_ACTIONS:
            mask[a] = True

        with torch.no_grad():
            x = torch.FloatTensor(norm_obs).unsqueeze(0)
            m = mask.unsqueeze(0)
            logits, value = network(x, action_mask=m)
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(torch.tensor(action))

        transitions.append(Transition(
            obs=obs,
            action=action,
            log_prob=log_prob.item(),
            value=value.item(),
            reward=reward,
            done=True,
            context="entry",
        ))

        cumulative_pnl += pnl

    n_enter = sum(1 for t in transitions if t.action == ACTION_ENTER_NORMAL)
    n_pass = sum(1 for t in transitions if t.action == ACTION_PASS)
    rewards = [t.reward for t in transitions]
    logger.info(f"Built {len(transitions)} transitions: "
                f"{n_enter} ENTER_NORMAL, {n_pass} PASS")
    logger.info(f"Reward stats: mean={np.mean(rewards):.4f}, "
                f"std={np.std(rewards):.4f}, "
                f"min={np.min(rewards):.4f}, max={np.max(rewards):.4f}")

    # Training epochs
    for epoch in range(args.epochs):
        # Recompute log_probs and values with current network each epoch
        updated = []
        for t in transitions:
            normalizer_copy = normalizer  # normalizer already fitted
            norm_obs = normalizer_copy.normalize(t.obs)
            mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
            for a in ENTRY_ACTIONS:
                mask[a] = True
            with torch.no_grad():
                x = torch.FloatTensor(norm_obs).unsqueeze(0)
                m = mask.unsqueeze(0)
                logits, value = network(x, action_mask=m)
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(torch.tensor(t.action))
            updated.append(Transition(
                obs=t.obs, action=t.action,
                log_prob=log_prob.item(), value=value.item(),
                reward=t.reward, done=True, context="entry",
            ))

        avg_loss = ppo_update(network, optimizer, normalizer, updated)

        # Check policy distribution after update
        with torch.no_grad():
            sample_obs = normalizer.normalize(transitions[0].obs)
            mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
            for a in ENTRY_ACTIONS:
                mask[a] = True
            logits, _ = network(
                torch.FloatTensor(sample_obs).unsqueeze(0),
                action_mask=mask.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze()
            prob_str = " ".join(
                f"a{a}={probs[a]:.3f}" for a in ENTRY_ACTIONS)

        logger.info(f"Epoch {epoch + 1}/{args.epochs}: "
                    f"avg_loss={avg_loss:.4f} | probs=[{prob_str}]")

    # Save model
    torch.save(network.state_dict(), f"{args.model_path}.pt")
    torch.save(optimizer.state_dict(), f"{args.model_path}_optimizer.pt")
    meta = {
        'normalizer': normalizer.state_dict(),
        'total_decisions': len(transitions),
        'train_updates': args.epochs,
        'num_closed': len(transitions),
    }
    with open(f"{args.model_path}_meta.json", "w") as f:
        json.dump(meta, f)

    logger.info(f"Model saved to {args.model_path}.pt")
    logger.info(f"Pre-training complete: {len(transitions)} positions, "
                f"{args.epochs} epochs, cumulative PnL=${cumulative_pnl:.2f}")


if __name__ == "__main__":
    main()
