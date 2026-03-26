#!/usr/bin/env python3
"""
Supervised profitability classifier for trade entry filtering.

Trains a gradient-boosted classifier on historical closed positions to predict
whether a trade will be profitable. Used as a pre-filter before RL sizing.

Features are derived from the same observation vector as the RL agent, plus
per-token slippage data from the execution log.

Usage:
    # Train and save model
    python3 trade_classifier.py --db exp_at.db --scanner-db ../arbitrage_tracker/arb_tracker.db \
        --model-path trade_clf.pkl

    # Train on multiple DBs
    python3 trade_classifier.py --db exp_at.db exp_ll.db --model-path trade_clf.pkl
"""

import argparse
import json
import logging
import math
import os
import pickle
import sqlite3
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Feature names for interpretability
FEATURE_NAMES = [
    "zscore", "abs_zscore", "spread_std", "half_life_norm", "p_value",
    "basket_size", "worst_rt_bps", "avg_rt_bps",
    "edge_bps",
    "hour_sin", "hour_cos",
]

NUM_FEATURES = len(FEATURE_NAMES)


class TradeClassifier:
    """Profitability classifier — supports sklearn (.pkl) and PyTorch (.pt) models."""

    def __init__(self, model_path: str = "trade_clf.pkl"):
        self.model_path = model_path
        self.model = None
        self.model_type = None  # "sklearn" or "pytorch"
        self._dl_model = None
        self._dl_mu = None
        self._dl_std = None
        self._dl_threshold = 0.5
        self._try_load()

    def _try_load(self):
        if not os.path.exists(self.model_path):
            return
        try:
            if self.model_path.endswith(".pt"):
                self._load_pytorch()
            else:
                self._load_sklearn()
        except Exception as e:
            logger.warning(f"Failed to load classifier: {e}")

    def _load_sklearn(self):
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        self.model_type = "sklearn"
        logger.info(f"Trade classifier (sklearn) loaded from {self.model_path}")

    def _load_pytorch(self):
        import torch
        from train_dl_classifier import ProfitNet
        data = torch.load(self.model_path, weights_only=False, map_location="cpu")
        self._dl_model = ProfitNet(data["dim"], hidden=data["hidden"])
        self._dl_model.load_state_dict(data["model_state"])
        self._dl_model.eval()
        self._dl_mu = data["mu"]
        self._dl_std = data["std"]
        self._dl_threshold = data.get("threshold", 0.5)
        self.model_type = "pytorch"
        self.model = True  # flag as loaded
        logger.info(f"Trade classifier (DL) loaded from {self.model_path} "
                    f"(threshold={self._dl_threshold})")

    def is_ready(self) -> bool:
        return self.model is not None

    def predict_profitable(self, features: np.ndarray) -> bool:
        if self.model is None:
            return True
        return self.predict_proba(features) >= self._dl_threshold

    def predict_proba(self, features: np.ndarray) -> float:
        if self.model is None:
            return 0.5
        if self.model_type == "pytorch":
            return self._predict_pytorch(features)
        return self.model.predict_proba(features.reshape(1, -1))[0, 1]

    def _predict_pytorch(self, features: np.ndarray) -> float:
        import torch
        x = (features - self._dl_mu) / self._dl_std
        with torch.no_grad():
            logit = self._dl_model(torch.FloatTensor(x).unsqueeze(0))
            return torch.sigmoid(logit).item()

    def build_features(self, signal, basket_state, portfolio=None,
                       slippage_monitor=None,
                       token_slippage_map: Dict[str, float] = None) -> np.ndarray:
        """Build feature vector from live signal data."""
        z = signal.zscore if signal else 0.0
        spread_std = basket_state.spread_std if basket_state else 0.01
        half_life = basket_state.half_life if basket_state else 500.0
        p_value = basket_state.eg_p_value if basket_state else 0.05
        basket_size = signal.basket_size if signal else 2

        hl_norm = half_life * 0.4 / 1800.0
        edge_bps = abs(z) * spread_std * 10000

        # Slippage features
        sol_mint = "So11111111111111111111111111111111111111112"
        worst_rt = 0.0
        avg_rt = 0.0
        mints = signal.mints if signal else []
        if slippage_monitor:
            from constants import STABLECOIN_MINTS
            rts = []
            for m in mints:
                if m in STABLECOIN_MINTS or m == sol_mint:
                    continue
                rt = slippage_monitor.get_slippage_at_size(m, 500)
                if rt is not None:
                    rts.append(rt)
            if rts:
                worst_rt = max(rts) / 100.0
                avg_rt = sum(rts) / len(rts) / 100.0
        elif token_slippage_map:
            rts = [token_slippage_map.get(m, 50.0) for m in mints if m != sol_mint]
            if rts:
                worst_rt = max(rts) / 100.0
                avg_rt = sum(rts) / len(rts) / 100.0

        # Derived features
        min_rt = worst_rt  # approximate if no per-token data
        if slippage_monitor:
            rts_list = []
            for m in mints:
                if m in STABLECOIN_MINTS or m == sol_mint:
                    continue
                rt = slippage_monitor.get_slippage_at_size(m, 500)
                if rt is not None:
                    rts_list.append(rt / 100.0)
            if rts_list:
                min_rt = min(rts_list)
        elif token_slippage_map:
            rts_list = [token_slippage_map.get(m, 50.0) / 100.0 for m in mints if m != sol_mint]
            if rts_list:
                min_rt = min(rts_list)

        net_edge = edge_bps - (worst_rt * 100)
        slip_to_edge = (worst_rt * 100) / max(edge_bps, 0.01)
        d_long = 1.0 if (signal and signal.signal_type.value == "entry_long") else 0.0
        entry_value_k = 0.0

        import time
        hour = (time.time() % 86400) / 3600
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)

        features = np.array([
            z, abs(z), spread_std, hl_norm, p_value,
            basket_size, basket_size == 2, basket_size == 3, basket_size == 4,
            worst_rt, avg_rt, min_rt,
            edge_bps, net_edge, slip_to_edge,
            entry_value_k, d_long,
            hour_sin, hour_cos,
        ], dtype=np.float32)

        return features


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
    return {mint: avg_slip * 2 for mint, avg_slip in rows}


def load_scanner_data(scanner_db_path: str) -> dict:
    try:
        conn = sqlite3.connect(f"file:{scanner_db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return {}
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT basket_key, spread_std, half_life, eg_p_value
            FROM cointegration_results
        """).fetchall()
    except Exception:
        return {}
    finally:
        conn.close()
    return {r['basket_key']: {
        'spread_std': r['spread_std'] or 0.01,
        'half_life': r['half_life'] or 500.0,
        'eg_p_value': r['eg_p_value'] or 0.05,
    } for r in rows}


def build_training_data(db_paths: List[str], scanner_db_path: str):
    """Build feature matrix and labels from closed positions."""
    scanner_data = load_scanner_data(scanner_db_path)
    logger.info(f"Loaded {len(scanner_data)} cointegration results")

    all_features = []
    all_labels = []
    all_pnls = []

    for db_path in db_paths:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        positions = conn.execute("""
            SELECT pair_key, basket_size, mints_json, direction,
                   entry_time, exit_time, entry_zscore,
                   entry_values_json, realized_pnl, status
            FROM positions
            WHERE status IN ('closed', 'stopped_out')
            ORDER BY entry_time ASC
        """).fetchall()
        positions = [dict(r) for r in positions]
        conn.close()

        token_slippage = load_token_slippage(db_path)
        logger.info(f"Loaded {len(positions)} positions from {db_path}, "
                     f"slippage for {len(token_slippage)} tokens")

        sol_mint = "So11111111111111111111111111111111111111112"
        cumulative_pnl = 0.0
        capital = 10000.0

        for i, pos in enumerate(positions):
            pnl = pos['realized_pnl'] or 0.0
            scan = scanner_data.get(pos['pair_key'], {})
            spread_std = scan.get('spread_std', 0.01)
            half_life = scan.get('half_life', 500.0)
            p_value = scan.get('eg_p_value', 0.05)

            z = pos['entry_zscore'] or 0.0
            hl_norm = half_life * 0.4 / 1800.0
            edge_bps = abs(z) * spread_std * 10000
            basket_size = pos['basket_size'] or 2

            mints = json.loads(pos['mints_json']) if pos.get('mints_json') else []
            rts = [token_slippage.get(m, 50.0) for m in mints if m != sol_mint]
            worst_rt = max(rts) / 100.0 if rts else 0.5
            avg_rt = (sum(rts) / len(rts) / 100.0) if rts else 0.3

            entry_time = pos['entry_time'] or 0
            hour = (entry_time % 86400) / 3600
            hour_sin = math.sin(2 * math.pi * hour / 24)
            hour_cos = math.cos(2 * math.pi * hour / 24)

            features = np.array([
                z, abs(z), spread_std, hl_norm, p_value,
                basket_size, worst_rt, avg_rt,
                edge_bps,
                hour_sin, hour_cos,
            ], dtype=np.float32)

            all_features.append(features)
            all_labels.append(1 if pnl > 0 else 0)
            all_pnls.append(pnl)
            cumulative_pnl += pnl

    X = np.array(all_features)
    y = np.array(all_labels)
    pnls = np.array(all_pnls)
    return X, y, pnls


def main():
    parser = argparse.ArgumentParser(
        description="Train profitability classifier from historical trades")
    parser.add_argument("--db", required=True, nargs="+",
                        help="Path(s) to experiment DB(s)")
    parser.add_argument("--scanner-db",
                        default="../arbitrage_tracker/arb_tracker.db")
    parser.add_argument("--model-path", default="trade_clf.pkl")
    args = parser.parse_args()

    X, y, pnls = build_training_data(args.db, args.scanner_db)
    logger.info(f"Training data: {len(X)} samples, {y.sum()} profitable ({100*y.mean():.0f}%)")

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
    )

    # Cross-validation
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    logger.info(f"Cross-val accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    # Profit-based evaluation: simulate filtering
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    filtered_pnl = 0.0
    filtered_trades = 0
    unfiltered_pnl = pnls.sum()
    for train_idx, test_idx in kf.split(X):
        clf_fold = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=10, subsample=0.8, random_state=42)
        clf_fold.fit(X[train_idx], y[train_idx])
        preds = clf_fold.predict(X[test_idx])
        for j, idx in enumerate(test_idx):
            if preds[j] == 1:  # classifier says enter
                filtered_pnl += pnls[idx]
                filtered_trades += 1

    logger.info(f"Unfiltered: {len(X)} trades, PnL=${unfiltered_pnl:+.2f}, "
                f"Avg=${unfiltered_pnl/len(X):+.3f}")
    logger.info(f"Filtered:   {filtered_trades} trades, PnL=${filtered_pnl:+.2f}, "
                f"Avg=${filtered_pnl/filtered_trades:+.3f}" if filtered_trades else "No trades passed filter")

    # Train final model on all data
    clf.fit(X, y)

    # Feature importance
    logger.info("Feature importance:")
    importances = sorted(zip(FEATURE_NAMES, clf.feature_importances_),
                         key=lambda x: -x[1])
    for name, imp in importances:
        if imp > 0.01:
            logger.info(f"  {name:<20s} {imp:.3f}")

    # Save
    with open(args.model_path, "wb") as f:
        pickle.dump(clf, f)
    logger.info(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    main()
