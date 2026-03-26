#!/usr/bin/env python3
"""
Train deep learning profitability classifier and save for deployment.

Usage:
    python3 train_dl_classifier.py --db exp_at.db exp_ll.db exp_rl.db \
        --scanner-db ../arbitrage_tracker/arb_tracker.db \
        --model-path trade_dl.pt --threshold 0.5
"""

import argparse
import json
import math
import logging
import os
import random
import sqlite3
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SOL_MINT = "So11111111111111111111111111111111111111112"

FEATURE_NAMES = [
    "zscore", "abs_zscore", "spread_std", "half_life_norm", "p_value",
    "basket_size", "bs_2", "bs_3", "bs_4",
    "worst_rt", "avg_rt", "min_rt",
    "edge_bps", "net_edge", "slip_to_edge",
    "entry_value_k", "direction_long",
    "hour_sin", "hour_cos",
]


class ProfitNet(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_all_data(db_paths, scanner_db_path):
    # Scanner
    sc = sqlite3.connect(f"file:{scanner_db_path}?mode=ro", uri=True)
    sc.row_factory = sqlite3.Row
    scanner = {}
    for r in sc.execute("SELECT basket_key, spread_std, half_life, eg_p_value FROM cointegration_results").fetchall():
        scanner[r["basket_key"]] = {"spread_std": r["spread_std"] or 0.01, "half_life": r["half_life"] or 500, "p_value": r["eg_p_value"] or 0.05}
    sc.close()

    # Token slippage
    token_slip = {}
    positions = []
    for db_path in db_paths:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        positions.extend([dict(r) for r in conn.execute("""
            SELECT pair_key, basket_size, mints_json, direction,
                   entry_time, exit_time, entry_zscore,
                   entry_values_json, realized_pnl, status
            FROM positions WHERE status IN ('closed', 'stopped_out')
            ORDER BY entry_time""").fetchall()])
        for m, s in conn.execute("SELECT token_mint, AVG(ABS(slippage_bps)) FROM execution_log WHERE slippage_bps IS NOT NULL GROUP BY token_mint").fetchall():
            token_slip[m] = max(token_slip.get(m, 0), s * 2)
        conn.close()

    logger.info(f"Loaded {len(positions)} positions from {len(db_paths)} DBs, {len(scanner)} scanner results")

    def build_features(pos):
        scan = scanner.get(pos["pair_key"], {})
        z = pos["entry_zscore"] or 0
        ss = scan.get("spread_std", 0.01)
        hl = scan.get("half_life", 500)
        pv = scan.get("p_value", 0.05)
        bs = pos["basket_size"] or 2
        mints = json.loads(pos["mints_json"]) if pos.get("mints_json") else []
        rts = [token_slip.get(m, 50) for m in mints if m != SOL_MINT]
        wrt = max(rts) / 100 if rts else 0.5
        art = (sum(rts) / len(rts) / 100) if rts else 0.3
        mrt = min(rts) / 100 if rts else 0.01
        edge = abs(z) * ss * 10000
        net_edge = edge - (wrt * 100)
        slip_to_edge = (wrt * 100) / max(edge, 0.01)
        et = pos["entry_time"] or 0
        hr = (et % 86400) / 3600
        d_long = 1.0 if pos["direction"] == "long" else 0.0
        ev = json.loads(pos["entry_values_json"]) if pos.get("entry_values_json") else []
        total_val = sum(ev) if ev else 0
        return np.array([
            z, abs(z), ss, hl * 0.4 / 1800, pv,
            bs, bs == 2, bs == 3, bs == 4,
            wrt, art, mrt,
            edge, net_edge, slip_to_edge,
            total_val / 1000, d_long,
            math.sin(2 * math.pi * hr / 24), math.cos(2 * math.pi * hr / 24),
        ], dtype=np.float32)

    X = np.array([build_features(p) for p in positions])
    y = np.array([1 if p["realized_pnl"] > 0 else 0 for p in positions])
    pnls = np.array([p["realized_pnl"] for p in positions])
    return X, y, pnls, token_slip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, nargs="+")
    parser.add_argument("--scanner-db", default="../arbitrage_tracker/arb_tracker.db")
    parser.add_argument("--model-path", default="trade_dl.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden", type=int, default=128)
    args = parser.parse_args()

    X, y, pnls, token_slip = load_all_data(args.db, args.scanner_db)
    logger.info(f"Data: {len(X)} samples, {y.sum()} profitable ({100*y.mean():.0f}%)")

    # Synthesize 4x
    np.random.seed(42)
    random.seed(42)
    X_synth, y_synth = [], []
    for i in range(len(X)):
        for _ in range(4):
            noise = np.random.normal(0, 0.03, X[i].shape) * (np.abs(X[i]) + 0.01)
            xn = X[i] + noise
            for j in [5, 6, 7, 8, 16]:  # keep discrete
                xn[j] = X[i][j]
            X_synth.append(xn)
            y_synth.append(y[i])
    X_synth = np.array(X_synth)
    y_synth = np.array(y_synth)

    # Split real data
    idx = np.arange(len(X))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    idx_tr2, idx_va = train_test_split(idx_tr, test_size=0.25, random_state=42, stratify=y[idx_tr])

    synth_tr = []
    for i in idx_tr2:
        for j in range(4):
            synth_tr.append(i * 4 + j)

    X_train = np.vstack([X[idx_tr2], X_synth[synth_tr]])
    y_train = np.concatenate([y[idx_tr2], y_synth[synth_tr]])
    X_val, y_val, pnls_val = X[idx_va], y[idx_va], pnls[idx_va]
    X_test, y_test, pnls_test = X[idx_te], y[idx_te], pnls[idx_te]

    logger.info(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # Normalize
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mu) / std
    X_val_n = (X_val - mu) / std
    X_test_n = (X_test - mu) / std

    # Model
    DIM = X_train.shape[1]
    model = ProfitNet(DIM, hidden=args.hidden)
    pos_weight = torch.tensor([(1 - y_train.mean()) / max(y_train.mean(), 0.01)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    train_dl = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
                          batch_size=64, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    patience = 0

    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_dl:
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(torch.FloatTensor(X_val_n)), torch.FloatTensor(y_val)).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if (epoch + 1) % 25 == 0:
            logger.info(f"Epoch {epoch+1}: val_loss={val_loss:.4f}")

        if patience >= 20:
            logger.info(f"Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)

    # Evaluate
    model.eval()
    for name, Xn, ye, pe in [("Val", X_val_n, y_val, pnls_val), ("Test", X_test_n, y_test, pnls_test)]:
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.FloatTensor(Xn))).numpy()
        try:
            auc = roc_auc_score(ye, probs)
        except:
            auc = 0
        preds = (probs >= args.threshold).astype(int)
        entered = pe[preds == 1]
        logger.info(f"{name}: AUC={auc:.3f} | entered={len(entered)} PnL=${entered.sum():+.2f} "
                    f"avg=${entered.mean():+.3f} | unfiltered=${pe.sum():+.2f}")

    # Save
    save_data = {
        "model_state": best_state,
        "mu": mu,
        "std": std,
        "threshold": args.threshold,
        "dim": DIM,
        "hidden": args.hidden,
        "feature_names": FEATURE_NAMES,
    }
    torch.save(save_data, args.model_path)
    logger.info(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    main()
