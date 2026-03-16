#!/usr/bin/env python3
"""Measure real Jupiter slippage by comparing quote prices to mid-market prices.

For each token, gets Jupiter price (mid-market) then fetches quotes at
various trade sizes to measure actual execution slippage.

Usage:
    python3 measure_slippage.py
"""

import json
import os
import time
import urllib.request
from dotenv import load_dotenv

load_dotenv()

SOL_MINT = "So11111111111111111111111111111111111111112"
JUPITER_API_KEY = os.getenv("JUPITER_API_KEY", "")
JUPITER_URL = os.getenv("JUPITER_QUOTE_URL", "https://api.jup.ag").rstrip("/")

# Tokens we trade in 2-token pairs (from scanner results)
TOKENS = {
    "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj": ("stSOL", 9),
    "J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn": ("jitoSOL", 9),
    "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So": ("mSOL", 9),
    "bSo13r4TkiE4KumL71LsHTPpL2euBYLFx6h9HP3piy1": ("bSOL", 9),
    "jupSoLaHXQiZZTSfEWMTRRgpnyFm8f6sZdosWBjx93v": ("jupSOL", 9),
    "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm": ("WIF", 6),
    "7GCihgDB8fe6LNa32gd7QZIk2sg3R4bfETkfso6nxXvf": ("FARTCOIN", 6),  # corrected decimals if needed
    "MEW1gQWJ3nEXg2qgERiKu7FAFj79PHvQVREQUzScPP5": ("MEW", 5),
    "rndrizKT3MK1iimdxRdWabcF7Zg7AR5T4nud4EkHBof": ("RENDER", 8),
    "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs": ("RAY", 6),
    "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3": ("PYTH", 6),
    "jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL": ("JTO", 9),
    "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263": ("BONK", 5),
}

# Trade sizes in USD to test
TRADE_SIZES_USD = [10, 25, 50, 100, 250, 500]


def get_jupiter_prices(mints):
    """Get mid-market prices from Jupiter Price API v3."""
    ids = ",".join(mints)
    url = f"https://api.jup.ag/price/v3?ids={ids}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "statalyzer/1.0",
    }
    if JUPITER_API_KEY:
        headers["x-api-key"] = JUPITER_API_KEY
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    prices = {}
    for mint in mints:
        entry = data.get(mint)
        if entry and entry.get("usdPrice"):
            prices[mint] = float(entry["usdPrice"])
    return prices


def get_quote(input_mint, output_mint, amount_raw):
    """Get Jupiter quote for a swap."""
    params = (
        f"?inputMint={input_mint}"
        f"&outputMint={output_mint}"
        f"&amount={amount_raw}"
        f"&slippageBps=1000"  # high tolerance so quote isn't rejected
    )
    url = f"{JUPITER_URL}/swap/v1/quote{params}"
    headers = {"Accept": "application/json"}
    if JUPITER_API_KEY:
        headers["x-api-key"] = JUPITER_API_KEY
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return None


def measure_slippage():
    mints = list(TOKENS.keys())
    print("Fetching mid-market prices...")
    prices = get_jupiter_prices(mints + [SOL_MINT])
    sol_price = prices.get(SOL_MINT, 88.0)
    print(f"SOL/USD: ${sol_price:.2f}\n")

    print(f"{'Token':<10} {'Price':>10} {'Size':>6} {'QuotePrice':>12} {'Slip(bps)':>10} {'Direction'}")
    print("-" * 70)

    for mint, (symbol, decimals) in TOKENS.items():
        mid_price = prices.get(mint)
        if not mid_price:
            print(f"{symbol:<10} {'no price':>10}")
            continue

        for size_usd in TRADE_SIZES_USD:
            # BUY direction: SOL -> Token
            sol_amount_raw = int(size_usd / sol_price * 1e9)
            expected_tokens = size_usd / mid_price
            expected_raw = int(expected_tokens * (10 ** decimals))

            quote = get_quote(SOL_MINT, mint, sol_amount_raw)
            time.sleep(0.15)  # rate limit

            if quote:
                out_raw = int(quote.get("outAmount", 0))
                out_tokens = out_raw / (10 ** decimals)
                quote_price = size_usd / out_tokens if out_tokens > 0 else 0
                slip_bps = (quote_price - mid_price) / mid_price * 10000 if mid_price > 0 else 0
                print(f"{symbol:<10} ${mid_price:>9.6f} ${size_usd:>4} ${quote_price:>11.6f} {slip_bps:>+9.1f}  BUY")
            else:
                print(f"{symbol:<10} ${mid_price:>9.6f} ${size_usd:>4} {'FAILED':>12} {'':>10}  BUY")

            # SELL direction: Token -> SOL
            token_amount_raw = int(size_usd / mid_price * (10 ** decimals))
            quote = get_quote(mint, SOL_MINT, token_amount_raw)
            time.sleep(0.15)

            if quote:
                out_raw = int(quote.get("outAmount", 0))
                sol_out = out_raw / 1e9
                usd_out = sol_out * sol_price
                quote_price = usd_out / (token_amount_raw / (10 ** decimals)) if token_amount_raw > 0 else 0
                slip_bps = (mid_price - quote_price) / mid_price * 10000 if mid_price > 0 else 0
                print(f"{symbol:<10} ${mid_price:>9.6f} ${size_usd:>4} ${quote_price:>11.6f} {slip_bps:>+9.1f}  SELL")
            else:
                print(f"{symbol:<10} ${mid_price:>9.6f} ${size_usd:>4} {'FAILED':>12} {'':>10}  SELL")

        print()


if __name__ == "__main__":
    measure_slippage()
