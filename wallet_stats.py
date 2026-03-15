#!/usr/bin/env python3
"""Query on-chain wallet stats for the live trading wallet."""

import json
import os
import time
import urllib.request

from dotenv import load_dotenv
load_dotenv()

WALLET = "2vesMyZ8TDqJm1NF1LoWeEVsPVknXRXVMtrmhCQSRhs7"
SOL_MINT = "So11111111111111111111111111111111111111112"


def rpc(method, params):
    endpoint = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(endpoint, data=payload,
                                headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def get_sol_price():
    api_key = os.getenv("JUPITER_API_KEY", "")
    url = f"https://api.jup.ag/price/v3?ids={SOL_MINT}"
    headers = {"Accept": "application/json", "User-Agent": "statalyzer/1.0"}
    if api_key:
        headers["x-api-key"] = api_key
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    return float(data[SOL_MINT]["usdPrice"])


def main():
    # Current SOL balance
    result = rpc("getBalance", [WALLET])
    sol_balance = result["result"]["value"] / 1e9
    sol_usd = get_sol_price()

    # Get all transactions
    all_sigs = []
    before = None
    while True:
        params = [WALLET, {"limit": 1000}]
        if before:
            params[1]["before"] = before
        result = rpc("getSignaturesForAddress", params)
        batch = result["result"]
        if not batch:
            break
        all_sigs.extend(batch)
        if len(batch) < 1000:
            break
        before = batch[-1]["signature"]

    ok_count = sum(1 for s in all_sigs if not s.get("err"))
    fail_count = sum(1 for s in all_sigs if s.get("err"))

    # Get first transaction to find initial deposit
    oldest_sig = all_sigs[-1]["signature"]
    tx_result = rpc("getTransaction", [oldest_sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}])
    tx = tx_result.get("result", {})
    meta = tx.get("meta", {})
    pre = meta.get("preBalances", [])
    post = meta.get("postBalances", [])
    first_time = tx.get("blockTime", 0)

    # Account index 0 is the fee payer. For a transfer TO this wallet,
    # we need to find the wallet's index in the account keys
    msg = tx.get("transaction", {}).get("message", {})
    account_keys = msg.get("accountKeys", [])
    wallet_idx = None
    for i, ak in enumerate(account_keys):
        key = ak.get("pubkey", ak) if isinstance(ak, dict) else ak
        if key == WALLET:
            wallet_idx = i
            break

    initial_deposit = 0
    if wallet_idx is not None and wallet_idx < len(pre) and wallet_idx < len(post):
        initial_deposit = (post[wallet_idx] - pre[wallet_idx]) / 1e9

    # Sum all SOL changes across all successful transactions
    # But that's too many RPC calls. Instead, use: current_balance = initial_deposit - fees_spent + net_transfers
    # Simplest: initial deposit vs current balance
    newest_time = all_sigs[0].get("blockTime", 0) if all_sigs else 0
    duration_s = newest_time - first_time if first_time else 0
    duration_h = duration_s / 3600

    # Get token account values
    tok_result = rpc("getTokenAccountsByOwner", [
        WALLET,
        {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
        {"encoding": "jsonParsed"},
    ])
    token_mints = []
    for item in tok_result["result"]["value"]:
        info = item["account"]["data"]["parsed"]["info"]
        amt = float(info["tokenAmount"]["uiAmountString"])
        if amt > 0 and info["mint"] != SOL_MINT:
            token_mints.append((info["mint"], amt))

    # Get USD values for remaining tokens
    token_value_sol = 0
    if token_mints:
        api_key = os.getenv("JUPITER_API_KEY", "")
        quote_url = os.getenv("JUPITER_QUOTE_URL", "https://api.jup.ag").rstrip("/")
        for mint, amt_raw_ui in token_mints:
            # Get the raw amount
            for item in tok_result["result"]["value"]:
                info = item["account"]["data"]["parsed"]["info"]
                if info["mint"] == mint:
                    raw_amt = int(info["tokenAmount"]["amount"])
                    break
            try:
                url = (f"{quote_url}/swap/v1/quote"
                       f"?inputMint={mint}&outputMint={SOL_MINT}"
                       f"&amount={raw_amt}&slippageBps=200")
                headers = {"Accept": "application/json"}
                if api_key:
                    headers["x-api-key"] = api_key
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    quote = json.loads(resp.read())
                out_sol = int(quote.get("outAmount", 0)) / 1e9
                token_value_sol += out_sol
            except Exception:
                pass

    total_sol = sol_balance + token_value_sol
    total_usd = total_sol * sol_usd
    initial_usd = initial_deposit * sol_usd  # approximate (SOL price changed)
    pnl_sol = total_sol - initial_deposit
    pnl_usd = pnl_sol * sol_usd

    print(f"LIVE WALLET STATS: {WALLET}")
    print(f"{'=' * 60}")
    print(f"  First tx:         {time.strftime('%Y-%m-%d %H:%M', time.localtime(first_time))}")
    print(f"  Latest tx:        {time.strftime('%Y-%m-%d %H:%M', time.localtime(newest_time))}")
    print(f"  Duration:         {duration_h:.1f} hours")
    print(f"  Transactions:     {len(all_sigs)} total ({ok_count} ok, {fail_count} failed)")
    print(f"  Initial deposit:  {initial_deposit:.6f} SOL")
    print(f"  Current SOL:      {sol_balance:.6f} SOL")
    print(f"  Token value:      {token_value_sol:.6f} SOL ({len(token_mints)} tokens)")
    print(f"  Total value:      {total_sol:.6f} SOL (${total_usd:.2f})")
    print(f"  SOL/USD:          ${sol_usd:.2f}")
    print(f"  P&L (SOL):        {pnl_sol:+.6f} SOL")
    print(f"  P&L (USD@now):    ${pnl_usd:+.2f}")
    if duration_h > 0:
        print(f"  P&L/hr (SOL):     {pnl_sol/duration_h:+.6f} SOL/hr")
        print(f"  P&L/hr (USD):     ${pnl_usd/duration_h:+.2f}/hr")
        print(f"  P&L/day (USD):    ${pnl_usd/duration_h*24:+.2f}/day")
    print(f"  Failed tx fees:   ~{fail_count * 15000 / 1e9:.6f} SOL (${fail_count * 15000 / 1e9 * sol_usd:.2f})")


if __name__ == "__main__":
    main()
