#!/usr/bin/env python3
"""
Sell all RAY tokens for SOL, paying transaction fees in RAY (via Jupiter's dynamic fee).

Usage:
    python sell_ray.py                # Dry run
    python sell_ray.py --execute      # Execute the swap
    python sell_ray.py --execute --slippage 300  # 3% slippage
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.stdout.reconfigure(line_buffering=True)

SOL_MINT = "So11111111111111111111111111111111111111112"
RAY_MINT = "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"
RAY_DECIMALS = 6


def rpc_call(endpoint, method, params):
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(endpoint, data=payload,
                                headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def get_ray_balance(rpc_endpoint, wallet):
    """Get RAY token balance for wallet."""
    result = rpc_call(rpc_endpoint, "getTokenAccountsByOwner", [
        wallet,
        {"mint": RAY_MINT},
        {"encoding": "jsonParsed"},
    ])
    for item in result.get("result", {}).get("value", []):
        info = item["account"]["data"]["parsed"]["info"]
        amount_raw = int(info["tokenAmount"]["amount"])
        ui_amount = info["tokenAmount"].get("uiAmount", 0) or 0
        return amount_raw, ui_amount, item["pubkey"]
    return 0, 0.0, None


def get_sol_balance(rpc_endpoint, wallet):
    result = rpc_call(rpc_endpoint, "getBalance", [wallet])
    return result["result"]["value"] / 1e9


def get_quote(quote_url, amount_raw, slippage_bps, api_key=""):
    """Get Jupiter quote for RAY → SOL."""
    url = (f"{quote_url}/swap/v1/quote"
           f"?inputMint={RAY_MINT}&outputMint={SOL_MINT}"
           f"&amount={amount_raw}&slippageBps={slippage_bps}")
    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def get_swap_tx(swap_url, quote, pubkey, api_key=""):
    """Get swap transaction from Jupiter, paying priority fees automatically."""
    body = {
        "quoteResponse": quote,
        "userPublicKey": pubkey,
        "wrapAndUnwrapSol": True,
    }
    payload = json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    req = urllib.request.Request(f"{swap_url}/swap/v1/swap", data=payload,
                                headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
        return data.get("swapTransaction")


def submit_tx(rpc_endpoint, signed_b64):
    result = rpc_call(rpc_endpoint, "sendTransaction", [
        signed_b64, {"encoding": "base64", "skipPreflight": True, "maxRetries": 3},
    ])
    if result.get("error"):
        raise RuntimeError(f"sendTransaction error: {result['error']}")
    return result.get("result", "")


def confirm_tx(rpc_endpoint, signature, timeout_s=30):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            result = rpc_call(rpc_endpoint, "getSignatureStatuses", [
                [signature], {"searchTransactionHistory": False},
            ])
            statuses = result.get("result", {}).get("value", [])
            if statuses and statuses[0]:
                status = statuses[0]
                if status.get("err"):
                    return False, f"on-chain error: {status['err']}"
                conf = status.get("confirmationStatus", "")
                if conf in ("confirmed", "finalized"):
                    return True, conf
        except Exception:
            pass
        time.sleep(1.0)
    return False, "timeout"


def main():
    from dotenv import load_dotenv
    try:
        load_dotenv()
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Sell all RAY for SOL")
    parser.add_argument("--execute", action="store_true", help="Execute the swap (default: dry run)")
    parser.add_argument("--slippage", type=int, default=200, help="Slippage in bps (default: 200 = 2%%)")
    args = parser.parse_args()

    rpc_endpoint = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    quote_url = os.getenv("JUPITER_QUOTE_URL", "https://lite-api.jup.ag").rstrip("/")
    swap_url = quote_url
    api_key = os.getenv("JUPITER_API_KEY", "")
    kp_path = os.getenv("WALLET_KEYPAIR_PATH", "")

    # Load keypair
    keypair = None
    pubkey = None
    if args.execute:
        if not kp_path:
            print("Error: WALLET_KEYPAIR_PATH required for --execute")
            sys.exit(1)
    if kp_path:
        from solders.keypair import Keypair
        with open(kp_path) as f:
            keypair = Keypair.from_bytes(bytes(json.load(f)))
        pubkey = str(keypair.pubkey())
    else:
        pubkey = "2vesMyZ8TDqJm1NF1LoWeEVsPVknXRXVMtrmhCQSRhs7"

    print(f"\n{'=' * 60}")
    print(f"  SELL RAY → SOL — {'EXECUTE' if args.execute else 'DRY RUN'}")
    print(f"  Wallet: {pubkey}")
    print(f"{'=' * 60}\n")

    # Check balances
    sol_balance = get_sol_balance(rpc_endpoint, pubkey)
    ray_raw, ray_ui, ray_account = get_ray_balance(rpc_endpoint, pubkey)

    print(f"  SOL balance:  {sol_balance:.6f} SOL")
    print(f"  RAY balance:  {ray_ui:.6f} RAY ({ray_raw} raw)")

    if ray_raw == 0:
        print("\n  No RAY to sell.")
        return

    # Get quote
    print(f"\n  Getting quote for {ray_ui:.6f} RAY → SOL...")
    try:
        quote = get_quote(quote_url, ray_raw, args.slippage, api_key)
    except Exception as e:
        print(f"  Quote failed: {e}")
        return

    out_sol = int(quote.get("outAmount", 0)) / 1e9
    price_impact = quote.get("priceImpactPct", "?")
    print(f"  Estimated output:  {out_sol:.6f} SOL")
    print(f"  Price impact:      {price_impact}%")
    print(f"  Slippage:          {args.slippage} bps")

    if not args.execute:
        print(f"\n  DRY RUN — re-run with --execute to swap")
        return

    # Build and sign swap tx
    print(f"\n  Building swap transaction...")
    swap_tx_b64 = get_swap_tx(swap_url, quote, pubkey, api_key)
    if not swap_tx_b64:
        print("  FAILED: no swap transaction returned")
        return

    from solders.transaction import VersionedTransaction
    tx_bytes = base64.b64decode(swap_tx_b64)
    tx = VersionedTransaction.from_bytes(tx_bytes)
    signed_tx = VersionedTransaction(tx.message, [keypair])
    signed_b64 = base64.b64encode(bytes(signed_tx)).decode()
    signature = str(signed_tx.signatures[0])

    # Submit
    print(f"  Submitting transaction...")
    try:
        submit_tx(rpc_endpoint, signed_b64)
    except RuntimeError as e:
        print(f"  FAILED: {e}")
        return

    print(f"  tx: {signature}")

    # Confirm
    print(f"  Waiting for confirmation...")
    confirmed, status = confirm_tx(rpc_endpoint, signature, timeout_s=30)
    if confirmed:
        print(f"  CONFIRMED ({status}) — ~{out_sol:.6f} SOL received")
        # Check final balances
        time.sleep(1)
        sol_after = get_sol_balance(rpc_endpoint, pubkey)
        ray_after_raw, ray_after_ui, _ = get_ray_balance(rpc_endpoint, pubkey)
        print(f"\n  Final SOL balance: {sol_after:.6f} SOL (was {sol_balance:.6f})")
        print(f"  Final RAY balance: {ray_after_ui:.6f} RAY (was {ray_ui:.6f})")
    else:
        print(f"  NOT CONFIRMED: {status}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
