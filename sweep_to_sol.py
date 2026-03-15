#!/usr/bin/env python3
"""
Sweep all non-SOL tokens in a wallet back to SOL via Jupiter swaps.

Usage:
    python sweep_to_sol.py                          # Dry run (show what would be swapped)
    python sweep_to_sol.py --execute                # Actually execute swaps (sequential via RPC)
    python sweep_to_sol.py --execute --batch         # Submit all swaps in parallel via sendTransaction
    python sweep_to_sol.py --execute --min-value 0.01  # Skip tokens worth less than 0.01 SOL
"""

import argparse
import base64
import json
import logging
import os
import random
import sys
import urllib.error
import time
import urllib.request

# Unbuffered stdout so output streams over SSH
sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

SOL_MINT = "So11111111111111111111111111111111111111112"
WALLET = "2vesMyZ8TDqJm1NF1LoWeEVsPVknXRXVMtrmhCQSRhs7"

LUNAR_LANDER_TIP_ACCOUNTS = [
    "moon17L6BgxXRX5uHKudAmqVF96xia9h8ygcmG2sL3F",
    "moon26Sek222Md7ZydcAGxoKG832DK36CkLrS3PQY4c",
    "moon7fwyajcVstMoBnVy7UBcTx87SBtNoGGAaH2Cb8V",
    "moonBtH9HvLHjLqi9ivyrMVKgFUsSfrz9BwQ9khhn1u",
    "moonCJg8476LNFLptX1qrK8PdRsA1HD1R6XWyu9MB93",
    "moonF2sz7qwAtdETnrgxNbjonnhGGjd6r4W4UC9284s",
    "moonKfftMiGSak3cezvhEqvkPSzwrmQxQHXuspC96yj",
    "moonQBUKBpkifLcTd78bfxxt4PYLwmJ5admLW6cBBs8",
    "moonXwpKwoVkMegt5Bc776cSW793X1irL5hHV1vJ3JA",
    "moonZ6u9E2fgk6eWd82621eLPHt9zuJuYECXAYjMY1C",
]

# Well-known token metadata for display
KNOWN_SYMBOLS = {
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": "USDC",
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": "USDT",
    "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs": "ETH",
    "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E": "BTC",
    "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R": "RAY",
    "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263": "BONK",
    "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So": "mSOL",
    "jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL": "JTO",
    "J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn": "jitoSOL",
    "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm": "WIF",
    "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN": "JUP",
    "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3": "PYTH",
    "HaP8r3ksG76PhQLTqR8FYBeNiQpejcFbQmiHbg787Ut1": "TRUMP",
    "9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump": "FARTCOIN",
    "jupSoLaHXQiZZTSfEWMTRRgpnyFm8f6sZdosWBjx93v": "jupSOL",
    "bSo13r4TkiE4KumL71LsHTPpL2euBYLFx6h9HP3piy1": "bSOL",
    "HeLp6NuQkmYB4pYWo2zYs22mESHXPQYzXbB8n4V98jwC": "ai16z",
    "7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr": "POPCAT",
    "MEW1gQWJ3nEXg2qgERiKu7FAFj79PHvQVREQUzScPP5": "MEW",
}


def rpc_call(endpoint, method, params):
    """Make a JSON-RPC call to Solana."""
    payload = json.dumps({
        "jsonrpc": "2.0", "id": 1,
        "method": method, "params": params,
    }).encode()
    req = urllib.request.Request(
        endpoint, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def get_token_accounts(rpc_endpoint, wallet):
    """Get all token accounts for a wallet (both SPL Token and Token-2022)."""
    TOKEN_PROGRAMS = [
        "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",   # SPL Token
        "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",   # Token-2022
    ]
    accounts = []
    seen_mints = set()
    for program_id in TOKEN_PROGRAMS:
        result = rpc_call(rpc_endpoint, "getTokenAccountsByOwner", [
            wallet,
            {"programId": program_id},
            {"encoding": "jsonParsed"},
        ])
        for item in result.get("result", {}).get("value", []):
            info = item["account"]["data"]["parsed"]["info"]
            mint = info["mint"]
            amount = int(info["tokenAmount"]["amount"])
            decimals = info["tokenAmount"]["decimals"]
            ui_amount = info["tokenAmount"].get("uiAmount", 0) or 0
            if amount > 0 and mint != SOL_MINT and mint not in seen_mints:
                seen_mints.add(mint)
                accounts.append({
                    "mint": mint,
                    "amount_raw": amount,
                    "decimals": decimals,
                    "ui_amount": ui_amount,
                    "token_account": item["pubkey"],
                })
    return accounts


def get_jupiter_quote(quote_url, input_mint, output_mint, amount_raw, slippage_bps=100, api_key=""):
    """Get a swap quote from Jupiter."""
    url = (f"{quote_url}/swap/v1/quote"
           f"?inputMint={input_mint}&outputMint={output_mint}"
           f"&amount={amount_raw}&slippageBps={slippage_bps}")
    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def get_jupiter_swap_tx(swap_url, quote, pubkey, priority_fee=100000, api_key=""):
    """Get a swap transaction from Jupiter."""
    body = {
        "quoteResponse": quote,
        "userPublicKey": pubkey,
        "wrapAndUnwrapSol": True,
        "dynamicComputeUnitLimit": True,
    }
    if priority_fee == 0:
        body["prioritizationFeeLamports"] = "auto"
    else:
        body["prioritizationFeeLamports"] = priority_fee
    payload = json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    req = urllib.request.Request(
        f"{swap_url}/swap/v1/swap", data=payload,
        headers=headers, method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
        return data.get("swapTransaction")


def submit_transaction(rpc_endpoint, signed_tx_b64):
    """Submit a signed transaction via Solana RPC."""
    result = rpc_call(rpc_endpoint, "sendTransaction", [
        signed_tx_b64,
        {"encoding": "base64", "skipPreflight": True, "maxRetries": 2},
    ])
    if result.get("result"):
        return True, result["result"]
    error = result.get("error", {})
    return False, error.get("message", str(error))


def submit_lunar_bundle(endpoint, api_key, signed_txs_b64):
    """Submit a bundle of base64-encoded signed transactions to Lunar Lander.
    Uses JSON-RPC format per https://docs.hellomoon.io/reference/send-bundle-api
    Max 4 txs per bundle (atomic execution)."""
    url = f"{endpoint}/sendbundle"
    if api_key:
        url += f"?api-key={api_key}"

    payload = json.dumps({
        "jsonrpc": "2.0", "id": 1,
        "method": "sendBundle",
        "params": [
            signed_txs_b64,
            {"encoding": "base64"},
        ],
    }).encode()

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
            return body
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        return {"error": f"HTTP {e.code}: {error_body[:500]}"}


def build_tip_tx(keypair, tip_lamports, rpc_endpoint):
    """Build a tip transaction transferring SOL to a random Lunar Lander moon* account."""
    from solders.transaction import VersionedTransaction
    from solders.message import MessageV0
    from solders.system_program import transfer, TransferParams
    from solders.pubkey import Pubkey
    from solders.hash import Hash

    tip_account = random.choice(LUNAR_LANDER_TIP_ACCOUNTS)
    tip_ix = transfer(TransferParams(
        from_pubkey=keypair.pubkey(),
        to_pubkey=Pubkey.from_string(tip_account),
        lamports=tip_lamports,
    ))

    bh_result = rpc_call(rpc_endpoint, "getLatestBlockhash", [{"commitment": "confirmed"}])
    blockhash_str = bh_result["result"]["value"]["blockhash"]

    tip_msg = MessageV0.try_compile(
        payer=keypair.pubkey(),
        instructions=[tip_ix],
        address_lookup_table_accounts=[],
        recent_blockhash=Hash.from_string(blockhash_str),
    )
    tip_tx = VersionedTransaction(tip_msg, [keypair])
    return base64.b64encode(bytes(tip_tx)).decode(), tip_account


def confirm_transaction(rpc_endpoint, signature, timeout_s=15):
    """Poll for transaction confirmation."""
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


def execute_sequential(sweep_targets, keypair, pubkey, rpc_endpoint, quote_url, swap_url,
                       api_key, args):
    """Execute swaps one at a time via RPC."""
    from solders.transaction import VersionedTransaction

    success_count = 0
    total_sol_received = 0.0

    for i, target in enumerate(sweep_targets):
        acct = target["account"]
        mint = acct["mint"]
        symbol = KNOWN_SYMBOLS.get(mint, mint[:8] + "..")

        print(f"  [{i+1}/{len(sweep_targets)}] {symbol} ({acct['ui_amount']:.6f}) → SOL...")

        try:
            quote = get_jupiter_quote(quote_url, mint, SOL_MINT, acct["amount_raw"],
                                      slippage_bps=args.slippage, api_key=api_key)

            swap_tx_b64 = get_jupiter_swap_tx(swap_url, quote, pubkey,
                                              priority_fee=args.priority_fee, api_key=api_key)
            if not swap_tx_b64:
                print(f"    FAILED: no swap transaction returned")
                continue

            tx_bytes = base64.b64decode(swap_tx_b64)
            tx = VersionedTransaction.from_bytes(tx_bytes)
            signed_tx = VersionedTransaction(tx.message, [keypair])
            signed_b64 = base64.b64encode(bytes(signed_tx)).decode()

            # Get current slot and timestamp at submit time
            submit_ts = time.time()
            try:
                slot_result = rpc_call(rpc_endpoint, "getSlot", [{"commitment": "confirmed"}])
                submit_slot = slot_result.get("result", 0)
            except Exception:
                submit_slot = 0

            ok, result = submit_transaction(rpc_endpoint, signed_b64)
            if not ok:
                print(f"    FAILED: {result}")
                continue

            signature = result
            print(f"    Slot:      {submit_slot}")
            print(f"    Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(submit_ts))}")
            print(f"    Signature: {signature}")

            confirmed, status = confirm_transaction(rpc_endpoint, signature, timeout_s=15)
            if confirmed:
                out_sol = int(quote.get("outAmount", 0)) / 1e9
                total_sol_received += out_sol
                print(f"    CONFIRMED ({status}) — ~{out_sol:.6f} SOL | tx: {signature}")
                success_count += 1
            else:
                print(f"    NOT CONFIRMED: {status} | tx: {signature}")

            if i < len(sweep_targets) - 1:
                time.sleep(2)

        except Exception as e:
            print(f"    ERROR: {e}")

    return success_count, total_sol_received


def execute_bundle(sweep_targets, keypair, pubkey, rpc_endpoint, quote_url, swap_url,
                   api_key, args, lunar_endpoint, lunar_api_key):
    """Execute swaps via Lunar Lander sendBundle, one [swap, tip] bundle per token."""
    from solders.transaction import VersionedTransaction

    success_count = 0
    total_sol_received = 0.0

    for i, target in enumerate(sweep_targets):
        acct = target["account"]
        mint = acct["mint"]
        symbol = KNOWN_SYMBOLS.get(mint, mint[:8] + "..")

        print(f"\n  [{i+1}/{len(sweep_targets)}] {symbol} ({acct['ui_amount']:.6f}) → SOL...")

        try:
            quote = get_jupiter_quote(quote_url, mint, SOL_MINT, acct["amount_raw"],
                                      slippage_bps=args.slippage, api_key=api_key)
            est_sol = int(quote.get("outAmount", 0)) / 1e9

            swap_tx_b64 = get_jupiter_swap_tx(swap_url, quote, pubkey,
                                              priority_fee=args.priority_fee, api_key=api_key)
            if not swap_tx_b64:
                print(f"    SKIPPED: no swap tx returned")
                continue

            tx_bytes = base64.b64decode(swap_tx_b64)
            tx = VersionedTransaction.from_bytes(tx_bytes)
            signed_tx = VersionedTransaction(tx.message, [keypair])
            sig_str = str(signed_tx.signatures[0])
            signed_b64 = base64.b64encode(bytes(signed_tx)).decode()

            # Build tip tx
            tip_b64, tip_account = build_tip_tx(keypair, args.tip, rpc_endpoint)
            print(f"    Tip: {args.tip} lamports → {tip_account[:12]}...")

            # Submit [swap, tip] bundle
            result = submit_lunar_bundle(lunar_endpoint, lunar_api_key, [signed_b64, tip_b64])
            if result.get("error"):
                print(f"    BUNDLE FAILED: {result['error']}")
                continue

            print(f"    tx: {sig_str}")
            print(f"    Bundle ID: {result.get('result', '')}")

            confirmed, status = confirm_transaction(rpc_endpoint, sig_str, timeout_s=30)
            if confirmed:
                success_count += 1
                total_sol_received += est_sol
                print(f"    CONFIRMED ({status}) — ~{est_sol:.6f} SOL")
            else:
                print(f"    NOT CONFIRMED: {status}")

        except Exception as e:
            print(f"    ERROR: {e}")

    return success_count, total_sol_received


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Sweep non-SOL tokens back to SOL")
    parser.add_argument("--execute", action="store_true", help="Actually execute swaps (default: dry run)")
    parser.add_argument("--bundle", action="store_true",
                        help="Use Lunar Lander sendBundle (1 swap + 1 tip per bundle)")
    parser.add_argument("--tip", type=int, default=1000000,
                        help="Tip amount in lamports for bundle mode (default: 1000000 = 0.001 SOL)")
    parser.add_argument("--min-value", type=float, default=0.0,
                        help="Skip tokens worth less than this many SOL (default: 0 = sweep all)")
    parser.add_argument("--slippage", type=int, default=200,
                        help="Slippage tolerance in bps (default: 200 = 2%%)")
    parser.add_argument("--priority-fee", type=int, default=100000,
                        help="Priority fee in lamports (default: 100000)")
    args = parser.parse_args()

    rpc_endpoint = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    quote_url = os.getenv("JUPITER_QUOTE_URL", "https://lite-api.jup.ag").rstrip("/")
    swap_url = quote_url
    api_key = os.getenv("JUPITER_API_KEY", "")
    kp_path = os.getenv("WALLET_KEYPAIR_PATH", "")
    lunar_endpoint = os.getenv("LUNAR_LANDER_ENDPOINT", "http://fra.lunar-lander.hellomoon.io").rstrip("/")
    lunar_api_key = os.getenv("LUNAR_LANDER_API_KEY", "")

    # Load keypair if executing
    keypair = None
    pubkey = WALLET
    if args.execute:
        if not kp_path:
            print("Error: WALLET_KEYPAIR_PATH env var required for --execute")
            sys.exit(1)
        import base58
        from solders.keypair import Keypair
        with open(kp_path, "r") as f:
            keypair_data = json.load(f)
        keypair = Keypair.from_bytes(bytes(keypair_data))
        pubkey = str(keypair.pubkey())
        if pubkey != WALLET:
            print(f"Warning: keypair pubkey {pubkey} != expected wallet {WALLET}")
            confirm = input("Continue? [y/N] ").strip().lower()
            if confirm != "y":
                sys.exit(0)

    mode = "BUNDLE" if args.bundle else "SEQUENTIAL"
    print(f"\n{'=' * 70}")
    print(f"  SWEEP TO SOL — {'EXECUTE (' + mode + ')' if args.execute else 'DRY RUN'}")
    print(f"  Wallet: {WALLET}")
    if args.bundle:
        print(f"  Lunar Lander: {lunar_endpoint} (sendBundle, tip={args.tip} lamports)")
    print(f"{'=' * 70}\n")

    # Get all token accounts
    print("Fetching token accounts...")
    accounts = get_token_accounts(rpc_endpoint, WALLET)

    if not accounts:
        print("No non-SOL token balances found. Wallet is clean.")
        return

    print(f"Found {len(accounts)} non-SOL token(s) with balance:\n")
    print(f"  {'Token':<12} {'Mint':<46} {'Balance':>18}")
    print(f"  {'-'*12} {'-'*46} {'-'*18}")

    # Get Jupiter quotes for all tokens to estimate SOL value
    sweep_targets = []
    total_sol_estimate = 0.0

    for acct in accounts:
        mint = acct["mint"]
        symbol = KNOWN_SYMBOLS.get(mint, mint[:8] + "..")
        ui_amount = acct["ui_amount"]

        print(f"  {symbol:<12} {mint:<46} {ui_amount:>18.6f}")

        try:
            quote = get_jupiter_quote(quote_url, mint, SOL_MINT, acct["amount_raw"],
                                      slippage_bps=args.slippage, api_key=api_key)
            out_sol = int(quote.get("outAmount", 0)) / 1e9
            total_sol_estimate += out_sol

            if args.min_value > 0 and out_sol < args.min_value:
                print(f"    → ~{out_sol:.6f} SOL (skipping, below min-value {args.min_value})")
                continue

            print(f"    → ~{out_sol:.6f} SOL")
            sweep_targets.append({"account": acct, "quote": quote, "est_sol": out_sol})
        except Exception as e:
            print(f"    → Quote failed: {e} (skipping)")

    print(f"\n  Total estimated: ~{total_sol_estimate:.6f} SOL")
    print(f"  Tokens to sweep: {len(sweep_targets)}")

    if not sweep_targets:
        print("\nNothing to sweep.")
        return

    if not args.execute:
        print(f"\n  DRY RUN — re-run with --execute to perform swaps")
        return

    print(f"\n  Executing {len(sweep_targets)} swap(s) ({mode})...\n")

    if args.bundle:
        success_count, total_sol_received = execute_bundle(
            sweep_targets, keypair, pubkey, rpc_endpoint, quote_url, swap_url,
            api_key, args, lunar_endpoint, lunar_api_key)
    else:
        success_count, total_sol_received = execute_sequential(
            sweep_targets, keypair, pubkey, rpc_endpoint, quote_url, swap_url,
            api_key, args)

    print(f"\n{'=' * 70}")
    print(f"  Sweep complete: {success_count}/{len(sweep_targets)} swaps succeeded")
    print(f"  Estimated SOL recovered: ~{total_sol_received:.6f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
