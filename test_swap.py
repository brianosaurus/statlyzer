"""
Quick test: buy a tiny amount of BONK with SOL via direct sendTransaction (no Lunar Lander),
then sell it back. Tests the thread-pool parallel execution + confirmation.
"""
import asyncio
import base64
import json
import logging
import os
import sys
import time
import urllib.request

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from db import Database
from executor import LiveExecutor
from constants import SOL_MINT, BONK_MINT, WIF_MINT

logger = logging.getLogger("test_swap")


def _price_from_quote(executor, mint, sol_price):
    """Get token price by quoting 1 SOL -> token via Jupiter."""
    one_sol = 1_000_000_000
    quote = executor._get_quote(SOL_MINT, mint, one_sol)
    if not quote:
        return 0
    out_amount = int(quote.get("outAmount", 0))
    from constants import WELL_KNOWN_TOKENS
    decimals = WELL_KNOWN_TOKENS.get(mint, {}).get("decimals", 9)
    tokens_per_sol = out_amount / (10 ** decimals)
    if tokens_per_sol <= 0:
        return 0
    return sol_price / tokens_per_sol


def _send_tx_direct(rpc_url, signed_b64):
    """Send a signed transaction directly via RPC sendTransaction (no bundle)."""
    payload = json.dumps({
        "jsonrpc": "2.0", "id": 1,
        "method": "sendTransaction",
        "params": [signed_b64, {"encoding": "base64", "skipPreflight": True,
                                 "maxRetries": 3}],
    }).encode()
    req = urllib.request.Request(rpc_url, data=payload,
                                headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        result = json.loads(resp.read())
    if result.get("error"):
        raise RuntimeError(f"sendTransaction error: {result['error']}")
    return result.get("result", "")


def _do_swap_direct(executor, input_mint, output_mint, amount_raw):
    """Build a swap, send it directly (no Lunar Lander), confirm it."""
    from solders.transaction import VersionedTransaction

    t0 = time.monotonic()
    quote = executor._get_quote(input_mint, output_mint, amount_raw)
    if not quote:
        raise RuntimeError("Quote failed")
    t_quote = time.monotonic()

    swap_tx_b64 = executor._get_swap_transaction(quote)
    if not swap_tx_b64:
        raise RuntimeError("Swap tx failed")
    t_swap = time.monotonic()

    tx_bytes = base64.b64decode(swap_tx_b64)
    tx = VersionedTransaction.from_bytes(tx_bytes)
    signed_tx = VersionedTransaction(tx.message, [executor._keypair])
    signed_b64 = base64.b64encode(bytes(signed_tx)).decode()
    signature = str(signed_tx.signatures[0])
    t_sign = time.monotonic()

    # Send directly (no bundle, no tip)
    try:
        sig_result = _send_tx_direct(executor._rpc_endpoint, signed_b64)
        logger.info(f"sendTransaction: {sig_result}")
    except RuntimeError as e:
        logger.error(f"sendTransaction failed: {e}")
        raise
    t_submit = time.monotonic()

    # Confirm
    confirmed, err = executor._confirm_transaction_sync(signature, timeout_s=30)
    t_confirm = time.monotonic()

    logger.info(f"TX timing: quote={int((t_quote-t0)*1000)}ms swap={int((t_swap-t_quote)*1000)}ms "
                f"sign={int((t_sign-t_swap)*1000)}ms submit={int((t_submit-t_sign)*1000)}ms "
                f"confirm={int((t_confirm-t_submit)*1000)}ms total={int((t_confirm-t0)*1000)}ms "
                f"confirmed={confirmed}")

    if err:
        raise RuntimeError(f"On-chain error: {err}")
    if not confirmed:
        raise RuntimeError(f"Not confirmed after 30s (tx: {signature})")

    return quote, signature


def _do_swap_bundle(executor, input_mint, output_mint, amount_raw):
    """Build a swap, send via Lunar Lander bundle, confirm it."""
    from solders.transaction import VersionedTransaction

    t0 = time.monotonic()
    quote = executor._get_quote(input_mint, output_mint, amount_raw)
    if not quote:
        raise RuntimeError("Quote failed")
    t_quote = time.monotonic()

    swap_tx_b64 = executor._get_swap_transaction(quote)
    if not swap_tx_b64:
        raise RuntimeError("Swap tx failed")
    t_swap = time.monotonic()

    tx_bytes = base64.b64decode(swap_tx_b64)
    tx = VersionedTransaction.from_bytes(tx_bytes)
    signed_tx = VersionedTransaction(tx.message, [executor._keypair])
    signed_b64 = base64.b64encode(bytes(signed_tx)).decode()
    signature = str(signed_tx.signatures[0])

    # Build tip and submit as bundle
    tip_b64 = executor._build_tip_tx()
    t_sign = time.monotonic()

    ok = executor._submit_lunar_bundle([signed_b64, tip_b64])
    t_submit = time.monotonic()
    if not ok:
        raise RuntimeError("Bundle submission failed")

    # Confirm
    confirmed, err = executor._confirm_transaction_sync(signature, timeout_s=30)
    t_confirm = time.monotonic()

    logger.info(f"TX timing: quote={int((t_quote-t0)*1000)}ms swap={int((t_swap-t_quote)*1000)}ms "
                f"sign={int((t_sign-t_swap)*1000)}ms submit={int((t_submit-t_sign)*1000)}ms "
                f"confirm={int((t_confirm-t_submit)*1000)}ms total={int((t_confirm-t0)*1000)}ms "
                f"confirmed={confirmed}")

    if err:
        raise RuntimeError(f"On-chain error: {err}")
    if not confirmed:
        raise RuntimeError(f"Not confirmed after 30s (tx: {signature})")

    return quote, signature


async def main():
    config = Config()
    db = Database(":memory:")
    executor = LiveExecutor(config, db)

    # Get SOL price
    from constants import USDC_MINT
    quote = executor._get_quote(SOL_MINT, USDC_MINT, 1_000_000_000)
    if not quote:
        print("ERROR: Could not get SOL/USDC quote")
        return
    sol_price = int(quote["outAmount"]) / 1e6
    executor.sol_usd_price = sol_price
    print(f"SOL price: ${sol_price:.2f}")

    # Get BONK price
    bonk_price = _price_from_quote(executor, BONK_MINT, sol_price)
    print(f"BONK price: ${bonk_price:.8f}")

    # Test 1: Direct sendTransaction (no Lunar Lander)
    usd_amount = 0.50
    sol_lamports = int((usd_amount / sol_price) * 1e9)
    print(f"\n=== TEST 1: Direct sendTransaction (buy ${usd_amount} BONK) ===")
    print(f"Swapping {sol_lamports} lamports ({sol_lamports/1e9:.6f} SOL)")
    try:
        quote, sig = _do_swap_direct(executor, SOL_MINT, BONK_MINT, sol_lamports)
        out_amount = int(quote["outAmount"])
        print(f"SUCCESS: tx={sig[:30]}... outAmount={out_amount}")
    except RuntimeError as e:
        print(f"FAILED: {e}")

    print(f"\n=== TEST 2: Lunar Lander bundle (buy ${usd_amount} BONK) ===")
    try:
        quote, sig = _do_swap_bundle(executor, SOL_MINT, BONK_MINT, sol_lamports)
        out_amount = int(quote["outAmount"])
        print(f"SUCCESS: tx={sig[:30]}... outAmount={out_amount}")
    except RuntimeError as e:
        print(f"FAILED: {e}")

    # Test 3: Full parallel entry via thread pool
    from signals import SignalType, Signal
    from position import PositionSize, get_decimals

    wif_price = _price_from_quote(executor, WIF_MINT, sol_price)
    print(f"\nWIF price: ${wif_price:.6f}")

    mints = sorted([BONK_MINT, WIF_MINT])
    from constants import WELL_KNOWN_TOKENS
    symbols = [WELL_KNOWN_TOKENS[m]['symbol'] for m in mints]

    signal = Signal(
        signal_type=SignalType.ENTRY_LONG, pair_key=",".join(mints),
        basket_size=2, mints=mints, symbols=symbols,
        hedge_ratios=[1.0, 1.0], zscore=-2.5,
        spread=0.0, spread_mean=0.0, spread_std=1.0,
        timestamp=int(time.time()), slot=0,
    )

    usd_per_leg = 0.50
    amounts, amounts_raw, prices_list = [], [], []
    for m in mints:
        p = bonk_price if m == BONK_MINT else wif_price
        qty = usd_per_leg / p
        decimals = get_decimals(m)
        amounts.append(qty)
        amounts_raw.append(int(qty * 10**decimals))
        prices_list.append(p)

    pos_size = PositionSize(amounts=amounts, amounts_raw=amounts_raw,
                            dollar_amounts=[usd_per_leg] * 2, total_exposure_usd=1.0)

    print(f"\n=== TEST 3: Parallel entry (2 legs, ${usd_per_leg} each) ===")
    t0 = time.monotonic()
    try:
        execution = await executor.execute_entry(signal, pos_size, prices_list)
        elapsed = time.monotonic() - t0
        print(f"Entry done in {elapsed:.1f}s — {len(execution.fills)} fills")
        for f in execution.fills:
            sym = WELL_KNOWN_TOKENS.get(f.token_mint, {}).get('symbol', f.token_mint[:8])
            print(f"  {f.side} {sym} qty={f.quantity:.4f} @ ${f.price:.8f} tx={f.tx_signature[:20] if f.tx_signature else 'N/A'}...")
    except RuntimeError as e:
        elapsed = time.monotonic() - t0
        print(f"Entry FAILED after {elapsed:.1f}s: {e}")
        return

    # Test 4: Parallel exit
    from portfolio import Position
    position = Position(
        id=0, pair_key=signal.pair_key, basket_size=2, mints=mints,
        direction="long", hedge_ratios=[1.0, 1.0],
        entry_time=int(time.time()), entry_slot=0, entry_zscore=-2.5,
        entry_prices=prices_list,
        quantities=[f.quantity for f in execution.fills],
        quantities_raw=[f.quantity_raw for f in execution.fills],
        entry_values=[f.quantity * f.price for f in execution.fills],
    )

    print(f"\n=== TEST 4: Parallel exit (sell back) ===")
    t0 = time.monotonic()
    try:
        exit_exec = await executor.execute_exit(position, prices_list)
        elapsed = time.monotonic() - t0
        print(f"Exit done in {elapsed:.1f}s — {len(exit_exec.fills)} fills")
        for f in exit_exec.fills:
            sym = WELL_KNOWN_TOKENS.get(f.token_mint, {}).get('symbol', f.token_mint[:8])
            print(f"  {f.side} {sym} qty={f.quantity:.4f} @ ${f.price:.8f} tx={f.tx_signature[:20] if f.tx_signature else 'N/A'}...")
    except RuntimeError as e:
        elapsed = time.monotonic() - t0
        print(f"Exit FAILED after {elapsed:.1f}s: {e}")

    print("\nDone.")

if __name__ == "__main__":
    asyncio.run(main())
