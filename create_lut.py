#!/usr/bin/env python3
"""
Create a Solana Address Lookup Table (LUT) for statalyzer and populate it
with Lunar Lander tip accounts.

One-time setup script. Run once, then paste the printed LUT address into
constants.py as STATALYZER_LUT_ADDRESS.

Usage:
    python create_lut.py
"""

import base64
import json
import os
import struct
import time
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


ALT_PROGRAM_ID_STR = "AddressLookupTab1e1111111111111111111111111"
RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")


def rpc(method, params):
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(
        RPC_URL, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    if "error" in result:
        raise RuntimeError(f"RPC {method} error: {result['error']}")
    return result["result"]


def get_finalized_slot() -> int:
    return rpc("getSlot", [{"commitment": "finalized"}])


def get_blockhash() -> str:
    return rpc("getLatestBlockhash", [{"commitment": "confirmed"}])["value"]["blockhash"]


def send_and_confirm(tx_b64: str, label: str) -> str:
    sig = rpc("sendTransaction", [tx_b64, {"encoding": "base64", "skipPreflight": False, "maxRetries": 5}])
    print(f"  {label}: {sig}")
    deadline = time.time() + 60
    while time.time() < deadline:
        statuses = rpc("getSignatureStatuses", [[sig], {"searchTransactionHistory": False}])["value"]
        if statuses and statuses[0]:
            st = statuses[0]
            if st.get("err"):
                raise RuntimeError(f"Transaction failed on-chain: {st['err']}")
            if st.get("confirmationStatus") in ("confirmed", "finalized"):
                print(f"  Confirmed ({st['confirmationStatus']})")
                return sig
        time.sleep(2)
    raise RuntimeError(f"Confirmation timeout: {sig}")


def main():
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey
    from solders.hash import Hash
    from solders.instruction import Instruction, AccountMeta
    from solders.message import Message
    from solders.transaction import Transaction
    from constants import LUNAR_LANDER_TIP_ACCOUNTS

    kp_path = os.getenv("WALLET_KEYPAIR_PATH")
    if not kp_path:
        raise SystemExit("Set WALLET_KEYPAIR_PATH in .env")

    with open(kp_path) as f:
        keypair = Keypair.from_bytes(bytes(json.load(f)))
    authority = keypair.pubkey()
    print(f"Authority: {authority}")

    alt_program = Pubkey.from_string(ALT_PROGRAM_ID_STR)
    system_program = Pubkey.from_string("11111111111111111111111111111111")

    # Derive LUT address from authority + recent finalized slot
    # Seeds: [authority_bytes, slot_le_bytes] (no prefix)
    slot = get_finalized_slot()
    slot_bytes = struct.pack("<Q", slot)
    lut_address, bump = Pubkey.find_program_address(
        [bytes(authority), slot_bytes], alt_program
    )
    print(f"Slot: {slot}  LUT: {lut_address}  bump: {bump}")

    # --- Create LUT ---
    # CreateLookupTable: discriminant u32=0, recent_slot u64, bump_seed u8
    create_data = struct.pack("<IQB", 0, slot, bump)
    create_ix = Instruction(
        alt_program,
        create_data,
        [
            AccountMeta(lut_address, is_signer=False, is_writable=True),
            AccountMeta(authority, is_signer=True, is_writable=False),
            AccountMeta(authority, is_signer=True, is_writable=True),
            AccountMeta(system_program, is_signer=False, is_writable=False),
        ],
    )

    blockhash = get_blockhash()
    msg = Message.new_with_blockhash([create_ix], authority, Hash.from_string(blockhash))
    tx = Transaction.new_unsigned(msg)
    tx.sign([keypair], Hash.from_string(blockhash))
    tx_b64 = base64.b64encode(bytes(tx)).decode()

    print("Creating LUT...")
    send_and_confirm(tx_b64, "create")

    # Wait a moment for the account to be visible
    time.sleep(4)

    # --- Extend LUT with Lunar Lander tip accounts ---
    # ExtendLookupTable: discriminant u32=2, count u64, addresses (32 bytes each)
    tip_pubkeys = [Pubkey.from_string(a) for a in LUNAR_LANDER_TIP_ACCOUNTS]
    extend_data = struct.pack("<IQ", 2, len(tip_pubkeys)) + b"".join(bytes(pk) for pk in tip_pubkeys)
    extend_ix = Instruction(
        alt_program,
        extend_data,
        [
            AccountMeta(lut_address, is_signer=False, is_writable=True),
            AccountMeta(authority, is_signer=True, is_writable=False),
            AccountMeta(authority, is_signer=True, is_writable=True),
            AccountMeta(system_program, is_signer=False, is_writable=False),
        ],
    )

    blockhash = get_blockhash()
    msg = Message.new_with_blockhash([extend_ix], authority, Hash.from_string(blockhash))
    tx = Transaction.new_unsigned(msg)
    tx.sign([keypair], Hash.from_string(blockhash))
    tx_b64 = base64.b64encode(bytes(tx)).decode()

    print(f"Extending LUT with {len(tip_pubkeys)} Lunar Lander tip accounts...")
    send_and_confirm(tx_b64, "extend")

    print(f"""
LUT created. Add to constants.py:

    STATALYZER_LUT_ADDRESS = "{lut_address}"
""")


if __name__ == "__main__":
    main()
