#!/usr/bin/env python3
"""
Convert a Solana private key to the JSON array format used by this bot.

Accepts:
  - Base58 private key string
  - Hex private key string
  - Existing JSON array file

Usage:
    python3 convert_key.py <private_key_base58>
    python3 convert_key.py --hex <private_key_hex>
    python3 convert_key.py --file <path_to_key_file>
    python3 convert_key.py <private_key_base58> --output wallet.json
"""

import argparse
import json
import sys


def base58_decode(s: str) -> bytes:
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    n = 0
    for c in s:
        n = n * 58 + alphabet.index(c)
    result = n.to_bytes(max(1, (n.bit_length() + 7) // 8), "big")
    # Preserve leading zeros
    pad = len(s) - len(s.lstrip("1"))
    return b"\x00" * pad + result


def main():
    parser = argparse.ArgumentParser(description="Convert Solana private key to JSON array")
    parser.add_argument("key", nargs="?", help="Base58 private key")
    parser.add_argument("--hex", dest="hex_key", help="Hex-encoded private key")
    parser.add_argument("--file", dest="file_path", help="Path to existing key file")
    parser.add_argument("--output", "-o", default=None, help="Output file (default: stdout)")
    args = parser.parse_args()

    if args.file_path:
        with open(args.file_path) as f:
            content = f.read().strip()
        try:
            key_bytes = json.loads(content)
            if isinstance(key_bytes, list):
                print("Already in array format")
            else:
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            # Try base58
            key_bytes = list(base58_decode(content))
    elif args.hex_key:
        key_bytes = list(bytes.fromhex(args.hex_key))
    elif args.key:
        key_bytes = list(base58_decode(args.key))
    else:
        parser.print_help()
        sys.exit(1)

    if len(key_bytes) == 32:
        print("Warning: 32-byte secret key only (no public key). Most wallets use 64 bytes.", file=sys.stderr)
    elif len(key_bytes) != 64:
        print(f"Warning: unexpected key length {len(key_bytes)} (expected 64)", file=sys.stderr)

    output = json.dumps(key_bytes, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output + "\n")
        print(f"Written to {args.output} ({len(key_bytes)} bytes)")
    else:
        print(output)


if __name__ == "__main__":
    main()
