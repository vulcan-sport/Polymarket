#!/usr/bin/env python3
"""
One-time script to generate Polymarket CLOB API L2 credentials.

Run this once after filling in POLYMARKET_PRIVATE_KEY and POLYMARKET_WALLET_ADDRESS in .env.
It will print the L2 credentials (API_KEY, API_SECRET, API_PASSPHRASE) to copy into your .env.

Usage:
    uv run python scripts/setup_credentials.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import os

from py_clob_client.client import ClobClient


def main() -> None:
    private_key = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
    wallet_address = os.environ.get("POLYMARKET_WALLET_ADDRESS", "")

    if not private_key or private_key == "0x...":
        print("ERROR: POLYMARKET_PRIVATE_KEY not set in .env")
        print("Edit .env and add your wallet private key, then re-run.")
        sys.exit(1)

    if not wallet_address or wallet_address == "0x...":
        print("ERROR: POLYMARKET_WALLET_ADDRESS not set in .env")
        sys.exit(1)

    print(f"Wallet address: {wallet_address}")
    print("Connecting to Polymarket CLOB API...")

    client = ClobClient(
        host="https://clob.polymarket.com",
        key=private_key,
        chain_id=137,
        signature_type=0,  # EOA wallet
    )

    print("Deriving L2 API credentials (this signs a message with your private key)...")
    try:
        creds = client.create_or_derive_api_creds()
    except Exception as e:
        print(f"ERROR: Failed to derive credentials: {e}")
        print(
            "\nTroubleshooting:"
            "\n  1. Ensure your private key starts with 0x"
            "\n  2. Ensure the wallet has a Polymarket account (visit polymarket.com once)"
            "\n  3. Check your internet connection"
        )
        sys.exit(1)

    print("\n" + "=" * 60)
    print("SUCCESS! Add these to your .env file:")
    print("=" * 60)
    print(f"POLYMARKET_API_KEY={creds.api_key}")
    print(f"POLYMARKET_API_SECRET={creds.api_secret}")
    print(f"POLYMARKET_API_PASSPHRASE={creds.api_passphrase}")
    print("=" * 60)
    print("\nOnce saved to .env, your MCP server will use these for all trading operations.")

    # Verify the credentials work
    print("\nVerifying credentials...")
    client.set_api_creds(creds)
    try:
        balance = client.get_balance_allowance(params={"asset_type": 0})
        usdc_balance = float(balance.get("balance", 0)) / 1e6
        print(f"Balance check passed. USDC available: ${usdc_balance:.2f}")
    except Exception as e:
        print(f"WARNING: Balance check failed ({e}) — credentials may still be valid.")

    print("\nSetup complete!")


if __name__ == "__main__":
    main()
