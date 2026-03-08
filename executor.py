"""
Execution layer for statalyzer.
Paper mode: simulates fills with slippage model.
Quote mode: fetches real Jupiter quotes for pricing (no execution).
Live mode: Jupiter swap API + swQoS submission (same infra as arbito).
"""

import base64
import json
import logging
import os
import random
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Fill:
    token_mint: str
    side: str           # "buy" or "sell"
    price: float        # fill price after slippage
    quantity: float     # human-readable amount
    quantity_raw: int   # raw lamports/units
    slippage_bps: float # actual slippage applied
    timestamp: int
    tx_signature: Optional[str] = None  # None for paper


@dataclass
class PairExecution:
    fill_a: Fill
    fill_b: Fill
    is_paper: bool = True
    estimated_fees_usd: float = 0.0  # total estimated tx fees for both legs


class PaperExecutor:
    """Simulates trade execution with a random slippage model."""

    def __init__(self, config, db):
        self.config = config
        self.db = db

    def execute_entry(self, signal, position_size, price_a: float, price_b: float) -> PairExecution:
        """Simulate entry fills for both legs."""
        from signals import SignalType

        now = int(time.time())

        # For ENTRY_LONG (z < -threshold): buy A, sell B
        # For ENTRY_SHORT (z > +threshold): sell A, buy B
        if signal.signal_type == SignalType.ENTRY_LONG:
            side_a, side_b = "buy", "sell"
        else:
            side_a, side_b = "sell", "buy"

        slip_a = self._random_slippage()
        slip_b = self._random_slippage()

        # Apply slippage: buying costs more, selling gets less
        fill_price_a = price_a * (1 + slip_a / 10000) if side_a == "buy" else price_a * (1 - slip_a / 10000)
        fill_price_b = price_b * (1 + slip_b / 10000) if side_b == "buy" else price_b * (1 - slip_b / 10000)

        fill_a = Fill(
            token_mint=signal.token_a_mint,
            side=side_a,
            price=fill_price_a,
            quantity=position_size.token_a_amount,
            quantity_raw=position_size.token_a_raw,
            slippage_bps=slip_a,
            timestamp=now,
        )
        fill_b = Fill(
            token_mint=signal.token_b_mint,
            side=side_b,
            price=fill_price_b,
            quantity=position_size.token_b_amount,
            quantity_raw=position_size.token_b_raw,
            slippage_bps=slip_b,
            timestamp=now,
        )

        return PairExecution(fill_a=fill_a, fill_b=fill_b, is_paper=True)

    def execute_exit(self, position, price_a: float, price_b: float) -> PairExecution:
        """Simulate exit fills for both legs (reverse of entry)."""
        now = int(time.time())

        # Reverse of entry: long exit = sell A, buy B
        if position.direction == "long":
            side_a, side_b = "sell", "buy"
        else:
            side_a, side_b = "buy", "sell"

        slip_a = self._random_slippage()
        slip_b = self._random_slippage()

        fill_price_a = price_a * (1 + slip_a / 10000) if side_a == "buy" else price_a * (1 - slip_a / 10000)
        fill_price_b = price_b * (1 + slip_b / 10000) if side_b == "buy" else price_b * (1 - slip_b / 10000)

        fill_a = Fill(
            token_mint=position.token_a_mint,
            side=side_a,
            price=fill_price_a,
            quantity=position.quantity_a,
            quantity_raw=position.quantity_a_raw,
            slippage_bps=slip_a,
            timestamp=now,
        )
        fill_b = Fill(
            token_mint=position.token_b_mint,
            side=side_b,
            price=fill_price_b,
            quantity=position.quantity_b,
            quantity_raw=position.quantity_b_raw,
            slippage_bps=slip_b,
            timestamp=now,
        )

        return PairExecution(fill_a=fill_a, fill_b=fill_b, is_paper=True)

    def log_execution(self, position_id: int, execution: PairExecution, slot: int):
        """Persist both fills to the execution_log table."""
        for leg, fill in [("A", execution.fill_a), ("B", execution.fill_b)]:
            self.db.save_execution(
                position_id=position_id,
                leg=leg,
                side=fill.side,
                token_mint=fill.token_mint,
                amount_raw=fill.quantity_raw,
                price=fill.price,
                dex="paper",
                pool_address="",
                signature=fill.tx_signature or "",
                slot=slot,
                timestamp=fill.timestamp,
                slippage_bps=fill.slippage_bps,
                fee_lamports=0,
                is_paper=True,
            )

    def _random_slippage(self) -> float:
        """Random slippage in bps, normal distribution centered at 0."""
        return abs(random.gauss(0, self.config.slippage_bps / 2))


class QuoteExecutor:
    """Uses Jupiter Price API v3 for real USD pricing instead of simulated slippage.
    Tracks positions like paper mode but with actual market prices."""

    PRICE_API_URL = "https://api.jup.ag/price/v3"
    BASE_TX_FEE_LAMPORTS = 5000  # Solana base fee per transaction

    def __init__(self, config, db):
        self.config = config
        self.db = db
        self._jupiter_api_key = os.getenv('JUPITER_API_KEY', '')
        self.sol_usd_price: float = 0.0  # set by statalyzer main loop

    def _estimate_fees_usd(self) -> float:
        """Estimate total transaction fees in USD for a pair trade (2 swap legs).
        With Jito bundles there's also a 3rd tip transaction per bundle."""
        if self.sol_usd_price <= 0:
            return 0.0
        per_leg_lamports = self.BASE_TX_FEE_LAMPORTS + self.config.priority_fee_lamports
        total_lamports = per_leg_lamports * 2  # two swap legs
        if self.config.use_jito:
            # Jito tip is paid once per bundle (not per leg), plus base fee for the tip tx
            total_lamports += self.config.jito_tip_lamports + self.BASE_TX_FEE_LAMPORTS
        return (total_lamports / 1e9) * self.sol_usd_price

    def _get_prices(self, *mints: str) -> dict:
        """Fetch USD prices for one or more token mints via Jupiter Price API v3.
        Returns dict of {mint: usd_price}."""
        ids = ",".join(mints)
        url = f"{self.PRICE_API_URL}?ids={ids}"
        try:
            headers = {
                "Accept": "application/json",
                "User-Agent": "statalyzer/1.0",
            }
            if self._jupiter_api_key:
                headers["x-api-key"] = self._jupiter_api_key
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            prices = {}
            for mint in mints:
                entry = data.get(mint)
                if entry and entry.get("usdPrice"):
                    prices[mint] = float(entry["usdPrice"])
            return prices
        except Exception as e:
            logger.warning(f"Jupiter price API failed: {e}")
            return {}

    def _validate_price(self, mint: str, jup_price: float, onchain_price: float) -> Optional[float]:
        """Cross-validate Jupiter vs on-chain price. Returns best price or None if both suspect."""
        if not jup_price and (not onchain_price or onchain_price <= 0):
            logger.warning(f"No price available for {mint[:8]}..")
            return None
        if not jup_price:
            logger.warning(f"Jupiter price failed for {mint[:8]}.., using on-chain ${onchain_price:.6f}")
            return onchain_price
        if not onchain_price or onchain_price <= 0:
            return jup_price
        # If Jupiter and on-chain differ by >3x, REJECT — can't trust either source
        ratio = jup_price / onchain_price
        if ratio > 3.0 or ratio < 1/3.0:
            logger.warning(f"Jupiter/on-chain price mismatch for {mint[:8]}..: "
                           f"Jupiter=${jup_price:.6f} vs on-chain=${onchain_price:.6f} "
                           f"(ratio={ratio:.1f}x) — REJECTING (can't trust either)")
            return None
        logger.info(f"Jupiter price {mint[:8]}.. ${jup_price:.6f} (on-chain ${onchain_price:.6f})")
        return jup_price

    def _make_fill(self, token_mint: str, side: str, price: float,
                   quantity: float, quantity_raw: int) -> Fill:
        return Fill(
            token_mint=token_mint, side=side, price=price,
            quantity=quantity, quantity_raw=quantity_raw,
            slippage_bps=0, timestamp=int(time.time()),
        )

    def execute_entry(self, signal, position_size, price_a: float, price_b: float) -> Optional[PairExecution]:
        """Get Jupiter prices for both tokens and create fills. Returns None if prices unreliable."""
        from signals import SignalType

        if signal.signal_type == SignalType.ENTRY_LONG:
            side_a, side_b = "buy", "sell"
        else:
            side_a, side_b = "sell", "buy"

        # Batch fetch both prices in one call
        prices = self._get_prices(signal.token_a_mint, signal.token_b_mint)
        jup_price_a = prices.get(signal.token_a_mint)
        jup_price_b = prices.get(signal.token_b_mint)

        fill_price_a = self._validate_price(signal.token_a_mint, jup_price_a, price_a)
        fill_price_b = self._validate_price(signal.token_b_mint, jup_price_b, price_b)

        if fill_price_a is None or fill_price_b is None:
            logger.warning(f"Skipping entry {signal.pair_key}: unreliable prices")
            return None

        fill_a = self._make_fill(signal.token_a_mint, side_a, fill_price_a,
                                 position_size.token_a_amount, position_size.token_a_raw)
        fill_b = self._make_fill(signal.token_b_mint, side_b, fill_price_b,
                                 position_size.token_b_amount, position_size.token_b_raw)

        return PairExecution(fill_a=fill_a, fill_b=fill_b, is_paper=True,
                             estimated_fees_usd=self._estimate_fees_usd())

    def execute_exit(self, position, price_a: float, price_b: float) -> PairExecution:
        """Get Jupiter prices for both tokens on exit.
        For exits we must close — use entry price as fallback if both sources unreliable."""
        if position.direction == "long":
            side_a, side_b = "sell", "buy"
        else:
            side_a, side_b = "buy", "sell"

        prices = self._get_prices(position.token_a_mint, position.token_b_mint)
        fill_price_a = self._validate_price(position.token_a_mint,
                                            prices.get(position.token_a_mint), price_a)
        fill_price_b = self._validate_price(position.token_b_mint,
                                            prices.get(position.token_b_mint), price_b)

        # For exits, fall back to entry price (0 P&L) rather than refusing to close
        if fill_price_a is None:
            logger.warning(f"Exit price unreliable for {position.token_a_mint[:8]}.., using entry price")
            fill_price_a = position.entry_price_a
        if fill_price_b is None:
            logger.warning(f"Exit price unreliable for {position.token_b_mint[:8]}.., using entry price")
            fill_price_b = position.entry_price_b

        fill_a = self._make_fill(position.token_a_mint, side_a, fill_price_a,
                                 position.quantity_a, position.quantity_a_raw)
        fill_b = self._make_fill(position.token_b_mint, side_b, fill_price_b,
                                 position.quantity_b, position.quantity_b_raw)

        return PairExecution(fill_a=fill_a, fill_b=fill_b, is_paper=True,
                             estimated_fees_usd=self._estimate_fees_usd())

    def log_execution(self, position_id: int, execution: PairExecution, slot: int):
        """Persist both fills to the execution_log table."""
        for leg, fill in [("A", execution.fill_a), ("B", execution.fill_b)]:
            self.db.save_execution(
                position_id=position_id,
                leg=leg,
                side=fill.side,
                token_mint=fill.token_mint,
                amount_raw=fill.quantity_raw,
                price=fill.price,
                dex="jupiter_price",
                pool_address="",
                signature=fill.tx_signature or "",
                slot=slot,
                timestamp=fill.timestamp,
                slippage_bps=fill.slippage_bps,
                fee_lamports=0,
                is_paper=True,
            )


class LiveExecutor:
    """Live execution via Jupiter swap API + swQoS submission.
    Uses arbito's keypair loading and transaction submission patterns.
    Routes all swaps through SOL (native wallet currency)."""

    def __init__(self, config, db):
        self.config = config
        self.db = db
        self._jupiter_quote_url = os.getenv('JUPITER_QUOTE_URL', 'https://lite-api.jup.ag').rstrip('/')
        self._jupiter_swap_url = os.getenv('JUPITER_SWAP_URL', 'https://lite-api.jup.ag').rstrip('/')
        self._jupiter_api_key = os.getenv('JUPITER_API_KEY', '')
        self._swqos_endpoint = os.getenv('SWQOS_ENDPOINT', config.rpc_url)
        self._rpc_endpoint = config.rpc_url
        self.sol_usd_price: float = 0.0  # set by statalyzer main loop

        # Load keypair (same pattern as arbito/config.py)
        import base58
        from solders.keypair import Keypair
        kp_path = config.wallet_keypair_path
        if not kp_path:
            raise RuntimeError("WALLET_KEYPAIR_PATH required for live trading")
        with open(kp_path, 'r') as f:
            keypair_data = json.load(f)
        self._keypair = Keypair.from_bytes(bytes(keypair_data))
        self._pubkey = str(self._keypair.pubkey())
        logger.info(f"Live executor loaded wallet: {self._pubkey[:8]}..{self._pubkey[-4:]}")

    def _get_quote(self, input_mint: str, output_mint: str, amount_raw: int) -> Optional[dict]:
        """Fetch a quote from Jupiter API."""
        params = (
            f"?inputMint={input_mint}"
            f"&outputMint={output_mint}"
            f"&amount={amount_raw}"
            f"&slippageBps={self.config.slippage_bps}"
        )
        url = f"{self._jupiter_quote_url}/quote/v6{params}"
        try:
            headers = {"Accept": "application/json"}
            if self._jupiter_api_key:
                headers["x-api-key"] = self._jupiter_api_key
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except Exception as e:
            logger.error(f"Jupiter quote failed: {e}")
            return None

    def _get_swap_transaction(self, quote: dict) -> Optional[str]:
        """POST quote to Jupiter swap API to get a serialized transaction."""
        payload = json.dumps({
            "quoteResponse": quote,
            "userPublicKey": self._pubkey,
            "wrapAndUnwrapSol": True,
            "dynamicComputeUnitLimit": True,
            "prioritizationFeeLamports": self.config.priority_fee_lamports,
        }).encode()
        try:
            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            if self._jupiter_api_key:
                headers["x-api-key"] = self._jupiter_api_key
            req = urllib.request.Request(
                f"{self._jupiter_swap_url}/swap/v6",
                data=payload,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                return data.get("swapTransaction")
        except Exception as e:
            logger.error(f"Jupiter swap API failed: {e}")
            return None

    async def _submit_transaction(self, signed_tx_b64: str) -> tuple:
        """Submit signed transaction via swQoS endpoint (same as arbito/jito.py)."""
        import aiohttp
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendTransaction",
            "params": [
                signed_tx_b64,
                {"encoding": "base64", "skipPreflight": True, "maxRetries": 2},
            ],
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._swqos_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("result"):
                            return True, result["result"], None
                        error = result.get("error", {})
                        return False, None, error.get("message", str(error))
                    return False, None, f"HTTP {response.status}"
        except Exception as e:
            return False, None, str(e)

    async def _confirm_transaction(self, signature: str, timeout_s: int = 30) -> bool:
        """Poll RPC for transaction confirmation."""
        import aiohttp, asyncio
        deadline = time.time() + timeout_s
        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "getSignatureStatuses",
            "params": [[signature], {"searchTransactionHistory": False}],
        }
        async with aiohttp.ClientSession() as session:
            while time.time() < deadline:
                try:
                    async with session.post(
                        self._rpc_endpoint, json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=5.0),
                    ) as resp:
                        result = await resp.json()
                        statuses = result.get("result", {}).get("value", [])
                        if statuses and statuses[0]:
                            status = statuses[0]
                            if status.get("err"):
                                logger.error(f"Transaction failed on-chain: {status['err']}")
                                return False
                            conf = status.get("confirmationStatus", "")
                            if conf in ("confirmed", "finalized"):
                                return True
                except Exception:
                    pass
                await asyncio.sleep(1.0)
        logger.warning(f"Transaction confirmation timeout: {signature}")
        return False

    def _execute_swap(self, input_mint: str, output_mint: str, amount_raw: int,
                      token_mint: str, side: str, quantity: float, quantity_raw: int,
                      price_fallback: float) -> tuple:
        """Get quote, build swap tx, sign, submit. Returns (Fill, signature)."""
        import asyncio
        import base64
        from solders.transaction import VersionedTransaction
        from position import get_decimals

        quote = self._get_quote(input_mint, output_mint, amount_raw)
        if not quote:
            raise RuntimeError(f"Jupiter quote failed for {input_mint[:8]}→{output_mint[:8]}")

        swap_tx_b64 = self._get_swap_transaction(quote)
        if not swap_tx_b64:
            raise RuntimeError(f"Jupiter swap tx failed for {input_mint[:8]}→{output_mint[:8]}")

        # Deserialize, sign, re-serialize
        tx_bytes = base64.b64decode(swap_tx_b64)
        tx = VersionedTransaction.from_bytes(tx_bytes)
        # Jupiter returns an unsigned tx; sign it with our keypair
        signed_tx = VersionedTransaction(tx.message, [self._keypair])
        signed_b64 = base64.b64encode(bytes(signed_tx)).decode()

        # Submit
        loop = asyncio.get_event_loop()
        success, signature, error = loop.run_until_complete(self._submit_transaction(signed_b64))
        if not success:
            raise RuntimeError(f"Transaction submission failed: {error}")

        logger.info(f"Swap submitted: {signature}")

        # Confirm
        confirmed = loop.run_until_complete(self._confirm_transaction(signature))
        if not confirmed:
            logger.warning(f"Swap not confirmed within timeout, proceeding with quote price")

        # Compute fill price from quote
        in_amt = int(quote.get("inAmount", 0))
        out_amt = int(quote.get("outAmount", 0))
        in_dec = get_decimals(input_mint)
        out_dec = get_decimals(output_mint)
        price_impact = float(quote.get("priceImpactPct", 0))

        # Price in USD: convert SOL amounts via SOL/USD price
        from constants import SOL_MINT
        if input_mint == SOL_MINT and out_amt > 0 and self.sol_usd_price > 0:
            # Bought token with SOL: price = (SOL spent * SOL/USD) / tokens received
            sol_spent = in_amt / 10**9
            fill_price = (sol_spent * self.sol_usd_price) / (out_amt / 10**out_dec)
        elif output_mint == SOL_MINT and in_amt > 0 and self.sol_usd_price > 0:
            # Sold token for SOL: price = (SOL received * SOL/USD) / tokens sold
            sol_received = out_amt / 10**9
            fill_price = (sol_received * self.sol_usd_price) / (in_amt / 10**in_dec)
        else:
            fill_price = price_fallback

        fill = Fill(
            token_mint=token_mint,
            side=side,
            price=fill_price,
            quantity=quantity,
            quantity_raw=quantity_raw,
            slippage_bps=abs(price_impact) * 10000,
            timestamp=int(time.time()),
            tx_signature=signature,
        )
        return fill, signature

    # --- Jito Bundle Methods ---

    def _build_tip_instruction(self):
        """Create a SOL transfer instruction to a random Jito tip account."""
        from solders.pubkey import Pubkey
        from solders.system_program import transfer, TransferParams
        from constants import JITO_TIP_ACCOUNTS

        tip_account = random.choice(list(JITO_TIP_ACCOUNTS))
        return transfer(TransferParams(
            from_pubkey=Pubkey.from_string(self._pubkey),
            to_pubkey=Pubkey.from_string(tip_account),
            lamports=self.config.jito_tip_lamports,
        ))

    def _submit_bundle(self, signed_txs_b64: list) -> Optional[str]:
        """Submit a bundle of base64-encoded signed transactions to Jito block engine.
        Returns bundle_id on success, None on failure."""
        url = f"{self.config.jito_block_engine}/api/v1/bundles"
        payload = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendBundle",
            "params": [signed_txs_b64],
        }).encode()
        try:
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                if result.get("result"):
                    bundle_id = result["result"]
                    logger.info(f"Jito bundle submitted: {bundle_id}")
                    return bundle_id
                error = result.get("error", {})
                logger.error(f"Jito sendBundle failed: {error}")
                return None
        except Exception as e:
            logger.error(f"Jito bundle submission failed: {e}")
            return None

    def _confirm_bundle(self, bundle_id: str, timeout_s: int = 30) -> tuple:
        """Poll Jito for bundle landing status.
        Returns (landed: bool, signatures: list)."""
        url = f"{self.config.jito_block_engine}/api/v1/bundles"
        deadline = time.time() + timeout_s

        while time.time() < deadline:
            payload = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getBundleStatuses",
                "params": [[bundle_id]],
            }).encode()
            try:
                req = urllib.request.Request(
                    url, data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    result = json.loads(resp.read())
                    statuses = result.get("result", {}).get("value", [])
                    if statuses and statuses[0]:
                        status = statuses[0]
                        conf = status.get("confirmation_status", "")
                        if conf in ("confirmed", "finalized"):
                            txs = status.get("transactions", [])
                            logger.info(f"Jito bundle landed: {bundle_id} ({conf}, {len(txs)} txs)")
                            return True, txs
                        if conf == "failed":
                            logger.error(f"Jito bundle failed: {bundle_id}")
                            return False, []
            except Exception as e:
                logger.warning(f"Jito bundle status check error: {e}")
            time.sleep(2.0)

        logger.warning(f"Jito bundle confirmation timeout: {bundle_id}")
        return False, []

    def _execute_swap_no_submit(self, input_mint: str, output_mint: str, amount_raw: int) -> tuple:
        """Get quote and build signed swap transaction without submitting.
        Returns (signed_tx_b64, quote) for use in bundles."""
        from solders.transaction import VersionedTransaction

        quote = self._get_quote(input_mint, output_mint, amount_raw)
        if not quote:
            raise RuntimeError(f"Jupiter quote failed for {input_mint[:8]}→{output_mint[:8]}")

        swap_tx_b64 = self._get_swap_transaction(quote)
        if not swap_tx_b64:
            raise RuntimeError(f"Jupiter swap tx failed for {input_mint[:8]}→{output_mint[:8]}")

        # Deserialize, sign, re-serialize
        tx_bytes = base64.b64decode(swap_tx_b64)
        tx = VersionedTransaction.from_bytes(tx_bytes)
        signed_tx = VersionedTransaction(tx.message, [self._keypair])
        signed_b64 = base64.b64encode(bytes(signed_tx)).decode()

        return signed_b64, quote

    def _fill_from_quote(self, quote: dict, token_mint: str, side: str,
                         quantity: float, quantity_raw: int,
                         price_fallback: float, tx_signature: str = None) -> Fill:
        """Create a Fill from a Jupiter quote response."""
        from position import get_decimals
        from constants import SOL_MINT

        in_amt = int(quote.get("inAmount", 0))
        out_amt = int(quote.get("outAmount", 0))
        in_mint = quote.get("inputMint", "")
        out_mint = quote.get("outputMint", "")
        in_dec = get_decimals(in_mint)
        out_dec = get_decimals(out_mint)
        price_impact = float(quote.get("priceImpactPct", 0))

        if in_mint == SOL_MINT and out_amt > 0 and self.sol_usd_price > 0:
            sol_spent = in_amt / 10**9
            fill_price = (sol_spent * self.sol_usd_price) / (out_amt / 10**out_dec)
        elif out_mint == SOL_MINT and in_amt > 0 and self.sol_usd_price > 0:
            sol_received = out_amt / 10**9
            fill_price = (sol_received * self.sol_usd_price) / (in_amt / 10**in_dec)
        else:
            fill_price = price_fallback

        return Fill(
            token_mint=token_mint, side=side, price=fill_price,
            quantity=quantity, quantity_raw=quantity_raw,
            slippage_bps=abs(price_impact) * 10000,
            timestamp=int(time.time()), tx_signature=tx_signature,
        )

    def _execute_pair_bundle(self, legs: list) -> PairExecution:
        """Execute both legs atomically via Jito bundle.
        legs: list of (input_mint, output_mint, amount_raw, token_mint, side, quantity, quantity_raw, price_fallback)
        """
        # Build signed transactions for both legs
        signed_txs = []
        quotes = []
        for input_mint, output_mint, amount_raw, *_ in legs:
            signed_b64, quote = self._execute_swap_no_submit(input_mint, output_mint, amount_raw)
            signed_txs.append(signed_b64)
            quotes.append(quote)

        # TODO: Ideally add tip instruction to the last transaction.
        # For now, create a separate tip transaction as the third tx in the bundle.
        # This is simpler and Jito supports up to 5 txs per bundle.
        from solders.pubkey import Pubkey
        from solders.system_program import transfer, TransferParams
        from solders.transaction import Transaction
        from solders.message import Message
        from solders.hash import Hash
        from constants import JITO_TIP_ACCOUNTS

        tip_account = random.choice(list(JITO_TIP_ACCOUNTS))
        tip_ix = transfer(TransferParams(
            from_pubkey=Pubkey.from_string(self._pubkey),
            to_pubkey=Pubkey.from_string(tip_account),
            lamports=self.config.jito_tip_lamports,
        ))

        # Get recent blockhash for tip tx
        try:
            payload = json.dumps({
                "jsonrpc": "2.0", "id": 1,
                "method": "getLatestBlockhash",
                "params": [{"commitment": "confirmed"}],
            }).encode()
            req = urllib.request.Request(
                self._rpc_endpoint, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                blockhash = result["result"]["value"]["blockhash"]
        except Exception as e:
            raise RuntimeError(f"Failed to get blockhash for tip tx: {e}")

        msg = Message.new_with_blockhash(
            [tip_ix],
            Pubkey.from_string(self._pubkey),
            Hash.from_string(blockhash),
        )
        tip_tx = Transaction.new_unsigned(msg)
        tip_tx.sign([self._keypair], Hash.from_string(blockhash))
        tip_b64 = base64.b64encode(bytes(tip_tx)).decode()
        signed_txs.append(tip_b64)

        # Submit bundle
        bundle_id = self._submit_bundle(signed_txs)
        if not bundle_id:
            raise RuntimeError("Jito bundle submission failed")

        # Confirm bundle
        landed, tx_sigs = self._confirm_bundle(bundle_id)
        if not landed:
            raise RuntimeError(f"Jito bundle did not land: {bundle_id}")

        # Build fills from quotes
        sig_a = tx_sigs[0] if len(tx_sigs) > 0 else None
        sig_b = tx_sigs[1] if len(tx_sigs) > 1 else None
        _, _, _, token_mint_a, side_a, qty_a, qty_raw_a, price_fb_a = legs[0]
        _, _, _, token_mint_b, side_b, qty_b, qty_raw_b, price_fb_b = legs[1]

        fill_a = self._fill_from_quote(quotes[0], token_mint_a, side_a, qty_a, qty_raw_a, price_fb_a, sig_a)
        fill_b = self._fill_from_quote(quotes[1], token_mint_b, side_b, qty_b, qty_raw_b, price_fb_b, sig_b)

        logger.info(f"Bundle landed: {side_a} {token_mint_a[:8]}.. @ ${fill_a.price:.6f}, "
                     f"{side_b} {token_mint_b[:8]}.. @ ${fill_b.price:.6f}")

        return PairExecution(fill_a=fill_a, fill_b=fill_b, is_paper=False)

    # --- Entry/Exit Methods ---

    def execute_entry(self, signal, position_size, price_a: float, price_b: float) -> PairExecution:
        """Execute entry via Jupiter swaps (Jito bundle or sequential).
        Routes through SOL: buy both tokens with SOL on entry."""
        from signals import SignalType
        from constants import SOL_MINT

        if signal.signal_type == SignalType.ENTRY_LONG:
            side_a, side_b = "buy", "sell"
        else:
            side_a, side_b = "sell", "buy"

        # For token/token stat-arb, entry always buys both tokens from SOL.
        # "buy" = SOL → Token, "sell" = SOL → Token (we buy both sides on entry)
        # The direction (long/short) only affects P&L calculation, not execution.
        # Convert USD value to SOL lamports for input amount
        def _usd_to_sol_lamports(usd_value: float) -> int:
            if self.sol_usd_price <= 0:
                raise RuntimeError("SOL/USD price not available for live execution")
            return int((usd_value / self.sol_usd_price) * 10**9)

        sol_amount_a = _usd_to_sol_lamports(position_size.token_a_amount * price_a)
        sol_amount_b = _usd_to_sol_lamports(position_size.token_b_amount * price_b)

        if self.config.use_jito:
            leg_a = (SOL_MINT, signal.token_a_mint, sol_amount_a,
                     signal.token_a_mint, side_a, position_size.token_a_amount, position_size.token_a_raw, price_a)
            leg_b = (SOL_MINT, signal.token_b_mint, sol_amount_b,
                     signal.token_b_mint, side_b, position_size.token_b_amount, position_size.token_b_raw, price_b)
            return self._execute_pair_bundle([leg_a, leg_b])

        # Sequential fallback (no Jito)
        fill_a, sig_a = self._execute_swap(
            SOL_MINT, signal.token_a_mint, sol_amount_a,
            signal.token_a_mint, side_a,
            position_size.token_a_amount, position_size.token_a_raw, price_a)
        logger.info(f"Leg A done: {side_a} {signal.token_a_mint[:8]}.. @ ${fill_a.price:.6f} | tx: {sig_a}")

        fill_b, sig_b = self._execute_swap(
            SOL_MINT, signal.token_b_mint, sol_amount_b,
            signal.token_b_mint, side_b,
            position_size.token_b_amount, position_size.token_b_raw, price_b)
        logger.info(f"Leg B done: {side_b} {signal.token_b_mint[:8]}.. @ ${fill_b.price:.6f} | tx: {sig_b}")

        return PairExecution(fill_a=fill_a, fill_b=fill_b, is_paper=False)

    def execute_exit(self, position, price_a: float, price_b: float) -> PairExecution:
        """Execute exit via Jupiter swaps (Jito bundle or sequential).
        Routes through SOL: sell both tokens back to SOL on exit."""
        from constants import SOL_MINT

        if position.direction == "long":
            side_a, side_b = "sell", "buy"
        else:
            side_a, side_b = "buy", "sell"

        # Exit always sells both tokens back to SOL
        if self.config.use_jito:
            leg_a = (position.token_a_mint, SOL_MINT, position.quantity_a_raw,
                     position.token_a_mint, side_a, position.quantity_a, position.quantity_a_raw, price_a)
            leg_b = (position.token_b_mint, SOL_MINT, position.quantity_b_raw,
                     position.token_b_mint, side_b, position.quantity_b, position.quantity_b_raw, price_b)
            return self._execute_pair_bundle([leg_a, leg_b])

        # Sequential fallback (no Jito)
        fill_a, sig_a = self._execute_swap(
            position.token_a_mint, SOL_MINT, position.quantity_a_raw,
            position.token_a_mint, side_a,
            position.quantity_a, position.quantity_a_raw, price_a)

        fill_b, sig_b = self._execute_swap(
            position.token_b_mint, SOL_MINT, position.quantity_b_raw,
            position.token_b_mint, side_b,
            position.quantity_b, position.quantity_b_raw, price_b)

        return PairExecution(fill_a=fill_a, fill_b=fill_b, is_paper=False)

    def log_execution(self, position_id: int, execution: PairExecution, slot: int):
        """Persist both fills to the execution_log table."""
        for leg, fill in [("A", execution.fill_a), ("B", execution.fill_b)]:
            self.db.save_execution(
                position_id=position_id,
                leg=leg,
                side=fill.side,
                token_mint=fill.token_mint,
                amount_raw=fill.quantity_raw,
                price=fill.price,
                dex="jupiter_live",
                pool_address="",
                signature=fill.tx_signature or "",
                slot=slot,
                timestamp=fill.timestamp,
                slippage_bps=fill.slippage_bps,
                fee_lamports=self.config.priority_fee_lamports,
                is_paper=False,
            )
