"""
Execution layer for statalyzer.
Paper mode: simulates fills with slippage model.
Quote mode: fetches real Jupiter quotes for pricing (no execution).
Live mode: Jupiter swap API + Lunar Lander sendBundle submission.

Supports 2-4 token baskets.
"""

import base64
import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


def _ms(t0: float, t1: float) -> int:
    """Convert monotonic time delta to milliseconds."""
    return int((t1 - t0) * 1000)


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
class BasketExecution:
    fills: List[Fill]
    is_paper: bool = True
    estimated_fees_usd: float = 0.0


class PaperExecutor:
    """Simulates trade execution with a random slippage model."""

    def __init__(self, config, db):
        self.config = config
        self.db = db

    async def execute_entry(self, signal, position_size, prices: List[float]) -> BasketExecution:
        """Simulate entry fills for all legs."""
        from signals import SignalType

        now = int(time.time())
        hr = signal.hedge_ratios

        fills = []
        for i in range(signal.basket_size):
            # Determine side from hedge ratio sign and signal direction
            signed_hr = hr[i] if signal.signal_type == SignalType.ENTRY_LONG else -hr[i]
            side = "buy" if signed_hr > 0 else "sell"

            slip = self._random_slippage()
            fill_price = prices[i] * (1 + slip / 10000) if side == "buy" else prices[i] * (1 - slip / 10000)

            fills.append(Fill(
                token_mint=signal.mints[i],
                side=side,
                price=fill_price,
                quantity=position_size.amounts[i],
                quantity_raw=position_size.amounts_raw[i],
                slippage_bps=slip,
                timestamp=now,
            ))

        return BasketExecution(fills=fills, is_paper=True)

    async def execute_exit(self, position, prices: List[float]) -> BasketExecution:
        """Simulate exit fills for all legs (reverse of entry)."""
        now = int(time.time())
        hr = position.hedge_ratios

        fills = []
        for i in range(position.basket_size):
            # Reverse of entry: flip the sign
            signed_hr = hr[i] if position.direction == "long" else -hr[i]
            side = "sell" if signed_hr > 0 else "buy"  # reverse: sell what we bought

            slip = self._random_slippage()
            fill_price = prices[i] * (1 + slip / 10000) if side == "buy" else prices[i] * (1 - slip / 10000)

            fills.append(Fill(
                token_mint=position.mints[i],
                side=side,
                price=fill_price,
                quantity=position.quantities[i],
                quantity_raw=position.quantities_raw[i],
                slippage_bps=slip,
                timestamp=now,
            ))

        return BasketExecution(fills=fills, is_paper=True)

    def log_execution(self, position_id: int, execution: BasketExecution, slot: int):
        """Persist all fills to the execution_log table."""
        for i, fill in enumerate(execution.fills):
            self.db.save_execution(
                position_id=position_id,
                leg=str(i),
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

    def _estimate_fees_usd(self, num_legs: int = 2) -> float:
        """Estimate total transaction fees in USD for a basket trade.
        With Jito bundles there's also a tip transaction per bundle."""
        if self.sol_usd_price <= 0:
            return 0.0
        per_leg_lamports = self.BASE_TX_FEE_LAMPORTS + self.config.priority_fee_lamports
        total_lamports = per_leg_lamports * num_legs
        if self.config.use_lunar_lander:
            total_lamports += self.config.lunar_lander_tip_lamports + self.BASE_TX_FEE_LAMPORTS
        return (total_lamports / 1e9) * self.sol_usd_price

    def _get_prices(self, *mints: str) -> dict:
        """Fetch USD prices for one or more token mints via Jupiter Price API v3."""
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

    async def execute_entry(self, signal, position_size, prices: List[float]) -> Optional[BasketExecution]:
        """Get Jupiter prices for all tokens and create fills."""
        from signals import SignalType

        hr = signal.hedge_ratios

        # Batch fetch all prices
        jup_prices = self._get_prices(*signal.mints)

        fills = []
        for i in range(signal.basket_size):
            signed_hr = hr[i] if signal.signal_type == SignalType.ENTRY_LONG else -hr[i]
            side = "buy" if signed_hr > 0 else "sell"

            fill_price = self._validate_price(
                signal.mints[i], jup_prices.get(signal.mints[i]), prices[i])
            if fill_price is None:
                logger.warning(f"Skipping entry {signal.pair_key}: unreliable price for leg {i}")
                return None

            fills.append(self._make_fill(
                signal.mints[i], side, fill_price,
                position_size.amounts[i], position_size.amounts_raw[i]))

        return BasketExecution(
            fills=fills, is_paper=True,
            estimated_fees_usd=self._estimate_fees_usd(signal.basket_size))

    async def execute_exit(self, position, prices: List[float]) -> BasketExecution:
        """Get Jupiter prices for all tokens on exit."""
        hr = position.hedge_ratios

        jup_prices = self._get_prices(*position.mints)

        fills = []
        for i in range(position.basket_size):
            signed_hr = hr[i] if position.direction == "long" else -hr[i]
            side = "sell" if signed_hr > 0 else "buy"

            fill_price = self._validate_price(
                position.mints[i], jup_prices.get(position.mints[i]), prices[i])
            if fill_price is None:
                logger.warning(f"Exit price unreliable for {position.mints[i][:8]}.., using entry price")
                fill_price = position.entry_prices[i]

            fills.append(self._make_fill(
                position.mints[i], side, fill_price,
                position.quantities[i], position.quantities_raw[i]))

        return BasketExecution(
            fills=fills, is_paper=True,
            estimated_fees_usd=self._estimate_fees_usd(position.basket_size))

    def log_execution(self, position_id: int, execution: BasketExecution, slot: int):
        """Persist all fills to the execution_log table."""
        for i, fill in enumerate(execution.fills):
            self.db.save_execution(
                position_id=position_id,
                leg=str(i),
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
    """Live execution via Jupiter swap API + Lunar Lander bundle submission.
    Uses arbito's keypair loading and transaction submission patterns.
    Routes all swaps through SOL (native wallet currency).
    Supports 2-4 token baskets (max 5 txs/bundle = 4 swaps + 1 tip)."""

    def __init__(self, config, db):
        self.config = config
        self.db = db
        self._jupiter_quote_url = os.getenv('JUPITER_QUOTE_URL', 'https://lite-api.jup.ag').rstrip('/')
        self._jupiter_swap_url = os.getenv('JUPITER_SWAP_URL', 'https://lite-api.jup.ag').rstrip('/')
        self._jupiter_api_key = os.getenv('JUPITER_API_KEY', '')
        self._lunar_lander_base = config.lunar_lander_endpoint.rstrip('/')
        self._lunar_lander_api_key = config.lunar_lander_api_key or os.getenv('LUNAR_LANDER_API_KEY', '')
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
        """Fetch a quote from Jupiter API with retry on rate-limit (429)."""
        params = (
            f"?inputMint={input_mint}"
            f"&outputMint={output_mint}"
            f"&amount={amount_raw}"
            f"&slippageBps={self.config.slippage_bps}"
        )
        url = f"{self._jupiter_quote_url}/swap/v1/quote{params}"
        headers = {"Accept": "application/json"}
        if self._jupiter_api_key:
            headers["x-api-key"] = self._jupiter_api_key
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 3:
                    delay = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s
                    logger.warning(f"Jupiter quote 429, retrying in {delay:.1f}s (attempt {attempt + 1})")
                    time.sleep(delay)
                    continue
                logger.error(f"Jupiter quote failed: {e}")
                return None
            except Exception as e:
                logger.error(f"Jupiter quote failed: {e}")
                return None
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
                f"{self._jupiter_quote_url}/swap/v1/swap",
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

    def _send_transaction_sync(self, signed_tx_b64: str) -> str:
        """Submit signed transaction via Solana RPC sendTransaction (synchronous).
        Returns the signature string on success, raises RuntimeError on failure."""
        payload = json.dumps({
            "jsonrpc": "2.0", "id": 1,
            "method": "sendTransaction",
            "params": [signed_tx_b64, {"encoding": "base64", "skipPreflight": True,
                                       "maxRetries": 3}],
        }).encode()
        req = urllib.request.Request(self._rpc_endpoint, data=payload,
                                    headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
        if result.get("error"):
            raise RuntimeError(f"sendTransaction error: {result['error']}")
        return result.get("result", "")

    async def _submit_transaction(self, signed_tx_b64: str) -> tuple:
        """Submit signed transaction via Solana RPC sendTransaction."""
        import aiohttp
        url = self._rpc_endpoint
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
                    url,
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

    def _confirm_transaction_sync(self, signature: str, timeout_s: int = 30) -> tuple:
        """Poll RPC for transaction confirmation (synchronous, thread-safe).
        Returns (confirmed: bool, error: str|None).
        confirmed=True means landed successfully.
        confirmed=False + error=None means timeout (unknown state).
        confirmed=False + error=str means on-chain failure."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                result = self._rpc_call_sync("getSignatureStatuses",
                                             [[signature], {"searchTransactionHistory": False}])
                statuses = result.get("result", {}).get("value", [])
                if statuses and statuses[0]:
                    status = statuses[0]
                    if status.get("err"):
                        err_msg = str(status['err'])
                        logger.error(f"Transaction failed on-chain: {err_msg}")
                        return False, err_msg
                    conf = status.get("confirmationStatus", "")
                    if conf in ("confirmed", "finalized"):
                        return True, None
            except Exception:
                pass
            time.sleep(1.0)
        logger.warning(f"Transaction confirmation timeout: {signature}")
        return False, None

    def _get_tx_slot_sync(self, signature: str) -> Optional[int]:
        """Get the slot a transaction landed in (synchronous)."""
        try:
            result = self._rpc_call_sync("getSignatureStatuses",
                                         [[signature], {"searchTransactionHistory": False}])
            statuses = result.get("result", {}).get("value", [])
            if statuses and statuses[0]:
                return statuses[0].get("slot")
        except Exception:
            pass
        return None

    def _execute_swap_sync(self, input_mint: str, output_mint: str, amount_raw: int,
                           token_mint: str, side: str, quantity: float, quantity_raw: int,
                           price_fallback: float) -> tuple:
        """Get quote, build swap tx, sign, submit via sendBundle, confirm — all synchronous.
        Designed to run in a thread pool so multiple legs execute in parallel.
        Returns (Fill, signature)."""
        from solders.transaction import VersionedTransaction
        from constants import SOL_MINT

        # Guard: block swaps FROM SOL if wallet balance would drop below $10
        if input_mint == SOL_MINT and self.sol_usd_price > 0:
            try:
                sol_balance = self.get_sol_balance_sync()
                swap_sol = amount_raw / 1e9
                remaining_usd = (sol_balance - swap_sol) * self.sol_usd_price
                if remaining_usd < 10.0:
                    raise RuntimeError(
                        f"SOL balance too low: {sol_balance:.6f} SOL "
                        f"(${sol_balance * self.sol_usd_price:.2f}) — "
                        f"swap would leave ${remaining_usd:.2f} < $10.00 minimum")
            except RuntimeError:
                raise
            except Exception as e:
                logger.warning(f"Could not check SOL balance: {e}")

        last_error = None
        for attempt in range(3):
            t0 = time.monotonic()

            quote = self._get_quote(input_mint, output_mint, amount_raw)
            if not quote:
                last_error = f"Jupiter quote failed for {input_mint[:8]}→{output_mint[:8]}"
                if attempt < 2:
                    logger.warning(f"{last_error}, retry {attempt + 1}")
                    time.sleep(0.5)
                    continue
                raise RuntimeError(last_error)

            t_quote = time.monotonic()

            swap_tx_b64 = self._get_swap_transaction(quote)
            if not swap_tx_b64:
                last_error = f"Jupiter swap tx failed for {input_mint[:8]}→{output_mint[:8]}"
                if attempt < 2:
                    logger.warning(f"{last_error}, retry {attempt + 1}")
                    time.sleep(0.5)
                    continue
                raise RuntimeError(last_error)

            t_swap_api = time.monotonic()

            tx_bytes = base64.b64decode(swap_tx_b64)
            tx = VersionedTransaction.from_bytes(tx_bytes)
            signed_tx = VersionedTransaction(tx.message, [self._keypair])
            signed_b64 = base64.b64encode(bytes(signed_tx)).decode()
            signature = str(signed_tx.signatures[0])

            t_sign = time.monotonic()
            submit_timestamp = time.time()

            if self.config.use_lunar_lander:
                # Build tip tx and submit as bundle
                tip_b64 = self._build_tip_tx()
                ok = self._submit_lunar_bundle([signed_b64, tip_b64])
                t_submit = time.monotonic()

                if not ok:
                    last_error = f"sendBundle failed for {input_mint[:8]}→{output_mint[:8]}"
                    logger.info(f"TX timing: quote={_ms(t0,t_quote)}ms swap_api={_ms(t_quote,t_swap_api)}ms "
                                f"sign={_ms(t_swap_api,t_sign)}ms submit={_ms(t_sign,t_submit)}ms (FAILED)")
                    if attempt < 2:
                        logger.warning(f"{last_error}, retry {attempt + 1}")
                        time.sleep(1.0)
                        continue
                    raise RuntimeError(last_error)

                logger.info(f"Swap bundle submitted: {signature}")
            else:
                # Direct sendTransaction (no tip, no bundle)
                try:
                    self._send_transaction_sync(signed_b64)
                    t_submit = time.monotonic()
                    logger.info(f"Swap tx submitted: {signature}")
                except RuntimeError as e:
                    t_submit = time.monotonic()
                    last_error = f"sendTransaction failed for {input_mint[:8]}→{output_mint[:8]}: {e}"
                    logger.info(f"TX timing: quote={_ms(t0,t_quote)}ms swap_api={_ms(t_quote,t_swap_api)}ms "
                                f"sign={_ms(t_swap_api,t_sign)}ms submit={_ms(t_sign,t_submit)}ms (FAILED)")
                    if attempt < 2:
                        logger.warning(f"{last_error}, retry {attempt + 1}")
                        time.sleep(1.0)
                        continue
                    raise RuntimeError(last_error)

            # Save transaction to DB (may fail from thread — non-critical)
            slot = 0
            try:
                self.db.save_transaction(submit_timestamp, slot, signature)
            except Exception:
                pass  # SQLite thread safety — will be saved on confirm

            # Confirm in same thread — blocks until confirmed or timeout
            confirmed, on_chain_err = self._confirm_transaction_sync(signature, timeout_s=30)
            t_confirm = time.monotonic()

            confirmation_timestamp = time.time() if confirmed else None

            logger.info(f"TX timing: quote={_ms(t0,t_quote)}ms swap_api={_ms(t_quote,t_swap_api)}ms "
                        f"sign={_ms(t_swap_api,t_sign)}ms submit={_ms(t_sign,t_submit)}ms "
                        f"confirm={_ms(t_submit,t_confirm)}ms total={_ms(t0,t_confirm)}ms")

            if on_chain_err:
                last_error = f"Swap failed on-chain: {on_chain_err} (tx: {signature})"
                if attempt < 2:
                    logger.warning(f"{last_error}, retry {attempt + 1}")
                    time.sleep(1.0)
                    continue
                raise RuntimeError(last_error)
            if not confirmed:
                last_error = f"Swap not confirmed after 30s (tx: {signature})"
                if attempt < 2:
                    logger.warning(f"{last_error}, retry {attempt + 1}")
                    time.sleep(1.0)
                    continue
                raise RuntimeError(last_error)

            # Update confirmation in DB
            if confirmed and confirmation_timestamp:
                try:
                    conf_slot = self._get_tx_slot_sync(signature)
                    self.db.confirm_transaction(signature, conf_slot or 0, confirmation_timestamp)
                except Exception as e:
                    logger.debug(f"Failed to update transaction confirmation: {e}")

            # Success — exit retry loop
            break

        else:
            # All 3 attempts exhausted without success
            raise RuntimeError(f"_execute_swap failed after 3 attempts: {last_error}")

        fill = self._fill_from_quote(quote, token_mint, side, quantity, quantity_raw,
                                     price_fallback, signature)
        return fill, signature

    def _fill_from_quote(self, quote: dict, token_mint: str, side: str,
                         quantity: float, quantity_raw: int,
                         price_fallback: float, tx_signature: str = None) -> Fill:
        """Create a Fill from a Jupiter quote response.
        Uses actual amounts from Jupiter quote instead of estimates so that
        position quantities match real wallet balances."""
        from position import get_decimals
        from constants import SOL_MINT

        in_amt = int(quote.get("inAmount", 0))
        out_amt = int(quote.get("outAmount", 0))
        in_mint = quote.get("inputMint", "")
        out_mint = quote.get("outputMint", "")
        in_dec = get_decimals(in_mint)
        out_dec = get_decimals(out_mint)
        price_impact = float(quote.get("priceImpactPct", 0))

        # Use actual swap amounts from Jupiter, not estimates
        if in_mint == SOL_MINT and out_amt > 0:
            # Entry: bought tokens with SOL — use actual received amount
            actual_quantity_raw = out_amt
            actual_quantity = out_amt / 10**out_dec
            if self.sol_usd_price > 0:
                sol_spent = in_amt / 10**9
                fill_price = (sol_spent * self.sol_usd_price) / actual_quantity
            else:
                fill_price = price_fallback
        elif out_mint == SOL_MINT and in_amt > 0:
            # Exit: sold tokens for SOL — use actual sold amount
            actual_quantity_raw = in_amt
            actual_quantity = in_amt / 10**in_dec
            if self.sol_usd_price > 0:
                sol_received = out_amt / 10**9
                fill_price = (sol_received * self.sol_usd_price) / actual_quantity
            else:
                fill_price = price_fallback
        else:
            actual_quantity = quantity
            actual_quantity_raw = quantity_raw
            fill_price = price_fallback

        return Fill(
            token_mint=token_mint, side=side, price=fill_price,
            quantity=actual_quantity, quantity_raw=actual_quantity_raw,
            slippage_bps=abs(price_impact) * 10000,
            timestamp=int(time.time()), tx_signature=tx_signature,
        )

    def _build_tip_tx(self) -> str:
        """Build a tip transaction to a random Lunar Lander moon* account.
        Returns base64-encoded signed transaction."""
        from solders.pubkey import Pubkey
        from solders.system_program import transfer, TransferParams
        from solders.transaction import VersionedTransaction
        from solders.message import MessageV0
        from solders.hash import Hash
        from constants import LUNAR_LANDER_TIP_ACCOUNTS

        tip_account = random.choice(LUNAR_LANDER_TIP_ACCOUNTS)
        tip_ix = transfer(TransferParams(
            from_pubkey=Pubkey.from_string(self._pubkey),
            to_pubkey=Pubkey.from_string(tip_account),
            lamports=self.config.lunar_lander_tip_lamports,
        ))

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

        tip_msg = MessageV0.try_compile(
            payer=Pubkey.from_string(self._pubkey),
            instructions=[tip_ix],
            address_lookup_table_accounts=[],
            recent_blockhash=Hash.from_string(blockhash),
        )
        tip_tx = VersionedTransaction(tip_msg, [self._keypair])
        tip_b64 = base64.b64encode(bytes(tip_tx)).decode()
        logger.info(f"Tip tx built: {self.config.lunar_lander_tip_lamports} lamports → {tip_account[:12]}...")
        return tip_b64

    # --- Jito Bundle Methods ---

    def _execute_swap_no_submit(self, input_mint: str, output_mint: str, amount_raw: int) -> tuple:
        """Get quote and build signed swap transaction without submitting.
        Returns (signed_tx_b64, quote, signature_str) for use in bundles."""
        from solders.transaction import VersionedTransaction

        quote = self._get_quote(input_mint, output_mint, amount_raw)
        if not quote:
            raise RuntimeError(f"Jupiter quote failed for {input_mint[:8]}→{output_mint[:8]}")

        swap_tx_b64 = self._get_swap_transaction(quote)
        if not swap_tx_b64:
            raise RuntimeError(f"Jupiter swap tx failed for {input_mint[:8]}→{output_mint[:8]}")

        tx_bytes = base64.b64decode(swap_tx_b64)
        tx = VersionedTransaction.from_bytes(tx_bytes)
        signed_tx = VersionedTransaction(tx.message, [self._keypair])
        signed_b64 = base64.b64encode(bytes(signed_tx)).decode()
        sig_str = str(signed_tx.signatures[0])

        return signed_b64, quote, sig_str

    def _submit_lunar_bundle(self, signed_txs_b64: list) -> bool:
        """Submit a bundle of base64-encoded signed transactions to Lunar Lander.
        Uses JSON-RPC format per https://docs.hellomoon.io/reference/send-bundle-api"""
        api_key = self._lunar_lander_api_key
        url = f"{self._lunar_lander_base}/sendbundle"
        if api_key:
            url += f"?api-key={api_key}"
        payload = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendBundle",
            "params": [
                signed_txs_b64,
                {"encoding": "base64"},
            ],
        }).encode()
        try:
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read())
                if body.get("error"):
                    logger.error(f"Lunar Lander bundle error: {body['error']}")
                    return False
                logger.info(f"Lunar Lander bundle submitted: {body.get('result', '')}")
                return True
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()
            except Exception:
                pass
            logger.error(f"Lunar Lander bundle HTTP {e.code}: {body[:500]}")
            return False
        except Exception as e:
            logger.error(f"Lunar Lander bundle submission failed: {e}")
            return False

    async def _execute_basket_bundle(self, legs: list) -> BasketExecution:
        """Execute legs via Lunar Lander bundles.
        Each leg is sent as its own [swap_tx, tip_tx] bundle for reliability.
        legs: list of (input_mint, output_mint, amount_raw, token_mint, side, quantity, quantity_raw, price_fallback)
        """
        all_fills = []
        for i, leg in enumerate(legs):
            input_mint, output_mint, amount_raw, token_mint, side, qty, qty_raw, price_fb = leg

            # Build and sign swap tx
            signed_b64, quote, sig_str = self._execute_swap_no_submit(input_mint, output_mint, amount_raw)

            # Build tip tx with fresh blockhash
            tip_b64 = self._build_tip_tx()

            # Submit [swap, tip] bundle
            ok = self._submit_lunar_bundle([signed_b64, tip_b64])
            if not ok:
                raise RuntimeError(f"Lunar Lander bundle failed for leg {i + 1}/{len(legs)} ({token_mint[:8]})")

            # Confirm landing
            confirmed, on_chain_err = self._confirm_transaction_sync(sig_str, timeout_s=30)
            if on_chain_err:
                raise RuntimeError(f"Leg {i + 1} failed on-chain: {on_chain_err} (tx: {sig_str})")
            if not confirmed:
                raise RuntimeError(f"Leg {i + 1} not confirmed after 30s (tx: {sig_str})")

            fill = self._fill_from_quote(quote, token_mint, side, qty, qty_raw, price_fb, sig_str)
            all_fills.append(fill)
            logger.info(f"Leg {i + 1}/{len(legs)} done: {side} {token_mint[:8]}.. @ ${fill.price:.6f} | tx: {sig_str[:20]}...")

        return BasketExecution(fills=all_fills, is_paper=False)

    # --- Entry/Exit Methods ---

    async def execute_entry(self, signal, position_size, prices: List[float]) -> BasketExecution:
        """Execute entry via Jupiter swaps (Lunar Lander bundle or sequential).
        Routes through SOL: buy all tokens with SOL on entry.
        All legs run in parallel via thread pool; each thread handles its own
        swap + confirmation."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        from signals import SignalType
        from constants import SOL_MINT

        hr = signal.hedge_ratios

        def _usd_to_sol_lamports(usd_value: float) -> int:
            if self.sol_usd_price <= 0:
                raise RuntimeError("SOL/USD price not available for live execution")
            return int((usd_value / self.sol_usd_price) * 10**9)

        legs = []
        sol_legs = []  # SOL legs need no swap
        for i in range(signal.basket_size):
            signed_hr = hr[i] if signal.signal_type == SignalType.ENTRY_LONG else -hr[i]
            side = "buy" if signed_hr > 0 else "sell"
            if signal.mints[i] == SOL_MINT:
                # No swap needed for SOL — we already hold it
                sol_legs.append((i, side, position_size.amounts[i],
                                 position_size.amounts_raw[i], prices[i]))
                continue
            sol_amount = _usd_to_sol_lamports(position_size.amounts[i] * prices[i])
            legs.append((i, SOL_MINT, signal.mints[i], sol_amount,
                         signal.mints[i], side, position_size.amounts[i],
                         position_size.amounts_raw[i], prices[i]))

        # Create paper fills for SOL legs (no swap needed)
        sol_fills = []
        for idx, side, qty, qty_raw, price in sol_legs:
            fill = Fill(
                token_mint=SOL_MINT, side=side, price=price,
                quantity=qty, quantity_raw=qty_raw,
                slippage_bps=0, timestamp=int(time.time()),
                tx_signature=None,
            )
            sol_fills.append(fill)
            logger.info(f"Leg {idx + 1}/{signal.basket_size}: SOL leg (no swap needed) @ ${price:.4f}")

        # Execute non-SOL legs in parallel via thread pool
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=max(len(legs), 1)) as pool:
            futures = []
            for leg in legs:
                idx, input_mint, output_mint, amount_raw, token_mint, side, qty, qty_raw, price_fb = leg
                fut = loop.run_in_executor(pool, self._execute_swap_sync,
                                           input_mint, output_mint, amount_raw,
                                           token_mint, side, qty, qty_raw, price_fb)
                futures.append((idx, fut, token_mint, side))

            fills = []
            errors = []
            for idx, fut, token_mint, side in futures:
                try:
                    fill, sig = await fut
                    logger.info(f"Leg {idx + 1}/{signal.basket_size} done: {side} {token_mint[:8]}.. @ ${fill.price:.6f} | tx: {sig}")
                    fills.append(fill)
                except RuntimeError as e:
                    logger.error(f"Leg {idx + 1}/{signal.basket_size} failed: {e}")
                    errors.append(str(e))

        # Merge SOL fills + swap fills
        fills = sol_fills + fills

        if not fills:
            raise RuntimeError(f"All entry legs failed — {'; '.join(errors)}")

        if errors:
            # Partial entry is invalid for pair trading — all legs must succeed
            logger.error(f"{len(errors)}/{len(legs)} entry legs failed — aborting entry. "
                         f"Successful legs will need manual cleanup. Errors: {'; '.join(errors)}")
            raise RuntimeError(f"Partial entry ({len(fills)}/{len(legs)} legs) — aborting: {'; '.join(errors)}")

        return BasketExecution(fills=fills, is_paper=False)

    async def execute_exit(self, position, prices: List[float]) -> BasketExecution:
        """Execute exit via Jupiter swaps (Lunar Lander bundle or sequential).
        Routes through SOL: sell all tokens back to SOL on exit.
        All legs run in parallel via thread pool. Falls back to paper fills
        for any leg that fails after retries."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        from constants import SOL_MINT

        hr = position.hedge_ratios

        legs = []
        sol_fills = []  # SOL legs need no swap
        for i in range(position.basket_size):
            signed_hr = hr[i] if position.direction == "long" else -hr[i]
            side = "sell" if signed_hr > 0 else "buy"
            if position.mints[i] == SOL_MINT:
                # No swap needed for SOL — already in SOL
                fill = Fill(
                    token_mint=SOL_MINT, side=side, price=prices[i],
                    quantity=position.quantities[i], quantity_raw=position.quantities_raw[i],
                    slippage_bps=0, timestamp=int(time.time()),
                    tx_signature=None,
                )
                sol_fills.append((i, fill))
                logger.info(f"Exit leg {i}: SOL leg (no swap needed) @ ${prices[i]:.4f}")
                continue
            legs.append((position.mints[i], SOL_MINT, position.quantities_raw[i],
                         position.mints[i], side, position.quantities[i],
                         position.quantities_raw[i], prices[i]))

        def _exit_leg_sync(i, leg):
            """Execute a single exit leg with slippage escalation. Runs in thread."""
            input_mint, output_mint, amount_raw, token_mint, side, qty, qty_raw, price_fb = leg
            saved_slippage = self.config.slippage_bps
            for attempt, multiplier in enumerate([1, 2, 4]):
                try:
                    self.config.slippage_bps = saved_slippage * multiplier
                    fill, sig = self._execute_swap_sync(
                        input_mint, output_mint, amount_raw,
                        token_mint, side, qty, qty_raw, price_fb)
                    if multiplier > 1:
                        logger.info(f"Exit leg {i} succeeded with {self.config.slippage_bps}bps slippage")
                    return fill
                except RuntimeError as e:
                    if '6024' in str(e) and multiplier < 4:
                        logger.warning(f"Exit leg {i} slippage error, retrying with "
                                       f"{saved_slippage * (multiplier * 2)}bps...")
                        continue
                    logger.error(f"Exit leg {i} failed after {attempt + 1} attempts: {e}")
                    break
                finally:
                    self.config.slippage_bps = saved_slippage
            # All attempts failed — return None so caller uses paper fill
            return None

        # Run all non-SOL exit legs in parallel
        if legs:
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=len(legs)) as pool:
                futures = [loop.run_in_executor(pool, _exit_leg_sync, i, leg)
                           for i, leg in enumerate(legs)]
                results = await asyncio.gather(*futures)
        else:
            results = []

        swap_fills = []
        for i, (fill, leg) in enumerate(zip(results, legs)):
            if fill is None:
                input_mint, output_mint, amount_raw, token_mint, side, qty, qty_raw, price_fb = leg
                logger.warning(f"Exit leg {i} {token_mint[:8]}.. using paper fill (on-chain failed)")
                fill = Fill(
                    token_mint=token_mint, side=side, price=price_fb,
                    quantity=qty, quantity_raw=qty_raw,
                    slippage_bps=0, timestamp=int(time.time()),
                )
            swap_fills.append(fill)

        # Merge SOL fills + swap fills in original leg order
        fills = [f for _, f in sol_fills] + swap_fills

        return BasketExecution(fills=fills, is_paper=False)

    def _rpc_call_sync(self, method, params):
        """Synchronous JSON-RPC call (for use in background threads)."""
        payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
        req = urllib.request.Request(
            self._rpc_endpoint, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def _get_token_balance_sync(self, mint):
        """Get wallet's on-chain balance of a specific token mint. Returns (raw_amount, decimals).
        Uses mint filter which works for both SPL Token and Token-2022."""
        result = self._rpc_call_sync("getTokenAccountsByOwner", [
            self._pubkey,
            {"mint": mint},
            {"encoding": "jsonParsed"},
        ])
        total_raw = 0
        decimals = 0
        for item in result.get("result", {}).get("value", []):
            info = item["account"]["data"]["parsed"]["info"]
            total_raw += int(info["tokenAmount"]["amount"])
            decimals = info["tokenAmount"]["decimals"]
        return total_raw, decimals

    def _sweep_token_sync(self, mint, amount_raw):
        """Swap a single token back to SOL via sendBundle. Synchronous, for background thread use.
        Retries with escalating slippage (500, 1000, 2000 bps) on 6024 errors."""
        from solders.transaction import VersionedTransaction
        from constants import SOL_MINT

        for slippage_bps in [500, 1000, 2000]:
            saved = self.config.slippage_bps
            self.config.slippage_bps = slippage_bps
            try:
                quote = self._get_quote(mint, SOL_MINT, amount_raw)
            finally:
                self.config.slippage_bps = saved
            if not quote:
                return False

            swap_tx_b64 = self._get_swap_transaction(quote)
            if not swap_tx_b64:
                return False

            tx_bytes = base64.b64decode(swap_tx_b64)
            tx = VersionedTransaction.from_bytes(tx_bytes)
            signed_tx = VersionedTransaction(tx.message, [self._keypair])
            signed_b64 = base64.b64encode(bytes(signed_tx)).decode()
            sig = str(signed_tx.signatures[0])

            # Build tip and submit as bundle
            tip_b64 = self._build_tip_tx()
            ok = self._submit_lunar_bundle([signed_b64, tip_b64])
            if not ok:
                logger.warning(f"Sweep bundle submit failed for {mint[:8]}..")
                return False

            logger.info(f"Sweep bundle submitted for {mint[:8]}.. ({slippage_bps}bps): {sig}")

            # Poll for confirmation
            deadline = time.time() + 30
            confirmed = False
            failed_6024 = False
            while time.time() < deadline:
                try:
                    status_result = self._rpc_call_sync("getSignatureStatuses",
                                                         [[sig], {"searchTransactionHistory": False}])
                    statuses = status_result.get("result", {}).get("value", [])
                    if statuses and statuses[0]:
                        if statuses[0].get("err"):
                            err = statuses[0]["err"]
                            err_str = str(err)
                            if "6024" in err_str and slippage_bps < 2000:
                                logger.warning(f"Sweep 6024 for {mint[:8]}.. at {slippage_bps}bps, retrying higher")
                                failed_6024 = True
                                break
                            logger.warning(f"Sweep tx failed on-chain for {mint[:8]}..: {err}")
                            return False
                        if statuses[0].get("confirmationStatus") in ("confirmed", "finalized"):
                            logger.info(f"Sweep confirmed for {mint[:8]}..")
                            return True
                except Exception:
                    pass
                time.sleep(1)

            if failed_6024:
                time.sleep(1)
                continue  # retry with higher slippage

            logger.warning(f"Sweep tx not confirmed within timeout for {mint[:8]}..")
            return False

        return False

    def get_sol_balance_sync(self):
        """Get wallet SOL balance in SOL units. Synchronous."""
        result = self._rpc_call_sync("getBalance", [self._pubkey])
        return result["result"]["value"] / 1e9

    def get_all_token_mints_sync(self):
        """Get all non-SOL token mints with balance in the wallet.
        Checks both SPL Token and Token-2022 programs."""
        from constants import SOL_MINT
        TOKEN_PROGRAMS = [
            "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
            "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",
        ]
        mints = []
        seen = set()
        for program in TOKEN_PROGRAMS:
            try:
                result = self._rpc_call_sync("getTokenAccountsByOwner", [
                    self._pubkey,
                    {"programId": program},
                    {"encoding": "jsonParsed"},
                ])
                for item in result.get("result", {}).get("value", []):
                    info = item["account"]["data"]["parsed"]["info"]
                    mint = info["mint"]
                    amt = int(info["tokenAmount"]["amount"])
                    if amt > 0 and mint != SOL_MINT and mint not in seen:
                        seen.add(mint)
                        mints.append(mint)
            except Exception as e:
                logger.warning(f"Failed to list token accounts for {program[:8]}..: {e}")
        return mints

    def sweep_and_update_capital(self, mints, portfolio, position_id=None):
        """Sweep remaining token balances back to SOL, then update portfolio stats.
        Updates the closed position's realized_pnl, total_realized_pnl, and capital
        so that P&L, win rate, hourly/daily projections all reflect sweep recovery.
        Runs synchronously — intended to be called from a background thread."""
        import sqlite3
        from constants import SOL_MINT

        sol_before = self.get_sol_balance_sync()

        swept_count = 0
        for mint in mints:
            if mint == SOL_MINT:
                continue
            try:
                balance, decimals = self._get_token_balance_sync(mint)
                if balance <= 0:
                    continue
                ui_amount = balance / 10**decimals if decimals > 0 else balance
                logger.info(f"Sweeping {ui_amount:.6f} of {mint[:8]}.. back to SOL")
                if self._sweep_token_sync(mint, balance):
                    swept_count += 1
                else:
                    logger.warning(f"Failed to sweep {mint[:8]}..")
            except Exception as e:
                logger.warning(f"Sweep error for {mint[:8]}..: {e}")
            time.sleep(1)  # Rate limit between sweeps

        # Measure recovered SOL and add to capital + position P&L
        sol_after = self.get_sol_balance_sync()
        recovered_sol = sol_after - sol_before
        if recovered_sol > 0 and self.sol_usd_price > 0:
            recovered_usd = recovered_sol * self.sol_usd_price

            # Use a thread-local SQLite connection (can't share across threads)
            try:
                conn = sqlite3.connect(self.db.db_path)
                conn.execute("PRAGMA journal_mode=WAL")

                # Update the closed position's realized_pnl in DB so all stats
                # (P&L, win rate, hourly/daily, profit factor) reflect the recovery
                if position_id is not None:
                    row = conn.execute(
                        "SELECT realized_pnl FROM positions WHERE id = ?",
                        (position_id,)
                    ).fetchone()
                    if row:
                        old_pnl = float(row[0])
                        new_pnl = old_pnl + recovered_usd
                        conn.execute(
                            "UPDATE positions SET realized_pnl = ? WHERE id = ?",
                            (new_pnl, position_id)
                        )
                        logger.info(f"Position {position_id} P&L updated: "
                                    f"${old_pnl:+.4f} → ${new_pnl:+.4f} (+${recovered_usd:.4f} sweep)")

                # Update total_realized_pnl and capital in config_state
                portfolio.total_realized_pnl += recovered_usd
                conn.execute(
                    "INSERT OR REPLACE INTO config_state (key, value) VALUES (?, ?)",
                    ('total_realized_pnl', str(portfolio.total_realized_pnl))
                )

                portfolio.initial_capital += recovered_usd
                conn.execute(
                    "INSERT OR REPLACE INTO config_state (key, value) VALUES (?, ?)",
                    ('initial_capital', str(portfolio.initial_capital))
                )

                conn.commit()
                conn.close()
            except Exception as e:
                logger.warning(f"Failed to update DB after sweep: {e}")

            logger.info(f"Sweep recovered {recovered_sol:.6f} SOL (${recovered_usd:.2f}) "
                        f"→ capital now ${portfolio.initial_capital:.2f}")
        elif swept_count > 0:
            logger.info(f"Swept {swept_count} tokens but SOL balance unchanged "
                        f"(before={sol_before:.6f}, after={sol_after:.6f})")

    def log_execution(self, position_id: int, execution: BasketExecution, slot: int):
        """Persist all fills to the execution_log table."""
        for i, fill in enumerate(execution.fills):
            self.db.save_execution(
                position_id=position_id,
                leg=str(i),
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
        sigs = [f.tx_signature for f in execution.fills if f.tx_signature]
        if sigs:
            logger.info(f"Position {position_id} txs: {sigs}")
