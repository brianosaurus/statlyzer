"""
Swap detector — adapted from arbito/swap_detector.py
Detects swaps and extracts pool/vault info from gRPC transaction data
"""

import base58
from typing import List, Dict, Optional, Any
from constants import (
    PUMPSWAP, RAYDIUM_AMM_V4, METEORA_DLMM, RAYDIUM_CLMM, RAYDIUM_CPMM, WHIRLPOOL,
    RAYDIUM_AMM_V4_PROGRAM, METEORA_DLMM_PROGRAM, METEORA_DAMM_PROGRAM,
    WHIRLPOOL_PROGRAM, ORCA_CLMM_PROGRAM, JUPITER_V6_PROGRAM, JUPITER_V4_PROGRAM,
    SERUM_V3_PROGRAM, SERUM_V2_PROGRAM, SERUM_V1_PROGRAM,
    PHOENIX_PROGRAM, OPENBOOK_PROGRAM, ORCA_V1_PROGRAM, ORCA_V2_PROGRAM,
    SUPPORTED_DEXS, SWAP_DISCRIMINATORS, DEX_PROGRAMS, SYSTEM_ACCOUNTS,
)


class SwapDetector:
    """Detects swaps and extracts pool addresses from Solana gRPC transactions"""

    @staticmethod
    def extract_bonding_curve(tx) -> Optional[str]:
        """Extract the bonding curve address from a PumpSwap transaction."""
        try:
            if not hasattr(tx.meta, 'pre_token_balances'):
                return None

            excluded_mints = {
                "So11111111111111111111111111111111111111112",
                "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            }

            for token_balance in tx.meta.pre_token_balances:
                if not hasattr(token_balance, 'owner') or not hasattr(token_balance, 'mint'):
                    continue
                if token_balance.mint in excluded_mints:
                    continue
                if (hasattr(token_balance, 'ui_token_amount') and
                        hasattr(token_balance.ui_token_amount, 'ui_amount')):
                    try:
                        ui_amount = float(token_balance.ui_token_amount.ui_amount)
                        if ui_amount > 1_000_000:
                            return token_balance.owner
                    except (ValueError, TypeError, AttributeError):
                        continue
        except Exception:
            pass
        return None

    @staticmethod
    def get_raydium_clmm_pool(tx) -> Optional[str]:
        """Find Raydium CLMM/CPMM pool address from token balance changes."""
        try:
            if not hasattr(tx, 'meta') or not hasattr(tx.meta, 'pre_token_balances'):
                return None

            owners_tokens = {}
            for balance in tx.meta.pre_token_balances:
                if not hasattr(balance, 'owner') or not hasattr(balance, 'mint'):
                    continue
                if not hasattr(balance, 'ui_token_amount') or not hasattr(balance.ui_token_amount, 'amount'):
                    continue
                try:
                    amount = int(balance.ui_token_amount.amount)
                except (ValueError, TypeError):
                    continue
                if amount < 1000000:
                    continue
                if balance.owner not in owners_tokens:
                    owners_tokens[balance.owner] = set()
                owners_tokens[balance.owner].add(balance.mint)

            for owner, mints in owners_tokens.items():
                if len(mints) == 2:
                    return owner
        except Exception:
            pass
        return None

    def bytes_to_address(self, byte_data: bytes) -> str:
        if len(byte_data) != 32:
            raise ValueError(f"Address must be 32 bytes, got {len(byte_data)}")
        return base58.b58encode(byte_data).decode('utf-8')

    def get_account_at_index(self, transaction_data, index: int) -> Optional[str]:
        """Get account address at given index, handling base accounts and loaded addresses"""
        account_keys = transaction_data.transaction.message.account_keys
        base_count = len(account_keys)

        if index < base_count:
            account_bytes = account_keys[index]
            if isinstance(account_bytes, str):
                account_bytes = account_bytes.encode('latin1')
            return self.bytes_to_address(account_bytes)

        loaded_index = index - base_count
        loaded_writable = transaction_data.meta.loaded_writable_addresses
        if loaded_index < len(loaded_writable):
            addr_bytes = loaded_writable[loaded_index]
            if isinstance(addr_bytes, str):
                addr_bytes = addr_bytes.encode('latin1')
            return self.bytes_to_address(addr_bytes)

        loaded_readonly = transaction_data.meta.loaded_readonly_addresses
        readonly_index = loaded_index - len(loaded_writable)
        if readonly_index < len(loaded_readonly):
            addr_bytes = loaded_readonly[readonly_index]
            if isinstance(addr_bytes, str):
                addr_bytes = addr_bytes.encode('latin1')
            return self.bytes_to_address(addr_bytes)

        return None

    def decode_instruction_accounts(self, accounts_data: bytes) -> List[int]:
        if isinstance(accounts_data, str):
            accounts_data = accounts_data.encode('latin1')
        return [b for b in accounts_data]

    def get_program_id(self, transaction_data, program_id_index: int) -> Optional[str]:
        return self.get_account_at_index(transaction_data, program_id_index)

    def identify_dex(self, program_id: str) -> Optional[str]:
        return DEX_PROGRAMS.get(program_id)

    def extract_vault_addresses(self, transaction_data, instruction, dex_name: str) -> Dict[str, Any]:
        """Extract vault addresses and balance changes from swap instruction accounts"""
        accounts_data = instruction.accounts
        account_indices = self.decode_instruction_accounts(accounts_data)

        vault_info = {
            'vaults': [],
            'vault_balance_changes': {},
            'instruction_accounts': account_indices,
        }

        if not account_indices:
            return vault_info

        instruction_account_addresses = []
        for idx in account_indices:
            addr = self.get_account_at_index(transaction_data, idx)
            if addr:
                instruction_account_addresses.append((idx, addr))

        pre_token_balances = transaction_data.meta.pre_token_balances
        post_token_balances = transaction_data.meta.post_token_balances

        balance_changes = {}
        addr_set = {addr for _, addr in instruction_account_addresses}

        for balance in pre_token_balances:
            account_addr = self.get_account_at_index(transaction_data, balance.account_index)
            if account_addr in addr_set:
                amount_str = balance.ui_token_amount.amount if balance.ui_token_amount.amount else "0"
                balance_changes[account_addr] = {
                    'pre_amount': int(amount_str),
                    'mint': balance.mint,
                    'decimals': balance.ui_token_amount.decimals,
                }

        for balance in post_token_balances:
            account_addr = self.get_account_at_index(transaction_data, balance.account_index)
            if account_addr in addr_set:
                amount_str = balance.ui_token_amount.amount if balance.ui_token_amount.amount else "0"
                if account_addr in balance_changes:
                    balance_changes[account_addr]['post_amount'] = int(amount_str)
                else:
                    balance_changes[account_addr] = {
                        'pre_amount': 0,
                        'post_amount': int(amount_str),
                        'mint': balance.mint,
                        'decimals': balance.ui_token_amount.decimals,
                    }

        vaults = []
        for account_addr, balances in balance_changes.items():
            pre_amount = balances.get('pre_amount', 0)
            post_amount = balances.get('post_amount', 0)
            change = post_amount - pre_amount
            if change != 0:
                vault_info_entry = {
                    'vault_address': account_addr,
                    'mint': balances.get('mint'),
                    'pre_balance': pre_amount,
                    'post_balance': post_amount,
                    'balance_change': change,
                    'decimals': balances.get('decimals', 0),
                    'is_vault': pre_amount > 100000 or post_amount > 100000,
                }
                vaults.append(vault_info_entry)
                vault_info['vault_balance_changes'][account_addr] = vault_info_entry

        vaults.sort(key=lambda x: max(x['pre_balance'], x['post_balance']), reverse=True)
        vault_info['vaults'] = vaults
        return vault_info

    def extract_pool_from_instruction(self, transaction_data, instruction, dex_name: str) -> Optional[str]:
        """Extract pool address from swap instruction based on DEX type"""
        accounts_data = instruction.accounts
        account_indices = self.decode_instruction_accounts(accounts_data)
        if not account_indices:
            return None

        if dex_name == RAYDIUM_AMM_V4:
            for pool_pos in [1, 2, 3, 4, 5]:
                if pool_pos < len(account_indices):
                    pool_address = self.get_account_at_index(transaction_data, account_indices[pool_pos])
                    if pool_address and self.is_likely_pool_account(pool_address, dex_name):
                        return pool_address

        elif dex_name == WHIRLPOOL:
            for pool_pos in [2, 4, 0, 1]:
                if pool_pos < len(account_indices):
                    pool_address = self.get_account_at_index(transaction_data, account_indices[pool_pos])
                    if pool_address and self.is_likely_pool_account(pool_address, dex_name):
                        return pool_address

        elif dex_name == METEORA_DLMM:
            for pool_pos in [0, 1, 2, 3]:
                if pool_pos < len(account_indices):
                    pool_address = self.get_account_at_index(transaction_data, account_indices[pool_pos])
                    if pool_address and self.is_likely_pool_account(pool_address, dex_name):
                        return pool_address

        elif dex_name == PUMPSWAP:
            bonding_curve = self.extract_bonding_curve(transaction_data)
            if bonding_curve:
                return bonding_curve
            for pool_pos in [6]:
                if pool_pos < len(account_indices):
                    pool_address = self.get_account_at_index(transaction_data, account_indices[pool_pos])
                    if pool_address and self.is_likely_pool_account(pool_address, dex_name):
                        return pool_address

        elif dex_name == RAYDIUM_CLMM:
            pool_address = self.get_raydium_clmm_pool(transaction_data)
            if pool_address:
                return pool_address

        elif dex_name == RAYDIUM_CPMM:
            pool_address = self.get_raydium_clmm_pool(transaction_data)
            if pool_address:
                return pool_address

        return None

    def is_likely_pool_account(self, address: str, dex_name: str) -> bool:
        if address in SYSTEM_ACCOUNTS:
            return False
        return True

    def is_swap_instruction(self, instruction_data: bytes, dex_name: str) -> bool:
        """Determine if instruction data represents a swap"""
        if len(instruction_data) == 0:
            return False

        for discriminator in SWAP_DISCRIMINATORS:
            if instruction_data.startswith(discriminator):
                return True

        if dex_name == RAYDIUM_AMM_V4:
            return len(instruction_data) >= 16
        elif dex_name == WHIRLPOOL:
            return len(instruction_data) >= 8
        elif METEORA_DLMM in dex_name:
            if len(instruction_data) >= 16:
                first_byte = instruction_data[0]
                return first_byte in [0x09, 0x84, 0xa8, 0x2e, 0x4c, 0x5d, 0x6e] or len(instruction_data) in [24, 32, 40, 48]
            return False

        return len(instruction_data) > 0

    def get_swap_type(self, instruction_data: bytes, dex_name: str) -> str:
        for discriminator, swap_type in SWAP_DISCRIMINATORS.items():
            if instruction_data.startswith(discriminator):
                if "Meteora" in dex_name:
                    return f"{METEORA_DLMM} Swap"
                return swap_type
        if "Meteora" in dex_name:
            return f"{METEORA_DLMM} Swap"
        return f"{dex_name} Swap"

    def analyze_instruction(self, transaction_data, instruction) -> Optional[Dict]:
        """Analyze a single instruction to detect swaps"""
        program_id_index = instruction.program_id_index
        if program_id_index is None:
            return None

        program_id = self.get_program_id(transaction_data, program_id_index)
        if not program_id:
            return None

        dex_name = self.identify_dex(program_id)
        if not dex_name:
            return None

        if dex_name not in SUPPORTED_DEXS:
            return None

        instruction_data = instruction.data
        if isinstance(instruction_data, str):
            instruction_data = instruction_data.encode('latin1')

        if not self.is_swap_instruction(instruction_data, dex_name):
            return None

        pool_address = self.extract_pool_from_instruction(transaction_data, instruction, dex_name)
        vault_info = self.extract_vault_addresses(transaction_data, instruction, dex_name)
        swap_type = self.get_swap_type(instruction_data, dex_name)

        return {
            'dex': dex_name,
            'program_id': program_id,
            'pool_address': pool_address,
            'vault_balance_changes': vault_info['vault_balance_changes'],
            'instruction_data_length': len(instruction_data),
            'swap_type': swap_type,
            'raw_instruction_data': instruction_data[:16].hex() if len(instruction_data) >= 16 else instruction_data.hex(),
        }

    def analyze_transaction(self, transaction_data) -> List[Dict]:
        """Analyze entire transaction to find all swaps"""
        swaps = []

        message = transaction_data.transaction.message
        for instruction in message.instructions:
            swap_info = self.analyze_instruction(transaction_data, instruction)
            if swap_info:
                swaps.append(swap_info)

        meta = transaction_data.meta
        for inner_group in meta.inner_instructions:
            for inner_instruction in inner_group.instructions:
                swap_info = self.analyze_instruction(transaction_data, inner_instruction)
                if swap_info:
                    swaps.append(swap_info)

        return swaps
