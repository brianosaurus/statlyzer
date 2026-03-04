"""
Configuration for statalayer statistical arbitrage bot.
"""

import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

try:
    load_dotenv()
except Exception:
    pass


@dataclass
class Config:
    # gRPC
    rpc_url: str = os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')
    grpc_endpoint: str = os.getenv('GRPC_ENDPOINT', 'api.mainnet-beta.solana.com:443')
    grpc_token: str = os.getenv('GRPC_TOKEN', '')

    # Wallet (for live trading)
    wallet_keypair_path: str = os.getenv('WALLET_KEYPAIR_PATH', '')

    # Scanner DB (read cointegrated pairs from here)
    scanner_db_path: str = os.getenv('SCANNER_DB_PATH', '../arbitrage_tracker/arb_tracker.db')

    # Signal thresholds
    entry_zscore: float = float(os.getenv('ENTRY_ZSCORE', '2.0'))
    exit_zscore: float = float(os.getenv('EXIT_ZSCORE', '0.5'))
    stop_loss_zscore: float = float(os.getenv('STOP_LOSS_ZSCORE', '4.0'))

    # Position sizing
    sizing_method: str = os.getenv('SIZING_METHOD', 'fixed_fraction')
    fixed_fraction: float = float(os.getenv('FIXED_FRACTION', '0.02'))
    max_position_usd: float = float(os.getenv('MAX_POSITION_USD', '1000'))
    max_total_exposure_usd: float = float(os.getenv('MAX_TOTAL_EXPOSURE_USD', '5000'))
    max_positions: int = int(os.getenv('MAX_POSITIONS', '10'))
    max_positions_per_hour: int = int(os.getenv('MAX_POSITIONS_PER_HOUR', '5'))

    # Risk
    max_drawdown_pct: float = float(os.getenv('MAX_DRAWDOWN_PCT', '0.10'))
    pair_staleness_hours: int = int(os.getenv('PAIR_STALENESS_HOURS', '24'))
    min_half_life: float = float(os.getenv('MIN_HALF_LIFE', '2.0'))
    max_half_life_ratio: float = float(os.getenv('MAX_HALF_LIFE_RATIO', '0.5'))

    # Execution
    slippage_bps: int = int(os.getenv('SLIPPAGE_BPS', '50'))
    priority_fee_lamports: int = int(os.getenv('PRIORITY_FEE', '10000'))
    use_jito: bool = os.getenv('USE_JITO', 'false').lower() == 'true'
    jito_tip_lamports: int = int(os.getenv('JITO_TIP', '10000'))

    # Mode
    paper_trade: bool = os.getenv('PAPER_TRADE', 'true').lower() == 'true'
    lookback_window: int = int(os.getenv('LOOKBACK_WINDOW', '60'))
    initial_capital: float = float(os.getenv('INITIAL_CAPITAL', '10000'))
