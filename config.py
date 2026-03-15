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
    entry_zscore: float = float(os.getenv('ENTRY_ZSCORE', '2.5'))
    exit_zscore: float = float(os.getenv('EXIT_ZSCORE', '0.5'))
    stop_loss_zscore: float = float(os.getenv('STOP_LOSS_ZSCORE', '4.0'))
    max_entry_zscore: float = float(os.getenv('MAX_ENTRY_ZSCORE', '3.0'))
    allowed_direction: str = os.getenv('ALLOWED_DIRECTION', 'both')  # both, long, short

    # Position sizing
    sizing_method: str = os.getenv('SIZING_METHOD', 'fixed_fraction')
    fixed_fraction: float = float(os.getenv('FIXED_FRACTION', '0.05'))
    max_position_usd: float = float(os.getenv('MAX_POSITION_USD', '1000'))
    max_exposure_ratio: float = float(os.getenv('MAX_EXPOSURE_RATIO', '1.0'))
    max_positions: int = int(os.getenv('MAX_POSITIONS', '10'))
    max_positions_per_hour: int = int(os.getenv('MAX_POSITIONS_PER_HOUR', '5'))

    # Risk
    max_drawdown_pct: float = float(os.getenv('MAX_DRAWDOWN_PCT', '0.10'))
    max_position_loss_pct: float = float(os.getenv('MAX_POSITION_LOSS_PCT', '0.15'))
    max_position_age_half_lives: float = float(os.getenv('MAX_POSITION_AGE_HALF_LIVES', '5.0'))
    pair_staleness_hours: int = int(os.getenv('PAIR_STALENESS_HOURS', '24'))
    min_half_life: float = float(os.getenv('MIN_HALF_LIFE', '200'))
    max_half_life_ratio: float = float(os.getenv('MAX_HALF_LIFE_RATIO', '0.5'))
    max_half_life_secs: float = float(os.getenv('MAX_HALF_LIFE_SECS', '1800'))  # 30min default
    max_positions_per_token: int = int(os.getenv('MAX_POSITIONS_PER_TOKEN', '3'))

    # Execution
    slippage_bps: int = int(os.getenv('SLIPPAGE_BPS', '50'))
    priority_fee_lamports: int = int(os.getenv('PRIORITY_FEE', '10000'))
    use_lunar_lander: bool = os.getenv('USE_LUNAR_LANDER', 'false').lower() == 'true'
    lunar_lander_tip_lamports: int = int(os.getenv('LUNAR_LANDER_TIP', '1000000'))  # 0.001 SOL minimum
    lunar_lander_endpoint: str = os.getenv('LUNAR_LANDER_ENDPOINT', 'http://fra.lunar-lander.hellomoon.io')
    lunar_lander_api_key: str = os.getenv('LUNAR_LANDER_API_KEY', '')

    # Price feed
    price_poll_interval: float = float(os.getenv('PRICE_POLL_INTERVAL', '6'))

    # Mode
    paper_trade: bool = os.getenv('PAPER_TRADE', 'true').lower() == 'true'
    lookback_window: int = int(os.getenv('LOOKBACK_WINDOW', '100'))
    signal_resample_secs: float = float(os.getenv('SIGNAL_RESAMPLE_SECS', '300'))  # 5-min candles
    entry_cooldown_slots: int = int(os.getenv('ENTRY_COOLDOWN_SLOTS', '750'))
    initial_capital: float = float(os.getenv('INITIAL_CAPITAL', '1000'))

    # Inline cointegration discovery
    coint_scan_interval: float = float(os.getenv('COINT_SCAN_INTERVAL', '300'))
    coint_resample_secs: float = float(os.getenv('COINT_RESAMPLE_SECS', '30'))
    coint_min_observations: int = int(os.getenv('COINT_MIN_OBSERVATIONS', '50'))
    coint_p_threshold: float = float(os.getenv('COINT_P_THRESHOLD', '0.05'))
    coint_history_capacity: int = int(os.getenv('COINT_HISTORY_CAPACITY', '10000'))
    coint_warmup_minutes: float = float(os.getenv('COINT_WARMUP_MINUTES', '30'))
    use_scanner_db: bool = os.getenv('USE_SCANNER_DB', 'true').lower() == 'true'
