#!/usr/bin/env python3
"""
GPU-Accelerated Gap Trading System Configuration
This module centralizes all configuration parameters for the trading system.
All time-sensitive parameters, risk limits, and GPU settings are defined here.
"""

import os
import torch
from datetime import time
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any
import pytz

# Load environment variables from .env file
env_path = Path('/home/ubuntu/moe-1/.env')
load_dotenv(dotenv_path=env_path)

class GPUTradingConfig:
    """Centralized configuration for GPU-accelerated gap trading system."""
    
    def __init__(self):
        # Verify GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This system requires GPU.")
        
        self.device = torch.device('cuda:0')
        self.gpu_properties = torch.cuda.get_device_properties(0)
        
        # API Configuration
        self.ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
        self.ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
        self.ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        self.POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
        
        if not all([self.ALPACA_API_KEY, self.ALPACA_SECRET_KEY, self.POLYGON_API_KEY]):
            raise ValueError("Missing required API keys in .env file")
        
        # Trading Universe Parameters
        self.UNIVERSE_SIZE_MIN = 150
        self.UNIVERSE_SIZE_MAX = 200
        self.MIN_STOCK_PRICE = 10.0
        self.MAX_STOCK_PRICE = 250.0
        self.MIN_AVG_VOLUME = 1_000_000  # shares
        self.MIN_DOLLAR_VOLUME = 10_000_000  # daily dollar volume
        self.MIN_ATR_PERCENT = 2.0  # minimum ATR as % of price
        self.UNIVERSE_UPDATE_FREQUENCY_DAYS = 30
        self.RUSSELL_1000_FILE = Path('/home/ubuntu/moe-1/russell_1000.json')
        
        # Gap Detection Parameters
        self.MIN_GAP_PERCENT = 2.0  # minimum gap size to consider
        self.MAX_GAP_PERCENT = 15.0  # maximum gap size (avoid anomalies)
        self.PRE_MARKET_VOLUME_MULTIPLIER = 0.1  # pre-market vol vs avg daily
        self.INSTITUTIONAL_BLOCK_SIZE = 5000  # shares to identify institutional
        
        # Trading Time Windows (all times in ET)
        self.TIMEZONE = pytz.timezone('US/Eastern')
        self.PRE_MARKET_START = time(4, 0)  # 4:00 AM ET
        self.FINAL_PREP_START = time(9, 0)  # 9:00 AM ET
        self.PRE_MARKET_ENTRY_START = time(9, 20)  # 9:20 AM ET
        self.MARKET_OPEN = time(9, 30)  # 9:30 AM ET
        self.PRIMARY_ENTRY_END = time(10, 0)  # 10:00 AM ET
        self.POSITION_MANAGEMENT_END = time(11, 0)  # 11:00 AM ET
        self.FORCE_EXIT_TIME = time(11, 30)  # 11:30 AM ET
        
        # Risk Management Parameters
        self.MAX_PORTFOLIO_HEAT = 0.15  # max 15% of portfolio at risk
        self.MAX_SINGLE_POSITION_RISK = 0.02  # max 2% portfolio risk per trade
        self.MAX_CORRELATED_POSITIONS = 3  # max positions in same sector
        self.CORRELATION_THRESHOLD = 0.7  # correlation coefficient threshold
        
        # Position Sizing Parameters
        self.TIER1_ALLOCATION = 0.4  # 40% for top 2-3 opportunities
        self.TIER2_ALLOCATION = 0.4  # 40% for next 4-5 opportunities
        self.RESERVE_ALLOCATION = 0.2  # 20% cash reserve
        self.MIN_POSITION_SIZE = 1000  # minimum $1000 position
        self.MAX_POSITION_SIZE_PERCENT = 0.15  # max 15% in single position
        
        # Stop Loss Configuration
        self.INITIAL_TIME_STOP_MINUTES = 20  # exit if no continuation
        self.ATR_STOP_MULTIPLIER = 1.5  # trailing stop at 1.5x ATR
        self.MAX_ADVERSE_MOVE_PERCENT = 0.5  # max 50% of gap size
        self.PROFIT_TARGET_ATR_MULTIPLIER = 3.0  # take profits at 3x ATR
        
        # Model Parameters
        self.BAYESIAN_PRIOR_STRENGTH = 0.1  # prior weight in Bayesian model
        self.ENSEMBLE_MODEL_COUNT = 100  # parallel models in ensemble
        self.CONFIDENCE_THRESHOLD = 0.65  # minimum confidence to take trade
        self.ONLINE_LEARNING_RATE = 0.001  # adaptation rate for models
        self.LOOKBACK_DAYS = 90  # historical data for model training
        
        # GPU Configuration
        self.GPU_BATCH_SIZE = 256  # stocks to process in parallel
        self.TENSOR_DTYPE = torch.float32
        self.USE_MIXED_PRECISION = True  # use FP16 where appropriate
        self.GPU_MEMORY_FRACTION = 0.8  # reserve 20% GPU memory
        self.CUDA_STREAMS = 4  # parallel CUDA streams
        
        # Polygon.io Data Pipeline Configuration
        self.POLYGON_OPTIONS_ENABLED = True  # Enable options data integration
        self.POLYGON_TICK_DATA_ENABLED = True  # Enable tick-by-tick data
        self.OPTIONS_CHAIN_DEPTH = 10  # Strike levels above/below current price
        self.UNUSUAL_OPTIONS_THRESHOLD = 100000  # $100k premium threshold for unusual activity
        self.TICK_BUFFER_SIZE = 1000  # Number of ticks to buffer per symbol
        self.INSTITUTIONAL_BLOCK_SIZE = 10000  # Minimum size for institutional trades
        self.OPTIONS_FLOW_WINDOW_MINUTES = 5  # Window for options flow analysis
        
        # Feature Engineering Parameters
        self.FEATURE_WINDOW_SIZES = [5, 15, 30, 60]  # minutes for rolling features
        self.VOLUME_PROFILE_BINS = 10  # bins for volume distribution
        self.MICROSTRUCTURE_TICK_WINDOW = 100  # ticks for spread analysis
        self.SECTOR_CORRELATION_WINDOW = 20  # days for sector correlation
        
        # Execution Parameters
        self.USE_MARKETABLE_LIMITS = True  # aggressive fills in first 5 min
        self.MARKETABLE_LIMIT_OFFSET = 0.02  # 2 cents through market
        self.LIMIT_ORDER_SPREAD_FRACTION = 0.5  # place at mid-spread
        self.MAX_SLIPPAGE_PERCENT = 0.1  # cancel if slippage exceeds
        self.ORDER_TIMEOUT_SECONDS = 5  # cancel unfilled orders
        
        # Database Configuration
        self.DB_HOST = os.getenv('DB_HOST', 'localhost')
        self.DB_PORT = int(os.getenv('DB_PORT', 5432))
        self.DB_NAME = os.getenv('DB_NAME', 'gap_trading')
        self.DB_USER = os.getenv('DB_USER', 'gap_trader')
        self.DB_PASSWORD = os.getenv('DB_PASSWORD')
        self.DB_SSL_MODE = os.getenv('DB_SSL_MODE', 'prefer')
        
        if not self.DB_PASSWORD:
            raise ValueError("Database password not found in .env file")
        
        # Data Storage Configuration (for files not in database)
        self.DATA_ROOT = Path('/home/ubuntu/gap_trading_data')
        self.FEATURES_DIR = self.DATA_ROOT / 'features'
        self.MODELS_DIR = self.DATA_ROOT / 'models'
        self.CACHE_DIR = self.DATA_ROOT / 'cache'
        
        # Create directories if they don't exist
        for directory in [self.DATA_ROOT, self.FEATURES_DIR, self.MODELS_DIR, self.CACHE_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Market Regime Parameters
        self.REGIME_STATES = ['TRENDING', 'VOLATILE', 'QUIET']
        self.REGIME_LOOKBACK_DAYS = 20
        self.VOLATILITY_PERCENTILE_THRESHOLD = 75  # for volatile regime
        self.TREND_STRENGTH_THRESHOLD = 0.6  # R-squared for trending
        
        # Performance Tracking
        self.MIN_TRADES_FOR_STATS = 20  # before calculating metrics
        self.SHARPE_CALCULATION_DAYS = 60  # rolling Sharpe ratio
        self.MAX_DRAWDOWN_LIMIT = 0.1  # stop trading if DD exceeds 10%
        
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate all configuration parameters and GPU setup."""
        validation_results = {
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': self.gpu_properties.name,
            'gpu_memory_gb': self.gpu_properties.total_memory / 1e9,
            'cuda_version': torch.version.cuda,
            'api_keys_present': bool(self.ALPACA_API_KEY and self.POLYGON_API_KEY),
            'directories_created': all(d.exists() for d in [
                self.DATA_ROOT, self.FEATURES_DIR, self.MODELS_DIR
            ]),
            'time_windows_valid': self._validate_time_windows(),
            'risk_parameters_valid': self._validate_risk_parameters(),
            'gpu_memory_sufficient': self.gpu_properties.total_memory > 8e9  # 8GB minimum
        }
        
        validation_results['all_valid'] = all(validation_results.values())
        return validation_results
    
    def _validate_time_windows(self) -> bool:
        """Ensure trading time windows are in correct order."""
        times = [
            self.PRE_MARKET_START,
            self.FINAL_PREP_START,
            self.PRE_MARKET_ENTRY_START,
            self.MARKET_OPEN,
            self.PRIMARY_ENTRY_END,
            self.POSITION_MANAGEMENT_END,
            self.FORCE_EXIT_TIME
        ]
        return all(times[i] < times[i+1] for i in range(len(times)-1))
    
    def _validate_risk_parameters(self) -> bool:
        """Ensure risk parameters are internally consistent."""
        checks = [
            self.MAX_SINGLE_POSITION_RISK <= self.MAX_PORTFOLIO_HEAT,
            self.TIER1_ALLOCATION + self.TIER2_ALLOCATION + self.RESERVE_ALLOCATION == 1.0,
            self.MIN_POSITION_SIZE < self.MAX_POSITION_SIZE_PERCENT * 50000,  # assuming $50k account
            self.CONFIDENCE_THRESHOLD > 0.5 and self.CONFIDENCE_THRESHOLD < 1.0
        ]
        return all(checks)
    
    def get_tensor_device_config(self) -> Dict[str, Any]:
        """Get PyTorch tensor and device configuration."""
        return {
            'device': str(self.device),
            'dtype': self.TENSOR_DTYPE,
            'mixed_precision': self.USE_MIXED_PRECISION,
            'batch_size': self.GPU_BATCH_SIZE,
            'memory_fraction': self.GPU_MEMORY_FRACTION,
            'cuda_streams': self.CUDA_STREAMS
        }
    
    def get_current_trading_phase(self, current_time: time) -> str:
        """Determine current trading phase based on time."""
        if current_time < self.PRE_MARKET_START:
            return "CLOSED"
        elif current_time < self.FINAL_PREP_START:
            return "PRE_MARKET_SCAN"
        elif current_time < self.PRE_MARKET_ENTRY_START:
            return "FINAL_PREPARATION"
        elif current_time < self.MARKET_OPEN:
            return "PRE_MARKET_ENTRY"
        elif current_time < self.PRIMARY_ENTRY_END:
            return "PRIMARY_ENTRY"
        elif current_time < self.POSITION_MANAGEMENT_END:
            return "POSITION_MANAGEMENT"
        elif current_time < self.FORCE_EXIT_TIME:
            return "EXIT_WINDOW"
        else:
            return "POST_MARKET"
    
    def adjust_for_account_size(self, account_value: float) -> None:
        """Dynamically adjust position sizing based on account value."""
        # Scale position sizes with account value
        base_account = 50000  # baseline account size
        scale_factor = account_value / base_account
        
        self.MIN_POSITION_SIZE = max(1000, 1000 * scale_factor)
        self.MAX_POSITION_SIZE_PERCENT = min(0.15, 0.15 * (base_account / account_value) ** 0.5)
        
        # Adjust tier allocations for larger accounts (more diversification)
        if account_value > 100000:
            self.TIER1_ALLOCATION = 0.35
            self.TIER2_ALLOCATION = 0.45
            self.RESERVE_ALLOCATION = 0.20


# Example usage when file is run directly
if __name__ == "__main__":
    print("GPU-Accelerated Gap Trading Configuration")
    print("=" * 50)
    
    # Initialize configuration
    config = GPUTradingConfig()
    
    # Validate configuration
    validation = config.validate_configuration()
    print("\nConfiguration Validation:")
    for key, value in validation.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value}")
    
    # Display GPU information
    print(f"\nGPU Configuration:")
    print(f"  Device: {config.gpu_properties.name}")
    print(f"  Memory: {config.gpu_properties.total_memory / 1e9:.1f} GB")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  PyTorch Version: {torch.__version__}")
    
    # Display trading windows
    print(f"\nTrading Schedule (ET):")
    print(f"  Pre-market Scan: {config.PRE_MARKET_START}")
    print(f"  Final Prep: {config.FINAL_PREP_START}")
    print(f"  Pre-market Entry: {config.PRE_MARKET_ENTRY_START}")
    print(f"  Market Open: {config.MARKET_OPEN}")
    print(f"  Primary Entry End: {config.PRIMARY_ENTRY_END}")
    print(f"  Management End: {config.POSITION_MANAGEMENT_END}")
    print(f"  Force Exit: {config.FORCE_EXIT_TIME}")
    
    # Display risk parameters
    print(f"\nRisk Management:")
    print(f"  Max Portfolio Heat: {config.MAX_PORTFOLIO_HEAT:.1%}")
    print(f"  Max Position Risk: {config.MAX_SINGLE_POSITION_RISK:.1%}")
    print(f"  Tier 1 Allocation: {config.TIER1_ALLOCATION:.1%}")
    print(f"  Tier 2 Allocation: {config.TIER2_ALLOCATION:.1%}")
    print(f"  Cash Reserve: {config.RESERVE_ALLOCATION:.1%}")
    
    # Test current trading phase
    from datetime import datetime
    current_time = datetime.now(config.TIMEZONE).time()
    current_phase = config.get_current_trading_phase(current_time)
    print(f"\nCurrent Trading Phase: {current_phase}")
    
    # Test account size adjustment
    print(f"\nPosition Sizing for Different Account Sizes:")
    for account_size in [25000, 50000, 100000, 250000]:
        config.adjust_for_account_size(account_size)
        print(f"  ${account_size:,}: Min=${config.MIN_POSITION_SIZE:,.0f}, "
              f"Max={config.MAX_POSITION_SIZE_PERCENT:.1%}")
