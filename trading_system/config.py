"""
Pure Momentum Trading System Configuration
Optimized for CRCL-type explosive moves with 10% trailing stops (Alpaca minimum)
Focus: Numerical indicators only, no news dependency
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')
ALPACA_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET = os.getenv('ALPACA_SECRET_KEY', '')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

# Account Configuration
ACCOUNT_SIZE = 25000  # $25,000 starting capital

# Universe Discovery (Pure Numerical) - Small to Mid-Cap Focus
UNIVERSE_CONFIG = {
    'max_universe_size': 2000,
    'min_daily_volume': 500000,    # 500K minimum volume
    'min_price': 5.0,              # $5 minimum price (small to mid-cap)
    'max_price': 300.0,            # $300 maximum price (expanded range)
    'volume_spike_threshold': 3.0,  # 3x volume spike detection
    'price_move_threshold': 0.05,   # 5% price move detection
    'refresh_interval_seconds': 60  # Update universe every minute
}

# Momentum Detection Parameters
MOMENTUM_CONFIG = {
    # Core thresholds - RAISED FOR BETTER QUALITY
    'entry_threshold': 40,          # Enter when momentum score >= 40 (more selective)
    'quality_threshold': 50,        # Minimum for high-quality trades
    'min_data_points': 1,           # Work with minimal data (IPO support)
    'max_data_days': 3,             # Maximum 3 days historical data
    
    # Feature extraction
    'price_velocity_windows': [1, 4, 24],  # 1h, 4h, 24h price velocity
    'volume_explosion_threshold': 2.5,      # 2.5x volume explosion (raised)
    'volatility_expansion_threshold': 1.8,  # 1.8x volatility expansion (raised)
    'gap_threshold': 0.03,                  # 3% gap detection
    
    # Model parameters
    'retrain_hours': 24,            # Retrain model daily
    'confidence_threshold': 0.7,    # Minimum model confidence (raised)
    
    # Quality filters
    'min_volume_consistency': 0.7,  # Volume should be consistent
    'max_price_choppiness': 0.3,    # Price action shouldn't be too choppy
    'min_trend_strength': 0.6,      # R-squared of price trend
    'min_liquidity_score': 0.8,     # Real-time liquidity check
}

# Risk Management
RISK_PARAMS = {
    'max_positions': 15,            # Maximum concurrent positions
    'max_position_size_pct': 0.15,  # 15% max position size
    'trailing_stop_pct': 0.10,     # 10% trailing stop from high (Alpaca minimum)
    'max_daily_loss_pct': 0.08,    # 8% daily loss limit
    'min_position_size': 1000,     # $1000 minimum position
}

# Execution Parameters
EXECUTION_PARAMS = {
    'market_order_pct': 0.8,       # 80% market orders for speed
    'limit_order_pct': 0.2,        # 20% limit orders for better fills
    'limit_spread': 0.001,          # 0.1% spread for limit orders
    'order_timeout_seconds': 30,    # Cancel unfilled orders after 30s
}

# XGBoost Model Configuration
XGBOOST_CONFIG = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

# Real-time Processing
REALTIME_CONFIG = {
    'update_frequency_seconds': 10,  # Check positions every 10 seconds
    'scan_frequency_seconds': 30,    # Scan universe every 30 seconds
    'max_concurrent_analysis': 100,  # Analyze 100 symbols in parallel
    'price_update_timeout': 5,       # 5 second timeout for price updates
}

# System Configuration
SYSTEM_CONFIG = {
    'timezone': 'US/Eastern',
    'market_open': '09:30',
    'market_close': '16:00',
    'max_memory_gb': 8,
    'enable_gpu': True,
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'momentum_trading.log',
    'max_size_mb': 50,
    'backup_count': 5,
}

# Momentum Feature Configuration
MOMENTUM_FEATURES = {
    # Price-based features
    'price_velocity_1h': True,      # 1-hour price change rate
    'price_velocity_4h': True,      # 4-hour price change rate  
    'price_velocity_24h': True,     # 24-hour price change rate
    'price_acceleration': True,     # Price acceleration (d²price/dt²)
    
    # Volume-based features
    'volume_explosion_1h': True,    # 1-hour volume vs average
    'volume_explosion_4h': True,    # 4-hour volume vs average
    'volume_price_correlation': True, # Volume-price correlation
    
    # Volatility features
    'volatility_expansion': True,   # Recent vs historical volatility
    'range_expansion': True,        # Price range expansion
    'gap_strength': True,           # Gap up/down strength
    
    # Technical features
    'breakout_strength': True,      # Breakout above resistance
    'momentum_consistency': True,   # Consistent upward moves
    'trend_strength': True,         # Linear trend strength (R²)
}