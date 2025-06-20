"""
System Configuration
Production settings for GPU momentum trading
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from dotenv import load_dotenv

# API Configuration
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

# Account Settings
INITIAL_CAPITAL = 25000
MIN_CASH_RESERVE = 100  # Always keep $100 in cash

# Position Sizing
POSITION_CONFIG = {
    'base_allocation_pct': 0.15,  # 15% base position (up from 10%)
    'max_position_pct': 0.25,     # 25% max per position (up from 20%)
    'min_position_size': 1500,    # $1,500 minimum (up from $500)
    # max_positions removed - rely on available cash only
}

# Momentum Thresholds
MOMENTUM_CONFIG = {
    'min_score': 60,              # Minimum momentum score - MEDIUM QUALITY
    'strong_momentum': 70,        # Strong momentum threshold
    'extreme_momentum': 80,       # Extreme momentum threshold
}

# Risk Management
RISK_CONFIG = {
    'trailing_stop_pct': 5.0,     # 5% trailing stop only
    'max_spread_pct': 0.01,       # 1% maximum bid-ask spread
    'ioc_orders': True,           # Use Immediate-or-Cancel orders
}

# Market Filters
MARKET_FILTERS = {
    'min_price': 5.0,             # Minimum stock price $5-$500 range
    'max_price': 500.0,           # Maximum stock price $5-$500 range
    'min_volume': 500000,         # Minimum daily volume (up from 100K)
    'min_market_cap': 500000000,  # $500M minimum market cap (up from $100M)
}

# GPU Settings
GPU_CONFIG = {
    'max_symbols': 15000,         # Maximum symbols to track
    'batch_size': 1000,           # GPU batch processing size
    'update_frequency': 0.1,      # Update frequency in seconds
}

# Performance Tracking
PERFORMANCE_CONFIG = {
    'log_interval': 60,           # Log performance every 60 seconds
    'snapshot_interval': 30,      # Market snapshot every 30 seconds
    'position_update': 10,        # Update positions every 10 seconds
}