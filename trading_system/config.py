"""
60% Win Rate Momentum Detection System Configuration
Optimized for balanced selectivity and opportunity capture
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

# 60% Win Rate Universe Configuration
UNIVERSE_CONFIG = {
    'min_price': 5.0,               # $5 minimum price
    'max_price': 300.0,             # $300 maximum price
    'min_daily_volume': 100000,     # 100K minimum volume
    'volume_spike_threshold': 2.0,   # 2x volume spike minimum
    'price_move_threshold': 0.003,   # 0.3% minimum move (balanced)
    'instant_filter_enabled': True,
}

# 60% Win Rate Momentum Detection Configuration
MOMENTUM_CONFIG = {
    # Balanced thresholds for 60% win rate
    'entry_threshold': 65,          # Moderate selectivity (was 25)
    'instant_signal_threshold': 50, # Initial screening threshold
    'quality_threshold': 60,        # Balanced quality requirement
    'min_data_points': 10,          # Sufficient data for analysis
    'max_data_days': 3,             # 3-day maximum lookback
    
    # Enhanced feature extraction
    'price_velocity_windows': [5, 15, 30, 60],  # Multiple timeframes
    'volume_explosion_threshold': 2.0,          # 2x volume minimum
    'volatility_expansion_threshold': 1.3,      # Moderate volatility requirement
    'gap_threshold': 0.02,                      # 2% gap detection
    
    # Model parameters optimized for 60% win rate
    'confidence_threshold': 0.70,    # 70% confidence minimum
    'xgboost_enabled': True,         # XGBoost as primary decision maker
    
    # Balanced quality filters
    'min_volume_consistency': 0.4,  # Moderate volume requirement
    'max_price_choppiness': 0.7,    # Allow some volatility
    'min_trend_strength': 0.4,      # Moderate trend requirement
    'min_liquidity_score': 0.4,     # Reasonable liquidity
    
    # Risk management for entries
    'max_daily_high_position': 0.70,  # Allow entries up to 70% of daily range
    'min_consolidation_periods': 2,   # Reduced consolidation requirement
    'max_recent_volatility': 0.08,    # 8% volatility tolerance
    'avoid_momentum_deceleration': True,
    'max_gap_up_pct': 0.05,          # Allow 5% gap ups
    'min_pullback_from_high': 0.05,  # 5% pullback minimum
}

# 60% Win Rate Risk Management
RISK_PARAMS = {
    'max_positions': 8,            # 5-8 positions for balance
    'max_position_size_pct': 0.20, # 20% maximum position size
    'stop_loss_pct': 3.0,         # 3% stop loss for 2:1 risk/reward
    'take_profit_pct': 6.0,       # 6% take profit target
    'max_daily_loss_pct': 0.06,   # 6% daily loss limit
    'min_position_size': 1500,    # $1,500 minimum position
    'max_simultaneous_entries': 3, # Max 3 new entries per day
    'min_cash_reserve_pct': 0.10, # 10% cash reserve
    'order_cooldown_seconds': 300, # 5 minute cooldown
    'bracket_orders_enabled': True,
}

# Position Sizing for 60% Win Rate
POSITION_SIZING = {
    'high_conviction_pct': 0.20,   # 20% for scores 85+
    'good_setup_pct': 0.15,       # 15% for scores 75-84
    'decent_setup_pct': 0.12,     # 12% for scores 65-74
    'min_score_for_position': 65, # No positions below 65
    'quality_multipliers': {
        'A': 1.2,   # 20% size increase for Grade A
        'B': 1.0,   # Base size for Grade B
        'C': 0.8,   # 20% reduction for Grade C
        'D': 0.0,   # No positions for Grade D
        'F': 0.0    # No positions for Grade F
    }
}

# Execution Parameters (Optimized for Fill Rate)
EXECUTION_PARAMS = {
    'market_order_pct': 0.3,       # 30% market orders for momentum
    'limit_order_pct': 0.7,        # 70% limit orders for price protection
    'limit_spread_max': 0.008,      # 0.8% spread tolerance
    'limit_price_buffer': 0.003,   # 0.3% buffer for fills
    'order_timeout_seconds': 45,    # Longer timeout for limit orders
    'instant_execution_enabled': True,
    'max_spread_pct': 0.015,       # 1.5% max spread
    'price_improvement_required': False, # Allow market execution
}

# XGBoost Configuration for 60% Win Rate
XGBOOST_CONFIG = {
    'max_depth': 5,                 # Moderate complexity
    'learning_rate': 0.08,          # Balanced learning rate
    'n_estimators': 200,            # Sufficient trees
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,          # Prevent overfitting
    'gamma': 0.05,                  # Light regularization
    'reg_alpha': 0.05,              # L1 regularization
    'reg_lambda': 0.05,             # L2 regularization
    'scale_pos_weight': 1.5,        # Slight bias toward signals
    'random_state': 42,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'predictor': 'cpu_predictor'
}

# Real-time Processing Configuration
REALTIME_CONFIG = {
    'update_frequency_seconds': 2,   # Faster updates for momentum
    'max_concurrent_analysis': 150,  # Balanced concurrency
    'price_update_timeout': 3,       # Reasonable timeout
    'websocket_buffer_size': 2000,   # Large buffer for data
    'instant_callback_enabled': True,
    'historical_cache_size': 300,    # Cache for 300 symbols
}

# System Configuration
SYSTEM_CONFIG = {
    'timezone': 'US/Eastern',
    'market_open': '09:30',
    'market_close': '16:00',
    'max_memory_gb': 8,
    'enable_gpu': False,
    'event_driven_mode': True,
    'win_rate_target': 0.60,        # 60% win rate target
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'momentum_trading.log',
    'max_size_mb': 100,
    'backup_count': 10,
}

# Enhanced Feature Configuration for 60% Win Rate
FEATURE_CONFIG = {
    # Core momentum features (15)
    'price_velocity_features': True,
    'volume_explosion_features': True,
    'price_acceleration_features': True,
    'breakout_strength_features': True,
    'momentum_persistence_features': True,
    
    # Risk management features (10)
    'daily_position_features': True,
    'pullback_analysis_features': True,
    'support_resistance_features': True,
    'volatility_features': True,
    'time_of_day_features': True,
    
    # Market context features (15)
    'sector_strength_features': True,
    'market_sentiment_features': True,
    'institutional_activity_features': True,
    'bid_ask_analysis_features': True,
    'news_catalyst_features': True,
    
    # Technical analysis features
    'moving_average_features': True,
    'bollinger_band_features': True,
    'rsi_features': True,
    'fibonacci_features': True,
    'pattern_recognition_features': True,
}

# Performance Monitoring for 60% Win Rate
PERFORMANCE_CONFIG = {
    'track_win_rate': True,          # Primary metric
    'win_rate_window': 20,           # Rolling 20-trade window
    'target_win_rate': 0.60,        # 60% target
    'win_rate_tolerance': 0.05,     # ±5% tolerance (55-65%)
    'adjust_thresholds_enabled': True, # Dynamic threshold adjustment
    'track_signal_latency': True,
    'track_execution_latency': True,
    'performance_log_interval': 600, # Log every 10 minutes
}

# Dynamic Threshold Adjustment
THRESHOLD_CONFIG = {
    'initial_threshold': 65,         # Starting threshold
    'min_threshold': 60,             # Minimum threshold
    'max_threshold': 75,             # Maximum threshold
    'adjustment_step': 2,            # Step size for adjustments
    'adjustment_frequency': 20,      # Adjust every 20 trades
    'win_rate_upper_bound': 0.70,    # Reduce threshold if above
    'win_rate_lower_bound': 0.50,    # Increase threshold if below
}

# Historical Training Configuration
TRAINING_CONFIG = {
    'lookback_days': 180,            # 6 months of training data
    'validation_split': 0.2,        # 20% for validation
    'test_split': 0.1,              # 10% for testing
    'min_training_samples': 500,    # Minimum samples needed
    'retrain_frequency_days': 7,    # Retrain weekly
    'explosive_move_threshold': 0.15, # 15% move for positive labels
    'momentum_duration_hours': 2,    # Minimum momentum duration
}