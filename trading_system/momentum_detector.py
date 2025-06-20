"""
Live Momentum Detection with XGBoost
Based on CRCL test example - adapted for 3-day max + live data
Detects CRCL-type explosive moves in real-time
"""

import asyncio
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from trading_system import config

logger = logging.getLogger(__name__)


class LiveMomentumDetector:
    """XGBoost momentum detector optimized for live data + 3-day max lookback"""
    
    # Historical explosive moves for bootstrap training
    HISTORICAL_EXPLOSIVE_MOVES = [
        {'symbol': 'CRCL', 'date': '2024-11-18', 'start_price': 3.5, 'peak_price': 11.0, 'days': 1},
        {'symbol': 'COIN', 'date': '2025-06-05', 'start_price': 244.20, 'peak_price': 295.29, 'days': 10},
        {'symbol': 'SMCI', 'date': '2024-01-15', 'start_price': 280, 'peak_price': 420, 'days': 5},
        {'symbol': 'NVDA', 'date': '2023-05-25', 'start_price': 305, 'peak_price': 419, 'days': 3},
        {'symbol': 'ARM', 'date': '2024-02-07', 'start_price': 95, 'peak_price': 164, 'days': 2},
        {'symbol': 'MARA', 'date': '2024-02-28', 'start_price': 18, 'peak_price': 34, 'days': 4},
        {'symbol': 'LUNR', 'date': '2024-08-15', 'start_price': 7.50, 'peak_price': 19.74, 'days': 14},
    ]
    
    def __init__(self, polygon_client):
        self.polygon = polygon_client
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.feature_importance = {}
        self.is_trained = False
        
        # Model persistence
        self.model_path = 'live_momentum_model.pkl'
        self.scaler_path = 'live_momentum_scaler.pkl'
        
        # Load existing model if available (training is optional)
        self._load_model()
        
        # If no model exists, we'll train during first detection
        if not self.is_trained:
            logger.info("🔄 No pre-trained model found - will train on first momentum detection")
    
    def _load_model(self):
        """Load pre-trained model if available"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.feature_names = model_data['feature_names']
                    self.feature_importance = model_data['feature_importance']
                
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                self.is_trained = True
                logger.info("Loaded pre-trained momentum model")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
    
    def _save_model(self):
        """Save trained model"""
        try:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
                
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    async def extract_live_features(self, symbol: str, current_time: datetime = None) -> Optional[Dict]:
        """Extract live momentum features with 3-day max lookback"""
        if current_time is None:
            current_time = datetime.now()
        
        try:
            # Get sufficient daily bars for testing (extend lookback for early dates)
            start_date = current_time - timedelta(days=15)  # Get more context
            bars = await self.polygon.list_aggs(
                symbol, 1, 'day', start_date, current_time + timedelta(days=1), limit=30
            )
            
            # Debug: Check for data issues
            if len(bars) > 0:
                logger.debug(f"Feature extraction for {symbol}: {len(bars)} bars, latest close: ${bars[-1].close:.2f}")
            
            if len(bars) < 1:  # Work with minimal data (even 1 bar for IPOs)
                return None
            
            prices = np.array([bar.close for bar in bars])
            volumes = np.array([bar.volume for bar in bars])
            highs = np.array([bar.high for bar in bars])
            lows = np.array([bar.low for bar in bars])
            
            features = {}
            
            # === LIVE MOMENTUM FEATURES ===
            
            # 1. PRICE VELOCITY (daily timeframe)
            if len(prices) >= 2:
                features['live_velocity_5min'] = (prices[-1] - prices[-2]) / prices[-2]  # 1-day velocity
                
                if len(prices) >= 3:  # 3 days
                    features['live_velocity_30min'] = (prices[-1] - prices[-3]) / prices[-3]
                else:
                    features['live_velocity_30min'] = features['live_velocity_5min']
                
                if len(prices) >= 5:  # 5 days (1 week)
                    features['live_velocity_1h'] = (prices[-1] - prices[-5]) / prices[-5]
                else:
                    features['live_velocity_1h'] = features['live_velocity_30min']
            else:
                features['live_velocity_5min'] = 0
                features['live_velocity_30min'] = 0
                features['live_velocity_1h'] = 0
            
            # 2. VOLUME EXPLOSION (daily timeframe)
            current_volume = volumes[-1]
            if len(volumes) > 5:  # Need 5+ days of data
                recent_avg_volume = np.mean(volumes[-6:-1])  # Last 5 days avg
                features['live_volume_spike'] = current_volume / recent_avg_volume if recent_avg_volume > 0 else 1
            else:
                avg_volume = np.mean(volumes[:-1]) if len(volumes) > 1 else volumes[-1]
                features['live_volume_spike'] = current_volume / avg_volume if avg_volume > 0 else 1
            
            # 3. LIVE PRICE ACCELERATION
            if len(prices) >= 3:
                recent_velocity = (prices[-1] - prices[-2]) / prices[-2]
                prev_velocity = (prices[-2] - prices[-3]) / prices[-3]
                features['live_acceleration'] = recent_velocity - prev_velocity
            else:
                features['live_acceleration'] = 0
            
            # 4. RANGE EXPANSION (daily timeframe)
            current_range = (highs[-1] - lows[-1]) / prices[-1]
            if len(prices) > 5:
                avg_range = np.mean((highs[-6:-1] - lows[-6:-1]) / prices[-6:-1])
                features['live_range_expansion'] = current_range / avg_range if avg_range > 0 else 1
            else:
                features['live_range_expansion'] = current_range
            
            # === 3-DAY MAX FEATURES ===
            
            # 5. 3-DAY HIGH BREAKOUT
            if len(prices) >= 3:  # 3+ days of data
                lookback_days = min(3, len(prices))  # 3 days max
                recent_high = np.max(prices[-lookback_days:])
                features['new_3d_high'] = 1 if prices[-1] >= recent_high else 0
                features['pct_from_3d_high'] = (prices[-1] - recent_high) / recent_high
            else:
                features['new_3d_high'] = 1  # Assume new high for new stocks
                features['pct_from_3d_high'] = 0
            
            # 6. 3-DAY VOLUME COMPARISON
            if len(volumes) >= 3:  # 3+ days
                three_day_avg_volume = np.mean(volumes[-3:])
                features['volume_vs_3d'] = current_volume / three_day_avg_volume if three_day_avg_volume > 0 else 1
            else:
                features['volume_vs_3d'] = features['live_volume_spike']
            
            # 7. 3-DAY PRICE MOMENTUM
            if len(prices) >= 3:  # 3+ days
                features['momentum_3d'] = (prices[-1] - prices[-3]) / prices[-3]
            elif len(prices) >= 2:  # 2+ days
                features['momentum_3d'] = (prices[-1] - prices[-2]) / prices[-2]
            else:
                features['momentum_3d'] = features['live_velocity_1h']
            
            # === INTRADAY FEATURES ===
            
            # 8. CONSECUTIVE UP PERIODS
            if len(prices) >= 3:
                consecutive_up = 0
                for i in range(len(prices) - 1, 0, -1):
                    if prices[i] > prices[i-1]:
                        consecutive_up += 1
                    else:
                        break
                features['consecutive_up_periods'] = consecutive_up
            else:
                features['consecutive_up_periods'] = 0
            
            # 9. MOMENTUM CONSISTENCY
            if len(prices) >= 5:
                recent_returns = np.diff(prices[-5:]) / prices[-5:-1]
                positive_moves = sum(1 for r in recent_returns if r > 0)
                features['momentum_consistency'] = positive_moves / len(recent_returns)
            else:
                features['momentum_consistency'] = 0.5
            
            # 10. BREAKOUT STRENGTH
            features['breakout_strength'] = features['live_velocity_1h'] * features['live_volume_spike']
            
            # 11. VOLATILITY SURGE
            if len(prices) >= 6:
                recent_volatility = np.std(prices[-3:]) / np.mean(prices[-3:])
                historical_volatility = np.std(prices[-6:-3]) / np.mean(prices[-6:-3])
                features['volatility_surge'] = recent_volatility / historical_volatility if historical_volatility > 0 else 1
            else:
                features['volatility_surge'] = 1
            
            # 12. DOLLAR VOLUME EXPLOSION
            dollar_volume = prices[-1] * volumes[-1]
            if len(prices) > 3:
                avg_dollar_volume = np.mean(prices[-4:-1] * volumes[-4:-1])
                features['dollar_volume_ratio'] = dollar_volume / avg_dollar_volume if avg_dollar_volume > 0 else 1
            else:
                features['dollar_volume_ratio'] = features['live_volume_spike']
            
            # === ENHANCED PEAK/VALLEY DETECTION FEATURES ===
            
            # 13. LOCAL PEAK/VALLEY DETECTION (FIXED LOGIC)
            if len(prices) >= 5:
                # Detect if current price is a local peak or valley
                is_peak = (prices[-1] > prices[-2] and prices[-2] > prices[-3] and
                          prices[-1] > prices[-4] and prices[-1] > prices[-5])
                is_valley = (prices[-1] < prices[-2] and prices[-2] < prices[-3] and
                            prices[-1] < prices[-4] and prices[-1] < prices[-5])
                
                # INVERT PEAK LOGIC: Peak = BAD for buying, Valley = GOOD for buying
                features['avoid_peak_signal'] = 1 if is_peak else 0  # HIGH = avoid buying
                features['buy_valley_signal'] = 1 if is_valley else 0  # HIGH = good for buying
                
                # Distance from recent peak/valley (CORRECTED)
                recent_max = np.max(prices[-10:]) if len(prices) >= 10 else np.max(prices)
                recent_min = np.min(prices[-10:]) if len(prices) >= 10 else np.min(prices)
                
                # Higher distance from peak = BETTER for buying
                features['safe_distance_from_peak'] = (recent_max - prices[-1]) / recent_max if recent_max > 0 else 0
                # Lower distance from valley = BETTER for buying
                features['close_to_valley'] = 1 - ((prices[-1] - recent_min) / (recent_max - recent_min)) if recent_max > recent_min else 0.5
            else:
                features['avoid_peak_signal'] = 0
                features['buy_valley_signal'] = 0
                features['safe_distance_from_peak'] = 0.5
                features['close_to_valley'] = 0.5
            
            # 14. PRICE PATTERN RECOGNITION
            if len(prices) >= 6:
                # Higher highs, higher lows pattern (bullish)
                recent_highs = [max(highs[i-1:i+2]) for i in range(2, len(highs)-2)]
                recent_lows = [min(lows[i-1:i+2]) for i in range(2, len(lows)-2)]
                
                if len(recent_highs) >= 2:
                    higher_highs = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1])
                    higher_lows = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] > recent_lows[i-1])
                    features['higher_highs_pattern'] = higher_highs / max(1, len(recent_highs)-1)
                    features['higher_lows_pattern'] = higher_lows / max(1, len(recent_lows)-1)
                else:
                    features['higher_highs_pattern'] = 0
                    features['higher_lows_pattern'] = 0
                
                # Price position within recent range
                recent_range_high = np.max(highs[-6:])
                recent_range_low = np.min(lows[-6:])
                if recent_range_high > recent_range_low:
                    position_in_range = (prices[-1] - recent_range_low) / (recent_range_high - recent_range_low)
                    features['price_position_in_range'] = position_in_range
                else:
                    features['price_position_in_range'] = 0.5
            else:
                features['higher_highs_pattern'] = 0
                features['higher_lows_pattern'] = 0
                features['price_position_in_range'] = 0.5
            
            # 15. MOVING AVERAGE ANALYSIS
            if len(prices) >= 5:
                # Simple moving averages
                ma_3 = np.mean(prices[-3:])
                ma_5 = np.mean(prices[-5:])
                
                features['price_vs_ma3'] = (prices[-1] - ma_3) / ma_3
                features['price_vs_ma5'] = (prices[-1] - ma_5) / ma_5
                features['ma3_vs_ma5'] = (ma_3 - ma_5) / ma_5 if ma_5 > 0 else 0
                
                # MA slope (trend direction)
                if len(prices) >= 7:
                    ma_5_prev = np.mean(prices[-7:-2])
                    features['ma5_slope'] = (ma_5 - ma_5_prev) / ma_5_prev if ma_5_prev > 0 else 0
                else:
                    features['ma5_slope'] = 0
            else:
                features['price_vs_ma3'] = 0
                features['price_vs_ma5'] = 0
                features['ma3_vs_ma5'] = 0
                features['ma5_slope'] = 0
            
            # 16. BOLLINGER BAND POSITION
            if len(prices) >= 10:
                period = min(10, len(prices))
                sma = np.mean(prices[-period:])
                std = np.std(prices[-period:])
                
                upper_band = sma + (2 * std)
                lower_band = sma - (2 * std)
                
                if upper_band > lower_band:
                    features['bollinger_position'] = (prices[-1] - lower_band) / (upper_band - lower_band)
                    features['bollinger_squeeze'] = std / sma if sma > 0 else 0
                else:
                    features['bollinger_position'] = 0.5
                    features['bollinger_squeeze'] = 0
                
                # Bollinger breakout detection - use consistent naming
                features['above_upper_bollinger'] = 1 if prices[-1] > upper_band else 0
                features['below_lower_bollinger'] = 1 if prices[-1] < lower_band else 0
            else:
                features['bollinger_position'] = 0.5
                features['bollinger_squeeze'] = 0
                features['above_upper_bollinger'] = 0
                features['below_lower_bollinger'] = 0
            
            # 17. RSI-LIKE MOMENTUM INDICATOR
            if len(prices) >= 8:
                price_changes = np.diff(prices[-8:])
                gains = np.where(price_changes > 0, price_changes, 0)
                losses = np.where(price_changes < 0, -price_changes, 0)
                
                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                avg_loss = np.mean(losses) if len(losses) > 0 else 0
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    features['rsi_momentum'] = rsi / 100  # Normalize to 0-1
                else:
                    features['rsi_momentum'] = 1.0 if avg_gain > 0 else 0.5
            else:
                features['rsi_momentum'] = 0.5
            
            # 18. FIBONACCI RETRACEMENT LEVELS
            if len(prices) >= 10:
                period_high = np.max(prices[-10:])
                period_low = np.min(prices[-10:])
                
                if period_high > period_low:
                    fib_range = period_high - period_low
                    
                    # Key Fibonacci levels
                    fib_236 = period_high - (0.236 * fib_range)
                    fib_382 = period_high - (0.382 * fib_range)
                    fib_618 = period_high - (0.618 * fib_range)
                    
                    # Distance to nearest Fibonacci level
                    fib_levels = [fib_236, fib_382, fib_618]
                    distances = [abs(prices[-1] - level) / period_high for level in fib_levels]
                    features['nearest_fib_distance'] = min(distances)
                    
                    # Which Fibonacci zone we're in
                    if prices[-1] >= fib_236:
                        features['fib_zone'] = 0.8  # Near high
                    elif prices[-1] >= fib_382:
                        features['fib_zone'] = 0.6  # Upper zone
                    elif prices[-1] >= fib_618:
                        features['fib_zone'] = 0.4  # Middle zone
                    else:
                        features['fib_zone'] = 0.2  # Lower zone
                else:
                    features['nearest_fib_distance'] = 0
                    features['fib_zone'] = 0.5
            else:
                features['nearest_fib_distance'] = 0
                features['fib_zone'] = 0.5
            
            # 19. SUPPORT/RESISTANCE ANALYSIS
            if len(prices) >= 8:
                # Find potential support/resistance levels
                highs_recent = highs[-8:]
                lows_recent = lows[-8:]
                
                # Count touches near current price (within 2%)
                current_price = prices[-1]
                tolerance = current_price * 0.02
                
                resistance_touches = sum(1 for h in highs_recent if abs(h - current_price) <= tolerance)
                support_touches = sum(1 for l in lows_recent if abs(l - current_price) <= tolerance)
                
                features['resistance_strength'] = resistance_touches / len(highs_recent)
                features['support_strength'] = support_touches / len(lows_recent)
                
                # Breakout potential
                max_resistance = np.max(highs_recent)
                min_support = np.min(lows_recent)
                
                # Support/Resistance analysis - use consistent naming
                features['resistance_breakout_potential'] = 1 if current_price >= max_resistance * 0.98 else 0
                features['support_breakdown_risk'] = 1 if current_price <= min_support * 1.02 else 0
            else:
                features['resistance_strength'] = 0
                features['support_strength'] = 0
                features['resistance_breakout_potential'] = 0
                features['support_breakdown_risk'] = 0
            
            # 20. TREND REVERSAL INDICATORS
            if len(prices) >= 6:
                # Doji-like patterns (small body relative to range)
                body_size = abs(prices[-1] - prices[-2]) if len(prices) >= 2 else 0
                range_size = highs[-1] - lows[-1]
                features['doji_pattern'] = 1 - (body_size / range_size) if range_size > 0 else 0
                
                # Hammer/shooting star patterns
                if range_size > 0:
                    upper_wick = highs[-1] - max(prices[-1], prices[-2] if len(prices) >= 2 else prices[-1])
                    lower_wick = min(prices[-1], prices[-2] if len(prices) >= 2 else prices[-1]) - lows[-1]
                    
                    features['hammer_pattern'] = lower_wick / range_size
                    features['shooting_star_pattern'] = upper_wick / range_size
                else:
                    features['hammer_pattern'] = 0
                    features['shooting_star_pattern'] = 0
                
                # Divergence detection (price vs volume)
                if len(volumes) >= 6:
                    price_trend = (prices[-1] - prices[-6]) / prices[-6] if prices[-6] > 0 else 0
                    volume_trend = (volumes[-1] - np.mean(volumes[-6:-1])) / np.mean(volumes[-6:-1]) if np.mean(volumes[-6:-1]) > 0 else 0
                    
                    # Positive divergence: price down, volume up (bullish)
                    # Negative divergence: price up, volume down (bearish)
                    features['price_volume_divergence'] = price_trend - (volume_trend * 0.1)  # Scale volume trend
                else:
                    features['price_volume_divergence'] = 0
            else:
                features['doji_pattern'] = 0
                features['hammer_pattern'] = 0
                features['shooting_star_pattern'] = 0
                features['price_volume_divergence'] = 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting live features for {symbol}: {e}")
            return None
    
    async def detect_momentum(self, symbol: str, current_time: datetime = None) -> Dict:
        """Detect momentum for a symbol in real-time with quality analysis"""
        try:
            if current_time is None:
                current_time = datetime.now()
            
            # Extract live features
            features = await self.extract_live_features(symbol, current_time)
            
            if not features:
                return {'symbol': symbol, 'momentum_score': 0, 'error': 'No features extracted'}
            
            # Validate data quality before scoring
            if not self._validate_features(features):
                return {
                    'symbol': symbol,
                    'momentum_score': 0,
                    'error': 'Invalid or stale data detected',
                    'method': 'data_validation'
                }
            
            # QUALITY ANALYSIS - Check if this is a high-quality setup
            quality_analysis = await self._analyze_trade_quality(symbol, features, current_time)
            
            # PRIORITIZE XGBOOST - trained on actual explosive moves like CRCL
            if self.is_trained:
                # Use trained XGBoost model (PRIMARY METHOD)
                X = pd.DataFrame([features])[self.feature_names]
                X_scaled = self.scaler.transform(X)
                momentum_prob = self.model.predict_proba(X_scaled)[0, 1]
                momentum_score = momentum_prob * 100  # Convert to 0-100 scale
                
                # Log raw XGBoost output for analysis
                logger.debug(f"{symbol} XGBoost raw: prob={momentum_prob:.4f}, score={momentum_score:.1f}")
                
                # Apply quality adjustments to XGBoost score
                adjusted_score = self._apply_quality_adjustments(momentum_score, quality_analysis)
                
                # Get top contributing features from XGBoost
                top_factors = self._get_top_factors(features)
                
                return {
                    'symbol': symbol,
                    'momentum_score': adjusted_score,
                    'original_score': momentum_score,
                    'raw_xgboost_prob': momentum_prob,  # Include raw probability
                    'raw_xgboost_score': momentum_score,  # Include raw 0-100 score
                    'confidence': momentum_prob,
                    'method': 'xgboost',
                    'quality_analysis': quality_analysis,
                    'top_factors': top_factors,
                    'features': features,
                    'timestamp': datetime.now()
                }
            else:
                # Fallback to rule-based if model not trained
                score = self._rule_based_scoring(features)
                adjusted_score = self._apply_quality_adjustments(score, quality_analysis)
                
                return {
                    'symbol': symbol,
                    'momentum_score': adjusted_score,
                    'original_score': score,
                    'method': 'rule_based_fallback',
                    'quality_analysis': quality_analysis,
                    'features': features,
                    'timestamp': datetime.now()
                }
            
        except Exception as e:
            logger.error(f"Momentum detection failed for {symbol}: {e}")
            return {'symbol': symbol, 'momentum_score': 0, 'error': str(e)}
    
    def _rule_based_scoring(self, features: Dict) -> float:
        """Simplified rule-based scoring - minimal fallback when XGBoost unavailable"""
        score = 0
        
        # Simple price momentum (primary signal)
        daily_velocity = features.get('live_velocity_1h', 0)
        if daily_velocity > 0.05:  # 5%+ move
            score += 40
        
        # Volume confirmation
        volume_spike = features.get('live_volume_spike', 1)
        if volume_spike > 1.5:  # 1.5x+ volume
            score += 30
        
        # New high breakout
        if features.get('new_3d_high', 0) == 1:
            score += 30
        
        # Ensure rule-based score stays within 0-100 range
        return max(0, min(score, 100))
    
    def _validate_features(self, features: Dict) -> bool:
        """Validate feature quality to prevent fake signals during market closure"""
        try:
            # Check for basic feature presence
            required_features = ['live_velocity_1h', 'live_volume_spike', 'momentum_3d']
            for feature in required_features:
                if feature not in features:
                    logger.debug(f"Missing required feature: {feature}")
                    return False
            
            # Check for reasonable velocity values (not extreme outliers from stale data)
            velocity = features.get('live_velocity_1h', 0)
            if abs(velocity) > 2.0:  # 200%+ moves are suspicious during closure
                logger.debug(f"Suspicious velocity: {velocity}")
                return False
            
            # Check for reasonable volume ratios
            volume_spike = features.get('live_volume_spike', 1)
            if volume_spike > 100:  # 100x volume is unrealistic during closure
                logger.debug(f"Suspicious volume spike: {volume_spike}")
                return False
            
            # Check for data freshness indicators
            consecutive_up = features.get('consecutive_up_periods', 0)
            if consecutive_up > 10:  # Too many consecutive ups suggests stale data
                logger.debug(f"Suspicious consecutive periods: {consecutive_up}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Feature validation error: {e}")
            return False
    
    def _get_top_factors(self, features: Dict) -> List[str]:
        """Get top contributing factors"""
        if not self.feature_importance:
            return []
        
        factor_scores = []
        for feature, value in features.items():
            if feature in self.feature_importance:
                importance = self.feature_importance[feature]
                factor_scores.append((feature, value, importance, value * importance))
        
        # Sort by contribution (value * importance)
        factor_scores.sort(key=lambda x: x[3], reverse=True)
        
        # Return top 3 factors
        top_factors = []
        for feature, value, importance, contribution in factor_scores[:3]:
            top_factors.append(f"{feature}={value:.3f}")
        
        return top_factors
    
    async def train_model_from_historical_movers(self, lookback_days: int = 90):
        """Train model from historical explosive moves with enhanced data"""
        logger.info("📚 Training momentum model with enhanced historical data...")
        
        try:
            # Use enhanced training data creation
            X_train, y_train = await self._create_enhanced_training_data(lookback_days)
            
            if len(X_train) < 30:  # Lower threshold since we have guaranteed data
                logger.warning("Still insufficient training data after enhancement")
                # Force creation of minimal model with what we have
                if len(X_train) < 10:
                    logger.error("Cannot train with less than 10 samples")
                    return
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X_train)
            
            # Split for validation
            X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y_train, test_size=0.2, random_state=42)
            
            # Train XGBoost
            params = {
                'objective': 'binary:logistic',
                'max_depth': 4,
                'learning_rate': 0.1,
                'n_estimators': 150,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1) if sum(y_train == 1) > 0 else 1,
                'eval_metric': 'auc',
                'random_state': 42
            }
            
            self.model = xgb.XGBClassifier(**params)
            self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            # Store feature names and importance
            self.feature_names = list(X_train.columns)
            importance = self.model.feature_importances_
            self.feature_importance = dict(zip(self.feature_names, importance))
            
            self.is_trained = True
            
            # Save model
            self._save_model()
            
            # Log top features
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            logger.info("Top momentum features:")
            for i, (feature, imp) in enumerate(sorted_features[:5]):
                logger.info(f"  {i+1}. {feature}: {imp:.4f}")
            
            val_score = self.model.score(X_val, y_val)
            logger.info(f"Model trained! Validation accuracy: {val_score:.2%}")
            logger.info(f"Training samples: {len(X_train)}, Explosive moves: {sum(y_train)}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
    
    async def _create_enhanced_training_data(self, lookback_days: int) -> Tuple[pd.DataFrame, np.ndarray]:
        """Create training data with historical explosive moves + live market data"""
        all_features = []
        labels = []
        
        # First, add historical explosive moves for guaranteed training data
        logger.info("Adding historical explosive moves to training data...")
        
        for move_data in self.HISTORICAL_EXPLOSIVE_MOVES:
            try:
                symbol = move_data['symbol']
                move_date = datetime.strptime(move_data['date'], '%Y-%m-%d')
                
                # Get features for the day before the explosive move
                pre_move_date = move_date - timedelta(days=1)
                features = await self.extract_live_features(symbol, pre_move_date)
                
                if features:
                    # This was an explosive move, label as 1
                    all_features.append(features)
                    labels.append(1)
                    logger.debug(f"Added explosive move: {symbol} on {move_data['date']}")
                
                # Also add some negative examples from days without moves
                for days_offset in [7, 14, 21]:  # Random days likely without explosive moves
                    negative_date = move_date - timedelta(days=days_offset)
                    neg_features = await self.extract_live_features(symbol, negative_date)
                    
                    if neg_features:
                        all_features.append(neg_features)
                        labels.append(0)
                
                await asyncio.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Failed to process historical move {move_data}: {e}")
                continue
        
        logger.info(f"Added {sum(labels)} explosive moves from historical data")
        
        # Then add recent market data for current patterns
        try:
            # Get recent gainers for additional positive examples
            recent_gainers = await self.polygon.get_market_movers('gainers')
            
            # Sample recent dates
            end_date = datetime.now() - timedelta(days=1)
            
            for days_ago in range(1, min(lookback_days, 30), 3):  # Every 3 days
                sample_date = end_date - timedelta(days=days_ago)
                
                # Process top gainers
                for symbol in recent_gainers[:20]:  # Top 20
                    try:
                        features = await self.extract_live_features(symbol, sample_date)
                        
                        if features:
                            # Check if this led to explosive move
                            future_return = await self._get_future_return(symbol, sample_date, days=3)
                            
                            if future_return is not None:
                                # Label based on whether it had explosive move
                                label = 1 if future_return > 0.15 else 0  # 15%+ in 3 days
                                all_features.append(features)
                                labels.append(label)
                        
                        await asyncio.sleep(0.05)  # Rate limiting
                        
                    except Exception as e:
                        continue
            
        except Exception as e:
            logger.warning(f"Failed to get recent market data: {e}")
        
        if len(all_features) < 50:
            logger.warning(f"Only {len(all_features)} training samples collected")
            # Add synthetic augmentation if needed
            all_features, labels = self._augment_training_data(all_features, labels)
        
        df = pd.DataFrame(all_features)
        logger.info(f"Created {len(df)} training samples, {sum(labels)} explosive moves")
        
        return df, np.array(labels)
    
    def _augment_training_data(self, features_list: List[Dict], labels: List[int]) -> Tuple[List[Dict], List[int]]:
        """Augment training data with synthetic examples"""
        augmented_features = list(features_list)
        augmented_labels = list(labels)
        
        # Get feature names from existing data or use defaults
        if features_list:
            feature_names = list(features_list[0].keys())
        else:
            feature_names = [
                'live_velocity_5min', 'live_velocity_30min', 'live_velocity_1h',
                'live_volume_spike', 'live_acceleration', 'live_range_expansion',
                'new_3d_high', 'pct_from_3d_high', 'volume_vs_3d', 'momentum_3d',
                'consecutive_up_periods', 'momentum_consistency', 'breakout_strength',
                'volatility_surge', 'dollar_volume_ratio',
                # CORRECTED PEAK/VALLEY FEATURES
                'avoid_peak_signal', 'buy_valley_signal', 'safe_distance_from_peak', 'close_to_valley',
                'price_position_in_range', 'above_upper_bollinger', 'below_lower_bollinger',
                'resistance_breakout_potential', 'support_breakdown_risk'
            ]
        
        # Create synthetic positive examples by combining strong features
        for _ in range(20):  # Add 20 synthetic positives
            synthetic = {}
            for feature in feature_names:
                if 'velocity' in feature:
                    synthetic[feature] = np.random.uniform(0.05, 0.25)  # 5-25% moves
                elif 'volume' in feature:
                    synthetic[feature] = np.random.uniform(2.0, 5.0)  # 2-5x volume
                elif 'acceleration' in feature:
                    synthetic[feature] = np.random.uniform(0.01, 0.05)
                elif feature == 'new_3d_high':
                    synthetic[feature] = 1  # New highs are positive signals
                else:
                    synthetic[feature] = np.random.uniform(0.5, 1.5)
            
            augmented_features.append(synthetic)
            augmented_labels.append(1)
        
        # Create synthetic negative examples
        for _ in range(30):  # Add 30 synthetic negatives
            synthetic = {}
            for feature in feature_names:
                if 'velocity' in feature:
                    synthetic[feature] = np.random.uniform(-0.02, 0.02)  # Small moves
                elif 'volume' in feature:
                    synthetic[feature] = np.random.uniform(0.5, 1.5)  # Normal volume
                elif feature == 'new_3d_high':
                    synthetic[feature] = 0  # No new highs
                else:
                    synthetic[feature] = np.random.uniform(0.8, 1.2)
            
            augmented_features.append(synthetic)
            augmented_labels.append(0)
        
        return augmented_features, augmented_labels
    
    async def _analyze_trade_quality(self, symbol: str, features: Dict, current_time: datetime) -> Dict:
        """Analyze trade quality to filter out low-quality setups and avoid peak buying"""
        try:
            quality = {
                'volume_consistency': 0.5,
                'price_action_quality': 0.5,
                'trend_strength': 0.5,
                'liquidity_score': 0.5,
                'timing_score': 0.5,
                'peak_avoidance_score': 0.5,
                'overall_grade': 'C',
                'penalties': [],
                'bonuses': []
            }
            
            # Get enhanced price data for analysis
            end_time = current_time
            start_time = end_time - timedelta(days=5)
            
            try:
                bars = await self.polygon.list_aggs(symbol, 1, 'day', start_time, end_time, limit=10)
                if len(bars) < 3:
                    quality['penalties'].append('Insufficient price history')
                    return quality
                
                prices = np.array([bar.close for bar in bars])
                volumes = np.array([bar.volume for bar in bars])
                highs = np.array([bar.high for bar in bars])
                lows = np.array([bar.low for bar in bars])
                
                # === STRICT PEAK AVOIDANCE ANALYSIS ===
                current_price = prices[-1]
                daily_high = highs[-1]
                daily_low = lows[-1]
                
                # 1. DAILY HIGH POSITION CHECK (STRICT)
                if daily_high > daily_low:  # Avoid division by zero
                    price_position = (current_price - daily_low) / (daily_high - daily_low)
                    max_position = config.MOMENTUM_CONFIG['max_daily_high_position']
                    
                    if price_position > max_position:
                        quality['penalties'].append(f'BLOCKED: Too close to daily high: {price_position:.1%} position (max: {max_position:.1%})')
                        quality['peak_avoidance_score'] = 0.0  # Complete block
                    elif price_position > 0.5:
                        quality['penalties'].append(f'WARNING: Above 50% of daily range: {price_position:.1%}')
                        quality['peak_avoidance_score'] = 0.3
                    elif price_position < 0.3:
                        quality['bonuses'].append(f'GOOD: Low in daily range: {price_position:.1%}')
                        quality['peak_avoidance_score'] = 1.0
                    else:
                        quality['peak_avoidance_score'] = 0.6
                
                # 2. GAP UP CHECK
                if len(prices) >= 2:
                    gap_pct = (prices[-1] - prices[-2]) / prices[-2]
                    max_gap_up = config.MOMENTUM_CONFIG.get('max_gap_up_pct', 0.03)
                    
                    if gap_pct > max_gap_up:
                        quality['penalties'].append(f'BLOCKED: Gap up too large: {gap_pct:.1%} (max: {max_gap_up:.1%})')
                        quality['peak_avoidance_score'] = min(quality['peak_avoidance_score'], 0.1)
                
                # 3. RECENT HIGH PULLBACK CHECK
                if len(prices) >= 10:
                    recent_high = np.max(prices[-10:])
                    min_pullback = config.MOMENTUM_CONFIG.get('min_pullback_from_high', 0.10)
                    pullback_from_high = (recent_high - current_price) / recent_high
                    
                    if pullback_from_high < min_pullback:
                        quality['penalties'].append(f'BLOCKED: Too close to recent high - pullback: {pullback_from_high:.1%} (min: {min_pullback:.1%})')
                        quality['peak_avoidance_score'] = min(quality['peak_avoidance_score'], 0.1)
                    elif pullback_from_high > 0.05:
                        quality['bonuses'].append(f'GOOD: Decent pullback from high: {pullback_from_high:.1%}')
                
                # 2. CONSOLIDATION CHECK
                if len(prices) >= 5:
                    recent_prices = prices[-5:]
                    price_volatility = np.std(recent_prices) / np.mean(recent_prices)
                    max_volatility = config.MOMENTUM_CONFIG['max_recent_volatility']
                    
                    if price_volatility > max_volatility:
                        quality['penalties'].append(f'High recent volatility: {price_volatility:.2%}')
                        quality['peak_avoidance_score'] *= 0.7
                    else:
                        quality['bonuses'].append('Good price consolidation')
                        quality['peak_avoidance_score'] *= 1.1
                
                # 3. MOMENTUM DECELERATION CHECK
                if config.MOMENTUM_CONFIG['avoid_momentum_deceleration']:
                    acceleration = features.get('live_acceleration', 0)
                    if acceleration < -0.01:  # Momentum is decelerating
                        quality['penalties'].append(f'Momentum decelerating: {acceleration:.3f}')
                        quality['peak_avoidance_score'] *= 0.6
                    elif acceleration > 0.01:
                        quality['bonuses'].append('Momentum accelerating')
                        quality['peak_avoidance_score'] *= 1.1
                
                # 1. VOLUME CONSISTENCY CHECK
                if len(volumes) >= 3:
                    recent_volume = volumes[-1]
                    avg_volume = np.mean(volumes[:-1])
                    volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
                    
                    # More forgiving volume analysis - sometimes low volume precedes explosive moves
                    if volume_ratio > 10:  # 10x+ volume might be news-driven
                        quality['penalties'].append(f'Extreme volume spike: {volume_ratio:.1f}x')
                        quality['volume_consistency'] = 0.2
                    elif volume_ratio >= 3:  # Good volume spike
                        quality['bonuses'].append(f'Strong volume: {volume_ratio:.1f}x')
                        quality['volume_consistency'] = 0.9
                    elif volume_ratio >= 1.5:  # Moderate volume
                        quality['volume_consistency'] = 0.7
                    elif volume_ratio >= 0.8:  # Normal/slightly low volume (acceptable for momentum)
                        quality['volume_consistency'] = 0.6
                    else:  # Very low volume (still penalized but less severely)
                        quality['penalties'].append(f'Low volume: {volume_ratio:.1f}x')
                        quality['volume_consistency'] = 0.4  # Reduced penalty from 0.3 to 0.4
                
                # 2. PRICE ACTION QUALITY
                if len(prices) >= 3:
                    # Check for choppy vs clean price action
                    price_changes = np.diff(prices) / prices[:-1]
                    direction_changes = sum(1 for i in range(1, len(price_changes))
                                          if (price_changes[i] > 0) != (price_changes[i-1] > 0))
                    choppiness = direction_changes / max(1, len(price_changes) - 1)
                    
                    # Use config values for choppiness thresholds with momentum-based tolerance
                    base_max_choppiness = config.MOMENTUM_CONFIG.get('max_price_choppiness', 0.6)
                    
                    # For high momentum, be more tolerant of choppiness (volatility can indicate explosive potential)
                    # Get momentum score from features to adjust tolerance
                    momentum_velocity = features.get('live_velocity_1h', 0) * 100  # Convert to percentage
                    if momentum_velocity > 20:  # Very high momentum (20%+ moves)
                        max_choppiness = base_max_choppiness * 2.0  # Double tolerance for explosive moves
                    elif momentum_velocity > 10:  # High momentum (10%+ moves)
                        max_choppiness = base_max_choppiness * 1.5  # 50% more tolerance
                    else:
                        max_choppiness = base_max_choppiness
                    
                    if choppiness > max_choppiness:  # Too choppy even with momentum adjustment
                        quality['penalties'].append(f'Choppy price action: {choppiness:.2f}')
                        quality['price_action_quality'] = 0.2
                    elif choppiness < 0.3:  # Clean trend
                        quality['bonuses'].append('Clean price action')
                        quality['price_action_quality'] = 0.9
                    else:
                        quality['price_action_quality'] = 0.6
                
                # 3. TREND STRENGTH (R-squared)
                if len(prices) >= 4:
                    x = np.arange(len(prices))
                    try:
                        correlation = np.corrcoef(x, prices)[0, 1]
                        r_squared = correlation ** 2
                        
                        # Use config values for trend strength thresholds
                        min_trend_strength = config.MOMENTUM_CONFIG.get('min_trend_strength', 0.6)
                        
                        if r_squared > 0.8:  # Strong trend
                            quality['bonuses'].append(f'Strong trend: R²={r_squared:.2f}')
                            quality['trend_strength'] = 0.9
                        elif r_squared > min_trend_strength:  # Moderate trend (uses config)
                            quality['trend_strength'] = 0.7
                        else:  # Weak trend
                            quality['penalties'].append(f'Weak trend: R²={r_squared:.2f}')
                            quality['trend_strength'] = 0.3
                    except:
                        quality['trend_strength'] = 0.5
                
                # 4. LIQUIDITY SCORE (based on bid-ask spread and volume)
                current_volume = volumes[-1]
                if current_volume > 1000000:  # 1M+ volume
                    quality['bonuses'].append('High liquidity')
                    quality['liquidity_score'] = 0.9
                elif current_volume > 500000:  # 500K+ volume
                    quality['liquidity_score'] = 0.7
                elif current_volume > 100000:  # 100K+ volume
                    quality['liquidity_score'] = 0.5
                else:  # Low volume
                    quality['penalties'].append('Low liquidity')
                    quality['liquidity_score'] = 0.3
                
                # 5. TIMING SCORE (avoid gap fills, prefer breakouts)
                if len(prices) >= 2:
                    current_price = prices[-1]
                    prev_close = prices[-2]
                    gap_pct = (current_price - prev_close) / prev_close
                    
                    if abs(gap_pct) > 0.05:  # 5%+ gap
                        if gap_pct > 0:  # Gap up
                            quality['bonuses'].append(f'Gap up: {gap_pct:.1%}')
                            quality['timing_score'] = 0.8
                        else:  # Gap down
                            quality['penalties'].append(f'Gap down: {gap_pct:.1%}')
                            quality['timing_score'] = 0.3
                    else:
                        quality['timing_score'] = 0.6
                
            except Exception as e:
                logger.debug(f"Quality analysis data error for {symbol}: {e}")
                quality['penalties'].append('Data quality issues')
            
            # Calculate overall grade with PEAK AVOIDANCE AS PRIMARY FILTER
            base_scores = [
                quality['volume_consistency'],
                quality['price_action_quality'],
                quality['trend_strength'],
                quality['liquidity_score'],
                quality['timing_score']
            ]
            
            # Peak avoidance is now the PRIMARY filter (weighted 5x)
            peak_score = quality['peak_avoidance_score']
            weighted_scores = base_scores + [peak_score] * 5  # 5x weight for peak avoidance
            avg_score = np.mean(weighted_scores)
            
            # ULTRA-STRICT grading - peak avoidance must be excellent
            if peak_score >= 0.8 and avg_score >= 0.75:
                quality['overall_grade'] = 'A'
            elif peak_score >= 0.6 and avg_score >= 0.65:
                quality['overall_grade'] = 'B'
            elif peak_score >= 0.5 and avg_score >= 0.55:
                quality['overall_grade'] = 'C'
            else:
                quality['overall_grade'] = 'F'
            
            # CRITICAL: Auto-fail if ANY peak avoidance issue
            if peak_score < 0.5:
                quality['overall_grade'] = 'F'
                quality['penalties'].append('CRITICAL: Failed strict peak avoidance check')
            
            # Additional auto-fail conditions for peak buying
            penalties_str = ' '.join(quality['penalties'])
            if 'BLOCKED:' in penalties_str:
                quality['overall_grade'] = 'F'
                quality['penalties'].append('BLOCKED: Peak buying protection activated')
            
            return quality
            
        except Exception as e:
            logger.error(f"Trade quality analysis failed for {symbol}: {e}")
            return {
                'volume_consistency': 0.3,
                'price_action_quality': 0.3,
                'trend_strength': 0.3,
                'liquidity_score': 0.3,
                'timing_score': 0.3,
                'overall_grade': 'F',
                'penalties': ['Analysis failed'],
                'bonuses': []
            }
    
    def _apply_quality_adjustments(self, momentum_score: float, quality_analysis: Dict) -> float:
        """Minimal quality adjustments - trust XGBoost predictions"""
        try:
            adjusted_score = momentum_score
            
            # Only apply severe penalty for completely failed analysis
            grade = quality_analysis.get('overall_grade', 'C')
            if grade == 'F':
                adjusted_score *= 0.8  # Light penalty for failed quality
            
            # Only penalize for critical blocking issues
            penalties = quality_analysis.get('penalties', [])
            critical_penalties = ['Analysis failed', 'Too close to daily high']
            has_critical = any(critical in penalty for penalty in penalties for critical in critical_penalties)
            
            if has_critical:
                adjusted_score *= 0.9  # Light penalty for critical issues
            
            # Ensure score stays within 0-100 scale
            adjusted_score = max(0, min(adjusted_score, 100))
            
            return adjusted_score
            
        except Exception as e:
            logger.error(f"Quality adjustment failed: {e}")
            return momentum_score  # Return original score if adjustment fails
    
    async def _get_future_return(self, symbol: str, date: datetime, days: int = 3) -> Optional[float]:
        """Get future return for labeling"""
        try:
            start_date = date
            end_date = date + timedelta(days=days + 2)
            
            bars = await self.polygon.list_aggs(symbol, 1, 'day', start_date, end_date)
            
            if len(bars) >= 2:
                start_price = bars[0].close
                end_idx = min(days, len(bars) - 1)
                end_price = bars[end_idx].close
                return (end_price - start_price) / start_price
            
            return None
        except:
            return None