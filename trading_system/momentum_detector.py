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
        
        # Load existing model if available
        self._load_model()
    
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
            
            # If model not trained, use simple rule-based scoring
            if not self.is_trained:
                score = self._rule_based_scoring(features)
                # Apply quality adjustments
                adjusted_score = self._apply_quality_adjustments(score, quality_analysis)
                
                return {
                    'symbol': symbol,
                    'momentum_score': adjusted_score,
                    'original_score': score,
                    'method': 'rule_based',
                    'quality_analysis': quality_analysis,
                    'features': features,
                    'timestamp': datetime.now()
                }
            
            # Use trained XGBoost model
            X = pd.DataFrame([features])[self.feature_names]
            X_scaled = self.scaler.transform(X)
            momentum_prob = self.model.predict_proba(X_scaled)[0, 1]
            momentum_score = momentum_prob * 100
            
            # Apply quality adjustments to XGBoost score too
            adjusted_score = self._apply_quality_adjustments(momentum_score, quality_analysis)
            
            # Get top contributing features
            top_factors = self._get_top_factors(features)
            
            return {
                'symbol': symbol,
                'momentum_score': adjusted_score,
                'original_score': momentum_score,
                'confidence': momentum_prob,
                'method': 'xgboost',
                'quality_analysis': quality_analysis,
                'top_factors': top_factors,
                'features': features,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Momentum detection failed for {symbol}: {e}")
            return {'symbol': symbol, 'momentum_score': 0, 'error': str(e)}
    
    def _rule_based_scoring(self, features: Dict) -> float:
        """AGGRESSIVE rule-based scoring for CRCL-type explosive moves"""
        score = 0
        
        # DAILY Price velocity scoring (more aggressive for daily moves)
        daily_velocity = features.get('live_velocity_1h', 0)  # Actually daily with our current setup
        if daily_velocity > 0.25:  # 25%+ daily move = EXPLOSIVE
            score += 60
        elif daily_velocity > 0.15:  # 15%+ daily move = STRONG
            score += 45
        elif daily_velocity > 0.10:  # 10%+ daily move = MODERATE
            score += 30
        elif daily_velocity > 0.05:  # 5%+ daily move = WEAK
            score += 15
        
        # Volume explosion scoring (more aggressive)
        volume_spike = features.get('live_volume_spike', 1)
        if volume_spike > 3:  # 3x+ volume = EXPLOSIVE
            score += 30
        elif volume_spike > 2:  # 2x+ volume = STRONG
            score += 20
        elif volume_spike > 1.5:  # 1.5x+ volume = MODERATE
            score += 10
        
        # Breakout scoring (more weight)
        if features.get('new_3d_high', 0) == 1:
            score += 25  # Increased from 20
        
        # Price acceleration (detect accelerating momentum)
        acceleration = features.get('live_acceleration', 0)
        if acceleration > 0.05:  # Strong acceleration
            score += 20
        elif acceleration > 0.02:  # Moderate acceleration
            score += 10
        
        # 3-day momentum (longer term strength)
        momentum_3d = features.get('momentum_3d', 0)
        if momentum_3d > 0.30:  # 30%+ over 3 days
            score += 25
        elif momentum_3d > 0.15:  # 15%+ over 3 days
            score += 15
        
        # Consecutive moves (momentum persistence)
        consecutive = features.get('consecutive_up_periods', 0)
        score += min(consecutive * 5, 20)  # Increased multiplier
        
        return min(score, 100)
    
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
                'volatility_surge', 'dollar_volume_ratio'
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
        """Analyze trade quality to filter out low-quality setups"""
        try:
            quality = {
                'volume_consistency': 0.5,
                'price_action_quality': 0.5,
                'trend_strength': 0.5,
                'liquidity_score': 0.5,
                'timing_score': 0.5,
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
                
                # 1. VOLUME CONSISTENCY CHECK
                if len(volumes) >= 3:
                    recent_volume = volumes[-1]
                    avg_volume = np.mean(volumes[:-1])
                    volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
                    
                    # Penalize extreme volume spikes that might be unsustainable
                    if volume_ratio > 10:  # 10x+ volume might be news-driven
                        quality['penalties'].append(f'Extreme volume spike: {volume_ratio:.1f}x')
                        quality['volume_consistency'] = 0.2
                    elif volume_ratio >= 3:  # Good volume spike
                        quality['bonuses'].append(f'Strong volume: {volume_ratio:.1f}x')
                        quality['volume_consistency'] = 0.9
                    elif volume_ratio >= 1.5:  # Moderate volume
                        quality['volume_consistency'] = 0.7
                    else:  # Low volume
                        quality['penalties'].append(f'Low volume: {volume_ratio:.1f}x')
                        quality['volume_consistency'] = 0.3
                
                # 2. PRICE ACTION QUALITY
                if len(prices) >= 3:
                    # Check for choppy vs clean price action
                    price_changes = np.diff(prices) / prices[:-1]
                    direction_changes = sum(1 for i in range(1, len(price_changes))
                                          if (price_changes[i] > 0) != (price_changes[i-1] > 0))
                    choppiness = direction_changes / max(1, len(price_changes) - 1)
                    
                    if choppiness > 0.6:  # Very choppy
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
                        
                        if r_squared > 0.8:  # Strong trend
                            quality['bonuses'].append(f'Strong trend: R²={r_squared:.2f}')
                            quality['trend_strength'] = 0.9
                        elif r_squared > 0.6:  # Moderate trend
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
            
            # Calculate overall grade
            scores = [
                quality['volume_consistency'],
                quality['price_action_quality'],
                quality['trend_strength'],
                quality['liquidity_score'],
                quality['timing_score']
            ]
            avg_score = np.mean(scores)
            
            if avg_score >= 0.8:
                quality['overall_grade'] = 'A'
            elif avg_score >= 0.7:
                quality['overall_grade'] = 'B'
            elif avg_score >= 0.6:
                quality['overall_grade'] = 'C'
            elif avg_score >= 0.4:
                quality['overall_grade'] = 'D'
            else:
                quality['overall_grade'] = 'F'
            
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
        """Apply quality adjustments to momentum score"""
        try:
            adjusted_score = momentum_score
            
            # Grade-based adjustments
            grade_multipliers = {
                'A': 1.2,   # 20% bonus for excellent quality
                'B': 1.1,   # 10% bonus for good quality
                'C': 1.0,   # No adjustment for average quality
                'D': 0.8,   # 20% penalty for poor quality
                'F': 0.5    # 50% penalty for failing quality
            }
            
            grade = quality_analysis.get('overall_grade', 'C')
            adjusted_score *= grade_multipliers.get(grade, 1.0)
            
            # Penalty adjustments
            penalty_count = len(quality_analysis.get('penalties', []))
            if penalty_count > 3:  # Multiple quality issues
                adjusted_score *= 0.7  # Additional 30% penalty
            elif penalty_count > 1:  # Some quality issues
                adjusted_score *= 0.85  # Additional 15% penalty
            
            # Bonus adjustments
            bonus_count = len(quality_analysis.get('bonuses', []))
            if bonus_count > 2:  # Multiple positive factors
                adjusted_score *= 1.15  # Additional 15% bonus
            
            # Ensure score doesn't go negative or exceed reasonable bounds
            adjusted_score = max(0, min(adjusted_score, 150))
            
            return adjusted_score
            
        except Exception as e:
            logger.error(f"Quality adjustment failed: {e}")
            return momentum_score * 0.5  # Conservative penalty if adjustment fails
    
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