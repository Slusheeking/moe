#!/usr/bin/env python3
"""
GPU-Accelerated Pattern Recognition Engine
Monitors all positions in real-time for exit signals, anomalies, and market patterns.
Uses parallel GPU processing to detect patterns across entire portfolio simultaneously.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpu_trading_config import GPUTradingConfig
from database.connection import get_db_manager
from gpu_core.parallel_data_pipeline import ParallelDataPipeline


class PatternType(Enum):
    """Types of patterns to monitor."""
    MOMENTUM_EXHAUSTION = "momentum_exhaustion"
    RANGE_BREAKDOWN = "range_breakdown"
    VOLUME_CLIMAX = "volume_climax"
    REVERSAL_SIGNAL = "reversal_signal"
    CORRELATION_BREAK = "correlation_break"
    SUPPORT_BREACH = "support_breach"
    RESISTANCE_HIT = "resistance_hit"
    GAP_FILL = "gap_fill"
    UNUSUAL_ACTIVITY = "unusual_activity"
    SECTOR_ROTATION = "sector_rotation"


@dataclass
class PatternAlert:
    """Alert for detected pattern."""
    timestamp: datetime
    symbol: str
    pattern_type: PatternType
    confidence: float
    urgency: float
    message: str
    action_required: str
    supporting_data: Dict


class PatternDetector(nn.Module):
    """Neural network for pattern detection."""
    
    def __init__(self, input_dim: int, hidden_dim: int, n_patterns: int,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        # Convolutional layers for time series pattern detection
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1).to(device).to(dtype)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2).to(device).to(dtype)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=7, padding=3).to(device).to(dtype)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim).to(device).to(dtype)
        self.bn2 = nn.BatchNorm1d(hidden_dim).to(device).to(dtype)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2).to(device).to(dtype)
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(hidden_dim // 2, hidden_dim // 4, 
                           num_layers=2, batch_first=True, 
                           bidirectional=True).to(device).to(dtype)
        
        # Pattern classification heads
        self.pattern_classifiers = nn.ModuleDict({
            pattern.name: nn.Linear(hidden_dim // 2, 1).to(device).to(dtype)
            for pattern in PatternType
        })
        
        # Confidence estimation
        self.confidence_head = nn.Linear(hidden_dim // 2, 1).to(device).to(dtype)
        
        # Attention mechanism for interpretability
        self.attention = nn.MultiheadAttention(hidden_dim // 2, num_heads=4, 
                                              batch_first=True).to(device).to(dtype)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for pattern detection.
        Args:
            x: Input tensor [batch_size, seq_len, features]
        Returns:
            Dictionary with pattern probabilities and confidence
        """
        # Transpose for conv1d (expects [batch, channels, length])
        x = x.transpose(1, 2)
        
        # Convolutional feature extraction
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        
        # Transpose back for LSTM
        h = h.transpose(1, 2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(h)
        
        # Self-attention
        attn_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global pooling
        # Max pooling for strong signals
        max_features = torch.max(attn_out, dim=1)[0]
        # Average pooling for overall trend
        avg_features = torch.mean(attn_out, dim=1)
        
        # Combine features
        combined_features = (max_features + avg_features) / 2
        
        # Pattern detection
        pattern_probs = {}
        for pattern_name, classifier in self.pattern_classifiers.items():
            pattern_probs[pattern_name] = torch.sigmoid(classifier(combined_features)).squeeze()
        
        # Confidence estimation
        confidence = torch.sigmoid(self.confidence_head(combined_features)).squeeze()
        
        return {
            'pattern_probabilities': pattern_probs,
            'confidence': confidence,
            'attention_weights': attention_weights,
            'features': combined_features
        }


class GPUPatternRecognitionEngine:
    """Real-time pattern recognition across all positions."""
    
    def __init__(self, config: GPUTradingConfig, data_pipeline: ParallelDataPipeline):
        self.config = config
        self.data_pipeline = data_pipeline
        self.device = config.device
        self.dtype = config.TENSOR_DTYPE
        
        # Database connection for alerts and events logging
        self.db_manager = get_db_manager()
        
        # Pattern detection parameters
        self.lookback_window = 30  # minutes of data for pattern detection
        self.min_confidence = 0.7  # minimum confidence for alerts
        
        # Initialize pattern detector
        self.pattern_detector = PatternDetector(
            input_dim=20,  # number of features per time step
            hidden_dim=64,
            n_patterns=len(PatternType),
            device=self.device,
            dtype=self.dtype
        )
        
        # Active positions tracking
        self.active_positions: Set[str] = set()
        self.position_entry_prices: Dict[str, float] = {}
        self.position_entry_times: Dict[str, datetime] = {}
        
        # Pattern history
        self.pattern_history: List[PatternAlert] = []
        self.alerts_by_symbol: Dict[str, List[PatternAlert]] = {}
        
        # GPU tensors for monitoring
        self._init_monitoring_tensors()
        
    def _init_monitoring_tensors(self):
        """Initialize GPU tensors for position monitoring."""
        n_symbols = len(self.data_pipeline.universe_symbols)
        
        # Time series buffers for each symbol
        self.price_buffer = torch.zeros((n_symbols, self.lookback_window), 
                                       device=self.device, dtype=self.dtype)
        self.volume_buffer = torch.zeros((n_symbols, self.lookback_window), 
                                        device=self.device, dtype=self.dtype)
        self.feature_buffer = torch.zeros((n_symbols, self.lookback_window, 20), 
                                         device=self.device, dtype=self.dtype)
        
        # Pattern detection results
        self.pattern_scores = torch.zeros((n_symbols, len(PatternType)), 
                                         device=self.device, dtype=self.dtype)
        
        # Exit urgency scores
        self.exit_urgency = torch.zeros(n_symbols, device=self.device, dtype=self.dtype)
        
    def add_position(self, symbol: str, entry_price: float, entry_time: Optional[datetime] = None):
        """Add position to monitoring."""
        self.active_positions.add(symbol)
        self.position_entry_prices[symbol] = entry_price
        self.position_entry_times[symbol] = entry_time or datetime.now(self.config.TIMEZONE)
        
        print(f"Added {symbol} to position monitoring (entry: ${entry_price:.2f})")
        
    def remove_position(self, symbol: str):
        """Remove position from monitoring."""
        self.active_positions.discard(symbol)
        self.position_entry_prices.pop(symbol, None)
        self.position_entry_times.pop(symbol, None)
        
        print(f"Removed {symbol} from position monitoring")
        
    def scan_patterns(self, current_time: Optional[datetime] = None) -> List[PatternAlert]:
        """Scan all positions for patterns and generate alerts."""
        if current_time is None:
            current_time = datetime.now(self.config.TIMEZONE)
        
        if not self.active_positions:
            return []
        
        alerts = []
        
        # Update feature buffers
        self._update_feature_buffers()
        
        # Get indices of active positions
        active_indices = torch.tensor([
            self.data_pipeline.symbol_to_idx[symbol] 
            for symbol in self.active_positions 
            if symbol in self.data_pipeline.symbol_to_idx
        ], device=self.device, dtype=torch.long)
        
        if len(active_indices) == 0:
            return []
        
        # Extract features for active positions
        active_features = self.feature_buffer[active_indices]
        
        # Run pattern detection
        self.pattern_detector.eval()
        with torch.no_grad():
            results = self.pattern_detector(active_features)
        
        # Process results for each position
        for i, symbol_idx in enumerate(active_indices):
            symbol = self.active_positions & set([self.data_pipeline.universe_symbols[symbol_idx]])
            if not symbol:
                continue
            symbol = list(symbol)[0]
            
            # Check each pattern type
            for pattern_type in PatternType:
                prob = results['pattern_probabilities'][pattern_type.name][i].item()
                confidence = results['confidence'][i].item()
                
                if prob > self.min_confidence and confidence > 0.6:
                    alert = self._create_pattern_alert(
                        symbol, pattern_type, prob, confidence, current_time
                    )
                    alerts.append(alert)
                    
                    # Store in history
                    self.pattern_history.append(alert)
                    if symbol not in self.alerts_by_symbol:
                        self.alerts_by_symbol[symbol] = []
                    self.alerts_by_symbol[symbol].append(alert)
        
        # Check for cross-position patterns
        if len(active_indices) > 1:
            cross_alerts = self._detect_cross_position_patterns(active_indices, current_time)
            alerts.extend(cross_alerts)
        
        return alerts
    
    def _update_feature_buffers(self):
        """Update feature buffers with latest data."""
        # Shift buffers
        self.price_buffer = torch.roll(self.price_buffer, -1, dims=1)
        self.volume_buffer = torch.roll(self.volume_buffer, -1, dims=1)
        self.feature_buffer = torch.roll(self.feature_buffer, -1, dims=1)
        
        # Add latest data
        current_prices = self.data_pipeline.realtime_features[:, 0]  # last_price
        current_volumes = self.data_pipeline.realtime_features[:, 5]  # volume_5min
        
        self.price_buffer[:, -1] = current_prices
        self.volume_buffer[:, -1] = current_volumes
        
        # Build comprehensive feature set
        for i in range(20):
            if i < self.data_pipeline.realtime_features.shape[1]:
                self.feature_buffer[:, -1, i] = self.data_pipeline.realtime_features[:, i]
        
    def _create_pattern_alert(self, symbol: str, pattern_type: PatternType,
                            probability: float, confidence: float,
                            timestamp: datetime) -> PatternAlert:
        """Create detailed pattern alert."""
        # Get current position info
        entry_price = self.position_entry_prices.get(symbol, 0)
        current_price = self._get_current_price(symbol)
        position_pnl = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        
        # Pattern-specific analysis
        if pattern_type == PatternType.MOMENTUM_EXHAUSTION:
            message = f"Momentum exhaustion detected in {symbol}"
            action = "Consider taking profits or tightening stop"
            urgency = 0.8 if position_pnl > 2 else 0.6
            
        elif pattern_type == PatternType.RANGE_BREAKDOWN:
            message = f"Price breaking out of established range in {symbol}"
            action = "Exit position if breakdown continues"
            urgency = 0.9
            
        elif pattern_type == PatternType.VOLUME_CLIMAX:
            message = f"Volume climax detected in {symbol}"
            action = "Potential reversal - consider exit"
            urgency = 0.85
            
        elif pattern_type == PatternType.REVERSAL_SIGNAL:
            message = f"Reversal pattern forming in {symbol}"
            action = "Exit immediately to preserve gains"
            urgency = 0.95
            
        elif pattern_type == PatternType.GAP_FILL:
            message = f"Gap fill complete in {symbol}"
            action = "Exit as gap momentum likely exhausted"
            urgency = 0.9
            
        else:
            message = f"{pattern_type.value} detected in {symbol}"
            action = "Monitor closely"
            urgency = 0.7
        
        # Add PnL context
        if position_pnl != 0:
            message += f" (PnL: {position_pnl:+.2f}%)"
        
        alert = PatternAlert(
            timestamp=timestamp,
            symbol=symbol,
            pattern_type=pattern_type,
            confidence=confidence,
            urgency=urgency * probability,  # Combined urgency
            message=message,
            action_required=action,
            supporting_data={
                'probability': probability,
                'entry_price': entry_price,
                'current_price': current_price,
                'position_pnl': position_pnl,
                'time_in_position': self._get_time_in_position(symbol)
            }
        )
        
        # Persist high-priority alerts to database
        if alert.urgency > 0.7:  # Only persist high-priority alerts
            self._persist_pattern_alert(alert)
        
        return alert
    
    def _detect_cross_position_patterns(self, active_indices: torch.Tensor,
                                      timestamp: datetime) -> List[PatternAlert]:
        """Detect patterns across multiple positions."""
        alerts = []
        
        # Calculate correlations between active positions
        if len(active_indices) > 1:
            active_prices = self.price_buffer[active_indices]
            
            # Normalize prices for correlation
            price_returns = torch.diff(active_prices, dim=1) / active_prices[:, :-1]
            
            # Calculate correlation matrix
            if price_returns.shape[1] > 5:  # Need enough data
                correlation_matrix = torch.corrcoef(price_returns)
                
                # Check for correlation breakdown
                for i in range(len(active_indices)):
                    for j in range(i+1, len(active_indices)):
                        historical_corr = correlation_matrix[i, j].item()
                        
                        # Recent correlation (last 10 periods)
                        recent_returns = price_returns[:, -10:]
                        if recent_returns.shape[1] > 2:
                            recent_corr = torch.corrcoef(recent_returns[[i, j]])[0, 1].item()
                            
                            # Significant correlation change
                            if abs(historical_corr - recent_corr) > 0.5:
                                symbol1 = self.data_pipeline.universe_symbols[active_indices[i]]
                                symbol2 = self.data_pipeline.universe_symbols[active_indices[j]]
                                
                                alert = PatternAlert(
                                    timestamp=timestamp,
                                    symbol=f"{symbol1}/{symbol2}",
                                    pattern_type=PatternType.CORRELATION_BREAK,
                                    confidence=0.8,
                                    urgency=0.7,
                                    message=f"Correlation breakdown between {symbol1} and {symbol2}",
                                    action_required="Review both positions for divergence",
                                    supporting_data={
                                        'historical_correlation': historical_corr,
                                        'recent_correlation': recent_corr,
                                        'correlation_change': recent_corr - historical_corr
                                    }
                                )
                                alerts.append(alert)
        
        # Sector rotation detection
        sector_performance = self._analyze_sector_performance(active_indices)
        if sector_performance['rotation_detected']:
            alert = PatternAlert(
                timestamp=timestamp,
                symbol="PORTFOLIO",
                pattern_type=PatternType.SECTOR_ROTATION,
                confidence=sector_performance['confidence'],
                urgency=0.6,
                message="Sector rotation detected in portfolio",
                action_required="Consider rebalancing sector exposure",
                supporting_data=sector_performance
            )
            alerts.append(alert)
        
        return alerts
    
    def _analyze_sector_performance(self, active_indices: torch.Tensor) -> Dict:
        """Analyze sector-level performance patterns."""
        # Group positions by sector
        sectors = {}
        for idx in active_indices:
            symbol = self.data_pipeline.universe_symbols[idx]
            if symbol in self.active_positions:
                sector = self.data_pipeline.universe_manager.sector_assignments[idx].item()
                if sector not in sectors:
                    sectors[sector] = []
                sectors[sector].append(idx.item())
        
        # Calculate sector performances
        sector_performances = {}
        for sector, indices in sectors.items():
            if indices:
                sector_prices = self.price_buffer[indices]
                sector_return = (sector_prices[:, -1] - sector_prices[:, 0]) / sector_prices[:, 0]
                sector_performances[sector] = sector_return.mean().item()
        
        # Detect rotation
        if len(sector_performances) > 1:
            perf_values = list(sector_performances.values())
            perf_std = np.std(perf_values)
            
            rotation_detected = perf_std > 0.02  # 2% performance dispersion
            confidence = min(perf_std * 20, 1.0)  # Scale to 0-1
            
            return {
                'rotation_detected': rotation_detected,
                'confidence': confidence,
                'sector_performances': sector_performances,
                'performance_dispersion': perf_std
            }
        
        return {'rotation_detected': False, 'confidence': 0}
    
    def get_exit_recommendations(self) -> Dict[str, Dict]:
        """Get exit recommendations for all positions."""
        recommendations = {}
        
        for symbol in self.active_positions:
            # Get recent alerts for symbol
            recent_alerts = [
                alert for alert in self.alerts_by_symbol.get(symbol, [])
                if (datetime.now(self.config.TIMEZONE) - alert.timestamp).total_seconds() < 300  # Last 5 min
            ]
            
            if recent_alerts:
                # Aggregate urgency
                max_urgency = max(alert.urgency for alert in recent_alerts)
                pattern_types = [alert.pattern_type.value for alert in recent_alerts]
                
                # Determine recommendation
                if max_urgency > 0.9:
                    recommendation = "EXIT_NOW"
                elif max_urgency > 0.8:
                    recommendation = "EXIT_SOON"
                elif max_urgency > 0.7:
                    recommendation = "TIGHTEN_STOP"
                else:
                    recommendation = "MONITOR"
                
                recommendations[symbol] = {
                    'recommendation': recommendation,
                    'urgency': max_urgency,
                    'patterns_detected': pattern_types,
                    'position_time': self._get_time_in_position(symbol),
                    'current_pnl': self._get_position_pnl(symbol)
                }
        
        return recommendations
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        if symbol in self.data_pipeline.symbol_to_idx:
            idx = self.data_pipeline.symbol_to_idx[symbol]
            return self.data_pipeline.realtime_features[idx, 0].item()
        return 0.0
    
    def _get_time_in_position(self, symbol: str) -> float:
        """Get time in position in minutes."""
        if symbol in self.position_entry_times:
            entry_time = self.position_entry_times[symbol]
            return (datetime.now(self.config.TIMEZONE) - entry_time).total_seconds() / 60
        return 0.0
    
    def _get_position_pnl(self, symbol: str) -> float:
        """Get position PnL percentage."""
        entry_price = self.position_entry_prices.get(symbol, 0)
        if entry_price > 0:
            current_price = self._get_current_price(symbol)
            return (current_price - entry_price) / entry_price * 100
        return 0.0
    
    def _persist_pattern_alert(self, alert: PatternAlert):
        """Persist high-priority pattern alert to database as system event."""
        try:
            # Prepare comprehensive metadata
            metadata = {
                'alert_details': {
                    'pattern_type': alert.pattern_type.value,
                    'confidence': float(alert.confidence),
                    'urgency': float(alert.urgency),
                    'message': alert.message,
                    'action_required': alert.action_required,
                },
                'position_details': alert.supporting_data,
                'system_context': {
                    'active_positions': len(self.active_positions),
                    'total_alerts_today': len([a for alerts in self.alerts_by_symbol.values() for a in alerts]),
                    'detection_latency_ms': 100.0,  # Placeholder - would measure actual latency
                },
                'pattern_engine_version': '1.0'
            }
            
            # Prepare database record as system event
            event_record = {
                'timestamp': alert.timestamp,
                'event_type': 'pattern_alert',
                'severity': 'high' if alert.urgency > 0.8 else 'medium',
                'description': f"{alert.pattern_type.value} detected in {alert.symbol}",
                'component': 'pattern_recognition_engine',
                'metadata': metadata
            }
            
            # Insert system event
            insert_query = """
                INSERT INTO system_events (
                    timestamp, event_type, severity, description, component, metadata
                ) VALUES (
                    %(timestamp)s, %(event_type)s, %(severity)s, %(description)s, %(component)s, %(metadata)s
                )
            """
            
            success = self.db_manager.execute_query(insert_query, event_record)
            if success:
                print(f"Persisted pattern alert: {alert.pattern_type.value} in {alert.symbol} (urgency: {alert.urgency:.2f})")
            else:
                print(f"Failed to persist pattern alert for {alert.symbol}")
                
        except Exception as e:
            print(f"Error persisting pattern alert to database: {e}")
            # Continue processing even if database persistence fails
    
    def visualize_patterns(self, symbol: str, save_path: Optional[str] = None):
        """Visualize detected patterns for a symbol."""
        if symbol not in self.data_pipeline.symbol_to_idx:
            print(f"Symbol {symbol} not found")
            return
        
        idx = self.data_pipeline.symbol_to_idx[symbol]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Price and patterns
        ax = axes[0]
        time_axis = np.arange(self.lookback_window)
        prices = self.price_buffer[idx].cpu().numpy()
        
        ax.plot(time_axis, prices, 'b-', linewidth=2, label='Price')
        
        # Mark detected patterns
        if symbol in self.alerts_by_symbol:
            for alert in self.alerts_by_symbol[symbol][-5:]:  # Last 5 alerts
                ax.axvline(x=self.lookback_window-1, color='r', alpha=0.3, 
                          linestyle='--', label=alert.pattern_type.value)
        
        ax.set_ylabel('Price')
        ax.set_title(f'Pattern Detection: {symbol}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Volume
        ax = axes[1]
        volumes = self.volume_buffer[idx].cpu().numpy()
        ax.bar(time_axis, volumes, alpha=0.6)
        ax.set_ylabel('Volume')
        ax.grid(True, alpha=0.3)
        
        # Pattern probabilities
        ax = axes[2]
        
        # Get latest pattern scores if available
        pattern_names = [p.value for p in PatternType]
        pattern_scores = np.random.rand(len(pattern_names))  # Placeholder
        
        ax.bar(range(len(pattern_names)), pattern_scores)
        ax.set_xticks(range(len(pattern_names)))
        ax.set_xticklabels(pattern_names, rotation=45, ha='right')
        ax.set_ylabel('Pattern Probability')
        ax.set_ylim(0, 1)
        ax.axhline(y=self.min_confidence, color='r', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


# Example usage when file is run directly
if __name__ == "__main__":
    print("GPU-Accelerated Pattern Recognition Engine Test")
    print("=" * 50)
    
    # Initialize components
    from gpu_core.tensor_universe_manager import TensorUniverseManager
    
    config = GPUTradingConfig()
    universe_manager = TensorUniverseManager(config)
    
    # Ensure universe is loaded
    if len(universe_manager.universe_symbols) == 0:
        print("Loading universe...")
        universe_manager.update_universe()
    
    # Create data pipeline
    data_pipeline = ParallelDataPipeline(config, universe_manager)
    
    # Create pattern recognition engine
    pattern_engine = GPUPatternRecognitionEngine(config, data_pipeline)
    
    # Add some test positions
    print("\nAdding test positions...")
    test_positions = [
        ("AAPL", 150.00),
        ("GOOGL", 140.00),
        ("MSFT", 380.00),
        ("AMZN", 170.00),
        ("TSLA", 250.00)
    ]
    
    for symbol, entry_price in test_positions:
        if symbol in universe_manager.universe_symbols:
            pattern_engine.add_position(symbol, entry_price)
    
    # Simulate market data
    print("\nSimulating market data...")
    
    # Create realistic patterns
    for _ in range(pattern_engine.lookback_window):
        # Update each position with synthetic data
        for symbol, entry_price in test_positions:
            if symbol in data_pipeline.symbol_to_idx:
                idx = data_pipeline.symbol_to_idx[symbol]
                
                # Simulate different patterns
                if symbol == "AAPL":
                    # Momentum exhaustion - slowing upward movement
                    price_change = np.random.randn() * 0.1 * (1 - _ / pattern_engine.lookback_window)
                elif symbol == "GOOGL":
                    # Range-bound movement
                    price_change = np.sin(_ / 5) * 0.5 + np.random.randn() * 0.1
                elif symbol == "MSFT":
                    # Strong trend
                    price_change = 0.2 + np.random.randn() * 0.1
                elif symbol == "AMZN":
                    # Reversal pattern
                    price_change = 0.3 - (_ / pattern_engine.lookback_window) * 0.6
                else:
                    # Normal movement
                    price_change = np.random.randn() * 0.2
                
                # Update features
                current_price = entry_price * (1 + price_change / 100)
                data_pipeline.realtime_features[idx, 0] = current_price
                data_pipeline.realtime_features[idx, 5] = 100000 + np.random.randint(0, 50000)
                data_pipeline.realtime_features[idx, 9] = price_change
                data_pipeline.realtime_features[idx, 18] = abs(price_change) * 10
        
        # Update buffers
        pattern_engine._update_feature_buffers()
    
    # Scan for patterns
    print("\nScanning for patterns...")
    alerts = pattern_engine.scan_patterns()
    
    # Display alerts
    if alerts:
        print(f"\nFound {len(alerts)} pattern alerts:")
        print("-" * 80)
        
        for alert in sorted(alerts, key=lambda x: x.urgency, reverse=True):
            print(f"\n{alert.timestamp.strftime('%H:%M:%S')} - {alert.symbol}")
            print(f"  Pattern: {alert.pattern_type.value}")
            print(f"  Confidence: {alert.confidence:.2%}")
            print(f"  Urgency: {alert.urgency:.2f}")
            print(f"  Message: {alert.message}")
            print(f"  Action: {alert.action_required}")
            
            if 'position_pnl' in alert.supporting_data:
                print(f"  PnL: {alert.supporting_data['position_pnl']:+.2f}%")
    else:
        print("\nNo significant patterns detected")
    
    # Get exit recommendations
    print("\nExit Recommendations:")
    print("-" * 60)
    
    recommendations = pattern_engine.get_exit_recommendations()
    
    for symbol, rec in sorted(recommendations.items(), 
                             key=lambda x: x[1]['urgency'], reverse=True):
        print(f"\n{symbol}:")
        print(f"  Recommendation: {rec['recommendation']}")
        print(f"  Urgency: {rec['urgency']:.2f}")
        print(f"  Time in Position: {rec['position_time']:.0f} minutes")
        print(f"  Current PnL: {rec['current_pnl']:+.2f}%")
        print(f"  Patterns: {', '.join(rec['patterns_detected'])}")
    
    # Performance summary
    print(f"\nPattern Recognition Summary:")
    print(f"  Active Positions: {len(pattern_engine.active_positions)}")
    print(f"  Total Alerts Generated: {len(pattern_engine.pattern_history)}")
    print(f"  Average Alert Confidence: {np.mean([a.confidence for a in pattern_engine.pattern_history]) if pattern_engine.pattern_history else 0:.2%}")
    
    # Pattern distribution
    if pattern_engine.pattern_history:
        pattern_counts = {}
        for alert in pattern_engine.pattern_history:
            pattern_counts[alert.pattern_type.value] = pattern_counts.get(alert.pattern_type.value, 0) + 1
        
        print(f"\nPattern Distribution:")
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pattern}: {count}")
