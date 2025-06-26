#!/usr/bin/env python3
"""
GPU-Accelerated Attention-Based Regime Classifier
Uses Transformer architecture to classify market regimes and identify
which market features are driving current conditions.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpu_trading_config import GPUTradingConfig
from database.connection import get_db_manager
from gpu_core.tensor_universe_manager import TensorUniverseManager
from gpu_core.parallel_data_pipeline import ParallelDataPipeline


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module for market feature relationships."""
    
    def __init__(self, d_model: int, n_heads: int, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.device = device
        self.dtype = dtype
        
        # Linear transformations for Q, K, V
        self.W_q = nn.Linear(d_model, d_model).to(device).to(dtype)
        self.W_k = nn.Linear(d_model, d_model).to(device).to(dtype)
        self.W_v = nn.Linear(d_model, d_model).to(device).to(dtype)
        self.W_o = nn.Linear(d_model, d_model).to(device).to(dtype)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model).to(device).to(dtype)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
        Returns:
            output: Attended features
            attention_weights: Attention weights for interpretability
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_head, dtype=self.dtype))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(x + self.dropout(output))
        
        # Average attention weights across heads for interpretability
        avg_attention = attention_weights.mean(dim=1)
        
        return output, avg_attention


class TransformerRegimeClassifier(nn.Module):
    """Transformer-based model for market regime classification."""
    
    def __init__(self, input_dim: int, d_model: int, n_heads: int, n_layers: int,
                 n_regimes: int, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_regimes = n_regimes
        self.device = device
        self.dtype = dtype
        
        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model).to(device).to(dtype)
        
        # Positional encoding for temporal features
        self.positional_encoding = self._create_positional_encoding(1000, d_model)
        
        # Transformer layers
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(d_model, n_heads, device, dtype)
            for _ in range(n_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 4, d_model),
                nn.LayerNorm(d_model)
            ).to(device).to(dtype)
            for _ in range(n_layers)
        ])
        
        # Global context attention (to aggregate across all symbols)
        self.global_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True).to(device).to(dtype)
        
        # Regime classification head
        self.regime_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, n_regimes)
        ).to(device).to(dtype)
        
        # Feature importance head
        self.feature_importance = nn.Linear(d_model, input_dim).to(device).to(dtype)
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model, device=self.device, dtype=self.dtype)
        position = torch.arange(0, max_len, device=self.device, dtype=self.dtype).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2, device=self.device, dtype=self.dtype) *
                           -(torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x: torch.Tensor, temporal_position: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through regime classifier.
        Args:
            x: Market features [batch_size, n_symbols, n_features]
            temporal_position: Optional temporal positions for each sample
        Returns:
            Dictionary with regime probabilities and attention maps
        """
        batch_size, n_symbols, _ = x.shape
        
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding if temporal positions provided
        if temporal_position is not None:
            x = x + self.positional_encoding[:, :n_symbols, :]
        
        # Store attention maps for interpretability
        attention_maps = []
        
        # Pass through transformer layers
        for i, (attn_layer, ffn_layer) in enumerate(zip(self.attention_layers, self.ffn_layers)):
            # Self-attention
            x, attn_weights = attn_layer(x)
            attention_maps.append(attn_weights)
            
            # Feed-forward
            x = x + ffn_layer(x)
        
        # Global attention to create market-wide representation
        # Use mean of all symbols as query
        global_query = x.mean(dim=1, keepdim=True)
        global_context, global_attention = self.global_attention(global_query, x, x)
        
        # Squeeze to get single vector per batch
        global_representation = global_context.squeeze(1)
        
        # Classify regime
        regime_logits = self.regime_classifier(global_representation)
        regime_probs = F.softmax(regime_logits, dim=-1)
        
        # Feature importance based on global representation
        feature_importance = torch.sigmoid(self.feature_importance(global_representation))
        
        return {
            'regime_probabilities': regime_probs,
            'regime_logits': regime_logits,
            'attention_maps': attention_maps,
            'global_attention': global_attention.squeeze(1),
            'feature_importance': feature_importance,
            'global_representation': global_representation
        }


class AttentionRegimeClassifier:
    """Main regime classifier for gap trading system."""
    
    def __init__(self, config: GPUTradingConfig, data_pipeline: ParallelDataPipeline):
        self.config = config
        self.data_pipeline = data_pipeline
        self.device = config.device
        self.dtype = config.TENSOR_DTYPE
        
        # Database connection for regime change logging
        self.db_manager = get_db_manager()
        
        # Market features to use for regime classification
        self.market_features = [
            'vix_level',          # Market volatility
            'spy_momentum',       # Market direction
            'sector_dispersion',  # Sector divergence
            'volume_ratio',       # Volume vs average
            'breadth',           # Advance/decline ratio
            'high_low_ratio',    # New highs vs lows
            'put_call_ratio',    # Options sentiment
            'dollar_strength',   # DXY level
            'bond_yield',        # 10Y yield
            'yield_curve',       # 10Y-2Y spread
            'gap_frequency',     # Overall gap activity
            'correlation_mean'   # Average stock correlation
        ]
        self.n_features = len(self.market_features)
        
        # Initialize transformer model
        self.model = TransformerRegimeClassifier(
            input_dim=self.n_features,
            d_model=128,
            n_heads=8,
            n_layers=4,
            n_regimes=len(config.REGIME_STATES),  # TRENDING, VOLATILE, QUIET
            device=self.device,
            dtype=self.dtype
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=0.01)
        
        # Current regime state
        self.current_regime = None
        self.regime_history = []
        self.regime_transition_matrix = torch.zeros(3, 3, device=self.device)  # Track transitions
        
        # Model path
        self.model_path = config.MODELS_DIR / 'regime_classifier.pth'
        self._load_model()
        
    def classify_regime(self, current_time: Optional[datetime] = None) -> Dict[str, any]:
        """
        Classify current market regime using all available data.
        Returns regime probabilities and driving factors.
        """
        if current_time is None:
            current_time = datetime.now(self.config.TIMEZONE)
        
        # Calculate market features
        market_features = self._calculate_market_features()
        
        # Prepare input tensor (using universe symbols as sequence)
        # Shape: [1, n_symbols, n_features]
        input_features = self._prepare_input_features(market_features)
        
        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_features)
        
        # Extract regime probabilities
        regime_probs = output['regime_probabilities'][0]  # Remove batch dimension
        predicted_regime = torch.argmax(regime_probs).item()
        
        # Map to regime names
        regime_names = self.config.REGIME_STATES
        regime_name = regime_names[predicted_regime]
        
        # Update regime history
        regime_changed = False
        if self.current_regime is not None:
            self.regime_transition_matrix[self.current_regime, predicted_regime] += 1
            if self.current_regime != predicted_regime:
                regime_changed = True
        else:
            regime_changed = True  # First regime detection
        
        self.current_regime = predicted_regime
        regime_entry = {
            'timestamp': current_time,
            'regime': regime_name,
            'probabilities': regime_probs.cpu().numpy(),
            'confidence': regime_probs.max().item()
        }
        self.regime_history.append(regime_entry)
        
        # Identify driving factors
        feature_importance = output['feature_importance'][0]
        top_features_idx = torch.argsort(feature_importance, descending=True)[:5]
        driving_factors = [(self.market_features[idx], feature_importance[idx].item()) 
                          for idx in top_features_idx]
        
        # Persist regime change to database if significant change occurred
        if regime_changed:
            self._persist_regime_change(regime_entry, driving_factors)
        
        # Get symbol-level attention to identify market leaders/laggards
        global_attention = output['global_attention'][0]  # [n_symbols]
        top_symbols_idx = torch.argsort(global_attention, descending=True)[:10]
        influential_symbols = [(self.data_pipeline.universe_symbols[idx], global_attention[idx].item())
                              for idx in top_symbols_idx if idx < len(self.data_pipeline.universe_symbols)]
        
        return {
            'regime': regime_name,
            'regime_index': predicted_regime,
            'probabilities': {name: prob.item() for name, prob in zip(regime_names, regime_probs)},
            'confidence': regime_probs.max().item(),
            'driving_factors': driving_factors,
            'influential_symbols': influential_symbols,
            'feature_importance': feature_importance.cpu().numpy(),
            'transition_matrix': self.regime_transition_matrix.cpu().numpy()
        }
    
    def _calculate_market_features(self) -> torch.Tensor:
        """Calculate aggregate market features from universe data."""
        # Get current features from pipeline
        current_features, _ = self.data_pipeline.get_current_features()
        
        # Calculate market-wide statistics
        market_features = torch.zeros(self.n_features, device=self.device, dtype=self.dtype)
        
        # VIX proxy (average volatility)
        volatilities = current_features[:, 18]  # price_volatility_5min
        market_features[0] = volatilities.mean()
        
        # Market momentum (average price change)
        momentum = self.data_pipeline.realtime_features[:, 9]  # price_momentum_5min
        market_features[1] = momentum.mean()
        
        # Sector dispersion (std of sector performances)
        # Group by sector and calculate dispersion
        sector_returns = []
        for sector in range(11):  # 11 sectors
            mask = self.data_pipeline.universe_manager.sector_assignments == sector
            if mask.sum() > 0:
                sector_returns.append(momentum[mask].mean())
        
        if sector_returns:
            market_features[2] = torch.tensor(sector_returns, device=self.device).std()
        
        # Volume ratio (current vs average)
        current_volumes = self.data_pipeline.realtime_features[:, 5]  # volume_5min
        avg_volumes = self.data_pipeline.universe_tensor[:, 1]  # average volume
        volume_ratios = current_volumes / (avg_volumes * 0.05 + 1)  # 5min vs 5% of daily
        market_features[3] = volume_ratios.mean()
        
        # Market breadth (% of positive movers)
        market_features[4] = (momentum > 0).float().mean()
        
        # High/low ratio (simplified - % near 5min high)
        high_5min = self.data_pipeline.realtime_features[:, 11]
        current_prices = self.data_pipeline.realtime_features[:, 0]
        near_high = (current_prices > high_5min * 0.99).float().mean()
        market_features[5] = near_high
        
        # Put/call ratio proxy (using volume skew as proxy)
        volume_skew = self.data_pipeline.realtime_features[:, 14]  # volume_profile_skew
        market_features[6] = volume_skew.mean()
        
        # Dollar strength (would need DXY data - using 0.5 as neutral)
        market_features[7] = 0.5
        
        # Bond yield proxy (using low volatility stocks as proxy)
        low_vol_mask = volatilities < volatilities.quantile(0.2)
        market_features[8] = momentum[low_vol_mask].mean() if low_vol_mask.sum() > 0 else 0
        
        # Yield curve (using momentum difference between sectors)
        market_features[9] = 0.0  # Simplified
        
        # Gap frequency (% of stocks with significant gaps)
        gap_sizes = torch.abs(self.data_pipeline.gap_analyzer.gap_features[:, 0]) if hasattr(self.data_pipeline, 'gap_analyzer') else torch.zeros_like(momentum)
        market_features[10] = (gap_sizes > 2.0).float().mean()
        
        # Average correlation (simplified - using momentum correlation)
        if len(momentum) > 1:
            momentum_centered = momentum - momentum.mean()
            correlation_sum = torch.matmul(momentum_centered, momentum_centered) / (momentum.std() ** 2 * len(momentum))
            market_features[11] = correlation_sum / len(momentum)
        
        return market_features
    
    def _persist_regime_change(self, regime_entry: Dict, driving_factors: List[Tuple[str, float]]):
        """Persist regime change to database as system event."""
        try:
            # Build comprehensive metadata
            metadata = {
                'regime_details': {
                    'regime': regime_entry['regime'],
                    'confidence': float(regime_entry['confidence']),
                    'probabilities': {name: float(prob) for name, prob in 
                                    zip(self.config.REGIME_STATES, regime_entry['probabilities'])},
                },
                'driving_factors': [
                    {'factor': factor, 'importance': float(importance)} 
                    for factor, importance in driving_factors
                ],
                'transition_history': {
                    'total_regime_changes': len(self.regime_history),
                    'transition_matrix': self.regime_transition_matrix.cpu().numpy().tolist(),
                },
                'market_context': {
                    'n_symbols_analyzed': len(self.data_pipeline.universe_symbols),
                    'model_confidence_threshold': 0.6,
                    'classification_latency_ms': 50.0,  # Placeholder - would measure actual
                },
                'classifier_version': '1.0'
            }
            
            # Determine severity based on confidence and regime type
            if regime_entry['confidence'] > 0.8:
                severity = 'high' if regime_entry['regime'] in ['VOLATILE', 'TRENDING'] else 'medium'
            else:
                severity = 'medium'
            
            # Prepare database record as system event
            event_record = {
                'timestamp': regime_entry['timestamp'],
                'event_type': 'regime_change',
                'severity': severity,
                'description': f"Market regime changed to {regime_entry['regime']} (confidence: {regime_entry['confidence']:.2f})",
                'component': 'attention_regime_classifier',
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
                print(f"Persisted regime change: {regime_entry['regime']} (confidence: {regime_entry['confidence']:.2f})")
            else:
                print(f"Failed to persist regime change to database")
                
        except Exception as e:
            print(f"Error persisting regime change to database: {e}")
            # Continue processing even if database persistence fails
    
    def _prepare_input_features(self, market_features: torch.Tensor) -> torch.Tensor:
        """Prepare input features for transformer model."""
        # Create symbol-level features that incorporate market features
        n_symbols = len(self.data_pipeline.universe_symbols)
        
        # Repeat market features for each symbol and add symbol-specific modulation
        symbol_features = market_features.unsqueeze(0).repeat(n_symbols, 1)
        
        # Add symbol-specific variations based on their characteristics
        if hasattr(self.data_pipeline, 'realtime_features'):
            # Modulate by symbol's momentum relative to market
            symbol_momentum = self.data_pipeline.realtime_features[:, 9]
            market_momentum = symbol_momentum.mean()
            momentum_diff = (symbol_momentum - market_momentum).unsqueeze(1)
            
            # Modulate features based on relative momentum
            symbol_features = symbol_features * (1 + 0.1 * torch.tanh(momentum_diff))
        
        # Add batch dimension
        return symbol_features.unsqueeze(0)
    
    def get_regime_adjusted_parameters(self) -> Dict[str, float]:
        """Get trading parameters adjusted for current regime."""
        if self.current_regime is None:
            return {}
        
        regime_name = self.config.REGIME_STATES[self.current_regime]
        
        adjustments = {}
        
        if regime_name == 'TRENDING':
            # In trending markets, gaps in trend direction are more reliable
            adjustments['confidence_threshold'] = self.config.CONFIDENCE_THRESHOLD * 0.9  # Lower threshold
            adjustments['position_size_multiplier'] = 1.2  # Larger positions
            adjustments['stop_loss_multiplier'] = 1.1  # Wider stops
            adjustments['gap_direction_bias'] = 0.7  # Prefer gaps in trend direction
            
        elif regime_name == 'VOLATILE':
            # In volatile markets, be more selective and use tighter risk management
            adjustments['confidence_threshold'] = self.config.CONFIDENCE_THRESHOLD * 1.1  # Higher threshold
            adjustments['position_size_multiplier'] = 0.8  # Smaller positions
            adjustments['stop_loss_multiplier'] = 0.9  # Tighter stops
            adjustments['gap_direction_bias'] = 0.5  # No directional bias
            
        elif regime_name == 'QUIET':
            # In quiet markets, gaps often start new moves
            adjustments['confidence_threshold'] = self.config.CONFIDENCE_THRESHOLD
            adjustments['position_size_multiplier'] = 1.0
            adjustments['stop_loss_multiplier'] = 1.0
            adjustments['gap_direction_bias'] = 0.5
            adjustments['gap_size_preference'] = 1.2  # Prefer larger gaps
        
        return adjustments
    
    def visualize_regime_analysis(self, save_path: Optional[Path] = None):
        """Visualize current regime analysis."""
        if not self.regime_history:
            print("No regime history to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Regime probability history
        ax = axes[0, 0]
        history_df = pd.DataFrame([
            {
                'timestamp': h['timestamp'],
                **{f'{regime}_prob': prob for regime, prob in h['probabilities'].items()}
            }
            for h in self.regime_history[-100:]  # Last 100 observations
        ])
        
        if len(history_df) > 0:
            for regime in self.config.REGIME_STATES:
                ax.plot(history_df['timestamp'], history_df[f'{regime}_prob'], label=regime)
            ax.set_ylabel('Probability')
            ax.set_title('Regime Probabilities Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Transition matrix heatmap
        ax = axes[0, 1]
        transition_probs = self.regime_transition_matrix.cpu().numpy()
        row_sums = transition_probs.sum(axis=1, keepdims=True)
        transition_probs = np.divide(transition_probs, row_sums, where=row_sums != 0)
        
        sns.heatmap(transition_probs, 
                    xticklabels=self.config.REGIME_STATES,
                    yticklabels=self.config.REGIME_STATES,
                    annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
        ax.set_title('Regime Transition Probabilities')
        ax.set_xlabel('To Regime')
        ax.set_ylabel('From Regime')
        
        # Feature importance
        ax = axes[1, 0]
        if self.regime_history:
            latest = self.regime_history[-1]
            feature_importance = latest.get('feature_importance', np.zeros(self.n_features))
            
            y_pos = np.arange(len(self.market_features))
            ax.barh(y_pos, feature_importance)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(self.market_features)
            ax.set_xlabel('Importance')
            ax.set_title('Current Market Feature Importance')
            ax.grid(True, alpha=0.3, axis='x')
        
        # Regime distribution pie chart
        ax = axes[1, 1]
        regime_counts = {}
        for h in self.regime_history:
            regime = h['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        if regime_counts:
            ax.pie(regime_counts.values(), labels=regime_counts.keys(), autopct='%1.1f%%')
            ax.set_title('Historical Regime Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def _save_model(self):
        """Save model state."""
        state = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'current_regime': self.current_regime,
            'regime_history': self.regime_history[-1000:],  # Keep last 1000
            'transition_matrix': self.regime_transition_matrix,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(state, self.model_path)
        print(f"Regime classifier saved to {self.model_path}")
    
    def _load_model(self):
        """Load model state."""
        if self.model_path.exists():
            try:
                state = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state['model_state'])
                self.optimizer.load_state_dict(state['optimizer_state'])
                self.current_regime = state.get('current_regime')
                self.regime_history = state.get('regime_history', [])
                self.regime_transition_matrix = state.get('transition_matrix', self.regime_transition_matrix)
                print(f"Regime classifier loaded from {self.model_path}")
            except Exception as e:
                print(f"Error loading regime classifier: {e}")


# Example usage when file is run directly
if __name__ == "__main__":
    print("GPU-Accelerated Attention Regime Classifier Test")
    print("=" * 50)
    
    # Initialize components
    config = GPUTradingConfig()
    universe_manager = TensorUniverseManager(config)
    
    # Ensure universe is loaded
    if len(universe_manager.universe_symbols) == 0:
        print("Loading universe...")
        universe_manager.update_universe()
    
    # Create pipeline
    data_pipeline = ParallelDataPipeline(config, universe_manager)
    
    # Simulate some market data
    print("\nSimulating market conditions...")
    n_symbols = len(universe_manager.universe_symbols)
    
    # Create different market scenarios
    scenarios = {
        'TRENDING': {
            'momentum': 2.5,
            'volatility': 15.0,
            'breadth': 0.7,
            'correlation': 0.6
        },
        'VOLATILE': {
            'momentum': 0.0,
            'volatility': 35.0,
            'breadth': 0.5,
            'correlation': 0.8
        },
        'QUIET': {
            'momentum': 0.5,
            'volatility': 10.0,
            'breadth': 0.5,
            'correlation': 0.3
        }
    }
    
    # Test each scenario
    for scenario_name, params in scenarios.items():
        print(f"\n{scenario_name} Market Scenario:")
        
        # Set up market conditions
        with torch.no_grad():
            # Price momentum
            data_pipeline.realtime_features[:, 9] = torch.randn(n_symbols, device=config.device) * params['volatility'] / 10 + params['momentum']
            
            # Volatility
            data_pipeline.realtime_features[:, 18] = torch.ones(n_symbols, device=config.device) * params['volatility']
            
            # Volume (higher in volatile markets)
            data_pipeline.realtime_features[:, 5] = torch.ones(n_symbols, device=config.device) * (1 + params['volatility'] / 20) * 100000
            
            # Correlation effect (higher correlation = more similar movements)
            base_movement = torch.randn(1, device=config.device) * params['correlation']
            data_pipeline.realtime_features[:, 9] = (
                data_pipeline.realtime_features[:, 9] * (1 - params['correlation']) +
                base_movement * params['correlation']
            )
    
    # Create regime classifier
    regime_classifier = AttentionRegimeClassifier(config, data_pipeline)
    
    # Classify regime
    print("\nClassifying market regime...")
    regime_analysis = regime_classifier.classify_regime()
    
    print(f"\nRegime Analysis:")
    print(f"  Current Regime: {regime_analysis['regime']}")
    print(f"  Confidence: {regime_analysis['confidence']:.2%}")
    
    print(f"\n  Regime Probabilities:")
    for regime, prob in regime_analysis['probabilities'].items():
        print(f"    {regime}: {prob:.2%}")
    
    print(f"\n  Top Driving Factors:")
    for factor, importance in regime_analysis['driving_factors']:
        print(f"    {factor}: {importance:.3f}")
    
    print(f"\n  Most Influential Symbols:")
    for i, (symbol, attention) in enumerate(regime_analysis['influential_symbols'][:5]):
        print(f"    {i+1}. {symbol}: {attention:.3f}")
    
    # Get adjusted parameters
    adjustments = regime_classifier.get_regime_adjusted_parameters()
    
    print(f"\n  Trading Adjustments for {regime_analysis['regime']} Regime:")
    for param, value in adjustments.items():
        print(f"    {param}: {value}")
    
    # Test regime transitions
    print(f"\nSimulating regime transitions...")
    
    # Simulate multiple time steps
    for i in range(10):
        # Gradually change market conditions
        volatility_change = np.random.randn() * 2
        data_pipeline.realtime_features[:, 18] += volatility_change
        
        momentum_change = np.random.randn() * 0.5
        data_pipeline.realtime_features[:, 9] += momentum_change
        
        # Classify again
        regime_analysis = regime_classifier.classify_regime()
        
        print(f"  Step {i+1}: {regime_analysis['regime']} (confidence: {regime_analysis['confidence']:.2%})")
    
    # Show transition matrix
    print(f"\nRegime Transition Matrix:")
    transition_matrix = regime_classifier.regime_transition_matrix.cpu().numpy()
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_probs = np.divide(transition_matrix, row_sums, where=row_sums != 0)
    
    regimes = config.REGIME_STATES
    print(f"{'From/To':<12}", end='')
    for regime in regimes:
        print(f"{regime:<12}", end='')
    print()
    
    for i, from_regime in enumerate(regimes):
        print(f"{from_regime:<12}", end='')
        for j in range(len(regimes)):
            print(f"{transition_probs[i, j]:<12.2f}", end='')
        print()
    
    # Save model
    regime_classifier._save_model()
    print(f"\nRegime classifier saved successfully!")
