#!/usr/bin/env python3
"""
GPU-Accelerated Neural Magnitude Predictor
Implements Mixture Density Network (MDN) for predicting gap continuation distributions.
Outputs probability distributions rather than point estimates for sophisticated position sizing.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpu_trading_config import GPUTradingConfig
from gpu_features.vectorized_gap_analyzer import VectorizedGapAnalyzer
from gpu_models.ensemble_continuation_filter import EnsembleContinuationFilter


class MixtureDensityNetwork(nn.Module):
    """
    Mixture Density Network for probabilistic gap magnitude prediction.
    Outputs parameters for a mixture of Gaussian distributions.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, n_gaussians: int, 
                 device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_gaussians = n_gaussians
        self.device = device
        self.dtype = dtype
        
        # Feature extraction layers with batch normalization
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        ).to(device).to(dtype)
        
        # Output heads for mixture parameters
        self.pi_head = nn.Linear(hidden_dim // 2, n_gaussians).to(device).to(dtype)  # mixing coefficients
        self.mu_head = nn.Linear(hidden_dim // 2, n_gaussians).to(device).to(dtype)  # means
        self.sigma_head = nn.Linear(hidden_dim // 2, n_gaussians).to(device).to(dtype)  # standard deviations
        
        # Initialize output heads carefully
        self._initialize_mixture_heads()
        
        # Attention mechanism for interpretability
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, input_dim),
            nn.Softmax(dim=1)
        ).to(device).to(dtype)
        
    def _initialize_mixture_heads(self):
        """Initialize mixture heads for stable training."""
        # Pi head: equal mixing initially
        nn.init.constant_(self.pi_head.weight, 0.0)
        nn.init.constant_(self.pi_head.bias, 0.0)
        
        # Mu head: spread across reasonable range (0-5% continuation)
        nn.init.normal_(self.mu_head.weight, 0.0, 0.1)
        with torch.no_grad():
            self.mu_head.bias.data = torch.linspace(0.5, 4.0, self.n_gaussians, 
                                                    device=self.device, dtype=self.dtype)
        
        # Sigma head: reasonable initial uncertainty
        nn.init.constant_(self.sigma_head.weight, 0.0)
        nn.init.constant_(self.sigma_head.bias, 0.5)  # 0.5% std dev initially
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MDN.
        Args:
            x: Input features [batch_size, input_dim]
        Returns:
            Dictionary with mixture parameters and attention weights
        """
        # Apply attention to highlight important features
        attention_weights = self.attention(x)
        x_attended = x * attention_weights
        
        # Extract features
        features = self.feature_net(x_attended)
        
        # Compute mixture parameters
        pi_logits = self.pi_head(features)
        pi = F.softmax(pi_logits, dim=1)  # [batch_size, n_gaussians]
        
        mu = self.mu_head(features)  # [batch_size, n_gaussians]
        
        # Ensure positive sigma with minimum value
        sigma_raw = self.sigma_head(features)
        sigma = F.softplus(sigma_raw) + 0.1  # minimum 0.1% std dev
        
        return {
            'pi': pi,
            'mu': mu,
            'sigma': sigma,
            'attention_weights': attention_weights,
            'features': features
        }
    
    def sample(self, x: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        """Sample from the predicted mixture distribution."""
        with torch.no_grad():
            output = self.forward(x)
            batch_size = x.shape[0]
            
            samples = torch.zeros(batch_size, n_samples, device=self.device, dtype=self.dtype)
            
            for i in range(batch_size):
                # Sample which Gaussian to use for each sample
                component_dist = Categorical(output['pi'][i])
                components = component_dist.sample((n_samples,))
                
                # Sample from the selected Gaussians
                for j in range(n_samples):
                    component = components[j]
                    mean = output['mu'][i, component]
                    std = output['sigma'][i, component]
                    samples[i, j] = Normal(mean, std).sample()
            
            return samples
    
    def get_percentiles(self, x: torch.Tensor, percentiles: List[float] = [10, 25, 50, 75, 90]) -> Dict[str, torch.Tensor]:
        """Get percentile estimates from the mixture distribution."""
        samples = self.sample(x, n_samples=1000)
        
        results = {}
        for p in percentiles:
            results[f'p{p}'] = torch.quantile(samples, p/100, dim=1)
        
        # Also compute mean and std
        results['mean'] = samples.mean(dim=1)
        results['std'] = samples.std(dim=1)
        
        return results
    
    def negative_log_likelihood(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute negative log likelihood loss for training."""
        output = self.forward(x)
        
        # Compute likelihood of y under each Gaussian
        likelihoods = torch.zeros(x.shape[0], self.n_gaussians, device=self.device, dtype=self.dtype)
        
        for k in range(self.n_gaussians):
            dist = Normal(output['mu'][:, k], output['sigma'][:, k])
            likelihoods[:, k] = dist.log_prob(y).exp()
        
        # Weight by mixing coefficients
        weighted_likelihoods = output['pi'] * likelihoods
        
        # Sum across components and take log
        log_likelihood = torch.log(weighted_likelihoods.sum(dim=1) + 1e-8)
        
        return -log_likelihood.mean()


class NeuralMagnitudePredictor:
    """Main magnitude predictor using MDN for gap continuation."""
    
    def __init__(self, config: GPUTradingConfig, gap_analyzer: VectorizedGapAnalyzer):
        self.config = config
        self.gap_analyzer = gap_analyzer
        self.device = config.device
        self.dtype = config.TENSOR_DTYPE
        
        # Select features for magnitude prediction
        # More features than continuation filter since we're predicting a complex distribution
        self.feature_indices = [
            0,   # gap_size
            1,   # gap_size_atr
            3,   # gap_quality_score
            5,   # institutional_footprint
            6,   # block_trade_ratio
            8,   # smart_money_flow
            10,  # relative_gap_strength
            11,  # sector_relative_strength
            13,  # peer_correlation
            15,  # gap_formation_minutes
            16,  # gap_stability
            17,  # gap_acceleration
            20,  # news_catalyst_score
            21,  # volume_surge_score
            23,  # momentum_alignment
            24   # continuation_probability
        ]
        self.n_features = len(self.feature_indices)
        
        # Initialize MDN
        self.model = MixtureDensityNetwork(
            input_dim=self.n_features,
            hidden_dim=128,
            n_gaussians=5,  # 5 components for flexible distributions
            device=self.device,
            dtype=self.dtype
        )
        
        # Optimizer with weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2
        )
        
        # Training history
        self.training_history = {
            'loss': [],
            'val_loss': [],
            'epoch': 0
        }
        
        # Model path
        self.model_path = config.MODELS_DIR / 'magnitude_mdn.pth'
        self._load_model()
        
    def predict_magnitude_distribution(self, gap_features: torch.Tensor, 
                                     mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Predict magnitude distribution for gaps.
        Args:
            gap_features: Full gap features from analyzer
            mask: Boolean mask of which gaps to predict (from continuation filter)
        Returns:
            Dictionary with distribution parameters and percentile estimates
        """
        # Extract relevant features
        selected_features = gap_features[:, self.feature_indices]
        
        # Apply mask if provided
        if mask is not None:
            selected_features = selected_features[mask]
        
        # Normalize features
        normalized_features = self._normalize_features(selected_features)
        
        # Get MDN predictions
        self.model.eval()
        with torch.no_grad():
            mdn_output = self.model(normalized_features)
            percentiles = self.model.get_percentiles(normalized_features)
        
        # Create comprehensive output
        predictions = {
            'mixture_weights': mdn_output['pi'],
            'component_means': mdn_output['mu'],
            'component_stds': mdn_output['sigma'],
            'attention_weights': mdn_output['attention_weights'],
            'expected_magnitude': percentiles['mean'],
            'magnitude_std': percentiles['std'],
            'p10': percentiles['p10'],  # 10th percentile (downside)
            'p25': percentiles['p25'],  # 25th percentile
            'p50': percentiles['p50'],  # median
            'p75': percentiles['p75'],  # 75th percentile
            'p90': percentiles['p90'],  # 90th percentile (upside)
            'features_used': normalized_features
        }
        
        # Add risk-adjusted metrics
        predictions['reward_risk_ratio'] = (predictions['p75'] - predictions['p50']) / (predictions['p50'] - predictions['p25'] + 0.1)
        predictions['upside_potential'] = predictions['p90'] - predictions['p50']
        predictions['downside_risk'] = predictions['p50'] - predictions['p10']
        predictions['skewness'] = (predictions['p90'] - 2*predictions['p50'] + predictions['p10']) / (predictions['magnitude_std'] + 0.1)
        
        return predictions
    
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features for neural network."""
        # Use layer normalization for stability
        normalized = F.layer_norm(features, features.shape[1:])
        
        # Apply feature-specific transformations
        with torch.no_grad():
            # Gap size features (indices 0, 1) - cap at reasonable values
            if features.shape[1] > 1:
                normalized[:, 0] = torch.clamp(normalized[:, 0], -3, 3)
                normalized[:, 1] = torch.clamp(normalized[:, 1], -3, 3)
        
        return normalized
    
    def train_on_historical_data(self, historical_gaps: pd.DataFrame, epochs: int = 100):
        """Train MDN on historical gap outcomes."""
        print(f"Training magnitude predictor on {len(historical_gaps)} historical gaps...")
        
        # Prepare training data
        X, y = self._prepare_training_data(historical_gaps)
        
        # Split into train/validation
        n_samples = X.shape[0]
        n_train = int(0.8 * n_samples)
        
        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # Training loop
        self.model.train()
        
        for epoch in range(epochs):
            # Training step
            self.optimizer.zero_grad()
            loss = self.model.negative_log_likelihood(X_train, y_train)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Validation step
            if epoch % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_loss = self.model.negative_log_likelihood(X_val, y_val)
                self.model.train()
                
                self.training_history['loss'].append(loss.item())
                self.training_history['val_loss'].append(val_loss.item())
                
                print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")
        
        self.training_history['epoch'] = epoch + 1
        self._save_model()
        
    def _prepare_training_data(self, historical_gaps: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare historical data for training."""
        # This is a placeholder - in production, you would load actual historical gap data
        # with features and outcomes
        
        n_samples = 1000
        n_features = self.n_features
        
        # Simulate realistic training data
        X = torch.randn(n_samples, n_features, device=self.device, dtype=self.dtype)
        
        # Simulate outcomes based on features (gap continuation percentages)
        # In reality, this would be actual measured continuations
        true_weights = torch.randn(n_features, device=self.device) * 0.1
        base_continuation = torch.matmul(X, true_weights) + 2.0  # Base around 2%
        
        # Add noise and ensure positive
        noise = torch.randn(n_samples, device=self.device) * 0.5
        y = torch.abs(base_continuation + noise)
        
        # Cap at reasonable values
        y = torch.clamp(y, 0.1, 10.0)  # 0.1% to 10% continuation
        
        return X, y
    
    def get_position_size_recommendation(self, predictions: Dict[str, torch.Tensor], 
                                       account_value: float) -> torch.Tensor:
        """Calculate recommended position sizes based on magnitude predictions."""
        # Kelly-inspired sizing with safety constraints
        expected_return = predictions['expected_magnitude'] / 100  # Convert to decimal
        downside_risk = predictions['downside_risk'] / 100
        
        # Avoid division by zero
        downside_risk = torch.maximum(downside_risk, torch.tensor(0.001, device=self.device))
        
        # Kelly fraction (simplified)
        kelly_fraction = expected_return / (downside_risk ** 2)
        
        # Apply safety scaling
        # - Never more than 25% Kelly
        # - Scale by confidence (reward/risk ratio)
        # - Account for skewness
        safety_factor = 0.25
        confidence_scale = torch.sigmoid(predictions['reward_risk_ratio'] - 1)
        skewness_adjustment = 1 + torch.tanh(predictions['skewness']) * 0.2
        
        adjusted_fraction = kelly_fraction * safety_factor * confidence_scale * skewness_adjustment
        
        # Convert to position sizes
        position_sizes = adjusted_fraction * account_value
        
        # Apply constraints
        min_size = self.config.MIN_POSITION_SIZE
        max_size = account_value * self.config.MAX_POSITION_SIZE_PERCENT
        
        position_sizes = torch.clamp(position_sizes, min_size, max_size)
        
        return position_sizes
    
    def visualize_predictions(self, predictions: Dict[str, torch.Tensor], 
                            symbol: str, save_path: Optional[Path] = None):
        """Visualize the predicted magnitude distribution."""
        plt.figure(figsize=(12, 8))
        
        # Get single prediction for visualization
        idx = 0  # First prediction
        
        # Sample from distribution
        samples = self.model.sample(predictions['features_used'][idx:idx+1], n_samples=1000).squeeze()
        
        # Plot histogram of samples
        plt.subplot(2, 2, 1)
        plt.hist(samples.cpu().numpy(), bins=50, density=True, alpha=0.7, color='blue')
        plt.axvline(predictions['expected_magnitude'][idx].cpu(), color='red', linestyle='--', label='Expected')
        plt.axvline(predictions['p10'][idx].cpu(), color='green', linestyle=':', label='P10')
        plt.axvline(predictions['p90'][idx].cpu(), color='orange', linestyle=':', label='P90')
        plt.xlabel('Gap Continuation %')
        plt.ylabel('Probability Density')
        plt.title(f'{symbol} - Predicted Magnitude Distribution')
        plt.legend()
        
        # Plot mixture components
        plt.subplot(2, 2, 2)
        x_range = torch.linspace(0, 10, 200, device=self.device)
        
        for k in range(self.model.n_gaussians):
            weight = predictions['mixture_weights'][idx, k].cpu()
            mean = predictions['component_means'][idx, k].cpu()
            std = predictions['component_stds'][idx, k].cpu()
            
            if weight > 0.05:  # Only plot significant components
                from scipy.stats import norm
                y = weight * norm.pdf(x_range.cpu(), mean, std)
                plt.plot(x_range.cpu(), y, label=f'Component {k+1} (w={weight:.2f})')
        
        plt.xlabel('Gap Continuation %')
        plt.ylabel('Weighted Density')
        plt.title('Mixture Components')
        plt.legend()
        
        # Plot attention weights
        plt.subplot(2, 2, 3)
        feature_names = [self.gap_analyzer.gap_feature_names[i] for i in self.feature_indices]
        attention = predictions['attention_weights'][idx].cpu().numpy()
        plt.bar(range(len(feature_names)), attention)
        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
        plt.ylabel('Attention Weight')
        plt.title('Feature Importance (Attention)')
        plt.tight_layout()
        
        # Plot percentile ranges
        plt.subplot(2, 2, 4)
        percentiles = [10, 25, 50, 75, 90]
        values = [predictions[f'p{p}'][idx].cpu() for p in percentiles]
        plt.plot(percentiles, values, 'o-', markersize=8)
        plt.fill_between([10, 90], [values[0]]*2, [values[-1]]*2, alpha=0.2)
        plt.fill_between([25, 75], [values[1]]*2, [values[-2]]*2, alpha=0.3)
        plt.xlabel('Percentile')
        plt.ylabel('Gap Continuation %')
        plt.title('Percentile Estimates')
        plt.grid(True, alpha=0.3)
        
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
            'scheduler_state': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'feature_indices': self.feature_indices,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(state, self.model_path)
        print(f"Magnitude model saved to {self.model_path}")
    
    def _load_model(self):
        """Load model state."""
        if self.model_path.exists():
            try:
                state = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state['model_state'])
                self.optimizer.load_state_dict(state['optimizer_state'])
                self.scheduler.load_state_dict(state['scheduler_state'])
                self.training_history = state.get('training_history', self.training_history)
                print(f"Magnitude model loaded from {self.model_path}")
            except Exception as e:
                print(f"Error loading magnitude model: {e}")


# Example usage when file is run directly
if __name__ == "__main__":
    print("GPU-Accelerated Neural Magnitude Predictor Test")
    print("=" * 50)
    
    # Initialize components
    from gpu_core.tensor_universe_manager import TensorUniverseManager
    from gpu_core.parallel_data_pipeline import ParallelDataPipeline
    
    config = GPUTradingConfig()
    universe_manager = TensorUniverseManager(config)
    
    # Ensure universe is loaded
    if len(universe_manager.universe_symbols) == 0:
        print("Loading universe...")
        universe_manager.update_universe()
    
    # Create pipeline and analyzer
    data_pipeline = ParallelDataPipeline(config, universe_manager)
    gap_analyzer = VectorizedGapAnalyzer(config, universe_manager, data_pipeline)
    
    # Create synthetic gap features
    n_symbols = len(universe_manager.universe_symbols)
    synthetic_features = torch.randn(n_symbols, 25, device=config.device, dtype=config.TENSOR_DTYPE)
    
    # Make features realistic
    for i in [3, 5, 16, 20, 21, 23, 24]:  # probability-like features
        synthetic_features[:, i] = torch.sigmoid(synthetic_features[:, i])
    
    # Create predictor
    magnitude_predictor = NeuralMagnitudePredictor(config, gap_analyzer)
    
    # Create a mask for high-quality gaps (simulate continuation filter output)
    quality_scores = synthetic_features[:, 3]  # gap_quality_score
    high_quality_mask = quality_scores > 0.6
    n_selected = high_quality_mask.sum().item()
    
    print(f"\nPredicting magnitudes for {n_selected} high-quality gaps...")
    
    # Get predictions
    predictions = magnitude_predictor.predict_magnitude_distribution(
        synthetic_features, 
        mask=high_quality_mask
    )
    
    # Display results
    print(f"\nMagnitude Prediction Results:")
    print(f"  Gaps analyzed: {n_selected}")
    print(f"  Average expected magnitude: {predictions['expected_magnitude'].mean():.2f}%")
    print(f"  Average upside potential (P90-P50): {predictions['upside_potential'].mean():.2f}%")
    print(f"  Average downside risk (P50-P10): {predictions['downside_risk'].mean():.2f}%")
    print(f"  Average reward/risk ratio: {predictions['reward_risk_ratio'].mean():.2f}")
    
    # Show top opportunities by reward/risk
    top_indices = torch.argsort(predictions['reward_risk_ratio'], descending=True)[:5]
    selected_symbols = [universe_manager.universe_symbols[i] for i in torch.where(high_quality_mask)[0]]
    
    print(f"\nTop 5 Opportunities by Reward/Risk:")
    print("-" * 80)
    account_value = 100000  # Example account
    
    for i, idx in enumerate(top_indices):
        symbol = selected_symbols[idx]
        
        print(f"\n{i+1}. {symbol}")
        print(f"   Expected Magnitude: {predictions['expected_magnitude'][idx]:.2f}%")
        print(f"   P10/P50/P90: {predictions['p10'][idx]:.1f}% / {predictions['p50'][idx]:.1f}% / {predictions['p90'][idx]:.1f}%")
        print(f"   Reward/Risk Ratio: {predictions['reward_risk_ratio'][idx]:.2f}")
        print(f"   Skewness: {predictions['skewness'][idx]:.2f}")
        
        # Position size recommendation
        position_sizes = magnitude_predictor.get_position_size_recommendation(
            {k: v[idx:idx+1] for k, v in predictions.items()},
            account_value
        )
        print(f"   Recommended Position: ${position_sizes[0]:,.0f} ({position_sizes[0]/account_value*100:.1f}% of account)")
    
    # Visualize top prediction
    if n_selected > 0:
        print(f"\nGenerating visualization for {selected_symbols[top_indices[0]]}...")
        # Note: Visualization would be saved in production
        # magnitude_predictor.visualize_predictions(predictions, selected_symbols[top_indices[0]])
    
    # Feature importance from attention
    avg_attention = predictions['attention_weights'].mean(dim=0)
    feature_names = [gap_analyzer.gap_feature_names[i] for i in magnitude_predictor.feature_indices]
    
    print(f"\nAverage Feature Importance (Attention Weights):")
    importance_indices = torch.argsort(avg_attention, descending=True)[:5]
    for idx in importance_indices:
        print(f"  {feature_names[idx]}: {avg_attention[idx]:.3f}")