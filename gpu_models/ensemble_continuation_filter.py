#!/usr/bin/env python3
"""
GPU-Accelerated Ensemble Continuation Filter
Implements parallel Bayesian Logistic Regression ensemble for gap continuation prediction.
Makes binary decisions with uncertainty quantification using GPU acceleration.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Beta
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpu_trading_config import GPUTradingConfig
from gpu_core.tensor_universe_manager import TensorUniverseManager
from gpu_features.vectorized_gap_analyzer import VectorizedGapAnalyzer


class BayesianLogisticEnsemble(nn.Module):
    """Ensemble of Bayesian Logistic Regression models on GPU."""
    
    def __init__(self, n_features: int, n_models: int, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.n_features = n_features
        self.n_models = n_models
        self.device = device
        self.dtype = dtype
        
        # Initialize ensemble weights with Bayesian priors
        # Each model has slightly different initialization for diversity
        self.weights = nn.Parameter(torch.zeros(n_models, n_features, device=device, dtype=dtype))
        self.biases = nn.Parameter(torch.zeros(n_models, device=device, dtype=dtype))
        
        # Prior parameters (for regularization)
        self.weight_prior_mean = 0.0
        self.weight_prior_std = 1.0
        self.bias_prior_mean = 0.0
        self.bias_prior_std = 2.0
        
        # Initialize with diverse priors
        self._initialize_ensemble()
        
        # Uncertainty parameters for each model
        self.weight_uncertainty = nn.Parameter(torch.ones(n_models, n_features, device=device, dtype=dtype) * 0.1)
        self.bias_uncertainty = nn.Parameter(torch.ones(n_models, device=device, dtype=dtype) * 0.1)
        
        # Online learning momentum
        self.weight_momentum = torch.zeros_like(self.weights)
        self.bias_momentum = torch.zeros_like(self.biases)
        
    def _initialize_ensemble(self):
        """Initialize ensemble with diverse starting points."""
        # Base initialization from prior
        base_weight_init = Normal(self.weight_prior_mean, self.weight_prior_std)
        base_bias_init = Normal(self.bias_prior_mean, self.bias_prior_std)
        
        with torch.no_grad():
            # Each model gets slightly different initialization
            for i in range(self.n_models):
                # Add model-specific variation
                variation_scale = 0.1 + (i / self.n_models) * 0.2
                
                self.weights[i] = base_weight_init.sample((self.n_features,)) * (1 + variation_scale)
                self.biases[i] = base_bias_init.sample() * (1 + variation_scale)
                
                # Some models focus on different features
                if i % 5 == 0:  # Every 5th model
                    # Emphasize institutional features
                    self.weights[i, 1] *= 2.0  # institutional_footprint
                    self.weights[i, 5] *= 2.0  # smart_money_flow
                elif i % 5 == 1:
                    # Emphasize relative strength
                    self.weights[i, 2] *= 2.0  # relative_gap_strength
                elif i % 5 == 2:
                    # Emphasize stability
                    self.weights[i, 3] *= 2.0  # gap_stability
                elif i % 5 == 3:
                    # Emphasize volume
                    self.weights[i, 4] *= 2.0  # volume_surge_score
    
    def forward(self, x: torch.Tensor, return_all: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ensemble.
        Args:
            x: Input features [batch_size, n_features]
            return_all: If True, return all model predictions
        Returns:
            predictions: Mean predictions [batch_size]
            uncertainty: Prediction uncertainty [batch_size]
        """
        # Compute logits for all models in parallel
        # x: [batch_size, n_features]
        # weights: [n_models, n_features]
        # result: [n_models, batch_size]
        logits = torch.matmul(self.weights, x.T) + self.biases.unsqueeze(1)
        
        # Apply sigmoid for probabilities
        probs = torch.sigmoid(logits)
        
        if return_all:
            return probs.T, logits.T  # Return [batch_size, n_models]
        
        # Compute ensemble statistics
        mean_prob = probs.mean(dim=0)
        std_prob = probs.std(dim=0)
        
        # Uncertainty includes both aleatoric and epistemic
        # Epistemic: disagreement between models (std)
        # Aleatoric: average model uncertainty
        model_uncertainties = self._compute_model_uncertainties(x)
        total_uncertainty = torch.sqrt(std_prob**2 + model_uncertainties.mean(dim=0)**2)
        
        return mean_prob, total_uncertainty
    
    def _compute_model_uncertainties(self, x: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty for each model's predictions."""
        # Propagate parameter uncertainty through sigmoid
        # Using linear approximation for computational efficiency
        
        # Compute gradients of sigmoid at current predictions
        logits = torch.matmul(self.weights, x.T) + self.biases.unsqueeze(1)
        sigmoid_grads = torch.sigmoid(logits) * (1 - torch.sigmoid(logits))
        
        # Weight uncertainty contribution
        weight_var = (self.weight_uncertainty**2).sum(dim=1, keepdim=True)
        
        # Total uncertainty per model
        uncertainties = sigmoid_grads * torch.sqrt(weight_var)
        
        return uncertainties
    
    def get_prediction_with_confidence(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get detailed prediction with confidence intervals."""
        all_probs, all_logits = self.forward(x, return_all=True)
        
        # Compute percentiles for confidence intervals
        lower_ci = torch.quantile(all_probs, 0.025, dim=1)
        upper_ci = torch.quantile(all_probs, 0.975, dim=1)
        median = torch.quantile(all_probs, 0.5, dim=1)
        mean = all_probs.mean(dim=1)
        
        # Agreement score (how much models agree)
        agreement = 1 - all_probs.std(dim=1) / 0.5  # Normalized by max possible std
        
        return {
            'mean_probability': mean,
            'median_probability': median,
            'lower_95_ci': lower_ci,
            'upper_95_ci': upper_ci,
            'agreement_score': agreement,
            'all_predictions': all_probs
        }
    
    def update_online(self, x: torch.Tensor, y: torch.Tensor, learning_rate: float = 0.001):
        """Online learning update using Bayesian approach."""
        # Get current predictions
        probs, _ = self.forward(x, return_all=False)
        
        # Compute gradients
        errors = y - probs
        
        # Update each model with momentum
        for i in range(self.n_models):
            # Compute individual model predictions
            logits_i = torch.matmul(x, self.weights[i]) + self.biases[i]
            probs_i = torch.sigmoid(logits_i)
            errors_i = y - probs_i
            
            # Gradient with L2 regularization (Bayesian prior)
            weight_grad = -torch.matmul(x.T, errors_i) / x.shape[0]
            weight_grad += (self.weights[i] - self.weight_prior_mean) / (self.weight_prior_std**2)
            
            bias_grad = -errors_i.mean()
            bias_grad += (self.biases[i] - self.bias_prior_mean) / (self.bias_prior_std**2)
            
            # Momentum update
            momentum_beta = 0.9
            self.weight_momentum[i] = momentum_beta * self.weight_momentum[i] + (1 - momentum_beta) * weight_grad
            self.bias_momentum[i] = momentum_beta * self.bias_momentum[i] + (1 - momentum_beta) * bias_grad
            
            # Apply updates
            with torch.no_grad():
                self.weights[i] -= learning_rate * self.weight_momentum[i]
                self.biases[i] -= learning_rate * self.bias_momentum[i]
                
                # Update uncertainties based on prediction errors
                error_magnitude = torch.abs(errors_i).mean()
                self.weight_uncertainty[i] *= (1 + 0.01 * error_magnitude)
                self.weight_uncertainty[i] = torch.clamp(self.weight_uncertainty[i], 0.01, 2.0)


class EnsembleContinuationFilter:
    """Main continuation filter using Bayesian ensemble."""
    
    def __init__(self, config: GPUTradingConfig, gap_analyzer: VectorizedGapAnalyzer):
        self.config = config
        self.gap_analyzer = gap_analyzer
        self.device = config.device
        self.dtype = config.TENSOR_DTYPE
        
        # Select robust features for continuation prediction
        self.feature_indices = [
            3,   # gap_quality_score
            5,   # institutional_footprint  
            10,  # relative_gap_strength
            16,  # gap_stability
            21,  # volume_surge_score
            23,  # momentum_alignment
            24   # continuation_probability
        ]
        self.n_features = len(self.feature_indices)
        
        # Initialize ensemble
        self.ensemble = BayesianLogisticEnsemble(
            n_features=self.n_features,
            n_models=config.ENSEMBLE_MODEL_COUNT,
            device=self.device,
            dtype=self.dtype
        )
        
        # Training history for online learning
        self.prediction_history = []
        self.outcome_history = []
        self.max_history = 1000
        
        # Model performance tracking
        self.performance_stats = {
            'predictions_made': 0,
            'correct_predictions': 0,
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'avg_confidence_when_correct': 0.0,
            'avg_confidence_when_wrong': 0.0
        }
        
        # Load saved model if exists
        self.model_path = config.MODELS_DIR / 'continuation_ensemble.pth'
        self._load_model()
        
    def predict_continuation(self, gap_features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Predict gap continuation probability for all symbols.
        Args:
            gap_features: Full gap features tensor from analyzer
        Returns:
            should_trade: Boolean tensor of trade decisions
            predictions: Detailed predictions dictionary
        """
        # Extract relevant features
        selected_features = gap_features[:, self.feature_indices]
        
        # Normalize features (important for logistic regression)
        normalized_features = self._normalize_features(selected_features)
        
        # Get ensemble predictions
        predictions = self.ensemble.get_prediction_with_confidence(normalized_features)
        
        # Make trading decisions based on confidence threshold
        mean_probs = predictions['mean_probability']
        confidence_intervals = predictions['upper_95_ci'] - predictions['lower_95_ci']
        
        # Trade only when confident (high probability and tight confidence interval)
        should_trade = (
            (mean_probs > self.config.CONFIDENCE_THRESHOLD) &
            (confidence_intervals < 0.3) &  # Reasonable agreement
            (predictions['agreement_score'] > 0.7)  # Models agree
        )
        
        # Add decision metadata
        predictions['should_trade'] = should_trade
        predictions['confidence'] = mean_probs * predictions['agreement_score']
        
        return should_trade, predictions
    
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features for stable logistic regression."""
        # Use robust normalization (median and MAD)
        median = features.median(dim=0)[0]
        mad = (features - median).abs().median(dim=0)[0]
        
        # Avoid division by zero
        mad = torch.maximum(mad, torch.ones_like(mad) * 0.01)
        
        normalized = (features - median) / (1.4826 * mad)  # 1.4826 converts MAD to std
        
        # Clip extreme values
        normalized = torch.clamp(normalized, -5, 5)
        
        return normalized
    
    def update_with_outcome(self, symbol: str, predicted_prob: float, actual_continuation: bool):
        """Update model with actual outcome for online learning."""
        # Store in history
        self.prediction_history.append(predicted_prob)
        self.outcome_history.append(1.0 if actual_continuation else 0.0)
        
        # Update performance stats
        self.performance_stats['predictions_made'] += 1
        
        predicted_class = predicted_prob > self.config.CONFIDENCE_THRESHOLD
        
        if predicted_class == actual_continuation:
            self.performance_stats['correct_predictions'] += 1
            self.performance_stats['avg_confidence_when_correct'] = (
                (self.performance_stats['avg_confidence_when_correct'] * 
                 (self.performance_stats['correct_predictions'] - 1) + predicted_prob) /
                self.performance_stats['correct_predictions']
            )
        else:
            wrong_count = self.performance_stats['predictions_made'] - self.performance_stats['correct_predictions']
            self.performance_stats['avg_confidence_when_wrong'] = (
                (self.performance_stats['avg_confidence_when_wrong'] * (wrong_count - 1) + predicted_prob) /
                wrong_count
            )
        
        # Update confusion matrix
        if predicted_class and actual_continuation:
            self.performance_stats['true_positives'] += 1
        elif predicted_class and not actual_continuation:
            self.performance_stats['false_positives'] += 1
        elif not predicted_class and actual_continuation:
            self.performance_stats['false_negatives'] += 1
        else:
            self.performance_stats['true_negatives'] += 1
        
        # Perform online update if we have enough history
        if len(self.prediction_history) >= 20:
            self._perform_online_update()
        
        # Trim history if too long
        if len(self.prediction_history) > self.max_history:
            self.prediction_history = self.prediction_history[-self.max_history:]
            self.outcome_history = self.outcome_history[-self.max_history:]
    
    def _perform_online_update(self):
        """Perform online learning update using recent predictions."""
        # Get recent predictions and outcomes
        recent_predictions = torch.tensor(self.prediction_history[-100:], device=self.device, dtype=self.dtype)
        recent_outcomes = torch.tensor(self.outcome_history[-100:], device=self.device, dtype=self.dtype)
        
        # Reconstruct features (this is simplified - in production, store features too)
        # For now, create synthetic features based on predictions
        n_samples = len(recent_predictions)
        synthetic_features = torch.randn(n_samples, self.n_features, device=self.device, dtype=self.dtype)
        
        # Weight recent examples more
        time_weights = torch.linspace(0.5, 1.0, n_samples, device=self.device, dtype=self.dtype)
        
        # Update ensemble
        self.ensemble.update_online(
            synthetic_features,
            recent_outcomes,
            learning_rate=self.config.ONLINE_LEARNING_RATE
        )
    
    def get_model_diagnostics(self) -> Dict[str, any]:
        """Get detailed model diagnostics and performance metrics."""
        stats = self.performance_stats.copy()
        
        # Calculate additional metrics
        if stats['predictions_made'] > 0:
            stats['accuracy'] = stats['correct_predictions'] / stats['predictions_made']
            
            # Precision and recall
            if stats['true_positives'] + stats['false_positives'] > 0:
                stats['precision'] = stats['true_positives'] / (stats['true_positives'] + stats['false_positives'])
            else:
                stats['precision'] = 0.0
                
            if stats['true_positives'] + stats['false_negatives'] > 0:
                stats['recall'] = stats['true_positives'] / (stats['true_positives'] + stats['false_negatives'])
            else:
                stats['recall'] = 0.0
            
            # F1 score
            if stats['precision'] + stats['recall'] > 0:
                stats['f1_score'] = 2 * (stats['precision'] * stats['recall']) / (stats['precision'] + stats['recall'])
            else:
                stats['f1_score'] = 0.0
        
        # Model diversity metrics
        if hasattr(self.ensemble, 'weights'):
            weight_diversity = self.ensemble.weights.std(dim=0).mean().item()
            stats['ensemble_diversity'] = weight_diversity
        
        return stats
    
    def _save_model(self):
        """Save model state to disk."""
        state = {
            'ensemble_state': self.ensemble.state_dict(),
            'performance_stats': self.performance_stats,
            'prediction_history': self.prediction_history[-100:],  # Keep recent history
            'outcome_history': self.outcome_history[-100:],
            'feature_indices': self.feature_indices,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(state, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def _load_model(self):
        """Load model state from disk."""
        if self.model_path.exists():
            try:
                state = torch.load(self.model_path, map_location=self.device)
                self.ensemble.load_state_dict(state['ensemble_state'])
                self.performance_stats = state.get('performance_stats', self.performance_stats)
                self.prediction_history = state.get('prediction_history', [])
                self.outcome_history = state.get('outcome_history', [])
                print(f"Model loaded from {self.model_path}")
                print(f"Historical accuracy: {self.performance_stats.get('accuracy', 0):.2%}")
            except Exception as e:
                print(f"Error loading model: {e}")


# Example usage when file is run directly
if __name__ == "__main__":
    print("GPU-Accelerated Ensemble Continuation Filter Test")
    print("=" * 50)
    
    # Initialize components
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
    
    # Simulate some gap features
    print("\nSimulating gap features...")
    n_symbols = len(universe_manager.universe_symbols)
    
    # Create synthetic gap features for testing
    synthetic_features = torch.randn(n_symbols, 25, device=config.device, dtype=config.TENSOR_DTYPE)
    
    # Make some features more realistic
    synthetic_features[:, 3] = torch.sigmoid(synthetic_features[:, 3])  # gap_quality_score
    synthetic_features[:, 5] = torch.sigmoid(synthetic_features[:, 5])  # institutional_footprint
    synthetic_features[:, 24] = torch.sigmoid(synthetic_features[:, 24])  # continuation_probability
    
    # Create filter
    continuation_filter = EnsembleContinuationFilter(config, gap_analyzer)
    
    # Make predictions
    print("\nMaking continuation predictions...")
    should_trade, predictions = continuation_filter.predict_continuation(synthetic_features)
    
    # Display results
    print(f"\nPrediction Results:")
    print(f"  Symbols analyzed: {n_symbols}")
    print(f"  Symbols passing filter: {should_trade.sum().item()}")
    print(f"  Average confidence: {predictions['confidence'].mean():.3f}")
    print(f"  Average agreement: {predictions['agreement_score'].mean():.3f}")
    
    # Show top recommendations
    top_indices = torch.argsort(predictions['confidence'], descending=True)[:10]
    
    print(f"\nTop 10 High-Confidence Predictions:")
    print("-" * 60)
    for i, idx in enumerate(top_indices):
        if should_trade[idx]:
            symbol = universe_manager.universe_symbols[idx]
            mean_prob = predictions['mean_probability'][idx].item()
            confidence = predictions['confidence'][idx].item()
            ci_width = (predictions['upper_95_ci'][idx] - predictions['lower_95_ci'][idx]).item()
            
            print(f"{i+1}. {symbol}")
            print(f"   Continuation Probability: {mean_prob:.3f}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   95% CI Width: {ci_width:.3f}")
            print(f"   Decision: {'TRADE' if should_trade[idx] else 'SKIP'}")
    
    # Test online learning
    print(f"\nTesting online learning...")
    
    # Simulate some outcomes
    for i in range(20):
        # Random symbol and outcome
        symbol_idx = torch.randint(0, n_symbols, (1,)).item()
        symbol = universe_manager.universe_symbols[symbol_idx]
        predicted_prob = predictions['mean_probability'][symbol_idx].item()
        
        # Simulate outcome (biased by prediction for realism)
        actual_continuation = torch.rand(1).item() < (predicted_prob * 0.7 + 0.15)
        
        # Update model
        continuation_filter.update_with_outcome(symbol, predicted_prob, actual_continuation)
    
    # Show diagnostics
    diagnostics = continuation_filter.get_model_diagnostics()
    
    print(f"\nModel Diagnostics After Updates:")
    print(f"  Predictions made: {diagnostics['predictions_made']}")
    print(f"  Accuracy: {diagnostics.get('accuracy', 0):.2%}")
    print(f"  Precision: {diagnostics.get('precision', 0):.2%}")
    print(f"  Recall: {diagnostics.get('recall', 0):.2%}")
    print(f"  F1 Score: {diagnostics.get('f1_score', 0):.3f}")
    
    # Save model
    continuation_filter._save_model()
    print(f"\nModel saved successfully!")