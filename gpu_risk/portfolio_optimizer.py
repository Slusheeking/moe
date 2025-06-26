#!/usr/bin/env python3
"""
GPU-Accelerated Portfolio Optimizer
Performs real-time portfolio optimization using Monte Carlo simulations,
correlation analysis, and sophisticated risk management on GPU.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpu_trading_config import GPUTradingConfig
from database.connection import get_db_manager
from gpu_core.tensor_universe_manager import TensorUniverseManager
from gpu_models.neural_magnitude_predictor import NeuralMagnitudePredictor


@dataclass
class PortfolioState:
    """Current portfolio state and risk metrics."""
    positions: Dict[str, float]  # symbol -> position size
    total_value: float
    cash: float
    total_risk: float  # portfolio heat
    position_risks: Dict[str, float]  # symbol -> position risk
    correlations: torch.Tensor  # correlation matrix
    sector_exposures: Dict[int, float]  # sector -> exposure
    
    
class GPUPortfolioOptimizer:
    """GPU-accelerated portfolio optimization for gap trading."""
    
    def __init__(self, config: GPUTradingConfig, 
                 universe_manager: TensorUniverseManager,
                 initial_capital: float = 100000):
        self.config = config
        self.universe_manager = universe_manager
        self.device = config.device
        self.dtype = config.TENSOR_DTYPE
        self.initial_capital = initial_capital
        
        # Database connection for risk metrics persistence
        self.db_manager = get_db_manager()
        
        # Get universe information
        self.universe_tensor, self.universe_symbols = universe_manager.get_universe_tensor()
        self.n_symbols = len(self.universe_symbols)
        self.symbol_to_idx = {s: i for i, s in enumerate(self.universe_symbols)}
        
        # Initialize portfolio state
        self.portfolio_state = PortfolioState(
            positions={},
            total_value=initial_capital,
            cash=initial_capital,
            total_risk=0.0,
            position_risks={},
            correlations=torch.eye(self.n_symbols, device=self.device, dtype=self.dtype),
            sector_exposures={}
        )
        
        # Risk calculation tensors
        self._init_risk_tensors()
        
        # Monte Carlo simulation parameters
        self.n_simulations = 10000
        self.simulation_horizon = 390  # minutes in trading day
        
        # Correlation estimation
        self.correlation_window = 60  # days
        self.min_correlation_obs = 30  # minimum observations for correlation
        
    def _init_risk_tensors(self):
        """Initialize GPU tensors for risk calculations."""
        # Position tensor (1 if position exists, 0 otherwise)
        self.position_mask = torch.zeros(self.n_symbols, device=self.device, dtype=torch.bool)
        
        # Position sizes tensor
        self.position_sizes = torch.zeros(self.n_symbols, device=self.device, dtype=self.dtype)
        
        # Risk metrics tensors
        self.position_vars = torch.zeros(self.n_symbols, device=self.device, dtype=self.dtype)
        self.position_expected_returns = torch.zeros(self.n_symbols, device=self.device, dtype=self.dtype)
        
        # Correlation matrix (updated dynamically)
        self.correlation_matrix = torch.eye(self.n_symbols, device=self.device, dtype=self.dtype)
        
        # Sector exposure tensor
        self.sector_exposure = torch.zeros(11, device=self.device, dtype=self.dtype)  # 11 sectors
        
    def optimize_portfolio(self, 
                          gap_opportunities: Dict[str, Dict],
                          magnitude_predictions: Dict[str, torch.Tensor],
                          regime_adjustments: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize portfolio allocation across gap opportunities.
        
        Args:
            gap_opportunities: Dict of symbol -> gap features
            magnitude_predictions: Dict of symbol -> magnitude distribution predictions
            regime_adjustments: Current regime adjustments from classifier
            
        Returns:
            Dict of symbol -> recommended position size
        """
        print(f"Optimizing portfolio for {len(gap_opportunities)} opportunities...")
        
        # Update correlation matrix
        self._update_correlations()
        
        # Prepare optimization tensors
        opp_symbols = list(gap_opportunities.keys())
        opp_indices = torch.tensor([self.symbol_to_idx[s] for s in opp_symbols], 
                                  device=self.device, dtype=torch.long)
        n_opportunities = len(opp_symbols)
        
        if n_opportunities == 0:
            return {}
        
        # Extract expected returns and risks from magnitude predictions
        expected_returns = torch.zeros(n_opportunities, device=self.device, dtype=self.dtype)
        downside_risks = torch.zeros(n_opportunities, device=self.device, dtype=self.dtype)
        upside_potentials = torch.zeros(n_opportunities, device=self.device, dtype=self.dtype)
        
        for i, symbol in enumerate(opp_symbols):
            pred = magnitude_predictions[symbol]
            expected_returns[i] = pred['expected_magnitude'] / 100  # Convert to decimal
            downside_risks[i] = pred['downside_risk'] / 100
            upside_potentials[i] = pred['upside_potential'] / 100
        
        # Extract correlation submatrix for opportunities
        opp_correlations = self.correlation_matrix[opp_indices][:, opp_indices]
        
        # Run portfolio optimization
        optimal_weights = self._optimize_weights(
            expected_returns,
            downside_risks,
            upside_potentials,
            opp_correlations,
            regime_adjustments
        )
        
        # Apply position sizing constraints
        position_sizes = self._calculate_position_sizes(
            optimal_weights,
            opp_symbols,
            gap_opportunities,
            regime_adjustments
        )
        
        # Run Monte Carlo risk simulation
        risk_metrics = self._monte_carlo_risk_simulation(
            position_sizes,
            opp_symbols,
            magnitude_predictions
        )
        
        # Apply risk limits
        final_positions = self._apply_risk_limits(
            position_sizes,
            risk_metrics,
            opp_symbols
        )
        
        # Persist risk metrics to database
        self._persist_risk_metrics(risk_metrics, opp_symbols, final_positions)
        
        return final_positions
    
    def _optimize_weights(self,
                         expected_returns: torch.Tensor,
                         downside_risks: torch.Tensor,
                         upside_potentials: torch.Tensor,
                         correlations: torch.Tensor,
                         regime_adjustments: Dict[str, float]) -> torch.Tensor:
        """
        Optimize portfolio weights using modified Markowitz with downside focus.
        """
        n_assets = len(expected_returns)
        
        # Initialize with equal weights
        weights = torch.ones(n_assets, device=self.device, dtype=self.dtype) / n_assets
        
        # Optimization parameters
        risk_aversion = 2.0  # Higher = more risk averse
        learning_rate = 0.01
        n_iterations = 1000
        
        # Adjust risk aversion based on regime
        regime_multiplier = regime_adjustments.get('position_size_multiplier', 1.0)
        risk_aversion = risk_aversion / regime_multiplier  # Less risk averse in trending markets
        
        # Create downside covariance matrix (semi-variance)
        # Using downside risks and correlations
        downside_vols = torch.sqrt(downside_risks)
        downside_cov = correlations * downside_vols.unsqueeze(1) * downside_vols.unsqueeze(0)
        
        # Add regularization to ensure positive definite
        downside_cov = downside_cov + torch.eye(n_assets, device=self.device) * 1e-4
        
        # Gradient descent optimization
        for _ in range(n_iterations):
            # Portfolio return
            portfolio_return = torch.sum(weights * expected_returns)
            
            # Portfolio downside risk (using semi-variance)
            portfolio_variance = torch.matmul(weights, torch.matmul(downside_cov, weights))
            portfolio_risk = torch.sqrt(portfolio_variance + 1e-8)
            
            # Objective: Maximize Sharpe-like ratio with downside focus
            # Using Sortino ratio concept
            objective = portfolio_return / (portfolio_risk + 1e-8)
            
            # Add penalty for concentration
            concentration_penalty = torch.sum(weights ** 2) * 0.1
            
            # Total objective (negative for minimization)
            total_objective = -objective + concentration_penalty
            
            # Compute gradients
            weights.requires_grad_(True)
            total_objective.backward()
            
            # Update weights
            with torch.no_grad():
                weights -= learning_rate * weights.grad
                weights = F.softmax(weights, dim=0)  # Ensure sum to 1 and positive
                
            weights.requires_grad_(False)
        
        # Apply regime-based adjustments
        if 'gap_direction_bias' in regime_adjustments:
            # In trending markets, overweight positions in trend direction
            # This would require gap direction information
            pass
        
        return weights
    
    def _calculate_position_sizes(self,
                                optimal_weights: torch.Tensor,
                                symbols: List[str],
                                gap_opportunities: Dict[str, Dict],
                                regime_adjustments: Dict[str, float]) -> Dict[str, float]:
        """Convert optimal weights to actual position sizes with tier allocation."""
        available_capital = self.portfolio_state.cash * (1 - self.config.RESERVE_ALLOCATION)
        
        # Get gap quality scores for tiering
        quality_scores = torch.tensor([
            gap_opportunities[symbol].get('gap_quality_score', 0.5)
            for symbol in symbols
        ], device=self.device, dtype=self.dtype)
        
        # Rank opportunities
        quality_ranks = torch.argsort(quality_scores, descending=True)
        
        # Allocate capital by tiers
        position_sizes = {}
        
        # Tier 1: Top 2-3 opportunities (40% of available capital)
        tier1_count = min(3, len(symbols))
        tier1_capital = available_capital * self.config.TIER1_ALLOCATION
        
        for i in range(tier1_count):
            idx = quality_ranks[i].item()
            symbol = symbols[idx]
            
            # Combine optimal weight with tier allocation
            tier_weight = 1.0 / tier1_count
            combined_weight = (optimal_weights[idx] + tier_weight) / 2
            
            size = tier1_capital * combined_weight
            size = self._apply_position_constraints(size, symbol)
            position_sizes[symbol] = size
        
        # Tier 2: Next 4-5 opportunities (40% of available capital)
        tier2_start = tier1_count
        tier2_count = min(5, len(symbols) - tier1_count)
        tier2_capital = available_capital * self.config.TIER2_ALLOCATION
        
        if tier2_count > 0:
            for i in range(tier2_count):
                idx = quality_ranks[tier2_start + i].item()
                symbol = symbols[idx]
                
                tier_weight = 1.0 / tier2_count
                combined_weight = (optimal_weights[idx] + tier_weight) / 2
                
                size = tier2_capital * combined_weight
                size = self._apply_position_constraints(size, symbol)
                position_sizes[symbol] = size
        
        # Apply regime adjustments
        regime_multiplier = regime_adjustments.get('position_size_multiplier', 1.0)
        for symbol in position_sizes:
            position_sizes[symbol] *= regime_multiplier
        
        return position_sizes
    
    def _apply_position_constraints(self, size: float, symbol: str) -> float:
        """Apply min/max position size constraints."""
        # Account value for percentage calculations
        account_value = self.portfolio_state.total_value
        
        # Apply constraints
        min_size = self.config.MIN_POSITION_SIZE
        max_size = account_value * self.config.MAX_POSITION_SIZE_PERCENT
        
        return max(min_size, min(size, max_size))
    
    def _monte_carlo_risk_simulation(self,
                                   position_sizes: Dict[str, float],
                                   symbols: List[str],
                                   magnitude_predictions: Dict[str, torch.Tensor]) -> Dict[str, any]:
        """Run Monte Carlo simulation for portfolio risk assessment."""
        n_positions = len(position_sizes)
        if n_positions == 0:
            return {}
        
        # Prepare simulation tensors
        positions_tensor = torch.tensor([position_sizes[s] for s in symbols], 
                                      device=self.device, dtype=self.dtype)
        
        # Extract distribution parameters for each position
        means = torch.zeros(n_positions, device=self.device, dtype=self.dtype)
        stds = torch.zeros(n_positions, device=self.device, dtype=self.dtype)
        
        for i, symbol in enumerate(symbols):
            pred = magnitude_predictions[symbol]
            means[i] = pred['expected_magnitude'] / 100
            stds[i] = pred['magnitude_std'] / 100
        
        # Get correlation submatrix
        indices = torch.tensor([self.symbol_to_idx[s] for s in symbols], device=self.device)
        position_correlations = self.correlation_matrix[indices][:, indices]
        
        # Generate correlated random returns
        # Using Cholesky decomposition for correlated samples
        L = torch.linalg.cholesky(position_correlations + torch.eye(n_positions, device=self.device) * 1e-6)
        
        # Run simulations
        portfolio_returns = torch.zeros(self.n_simulations, device=self.device, dtype=self.dtype)
        
        for sim in range(0, self.n_simulations, 1000):  # Batch for memory efficiency
            batch_size = min(1000, self.n_simulations - sim)
            
            # Generate independent standard normal samples
            z = torch.randn(batch_size, n_positions, device=self.device, dtype=self.dtype)
            
            # Create correlated samples
            correlated_z = torch.matmul(z, L.T)
            
            # Convert to returns using each position's distribution
            returns = means.unsqueeze(0) + stds.unsqueeze(0) * correlated_z
            
            # Calculate portfolio returns
            portfolio_returns[sim:sim+batch_size] = torch.matmul(returns, positions_tensor) / self.portfolio_state.total_value
        
        # Calculate risk metrics
        var_95 = torch.quantile(portfolio_returns, 0.05)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        return {
            'portfolio_returns': portfolio_returns,
            'expected_return': portfolio_returns.mean().item(),
            'return_std': portfolio_returns.std().item(),
            'var_95': var_95.item(),
            'cvar_95': cvar_95.item(),
            'sharpe_ratio': portfolio_returns.mean() / (portfolio_returns.std() + 1e-8),
            'max_drawdown': (portfolio_returns.min() - portfolio_returns.max()).item(),
            'win_rate': (portfolio_returns > 0).float().mean().item(),
            'profit_factor': portfolio_returns[portfolio_returns > 0].sum() / (-portfolio_returns[portfolio_returns < 0].sum() + 1e-8)
        }
    
    def _apply_risk_limits(self,
                          position_sizes: Dict[str, float],
                          risk_metrics: Dict[str, any],
                          symbols: List[str]) -> Dict[str, float]:
        """Apply portfolio risk limits and adjust positions if needed."""
        if not risk_metrics:
            return position_sizes
        
        # Check portfolio heat
        total_risk = abs(risk_metrics['cvar_95']) * self.portfolio_state.total_value
        max_risk = self.portfolio_state.total_value * self.config.MAX_PORTFOLIO_HEAT
        
        if total_risk > max_risk:
            # Scale down all positions proportionally
            scale_factor = max_risk / total_risk
            print(f"Scaling positions by {scale_factor:.2f} to meet risk limits")
            
            for symbol in position_sizes:
                position_sizes[symbol] *= scale_factor
        
        # Check sector concentration
        sector_exposures = self._calculate_sector_exposures(position_sizes, symbols)
        
        # Apply maximum correlated positions limit
        correlations = self._get_position_correlations(symbols)
        highly_correlated = self._find_highly_correlated_positions(correlations, symbols)
        
        if len(highly_correlated) > self.config.MAX_CORRELATED_POSITIONS:
            # Keep only the best opportunities in correlated groups
            print(f"Reducing correlated positions from {len(highly_correlated)} to {self.config.MAX_CORRELATED_POSITIONS}")
            
            # Sort by expected return and keep top N
            sorted_positions = sorted(
                highly_correlated,
                key=lambda s: risk_metrics.get('expected_return', 0),
                reverse=True
            )
            
            for symbol in sorted_positions[self.config.MAX_CORRELATED_POSITIONS:]:
                position_sizes.pop(symbol, None)
        
        # Final size check
        final_positions = {}
        for symbol, size in position_sizes.items():
            if size >= self.config.MIN_POSITION_SIZE:
                final_positions[symbol] = size
        
        return final_positions
    
    def _update_correlations(self):
        """Update correlation matrix using recent price data."""
        # In production, this would use historical returns
        # For now, create a realistic correlation structure
        
        # Start with identity matrix
        self.correlation_matrix = torch.eye(self.n_symbols, device=self.device, dtype=self.dtype)
        
        # Add sector correlations
        sector_assignments = self.universe_manager.sector_assignments
        
        for i in range(self.n_symbols):
            for j in range(i+1, self.n_symbols):
                # Base correlation
                base_corr = 0.3
                
                # Same sector = higher correlation
                if sector_assignments[i] == sector_assignments[j]:
                    correlation = base_corr + 0.4
                else:
                    correlation = base_corr
                
                # Add some randomness
                correlation += torch.randn(1, device=self.device).item() * 0.1
                correlation = max(-1, min(1, correlation))
                
                self.correlation_matrix[i, j] = correlation
                self.correlation_matrix[j, i] = correlation
    
    def _calculate_sector_exposures(self, position_sizes: Dict[str, float], 
                                  symbols: List[str]) -> Dict[int, float]:
        """Calculate exposure by sector."""
        sector_exposures = {}
        
        for symbol, size in position_sizes.items():
            if symbol in self.symbol_to_idx:
                idx = self.symbol_to_idx[symbol]
                sector = self.universe_manager.sector_assignments[idx].item()
                sector_exposures[sector] = sector_exposures.get(sector, 0) + size
        
        return sector_exposures
    
    def _get_position_correlations(self, symbols: List[str]) -> torch.Tensor:
        """Get correlation matrix for specific symbols."""
        indices = [self.symbol_to_idx[s] for s in symbols]
        return self.correlation_matrix[indices][:, indices]
    
    def _find_highly_correlated_positions(self, correlations: torch.Tensor, 
                                        symbols: List[str]) -> List[str]:
        """Find positions with correlation above threshold."""
        high_corr_mask = correlations > self.config.CORRELATION_THRESHOLD
        
        # Find connected components (groups of correlated positions)
        correlated_groups = []
        visited = set()
        
        for i, symbol in enumerate(symbols):
            if i not in visited:
                group = self._find_correlation_group(i, high_corr_mask, symbols, visited)
                if len(group) > 1:
                    correlated_groups.extend(group)
        
        return list(set(correlated_groups))
    
    def _find_correlation_group(self, start_idx: int, corr_mask: torch.Tensor,
                               symbols: List[str], visited: set) -> List[str]:
        """Find all positions correlated with start_idx."""
        group = [symbols[start_idx]]
        visited.add(start_idx)
        
        for i in range(len(symbols)):
            if i not in visited and corr_mask[start_idx, i]:
                visited.add(i)
                group.extend(self._find_correlation_group(i, corr_mask, symbols, visited))
        
        return group
    
    def _persist_risk_metrics(self, risk_metrics: Dict, symbols: List[str], final_positions: Dict[str, float]):
        """Persist portfolio risk metrics to database."""
        try:
            from datetime import datetime
            
            # Calculate portfolio-level metrics
            total_portfolio_value = sum(final_positions.values())
            portfolio_var = risk_metrics.get('portfolio_var', 0.0)
            expected_return = risk_metrics.get('expected_return', 0.0)
            max_drawdown = risk_metrics.get('max_drawdown', 0.0)
            
            # Calculate position concentration
            position_concentration = max(final_positions.values()) / max(total_portfolio_value, 1) if final_positions else 0.0
            
            # Calculate sector exposures
            sector_exposures = self._calculate_sector_exposures(final_positions, symbols)
            max_sector_exposure = max(sector_exposures.values()) if sector_exposures else 0.0
            
            # Count correlations above threshold
            if len(symbols) > 1:
                correlations = self._get_position_correlations(symbols)
                high_corr_pairs = (correlations > self.config.CORRELATION_THRESHOLD).sum().item() - len(symbols)  # Exclude diagonal
            else:
                high_corr_pairs = 0
            
            # Build comprehensive metadata
            metadata = {
                'portfolio_composition': {
                    'n_positions': len(final_positions),
                    'symbols': list(final_positions.keys()),
                    'position_sizes': {k: float(v) for k, v in final_positions.items()},
                },
                'sector_analysis': {
                    'sector_exposures': {str(k): float(v) for k, v in sector_exposures.items()},
                    'max_sector_exposure': float(max_sector_exposure),
                    'sector_diversification': len(sector_exposures),
                },
                'correlation_analysis': {
                    'high_correlation_pairs': int(high_corr_pairs),
                    'correlation_threshold': float(self.config.CORRELATION_THRESHOLD),
                    'avg_correlation': float(correlations.mean().item()) if len(symbols) > 1 else 0.0,
                },
                'simulation_params': {
                    'n_simulations': self.n_simulations,
                    'simulation_horizon_minutes': self.simulation_horizon,
                    'correlation_window_days': self.correlation_window,
                },
                'optimizer_version': '1.0'
            }
            
            # Prepare database record
            risk_record = {
                'timestamp': datetime.now(),
                'portfolio_var': float(portfolio_var),
                'expected_return': float(expected_return),
                'max_drawdown': float(max_drawdown),
                'position_concentration': float(position_concentration),
                'correlation_risk': float(high_corr_pairs / max(len(symbols) * (len(symbols) - 1) / 2, 1)),
                'n_positions': len(final_positions),
                'total_exposure': float(total_portfolio_value),
                'metadata': metadata
            }
            
            # Insert risk metrics
            insert_query = """
                INSERT INTO risk_metrics (
                    timestamp, portfolio_var, expected_return, max_drawdown,
                    position_concentration, correlation_risk, n_positions, total_exposure, metadata
                ) VALUES (
                    %(timestamp)s, %(portfolio_var)s, %(expected_return)s, %(max_drawdown)s,
                    %(position_concentration)s, %(correlation_risk)s, %(n_positions)s, %(total_exposure)s, %(metadata)s
                )
            """
            
            success = self.db_manager.execute_query(insert_query, risk_record)
            if success:
                print(f"Persisted risk metrics: {len(final_positions)} positions, VaR: {portfolio_var:.4f}")
            else:
                print(f"Failed to persist risk metrics to database")
                
        except Exception as e:
            print(f"Error persisting risk metrics to database: {e}")
            # Continue processing even if database persistence fails
    
    def update_portfolio_state(self, executed_positions: Dict[str, float]):
        """Update portfolio state after trades are executed."""
        for symbol, size in executed_positions.items():
            self.portfolio_state.positions[symbol] = size
            self.portfolio_state.cash -= size
        
        self.portfolio_state.total_value = self.portfolio_state.cash + sum(self.portfolio_state.positions.values())
    
    def get_risk_dashboard(self) -> Dict[str, any]:
        """Get comprehensive risk metrics for current portfolio."""
        if not self.portfolio_state.positions:
            return {
                'total_positions': 0,
                'portfolio_heat': 0.0,
                'cash_percentage': 100.0,
                'largest_position': 0.0,
                'sector_concentration': 0.0
            }
        
        # Calculate metrics
        total_positions = len(self.portfolio_state.positions)
        total_exposure = sum(self.portfolio_state.positions.values())
        
        # Portfolio heat (simplified - in production would use actual stop losses)
        portfolio_heat = total_exposure / self.portfolio_state.total_value
        
        # Cash percentage
        cash_percentage = (self.portfolio_state.cash / self.portfolio_state.total_value) * 100
        
        # Largest position
        largest_position = max(self.portfolio_state.positions.values()) / self.portfolio_state.total_value
        
        # Sector concentration
        sector_exposures = self._calculate_sector_exposures(
            self.portfolio_state.positions,
            list(self.portfolio_state.positions.keys())
        )
        
        if sector_exposures:
            max_sector_exposure = max(sector_exposures.values())
            sector_concentration = max_sector_exposure / total_exposure
        else:
            sector_concentration = 0.0
        
        return {
            'total_positions': total_positions,
            'portfolio_heat': portfolio_heat,
            'cash_percentage': cash_percentage,
            'largest_position': largest_position,
            'sector_concentration': sector_concentration,
            'total_exposure': total_exposure,
            'sector_exposures': sector_exposures
        }


# Example usage when file is run directly
if __name__ == "__main__":
    print("GPU-Accelerated Portfolio Optimizer Test")
    print("=" * 50)
    
    # Initialize components
    config = GPUTradingConfig()
    universe_manager = TensorUniverseManager(config)
    
    # Ensure universe is loaded
    if len(universe_manager.universe_symbols) == 0:
        print("Loading universe...")
        universe_manager.update_universe()
    
    # Create portfolio optimizer
    optimizer = GPUPortfolioOptimizer(config, universe_manager, initial_capital=100000)
    
    # Create mock gap opportunities
    print("\nCreating mock gap opportunities...")
    
    # Select 10 random symbols
    selected_indices = torch.randperm(len(universe_manager.universe_symbols))[:10]
    selected_symbols = [universe_manager.universe_symbols[idx] for idx in selected_indices]
    
    # Create gap opportunities with varying quality
    gap_opportunities = {}
    magnitude_predictions = {}
    
    for i, symbol in enumerate(selected_symbols):
        # Mock gap features
        quality_score = 0.5 + (i / 20)  # Increasing quality
        gap_opportunities[symbol] = {
            'gap_quality_score': quality_score,
            'gap_size': 2.0 + i * 0.3,
            'institutional_footprint': 0.3 + i * 0.05
        }
        
        # Mock magnitude predictions
        expected_return = 2.0 + i * 0.2  # 2-4% expected continuation
        magnitude_predictions[symbol] = {
            'expected_magnitude': torch.tensor(expected_return, device=config.device),
            'magnitude_std': torch.tensor(1.0, device=config.device),
            'downside_risk': torch.tensor(1.5, device=config.device),
            'upside_potential': torch.tensor(3.0, device=config.device),
            'p10': torch.tensor(expected_return - 1.5, device=config.device),
            'p50': torch.tensor(expected_return, device=config.device),
            'p90': torch.tensor(expected_return + 2.0, device=config.device),
            'reward_risk_ratio': torch.tensor(1.5, device=config.device)
        }
    
    # Mock regime adjustments (trending market)
    regime_adjustments = {
        'confidence_threshold': 0.6,
        'position_size_multiplier': 1.2,
        'stop_loss_multiplier': 1.1,
        'gap_direction_bias': 0.7
    }
    
    # Run optimization
    print("\nRunning portfolio optimization...")
    optimal_positions = optimizer.optimize_portfolio(
        gap_opportunities,
        magnitude_predictions,
        regime_adjustments
    )
    
    # Display results
    print(f"\nOptimal Portfolio Allocation:")
    print(f"{'Symbol':<10} {'Position Size':>15} {'% of Portfolio':>15}")
    print("-" * 40)
    
    total_allocated = 0
    for symbol, size in sorted(optimal_positions.items(), key=lambda x: x[1], reverse=True):
        pct_of_portfolio = (size / optimizer.portfolio_state.total_value) * 100
        print(f"{symbol:<10} ${size:>14,.2f} {pct_of_portfolio:>14.2f}%")
        total_allocated += size
    
    print("-" * 40)
    print(f"{'Total':<10} ${total_allocated:>14,.2f} {(total_allocated/optimizer.portfolio_state.total_value)*100:>14.2f}%")
    
    # Show risk metrics
    print(f"\nRisk Analysis:")
    risk_dashboard = optimizer.get_risk_dashboard()
    
    # Simulate execution
    optimizer.update_portfolio_state(optimal_positions)
    risk_dashboard = optimizer.get_risk_dashboard()
    
    print(f"  Total Positions: {risk_dashboard['total_positions']}")
    print(f"  Portfolio Heat: {risk_dashboard['portfolio_heat']:.2%}")
    print(f"  Cash Remaining: {risk_dashboard['cash_percentage']:.1f}%")
    print(f"  Largest Position: {risk_dashboard['largest_position']:.2%}")
    print(f"  Sector Concentration: {risk_dashboard['sector_concentration']:.2%}")
    
    # Show sector breakdown
    if risk_dashboard['sector_exposures']:
        print(f"\nSector Exposures:")
        for sector, exposure in sorted(risk_dashboard['sector_exposures'].items(), 
                                     key=lambda x: x[1], reverse=True):
            pct = (exposure / risk_dashboard['total_exposure']) * 100
            print(f"  Sector {sector}: ${exposure:,.2f} ({pct:.1f}%)")
    
    # Show correlation matrix sample
    print(f"\nSample Correlation Matrix (first 5 positions):")
    if len(selected_symbols) >= 5:
        indices = [optimizer.symbol_to_idx[s] for s in selected_symbols[:5]]
        corr_sample = optimizer.correlation_matrix[indices][:, indices]
        
        print("       ", end="")
        for s in selected_symbols[:5]:
            print(f"{s:>8}", end="")
        print()
        
        for i, s in enumerate(selected_symbols[:5]):
            print(f"{s:>7}", end="")
            for j in range(5):
                print(f"{corr_sample[i,j]:>8.2f}", end="")
            print()
