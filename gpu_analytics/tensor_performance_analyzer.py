#!/usr/bin/env python3
"""
GPU-Accelerated Tensor Performance Analyzer
Comprehensive performance analysis using parallel GPU computations for
deep insights into trading system effectiveness.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpu_trading_config import GPUTradingConfig
from database.connection import get_db_manager


@dataclass
class TradeRecord:
    """Complete record of a trade for analysis."""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    position_size: float
    pnl: float
    pnl_percent: float
    gap_size: float
    gap_quality_score: float
    predicted_continuation: float
    predicted_magnitude: float
    actual_magnitude: float
    regime: str
    entry_slippage: float
    exit_slippage: float
    max_adverse_excursion: float
    max_favorable_excursion: float
    time_to_peak: float
    pattern_alerts: List[str]


class GPUTensorPerformanceAnalyzer:
    """Comprehensive performance analysis using GPU acceleration."""
    
    def __init__(self, config: GPUTradingConfig):
        self.config = config
        self.device = config.device
        self.dtype = config.TENSOR_DTYPE
        
        # Database connection for persistence
        self.db_manager = get_db_manager()
        
        # Performance data storage
        self.trade_records: List[TradeRecord] = []
        self.daily_pnl: Dict[datetime, float] = {}
        self.intraday_equity_curve: Dict[datetime, List[float]] = {}
        
        # GPU tensors for analysis
        self._init_analysis_tensors()
        
        # Analysis results cache
        self.analysis_cache = {}
        self.last_analysis_time = None
        
        # Visualization settings
        self.figure_dir = Path('/home/ubuntu/gap_trading_analysis')
        self.figure_dir.mkdir(exist_ok=True)
        
    def _init_analysis_tensors(self):
        """Initialize GPU tensors for performance analysis."""
        max_trades = 10000  # Pre-allocate for efficiency
        
        # Trade metrics tensors
        self.pnl_tensor = torch.zeros(max_trades, device=self.device, dtype=self.dtype)
        self.pnl_percent_tensor = torch.zeros(max_trades, device=self.device, dtype=self.dtype)
        self.holding_time_tensor = torch.zeros(max_trades, device=self.device, dtype=self.dtype)
        
        # Feature tensors
        self.gap_size_tensor = torch.zeros(max_trades, device=self.device, dtype=self.dtype)
        self.gap_quality_tensor = torch.zeros(max_trades, device=self.device, dtype=self.dtype)
        self.prediction_accuracy_tensor = torch.zeros(max_trades, device=self.device, dtype=self.dtype)
        
        # Risk metrics tensors
        self.mae_tensor = torch.zeros(max_trades, device=self.device, dtype=self.dtype)
        self.mfe_tensor = torch.zeros(max_trades, device=self.device, dtype=self.dtype)
        self.risk_reward_tensor = torch.zeros(max_trades, device=self.device, dtype=self.dtype)
        
        # Regime encoding
        self.regime_tensor = torch.zeros(max_trades, device=self.device, dtype=torch.long)
        
        # Current trade count
        self.n_trades = 0
        
    def add_trade(self, trade: TradeRecord):
        """Add trade record for analysis and persist to database."""
        self.trade_records.append(trade)
        
        # Persist to database first
        try:
            self.db_manager.insert_trade(
                trade_id=trade.trade_id,
                symbol=trade.symbol,
                entry_time=trade.entry_time,
                exit_time=trade.exit_time,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                quantity=trade.quantity,
                side='long',  # Assuming long positions for gap trading
                pnl=trade.pnl,
                commission=0.0,  # Add commission if available
                slippage_bps=(trade.entry_slippage + trade.exit_slippage) * 100,  # Convert to bps
                strategy='gap_trading',
                metadata={
                    'gap_size': trade.gap_size,
                    'gap_quality_score': trade.gap_quality_score,
                    'predicted_continuation': trade.predicted_continuation,
                    'predicted_magnitude': trade.predicted_magnitude,
                    'actual_magnitude': trade.actual_magnitude,
                    'regime': trade.regime,
                    'max_adverse_excursion': trade.max_adverse_excursion,
                    'max_favorable_excursion': trade.max_favorable_excursion,
                    'time_to_peak': trade.time_to_peak,
                    'pattern_alerts': trade.pattern_alerts
                }
            )
        except Exception as e:
            print(f"Error persisting trade to database: {e}")
        
        # Update GPU tensors
        idx = self.n_trades
        
        self.pnl_tensor[idx] = trade.pnl
        self.pnl_percent_tensor[idx] = trade.pnl_percent
        self.holding_time_tensor[idx] = (trade.exit_time - trade.entry_time).total_seconds() / 60
        
        self.gap_size_tensor[idx] = abs(trade.gap_size)
        self.gap_quality_tensor[idx] = trade.gap_quality_score
        self.prediction_accuracy_tensor[idx] = 1.0 - abs(trade.predicted_magnitude - trade.actual_magnitude) / (abs(trade.actual_magnitude) + 1)
        
        self.mae_tensor[idx] = trade.max_adverse_excursion
        self.mfe_tensor[idx] = trade.max_favorable_excursion
        
        if trade.max_adverse_excursion > 0:
            self.risk_reward_tensor[idx] = trade.max_favorable_excursion / trade.max_adverse_excursion
        else:
            self.risk_reward_tensor[idx] = trade.max_favorable_excursion
        
        # Encode regime
        regime_map = {'TRENDING': 0, 'VOLATILE': 1, 'QUIET': 2}
        self.regime_tensor[idx] = regime_map.get(trade.regime, 0)
        
        self.n_trades += 1
        
        # Update daily P&L
        trade_date = trade.exit_time.date()
        self.daily_pnl[trade_date] = self.daily_pnl.get(trade_date, 0) + trade.pnl
        
        # Clear cache as data has changed
        self.analysis_cache.clear()
    
    def analyze_performance(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Comprehensive performance analysis using GPU computations."""
        # Check cache
        if not force_refresh and self.analysis_cache and self.last_analysis_time:
            if (datetime.now() - self.last_analysis_time).total_seconds() < 300:  # 5 min cache
                return self.analysis_cache
        
        if self.n_trades == 0:
            return {'status': 'No trades to analyze'}
        
        print(f"Analyzing performance for {self.n_trades} trades...")
        
        # Get valid data slice
        pnl = self.pnl_tensor[:self.n_trades]
        pnl_pct = self.pnl_percent_tensor[:self.n_trades]
        holding_times = self.holding_time_tensor[:self.n_trades]
        
        # Basic statistics
        basic_stats = self._calculate_basic_statistics(pnl, pnl_pct)
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(pnl, pnl_pct)
        
        # Win/loss analysis
        win_loss_analysis = self._analyze_wins_losses(pnl, pnl_pct)
        
        # Time-based analysis
        time_analysis = self._analyze_time_patterns()
        
        # Feature correlation analysis
        feature_analysis = self._analyze_feature_correlations()
        
        # Regime performance
        regime_analysis = self._analyze_regime_performance()
        
        # Monte Carlo simulation
        monte_carlo = self._run_monte_carlo_simulation(pnl_pct)
        
        # Edge analysis
        edge_analysis = self._analyze_trading_edge()
        
        # Compile results
        analysis_results = {
            'basic_statistics': basic_stats,
            'risk_metrics': risk_metrics,
            'win_loss_analysis': win_loss_analysis,
            'time_analysis': time_analysis,
            'feature_analysis': feature_analysis,
            'regime_analysis': regime_analysis,
            'monte_carlo': monte_carlo,
            'edge_analysis': edge_analysis,
            'analysis_timestamp': datetime.now()
        }
        
        # Cache results
        self.analysis_cache = analysis_results
        self.last_analysis_time = datetime.now()
        
        return analysis_results
    
    def _calculate_basic_statistics(self, pnl: torch.Tensor, pnl_pct: torch.Tensor) -> Dict:
        """Calculate basic performance statistics on GPU."""
        return {
            'total_trades': self.n_trades,
            'total_pnl': pnl.sum().item(),
            'average_pnl': pnl.mean().item(),
            'average_pnl_percent': pnl_pct.mean().item(),
            'pnl_std': pnl.std().item(),
            'pnl_pct_std': pnl_pct.std().item(),
            'best_trade': pnl.max().item(),
            'worst_trade': pnl.min().item(),
            'median_pnl': pnl.median().item(),
            'pnl_skewness': self._calculate_skewness(pnl),
            'pnl_kurtosis': self._calculate_kurtosis(pnl)
        }
    
    def _calculate_risk_metrics(self, pnl: torch.Tensor, pnl_pct: torch.Tensor) -> Dict:
        """Calculate risk-adjusted performance metrics."""
        # Daily returns for Sharpe calculation
        daily_returns = self._get_daily_returns()
        
        # Sharpe Ratio (annualized)
        if len(daily_returns) > 1:
            daily_returns_tensor = torch.tensor(daily_returns, device=self.device, dtype=self.dtype)
            sharpe_ratio = (daily_returns_tensor.mean() / (daily_returns_tensor.std() + 1e-8)) * torch.sqrt(torch.tensor(252.0))
        else:
            sharpe_ratio = torch.tensor(0.0)
        
        # Sortino Ratio (downside deviation)
        negative_returns = pnl_pct[pnl_pct < 0]
        if len(negative_returns) > 0:
            downside_deviation = negative_returns.std()
            sortino_ratio = (pnl_pct.mean() / (downside_deviation + 1e-8)) * torch.sqrt(torch.tensor(252.0))
        else:
            sortino_ratio = pnl_pct.mean() * torch.sqrt(torch.tensor(252.0))
        
        # Maximum Drawdown
        cumulative_pnl = pnl.cumsum(0)
        running_max = cumulative_pnl.cummax(0)[0]
        drawdown = running_max - cumulative_pnl
        max_drawdown = drawdown.max().item()
        max_drawdown_pct = (max_drawdown / (running_max[drawdown.argmax()] + 1e-8) * 100).item()
        
        # Calmar Ratio
        annual_return = pnl.sum().item() * (252 / max(len(self.daily_pnl), 1))
        calmar_ratio = annual_return / (max_drawdown + 1e-8)
        
        # Value at Risk (95%)
        var_95 = torch.quantile(pnl, 0.05).item()
        cvar_95 = pnl[pnl <= var_95].mean().item() if (pnl <= var_95).any() else var_95
        
        return {
            'sharpe_ratio': sharpe_ratio.item(),
            'sortino_ratio': sortino_ratio.item(),
            'max_drawdown': max_drawdown,
            'max_drawdown_percent': max_drawdown_pct,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'profit_factor': self._calculate_profit_factor(pnl),
            'recovery_factor': pnl.sum().item() / (max_drawdown + 1e-8),
            'risk_reward_ratio': self.risk_reward_tensor[:self.n_trades].mean().item()
        }
    
    def _analyze_wins_losses(self, pnl: torch.Tensor, pnl_pct: torch.Tensor) -> Dict:
        """Analyze winning and losing trades."""
        wins = pnl > 0
        losses = pnl < 0
        
        win_count = wins.sum().item()
        loss_count = losses.sum().item()
        
        return {
            'win_rate': win_count / (self.n_trades + 1e-8),
            'win_count': win_count,
            'loss_count': loss_count,
            'average_win': pnl[wins].mean().item() if wins.any() else 0,
            'average_loss': pnl[losses].mean().item() if losses.any() else 0,
            'average_win_percent': pnl_pct[wins].mean().item() if wins.any() else 0,
            'average_loss_percent': pnl_pct[losses].mean().item() if losses.any() else 0,
            'largest_win': pnl[wins].max().item() if wins.any() else 0,
            'largest_loss': pnl[losses].min().item() if losses.any() else 0,
            'win_loss_ratio': abs(pnl[wins].mean() / (pnl[losses].mean() + 1e-8)) if losses.any() else float('inf'),
            'average_holding_time_wins': self.holding_time_tensor[:self.n_trades][wins].mean().item() if wins.any() else 0,
            'average_holding_time_losses': self.holding_time_tensor[:self.n_trades][losses].mean().item() if losses.any() else 0,
            'consecutive_wins_max': self._max_consecutive(wins),
            'consecutive_losses_max': self._max_consecutive(losses)
        }
    
    def _analyze_time_patterns(self) -> Dict:
        """Analyze performance patterns by time."""
        if not self.trade_records:
            return {}
        
        # Group by hour of day
        hourly_pnl = {}
        hourly_counts = {}
        
        for trade in self.trade_records:
            hour = trade.entry_time.hour
            hourly_pnl[hour] = hourly_pnl.get(hour, 0) + trade.pnl
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
        
        # Best/worst hours
        if hourly_pnl:
            best_hour = max(hourly_pnl.items(), key=lambda x: x[1])[0]
            worst_hour = min(hourly_pnl.items(), key=lambda x: x[1])[0]
        else:
            best_hour = worst_hour = None
        
        # Day of week analysis
        dow_pnl = {}
        for trade in self.trade_records:
            dow = trade.entry_time.strftime('%A')
            dow_pnl[dow] = dow_pnl.get(dow, 0) + trade.pnl
        
        return {
            'hourly_pnl': hourly_pnl,
            'hourly_trade_counts': hourly_counts,
            'best_hour': best_hour,
            'worst_hour': worst_hour,
            'day_of_week_pnl': dow_pnl,
            'average_holding_time': self.holding_time_tensor[:self.n_trades].mean().item(),
            'holding_time_std': self.holding_time_tensor[:self.n_trades].std().item()
        }
    
    def _analyze_feature_correlations(self) -> Dict:
        """Analyze correlations between features and outcomes."""
        # Stack features
        features = torch.stack([
            self.gap_size_tensor[:self.n_trades],
            self.gap_quality_tensor[:self.n_trades],
            self.prediction_accuracy_tensor[:self.n_trades],
            self.mae_tensor[:self.n_trades],
            self.mfe_tensor[:self.n_trades]
        ], dim=1)
        
        # Calculate correlations with P&L
        pnl = self.pnl_percent_tensor[:self.n_trades]
        
        correlations = {}
        feature_names = ['gap_size', 'gap_quality', 'prediction_accuracy', 'max_adverse', 'max_favorable']
        
        for i, name in enumerate(feature_names):
            if features[:, i].std() > 0:
                corr = torch.corrcoef(torch.stack([features[:, i], pnl]))[0, 1]
                correlations[name] = corr.item()
            else:
                correlations[name] = 0.0
        
        # Feature importance using simple regression
        if self.n_trades > 10:
            # Normalize features
            features_norm = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
            
            # Simple linear regression
            X = torch.cat([features_norm, torch.ones(self.n_trades, 1, device=self.device, dtype=self.dtype)], dim=1)
            coefficients = torch.linalg.lstsq(X, pnl).solution
            
            feature_importance = {
                name: abs(coefficients[i].item())
                for i, name in enumerate(feature_names)
            }
        else:
            feature_importance = {name: 0.0 for name in feature_names}
        
        return {
            'correlations': correlations,
            'feature_importance': feature_importance,
            'most_predictive_feature': max(correlations.items(), key=lambda x: abs(x[1]))[0] if correlations else None
        }
    
    def _analyze_regime_performance(self) -> Dict:
        """Analyze performance by market regime."""
        regime_names = ['TRENDING', 'VOLATILE', 'QUIET']
        regime_stats = {}
        
        for i, regime in enumerate(regime_names):
            mask = self.regime_tensor[:self.n_trades] == i
            
            if mask.any():
                regime_pnl = self.pnl_tensor[:self.n_trades][mask]
                regime_pnl_pct = self.pnl_percent_tensor[:self.n_trades][mask]
                
                regime_stats[regime] = {
                    'count': mask.sum().item(),
                    'total_pnl': regime_pnl.sum().item(),
                    'average_pnl': regime_pnl.mean().item(),
                    'win_rate': (regime_pnl > 0).float().mean().item(),
                    'sharpe': (regime_pnl_pct.mean() / (regime_pnl_pct.std() + 1e-8) * torch.sqrt(torch.tensor(252.0))).item()
                }
            else:
                regime_stats[regime] = {
                    'count': 0,
                    'total_pnl': 0,
                    'average_pnl': 0,
                    'win_rate': 0,
                    'sharpe': 0
                }
        
        # Best regime
        best_regime = max(regime_stats.items(), 
                         key=lambda x: x[1]['total_pnl'] if x[1]['count'] > 0 else float('-inf'))
        
        return {
            'regime_statistics': regime_stats,
            'best_regime': best_regime[0],
            'regime_distribution': {
                regime: stats['count'] / self.n_trades 
                for regime, stats in regime_stats.items()
            }
        }
    
    def _run_monte_carlo_simulation(self, pnl_pct: torch.Tensor, n_simulations: int = 1000) -> Dict:
        """Run Monte Carlo simulation for confidence intervals."""
        n_trades = len(pnl_pct)
        
        # Generate random paths
        simulation_results = torch.zeros(n_simulations, device=self.device, dtype=self.dtype)
        
        for i in range(n_simulations):
            # Resample with replacement
            indices = torch.randint(0, n_trades, (n_trades,), device=self.device)
            simulated_returns = pnl_pct[indices]
            simulation_results[i] = simulated_returns.sum()
        
        # Calculate confidence intervals
        percentiles = [5, 25, 50, 75, 95]
        ci_values = {}
        
        for p in percentiles:
            ci_values[f'p{p}'] = torch.quantile(simulation_results, p/100).item()
        
        # Probability of profit
        prob_profit = (simulation_results > 0).float().mean().item()
        
        # Expected shortfall
        var_5 = ci_values['p5']
        expected_shortfall = simulation_results[simulation_results <= var_5].mean().item()
        
        return {
            'confidence_intervals': ci_values,
            'probability_of_profit': prob_profit,
            'expected_return': simulation_results.mean().item(),
            'return_std': simulation_results.std().item(),
            'expected_shortfall': expected_shortfall,
            'best_case_scenario': simulation_results.max().item(),
            'worst_case_scenario': simulation_results.min().item()
        }
    
    def _analyze_trading_edge(self) -> Dict:
        """Analyze and quantify trading edge."""
        if self.n_trades < 20:
            return {'status': 'Insufficient data for edge analysis'}
        
        # Calculate expectancy
        pnl = self.pnl_tensor[:self.n_trades]
        win_rate = (pnl > 0).float().mean()
        avg_win = pnl[pnl > 0].mean() if (pnl > 0).any() else torch.tensor(0.0)
        avg_loss = abs(pnl[pnl < 0].mean()) if (pnl < 0).any() else torch.tensor(0.0)
        
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
        
        # Kelly Criterion
        if avg_loss > 0:
            kelly_fraction = (win_rate - (1 - win_rate) / (avg_win / avg_loss))
        else:
            kelly_fraction = win_rate
        
        # Edge persistence (rolling window analysis)
        window_size = min(20, self.n_trades // 5)
        rolling_returns = []
        
        for i in range(window_size, self.n_trades):
            window_pnl = pnl[i-window_size:i]
            rolling_returns.append(window_pnl.mean().item())
        
        if rolling_returns:
            edge_consistency = 1 - (np.std(rolling_returns) / (np.mean(rolling_returns) + 1e-8))
        else:
            edge_consistency = 0
        
        # Statistical significance (t-test)
        t_statistic = (pnl.mean() / (pnl.std() / torch.sqrt(torch.tensor(float(self.n_trades), device=self.device)))).item()
        
        # Rough p-value approximation
        p_value = 2 * (1 - self._normal_cdf(abs(t_statistic)))
        
        return {
            'expectancy': expectancy.item(),
            'expectancy_ratio': (expectancy / (avg_loss + 1e-8)).item(),
            'kelly_fraction': kelly_fraction.item(),
            'suggested_position_size': min(kelly_fraction * 0.25, 0.25).item(),  # Conservative Kelly
            'edge_consistency': edge_consistency,
            't_statistic': t_statistic,
            'p_value': p_value,
            'statistically_significant': p_value < 0.05,
            'minimum_trades_for_confidence': max(30, int(100 / (win_rate.item() + 0.1)))
        }
    
    def generate_performance_report(self, save_path: Optional[Path] = None):
        """Generate comprehensive performance report with visualizations."""
        analysis = self.analyze_performance()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Equity Curve
        ax1 = plt.subplot(4, 2, 1)
        self._plot_equity_curve(ax1)
        
        # 2. Daily Returns Distribution
        ax2 = plt.subplot(4, 2, 2)
        self._plot_returns_distribution(ax2)
        
        # 3. Hourly Performance Heatmap
        ax3 = plt.subplot(4, 2, 3)
        self._plot_hourly_heatmap(ax3, analysis['time_analysis'])
        
        # 4. Win/Loss Analysis
        ax4 = plt.subplot(4, 2, 4)
        self._plot_win_loss_analysis(ax4, analysis['win_loss_analysis'])
        
        # 5. Regime Performance
        ax5 = plt.subplot(4, 2, 5)
        self._plot_regime_performance(ax5, analysis['regime_analysis'])
        
        # 6. Feature Correlations
        ax6 = plt.subplot(4, 2, 6)
        self._plot_feature_correlations(ax6, analysis['feature_analysis'])
        
        # 7. Monte Carlo Results
        ax7 = plt.subplot(4, 2, 7)
        self._plot_monte_carlo(ax7, analysis['monte_carlo'])
        
        # 8. Risk Metrics Summary
        ax8 = plt.subplot(4, 2, 8)
        self._plot_risk_metrics(ax8, analysis['risk_metrics'])
        
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.figure_dir / f'performance_report_{timestamp}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
        
        # Generate text report
        self._generate_text_report(analysis, save_path.parent if save_path else self.figure_dir)
        
        return analysis
    
    def _plot_equity_curve(self, ax):
        """Plot cumulative equity curve."""
        if self.n_trades == 0:
            return
        
        cumulative_pnl = self.pnl_tensor[:self.n_trades].cumsum(0).cpu().numpy()
        trades = np.arange(self.n_trades)
        
        ax.plot(trades, cumulative_pnl, 'b-', linewidth=2)
        ax.fill_between(trades, 0, cumulative_pnl, alpha=0.3)
        
        # Add drawdown shading
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        
        ax2 = ax.twinx()
        ax2.fill_between(trades, 0, -drawdown, color='red', alpha=0.2)
        ax2.set_ylabel('Drawdown ($)', color='red')
        
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Cumulative P&L ($)', color='blue')
        ax.set_title('Equity Curve')
        ax.grid(True, alpha=0.3)
        
    def _plot_returns_distribution(self, ax):
        """Plot returns distribution."""
        if self.n_trades == 0:
            return
        
        returns = self.pnl_percent_tensor[:self.n_trades].cpu().numpy()
        
        ax.hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=returns.mean(), color='green', linestyle='--', label=f'Mean: {returns.mean():.2f}%')
        
        # Add normal distribution overlay
        from scipy import stats
        x = np.linspace(returns.min(), returns.max(), 100)
        ax2 = ax.twinx()
        ax2.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()) * len(returns) * (returns.max() - returns.min()) / 50,
                'r-', alpha=0.5, label='Normal')
        
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Returns Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_hourly_heatmap(self, ax, time_analysis):
        """Plot hourly performance heatmap."""
        if not time_analysis.get('hourly_pnl'):
            return
        
        # Create hourly matrix
        hours = list(range(24))
        hourly_data = [time_analysis['hourly_pnl'].get(h, 0) for h in hours]
        
        # Reshape for heatmap
        data_matrix = np.array(hourly_data).reshape(4, 6)
        
        sns.heatmap(data_matrix, annot=True, fmt='.0f', cmap='RdYlGn', center=0,
                   xticklabels=[f'{h}:00' for h in range(0, 24, 4)],
                   yticklabels=['Night', 'Morning', 'Afternoon', 'Evening'],
                   ax=ax)
        
        ax.set_title('P&L by Hour of Day')
        
    def _plot_win_loss_analysis(self, ax, win_loss):
        """Plot win/loss analysis."""
        categories = ['Wins', 'Losses']
        counts = [win_loss['win_count'], win_loss['loss_count']]
        avg_values = [win_loss['average_win'], abs(win_loss['average_loss'])]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, counts, width, label='Count', color=['green', 'red'], alpha=0.7)
        
        ax2 = ax.twinx()
        ax2.bar(x + width/2, avg_values, width, label='Avg Size ($)', color=['darkgreen', 'darkred'], alpha=0.7)
        
        ax.set_xlabel('Category')
        ax.set_ylabel('Count')
        ax2.set_ylabel('Average Size ($)')
        ax.set_title('Win/Loss Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
    def _plot_regime_performance(self, ax, regime_analysis):
        """Plot performance by regime."""
        regimes = list(regime_analysis['regime_statistics'].keys())
        pnls = [regime_analysis['regime_statistics'][r]['total_pnl'] for r in regimes]
        counts = [regime_analysis['regime_statistics'][r]['count'] for r in regimes]
        
        # Create pie chart
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        wedges, texts, autotexts = ax.pie(counts, labels=regimes, colors=colors,
                                          autopct='%1.1f%%', startangle=90)
        
        # Add P&L annotations
        for i, (regime, pnl) in enumerate(zip(regimes, pnls)):
            ax.text(0.7 * np.cos(np.pi/2 - 2*np.pi*sum(counts[:i+1])/sum(counts)),
                   0.7 * np.sin(np.pi/2 - 2*np.pi*sum(counts[:i+1])/sum(counts)),
                   f'${pnl:,.0f}', ha='center', va='center')
        
        ax.set_title('Performance by Market Regime')
        
    def _plot_feature_correlations(self, ax, feature_analysis):
        """Plot feature correlations."""
        correlations = feature_analysis['correlations']
        
        features = list(correlations.keys())
        values = list(correlations.values())
        
        colors = ['green' if v > 0 else 'red' for v in values]
        
        bars = ax.bar(features, values, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Correlation with P&L')
        ax.set_title('Feature Correlations')
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}', ha='center', va='bottom' if value > 0 else 'top')
        
    def _plot_monte_carlo(self, ax, monte_carlo):
        """Plot Monte Carlo simulation results."""
        # Create box plot data
        ci = monte_carlo['confidence_intervals']
        positions = [1]
        
        ax.boxplot([ci['p50']], positions=positions, widths=0.6,
                  showmeans=True, meanline=True)
        
        # Add confidence intervals
        ax.plot([1, 1], [ci['p5'], ci['p95']], 'b-', linewidth=2)
        ax.plot([0.9, 1.1], [ci['p5'], ci['p5']], 'b-', linewidth=2)
        ax.plot([0.9, 1.1], [ci['p95'], ci['p95']], 'b-', linewidth=2)
        
        # Add annotations
        ax.text(1.2, ci['p50'], f"Median: {ci['p50']:.1f}%", va='center')
        ax.text(1.2, ci['p95'], f"95%: {ci['p95']:.1f}%", va='center')
        ax.text(1.2, ci['p5'], f"5%: {ci['p5']:.1f}%", va='center')
        
        ax.set_ylabel('Expected Total Return (%)')
        ax.set_title(f'Monte Carlo Simulation (n={1000})')
        ax.set_xticklabels([''])
        ax.grid(True, alpha=0.3, axis='y')
        
    def _plot_risk_metrics(self, ax, risk_metrics):
        """Plot risk metrics summary."""
        ax.axis('off')
        
        # Create text summary
        metrics_text = f"""
Risk-Adjusted Performance Metrics

Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}
Sortino Ratio: {risk_metrics['sortino_ratio']:.2f}
Calmar Ratio: {risk_metrics['calmar_ratio']:.2f}

Maximum Drawdown: ${risk_metrics['max_drawdown']:,.2f} ({risk_metrics['max_drawdown_percent']:.1f}%)
Value at Risk (95%): ${risk_metrics['var_95']:,.2f}
Conditional VaR (95%): ${risk_metrics['cvar_95']:,.2f}

Profit Factor: {risk_metrics['profit_factor']:.2f}
Recovery Factor: {risk_metrics['recovery_factor']:.2f}
Risk/Reward Ratio: {risk_metrics['risk_reward_ratio']:.2f}
"""
        
        ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top', fontfamily='monospace')
        
    def _generate_text_report(self, analysis: Dict, save_dir: Path):
        """Generate detailed text report."""
        timestamp = datetime.now()
        
        report = f"""
GPU-ACCELERATED GAP TRADING SYSTEM
PERFORMANCE ANALYSIS REPORT
Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}

EXECUTIVE SUMMARY
-----------------
Total Trades: {analysis['basic_statistics']['total_trades']}
Total P&L: ${analysis['basic_statistics']['total_pnl']:,.2f}
Win Rate: {analysis['win_loss_analysis']['win_rate']:.1%}
Sharpe Ratio: {analysis['risk_metrics']['sharpe_ratio']:.2f}
Maximum Drawdown: ${analysis['risk_metrics']['max_drawdown']:,.2f} ({analysis['risk_metrics']['max_drawdown_percent']:.1f}%)

TRADING EDGE ANALYSIS
--------------------
Expectancy: ${analysis['edge_analysis']['expectancy']:.2f} per trade
Kelly Fraction: {analysis['edge_analysis']['kelly_fraction']:.2%}
Suggested Position Size: {analysis['edge_analysis']['suggested_position_size']:.1%} of capital
Edge Consistency: {analysis['edge_analysis']['edge_consistency']:.2f}
Statistical Significance: {'YES' if analysis['edge_analysis']['statistically_significant'] else 'NO'} (p={analysis['edge_analysis']['p_value']:.4f})

PERFORMANCE BY MARKET REGIME
---------------------------
{self._format_regime_table(analysis['regime_analysis'])}

MONTE CARLO SIMULATION RESULTS
-----------------------------
Expected Return: {analysis['monte_carlo']['expected_return']:.2f}%
Probability of Profit: {analysis['monte_carlo']['probability_of_profit']:.1%}
95% Confidence Interval: [{analysis['monte_carlo']['confidence_intervals']['p5']:.1f}%, {analysis['monte_carlo']['confidence_intervals']['p95']:.1f}%]
Expected Shortfall: {analysis['monte_carlo']['expected_shortfall']:.2f}%

FEATURE IMPORTANCE
-----------------
{self._format_feature_importance(analysis['feature_analysis'])}

RECOMMENDATIONS
--------------
{self._generate_recommendations(analysis)}

{'=' * 80}
End of Report
"""
        
        # Save report
        report_path = save_dir / f'performance_report_{timestamp.strftime("%Y%m%d_%H%M%S")}.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {report_path}")
    
    def _format_regime_table(self, regime_analysis: Dict) -> str:
        """Format regime performance table."""
        lines = []
        lines.append(f"{'Regime':<12} {'Trades':<8} {'Total P&L':<12} {'Win Rate':<10} {'Sharpe':<8}")
        lines.append("-" * 50)
        
        for regime, stats in regime_analysis['regime_statistics'].items():
            lines.append(f"{regime:<12} {stats['count']:<8} ${stats['total_pnl']:<11,.0f} "
                        f"{stats['win_rate']:<9.1%} {stats['sharpe']:<8.2f}")
        
        return "\n".join(lines)
    
    def _format_feature_importance(self, feature_analysis: Dict) -> str:
        """Format feature importance results."""
        lines = []
        
        for feature, correlation in feature_analysis['correlations'].items():
            importance = feature_analysis['feature_importance'].get(feature, 0)
            lines.append(f"{feature:<20} Correlation: {correlation:>6.3f}  Importance: {importance:>6.3f}")
        
        return "\n".join(lines)
    
    def _generate_recommendations(self, analysis: Dict) -> str:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Position sizing recommendation
        if analysis['edge_analysis']['expectancy'] > 0:
            recommendations.append(f"• Maintain position sizing at {analysis['edge_analysis']['suggested_position_size']:.1%} of capital")
        else:
            recommendations.append("• STOP TRADING - No positive expectancy detected")
        
        # Regime-based recommendations
        best_regime = analysis['regime_analysis']['best_regime']
        recommendations.append(f"• Focus trading during {best_regime} market conditions")
        
        # Time-based recommendations
        if 'best_hour' in analysis['time_analysis']:
            recommendations.append(f"• Prioritize entries around {analysis['time_analysis']['best_hour']}:00")
        
        # Feature-based recommendations
        most_predictive = analysis['feature_analysis'].get('most_predictive_feature')
        if most_predictive:
            recommendations.append(f"• Emphasize {most_predictive} in trade selection")
        
        # Risk recommendations
        if analysis['risk_metrics']['max_drawdown_percent'] > 20:
            recommendations.append("• Consider reducing position sizes - drawdown exceeds 20%")
        
        return "\n".join(recommendations)
    
    # Utility methods
    def _calculate_skewness(self, data: torch.Tensor) -> float:
        """Calculate skewness of distribution."""
        mean = data.mean()
        std = data.std()
        if std == 0:
            return 0
        return ((data - mean) ** 3).mean() / (std ** 3)
    
    def _calculate_kurtosis(self, data: torch.Tensor) -> float:
        """Calculate kurtosis of distribution."""
        mean = data.mean()
        std = data.std()
        if std == 0:
            return 0
        return ((data - mean) ** 4).mean() / (std ** 4) - 3
    
    def _calculate_profit_factor(self, pnl: torch.Tensor) -> float:
        """Calculate profit factor."""
        gross_profit = pnl[pnl > 0].sum()
        gross_loss = abs(pnl[pnl < 0].sum())
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
        
        return (gross_profit / gross_loss).item()
    
    def _max_consecutive(self, binary_tensor: torch.Tensor) -> int:
        """Calculate maximum consecutive True values."""
        if not binary_tensor.any():
            return 0
        
        # Convert to numpy for easier manipulation
        binary_array = binary_tensor.cpu().numpy()
        
        max_consecutive = 0
        current_consecutive = 0
        
        for value in binary_array:
            if value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _get_daily_returns(self) -> List[float]:
        """Get daily returns from P&L data."""
        if not self.daily_pnl:
            return []
        
        sorted_dates = sorted(self.daily_pnl.keys())
        return [self.daily_pnl[date] for date in sorted_dates]
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF."""
        return 0.5 * (1 + torch.erf(torch.tensor(x) / torch.sqrt(torch.tensor(2.0)))).item()


# Example usage when file is run directly
if __name__ == "__main__":
    print("GPU-Accelerated Tensor Performance Analyzer Test")
    print("=" * 50)
    
    # Initialize configuration
    config = GPUTradingConfig()
    
    # Create analyzer
    analyzer = GPUTensorPerformanceAnalyzer(config)
    
    # Generate synthetic trades for testing
    print("\nGenerating synthetic trade data...")
    
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(100):
        # Create realistic trade patterns
        win_probability = 0.55  # 55% win rate
        is_win = np.random.random() < win_probability
        
        if is_win:
            pnl = np.random.uniform(50, 500)
            pnl_pct = np.random.uniform(0.5, 3.0)
            mae = np.random.uniform(0, pnl * 0.3)
            mfe = pnl + np.random.uniform(0, pnl * 0.5)
        else:
            pnl = -np.random.uniform(30, 300)
            pnl_pct = -np.random.uniform(0.3, 2.0)
            mae = -pnl + np.random.uniform(0, -pnl * 0.5)
            mfe = np.random.uniform(0, -pnl * 0.3)
        
        trade = TradeRecord(
            trade_id=f"TRADE_{i:04d}",
            symbol=f"SYMBOL_{i % 20}",
            entry_time=base_time + timedelta(days=i//4, hours=9, minutes=30 + (i % 4) * 15),
            exit_time=base_time + timedelta(days=i//4, hours=10, minutes=30 + (i % 4) * 15),
            entry_price=100.0 + np.random.randn() * 10,
            exit_price=100.0 + np.random.randn() * 10 + pnl_pct,
            quantity=100,
            position_size=10000,
            pnl=pnl,
            pnl_percent=pnl_pct,
            gap_size=np.random.uniform(2, 5) * np.random.choice([-1, 1]),
            gap_quality_score=np.random.uniform(0.5, 0.9),
            predicted_continuation=np.random.uniform(0.6, 0.8),
            predicted_magnitude=np.random.uniform(1.5, 3.5),
            actual_magnitude=pnl_pct,
            regime=np.random.choice(['TRENDING', 'VOLATILE', 'QUIET']),
            entry_slippage=np.random.uniform(0, 0.1),
            exit_slippage=np.random.uniform(0, 0.15),
            max_adverse_excursion=mae,
            max_favorable_excursion=mfe,
            time_to_peak=np.random.uniform(5, 45),
            pattern_alerts=[]
        )
        
        analyzer.add_trade(trade)
    
    # Run analysis
    print("\nRunning comprehensive performance analysis...")
    analysis_results = analyzer.analyze_performance()
    
    # Display key metrics
    print(f"\nKey Performance Metrics:")
    print("-" * 50)
    
    basic = analysis_results['basic_statistics']
    print(f"Total Trades: {basic['total_trades']}")
    print(f"Total P&L: ${basic['total_pnl']:,.2f}")
    print(f"Average P&L: ${basic['average_pnl']:.2f}")
    print(f"Best Trade: ${basic['best_trade']:.2f}")
    print(f"Worst Trade: ${basic['worst_trade']:.2f}")
    
    print(f"\nRisk Metrics:")
    risk = analysis_results['risk_metrics']
    print(f"Sharpe Ratio: {risk['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {risk['sortino_ratio']:.2f}")
    print(f"Maximum Drawdown: ${risk['max_drawdown']:.2f} ({risk['max_drawdown_percent']:.1f}%)")
    print(f"Profit Factor: {risk['profit_factor']:.2f}")
    
    print(f"\nWin/Loss Analysis:")
    winloss = analysis_results['win_loss_analysis']
    print(f"Win Rate: {winloss['win_rate']:.1%}")
    print(f"Average Win: ${winloss['average_win']:.2f}")
    print(f"Average Loss: ${winloss['average_loss']:.2f}")
    print(f"Win/Loss Ratio: {winloss['win_loss_ratio']:.2f}")
    
    print(f"\nTrading Edge:")
    edge = analysis_results['edge_analysis']
    print(f"Expectancy: ${edge['expectancy']:.2f}")
    print(f"Kelly Fraction: {edge['kelly_fraction']:.2%}")
    print(f"Suggested Position Size: {edge['suggested_position_size']:.1%}")
    print(f"Statistically Significant: {edge['statistically_significant']}")
    
    # Generate visual report
    print("\nGenerating performance report...")
    analyzer.generate_performance_report()
    
    print(f"\nAnalysis complete! Reports saved to {analyzer.figure_dir}")