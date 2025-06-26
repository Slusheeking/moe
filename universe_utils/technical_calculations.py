#!/usr/bin/env python3
"""
Technical calculations for the Enhanced Universe Updater.
Contains GPU-optimized calculations and technical analysis functions.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# GPU Acceleration Imports
try:
    import cupy as cp
    import cudf
    import cuml
    from cuml.preprocessing import StandardScaler as CuStandardScaler
    from cuml.linear_model import LinearRegression as CuLinearRegression
    from numba import cuda, jit, njit, prange
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator
    
    # Configure RMM memory pool for GPU memory management
    # Use smaller pool to coexist with FinGPT (~13GB)
    try:
        # Check available GPU memory first
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            free_memory = gpu_memory - reserved
            
            # Only use RMM if sufficient memory and no large PyTorch allocations
            if free_memory > 15.0 and allocated < 1.0:
                rmm.reinitialize(
                    managed_memory=True,
                    pool_allocator=True,
                    initial_pool_size=2**27,  # 128MB initial (reduced)
                    maximum_pool_size=2**29   # 512MB maximum (reduced)
                )
                cp.cuda.set_allocator(rmm_cupy_allocator)
                GPU_AVAILABLE = True
                logger.info(f"🚀 GPU Acceleration ENABLED: RAPIDS/CuPy/Numba (Free: {free_memory:.1f}GB)")
            else:
                logger.warning(f"⚠️ Insufficient GPU memory for RMM ({free_memory:.1f}GB free, {allocated:.1f}GB allocated)")
                GPU_AVAILABLE = False
        else:
            GPU_AVAILABLE = False
    except Exception as e:
        logger.warning(f"Could not configure RMM memory pool: {e}")
        GPU_AVAILABLE = False
        
except ImportError as e:
    logger.warning(f"⚠️  GPU libraries not available: {e}")
    GPU_AVAILABLE = False
    # Fallback imports
    import numpy as cp  # Use numpy as cupy
    
    # Create dummy decorators if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def prange(*args, **kwargs):
        return range(*args, **kwargs)


# GPU-Optimized Numba Kernels
@njit(parallel=True, fastmath=True, cache=True)
def calculate_gaps_vectorized(opens: np.ndarray, closes_prev: np.ndarray) -> np.ndarray:
    """Ultra-fast gap calculation using Numba with parallel processing."""
    n = len(opens)
    if n == 0 or len(closes_prev) == 0:
        return np.empty(0, dtype=np.float32)
    
    # Ensure arrays are same length
    min_len = min(n, len(closes_prev))
    gaps = np.empty(min_len, dtype=np.float32)
    
    for i in prange(min_len):
        if i < len(closes_prev) and i < len(opens):
            if closes_prev[i] != 0 and not np.isnan(closes_prev[i]) and not np.isnan(opens[i]):
                gaps[i] = ((opens[i] - closes_prev[i]) / closes_prev[i]) * 100.0
            else:
                gaps[i] = 0.0
        else:
            gaps[i] = 0.0
    
    return gaps


@njit(parallel=True, fastmath=True, cache=True)
def calculate_returns_vectorized(prices: np.ndarray) -> np.ndarray:
    """Ultra-fast returns calculation using Numba."""
    n = len(prices)
    if n <= 1:
        return np.empty(0, dtype=np.float32)
    
    returns = np.empty(n-1, dtype=np.float32)
    
    for i in prange(n-1):
        if i < len(prices) - 1 and i + 1 < len(prices):
            if prices[i] != 0 and not np.isnan(prices[i]) and not np.isnan(prices[i+1]):
                returns[i] = (prices[i+1] - prices[i]) / prices[i]
            else:
                returns[i] = 0.0
        else:
            returns[i] = 0.0
    
    return returns


@njit(parallel=True, fastmath=True, cache=True)
def calculate_true_range_vectorized(highs: np.ndarray, lows: np.ndarray,
                                  closes_prev: np.ndarray) -> np.ndarray:
    """Ultra-fast True Range calculation using Numba."""
    n = len(highs)
    if n == 0 or len(lows) == 0 or len(closes_prev) == 0:
        return np.empty(0, dtype=np.float32)
    
    # Ensure all arrays are same length
    min_len = min(n, len(lows), len(closes_prev))
    tr = np.empty(min_len, dtype=np.float32)
    
    for i in prange(min_len):
        if i < len(highs) and i < len(lows):
            hl = highs[i] - lows[i]
            if i > 0 and i-1 < len(closes_prev) and not np.isnan(closes_prev[i-1]) and not np.isnan(highs[i]) and not np.isnan(lows[i]):
                hc = abs(highs[i] - closes_prev[i-1])
                lc = abs(lows[i] - closes_prev[i-1])
                tr[i] = max(hl, max(hc, lc))
            else:
                tr[i] = hl if not np.isnan(hl) else 0.0
        else:
            tr[i] = 0.0
    
    return tr


@njit(parallel=True, fastmath=True, cache=True)
def find_support_resistance_vectorized(highs: np.ndarray, lows: np.ndarray,
                                     window: int = 5) -> Tuple[List[float], List[float]]:
    """Ultra-fast support/resistance detection using Numba."""
    n = len(highs)
    if n == 0 or len(lows) == 0 or n <= 2 * window:
        # Return typed empty lists for Numba
        return [0.0][:0], [0.0][:0]
    
    # Ensure arrays are same length
    min_len = min(n, len(lows))
    # Initialize with typed lists for Numba
    resistance_levels = [0.0][:0]  # Empty float list
    support_levels = [0.0][:0]     # Empty float list
    
    for i in range(window, min_len - window):
        if i >= len(highs) or i >= len(lows):
            continue
            
        # Check for resistance (local maxima)
        is_resistance = True
        current_high = highs[i]
        
        if np.isnan(current_high):
            continue
        
        for j in range(max(0, i - window), min(min_len, i + window + 1)):
            if j != i and j < len(highs) and not np.isnan(highs[j]) and highs[j] >= current_high:
                is_resistance = False
                break
        
        if is_resistance:
            resistance_levels.append(float(current_high))
        
        # Check for support (local minima)
        is_support = True
        current_low = lows[i]
        
        if np.isnan(current_low):
            continue
        
        for j in range(max(0, i - window), min(min_len, i + window + 1)):
            if j != i and j < len(lows) and not np.isnan(lows[j]) and lows[j] <= current_low:
                is_support = False
                break
        
        if is_support:
            support_levels.append(float(current_low))
    
    return resistance_levels, support_levels


def calculate_volatility_profile(df: pd.DataFrame, 
                               period: int = 30) -> Tuple[float, float]:
    """
    Calculate realized volatility over different periods.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC data
    period : int
        Number of days for volatility calculation
        
    Returns:
    --------
    Tuple[float, float]
        (realized_volatility, volatility_ratio)
    """
    if len(df) < period:
        return 0.0, 1.0
    
    try:
        # Calculate returns
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < period:
            return 0.0, 1.0
        
        # Calculate realized volatility (annualized)
        realized_vol = returns.tail(period).std() * np.sqrt(252) * 100
        
        # Calculate volatility ratio (recent vs historical)
        if len(returns) >= period * 2:
            recent_vol = returns.tail(period).std()
            historical_vol = returns.tail(period * 2).head(period).std()
            
            if historical_vol > 0:
                vol_ratio = recent_vol / historical_vol
            else:
                vol_ratio = 1.0
        else:
            vol_ratio = 1.0
        
        return float(realized_vol), float(vol_ratio)
        
    except Exception as e:
        logger.warning(f"Error calculating volatility profile: {e}")
        return 0.0, 1.0


def calculate_trend_metrics(df: pd.DataFrame, 
                          lookback: int = 90) -> Dict[str, float]:
    """
    Calculate comprehensive trend metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC data
    lookback : int
        Number of days for trend analysis
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing trend metrics
    """
    metrics = {
        'trend_strength': 0.0,
        'trend_direction': 0.0,
        'trend_consistency': 0.0,
        'momentum_score': 0.0
    }
    
    if len(df) < lookback:
        return metrics
    
    try:
        # Get recent data
        recent_data = df.tail(lookback)
        prices = recent_data['close'].values
        
        # Linear regression for trend strength
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        
        # Trend strength (R-squared)
        metrics['trend_strength'] = float(abs(r_value))
        
        # Trend direction (normalized slope)
        avg_price = prices.mean()
        if avg_price > 0:
            metrics['trend_direction'] = float(slope / avg_price)
        
        # Trend consistency (percentage of days moving in trend direction)
        returns = recent_data['close'].pct_change().dropna()
        if len(returns) > 0:
            if slope > 0:
                consistent_days = (returns > 0).sum()
            else:
                consistent_days = (returns < 0).sum()
            
            metrics['trend_consistency'] = float(consistent_days / len(returns))
        
        # Momentum score (recent performance vs historical)
        if len(df) >= lookback * 2:
            recent_return = (prices[-1] / prices[0]) - 1
            historical_prices = df.tail(lookback * 2).head(lookback)['close'].values
            historical_return = (historical_prices[-1] / historical_prices[0]) - 1
            
            if abs(historical_return) > 0:
                metrics['momentum_score'] = float(recent_return / abs(historical_return))
            else:
                metrics['momentum_score'] = float(recent_return * 10)  # Scale up if no historical movement
        
        return metrics
        
    except Exception as e:
        logger.warning(f"Error calculating trend metrics: {e}")
        return metrics


def calculate_market_microstructure(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate market microstructure metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC and volume data
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing microstructure metrics
    """
    metrics = {
        'avg_spread_estimate': 0.0,
        'price_efficiency': 0.0,
        'volume_concentration': 0.0,
        'intraday_volatility': 0.0,
        'tick_size_ratio': 0.0
    }
    
    if len(df) < 20:
        return metrics
    
    try:
        # Estimate average spread from high-low range
        intraday_range = ((df['high'] - df['low']) / df['close']).tail(90)
        metrics['avg_spread_estimate'] = float(intraday_range.mean() * 10000 / 4)  # Convert to bps
        
        # Price efficiency (how quickly price incorporates information)
        # Measured by autocorrelation of returns
        returns = df['close'].pct_change().dropna()
        if len(returns) > 20:
            autocorr = returns.autocorr(lag=1)
            metrics['price_efficiency'] = float(1 - abs(autocorr))
        
        # Volume concentration (Herfindahl index approximation)
        volumes = df['volume'].tail(90)
        if volumes.sum() > 0:
            volume_shares = volumes / volumes.sum()
            metrics['volume_concentration'] = float(np.sum(volume_shares ** 2))
        
        # Intraday volatility (Parkinson estimator)
        log_hl = np.log(df['high'] / df['low'])
        metrics['intraday_volatility'] = float(
            np.sqrt(np.mean(log_hl.tail(30) ** 2) / (4 * np.log(2))) * np.sqrt(252) * 100
        )
        
        # Tick size ratio (price discreteness)
        price_changes = df['close'].diff().dropna()
        non_zero_changes = price_changes[price_changes != 0]
        if len(non_zero_changes) > 0:
            min_tick = abs(non_zero_changes).min()
            avg_price = df['close'].mean()
            if avg_price > 0:
                metrics['tick_size_ratio'] = float(min_tick / avg_price * 10000)  # In bps
        
        return metrics
        
    except Exception as e:
        logger.warning(f"Error calculating market microstructure: {e}")
        return metrics


def calculate_regime_indicators(df: pd.DataFrame, 
                              market_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """
    Calculate market regime indicators.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Stock DataFrame with OHLC data
    market_data : Optional[pd.DataFrame]
        Market benchmark data (e.g., SPY)
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing regime indicators
    """
    indicators = {
        'volatility_regime': 0.5,  # 0=low vol, 1=high vol
        'trend_regime': 0.5,       # 0=downtrend, 1=uptrend
        'correlation_regime': 0.5,  # 0=decorrelated, 1=highly correlated
        'liquidity_regime': 0.5,    # 0=illiquid, 1=liquid
        'regime_stability': 0.5     # 0=unstable, 1=stable
    }
    
    if len(df) < 60:
        return indicators
    
    try:
        returns = df['close'].pct_change().dropna()
        
        # Volatility regime
        current_vol = returns.tail(20).std()
        historical_vol = returns.tail(90).std()
        if historical_vol > 0:
            vol_percentile = stats.percentileofscore(
                returns.rolling(20).std().dropna(), 
                current_vol
            ) / 100
            indicators['volatility_regime'] = float(vol_percentile)
        
        # Trend regime
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        if len(sma_20) > 0 and len(sma_50) > 0:
            current_price = df['close'].iloc[-1]
            if sma_20.iloc[-1] > sma_50.iloc[-1] and current_price > sma_20.iloc[-1]:
                indicators['trend_regime'] = 0.8
            elif sma_20.iloc[-1] < sma_50.iloc[-1] and current_price < sma_20.iloc[-1]:
                indicators['trend_regime'] = 0.2
            else:
                indicators['trend_regime'] = 0.5
        
        # Correlation regime (if market data provided)
        if market_data is not None and len(market_data) > 30:
            market_returns = market_data['close'].pct_change().dropna()
            
            # Align dates
            common_dates = returns.index.intersection(market_returns.index)
            if len(common_dates) > 30:
                stock_rets = returns.loc[common_dates].tail(30)
                market_rets = market_returns.loc[common_dates].tail(30)
                
                correlation = stock_rets.corr(market_rets)
                indicators['correlation_regime'] = float(abs(correlation))
        
        # Liquidity regime (based on volume)
        current_volume = df['volume'].tail(20).mean()
        historical_volume = df['volume'].tail(90).mean()
        if historical_volume > 0:
            vol_ratio = current_volume / historical_volume
            indicators['liquidity_regime'] = float(min(1.0, vol_ratio))
        
        # Regime stability (how stable are the regimes)
        # Calculate rolling standard deviation of regime indicators
        vol_stability = 1 - returns.rolling(20).std().tail(30).std()
        indicators['regime_stability'] = float(max(0, min(1, vol_stability * 10)))
        
        return indicators
        
    except Exception as e:
        logger.warning(f"Error calculating regime indicators: {e}")
        return indicators


def calculate_gap_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive gap pattern analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC data
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing gap pattern metrics
    """
    patterns = {
        'gap_types': {'up': 0, 'down': 0, 'inside': 0},
        'gap_fills': {'same_day': 0, 'next_day': 0, 'within_week': 0, 'unfilled': 0},
        'gap_continuations': 0,
        'gap_reversals': 0,
        'average_gap_size': 0.0,
        'gap_volatility': 0.0,
        'gap_seasonality': {}
    }
    
    if len(df) < 30:
        return patterns
    
    try:
        # Calculate gaps
        opens = df['open'].values
        closes_prev = df['close'].shift(1).fillna(method='bfill').values
        gaps = calculate_gaps_vectorized(opens, closes_prev)
        
        # Convert to series for easier manipulation
        gaps_series = pd.Series(gaps[1:], index=df.index[1:])
        
        # Filter significant gaps (>1%)
        significant_gaps = gaps_series[np.abs(gaps_series) >= 1.0]
        
        if len(significant_gaps) == 0:
            return patterns
        
        # Classify gap types
        for gap_idx, gap_value in significant_gaps.items():
            idx_pos = df.index.get_loc(gap_idx)
            
            if gap_value > 0:
                patterns['gap_types']['up'] += 1
            else:
                patterns['gap_types']['down'] += 1
            
            # Check gap fills
            gap_filled = False
            gap_day_open = df.iloc[idx_pos]['open']
            prev_close = df.iloc[idx_pos - 1]['close']
            
            # Same day fill
            if gap_value > 0:  # Gap up
                if df.iloc[idx_pos]['low'] <= prev_close:
                    patterns['gap_fills']['same_day'] += 1
                    gap_filled = True
            else:  # Gap down
                if df.iloc[idx_pos]['high'] >= prev_close:
                    patterns['gap_fills']['same_day'] += 1
                    gap_filled = True
            
            # Next day fill
            if not gap_filled and idx_pos + 1 < len(df):
                if gap_value > 0:
                    if df.iloc[idx_pos + 1]['low'] <= prev_close:
                        patterns['gap_fills']['next_day'] += 1
                        gap_filled = True
                else:
                    if df.iloc[idx_pos + 1]['high'] >= prev_close:
                        patterns['gap_fills']['next_day'] += 1
                        gap_filled = True
            
            # Within week fill
            if not gap_filled and idx_pos + 5 < len(df):
                week_data = df.iloc[idx_pos + 1:idx_pos + 6]
                if gap_value > 0:
                    if week_data['low'].min() <= prev_close:
                        patterns['gap_fills']['within_week'] += 1
                        gap_filled = True
                else:
                    if week_data['high'].max() >= prev_close:
                        patterns['gap_fills']['within_week'] += 1
                        gap_filled = True
            
            if not gap_filled:
                patterns['gap_fills']['unfilled'] += 1
            
            # Check continuation vs reversal
            day_return = (df.iloc[idx_pos]['close'] - gap_day_open) / gap_day_open
            
            if (gap_value > 0 and day_return > 0) or (gap_value < 0 and day_return < 0):
                patterns['gap_continuations'] += 1
            else:
                patterns['gap_reversals'] += 1
        
        # Calculate statistics
        patterns['average_gap_size'] = float(np.mean(np.abs(significant_gaps)))
        patterns['gap_volatility'] = float(np.std(significant_gaps))
        
        # Gap seasonality by day of week
        if hasattr(df.index, 'day_name'):
            for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                day_gaps = significant_gaps[df.loc[significant_gaps.index].index.day_name() == day]
                if len(day_gaps) > 0:
                    patterns['gap_seasonality'][day] = {
                        'count': len(day_gaps),
                        'avg_size': float(np.mean(np.abs(day_gaps)))
                    }
        
        return patterns
        
    except Exception as e:
        logger.warning(f"Error calculating gap patterns: {e}")
        return patterns


def calculate_volume_profile(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate volume profile and volume-based indicators.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC and volume data
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing volume metrics
    """
    metrics = {
        'volume_trend': 0.0,
        'volume_volatility': 0.0,
        'price_volume_correlation': 0.0,
        'volume_concentration_ratio': 0.0,
        'abnormal_volume_days': 0,
        'volume_weighted_price_trend': 0.0
    }
    
    if len(df) < 30:
        return metrics
    
    try:
        # Volume trend (regression slope)
        volumes = df['volume'].tail(90).values
        if len(volumes) > 20:
            x = np.arange(len(volumes))
            slope, _, r_value, _, _ = stats.linregress(x, volumes)
            
            avg_volume = volumes.mean()
            if avg_volume > 0:
                metrics['volume_trend'] = float(slope / avg_volume)
        
        # Volume volatility
        volume_returns = df['volume'].pct_change().dropna().tail(90)
        metrics['volume_volatility'] = float(volume_returns.std())
        
        # Price-volume correlation
        price_returns = df['close'].pct_change().dropna()
        volume_changes = df['volume'].pct_change().dropna()
        
        common_idx = price_returns.index.intersection(volume_changes.index)
        if len(common_idx) > 20:
            corr = price_returns.loc[common_idx].corr(volume_changes.loc[common_idx])
            metrics['price_volume_correlation'] = float(corr)
        
        # Volume concentration (80/20 rule)
        sorted_volumes = np.sort(df['volume'].tail(90).values)[::-1]
        if len(sorted_volumes) > 0 and sorted_volumes.sum() > 0:
            cumsum = np.cumsum(sorted_volumes)
            total = sorted_volumes.sum()
            
            # Find how many days account for 80% of volume
            days_for_80pct = np.argmax(cumsum >= 0.8 * total) + 1
            metrics['volume_concentration_ratio'] = float(days_for_80pct / len(sorted_volumes))
        
        # Abnormal volume days (>2 std above mean)
        volume_mean = df['volume'].tail(90).mean()
        volume_std = df['volume'].tail(90).std()
        if volume_std > 0:
            abnormal = df['volume'].tail(90) > (volume_mean + 2 * volume_std)
            metrics['abnormal_volume_days'] = int(abnormal.sum())
        
        # Volume-weighted price trend
        vwap = (df['close'] * df['volume']).tail(90) / df['volume'].tail(90)
        if len(vwap) > 20:
            x = np.arange(len(vwap))
            slope, _, _, _, _ = stats.linregress(x, vwap.values)
            
            avg_vwap = vwap.mean()
            if avg_vwap > 0:
                metrics['volume_weighted_price_trend'] = float(slope / avg_vwap)
        
        return metrics
        
    except Exception as e:
        logger.warning(f"Error calculating volume profile: {e}")
        return metrics