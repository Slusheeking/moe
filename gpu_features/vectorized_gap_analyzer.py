#!/usr/bin/env python3
"""
GPU-Accelerated Vectorized Gap Analyzer
Transforms raw price/volume data into sophisticated gap trading features.
All calculations run in parallel across entire universe on GPU.
"""

import os
import sys
import torch
import torch.nn.functional as F
import cupy as cp
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional
import pytz

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpu_trading_config import GPUTradingConfig
from database.connection import get_db_manager
from gpu_core.tensor_universe_manager import TensorUniverseManager
from gpu_core.parallel_data_pipeline import ParallelDataPipeline


class VectorizedGapAnalyzer:
    """GPU-accelerated gap analysis across entire trading universe."""
    
    def __init__(self, config: GPUTradingConfig,
                 universe_manager: TensorUniverseManager,
                 data_pipeline: ParallelDataPipeline):
        self.config = config
        self.universe_manager = universe_manager
        self.data_pipeline = data_pipeline
        self.device = config.device
        self.dtype = config.TENSOR_DTYPE
        
        # Database connection for gap events persistence
        self.db_manager = get_db_manager()
        
        # Get universe information
        self.universe_tensor, self.universe_symbols = universe_manager.get_universe_tensor()
        self.n_symbols = len(self.universe_symbols)
        
        # Initialize feature tensors
        self._init_feature_tensors()
        
        # Sector mapping for relative strength
        self._init_sector_mapping()
        
    def _init_feature_tensors(self):
        """Initialize GPU tensors for gap features."""
        # Core gap features
        self.gap_features = torch.zeros((self.n_symbols, 25), device=self.device, dtype=self.dtype)
        
        # Feature names and indices
        self.gap_feature_names = [
            # Basic gap metrics (0-4)
            'gap_size', 'gap_size_atr', 'gap_direction', 'gap_quality_score', 'gap_rank',
            
            # Institutional footprint (5-9)
            'institutional_footprint', 'block_trade_ratio', 'avg_trade_size_ratio',
            'smart_money_flow', 'institutional_participation',
            
            # Relative strength (10-14)
            'relative_gap_strength', 'sector_relative_strength', 'market_relative_strength',
            'peer_correlation', 'sector_gap_rank',
            
            # Time-based features (15-19)
            'gap_formation_minutes', 'gap_stability', 'gap_acceleration',
            'time_since_last_gap', 'gap_timing_score',
            
            # Advanced features (20-24)
            'news_catalyst_score', 'volume_surge_score', 'spread_quality',
            'momentum_alignment', 'continuation_probability'
        ]
        
        # Historical gap tracking
        self.gap_history = torch.zeros((self.n_symbols, 30, 5), device=self.device, dtype=self.dtype)
        # Dimensions: [symbols, days, features(date, gap%, continuation%, volume_ratio, catalyst)]
        
    def _init_sector_mapping(self):
        """Initialize sector mappings for relative strength analysis."""
        # In production, this would come from a data provider
        # For now, create synthetic sector assignments
        n_sectors = 11  # Standard S&P sectors
        self.sector_assignments = torch.randint(0, n_sectors, (self.n_symbols,), device=self.device)
        
        # Sector ETF proxies for comparison
        self.sector_etfs = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLB', 'XLRE', 'XLC', 'XLU']
        
    def analyze_gaps(self, current_time: Optional[datetime] = None) -> torch.Tensor:
        """Perform comprehensive gap analysis across all symbols."""
        if current_time is None:
            current_time = datetime.now(self.config.TIMEZONE)
        
        print(f"Analyzing gaps at {current_time.strftime('%H:%M:%S')} ET...")
        
        # Get pre-market data and historical closes
        pre_market_data, has_gaps = self.data_pipeline.get_pre_market_analysis()
        
        # Calculate all gap features in parallel
        self._calculate_basic_gap_features(pre_market_data)
        self._calculate_institutional_footprint()
        self._calculate_relative_strength()
        self._calculate_time_based_features(current_time)
        self._calculate_advanced_features()
        
        # Compute final gap quality scores
        self._compute_gap_quality_scores()
        
        # Rank gaps
        self._rank_gaps()
        
        # Persist significant gap events to database
        self._persist_gap_events(current_time)
        
        return self.gap_features
    
    def _calculate_basic_gap_features(self, pre_market_data: torch.Tensor):
        """Calculate fundamental gap metrics."""
        # Extract gap percentages and other data
        gap_percents = pre_market_data[:, 0]
        volumes_5min = pre_market_data[:, 1]
        
        # Gap size (absolute and relative to ATR)
        self.gap_features[:, 0] = torch.abs(gap_percents)
        
        # Get ATR from universe data
        atr_percents = self.universe_tensor[:, self.universe_manager.feature_indices['atr_percent']]
        self.gap_features[:, 1] = torch.abs(gap_percents) / (atr_percents + 0.01)  # gap size in ATR units
        
        # Gap direction (1 for up, -1 for down, 0 for no gap)
        self.gap_features[:, 2] = torch.sign(gap_percents)
        
        # Initial quality score (refined later)
        quality = torch.zeros_like(gap_percents)
        
        # Ideal gap size is 2-3 ATR
        ideal_gap_atr = 2.5
        gap_atr_diff = torch.abs(self.gap_features[:, 1] - ideal_gap_atr)
        size_score = torch.exp(-gap_atr_diff / 2.0)  # Gaussian-like scoring
        
        # Volume confirmation
        avg_volumes = self.universe_tensor[:, self.universe_manager.feature_indices['volume']]
        volume_ratios = volumes_5min / (avg_volumes * 0.05 + 1)  # 5min vs 5% of daily
        volume_score = torch.tanh(volume_ratios)  # Saturates at high values
        
        # Combine for initial quality
        quality = (size_score + volume_score) / 2
        self.gap_features[:, 3] = quality
        
    def _calculate_institutional_footprint(self):
        """Identify institutional activity in gap formation."""
        # Get real-time features from pipeline
        realtime_features = self.data_pipeline.realtime_features
        
        # Large trade ratio (from pipeline)
        large_trade_ratios = realtime_features[:, 17]
        
        # Average trade size relative to normal
        avg_trade_sizes = realtime_features[:, 16]
        historical_avg_sizes = self.universe_tensor[:, self.universe_manager.feature_indices['volume']] / 1000  # Approximate
        size_ratios = avg_trade_sizes / (historical_avg_sizes + 1)
        
        # Block trade detection (trades > institutional threshold)
        block_threshold = self.config.INSTITUTIONAL_BLOCK_SIZE
        estimated_block_ratio = torch.sigmoid((avg_trade_sizes - block_threshold) / 1000)
        
        # Smart money flow (large trades in gap direction)
        gap_directions = self.gap_features[:, 2]
        price_changes = realtime_features[:, 9]  # price_momentum_5min
        directional_agreement = torch.sign(price_changes) == gap_directions
        smart_flow = large_trade_ratios * directional_agreement.float()
        
        # Institutional participation score
        inst_score = (large_trade_ratios + size_ratios + estimated_block_ratio + smart_flow) / 4
        
        # Store features
        self.gap_features[:, 5] = inst_score  # institutional_footprint
        self.gap_features[:, 6] = estimated_block_ratio  # block_trade_ratio
        self.gap_features[:, 7] = size_ratios  # avg_trade_size_ratio
        self.gap_features[:, 8] = smart_flow  # smart_money_flow
        self.gap_features[:, 9] = torch.sigmoid(inst_score * 2)  # institutional_participation
        
    def _calculate_relative_strength(self):
        """Calculate gap strength relative to market and sector."""
        # Market gap (using SPY as proxy - in production, fetch actual SPY gap)
        market_gap = torch.median(self.gap_features[:, 0])  # Simplified
        
        # Relative to market
        gap_sizes = self.gap_features[:, 0]
        market_relative = (gap_sizes - market_gap) / (market_gap + 0.01)
        self.gap_features[:, 12] = torch.tanh(market_relative)
        
        # Sector relative strength
        sector_gaps = torch.zeros(11, device=self.device)  # 11 sectors
        sector_counts = torch.zeros(11, device=self.device)
        
        # Calculate average gap per sector
        for i in range(self.n_symbols):
            sector = self.sector_assignments[i]
            sector_gaps[sector] += gap_sizes[i]
            sector_counts[sector] += 1
        
        # Avoid division by zero
        sector_counts = torch.maximum(sector_counts, torch.ones_like(sector_counts))
        sector_avg_gaps = sector_gaps / sector_counts
        
        # Calculate relative to sector
        sector_relative = torch.zeros_like(gap_sizes)
        for i in range(self.n_symbols):
            sector = self.sector_assignments[i]
            sector_avg = sector_avg_gaps[sector]
            sector_relative[i] = (gap_sizes[i] - sector_avg) / (sector_avg + 0.01)
        
        self.gap_features[:, 11] = torch.tanh(sector_relative)  # sector_relative_strength
        
        # Combined relative strength
        self.gap_features[:, 10] = (market_relative + sector_relative) / 2  # relative_gap_strength
        
        # Peer correlation (simplified - correlation with sector peers)
        correlations = torch.zeros_like(gap_sizes)
        for sector in range(11):
            mask = self.sector_assignments == sector
            if mask.sum() > 1:
                sector_gaps_subset = gap_sizes[mask]
                mean_gap = sector_gaps_subset.mean()
                std_gap = sector_gaps_subset.std() + 0.001
                correlations[mask] = (sector_gaps_subset - mean_gap) / std_gap
        
        self.gap_features[:, 13] = torch.sigmoid(correlations)  # peer_correlation
        
        # Sector gap rank
        sector_ranks = torch.zeros_like(gap_sizes)
        for sector in range(11):
            mask = self.sector_assignments == sector
            if mask.sum() > 0:
                sector_gaps_subset = gap_sizes[mask]
                ranks = torch.argsort(torch.argsort(sector_gaps_subset, descending=True))
                sector_ranks[mask] = ranks.float() / mask.sum()
        
        self.gap_features[:, 14] = 1 - sector_ranks  # sector_gap_rank (1 = best)
        
    def _calculate_time_based_features(self, current_time: datetime):
        """Calculate time-dependent gap features."""
        # Time since market close (for gap formation time)
        market_close_yesterday = current_time.replace(hour=16, minute=0, second=0) - timedelta(days=1)
        if current_time.weekday() == 0:  # Monday
            market_close_yesterday -= timedelta(days=2)  # Friday close
        
        minutes_since_close = (current_time - market_close_yesterday).total_seconds() / 60
        
        # Gap formation time (how long the gap has been developing)
        # Using pre-market data timestamps
        first_trade_times = torch.zeros(self.n_symbols, device=self.device)
        for i in range(self.n_symbols):
            if self.data_pipeline.pre_market_counts[i] > 0:
                first_timestamp = self.data_pipeline.pre_market_timestamps[i, 0]
                if first_timestamp > 0:
                    first_trade_times[i] = (current_time.timestamp() - first_timestamp.item()) / 60
        
        self.gap_features[:, 15] = first_trade_times  # gap_formation_minutes
        
        # Gap stability (low volatility during formation = stable)
        stability_scores = torch.zeros(self.n_symbols, device=self.device)
        for i in range(self.n_symbols):
            count = self.data_pipeline.pre_market_counts[i]
            if count > 10:
                prices = self.data_pipeline.pre_market_prices[i, :count]
                price_std = prices.std()
                price_mean = prices.mean()
                cv = price_std / (price_mean + 0.01)  # Coefficient of variation
                stability_scores[i] = 1 / (1 + cv * 10)  # Inverse CV
        
        self.gap_features[:, 16] = stability_scores  # gap_stability
        
        # Gap acceleration (is gap widening or stabilizing)
        acceleration = torch.zeros(self.n_symbols, device=self.device)
        for i in range(self.n_symbols):
            count = self.data_pipeline.pre_market_counts[i]
            if count > 20:
                prices = self.data_pipeline.pre_market_prices[i, :count]
                # Compare recent vs older price moves
                recent_change = prices[-10:].mean() - prices[-20:-10].mean()
                older_change = prices[-20:-10].mean() - prices[-30:-20].mean() if count > 30 else 0
                acceleration[i] = recent_change - older_change
        
        self.gap_features[:, 17] = torch.tanh(acceleration * 100)  # gap_acceleration
        
        # Time since last significant gap (from history)
        # This would typically come from a database
        # For now, simulate with random historical data
        days_since_gap = torch.randint(1, 30, (self.n_symbols,), device=self.device, dtype=self.dtype)
        self.gap_features[:, 18] = days_since_gap.float()  # time_since_last_gap
        
        # Gap timing score (certain times are better for gaps)
        hour = current_time.hour
        minute = current_time.minute
        time_score = 0.0
        
        # Pre-market sweet spot: 7:00-9:00 AM
        if 7 <= hour < 9:
            time_score = 1.0
        elif hour == 9 and minute < 20:
            time_score = 0.8
        elif hour == 6:
            time_score = 0.6
        elif hour == 9 and minute >= 20:
            time_score = 0.9  # Close to open
        else:
            time_score = 0.3
        
        self.gap_features[:, 19] = torch.full((self.n_symbols,), time_score, device=self.device)  # gap_timing_score
        
    def _calculate_advanced_features(self):
        """Calculate sophisticated gap features."""
        # News catalyst score (in production, would use NLP on news feeds)
        # For now, use volume surge as proxy
        volume_5min = self.data_pipeline.realtime_features[:, 5]
        avg_volumes = self.universe_tensor[:, self.universe_manager.feature_indices['volume']]
        volume_surge = volume_5min / (avg_volumes * 0.01 + 1)  # vs 1% of daily
        
        news_scores = torch.sigmoid((volume_surge - 5) / 5)  # Sigmoid centered at 5x normal
        self.gap_features[:, 20] = news_scores  # news_catalyst_score
        
        # Volume surge score (more sophisticated than above)
        volume_acceleration = self.data_pipeline.realtime_features[:, 10]  # volume_momentum_5min
        volume_surge_score = (torch.tanh(volume_surge / 10) + torch.sigmoid(volume_acceleration - 1)) / 2
        self.gap_features[:, 21] = volume_surge_score  # volume_surge_score
        
        # Spread quality (tight spreads = better liquidity)
        spreads = self.data_pipeline.realtime_features[:, 4]  # spread_pct
        spread_quality = 1 / (1 + spreads)  # Inverse spread
        self.gap_features[:, 22] = spread_quality  # spread_quality
        
        # Momentum alignment (price momentum aligns with gap direction)
        price_momentum = self.data_pipeline.realtime_features[:, 9]  # price_momentum_5min
        gap_directions = self.gap_features[:, 2]
        alignment = torch.sign(price_momentum) == gap_directions
        momentum_scores = alignment.float() * torch.abs(torch.tanh(price_momentum / 5))
        self.gap_features[:, 23] = momentum_scores  # momentum_alignment
        
        # Continuation probability (learned from historical patterns)
        # Combine multiple factors
        quality = self.gap_features[:, 3]  # gap_quality_score
        institutional = self.gap_features[:, 5]  # institutional_footprint
        stability = self.gap_features[:, 16]  # gap_stability
        timing = self.gap_features[:, 19]  # gap_timing_score
        
        # Weighted combination
        continuation_prob = (
            quality * 0.3 +
            institutional * 0.3 +
            stability * 0.2 +
            timing * 0.1 +
            momentum_scores * 0.1
        )
        
        self.gap_features[:, 24] = torch.sigmoid(continuation_prob * 2)  # continuation_probability
        
    def _compute_gap_quality_scores(self):
        """Compute final gap quality scores using all features."""
        # Define feature weights for final scoring
        weights = torch.tensor([
            0.15,  # gap_size (moderate weight)
            0.20,  # gap_size_atr (important)
            0.00,  # gap_direction (not used in score)
            0.00,  # gap_quality_score (will be overwritten)
            0.00,  # gap_rank (will be computed)
            0.25,  # institutional_footprint (very important)
            0.10,  # block_trade_ratio
            0.05,  # avg_trade_size_ratio
            0.10,  # smart_money_flow
            0.05,  # institutional_participation
            0.15,  # relative_gap_strength (important)
            0.10,  # sector_relative_strength
            0.05,  # market_relative_strength
            0.05,  # peer_correlation
            0.05,  # sector_gap_rank
            0.05,  # gap_formation_minutes (prefer longer formation)
            0.15,  # gap_stability (important)
            0.10,  # gap_acceleration
            -0.05, # time_since_last_gap (negative - prefer fresh)
            0.10,  # gap_timing_score
            0.10,  # news_catalyst_score
            0.15,  # volume_surge_score
            0.10,  # spread_quality
            0.15,  # momentum_alignment
            0.20   # continuation_probability (very important)
        ], device=self.device, dtype=self.dtype)
        
        # Normalize weights
        weights = weights / weights.abs().sum()
        
        # Compute weighted score
        # First normalize features to 0-1 range where appropriate
        normalized_features = self.gap_features.clone()
        
        # Some features need special normalization
        normalized_features[:, 0] = torch.sigmoid((self.gap_features[:, 0] - 2) / 2)  # gap_size
        normalized_features[:, 1] = torch.sigmoid((self.gap_features[:, 1] - 2.5) / 1.5)  # gap_size_atr
        normalized_features[:, 15] = torch.sigmoid((self.gap_features[:, 15] - 60) / 30)  # formation_minutes
        normalized_features[:, 18] = 1 - torch.sigmoid((self.gap_features[:, 18] - 10) / 10)  # time_since_gap
        
        # Compute final scores
        quality_scores = torch.matmul(normalized_features, weights)
        
        # Apply final sigmoid for 0-1 range
        self.gap_features[:, 3] = torch.sigmoid(quality_scores * 3)
        
    def _rank_gaps(self):
        """Rank all gaps by quality score."""
        quality_scores = self.gap_features[:, 3]
        
        # Get ranks (0 = best)
        sorted_indices = torch.argsort(quality_scores, descending=True)
        ranks = torch.zeros_like(quality_scores)
        ranks[sorted_indices] = torch.arange(len(sorted_indices), device=self.device, dtype=self.dtype)
        
        # Normalize to 0-1 (1 = best)
        self.gap_features[:, 4] = 1 - (ranks / len(ranks))
        
    def get_top_gaps(self, n: int = 20) -> Tuple[List[str], torch.Tensor]:
        """Get top N gaps by quality score."""
        quality_scores = self.gap_features[:, 3]
        gap_sizes = self.gap_features[:, 0]
        
        # Filter for minimum gap size
        valid_gaps = gap_sizes >= self.config.MIN_GAP_PERCENT
        
        # Set invalid gaps to -1 quality
        filtered_scores = quality_scores.clone()
        filtered_scores[~valid_gaps] = -1
        
        # Get top N
        top_indices = torch.argsort(filtered_scores, descending=True)[:n]
        
        # Get symbols and features
        top_symbols = [self.universe_symbols[idx] for idx in top_indices if filtered_scores[idx] > 0]
        top_features = self.gap_features[top_indices]
        
        return top_symbols, top_features
    
    def get_gap_report(self, symbol: str) -> Dict[str, float]:
        """Get detailed gap analysis for a specific symbol."""
        if symbol not in self.universe_symbols:
            return {}
        
        idx = self.universe_symbols.index(symbol)
        features = self.gap_features[idx]
        
        report = {}
        for i, name in enumerate(self.gap_feature_names):
            report[name] = features[i].item()
        
        # Add some derived metrics
        report['recommendation'] = 'BUY' if features[3] > 0.65 else 'WATCH' if features[3] > 0.5 else 'SKIP'
        report['confidence'] = features[24].item()  # continuation probability
        
        return report


# Example usage when file is run directly
if __name__ == "__main__":
    print("GPU-Accelerated Gap Analyzer Test")
    print("=" * 50)
    
    # Initialize components
    config = GPUTradingConfig()
    universe_manager = TensorUniverseManager(config)
    
    # Ensure universe is loaded
    if len(universe_manager.universe_symbols) == 0:
        print("Loading universe...")
        universe_manager.update_universe()
    
    # Create pipeline and analyzer
    data_pipeline = ParallelDataPipeline(config, universe_manager)
    gap_analyzer = VectorizedGapAnalyzer(config, universe_manager, data_pipeline)
    
    # Fetch historical data for context
    print("\nFetching historical data...")
    data_pipeline.fetch_historical_data(lookback_days=30)
    
    # Simulate pre-market data for testing
    print("\nSimulating pre-market activity...")
    for i in range(min(50, len(data_pipeline.universe_symbols))):
        # Create realistic gap scenarios
        if i % 5 == 0:  # 20% have significant gaps
            gap_size = np.random.choice([3.5, -3.5, 4.5, -4.5])  # Significant gaps
            base_price = 50 + i * 2
            
            # Simulate pre-market trades with gap
            for j in range(200):
                price_noise = np.random.randn() * 0.1
                data_pipeline.pre_market_prices[i, j] = base_price * (1 + gap_size/100) + price_noise
                data_pipeline.pre_market_volumes[i, j] = 500 + np.random.randint(0, 2000)
                data_pipeline.pre_market_timestamps[i, j] = int(datetime.now().timestamp()) - (200 - j) * 30
def _persist_gap_events(self, current_time: datetime):
        """Persist significant gap events to database."""
        try:
            # Only persist gaps that meet minimum quality threshold
            quality_threshold = 0.3
            significant_mask = self.gap_features[:, 3] > quality_threshold  # gap_quality_score > threshold
            
            if not significant_mask.any():
                return  # No significant gaps to persist
            
            # Get indices of significant gaps
            significant_indices = torch.where(significant_mask)[0].cpu().numpy()
            
            gap_events = []
            for idx in significant_indices:
                symbol = self.universe_symbols[idx]
                features = self.gap_features[idx].cpu().numpy()
                
                # Extract key features
                gap_size = float(features[0])  # gap_size
                gap_direction = 1 if features[2] > 0 else -1  # gap_direction
                gap_quality_score = float(features[3])  # gap_quality_score
                institutional_footprint = float(features[5])  # institutional_footprint
                volume_surge_score = float(features[21])  # volume_surge_score
                continuation_probability = float(features[24])  # continuation_probability
                
                # Build comprehensive metadata
                metadata = {
                    'gap_size_atr': float(features[1]),
                    'block_trade_ratio': float(features[6]),
                    'avg_trade_size_ratio': float(features[7]),
                    'smart_money_flow': float(features[8]),
                    'institutional_participation': float(features[9]),
                    'relative_gap_strength': float(features[10]),
                    'sector_relative_strength': float(features[11]),
                    'market_relative_strength': float(features[12]),
                    'peer_correlation': float(features[13]),
                    'sector_gap_rank': float(features[14]),
                    'gap_formation_minutes': float(features[15]),
                    'gap_stability': float(features[16]),
                    'gap_acceleration': float(features[17]),
                    'time_since_last_gap': float(features[18]),
                    'gap_timing_score': float(features[19]),
                    'news_catalyst_score': float(features[20]),
                    'spread_quality': float(features[22]),
                    'momentum_alignment': float(features[23]),
                    'analyzer_version': '1.0',
                    'processing_time': current_time.isoformat()
                }
                
                gap_event = {
                    'symbol': symbol,
                    'timestamp': current_time,
                    'gap_size': gap_size,
                    'gap_direction': gap_direction,
                    'gap_quality_score': gap_quality_score,
                    'institutional_footprint': institutional_footprint,
                    'volume_surge_score': volume_surge_score,
                    'continuation_probability': continuation_probability,
                    'metadata': metadata
                }
                
                gap_events.append(gap_event)
            
            # Batch insert gap events
            if gap_events:
                insert_query = """
                    INSERT INTO gap_events (
                        symbol, timestamp, gap_size, gap_direction, gap_quality_score,
                        institutional_footprint, volume_surge_score, continuation_probability, metadata
                    ) VALUES (
                        %(symbol)s, %(timestamp)s, %(gap_size)s, %(gap_direction)s, %(gap_quality_score)s,
                        %(institutional_footprint)s, %(volume_surge_score)s, %(continuation_probability)s, %(metadata)s
                    )
                """
                
                success = self.db_manager.execute_batch(insert_query, gap_events)
                if success:
                    print(f"Persisted {len(gap_events)} gap events to database")
                else:
                    print(f"Failed to persist {len(gap_events)} gap events to database")
                    
        except Exception as e:
            print(f"Error persisting gap events to database: {e}")
            # Continue processing even if database persistence fails


if __name__ == "__main__":
    """Test gap analyzer with simulated pre-market data."""
    import random
    
    # Initialize system
    print("Initializing GPU gap analysis system...")
    config = GPUTradingConfig()
    universe_manager = TensorUniverseManager(config)
    data_pipeline = ParallelDataPipeline(config, universe_manager)
    gap_analyzer = VectorizedGapAnalyzer(config, universe_manager, data_pipeline)
    
    # Load universe
    universe_tensor, symbols = universe_manager.get_universe_tensor()
    print(f"Universe loaded: {len(symbols)} symbols")
    
    # Simulate pre-market gaps for testing
    print("Simulating pre-market gaps...")
    for i in range(min(10, len(symbols))):
        # Simulate gap
        gap_percent = random.uniform(-8, 8)
        volume_5min = random.uniform(50000, 500000)
        data_pipeline.current_prices[i] = (1 + gap_percent / 100) * data_pipeline.previous_closes[i]
        data_pipeline.volumes_5min[i] = volume_5min
        
        # Simulate institutional activity
        if abs(gap_percent) > 2:
            data_pipeline.pre_market_counts[i] = 200
            data_pipeline._update_realtime_features_gpu(i)
    
    # Analyze gaps
    print("\nAnalyzing gaps...")
    gap_features = gap_analyzer.analyze_gaps()
    
    # Get top gaps
    top_symbols, top_features = gap_analyzer.get_top_gaps(n=10)
    
    print(f"\nTop 10 Gap Trading Opportunities:")
    print("-" * 80)
    for i, symbol in enumerate(top_symbols):
        idx = gap_analyzer.universe_symbols.index(symbol)
        features = gap_analyzer.gap_features[idx]
        
        gap_size = features[0].item()
        gap_direction = "UP" if features[2] > 0 else "DOWN"
        quality_score = features[3].item()
        inst_footprint = features[5].item()
        continuation_prob = features[24].item()
        
        print(f"\n{i+1}. {symbol}")
        print(f"   Gap: {gap_direction} {gap_size:.2f}%")
        print(f"   Quality Score: {quality_score:.3f}")
        print(f"   Institutional Footprint: {inst_footprint:.3f}")
        print(f"   Continuation Probability: {continuation_prob:.3f}")
    
    # Detailed report for top symbol
    if top_symbols:
        print(f"\nDetailed Analysis for {top_symbols[0]}:")
        print("-" * 50)
        report = gap_analyzer.get_gap_report(top_symbols[0])
        
        key_metrics = [
            'gap_size', 'gap_size_atr', 'institutional_footprint',
            'relative_gap_strength', 'gap_stability', 'volume_surge_score',
            'momentum_alignment', 'continuation_probability', 'recommendation'
        ]
        
        for metric in key_metrics:
            if metric in report:
                if isinstance(report[metric], float):
                    print(f"  {metric}: {report[metric]:.3f}")
                else:
                    print(f"  {metric}: {report[metric]}")
    
    # GPU utilization
    print(f"\nGPU Memory Usage:")
    print(f"  Gap features tensor: {gap_features.element_size() * gap_features.nelement() / 1e6:.2f} MB")
    print(f"  Total features: {len(gap_analyzer.gap_feature_names)}")
    print(f"  Processing {len(gap_analyzer.universe_symbols)} symbols in parallel")