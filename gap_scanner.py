"""
Optimized Real-time Gap Scanner with PatchTST
Scans entire market for gap trading opportunities with low latency
"""

import os
import asyncio
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
import pytz
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import pickle
import json
from dataclasses import dataclass
import aiofiles
from collections import defaultdict
import heapq
from dotenv import load_dotenv

# Import our modules
from polygon_data_client import PolygonClient, TickerFilter, calculate_gap_metrics
from gap_model_trainer import OptimizedPatchTSTModel, TrainingConfig, EnhancedFeatureEngineering

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScanConfig:
    """Scanner configuration"""
    # Model settings
    model_path: str = "models/gap_model_final.pth"
    scaler_path: str = "models/scaler.pkl"
    config_path: str = "models/model_config.json"
    
    # Scanning parameters
    min_gap_percent: float = 0.015  # 1.5% minimum gap
    min_premarket_volume: int = 50000
    max_concurrent_requests: int = 10
    batch_size: int = 32
    
    # Time windows
    premarket_start: time = time(4, 0)  # 4:00 AM ET
    market_open: time = time(9, 30)     # 9:30 AM ET
    scan_window_minutes: int = 30       # Scan 30 min after open
    
    # Output settings
    top_opportunities: int = 20
    save_results: bool = True
    results_dir: str = "scan_results"

@dataclass
class GapOpportunity:
    """Gap trading opportunity"""
    ticker: str
    current_price: float
    prev_close: float
    gap_percent: float
    premarket_volume: int
    prediction: Dict
    score: float
    timestamp: datetime
    
    def __lt__(self, other):
        return self.score > other.score  # Higher score is better

class OptimizedGapScanner:
    """High-performance gap scanner optimized for A10 GPU"""
    
    def __init__(self, config: ScanConfig = None):
        self.config = config or ScanConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and components
        self._load_model()
        self._load_scaler()
        
        # Results storage
        self.opportunities = []
        self.scan_stats = {}
        
        # Cache for efficiency
        self.ticker_cache = {}
        self.feature_cache = {}
        
        # Ensure results directory exists
        Path(self.config.results_dir).mkdir(exist_ok=True)
    
    def _load_model(self):
        """Load trained model with error handling"""
        try:
            # Load model config
            with open(self.config.config_path, 'r') as f:
                model_config = json.load(f)
            
            # Create config object
            training_config = TrainingConfig(**model_config)
            
            # Initialize model
            self.model = OptimizedPatchTSTModel(training_config)
            
            # Load checkpoint
            checkpoint = torch.load(self.config.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.model.to(self.device)
            self.model.eval()
            
            # Store config for later use
            self.model_config = training_config
            
            logger.info(f"Model loaded from {self.config.model_path}")
            
            if torch.cuda.is_available():
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_scaler(self):
        """Load feature scaler"""
        try:
            with open(self.config.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("Feature scaler loaded")
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            raise
    
    async def prepare_ticker_features(self, ticker: str, client: PolygonClient) -> Optional[torch.Tensor]:
        """Prepare features for a single ticker with caching"""
        
        # Check cache first
        cache_key = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H')}"
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        try:
            # Get historical data (60 days of hourly)
            df = await client.get_aggregates(
                ticker,
                timespan='hour',
                from_date=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'),
                limit=2000
            )
            
            if df.empty or len(df) < 100:
                return None
            
            # Get premarket data for today
            premarket_df = await client.get_premarket_data(ticker)
            
            # Combine data if we have premarket
            if not premarket_df.empty:
                # Resample premarket to hourly for consistency
                premarket_hourly = premarket_df.resample('H').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                # Append to main dataframe
                df = pd.concat([df, premarket_hourly])
                df = df.sort_index()
            
            # Engineer features
            df = EnhancedFeatureEngineering.create_gap_features(df)
            
            # Select features matching training
            feature_cols = [
                'returns', 'log_returns', 'overnight_gap', 'close_vs_open',
                'volume_ratio', 'volume_momentum', 'atr_14', 'volatility_ratio',
                'rsi', 'vwap_deviation', 'hour', 'minute', 'day_of_week',
                'is_premarket', 'is_first_hour', 'gap_mean_5d', 'gap_std_5d',
                'open_to_high', 'open_to_low', 'dollar_volume',
                'rsi_oversold', 'rsi_overbought', 'gap_streak', 'gap_max_5d'
            ]
            
            # Extract features
            features = df[feature_cols].fillna(0).values
            
            # Take last sequence_length points
            if len(features) >= self.model_config.sequence_length:
                features = features[-self.model_config.sequence_length:]
            else:
                # Pad if necessary
                padded = np.zeros((self.model_config.sequence_length, len(feature_cols)))
                padded[-len(features):] = features
                features = padded
            
            # Scale features
            features = self.scaler.transform(features)
            
            # Convert to tensor and create patches
            features_tensor = torch.FloatTensor(features)
            patches = features_tensor.unfold(
                0, 
                self.model_config.patch_length, 
                self.model_config.stride
            )
            patches = patches.reshape(patches.size(0), -1)
            
            # Cache result
            self.feature_cache[cache_key] = patches
            
            return patches
            
        except Exception as e:
            logger.error(f"Error preparing features for {ticker}: {e}")
            return None
    
    async def analyze_ticker(self, ticker: str, ticker_info: Dict, 
                           client: PolygonClient) -> Optional[GapOpportunity]:
        """Analyze a single ticker for gap opportunity"""
        
        try:
            # Get current snapshot
            snapshot = await client.get_snapshot(ticker)
            if not snapshot:
                return None
            
            # Extract key data
            day_data = snapshot.get('day', {})
            prev_day = snapshot.get('prevDay', {})
            
            if not day_data or not prev_day:
                return None
            
            current_price = day_data.get('c', day_data.get('l', 0))
            prev_close = prev_day.get('c', 0)
            
            if prev_close == 0:
                return None
            
            # Calculate gap
            gap_percent = (current_price - prev_close) / prev_close
            
            # Check minimum gap threshold
            if abs(gap_percent) < self.config.min_gap_percent:
                return None
            
            # Get premarket volume
            premarket_volume = day_data.get('v', 0)
            if premarket_volume < self.config.min_premarket_volume:
                return None
            
            # Prepare features
            features = await self.prepare_ticker_features(ticker, client)
            if features is None:
                return None
            
            # Make prediction
            with torch.no_grad():
                # Add batch dimension
                batch_features = features.unsqueeze(0).to(self.device)
                
                # Use mixed precision for inference
                with autocast():
                    outputs = self.model(batch_features)
                
                # Process outputs
                gap_probs = torch.softmax(outputs['gap_logits'], dim=-1).cpu().numpy()[0]
                momentum_score = outputs['momentum_score'].cpu().item()
                continuation_prob = outputs['continuation_prob'].cpu().item()
            
            # Create prediction dict
            prediction = {
                'gap_up_continuation': float(gap_probs[0]),
                'gap_fade': float(gap_probs[1]),
                'gap_down_continuation': float(gap_probs[2]),
                'volatile': float(gap_probs[3]),
                'momentum_score': momentum_score,
                'continuation_prob': continuation_prob
            }
            
            # Calculate composite score
            # Weight factors: gap size, volume, model confidence
            volume_factor = min(premarket_volume / 1000000, 2.0)  # Cap at 2x
            gap_factor = min(abs(gap_percent) * 20, 2.0)  # Cap at 2x
            
            if gap_percent > 0:
                model_confidence = prediction['gap_up_continuation'] + prediction['gap_fade'] * 0.5
            else:
                model_confidence = prediction['gap_down_continuation'] + prediction['gap_fade'] * 0.5
            
            score = (
                model_confidence * 
                momentum_score * 
                continuation_prob * 
                volume_factor * 
                gap_factor
            )
            
            # Create opportunity
            opportunity = GapOpportunity(
                ticker=ticker,
                current_price=current_price,
                prev_close=prev_close,
                gap_percent=gap_percent,
                premarket_volume=premarket_volume,
                prediction=prediction,
                score=score,
                timestamp=datetime.now()
            )
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            return None
    
    async def scan_market(self) -> List[GapOpportunity]:
        """Scan entire market for gap opportunities"""
        
        start_time = datetime.now()
        logger.info("="*60)
        logger.info(f"Starting gap scan at {start_time}")
        logger.info("="*60)
        
        async with PolygonClient() as client:
            # Get market hours
            market_hours = await client.get_market_hours()
            
            # Check if market is in appropriate time window
            et_tz = pytz.timezone('US/Eastern')
            now_et = datetime.now(et_tz)
            
            # Get all active tickers
            ticker_filter = TickerFilter(
                min_price=5.0,
                max_price=1000.0,
                min_volume=500000,
                min_market_cap=500_000_000
            )
            
            all_tickers = await client.get_all_tickers(ticker_filter)
            logger.info(f"Scanning {len(all_tickers)} tickers...")
            
            # Process in batches with semaphore for rate limiting
            semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
            
            async def process_ticker(ticker_info):
                async with semaphore:
                    return await self.analyze_ticker(
                        ticker_info['ticker'], 
                        ticker_info, 
                        client
                    )
            
            # Create tasks for all tickers
            tasks = [process_ticker(ticker_info) for ticker_info in all_tickers]
            
            # Process with progress tracking
            opportunities = []
            
            for i in range(0, len(tasks), 100):
                batch = tasks[i:i+100]
                results = await asyncio.gather(*batch, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, GapOpportunity):
                        opportunities.append(result)
                
                logger.info(f"Processed {min(i+100, len(tasks))}/{len(tasks)} tickers")
            
            # Sort by score and keep top N
            opportunities.sort()
            self.opportunities = opportunities[:self.config.top_opportunities]
            
            # Calculate scan statistics
            end_time = datetime.now()
            self.scan_stats = {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': (end_time - start_time).total_seconds(),
                'tickers_scanned': len(all_tickers),
                'opportunities_found': len(opportunities),
                'top_opportunities': len(self.opportunities)
            }
            
            # Save results if configured
            if self.config.save_results:
                self._save_results()
            
            # Display results
            self._display_results()
            
            return self.opportunities
    
    def _save_results(self):
        """Save scan results to files"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_data = {
            'scan_stats': self.scan_stats,
            'opportunities': [
                {
                    'ticker': opp.ticker,
                    'current_price': opp.current_price,
                    'prev_close': opp.prev_close,
                    'gap_percent': opp.gap_percent,
                    'premarket_volume': opp.premarket_volume,
                    'prediction': opp.prediction,
                    'score': opp.score,
                    'timestamp': opp.timestamp.isoformat()
                }
                for opp in self.opportunities
            ]
        }
        
        results_path = Path(self.config.results_dir) / f'gap_scan_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save summary CSV
        summary_data = []
        for opp in self.opportunities:
            summary_data.append({
                'Ticker': opp.ticker,
                'Gap %': f"{opp.gap_percent*100:.2f}",
                'Score': f"{opp.score:.2f}",
                'Momentum': f"{opp.prediction['momentum_score']:.1f}",
                'Continuation': f"{opp.prediction['continuation_prob']*100:.1f}",
                'Volume': f"{opp.premarket_volume:,}",
                'Price': f"${opp.current_price:.2f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        csv_path = Path(self.config.results_dir) / f'gap_summary_{timestamp}.csv'
        summary_df.to_csv(csv_path, index=False)
        
        logger.info(f"Results saved to {self.config.results_dir}/")
    
    def _display_results(self):
        """Display scan results in formatted output"""
        
        print("\n" + "="*100)
        print("TOP GAP TRADING OPPORTUNITIES")
        print("="*100)
        print(f"Scan Duration: {self.scan_stats['duration_seconds']:.1f} seconds | "
              f"Tickers Scanned: {self.scan_stats['tickers_scanned']} | "
              f"Opportunities Found: {self.scan_stats['opportunities_found']}")
        print("="*100)
        
        if not self.opportunities:
            print("No significant gap opportunities found.")
            return
        
        # Header
        print(f"{'Rank':<5} {'Ticker':<8} {'Gap %':<8} {'Score':<8} "
              f"{'Momentum':<10} {'Cont %':<8} {'Volume':<12} {'Price':<10} {'Signal':<20}")
        print("-"*100)
        
        # Display opportunities
        for i, opp in enumerate(self.opportunities, 1):
            # Determine trading signal
            if opp.gap_percent > 0:
                if opp.prediction['gap_up_continuation'] > 0.6:
                    signal = "STRONG BUY"
                    signal_color = "\033[92m"  # Green
                elif opp.prediction['gap_fade'] > 0.6:
                    signal = "FADE SHORT"
                    signal_color = "\033[93m"  # Yellow
                else:
                    signal = "MONITOR"
                    signal_color = "\033[94m"  # Blue
            else:
                if opp.prediction['gap_down_continuation'] > 0.6:
                    signal = "STRONG SHORT"
                    signal_color = "\033[91m"  # Red
                else:
                    signal = "MONITOR"
                    signal_color = "\033[94m"  # Blue
            
            print(f"{i:<5} "
                  f"{opp.ticker:<8} "
                  f"{opp.gap_percent*100:>6.2f}% "
                  f"{opp.score:>7.2f} "
                  f"{opp.prediction['momentum_score']:>9.1f} "
                  f"{opp.prediction['continuation_prob']*100:>6.1f}% "
                  f"{opp.premarket_volume:>11,} "
                  f"${opp.current_price:>8.2f} "
                  f"{signal_color}{signal:<20}\033[0m")
        
        print("\n" + "="*100)
        print("LEGEND:")
        print("- Gap %: Percentage gap from previous close")
        print("- Score: Composite score (higher is better)")
        print("- Momentum: Expected momentum strength (0-100)")
        print("- Cont %: Probability of gap continuation")
        print("- Volume: Premarket/current volume")
        print("="*100)

async def run_continuous_scan(scan_interval_minutes: int = 5):
    """Run continuous scanning during market hours"""
    
    scanner = OptimizedGapScanner()
    et_tz = pytz.timezone('US/Eastern')
    
    while True:
        now_et = datetime.now(et_tz)
        
        # Check if within scanning window
        market_open = now_et.replace(hour=9, minute=30, second=0)
        scan_end = market_open + timedelta(minutes=scanner.config.scan_window_minutes)
        
        if scanner.config.premarket_start <= now_et.time() <= scan_end.time():
            # Run scan
            await scanner.scan_market()
            
            # Wait for next scan
            logger.info(f"Next scan in {scan_interval_minutes} minutes...")
            await asyncio.sleep(scan_interval_minutes * 60)
        else:
            # Wait until next premarket
            if now_et.time() > scan_end.time():
                next_scan = (now_et + timedelta(days=1)).replace(
                    hour=scanner.config.premarket_start.hour,
                    minute=scanner.config.premarket_start.minute,
                    second=0
                )
            else:
                next_scan = now_et.replace(
                    hour=scanner.config.premarket_start.hour,
                    minute=scanner.config.premarket_start.minute,
                    second=0
                )
            
            wait_seconds = (next_scan - now_et).total_seconds()
            logger.info(f"Market closed. Next scan at {next_scan}. Waiting {wait_seconds/3600:.1f} hours...")
            await asyncio.sleep(wait_seconds)

async def main():
    """Main execution"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Gap Trading Scanner')
    parser.add_argument('--continuous', action='store_true', 
                       help='Run continuous scanning during market hours')
    parser.add_argument('--interval', type=int, default=5,
                       help='Scan interval in minutes for continuous mode')
    parser.add_argument('--model', type=str, default='models/gap_model_final.pth',
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    # Update config if custom model path
    config = ScanConfig()
    if args.model:
        config.model_path = args.model
    
    # Run scanner
    if args.continuous:
        logger.info("Starting continuous gap scanner...")
        await run_continuous_scan(args.interval)
    else:
        scanner = OptimizedGapScanner(config)
        await scanner.scan_market()

if __name__ == "__main__":
    asyncio.run(main())