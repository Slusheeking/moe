"""
GPU-Accelerated Enterprise Mid-Cap Universe Data Collector
Optimized for A10 GPU with RAPIDS cuDF, CuPy, and Numba acceleration
Collects and labels training data for 2000+ mid-cap stocks with ultra-fast processing
"""

import os
import asyncio
import warnings
import logging
from datetime import datetime, timedelta, time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
import json
from dotenv import load_dotenv

# GPU-accelerated imports with fallbacks
try:
    import cudf as pd
    import cupy as np
    from cudf import DataFrame
    print("🚀 Enterprise Collector: Using RAPIDS cuDF and CuPy for GPU acceleration!")
    GPU_AVAILABLE = True
except ImportError:
    import pandas as pd
    import numpy as np
    from pandas import DataFrame
    print("⚠️  Enterprise Collector: Using CPU fallback - install RAPIDS for GPU acceleration")
    GPU_AVAILABLE = False

try:
    from numba import cuda, jit
    NUMBA_AVAILABLE = cuda.is_available()
    if NUMBA_AVAILABLE:
        print(f"⚡ Enterprise Collector: Numba CUDA acceleration enabled")
except ImportError:
    NUMBA_AVAILABLE = False

# Import our GPU-optimized polygon client
from polygon_data_client import PolygonClient, TickerFilter, EnhancedFeatureEngineering

# Suppress GPU warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module='cudf')

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MidCapFilter:
    """Mid-cap stock universe filter for full market coverage"""
    min_market_cap: int = 2_000_000_000      # $2B
    max_market_cap: int = 10_000_000_000     # $10B
    min_daily_volume: int = 500_000          # 500K shares (lowered for broader coverage)
    min_price: float = 5.0                   # $5 minimum
    max_price: float = 500.0                 # $500 maximum (increased for growth stocks)
    min_avg_volume_30d: int = 250_000        # 250K avg volume (lowered for broader coverage)
    exclude_otc: bool = True                 # Exclude OTC stocks
    exclude_etfs: bool = True                # Exclude ETFs

@dataclass
class LabelingConfig:
    """Configuration for multi-horizon labeling"""
    # Time horizons for prediction (in minutes)
    horizons: List[int] = None
    
    # Movement threshold for UP/DOWN classification
    movement_threshold: float = 0.004        # 0.4% movement threshold
    
    # Data timespan - configurable for noise testing
    data_timespan: str = "minute"            # 'minute', '5minute', '15minute'
    
    def __post_init__(self):
        if self.horizons is None:
            self.horizons = [1, 5, 10]  # 1, 5, and 10 minute predictions

@dataclass
class DataCollectionConfig:
    """Configuration for enterprise data collection"""
    # Data collection
    data_dir: str = "data/midcap_universe"
    temp_dir: str = "data/temp"
    batch_size: int = 50                     # Stocks per batch
    max_workers: int = 20                    # Parallel workers
    lookback_days: int = 180                 # Exactly 6 months historical
    
    # Trading hours
    market_open: time = time(9, 40)          # 9:40 AM ET
    market_close: time = time(11, 40)        # 11:40 AM ET (2-hour trading window)
    
    # Multi-horizon labeling
    labeling: LabelingConfig = None
    
    # Storage optimization
    compression: str = "snappy"
    partition_cols: List[str] = None
    
    # Mid-cap filter
    midcap_filter: MidCapFilter = None
    
    def __post_init__(self):
        if self.partition_cols is None:
            self.partition_cols = ["year", "month"]
        if self.midcap_filter is None:
            self.midcap_filter = MidCapFilter()
        if self.labeling is None:
            self.labeling = LabelingConfig()

def generate_forward_looking_labels(df: pd.DataFrame, config: LabelingConfig) -> pd.DataFrame:
    """
    Generate multi-horizon forward-looking labels for price movement.
    Labels: 0=UP, 1=DOWN, 2=NEUTRAL
    """
    for horizon in config.horizons:
        # Calculate future rolling max and min prices
        future_max = df['high'].rolling(window=horizon, min_periods=1).max().shift(-horizon)
        future_min = df['low'].rolling(window=horizon, min_periods=1).min().shift(-horizon)

        # Calculate percentage change
        up_return = (future_max - df['close']) / df['close']
        down_return = (future_min - df['close']) / df['close']

        # Determine labels based on threshold
        label_col = f'label_{horizon}m'
        df[label_col] = 2  # Default to NEUTRAL
        df.loc[up_return > config.movement_threshold, label_col] = 0  # UP
        df.loc[down_return < -config.movement_threshold, label_col] = 1  # DOWN
        
        # Prioritize larger move if both thresholds are met
        up_mask = up_return > config.movement_threshold
        down_mask = down_return < -config.movement_threshold
        both_mask = up_mask & down_mask
        
        if not df[both_mask].empty:
            up_wins = abs(up_return[both_mask]) >= abs(down_return[both_mask])
            down_wins = ~up_wins
            df.loc[both_mask & up_wins, label_col] = 0
            df.loc[both_mask & down_wins, label_col] = 1

    return df

class EnterpriseDataCollector:
    """Main data collection orchestrator for mid-cap universe"""
    
    def __init__(self, config: DataCollectionConfig = None):
        self.config = config or DataCollectionConfig()
        
        # Setup directories
        Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.temp_dir).mkdir(parents=True, exist_ok=True)
        
        # Stats tracking
        self.collection_stats = {
            'stocks_processed': 0,
            'total_samples': 0,
            'signals_generated': 0,
            'errors': []
        }
    
    async def collect_midcap_universe(self) -> Dict:
        """Main entry point - collect entire mid-cap universe"""
        logger.info("Starting enterprise mid-cap data collection...")
        
        start_time = datetime.now()
        
        async with PolygonClient() as client:
            # Get mid-cap universe
            midcap_stocks = await self._get_midcap_universe(client)
            logger.info(f"Found {len(midcap_stocks)} mid-cap stocks")
            
            # Process in batches
            await self._process_stock_batches(client, midcap_stocks)
            
            # Consolidate and save metadata
            await self._finalize_dataset()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Final stats
        self.collection_stats.update({
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'stocks_total': len(midcap_stocks)
        })
        
        logger.info(f"Collection complete! Stats: {self.collection_stats}")
        return self.collection_stats
    
    async def _get_midcap_universe(self, client: PolygonClient) -> List[Dict]:
        """Get all mid-cap stocks matching our criteria with REAL market cap filtering"""
        filter_criteria = TickerFilter(
            min_price=self.config.midcap_filter.min_price,
            max_price=self.config.midcap_filter.max_price,
            min_volume=self.config.midcap_filter.min_daily_volume,
            min_market_cap=self.config.midcap_filter.min_market_cap,
            max_market_cap=self.config.midcap_filter.max_market_cap
        )
        
        all_tickers = await client.get_all_tickers(filter_criteria)
        
        logger.info(f"Got {len(all_tickers)} total tickers, applying strict mid-cap market cap filtering...")
        
        # Filter for actual mid-cap stocks with market cap verification
        midcap_stocks = []
        processed_count = 0
        skipped_no_market_cap = 0
        skipped_wrong_market_cap = 0
        
        for ticker_info in all_tickers:
            processed_count += 1
            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count}/{len(all_tickers)} tickers, found {len(midcap_stocks)} valid mid-caps")
            
            # Extract ticker symbol safely
            ticker = None
            if hasattr(ticker_info, 'ticker'):
                ticker = ticker_info.ticker
            elif isinstance(ticker_info, dict) and 'ticker' in ticker_info:
                ticker = ticker_info['ticker']
            else:
                continue
            
            # Basic symbol filtering for quality stocks
            if not (ticker and
                    len(ticker) <= 5 and  # Skip very long symbols
                    not ticker.endswith('.') and  # Skip OTC stocks
                    not ticker.startswith('SPY') and  # Skip major ETFs
                    not ticker.startswith('QQQ') and
                    not ticker.startswith('IWM') and  # Skip Russell ETFs
                    not ticker.startswith('VT') and   # Skip Vanguard ETFs
                    not ticker.endswith('X') and     # Skip many ETFs
                    ticker.isalpha()):               # Only alphabetic symbols
                continue
            
            # ✅ REAL MARKET CAP FILTERING - Get actual market cap from Polygon API
            ticker_details = await client.get_ticker_details(ticker)
            if not ticker_details:
                skipped_no_market_cap += 1
                continue
                
            market_cap = ticker_details.get('market_cap')
            if not market_cap:
                skipped_no_market_cap += 1
                continue
            
            # Apply strict mid-cap filter ($2B - $10B)
            if market_cap < self.config.midcap_filter.min_market_cap:
                skipped_wrong_market_cap += 1
                continue
            if market_cap > self.config.midcap_filter.max_market_cap:
                skipped_wrong_market_cap += 1
                continue
            
            # Add market cap info to ticker data
            if isinstance(ticker_info, dict):
                ticker_info['market_cap'] = market_cap
                ticker_info['market_cap_b'] = market_cap / 1_000_000_000  # In billions
            
            midcap_stocks.append(ticker_info)
            
            # Show progress for first few stocks
            if len(midcap_stocks) <= 10:
                market_cap_b = market_cap / 1_000_000_000
                logger.info(f"✅ {ticker}: ${market_cap_b:.1f}B market cap - VALID MID-CAP")
        
        logger.info(f"🎯 FINAL MID-CAP FILTERING RESULTS:")
        logger.info(f"   📊 Total processed: {processed_count}")
        logger.info(f"   ✅ Valid mid-caps: {len(midcap_stocks)}")
        logger.info(f"   ❌ No market cap data: {skipped_no_market_cap}")
        logger.info(f"   ❌ Wrong market cap range: {skipped_wrong_market_cap}")
        logger.info(f"   🎯 Target range: ${self.config.midcap_filter.min_market_cap/1e9:.1f}B - ${self.config.midcap_filter.max_market_cap/1e9:.1f}B")
        
        if len(midcap_stocks) == 0:
            logger.error("❌ NO MID-CAP STOCKS FOUND! Check API key and market cap filtering.")
            return []
        
        logger.info(f"🚀 Processing {len(midcap_stocks)} REAL mid-cap stocks")
        logger.info(f"   ⏰ Time window: 9:40 AM - 11:40 AM EST (2 hours daily)")
        logger.info(f"   📅 Historical period: {self.config.lookback_days} days (~6 months)")
        
        return midcap_stocks
    
    async def _process_stock_batches(self, client: PolygonClient, stocks: List[Dict]):
        """Process stocks in parallel batches"""
        
        # Create batches
        batches = [
            stocks[i:i + self.config.batch_size]
            for i in range(0, len(stocks), self.config.batch_size)
        ]
        
        logger.info(f"Processing {len(batches)} batches of {self.config.batch_size} stocks each")
        
        # Process batches with limited concurrency
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def process_batch(batch_idx, batch):
            async with semaphore:
                await self._process_single_batch(client, batch_idx, batch)
        
        # Run all batches
        tasks = [
            process_batch(idx, batch) 
            for idx, batch in enumerate(batches)
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_single_batch(self, client: PolygonClient, 
                                   batch_idx: int, batch: List[Dict]):
        """Process a single batch of stocks"""
        
        logger.info(f"Processing batch {batch_idx + 1} with {len(batch)} stocks")
        
        batch_data = []
        
        for stock_info in batch:
            try:
                ticker = stock_info['ticker']
                stock_data = await self._collect_stock_data(client, ticker)
                
                if stock_data is not None and not stock_data.empty:
                    batch_data.append(stock_data)
                    self.collection_stats['stocks_processed'] += 1
                
            except Exception as e:
                error_msg = f"Error processing {ticker}: {str(e)}"
                logger.error(error_msg)
                self.collection_stats['errors'].append(error_msg)
        
        # Save batch data
        if batch_data:
            await self._save_batch_data(batch_idx, batch_data)
        
        logger.info(f"Batch {batch_idx + 1} complete. Processed {len(batch_data)} stocks successfully")
    
    async def _collect_stock_data(self, client: PolygonClient, ticker: str) -> Optional[pd.DataFrame]:
        """Collect and process data for a single stock"""
        
        try:
            # Get historical minute data for lookback period
            from_date = (datetime.now() - timedelta(days=self.config.lookback_days)).strftime('%Y-%m-%d')
            
            df = await client.get_aggregates(
                ticker=ticker,
                timespan='minute',
                from_date=from_date,
                to_date=datetime.now().strftime('%Y-%m-%d')
            )
            
            if df.empty:
                return None
            
            # Filter to trading hours only
            df = self._filter_to_trading_hours(df)
            
            if len(df) < 100:  # Need minimum data
                return None
            
            # Engineer features
            df = EnhancedFeatureEngineering.create_gap_features(df)

            # Generate multi-horizon labels
            df = generate_forward_looking_labels(df, self.config.labeling)
            
            # Add metadata
            df['ticker'] = ticker
            df['year'] = df.index.year
            df['month'] = df.index.month
            df['day'] = df.index.day
            
            # Count signals for stats
            for horizon in self.config.labeling.horizons:
                label_col = f'label_{horizon}m'
                if label_col in df.columns:
                    signal_counts = df[label_col].value_counts()
                    self.collection_stats[f'signals_generated_{horizon}m'] = signal_counts.to_dict()
            
            return df
            
        except Exception as e:
            logger.error(f"Error collecting data for {ticker}: {e}")
            return None
    
    def _filter_to_trading_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to scalping window only"""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        
        # Convert to ET timezone if needed
        if df.index.tz is None:
            df.index = df.index.tz_localize('US/Eastern')
        elif df.index.tz != 'US/Eastern':
            df.index = df.index.tz_convert('US/Eastern')
        
        # Filter to scalping hours
        mask = (
            (df.index.time >= self.config.market_open) &
            (df.index.time <= self.config.market_close)
        )
        
        return df[mask]
    
    async def _save_batch_data(self, batch_idx: int, batch_data: List[pd.DataFrame]):
        """Save batch data to parquet files"""
        
        # Combine all batch data
        combined_df = pd.concat(batch_data, ignore_index=False)
        
        # Reset index to make timestamp a column
        combined_df = combined_df.reset_index()
        combined_df = combined_df.rename(columns={'index': 'timestamp'})
        
        # Save to parquet with partitioning
        filename = f"midcap_batch_{batch_idx:04d}.parquet"
        filepath = Path(self.config.temp_dir) / filename
        
        # Save with compression
        combined_df.to_parquet(
            filepath,
            compression=self.config.compression,
            engine='pyarrow'
        )
        
        logger.info(f"Saved batch {batch_idx} to {filepath} ({len(combined_df)} records)")
        
        # Update stats
        self.collection_stats['total_samples'] += len(combined_df)
    
    async def _finalize_dataset(self):
        """Consolidate all batch files into final partitioned dataset"""
        
        logger.info("Finalizing dataset...")
        
        # Load all batch files
        temp_files = list(Path(self.config.temp_dir).glob("midcap_batch_*.parquet"))
        
        if not temp_files:
            logger.warning("No batch files found to consolidate")
            return
        
        # Read and combine all batches
        all_data = []
        for file_path in temp_files:
            df = pd.read_parquet(file_path)
            all_data.append(df)
        
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Save final partitioned dataset
        final_path = Path(self.config.data_dir) / "partitioned"
        final_df.to_parquet(
            final_path,
            partition_cols=self.config.partition_cols,
            compression=self.config.compression,
            engine='pyarrow'
        )
        
        # Save metadata
        metadata = {
            'collection_stats': self.collection_stats,
            'config': {
                'lookback_days': self.config.lookback_days,
                'market_open': self.config.market_open.isoformat(),
                'market_close': self.config.market_close.isoformat(),
                'signal_classes': {
                    0: 'UP',
                    1: 'DOWN',
                    2: 'NEUTRAL'
                }
            },
            'schema': {
                'features': list(final_df.columns),
                'total_records': len(final_df),
                'date_range': {
                    'start': final_df['timestamp'].min().isoformat(),
                    'end': final_df['timestamp'].max().isoformat()
                }
            }
        }
        
        metadata_path = Path(self.config.data_dir) / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Cleanup temp files
        for file_path in temp_files:
            file_path.unlink()
        
        logger.info(f"Dataset finalized: {len(final_df)} total records")
        logger.info(f"Data saved to: {final_path}")
        logger.info(f"Metadata saved to: {metadata_path}")

async def main():
    """Main execution for full mid-cap universe data collection"""
    
    # Configuration for full mid-cap universe (2000-3000+ stocks)
    config = DataCollectionConfig(
        batch_size=25,          # Smaller batches for stability
        max_workers=15,         # Conservative worker count for API limits
        lookback_days=180       # Exactly 6 months
    )
    
    logger.info("="*80)
    logger.info("FULL MID-CAP UNIVERSE DATA COLLECTION")
    logger.info("="*80)
    logger.info(f"Target: ALL mid-cap stocks ($2B-$10B market cap)")
    logger.info(f"Time window: 9:40 AM - 11:40 AM EST (2 hours daily)")
    logger.info(f"Historical period: 6 months ({config.lookback_days} days)")
    logger.info(f"Expected dataset size: ~500GB+ (2000-3000 stocks)")
    logger.info("="*80)
    
    # Initialize collector
    collector = EnterpriseDataCollector(config)
    
    # Run collection
    stats = await collector.collect_midcap_universe()
    
    print("\n" + "="*80)
    print("FULL MID-CAP UNIVERSE DATA COLLECTION COMPLETE")
    print("="*80)
    print(f"Stocks processed: {stats['stocks_processed']}")
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Duration: {stats['duration_seconds']:.1f} seconds ({stats['duration_seconds']/3600:.1f} hours)")
    print(f"Errors: {len(stats['errors'])}")
    success_rate = (stats['stocks_processed']/stats['stocks_total']*100) if stats['stocks_total'] > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    print("="*80)
    print("WARNING: This dataset is suitable for production trading models")
    print("covering the complete mid-cap universe!")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())