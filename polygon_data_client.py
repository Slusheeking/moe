"""
GPU-Accelerated Polygon.io API Client with RAPIDS/CuPy Integration
Optimized for A10 GPU with cuDF, CuPy, and Numba acceleration
"""

import os
import json
import asyncio
import warnings
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from polygon import RESTClient
from polygon.rest.models import Ticker
import pickle

# GPU-accelerated imports with fallbacks
try:
    import cudf as pd
    import cupy as np
    print("🚀 Using RAPIDS cuDF and CuPy for GPU acceleration!")
    GPU_AVAILABLE = True
except ImportError:
    import pandas as pd
    import numpy as np
    print("⚠️  Using CPU fallback - install RAPIDS for GPU acceleration")
    GPU_AVAILABLE = False

try:
    from numba import cuda, jit
    NUMBA_AVAILABLE = cuda.is_available()
    if NUMBA_AVAILABLE:
        print(f"⚡ Numba CUDA available - GPU detected: {cuda.get_current_device().name.decode()}")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️  Numba not available")

# Suppress GPU warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN slice encountered')
warnings.filterwarnings('ignore', category=FutureWarning, module='cudf')

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TickerFilter:
    """Enhanced ticker filter with market cap support"""
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_volume: Optional[int] = None
    min_market_cap: Optional[int] = None
    max_market_cap: Optional[int] = None
    exclude_otc: bool = True
    exclude_etfs: bool = True

class PolygonClient:
    """
    Enhanced Polygon.io API client with market cap filtering support
    """
    def __init__(self):
        self.api_key = os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY environment variable not set.")
        
        self.client = RESTClient(self.api_key)
        self.cache_dir = Path("polygon_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache files
        self.tickers_cache_file = self.cache_dir / f"tickers_{datetime.now().strftime('%Y%m%d')}.pkl"
        self.ticker_details_cache = self.cache_dir / "ticker_details_cache.json"
        
        # Load cached ticker details if available
        self.details_cache = {}
        if self.ticker_details_cache.exists():
            try:
                with open(self.ticker_details_cache, 'r') as f:
                    self.details_cache = json.load(f)
                logger.info(f"Loaded {len(self.details_cache)} cached ticker details")
            except Exception as e:
                logger.warning(f"Could not load ticker details cache: {e}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Save ticker details cache
        try:
            with open(self.ticker_details_cache, 'w') as f:
                json.dump(self.details_cache, f, indent=2)
            logger.info(f"Saved {len(self.details_cache)} ticker details to cache")
        except Exception as e:
            logger.warning(f"Could not save ticker details cache: {e}")

    async def get_ticker_details(self, ticker: str) -> Optional[Dict]:
        """Get detailed ticker information including market cap"""
        if ticker in self.details_cache:
            return self.details_cache[ticker]
        
        try:
            # Use Polygon's ticker details endpoint
            details = self.client.get_ticker_details(ticker)
            
            if details:
                ticker_info = {
                    'ticker': ticker,
                    'market_cap': getattr(details, 'market_cap', None),
                    'share_class_shares_outstanding': getattr(details, 'share_class_shares_outstanding', None),
                    'weighted_shares_outstanding': getattr(details, 'weighted_shares_outstanding', None),
                    'type': getattr(details, 'type', None),
                    'active': getattr(details, 'active', True),
                    'currency_name': getattr(details, 'currency_name', 'USD'),
                    'last_updated': datetime.now().isoformat()
                }
                
                # Cache the result
                self.details_cache[ticker] = ticker_info
                return ticker_info
        except Exception as e:
            error_str = str(e)
            # Handle common "NOT_FOUND" errors more gracefully
            if "NOT_FOUND" in error_str or "Ticker not found" in error_str:
                # These are normal for delisted/inactive stocks - log at debug level
                logger.debug(f"Ticker {ticker} not found (likely delisted/inactive)")
            else:
                # Log unexpected errors at warning level
                logger.warning(f"Unexpected error getting details for {ticker}: {e}")
        
        return None

    async def get_all_tickers(self, ticker_filter: Optional[TickerFilter] = None) -> List[Dict]:
        """Get all tickers with market cap filtering support"""
        
        # Try to load from cache first
        if self.tickers_cache_file.exists():
            try:
                with open(self.tickers_cache_file, 'rb') as f:
                    cached_tickers = pickle.load(f)
                logger.info(f"Loaded {len(cached_tickers)} tickers from cache")
                return cached_tickers
            except Exception as e:
                logger.warning(f"Could not load tickers from cache: {e}")

        logger.info("Fetching fresh ticker data from Polygon...")
        
        # Get basic ticker list
        basic_tickers = []
        try:
            # Use v3 API for better filtering - only get ACTIVE common stocks
            import requests
            
            url = "https://api.polygon.io/v3/reference/tickers"
            params = {
                'market': 'stocks',     # Stocks only (not crypto/fx/indices)
                'active': 'true',       # Only actively traded stocks
                'type': 'CS',          # Common stocks only
                'order': 'asc',
                'limit': 1000,
                'sort': 'ticker',
                'apikey': self.api_key
            }
            
            all_v3_tickers = []
            next_url = None
            
            while True:
                if next_url:
                    response = requests.get(next_url)
                else:
                    response = requests.get(url, params=params)
                
                if response.status_code != 200:
                    logger.error(f"API Error: {response.status_code} - {response.text}")
                    break
                
                data = response.json()
                
                if 'results' not in data:
                    break
                
                # Process each ticker from v3 API
                for ticker_data in data['results']:
                    ticker = ticker_data.get('ticker')
                    active = ticker_data.get('active', False)
                    market = ticker_data.get('market')
                    type_code = ticker_data.get('type')
                    
                    # Strict filtering for quality stocks
                    if not (ticker and active and market == 'stocks' and type_code == 'CS'):
                        continue
                    
                    if len(ticker) > 5 or not ticker.isalpha() or ticker.endswith('.'):
                        continue
                    
                    all_v3_tickers.append(ticker_data)
                
                # Check for pagination
                next_url = data.get('next_url')
                if not next_url:
                    break
                
                # Add API key to next_url if not present
                if 'apikey=' not in next_url:
                    connector = '&' if '?' in next_url else '?'
                    next_url += f"{connector}apikey={self.api_key}"
            
            logger.info(f"Retrieved {len(all_v3_tickers)} quality active stocks from v3 API")
            
            # Convert v3 API results to our format
            for ticker_data in all_v3_tickers:
                ticker = ticker_data.get('ticker')
                
                # Create basic ticker info from v3 API data
                basic_ticker_info = {
                    'ticker': ticker,
                    'type': ticker_data.get('type', 'CS'),
                    'name': ticker_data.get('name'),
                    'market': ticker_data.get('market', 'stocks'),
                    'active': ticker_data.get('active', True)
                }
                
                # Apply basic filtering if we have the data - v3 API doesn't include price/volume
                # We'll get that during market cap filtering phase
                
                # ETF and OTC filtering
                if ticker_filter and ticker_filter.exclude_etfs:
                    # Common ETF patterns
                    if (ticker.endswith('X') or ticker.startswith('SPY') or
                        ticker.startswith('QQQ') or ticker.startswith('IWM') or
                        ticker.startswith('VT') or len(ticker) > 5):
                        continue
                
                if ticker_filter and ticker_filter.exclude_otc:
                    if ticker.endswith('.') or not ticker.isalpha():
                        continue
                
                basic_tickers.append(basic_ticker_info)
        
        except Exception as e:
            logger.error(f"Error fetching basic tickers: {e}")
            return []
        
        logger.info(f"Got {len(basic_tickers)} basic tickers, applying market cap filtering...")
        
        # Apply market cap filtering if needed
        filtered_tickers = []
        if ticker_filter and (ticker_filter.min_market_cap or ticker_filter.max_market_cap):
            
            # Process in batches to avoid API rate limits
            batch_size = 50
            processed = 0
            
            for i in range(0, len(basic_tickers), batch_size):
                batch = basic_tickers[i:i + batch_size]
                
                for ticker_info in batch:
                    processed += 1
                    if processed % 500 == 0:
                        logger.info(f"Processed {processed}/{len(basic_tickers)} tickers for market cap")
                    
                    # Get detailed info including market cap
                    details = await self.get_ticker_details(ticker_info['ticker'])
                    
                    if details and details.get('market_cap'):
                        market_cap = details['market_cap']
                        
                        # Apply market cap filters
                        if ticker_filter.min_market_cap and market_cap < ticker_filter.min_market_cap:
                            continue
                        if ticker_filter.max_market_cap and market_cap > ticker_filter.max_market_cap:
                            continue
                        
                        # Add market cap to ticker info
                        ticker_info['market_cap'] = market_cap
                        filtered_tickers.append(ticker_info)
                
                # Add small delay to respect API rate limits
                await asyncio.sleep(0.1)
        else:
            # No market cap filtering needed
            filtered_tickers = basic_tickers
        
        # Cache the results
        try:
            with open(self.tickers_cache_file, 'wb') as f:
                pickle.dump(filtered_tickers, f)
            logger.info(f"Cached {len(filtered_tickers)} filtered tickers")
        except Exception as e:
            logger.warning(f"Could not cache tickers: {e}")
        
        logger.info(f"Final filtered ticker count: {len(filtered_tickers)}")
        return filtered_tickers

    async def get_aggregates(self, ticker: str, timespan: str, from_date: str, to_date: Optional[str] = None) -> pd.DataFrame:
        """Get aggregate data for a ticker"""
        try:
            if to_date is None:
                to_date = datetime.now().strftime('%Y-%m-%d')
            
            aggs = []
            for a in self.client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan=timespan,
                from_=from_date,
                to=to_date,
                limit=50000
            ):
                # Convert Polygon Agg object to dict for cuDF compatibility
                agg_dict = {
                    'timestamp': a.timestamp,
                    'o': a.open,
                    'h': a.high,
                    'l': a.low,
                    'c': a.close,
                    'v': a.volume
                }
                aggs.append(agg_dict)
            
            if not aggs:
                return pd.DataFrame()

            df = pd.DataFrame(aggs)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Error fetching aggregates for {ticker}: {e}")
            return pd.DataFrame()

    def save_historical_gaps(self, gap_data: List[Dict], filename='historical_gaps.json'):
        """Save collected historical gap data"""
        filepath = self.cache_dir / filename
        with open(filepath, 'w') as f:
            json.dump(gap_data, f, indent=4)
        logger.info(f"Saved {len(gap_data)} historical gap records to {filepath}")

class EnhancedFeatureEngineering:
    """GPU-Accelerated feature engineering for A10 optimization"""
    
    @staticmethod
    def create_gap_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create specialized features with GPU acceleration"""
        
        # Basic price features - GPU optimized
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Gap-specific features
        df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['gap_fill'] = ((df['low'] <= df['close'].shift(1)) & (df['overnight_gap'] > 0)) | \
                         ((df['high'] >= df['close'].shift(1)) & (df['overnight_gap'] < 0))
        
        # Intraday momentum
        df['open_to_high'] = (df['high'] - df['open']) / df['open']
        df['open_to_low'] = (df['low'] - df['open']) / df['open']
        df['close_vs_open'] = (df['close'] - df['open']) / df['open']
        
        # Volume features with NaN handling
        df['volume_ma_20'] = df['volume'].rolling(20, min_periods=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20'].fillna(1.0)
        df['dollar_volume'] = df['close'] * df['volume']
        
        # GPU-optimized volume momentum
        if GPU_AVAILABLE:
            df['volume_momentum'] = df['volume'].pct_change().fillna(0.0)
        else:
            df['volume_momentum'] = df['volume'].pct_change(fill_method=None).fillna(0.0)
        
        # Volatility features with GPU optimization
        df['true_range'] = (df[['high', 'close']].shift(1)).max(axis=1) - \
                          (df[['low', 'close']].shift(1)).min(axis=1)
        df['atr_14'] = df['true_range'].rolling(14, min_periods=7).mean().fillna(0.0)
        df['volatility_20'] = df['returns'].rolling(20, min_periods=10).std().fillna(0.0)
        volatility_60_mean = df['volatility_20'].rolling(60, min_periods=30).mean().fillna(1.0)
        df['volatility_ratio'] = df['volatility_20'] / volatility_60_mean
        
        # GPU-accelerated technical indicators
        df['rsi'] = EnhancedFeatureEngineering._calculate_rsi_gpu(df['close'], 14)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # VWAP deviation with GPU optimization
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cumulative_volume = df['volume'].cumsum()
        cumulative_volume = cumulative_volume.where(cumulative_volume > 0, 1.0)
        df['vwap'] = (df['volume'] * typical_price).cumsum() / cumulative_volume
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap'].fillna(1.0)
        
        # Time-based features
        try:
            if hasattr(df.index, 'hour'):
                df['hour'] = df.index.hour
                df['minute'] = df.index.minute
                df['day_of_week'] = df.index.dayofweek
                df['is_premarket'] = ((df.index.hour >= 4) & (df.index.hour < 9)) | \
                                    ((df.index.hour == 9) & (df.index.minute < 30))
                df['is_first_hour'] = (df.index.hour == 9) & (df.index.minute >= 30) | \
                                     (df.index.hour == 10) & (df.index.minute < 30)
            else:
                # Fallback values
                df['hour'] = 9
                df['minute'] = 30
                df['day_of_week'] = 1
                df['is_premarket'] = False
                df['is_first_hour'] = True
        except Exception:
            # Safe fallback
            df['hour'] = 9
            df['minute'] = 30
            df['day_of_week'] = 1
            df['is_premarket'] = False
            df['is_first_hour'] = True
        
        # Gap history features - simplified for GPU compatibility
        df['gap_streak'] = (df['overnight_gap'] > 0.01).astype(int)
        
        # Rolling gap statistics with proper NaN handling
        df['gap_mean_5d'] = df['overnight_gap'].rolling(5, min_periods=3).mean().fillna(0.0)
        df['gap_std_5d'] = df['overnight_gap'].rolling(5, min_periods=3).std().fillna(0.0)
        df['gap_max_5d'] = df['overnight_gap'].rolling(5, min_periods=3).max().fillna(0.0)
        
        # Enhanced Momentum Detection Features
        df['price_momentum_5'] = (df['close'] / df['close'].shift(5)) - 1  # 5-period momentum
        df['price_momentum_20'] = (df['close'] / df['close'].shift(20)) - 1  # 20-period momentum
        df['volume_surge'] = df['volume'] / df['volume_ma_20']  # Volume surge factor
        df['volume_surge_signal'] = (df['volume_surge'] > 3.0).astype(int)  # 3x volume surge
        
        # Breakout detection
        df['price_range_20'] = df['high'].rolling(20).max() - df['low'].rolling(20).min()
        df['close_position'] = (df['close'] - df['low'].rolling(20).min()) / df['price_range_20']
        df['breakout_signal'] = (df['close_position'] > 0.8).astype(int)  # Near 20-day high
        
        # Relative strength (vs own history)
        df['relative_strength'] = df['close'] / df['close'].rolling(50).mean()  # vs 50-day MA
        df['momentum_strength'] = (df['relative_strength'] > 1.05).astype(int)  # 5% above MA
        
        return df
    
    @staticmethod
    def _calculate_rsi_gpu(prices, period=14):
        """GPU-accelerated RSI calculation"""
        delta = prices.diff().fillna(0.0)
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period//2).mean().fillna(0.0)
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period//2).mean().fillna(0.0)
        
        # GPU-safe division
        loss_safe = loss.where(loss > 0, 1e-10)
        rs = gain / loss_safe
        rsi = 100 - (100 / (1 + rs))
        
        # Handle edge cases
        rsi = rsi.fillna(50.0)  # Neutral RSI
        rsi = rsi.clip(0, 100)  # Ensure valid range
        
        return rsi

# GPU-accelerated Numba functions for heavy computations
if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def fast_rolling_mean(arr, window):
        """Ultra-fast rolling mean with Numba JIT"""
        n = len(arr)
        result = np.empty(n)
        result[:window-1] = np.nan
        
        for i in range(window-1, n):
            result[i] = np.mean(arr[i-window+1:i+1])
        
        return result
    
    @jit(nopython=True)
    def fast_rsi_calculation(prices, period=14):
        """Ultra-fast RSI calculation with Numba"""
        n = len(prices)
        rsi = np.empty(n)
        rsi[:period] = 50.0  # Default neutral
        
        deltas = np.diff(prices)
        
        for i in range(period, n):
            gains = deltas[i-period:i]
            gains = gains[gains > 0]
            losses = -deltas[i-period:i]
            losses = losses[losses > 0]
            
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
            
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    print("🔥 Numba JIT-compiled functions loaded for maximum performance!")
else:
    # CPU fallback functions
    def fast_rolling_mean(arr, window):
        return np.convolve(arr, np.ones(window)/window, mode='same')
    
    def fast_rsi_calculation(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.convolve(gain, np.ones(period)/period, mode='same')
        avg_loss = np.convolve(loss, np.ones(period)/period, mode='same')
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

def calculate_gap_metrics(previous_close: float, gap_open: float, next_high: float, next_low: float, final_close: float) -> Dict:
    """Calculate gap-related metrics"""
    gap_size = (gap_open - previous_close) / previous_close
    
    # Determine if the gap continued or faded
    if gap_size > 0:  # Up gap
        continuation = final_close > gap_open and next_high > gap_open
    else:  # Down gap
        continuation = final_close < gap_open and next_low < gap_open
        
    return {
        'gap_size': gap_size,
        'continuation': continuation,
        'gap_fill_percentage': 0.0
    }

async def test_midcap_filtering():
    """Test the market cap filtering functionality"""
    logger.info("Testing mid-cap market cap filtering...")
    
    async with PolygonClient() as client:
        # Test mid-cap filter
        midcap_filter = TickerFilter(
            min_market_cap=2_000_000_000,   # $2B
            max_market_cap=10_000_000_000,  # $10B
            min_price=5.0,
            max_price=500.0,
            min_volume=500_000,
            exclude_etfs=True,
            exclude_otc=True
        )
        
        tickers = await client.get_all_tickers(midcap_filter)
        
        logger.info(f"Found {len(tickers)} mid-cap stocks")
        
        # Show sample results
        for i, ticker in enumerate(tickers[:10]):
            market_cap = ticker.get('market_cap', 'Unknown')
            if isinstance(market_cap, (int, float)):
                market_cap_b = market_cap / 1_000_000_000
                logger.info(f"{ticker['ticker']}: ${market_cap_b:.1f}B market cap")
            else:
                logger.info(f"{ticker['ticker']}: Market cap {market_cap}")
        
        return tickers

if __name__ == "__main__":
    # Test the market cap filtering
    asyncio.run(test_midcap_filtering())