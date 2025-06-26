#!/usr/bin/env python3
"""
GPU-Accelerated Parallel Data Pipeline with Polygon Integration
Handles REST API historical data and WebSocket real-time streams using Polygon.io
Includes options data integration for enhanced gap analysis.
"""

import os
import sys
import torch
import cupy as cp
import numpy as np
import pandas as pd
import asyncio
import websockets
import json
import requests
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import pytz
from polygon import RESTClient, WebSocketClient
from polygon.websocket.models import Market, EquityTrade, EquityQuote, OptionsQuote
from concurrent.futures import ThreadPoolExecutor
import threading
import aiohttp

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpu_trading_config import GPUTradingConfig
from gpu_core.tensor_universe_manager import TensorUniverseManager


class PolygonDataPipeline:
    """GPU-accelerated data pipeline using Polygon.io for comprehensive market data."""
    
    def __init__(self, config: GPUTradingConfig, universe_manager: TensorUniverseManager):
        self.config = config
        self.universe_manager = universe_manager
        self.device = config.device
        self.dtype = config.TENSOR_DTYPE
        
        # Initialize Polygon clients
        self.polygon_rest = RESTClient(config.POLYGON_API_KEY)
        self.polygon_ws = None  # Will initialize when starting real-time data
        
        # Get current universe
        self.universe_tensor, self.universe_symbols = universe_manager.get_universe_tensor()
        self.symbol_to_idx = {symbol: idx for idx, symbol in enumerate(self.universe_symbols)}
        
        # Initialize GPU data structures
        self._init_gpu_structures()
        
        # WebSocket connection management
        self.ws_connected = False
        self.data_lock = threading.Lock()
        
        # Historical data cache
        self.historical_cache = {}
        self.options_cache = {}  # Cache for options data
        self.cache_lock = threading.Lock()
        
        # Options tracking for gap analysis
        self.unusual_options_activity = {}
        self.options_flow = defaultdict(lambda: {'calls': 0, 'puts': 0})
        
    def _init_gpu_structures(self):
        """Initialize GPU-resident data structures for real-time processing."""
        n_symbols = len(self.universe_symbols)
        
        # Pre-market data tensors (stored on GPU)
        self.pre_market_prices = torch.zeros((n_symbols, 1000), device=self.device, dtype=self.dtype)
        self.pre_market_volumes = torch.zeros((n_symbols, 1000), device=self.device, dtype=self.dtype)
        self.pre_market_timestamps = torch.zeros((n_symbols, 1000), device=self.device, dtype=torch.int64)
        self.pre_market_counts = torch.zeros(n_symbols, device=self.device, dtype=torch.int32)
        
        # Trade size tracking for institutional detection
        self.pre_market_trade_sizes = torch.zeros((n_symbols, 1000), device=self.device, dtype=self.dtype)
        
        # Intraday data tensors
        self.intraday_prices = torch.zeros((n_symbols, 10000), device=self.device, dtype=self.dtype)
        self.intraday_volumes = torch.zeros((n_symbols, 10000), device=self.device, dtype=self.dtype)
        self.intraday_timestamps = torch.zeros((n_symbols, 10000), device=self.device, dtype=torch.int64)
        self.intraday_counts = torch.zeros(n_symbols, device=self.device, dtype=torch.int32)
        
        # Real-time feature tensors (expanded to include options data)
        self.n_features = 25  # Increased from 20 to include options features
        self.realtime_features = torch.zeros((n_symbols, self.n_features), 
                                           device=self.device, dtype=self.dtype)
        
        # Feature names for reference
        self.feature_names = [
            'last_price', 'bid', 'ask', 'spread', 'spread_pct',
            'volume_5min', 'volume_15min', 'vwap_5min', 'vwap_15min',
            'price_momentum_5min', 'volume_momentum_5min',
            'high_5min', 'low_5min', 'range_5min', 'volume_profile_skew',
            'trade_count_5min', 'avg_trade_size', 'large_trade_ratio',
            'price_volatility_5min', 'microstructure_score',
            # New options-based features
            'options_volume_ratio', 'put_call_ratio', 'options_flow_score',
            'implied_volatility_rank', 'unusual_options_score'
        ]
        
        # Tick-by-tick storage for microstructure analysis
        self.tick_buffer_size = 1000
        self.tick_prices = torch.zeros((n_symbols, self.tick_buffer_size), device=self.device, dtype=self.dtype)
        self.tick_sizes = torch.zeros((n_symbols, self.tick_buffer_size), device=self.device, dtype=self.dtype)
        self.tick_timestamps = torch.zeros((n_symbols, self.tick_buffer_size), device=self.device, dtype=torch.int64)
        self.tick_counts = torch.zeros(n_symbols, device=self.device, dtype=torch.int32)
        
    async def start_realtime_data(self):
        """Start WebSocket connections for real-time data using Polygon."""
        print(f"Starting real-time data for {len(self.universe_symbols)} symbols via Polygon...")
        
        # Initialize Polygon WebSocket client
        self.polygon_ws = WebSocketClient(
            api_key=self.config.POLYGON_API_KEY,
            market=Market.Stocks,
            raw_data_handler=self._handle_polygon_message
        )
        
        # Subscribe to trades and quotes for all universe symbols
        # Polygon uses different subscription format: T.*, Q.*, etc.
        subscriptions = []
        
        for symbol in self.universe_symbols:
            subscriptions.append(f"T.{symbol}")  # Trades
            subscriptions.append(f"Q.{symbol}")  # Quotes
            subscriptions.append(f"A.{symbol}")  # Aggregate (minute bars)
            
            # Also subscribe to options flow for key symbols
            # This helps identify institutional positioning
            if self._is_high_volume_symbol(symbol):
                subscriptions.append(f"T.O:{symbol}*")  # Options trades
        
        # Connect and subscribe
        self.ws_connected = True
        try:
            await self.polygon_ws.connect()
            await self.polygon_ws.subscribe(subscriptions)
            
            # Keep the connection alive
            while self.ws_connected:
                await asyncio.sleep(1)
                
        except Exception as e:
            print(f"Polygon WebSocket error: {e}")
            self.ws_connected = False
            
    def _handle_polygon_message(self, message: List[Dict]):
        """Process raw Polygon WebSocket messages."""
        for msg in message:
            event_type = msg.get('ev')
            
            if event_type == 'T':  # Stock trade
                self._process_stock_trade(msg)
            elif event_type == 'Q':  # Stock quote
                self._process_stock_quote(msg)
            elif event_type == 'A':  # Aggregate bar
                self._process_aggregate_bar(msg)
            elif event_type == 'T' and msg.get('sym', '').startswith('O:'):  # Options trade
                self._process_options_trade(msg)
                
    def _process_stock_trade(self, trade_data: Dict):
        """Process stock trade data from Polygon."""
        symbol = trade_data.get('sym', '')
        if symbol not in self.symbol_to_idx:
            return
            
        idx = self.symbol_to_idx[symbol]
        
        # Extract trade information
        price = float(trade_data.get('p', 0))
        size = int(trade_data.get('s', 0))
        timestamp = int(trade_data.get('t', 0)) // 1000000  # Polygon uses nanoseconds
        conditions = trade_data.get('c', [])
        
        # Check if this is a block trade (institutional)
        is_block = size >= 10000 or any(c in [12, 13] for c in conditions)  # Polygon condition codes
        
        current_time = datetime.now(self.config.TIMEZONE)
        
        # Store tick data for microstructure analysis
        with self.data_lock:
            tick_idx = self.tick_counts[idx] % self.tick_buffer_size
            self.tick_prices[idx, tick_idx] = price
            self.tick_sizes[idx, tick_idx] = size
            self.tick_timestamps[idx, tick_idx] = timestamp
            self.tick_counts[idx] += 1
        
        # Determine if pre-market or regular hours
        if current_time.time() < self.config.MARKET_OPEN:
            # Pre-market data
            with self.data_lock:
                count = self.pre_market_counts[idx].item()
                if count < 1000:  # Prevent overflow
                    self.pre_market_prices[idx, count] = price
                    self.pre_market_volumes[idx, count] = size
                    self.pre_market_timestamps[idx, count] = timestamp
                    self.pre_market_trade_sizes[idx, count] = size if is_block else 0
                    self.pre_market_counts[idx] += 1
        else:
            # Regular hours data
            with self.data_lock:
                count = self.intraday_counts[idx].item()
                if count < 10000:  # Prevent overflow
                    self.intraday_prices[idx, count] = price
                    self.intraday_volumes[idx, count] = size
                    self.intraday_timestamps[idx, count] = timestamp
                    self.intraday_counts[idx] += 1
        
        # Update real-time features if we have enough data
        if self.pre_market_counts[idx] > 10 or self.intraday_counts[idx] > 10:
            self._update_realtime_features_gpu(idx)
            
    def _process_stock_quote(self, quote_data: Dict):
        """Process stock quote data from Polygon."""
        symbol = quote_data.get('sym', '')
        if symbol not in self.symbol_to_idx:
            return
            
        idx = self.symbol_to_idx[symbol]
        
        # Extract quote information
        bid_price = float(quote_data.get('bp', 0))
        ask_price = float(quote_data.get('ap', 0))
        bid_size = int(quote_data.get('bs', 0))
        ask_size = int(quote_data.get('as', 0))
        
        # Update bid/ask/spread in real-time features
        with self.data_lock:
            self.realtime_features[idx, 1] = bid_price  # bid
            self.realtime_features[idx, 2] = ask_price  # ask
            self.realtime_features[idx, 3] = ask_price - bid_price  # spread
            if bid_price > 0:
                self.realtime_features[idx, 4] = (ask_price - bid_price) / bid_price * 100  # spread_pct
                
            # Analyze quote imbalance for order flow
            total_size = bid_size + ask_size
            if total_size > 0:
                self.realtime_features[idx, 14] = (bid_size - ask_size) / total_size  # volume_profile_skew
                
    def _process_aggregate_bar(self, agg_data: Dict):
        """Process aggregate bar data from Polygon."""
        # This can be used for minute-by-minute analysis
        # Implementation depends on specific requirements
        pass
                
    def _process_options_trade(self, options_data: Dict):
        """Process options trade data to detect unusual activity."""
        # Extract the underlying symbol from options symbol
        options_symbol = options_data.get('sym', '')
        underlying = self._extract_underlying_from_options(options_symbol)
        
        if underlying not in self.symbol_to_idx:
            return
            
        idx = self.symbol_to_idx[underlying]
        
        # Track options flow
        size = int(options_data.get('s', 0))
        price = float(options_data.get('p', 0))
        
        # Determine if call or put
        if 'C' in options_symbol:
            self.options_flow[underlying]['calls'] += size
        else:
            self.options_flow[underlying]['puts'] += size
            
        # Detect unusual options activity (large trades)
        if size * price * 100 > 100000:  # Over $100k premium
            if underlying not in self.unusual_options_activity:
                self.unusual_options_activity[underlying] = []
            self.unusual_options_activity[underlying].append({
                'time': datetime.now(),
                'size': size,
                'premium': size * price * 100
            })
            
        # Update options-based features
        self._update_options_features(idx, underlying)
        
    def _update_realtime_features_gpu(self, symbol_idx: int):
        """Calculate real-time features for a symbol using GPU operations."""
        current_time = int(datetime.now().timestamp())
        
        # Determine which data to use (pre-market or intraday)
        if datetime.now(self.config.TIMEZONE).time() < self.config.MARKET_OPEN:
            prices = self.pre_market_prices[symbol_idx]
            volumes = self.pre_market_volumes[symbol_idx]
            timestamps = self.pre_market_timestamps[symbol_idx]
            count = self.pre_market_counts[symbol_idx]
            trade_sizes = self.pre_market_trade_sizes[symbol_idx]
        else:
            prices = self.intraday_prices[symbol_idx]
            volumes = self.intraday_volumes[symbol_idx]
            timestamps = self.intraday_timestamps[symbol_idx]
            count = self.intraday_counts[symbol_idx]
            trade_sizes = None  # Not tracked intraday for memory efficiency
        
        if count < 2:
            return
        
        # Get valid data (non-zero timestamps)
        valid_mask = timestamps > 0
        valid_prices = prices[valid_mask]
        valid_volumes = volumes[valid_mask]
        valid_timestamps = timestamps[valid_mask]
        
        # Calculate time-based masks for different windows
        time_5min = current_time - 300
        time_15min = current_time - 900
        
        mask_5min = valid_timestamps >= time_5min
        mask_15min = valid_timestamps >= time_15min
        
        # Update features using GPU operations
        with self.data_lock:
            # Last price
            self.realtime_features[symbol_idx, 0] = valid_prices[-1]
            
            # Volume calculations
            self.realtime_features[symbol_idx, 5] = valid_volumes[mask_5min].sum()  # volume_5min
            self.realtime_features[symbol_idx, 6] = valid_volumes[mask_15min].sum()  # volume_15min
            
            # VWAP calculations
            if mask_5min.sum() > 0:
                vwap_5min = (valid_prices[mask_5min] * valid_volumes[mask_5min]).sum() / valid_volumes[mask_5min].sum()
                self.realtime_features[symbol_idx, 7] = vwap_5min
            
            if mask_15min.sum() > 0:
                vwap_15min = (valid_prices[mask_15min] * valid_volumes[mask_15min]).sum() / valid_volumes[mask_15min].sum()
                self.realtime_features[symbol_idx, 8] = vwap_15min
            
            # Price momentum (5min price change)
            if mask_5min.sum() > 1:
                first_price_5min = valid_prices[mask_5min][0]
                last_price = valid_prices[-1]
                self.realtime_features[symbol_idx, 9] = (last_price - first_price_5min) / first_price_5min * 100
            
            # Volume momentum (5min volume acceleration)
            if mask_5min.sum() > 10:
                recent_vol = valid_volumes[mask_5min][-5:].mean()
                older_vol = valid_volumes[mask_5min][:5].mean()
                if older_vol > 0:
                    self.realtime_features[symbol_idx, 10] = recent_vol / older_vol
            
            # High/Low/Range
            if mask_5min.sum() > 0:
                self.realtime_features[symbol_idx, 11] = valid_prices[mask_5min].max()  # high_5min
                self.realtime_features[symbol_idx, 12] = valid_prices[mask_5min].min()  # low_5min
                self.realtime_features[symbol_idx, 13] = valid_prices[mask_5min].max() - valid_prices[mask_5min].min()  # range_5min
            
            # Trade statistics
            self.realtime_features[symbol_idx, 15] = mask_5min.sum()  # trade_count_5min
            if mask_5min.sum() > 0:
                self.realtime_features[symbol_idx, 16] = valid_volumes[mask_5min].mean()  # avg_trade_size
                
                # Large trade ratio (institutional footprint)
                if trade_sizes is not None:
                    valid_trade_sizes = trade_sizes[valid_mask]
                    large_trades = (valid_trade_sizes[mask_5min] > 0).sum()  # Non-zero means block trade
                    self.realtime_features[symbol_idx, 17] = large_trades / mask_5min.sum()
                else:
                    # Fallback: use volume threshold
                    large_trades = (valid_volumes[mask_5min] >= 1000).sum()
                    self.realtime_features[symbol_idx, 17] = large_trades / mask_5min.sum()
            
            # Price volatility (5min) using tick data for more accuracy
            if self.tick_counts[symbol_idx] > 10:
                recent_ticks = min(100, self.tick_counts[symbol_idx].item())
                tick_prices = self.tick_prices[symbol_idx, :recent_ticks]
                returns = torch.diff(torch.log(tick_prices[tick_prices > 0]))
                if len(returns) > 2:
                    self.realtime_features[symbol_idx, 18] = returns.std() * torch.sqrt(torch.tensor(252.0 * 78.0))  # annualized
            
            # Microstructure score (enhanced with tick data)
            spread_score = 1 / (1 + self.realtime_features[symbol_idx, 4])  # inverse spread percentage
            size_score = torch.tanh(self.realtime_features[symbol_idx, 16] / 1000)  # normalized trade size
            
            # Add tick frequency component
            tick_frequency = self.tick_counts[symbol_idx] / max(1, (current_time - self.tick_timestamps[symbol_idx, 0].item()))
            frequency_score = torch.tanh(tick_frequency * 100)  # Normalize to 0-1
            
            self.realtime_features[symbol_idx, 19] = (spread_score + size_score + frequency_score) / 3
            
    def _update_options_features(self, idx: int, symbol: str):
        """Update options-based features for gap analysis."""
        with self.data_lock:
            # Options volume ratio (options volume vs stock volume)
            stock_volume = self.realtime_features[idx, 5]  # volume_5min
            options_volume = self.options_flow[symbol]['calls'] + self.options_flow[symbol]['puts']
            if stock_volume > 0:
                self.realtime_features[idx, 20] = options_volume / stock_volume
            
            # Put/Call ratio
            calls = self.options_flow[symbol]['calls']
            puts = self.options_flow[symbol]['puts']
            if calls > 0:
                self.realtime_features[idx, 21] = puts / calls
            
            # Options flow score (bullish vs bearish)
            total_options = calls + puts
            if total_options > 0:
                self.realtime_features[idx, 22] = (calls - puts) / total_options
            
            # Unusual options activity score
            if symbol in self.unusual_options_activity:
                recent_unusual = len([a for a in self.unusual_options_activity[symbol] 
                                    if (datetime.now() - a['time']).seconds < 300])
                self.realtime_features[idx, 24] = min(1.0, recent_unusual / 5)  # Normalize to 0-1
                
    def fetch_historical_data(self, lookback_days: int = None) -> Dict[str, pd.DataFrame]:
        """Fetch historical daily data using Polygon aggregates."""
        if lookback_days is None:
            lookback_days = self.config.LOOKBACK_DAYS
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        print(f"Fetching historical data for {len(self.universe_symbols)} symbols from Polygon...")
        
        historical_data = {}
        
        # Use ThreadPoolExecutor for parallel API calls
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {}
            for symbol in self.universe_symbols:
                future = executor.submit(self._fetch_symbol_history_polygon, symbol, start_date, end_date)
                futures[future] = symbol
            
            # Collect results
            for future in futures:
                symbol = futures[future]
                try:
                    data = future.result()
                    if data is not None:
                        historical_data[symbol] = data
                except Exception as e:
                    print(f"Error fetching {symbol}: {e}")
        
        # Cache the data
        with self.cache_lock:
            self.historical_cache = historical_data
        
        print(f"Fetched historical data for {len(historical_data)} symbols")
        return historical_data
        
    def _fetch_symbol_history_polygon(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch historical data for a single symbol using Polygon."""
        try:
            # Polygon aggregates endpoint
            aggs = self.polygon_rest.list_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                limit=5000
            )
            
            # Convert to DataFrame
            data = []
            for agg in aggs:
                data.append({
                    'timestamp': datetime.fromtimestamp(agg.timestamp / 1000),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume,
                    'vwap': agg.vwap,
                    'transactions': agg.transactions
                })
            
            if len(data) > 20:  # Ensure sufficient data
                df = pd.DataFrame(data)
                df.set_index('timestamp', inplace=True)
                return df
                
        except Exception as e:
            print(f"Error fetching {symbol} from Polygon: {e}")
        
        return None
        
    def fetch_options_chain(self, symbol: str, expiration_date: Optional[str] = None) -> Dict:
        """Fetch options chain data for enhanced gap analysis."""
        try:
            # Get next Friday if no expiration specified
            if expiration_date is None:
                today = datetime.now()
                days_until_friday = (4 - today.weekday()) % 7
                if days_until_friday == 0:
                    days_until_friday = 7
                expiration = today + timedelta(days=days_until_friday)
                expiration_date = expiration.strftime('%Y-%m-%d')
            
            # Fetch options contracts
            contracts = self.polygon_rest.list_options_contracts(
                underlying_ticker=symbol,
                expiration_date=expiration_date,
                limit=1000
            )
            
            options_data = {'calls': [], 'puts': []}
            
            for contract in contracts:
                contract_data = {
                    'strike': contract.strike_price,
                    'expiration': contract.expiration_date,
                    'contract_type': contract.contract_type,
                    'ticker': contract.ticker
                }
                
                # Get latest quote for this contract
                try:
                    quote = self.polygon_rest.get_last_quote(contract.ticker)
                    contract_data.update({
                        'bid': quote.bid,
                        'ask': quote.ask,
                        'volume': quote.day_volume,
                        'open_interest': quote.open_interest,
                        'implied_volatility': quote.implied_volatility
                    })
                except:
                    pass
                
                if contract.contract_type == 'call':
                    options_data['calls'].append(contract_data)
                else:
                    options_data['puts'].append(contract_data)
                    
            return options_data
            
        except Exception as e:
            print(f"Error fetching options chain for {symbol}: {e}")
            return {'calls': [], 'puts': []}
            
    def analyze_options_flow(self, symbol: str) -> Dict[str, float]:
        """Analyze options flow to predict gap continuation."""
        options_chain = self.fetch_options_chain(symbol)
        
        analysis = {
            'bullish_flow': 0,
            'bearish_flow': 0,
            'smart_money_confidence': 0,
            'expected_move': 0
        }
        
        # Analyze call options
        for call in options_chain['calls']:
            if 'volume' in call and 'open_interest' in call:
                # High volume relative to open interest indicates new positioning
                if call['open_interest'] > 0:
                    flow_ratio = call['volume'] / call['open_interest']
                    if flow_ratio > 0.5:  # Significant new activity
                        analysis['bullish_flow'] += call['volume'] * call.get('ask', 0)
        
        # Analyze put options
        for put in options_chain['puts']:
            if 'volume' in put and 'open_interest' in put:
                if put['open_interest'] > 0:
                    flow_ratio = put['volume'] / put['open_interest']
                    if flow_ratio > 0.5:
                        analysis['bearish_flow'] += put['volume'] * put.get('ask', 0)
        
        # Calculate smart money confidence
        total_flow = analysis['bullish_flow'] + analysis['bearish_flow']
        if total_flow > 0:
            analysis['smart_money_confidence'] = (analysis['bullish_flow'] - analysis['bearish_flow']) / total_flow
        
        # Estimate expected move from at-the-money options
        current_price = self.realtime_features[self.symbol_to_idx[symbol], 0].item()
        if current_price > 0:
            atm_calls = [c for c in options_chain['calls'] 
                        if abs(c['strike'] - current_price) / current_price < 0.02]
            atm_puts = [p for p in options_chain['puts'] 
                       if abs(p['strike'] - current_price) / current_price < 0.02]
            
            if atm_calls and atm_puts:
                atm_call_price = np.mean([c.get('ask', 0) for c in atm_calls])
                atm_put_price = np.mean([p.get('ask', 0) for p in atm_puts])
                analysis['expected_move'] = (atm_call_price + atm_put_price) / current_price * 100
        
        return analysis
        
    def get_pre_market_analysis(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Analyze pre-market data and return gap candidates with options insights."""
        print("Analyzing pre-market gaps with options flow...")
        
        # Get yesterday's closing prices from historical data
        yesterday_closes = torch.zeros(len(self.universe_symbols), device=self.device, dtype=self.dtype)
        
        for idx, symbol in enumerate(self.universe_symbols):
            if symbol in self.historical_cache:
                hist_data = self.historical_cache[symbol]
                if len(hist_data) > 0:
                    yesterday_closes[idx] = hist_data['close'].iloc[-1]
        
        # Calculate gaps using current pre-market prices
        current_prices = self.realtime_features[:, 0]  # last_price
        
        # Avoid division by zero
        valid_mask = yesterday_closes > 0
        gap_percents = torch.zeros_like(current_prices)
        gap_percents[valid_mask] = ((current_prices[valid_mask] - yesterday_closes[valid_mask]) / 
                                    yesterday_closes[valid_mask] * 100)
        
        # Identify significant gaps
        significant_gaps = torch.abs(gap_percents) >= self.config.MIN_GAP_PERCENT
        
        # Create enhanced gap analysis tensor including options features
        gap_analysis = torch.stack([
            gap_percents,                          # gap percentage
            self.realtime_features[:, 5],          # volume_5min
            self.realtime_features[:, 17],         # large_trade_ratio
            self.realtime_features[:, 4],          # spread_pct
            self.realtime_features[:, 19],         # microstructure_score
            self.realtime_features[:, 20],         # options_volume_ratio
            self.realtime_features[:, 21],         # put_call_ratio
            self.realtime_features[:, 22],         # options_flow_score
            self.realtime_features[:, 24],         # unusual_options_score
        ], dim=1)
        
        return gap_analysis, significant_gaps
        
    def _extract_underlying_from_options(self, options_symbol: str) -> str:
        """Extract underlying symbol from options contract symbol."""
        # Polygon options format: O:AAPL230120C00150000
        # Extract the underlying symbol (AAPL in this case)
        if options_symbol.startswith('O:'):
            parts = options_symbol[2:].split(chr(ord('0')))
            if parts:
                # Find where the numbers start
                for i, char in enumerate(parts[0]):
                    if char.isdigit():
                        return parts[0][:i]
        return ''
        
    def _is_high_volume_symbol(self, symbol: str) -> bool:
        """Determine if a symbol is high volume for options monitoring."""
        # Check if this symbol typically has high options volume
        # In production, this would use actual volume data
        high_volume_symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT', 'AMZN', 'META', 'GOOGL']
        return symbol in high_volume_symbols
        
    def calculate_intraday_features(self) -> torch.Tensor:
        """Calculate comprehensive intraday features using GPU."""
        # Stack all feature types
        features = []
        
        # Real-time features (including options)
        features.append(self.realtime_features)
        
        # Technical indicators (calculated on GPU)
        tech_features = self._calculate_technical_features_gpu()
        features.append(tech_features)
        
        # Market microstructure features (enhanced with tick data)
        micro_features = self._calculate_microstructure_features_gpu()
        features.append(micro_features)
        
        # Concatenate all features
        all_features = torch.cat(features, dim=1)
        
        return all_features
        
    def _calculate_technical_features_gpu(self) -> torch.Tensor:
        """Calculate technical indicators on GPU with tick precision."""
        n_symbols = len(self.universe_symbols)
        tech_features = torch.zeros((n_symbols, 10), device=self.device, dtype=self.dtype)
        
        for idx in range(n_symbols):
            count = self.intraday_counts[idx]
            if count < 20:
                continue
            
            prices = self.intraday_prices[idx, :count]
            volumes = self.intraday_volumes[idx, :count]
            
            # Use tick data for more accurate calculations
            tick_count = min(self.tick_counts[idx].item(), self.tick_buffer_size)
            if tick_count > 50:
                tick_prices = self.tick_prices[idx, :tick_count]
                
                # Calculate true range using tick data
                tick_high = tick_prices.max()
                tick_low = tick_prices.min()
                true_range = tick_high - tick_low
                tech_features[idx, 5] = true_range / prices[-1] * 100  # True range %
                
                # Tick-based momentum
                recent_ticks = tick_prices[-20:]
                older_ticks = tick_prices[-40:-20] if tick_count > 40 else tick_prices[:20]
                tick_momentum = (recent_ticks.mean() - older_ticks.mean()) / older_ticks.mean() * 100
                tech_features[idx, 6] = tick_momentum
            
            # Standard technical indicators
            if count >= 20:
                tech_features[idx, 0] = prices[-20:].mean()  # SMA20
            if count >= 50:
                tech_features[idx, 1] = prices[-50:].mean()  # SMA50
            
            # RSI (simplified)
            if count >= 14:
                changes = prices[1:] - prices[:-1]
                gains = torch.where(changes > 0, changes, 0)
                losses = torch.where(changes < 0, -changes, 0)
                avg_gain = gains[-14:].mean()
                avg_loss = losses[-14:].mean()
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    tech_features[idx, 2] = 100 - (100 / (1 + rs))  # RSI
            
            # Volume-weighted momentum
            if count >= 10:
                recent_vwap = (prices[-10:] * volumes[-10:]).sum() / volumes[-10:].sum()
                older_vwap = (prices[-20:-10] * volumes[-20:-10]).sum() / volumes[-20:-10].sum()
                tech_features[idx, 3] = (recent_vwap - older_vwap) / older_vwap * 100
            
            # Bollinger Band position
            if count >= 20:
                sma = prices[-20:].mean()
                std = prices[-20:].std()
                current = prices[-1]
                tech_features[idx, 4] = (current - sma) / (2 * std) if std > 0 else 0
        
        return tech_features
        
    def _calculate_microstructure_features_gpu(self) -> torch.Tensor:
        """Calculate market microstructure features using tick-level data."""
        n_symbols = len(self.universe_symbols)
        micro_features = torch.zeros((n_symbols, 8), device=self.device, dtype=self.dtype)
        
        # Extract microstructure patterns from real-time data
        spreads = self.realtime_features[:, 4]  # spread_pct
        large_trade_ratios = self.realtime_features[:, 17]
        trade_counts = self.realtime_features[:, 15]
        avg_trade_sizes = self.realtime_features[:, 16]
        
        # Kyle's lambda (price impact coefficient) - enhanced with tick data
        for idx in range(n_symbols):
            tick_count = min(self.tick_counts[idx].item(), self.tick_buffer_size)
            if tick_count > 10:
                tick_prices = self.tick_prices[idx, :tick_count]
                tick_sizes = self.tick_sizes[idx, :tick_count]
                
                # Calculate price impact per unit volume
                price_changes = torch.diff(tick_prices)
                volume_chunks = tick_sizes[1:]
                
                valid_mask = volume_chunks > 0
                if valid_mask.sum() > 5:
                    impacts = torch.abs(price_changes[valid_mask]) / torch.sqrt(volume_chunks[valid_mask])
                    micro_features[idx, 0] = impacts.median()  # Robust Kyle's lambda
        
        # Amihud illiquidity ratio
        volume_5min = self.realtime_features[:, 5]
        price_change_5min = self.realtime_features[:, 9]
        micro_features[:, 1] = torch.abs(price_change_5min) / (volume_5min + 1)
        
        # Order flow imbalance (enhanced with tick data)
        micro_features[:, 2] = large_trade_ratios * torch.sign(price_change_5min)
        
        # Relative spread (current vs average)
        avg_spread = spreads.mean()
        micro_features[:, 3] = spreads / (avg_spread + 0.001)
        
        # Trade intensity
        avg_trade_count = trade_counts.mean()
        micro_features[:, 4] = trade_counts / (avg_trade_count + 1)
        
        # Trade size distribution score
        micro_features[:, 5] = avg_trade_sizes / (avg_trade_sizes.median() + 1)
        
        # Options-driven microstructure
        options_volume_ratio = self.realtime_features[:, 20]
        options_flow_score = self.realtime_features[:, 22]
        micro_features[:, 6] = options_volume_ratio * (1 + options_flow_score)  # Options pressure
        
        # Tick fragmentation (how spread across time are trades)
        for idx in range(n_symbols):
            tick_count = self.tick_counts[idx].item()
            if tick_count > 10:
                tick_times = self.tick_timestamps[idx, :min(tick_count, 100)]
                time_diffs = torch.diff(tick_times[tick_times > 0])
                if len(time_diffs) > 5:
                    micro_features[idx, 7] = time_diffs.std().float() / (time_diffs.mean().float() + 1)
        
        return micro_features
        
    def stop_realtime_data(self):
        """Stop WebSocket connections."""
        self.ws_connected = False
        if self.polygon_ws:
            asyncio.create_task(self.polygon_ws.close())
            
    def get_current_features(self) -> Tuple[torch.Tensor, List[str]]:
        """Get current feature tensor and names."""
        all_features = self.calculate_intraday_features()
        
        # Combine feature names
        all_feature_names = self.feature_names + [
            'sma20', 'sma50', 'rsi', 'volume_momentum', 'bb_position',
            'true_range_pct', 'tick_momentum', 'tech_feature_8', 'tech_feature_9', 'tech_feature_10',
            'kyle_lambda', 'amihud_ratio', 'order_flow_imbalance', 'relative_spread',
            'trade_intensity', 'trade_size_score', 'options_pressure', 'tick_fragmentation'
        ]
        
        return all_features, all_feature_names


# Update the import in other files to use PolygonDataPipeline instead of ParallelDataPipeline
ParallelDataPipeline = PolygonDataPipeline  # Alias for compatibility


# Example usage when file is run directly
if __name__ == "__main__":
    print("GPU-Accelerated Polygon Data Pipeline Test")
    print("=" * 50)
    
    # Initialize configuration and universe manager
    config = GPUTradingConfig()
    universe_manager = TensorUniverseManager(config)
    
    # Ensure universe is loaded
    if len(universe_manager.universe_symbols) == 0:
        print("Loading universe...")
        universe_manager.update_universe()
    
    # Create data pipeline
    pipeline = PolygonDataPipeline(config, universe_manager)
    
    # Test historical data fetching
    print("\nFetching historical data from Polygon...")
    historical_data = pipeline.fetch_historical_data(lookback_days=30)
    
    print(f"\nHistorical Data Summary:")
    print(f"  Symbols with data: {len(historical_data)}")
    if historical_data:
        sample_symbol = list(historical_data.keys())[0]
        sample_data = historical_data[sample_symbol]
        print(f"  Sample ({sample_symbol}): {len(sample_data)} days")
        print(f"  Latest close: ${sample_data['close'].iloc[-1]:.2f}")
        print(f"  Average volume: {sample_data['volume'].mean():,.0f}")
        print(f"  VWAP available: {'vwap' in sample_data.columns}")
    
    # Test options data fetching
    if historical_data:
        test_symbol = list(historical_data.keys())[0]
        print(f"\nFetching options chain for {test_symbol}...")
        options_chain = pipeline.fetch_options_chain(test_symbol)
        
        print(f"  Calls: {len(options_chain['calls'])} contracts")
        print(f"  Puts: {len(options_chain['puts'])} contracts")
        
        # Analyze options flow
        options_analysis = pipeline.analyze_options_flow(test_symbol)
        print(f"\nOptions Flow Analysis for {test_symbol}:")
        print(f"  Bullish Flow: ${options_analysis['bullish_flow']:,.0f}")
        print(f"  Bearish Flow: ${options_analysis['bearish_flow']:,.0f}")
        print(f"  Smart Money Confidence: {options_analysis['smart_money_confidence']:+.2f}")
        print(f"  Expected Move: {options_analysis['expected_move']:.2f}%")
    
    # Test real-time features initialization
    print("\nReal-time Features Structure:")
    print(f"  Feature tensor shape: {pipeline.realtime_features.shape}")
    print(f"  Number of features: {len(pipeline.feature_names)} (includes options features)")
    print(f"  Device: {pipeline.realtime_features.device}")
    print(f"  Memory usage: {pipeline.realtime_features.element_size() * pipeline.realtime_features.nelement() / 1e6:.2f} MB")
    
    # Simulate some pre-market data for testing
    print("\nSimulating pre-market data with institutional activity...")
    for i in range(min(5, len(pipeline.universe_symbols))):
        symbol = pipeline.universe_symbols[i]
        base_price = 100.0 + i * 10
        
        # Simulate 100 pre-market trades with some block trades
        for j in range(100):
            size = 100 + torch.randint(0, 900, (1,)).item()
            # Make some trades institutional size
            if j % 20 == 0:
                size = 10000 + torch.randint(0, 5000, (1,)).item()
                
            pipeline.pre_market_prices[i, j] = base_price + torch.randn(1).item() * 0.5
            pipeline.pre_market_volumes[i, j] = size
            pipeline.pre_market_timestamps[i, j] = int(datetime.now().timestamp()) - (100 - j)
            pipeline.pre_market_trade_sizes[i, j] = size if size >= 10000 else 0
        pipeline.pre_market_counts[i] = 100
        
        # Update features for this symbol
        pipeline._update_realtime_features_gpu(i)
        
        # Simulate some options activity
        pipeline.options_flow[symbol] = {'calls': 5000, 'puts': 3000}
        pipeline._update_options_features(i, symbol)
    
    # Test enhanced gap analysis
    gap_analysis, significant_gaps = pipeline.get_pre_market_analysis()
    
    print(f"\nEnhanced Pre-market Gap Analysis:")
    print(f"  Gap analysis shape: {gap_analysis.shape}")
    print(f"  Significant gaps found: {significant_gaps.sum().item()}")
    print(f"  Features include options data: Yes")
    
    # Show top gaps with options insights
    gap_percents = gap_analysis[:, 0]
    options_scores = gap_analysis[:, 7]  # options_flow_score
    top_gap_indices = torch.argsort(torch.abs(gap_percents), descending=True)[:5]
    
    print(f"\nTop 5 Pre-market Gaps with Options Analysis:")
    for i, idx in enumerate(top_gap_indices):
        if torch.abs(gap_percents[idx]) > 0:
            symbol = pipeline.universe_symbols[idx]
            gap_pct = gap_percents[idx].item()
            volume = gap_analysis[idx, 1].item()
            options_flow = options_scores[idx].item()
            unusual_options = gap_analysis[idx, 8].item()
            
            print(f"  {i+1}. {symbol}: {gap_pct:+.2f}% gap")
            print(f"     Volume: {volume:.0f} shares")
            print(f"     Options Flow: {options_flow:+.2f} ({'Bullish' if options_flow > 0 else 'Bearish'})")
            print(f"     Unusual Options Activity: {'Yes' if unusual_options > 0.5 else 'No'}")
    
    # Test feature calculation
    all_features, feature_names = pipeline.get_current_features()
    print(f"\nComplete Feature Set with Polygon Data:")
    print(f"  Total features: {len(feature_names)}")
    print(f"  Feature tensor shape: {all_features.shape}")
    print(f"  Non-zero features: {(all_features != 0).sum().item()}")
    print(f"  Tick data buffer: {pipeline.tick_buffer_size} ticks per symbol")
    
    # Note about WebSocket
    print("\nNote: Polygon WebSocket connection not started in test mode.")
    print("In production, call pipeline.start_realtime_data() in async context.")
    print("\nPolygon Integration Benefits:")
    print("  - True tick-by-tick data from all exchanges")
    print("  - Options flow analysis for institutional positioning")
    print("  - Enhanced microstructure features from tick data")
    print("  - More accurate VWAP and technical indicators")