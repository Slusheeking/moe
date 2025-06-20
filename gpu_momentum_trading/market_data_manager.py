"""
High-Performance Market Data Manager
Handles Polygon WebSocket and API data with GPU acceleration
"""

import asyncio
import websockets
import json
import aiohttp
import logging
from typing import Dict, List, Optional, Set, Callable
from datetime import datetime
import time
import os
from system_config import MOMENTUM_CONFIG
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockLatestQuoteRequest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class MarketDataManager:
    """Ultra-fast market data processing"""
    
    def __init__(self, polygon_api_key: str, gpu_scanner):
        self.api_key = polygon_api_key
        self.gpu_scanner = gpu_scanner
        self.websocket = None
        self.session = None
        self.running = False
        self.callbacks = []
        
        # Initialize Alpaca data client for reliable quotes
        self.alpaca_client = StockHistoricalDataClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY')
        )
        
        # Performance tracking
        self.messages_processed = 0
        self.last_snapshot_time = 0
        
    async def connect(self):
        """Connect to data sources"""
        # HTTP session for API calls
        self.session = aiohttp.ClientSession()
        
        # WebSocket for real-time data with robust connection settings
        ws_url = "wss://socket.polygon.io/stocks"
        self.websocket = await websockets.connect(
            ws_url,
            ping_interval=60,      # Longer ping interval
            ping_timeout=30,       # Longer timeout
            max_size=2**23,
            close_timeout=10
        )
        
        # Authenticate
        auth = {"action": "auth", "params": self.api_key}
        await self.websocket.send(json.dumps(auth))
        
        # Wait for auth confirmation
        auth_response = await self.websocket.recv()
        logger.info(f"WebSocket auth response: {auth_response}")
        
        # Subscribe to trades only (quotes handled by Alpaca API)
        subscribe = {"action": "subscribe", "params": "T.*"}
        await self.websocket.send(json.dumps(subscribe))
        
        self.running = True
        logger.info("Market data connected and subscribed")
        
        # Start processing
        asyncio.create_task(self._process_stream())
        asyncio.create_task(self._snapshot_updater())
    
    async def _process_stream(self):
        """Process WebSocket stream with GPU batching"""
        batch = []
        batch_size = 1000
        last_batch_time = time.time()
        last_debug_time = time.time()
        
        while self.running and self.websocket:
            try:
                message = await asyncio.wait_for(self.websocket.recv(), timeout=1)
                data = json.loads(message)
                
                # Debug logging every 30 seconds
                current_time = time.time()
                if current_time - last_debug_time >= 30:
                    logger.info(f"WebSocket active - messages processed: {self.messages_processed}")
                    last_debug_time = current_time
                
                if isinstance(data, list):
                    for item in data:
                        if item.get('ev') == 'T':  # Trade
                            batch.append({
                                'symbol': item.get('sym'),
                                'price': item.get('p'),
                                'volume': item.get('s', 0) * 100,  # Convert to shares
                                'timestamp': item.get('t', 0) / 1000
                            })
                            self.messages_processed += 1
                        elif item.get('ev') == 'Q':  # Quote
                            # Skip quote processing - using Alpaca API for quotes
                            pass
                        elif item.get('ev') == 'status':
                            logger.info(f"WebSocket status: {item}")
                
                # Process batch when full or timeout
                current_time = time.time()
                if len(batch) >= batch_size or (current_time - last_batch_time > 0.1):
                    if batch:
                        # Update GPU scanner
                        self.gpu_scanner.batch_update(batch)
                        
                        # Calculate momentum every 1000 messages
                        if self.messages_processed % 1000 == 0:
                            await self._check_momentum_signals()
                        
                        batch = []
                        last_batch_time = current_time
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Stream processing error: {e}")
                # Handle connection drops by reconnecting
                if "keepalive ping timeout" in str(e) or "connection" in str(e).lower():
                    logger.info("Attempting WebSocket reconnection...")
                    try:
                        await self._reconnect_websocket()
                    except Exception as reconnect_error:
                        logger.error(f"Reconnection failed: {reconnect_error}")
                        await asyncio.sleep(30)
                else:
                    await asyncio.sleep(1)
    
    async def _snapshot_updater(self):
        """Periodic market snapshot updates"""
        while self.running:
            try:
                current_time = time.time()
                if current_time - self.last_snapshot_time >= 30:  # Every 30 seconds
                    await self._update_market_snapshot()
                    self.last_snapshot_time = current_time
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Snapshot update error: {e}")
                await asyncio.sleep(30)
    
    async def _update_market_snapshot(self):
        """Get full market snapshot"""
        try:
            url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers"
            params = {'apikey': self.api_key}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'tickers' in data:
                        batch = []
                        for ticker in data['tickers']:
                            symbol = ticker.get('ticker')
                            day = ticker.get('day', {})
                            
                            if symbol and day:
                                price = day.get('c', 0)  # Close price
                                volume = day.get('v', 0)  # Volume
                                
                                # Filter for stocks AND ETFs $5-$500 range with high volume
                                if (price >= 5.0 and price <= 500.0 and volume >= 500000 and
                                    len(symbol) <= 5 and symbol.isalpha() and
                                    not any(char in symbol for char in ['.', '-', '_'])):
                                    batch.append({
                                        'symbol': symbol,
                                        'price': price,
                                        'volume': volume,
                                        'timestamp': time.time()
                                    })
                        
                        # Update GPU scanner
                        self.gpu_scanner.batch_update(batch)
                        logger.info(f"Market snapshot: {len(batch)} active symbols")
                        
                        # Check for momentum
                        await self._check_momentum_signals()
                        
        except Exception as e:
            logger.error(f"Snapshot failed: {e}")
    
    async def _check_momentum_signals(self):
        """Check for momentum signals using GPU"""
        try:
            # Get top movers from GPU
            top_movers = self.gpu_scanner.get_top_movers(50)
            
            if top_movers:
                # Log top signals for debugging
                top_3 = top_movers[:3]
                logger.info(f"Top momentum signals: {[(s, f'{sc:.1f}') for s, sc in top_3]}")
            
            signal_count = 0
            for symbol, score in top_movers:
                if score >= MOMENTUM_CONFIG['min_score']:  # Minimum momentum threshold from config
                    # Get detailed data - will use Alpaca API for quotes
                    data = self.gpu_scanner.get_symbol_data(symbol)
                    
                    if data:
                        signal_count += 1
                        logger.info(f"🔥 Momentum signal: {symbol} Score: {score:.1f} Price: ${data.get('price', 0):.2f}")
                        # Trigger callbacks
                        for callback in self.callbacks:
                            asyncio.create_task(callback(symbol, data))
            
            if signal_count == 0 and top_movers:
                logger.info(f"No signals above threshold {MOMENTUM_CONFIG['min_score']}. Highest score: {top_movers[0][1]:.1f}")
            
        except Exception as e:
            logger.error(f"Momentum check error: {e}")
    
    def add_momentum_callback(self, callback: Callable):
        """Add callback for momentum signals"""
        self.callbacks.append(callback)
    
    async def _reconnect_websocket(self):
        """Reconnect WebSocket connection"""
        try:
            # Close existing connection
            if self.websocket:
                await self.websocket.close()
            
            # Reconnect with same settings
            ws_url = "wss://socket.polygon.io/stocks"
            self.websocket = await websockets.connect(
                ws_url,
                ping_interval=60,      # Longer ping interval
                ping_timeout=30,       # Longer timeout
                max_size=2**23,
                close_timeout=10
            )
            
            # Re-authenticate
            auth = {"action": "auth", "params": self.api_key}
            await self.websocket.send(json.dumps(auth))
            
            # Wait for auth confirmation
            auth_response = await self.websocket.recv()
            logger.info(f"WebSocket reconnected - auth response: {auth_response}")
            
            # Re-subscribe to trades
            subscribe = {"action": "subscribe", "params": "T.*"}
            await self.websocket.send(json.dumps(subscribe))
            
            logger.info("WebSocket reconnection successful")
            
        except Exception as e:
            logger.error(f"WebSocket reconnection failed: {e}")
            raise
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote with validation and spread checking"""
        try:
            # Use Alpaca's direct API for guaranteed fresh quotes
            request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            quotes = self.alpaca_client.get_stock_latest_quote(request)
            
            if symbol in quotes:
                quote = quotes[symbol]
                bid = quote.bid_price
                ask = quote.ask_price
                
                # Enhanced quote validation
                if bid > 0 and ask > 0 and ask > bid:
                    # Check spread is reasonable (< 5%)
                    spread_pct = (ask - bid) / ask * 100
                    if spread_pct < 5.0:
                        logger.info(f"⚡ ALPACA QUOTE: {symbol} Bid ${bid:.2f}, Ask ${ask:.2f} (Spread: {spread_pct:.2f}%)")
                        return {
                            'symbol': symbol,
                            'bid': bid,
                            'ask': ask,
                            'bid_size': quote.bid_size,
                            'ask_size': quote.ask_size,
                            'timestamp': quote.timestamp.timestamp()
                        }
                    else:
                        logger.warning(f"❌ WIDE SPREAD: {symbol} Bid ${bid:.2f}, Ask ${ask:.2f} (Spread: {spread_pct:.2f}%)")
                else:
                    logger.warning(f"❌ INVALID QUOTE: {symbol} Bid ${bid:.2f}, Ask ${ask:.2f}")
            else:
                logger.warning(f"❌ NO QUOTE RETURNED: {symbol}")
                
        except Exception as e:
            logger.error(f"❌ ALPACA QUOTE ERROR for {symbol}: {e}")
        
        return None
    
    async def disconnect(self):
        """Disconnect from data sources"""
        self.running = False
        
        if self.websocket:
            await self.websocket.close()
        
        if self.session:
            await self.session.close()
        
        logger.info("Market data disconnected")