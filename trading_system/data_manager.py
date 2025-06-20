"""
Real-Time Data Manager for Event-Driven Momentum Detection
Streamlined pipeline: WebSocket → Filter → XGBoost → Execute
"""

import asyncio
import websockets
import json
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np

from trading_system import config

logger = logging.getLogger(__name__)


class PolygonStream:
    """Real-time data stream for event-driven momentum detection"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.websocket = None
        self.subscriptions = set()
        self.price_data = {}  # symbol -> recent prices
        self.volume_data = {}  # symbol -> recent volumes
        self.callbacks = []
        self.running = False
        
        # Enhanced data buffers for immediate analysis
        self.price_buffers = defaultdict(lambda: deque(maxlen=200))  # Keep more history
        self.volume_buffers = defaultdict(lambda: deque(maxlen=200))
        self.timestamp_buffers = defaultdict(lambda: deque(maxlen=200))
        
        # Real-time quote data buffers for precise execution
        self.quote_data = {}  # symbol -> latest bid/ask data
        self.bid_buffers = defaultdict(lambda: deque(maxlen=50))  # Recent bid prices
        self.ask_buffers = defaultdict(lambda: deque(maxlen=50))  # Recent ask prices
        self.quote_timestamps = defaultdict(lambda: deque(maxlen=50))
        
        # Performance tracking
        self.trades_processed = 0
        self.signals_generated = 0
        self.last_signal_time = None
        
    async def connect(self):
        """Connect to Polygon WebSocket for ALL market data"""
        try:
            ws_url = "wss://socket.polygon.io/stocks"
            
            self.websocket = await websockets.connect(
                ws_url,
                ping_interval=45,  # Increased from 30 to 45 seconds
                ping_timeout=30,   # Increased from 15 to 30 seconds - more tolerant
                close_timeout=30,  # Increased close timeout
                max_size=2**23,    # 8MB max message size
                read_limit=2**20,  # 1MB read buffer
                write_limit=2**20  # 1MB write buffer
            )
            
            # Authenticate
            auth_message = {
                "action": "auth",
                "params": self.api_key
            }
            await self.websocket.send(json.dumps(auth_message))
            
            # Wait for auth confirmation
            try:
                response = await asyncio.wait_for(self.websocket.recv(), timeout=10)
                auth_response = json.loads(response)
            except asyncio.TimeoutError:
                logger.error("WebSocket auth timeout")
                return False
            
            # Handle auth response
            authenticated = False
            for _ in range(3):
                message = json.loads(response) if 'response' in locals() else auth_response
                
                if isinstance(message, dict) and message.get("status") == "connected":
                    try:
                        response = await asyncio.wait_for(self.websocket.recv(), timeout=10)
                        continue
                    except asyncio.TimeoutError:
                        logger.error("Timeout waiting for auth response")
                        return False
                
                if isinstance(message, list) and len(message) > 0:
                    auth_data = message[0]
                elif isinstance(message, dict):
                    auth_data = message
                else:
                    break
                
                if auth_data.get("status") == "auth_success":
                    logger.info("✅ Connected to Polygon WebSocket - Event-Driven Mode")
                    self.running = True
                    authenticated = True
                    break
                elif auth_data.get("status") in ["auth_timeout", "auth_failed"]:
                    logger.error(f"WebSocket auth failed: {auth_data}")
                    return False
                else:
                    try:
                        response = await asyncio.wait_for(self.websocket.recv(), timeout=10)
                        continue
                    except asyncio.TimeoutError:
                        logger.error("Timeout waiting for next auth message")
                        return False
            
            if authenticated:
                # Start event processing
                asyncio.create_task(self._process_messages())
                asyncio.create_task(self._monitor_connection())
                return True
            else:
                logger.error("Failed to authenticate WebSocket")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False
    
    async def subscribe(self, symbols: List[str] = None):
        """Subscribe to ALL market trades for maximum coverage"""
        if not self.websocket:
            return
        
        try:
            # Subscribe to ALL trades AND quotes using wildcards
            subscribe_message = {
                "action": "subscribe",
                "params": "T.*,Q.*"  # ALL trade symbols AND all quotes (NBBO)
            }
            
            await self.websocket.send(json.dumps(subscribe_message))
            logger.info("🚀 Subscribed to ALL market trades + quotes (NBBO) - Real-time execution ready")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to market trades: {e}")
    
    async def _monitor_connection(self):
        """Monitor websocket connection health"""
        reconnect_attempts = 0
        max_reconnect_attempts = 10
        base_reconnect_delay = 5
        
        while self.running:
            try:
                if not self.websocket or self.websocket.closed:
                    reconnect_attempts += 1
                    logger.warning(f"🔄 WebSocket reconnecting (attempt {reconnect_attempts})...")
                    
                    # Exponential backoff
                    delay = min(base_reconnect_delay * (2 ** min(reconnect_attempts - 1, 4)), 60)
                    await asyncio.sleep(delay)
                    
                    # Attempt reconnection
                    if await self.connect():
                        reconnect_attempts = 0
                        await self.subscribe()  # Re-subscribe to all trades
                        logger.info("✅ WebSocket reconnected successfully")
                    else:
                        if reconnect_attempts >= max_reconnect_attempts:
                            logger.error("Max reconnection attempts reached")
                            await asyncio.sleep(300)  # Wait 5 minutes
                            reconnect_attempts = 0
                else:
                    if reconnect_attempts > 0:
                        reconnect_attempts = 0
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Connection monitor error: {e}")
                await asyncio.sleep(10)
    
    async def _process_messages(self):
        """Process incoming WebSocket messages with immediate analysis"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.running and self.websocket:
            try:
                # Receive message with timeout - reduced to avoid ping conflicts
                message = await asyncio.wait_for(self.websocket.recv(), timeout=40)
                data = json.loads(message)
                
                consecutive_errors = 0
                
                # Process trade and quote messages immediately
                if isinstance(data, list):
                    for item in data:
                        if item.get("ev") == "T":  # Trade event
                            await self._process_trade_immediate(item)
                        elif item.get("ev") == "Q":  # Quote event (NBBO)
                            await self._process_quote_immediate(item)
                elif isinstance(data, dict):
                    if data.get("ev") == "T":
                        await self._process_trade_immediate(data)
                    elif data.get("ev") == "Q":
                        await self._process_quote_immediate(data)
                
            except asyncio.TimeoutError:
                continue  # Timeout is normal
                
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                # Check if it's a ping timeout (1011)
                if e.code == 1011:
                    logger.warning("🔄 Ping timeout detected - will attempt reconnection")
                break
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                consecutive_errors += 1
                
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                consecutive_errors += 1
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("Too many consecutive errors, stopping")
                    break
                
                await asyncio.sleep(1)
        
        self.running = False
        logger.info("WebSocket message processing stopped")
    
    async def _process_trade_immediate(self, trade_data: Dict):
        """Immediate trade processing with instant filtering and momentum analysis"""
        try:
            symbol = trade_data.get("sym")
            price = trade_data.get("p")
            volume = trade_data.get("s", 1)
            timestamp = datetime.fromtimestamp(trade_data.get("t", 0) / 1000)
            
            if not symbol or not price:
                return
            
            self.trades_processed += 1
            
            # IMMEDIATE FILTER: Fast volume/price screening
            if not self._passes_immediate_filter(symbol, price, volume):
                return
            
            # Update data buffers
            self.price_buffers[symbol].append(price)
            self.volume_buffers[symbol].append(volume)
            self.timestamp_buffers[symbol].append(timestamp)
            self.price_data[symbol] = price
            
            # IMMEDIATE MOMENTUM ANALYSIS: Calculate on every qualifying trade
            if len(self.price_buffers[symbol]) >= 5:  # Reduced minimum data requirement
                momentum_data = self._calculate_enhanced_momentum(symbol)
                
                if momentum_data:
                    live_score = momentum_data.get('momentum_score', 0)
                    
                    # LOG SIGNIFICANT MOMENTUM CALCULATIONS (score > 5 to reduce noise)
                    if live_score > 5:
                        logger.info(f"📊 LIVE MOMENTUM: {symbol}")
                        logger.info(f"   🎯 Score: {live_score:.0f} | Price: ${price:.2f} | Volume: {volume:,}")
                        logger.info(f"   📈 Velocity: {momentum_data.get('short_velocity', 0)*100:.2f}% | Volume Ratio: {momentum_data.get('volume_ratio', 1):.1f}x")
                        logger.info(f"   🚀 Acceleration: {momentum_data.get('acceleration', 0)*100:.2f}%")
                    
                    # INSTANT SIGNAL GENERATION: More permissive threshold
                    if live_score >= 15:  # Even lower threshold for more signals
                        self.signals_generated += 1
                        self.last_signal_time = datetime.now()
                        
                        logger.info(f"⚡ TRIGGERING XGBOOST: {symbol} | Live Score: {live_score:.0f} → Sending to XGBoost analysis...")
                        
                        # Execute callbacks immediately - no delays
                        for callback in self.callbacks:
                            try:
                                asyncio.create_task(callback(symbol, momentum_data))
                            except Exception as e:
                                logger.error(f"Callback error for {symbol}: {e}")
                    elif live_score > 5:  # Log near-misses
                        logger.info(f"❌ {symbol}: Score {live_score:.0f} below XGBoost threshold (15)")
                else:
                    # Debug: Log when momentum calculation fails
                    logger.debug(f"⚠️ {symbol}: Momentum calculation failed")
            else:
                # Track data accumulation
                if len(self.price_buffers[symbol]) == 4:  # Log when almost ready
                    logger.debug(f"📈 {symbol}: Building data (4/5 required)")
            
        except Exception as e:
            logger.debug(f"Error processing trade for {symbol}: {e}")
    
    async def _process_quote_immediate(self, quote_data: Dict):
        """Process real-time NBBO quote data for precise execution"""
        try:
            symbol = quote_data.get("sym")
            bid_price = quote_data.get("bp")  # Bid price
            ask_price = quote_data.get("ap")  # Ask price
            bid_size = quote_data.get("bs", 0)  # Bid size (in round lots)
            ask_size = quote_data.get("as", 0)  # Ask size (in round lots)
            timestamp = datetime.fromtimestamp(quote_data.get("t", 0) / 1000)
            
            if not symbol or not bid_price or not ask_price:
                return
            
            # Validate bid/ask data
            if bid_price <= 0 or ask_price <= 0 or bid_price >= ask_price:
                return
            
            # IMMEDIATE FILTER: Only store quotes for symbols we care about
            if not self._is_valid_symbol_fast(symbol):
                return
            
            # Update real-time quote data
            self.quote_data[symbol] = {
                'symbol': symbol,
                'bid_price': bid_price,
                'ask_price': ask_price,
                'bid_size': bid_size * 100,  # Convert to shares (round lots to shares)
                'ask_size': ask_size * 100,  # Convert to shares
                'spread': ask_price - bid_price,
                'spread_pct': (ask_price - bid_price) / bid_price if bid_price > 0 else 0,
                'timestamp': timestamp,
                'last_updated': datetime.now()
            }
            
            # Update quote buffers for trend analysis
            self.bid_buffers[symbol].append(bid_price)
            self.ask_buffers[symbol].append(ask_price)
            self.quote_timestamps[symbol].append(timestamp)
            
            # Log significant spread changes (for monitoring)
            spread_pct = (ask_price - bid_price) / bid_price
            if spread_pct > 0.02:  # Log spreads wider than 2%
                logger.debug(f"📊 Wide spread: {symbol} bid=${bid_price:.2f} ask=${ask_price:.2f} ({spread_pct:.1%})")
            
        except Exception as e:
            logger.debug(f"Error processing quote for {symbol}: {e}")
    
    def _passes_immediate_filter(self, symbol: str, price: float, volume: int) -> bool:
        """Lightning-fast filtering to reduce noise"""
        # Price range filter
        if not (config.UNIVERSE_CONFIG['min_price'] <= price <= config.UNIVERSE_CONFIG['max_price']):
            return False
        
        # Minimum volume filter (more permissive)
        if volume < 10:  # Only skip very tiny trades
            return False
        
        # Basic symbol validation
        if not self._is_valid_symbol_fast(symbol):
            return False
        
        # More permissive movement check - allow accumulation of smaller moves
        if symbol in self.price_buffers and len(self.price_buffers[symbol]) > 5:
            last_price = self.price_buffers[symbol][-1]
            price_change = abs(price - last_price) / last_price if last_price > 0 else 0
            # Only filter out if no movement at all
            if price_change < 0.001:  # 0.1% minimum (much more permissive)
                return False
        
        return True
    
    def _is_valid_symbol_fast(self, symbol: str) -> bool:
        """Fast symbol validation"""
        if not symbol or not (1 <= len(symbol) <= 5):
            return False
        if not (symbol.isalpha() and symbol.isupper()):
            return False
        # Skip obvious derivatives
        if symbol.endswith(('W', 'WS', 'WT', 'WR', 'P', 'PR')):
            return False
        return True
    
    def _calculate_enhanced_momentum(self, symbol: str) -> Optional[Dict]:
        """Enhanced momentum calculation for immediate decision making"""
        try:
            prices = list(self.price_buffers[symbol])
            volumes = list(self.volume_buffers[symbol])
            
            if len(prices) < 10:
                return None
            
            current_price = prices[-1]
            momentum_score = 0
            
            # Price velocity analysis
            short_velocity = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
            long_velocity = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
            
            # Volume analysis
            current_volume = volumes[-1]
            avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Acceleration
            if len(prices) >= 3:
                recent_velocity = (prices[-1] - prices[-2]) / prices[-2]
                prev_velocity = (prices[-2] - prices[-3]) / prices[-3]
                acceleration = recent_velocity - prev_velocity
            else:
                acceleration = 0
            
            # SCORING: Immediate momentum scoring
            if abs(short_velocity) > 0.02:  # 2%+ short-term move
                momentum_score += 40
            elif abs(short_velocity) > 0.01:  # 1%+ move
                momentum_score += 20
            
            if volume_ratio > 3:  # 3x+ volume
                momentum_score += 35
            elif volume_ratio > 2:  # 2x+ volume
                momentum_score += 20
            
            if acceleration > 0.01:  # Positive acceleration
                momentum_score += 25
            
            # Consistency check
            if len(prices) >= 5:
                recent_moves = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(-4, 0)]
                positive_moves = sum(1 for move in recent_moves if move > 0)
                if positive_moves >= 3:  # 3/4 positive moves
                    momentum_score += 15
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'short_velocity': short_velocity,
                'long_velocity': long_velocity,
                'volume_ratio': volume_ratio,
                'acceleration': acceleration,
                'momentum_score': momentum_score,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Enhanced momentum calculation failed for {symbol}: {e}")
            return None
    
    def add_momentum_callback(self, callback: Callable):
        """Add callback for momentum events"""
        self.callbacks.append(callback)
    
    def remove_momentum_callback(self, callback: Callable):
        """Remove momentum callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    async def get_last_price(self, symbol: str) -> Optional[float]:
        """Get last known price for symbol"""
        return self.price_data.get(symbol)
    
    def get_latest_quote(self, symbol: str) -> Optional[Dict]:
        """Get latest real-time bid/ask quote for symbol"""
        quote = self.quote_data.get(symbol)
        if quote:
            # Check if quote is recent (within last 5 seconds)
            if (datetime.now() - quote['last_updated']).total_seconds() <= 5:
                return quote
        return None
    
    def get_quote_stats(self) -> Dict:
        """Get quote processing statistics"""
        return {
            'active_quotes': len(self.quote_data),
            'total_symbols_with_quotes': len([s for s in self.quote_data.keys() if self.quote_data[s]]),
            'recent_quotes': len([q for q in self.quote_data.values()
                                if (datetime.now() - q['last_updated']).total_seconds() <= 10])
        }
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            'trades_processed': self.trades_processed,
            'signals_generated': self.signals_generated,
            'last_signal_time': self.last_signal_time,
            'active_symbols': len(self.price_buffers),
            'callbacks_registered': len(self.callbacks)
        }
    
    async def disconnect(self):
        """Gracefully disconnect from WebSocket"""
        self.running = False
        
        if self.websocket:
            try:
                if not self.websocket.closed:
                    unsubscribe_message = {
                        "action": "unsubscribe",
                        "params": "T.*,Q.*"
                    }
                    try:
                        await asyncio.wait_for(
                            self.websocket.send(json.dumps(unsubscribe_message)),
                            timeout=1
                        )
                    except asyncio.TimeoutError:
                        pass
                
                try:
                    await asyncio.wait_for(self.websocket.close(), timeout=2)
                except asyncio.TimeoutError:
                    pass
                
                logger.info("WebSocket disconnected")
                
            except Exception as e:
                logger.debug(f"WebSocket disconnect: {e}")
            finally:
                self.websocket = None
                self.subscriptions.clear()


class DataAggregator:
    """Real-time data aggregation for enhanced analysis"""
    
    def __init__(self):
        self.symbol_data = defaultdict(dict)
        self.last_update = {}
    
    def update_symbol_data(self, symbol: str, data: Dict):
        """Update data for a symbol"""
        self.symbol_data[symbol].update(data)
        self.last_update[symbol] = datetime.now()
    
    def get_symbol_data(self, symbol: str) -> Dict:
        """Get current data for symbol"""
        return self.symbol_data.get(symbol, {})
    
    def get_top_movers(self, metric: str = 'momentum_score', limit: int = 20) -> List[Dict]:
        """Get top symbols by specified metric"""
        scored_symbols = []
        
        for symbol, data in self.symbol_data.items():
            if metric in data and isinstance(data[metric], (int, float)):
                scored_symbols.append({
                    'symbol': symbol,
                    'score': data[metric],
                    'data': data
                })
        
        scored_symbols.sort(key=lambda x: x['score'], reverse=True)
        return scored_symbols[:limit]
    
    def cleanup_old_data(self, max_age_minutes: int = 30):
        """Clean up old data entries"""
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
        
        symbols_to_remove = []
        for symbol, last_update in self.last_update.items():
            if last_update < cutoff_time:
                symbols_to_remove.append(symbol)
        
        for symbol in symbols_to_remove:
            del self.symbol_data[symbol]
            del self.last_update[symbol]
        
        if symbols_to_remove:
            logger.debug(f"Cleaned up {len(symbols_to_remove)} old data entries")