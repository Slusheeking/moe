"""
Real-Time Data Manager for Momentum Detection
Optimized for live momentum scanning with minimal latency
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
    """Real-time data stream for momentum detection"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.websocket = None
        self.subscriptions = set()
        self.price_data = {}  # symbol -> recent prices
        self.volume_data = {}  # symbol -> recent volumes
        self.callbacks = []
        self.running = False
        
        # Data buffers for momentum calculation
        self.price_buffers = defaultdict(lambda: deque(maxlen=100))  # Keep last 100 prices
        self.volume_buffers = defaultdict(lambda: deque(maxlen=100))
        self.timestamp_buffers = defaultdict(lambda: deque(maxlen=100))
        
    async def connect(self):
        """Connect to Polygon WebSocket"""
        try:
            ws_url = "wss://socket.polygon.io/stocks"
            self.websocket = await websockets.connect(ws_url)
            
            # Authenticate
            auth_message = {
                "action": "auth",
                "params": self.api_key
            }
            await self.websocket.send(json.dumps(auth_message))
            
            # Wait for auth confirmation
            response = await self.websocket.recv()
            auth_response = json.loads(response)
            
            # Handle connection status message first, then auth response
            authenticated = False
            for _ in range(3):  # Try up to 3 messages
                message = json.loads(response) if 'response' in locals() else auth_response
                
                # Handle connection status message
                if isinstance(message, dict) and message.get("status") == "connected":
                    logger.debug("WebSocket connected, waiting for auth response...")
                    response = await self.websocket.recv()
                    continue
                
                # Handle auth response (could be list or dict)
                if isinstance(message, list) and len(message) > 0:
                    auth_data = message[0]
                elif isinstance(message, dict):
                    auth_data = message
                else:
                    break
                
                if auth_data.get("status") == "auth_success":
                    logger.info("Connected to Polygon WebSocket")
                    self.running = True
                    authenticated = True
                    break
                elif auth_data.get("status") in ["auth_timeout", "auth_failed"]:
                    logger.error(f"WebSocket auth failed: {auth_data}")
                    return False
                else:
                    # Not an auth message, try next
                    response = await self.websocket.recv()
                    continue
            
            if authenticated:
                # Start message processing
                asyncio.create_task(self._process_messages())
                return True
            else:
                logger.error("Failed to authenticate WebSocket within timeout")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False
    
    async def subscribe(self, symbols: List[str]):
        """Subscribe to real-time data for symbols"""
        if not self.websocket or not symbols:
            return
        
        try:
            # Subscribe to trades for momentum detection
            trade_symbols = [f"T.{symbol}" for symbol in symbols[:1000]]  # Limit to 1000
            
            subscribe_message = {
                "action": "subscribe",
                "params": ",".join(trade_symbols)
            }
            
            await self.websocket.send(json.dumps(subscribe_message))
            self.subscriptions.update(symbols)
            
            logger.info(f"Subscribed to {len(symbols)} symbols for real-time momentum data")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to symbols: {e}")
    
    async def update_subscriptions(self, new_symbols: List[str]):
        """Update subscriptions to new symbol list"""
        if not self.websocket:
            return
        
        try:
            # Unsubscribe from old symbols
            if self.subscriptions:
                old_trade_symbols = [f"T.{symbol}" for symbol in self.subscriptions]
                unsubscribe_message = {
                    "action": "unsubscribe",
                    "params": ",".join(old_trade_symbols)
                }
                await self.websocket.send(json.dumps(unsubscribe_message))
            
            # Subscribe to new symbols
            await self.subscribe(new_symbols)
            
        except Exception as e:
            logger.error(f"Failed to update subscriptions: {e}")
    
    async def _process_messages(self):
        """Process incoming WebSocket messages"""
        while self.running and self.websocket:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                # Handle trade messages
                if isinstance(data, list):
                    for item in data:
                        if item.get("ev") == "T":  # Trade event
                            await self._process_trade(item)
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await asyncio.sleep(1)
    
    async def _process_trade(self, trade_data: Dict):
        """Process individual trade for momentum calculation"""
        try:
            symbol = trade_data.get("sym")
            price = trade_data.get("p")
            volume = trade_data.get("s", 1)  # Size
            timestamp = datetime.fromtimestamp(trade_data.get("t", 0) / 1000)
            
            if not symbol or not price:
                return
            
            # Update data buffers
            self.price_buffers[symbol].append(price)
            self.volume_buffers[symbol].append(volume)
            self.timestamp_buffers[symbol].append(timestamp)
            
            # Update current price
            self.price_data[symbol] = price
            
            # Calculate live momentum if we have enough data
            if len(self.price_buffers[symbol]) >= 5:
                momentum_data = self._calculate_live_momentum(symbol)
                
                # Notify callbacks if significant momentum detected
                if momentum_data and momentum_data.get('momentum_score', 0) > 30:
                    for callback in self.callbacks:
                        try:
                            await callback(symbol, momentum_data)
                        except Exception as e:
                            logger.error(f"Callback error for {symbol}: {e}")
            
        except Exception as e:
            logger.error(f"Error processing trade for {symbol}: {e}")
    
    def _calculate_live_momentum(self, symbol: str) -> Optional[Dict]:
        """Calculate live momentum from recent trades"""
        try:
            prices = list(self.price_buffers[symbol])
            volumes = list(self.volume_buffers[symbol])
            
            if len(prices) < 5:
                return None
            
            current_price = prices[-1]
            
            # Calculate price velocity (5-minute momentum)
            if len(prices) >= 10:
                price_velocity = (prices[-1] - prices[-10]) / prices[-10]
            else:
                price_velocity = (prices[-1] - prices[0]) / prices[0]
            
            # Calculate volume explosion
            current_volume = volumes[-1]
            avg_volume = np.mean(volumes[:-1]) if len(volumes) > 1 else current_volume
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Calculate price acceleration
            if len(prices) >= 3:
                recent_velocity = (prices[-1] - prices[-2]) / prices[-2]
                prev_velocity = (prices[-2] - prices[-3]) / prices[-3]
                acceleration = recent_velocity - prev_velocity
            else:
                acceleration = 0
            
            # Simple momentum score
            momentum_score = 0
            
            # Price movement scoring
            if abs(price_velocity) > 0.02:  # 2%+ move
                momentum_score += 30
            elif abs(price_velocity) > 0.01:  # 1%+ move
                momentum_score += 15
            
            # Volume scoring
            if volume_ratio > 3:
                momentum_score += 25
            elif volume_ratio > 2:
                momentum_score += 15
            
            # Acceleration scoring
            if acceleration > 0.005:  # Positive acceleration
                momentum_score += 20
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'price_velocity': price_velocity,
                'volume_ratio': volume_ratio,
                'acceleration': acceleration,
                'momentum_score': momentum_score,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Live momentum calculation failed for {symbol}: {e}")
            return None
    
    async def get_last_price(self, symbol: str) -> Optional[float]:
        """Get last known price for symbol"""
        return self.price_data.get(symbol)
    
    def get_price_history(self, symbol: str, periods: int = 20) -> List[float]:
        """Get recent price history for symbol"""
        if symbol in self.price_buffers:
            prices = list(self.price_buffers[symbol])
            return prices[-periods:] if len(prices) >= periods else prices
        return []
    
    def get_volume_history(self, symbol: str, periods: int = 20) -> List[float]:
        """Get recent volume history for symbol"""
        if symbol in self.volume_buffers:
            volumes = list(self.volume_buffers[symbol])
            return volumes[-periods:] if len(volumes) >= periods else volumes
        return []
    
    def add_momentum_callback(self, callback: Callable):
        """Add callback for momentum events"""
        self.callbacks.append(callback)
    
    def remove_momentum_callback(self, callback: Callable):
        """Remove momentum callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        self.running = False
        
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("WebSocket disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting WebSocket: {e}")


class DataAggregator:
    """Aggregate and process market data for momentum detection"""
    
    def __init__(self):
        self.symbol_data = defaultdict(dict)
        self.market_snapshots = {}
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
        
        # Sort by score descending
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