"""
Polygon API Client for Momentum Detection
Optimized for real-time momentum scanning and minimal data requirements
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class PolygonClient:
    """Async Polygon API client optimized for momentum detection"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.session = None
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.last_request_time = 0
        
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self):
        """Initialize HTTP session with extended timeout"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=60)  # Increased from 30 to 60 seconds
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.info("Polygon API client connected")
    
    async def disconnect(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Polygon API client disconnected")
    
    async def _make_request(self, endpoint: str, params: Dict = None, retries: int = 2) -> Optional[Dict]:
        """Make rate-limited API request with enhanced error handling"""
        if not self.session:
            await self.connect()
        
        # More conservative rate limiting to prevent API issues
        time_since_last = time.time() - self.last_request_time
        min_delay = 0.2  # Increased from 0.1 to 0.2 seconds
        if time_since_last < min_delay:
            await asyncio.sleep(min_delay - time_since_last)
        
        for attempt in range(retries + 1):
            try:
                url = f"{self.base_url}{endpoint}"
                params = params or {}
                params['apikey'] = self.api_key
                
                self.last_request_time = time.time()
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    elif response.status == 429:  # Rate limited
                        wait_time = min(2 ** attempt, 10)  # Exponential backoff up to 10s
                        logger.warning(f"Rate limited, waiting {wait_time} seconds... (attempt {attempt + 1})")
                        await asyncio.sleep(wait_time)
                        continue  # Retry
                    elif response.status == 404:  # Not found - common for delisted/invalid symbols
                        logger.debug(f"Symbol not found (404) for {endpoint}")
                        return None
                    elif response.status in [500, 502, 503, 504]:  # Server errors - retry
                        if attempt < retries:
                            wait_time = min(2 ** attempt, 5)
                            logger.warning(f"Server error {response.status}, retrying in {wait_time}s... (attempt {attempt + 1})")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"Server error {response.status} after {retries} retries: {endpoint}")
                            return None
                    else:
                        error_text = await response.text()
                        if attempt < retries:
                            logger.warning(f"API error {response.status}, retrying... (attempt {attempt + 1})")
                            await asyncio.sleep(1)
                            continue
                        else:
                            logger.error(f"API error {response.status} after retries: {error_text[:200]}")
                            return None
                        
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.debug(f"Request timeout, retrying... (attempt {attempt + 1})")
                    await asyncio.sleep(1)
                    continue
                else:
                    logger.warning(f"Request timeout after {retries} retries: {endpoint}")
                    return None
            except Exception as e:
                if attempt < retries:
                    logger.warning(f"Request failed, retrying... (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(1)
                    continue
                else:
                    logger.error(f"Request failed after {retries} retries for {endpoint}: {e}")
                    return None
        
        return None
    
    async def get_market_movers(self, direction: str = 'gainers') -> List[str]:
        """Get market gainers or losers for momentum universe"""
        try:
            endpoint = f"/v2/snapshot/locale/us/markets/stocks/{direction}"
            data = await self._make_request(endpoint)
            
            if data and 'tickers' in data:
                symbols = []
                for item in data['tickers']:
                    symbol = item.get('ticker', '')
                    if symbol and len(symbol) <= 5:  # Filter out complex symbols
                        symbols.append(symbol)
                return symbols[:100]  # Top 100
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get market movers: {e}")
            return []
    
    async def get_snapshot_all_tickers(self) -> List[Any]:
        """Get snapshot of all tickers for momentum scanning"""
        try:
            endpoint = "/v2/snapshot/locale/us/markets/stocks/tickers"
            data = await self._make_request(endpoint)
            
            if data and 'tickers' in data:
                # Convert to simple objects for processing
                snapshots = []
                for item in data['tickers']:
                    snapshot = SimpleSnapshot(item)
                    snapshots.append(snapshot)
                return snapshots
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get all tickers snapshot: {e}")
            return []
    
    async def get_snapshot_ticker(self, symbol: str) -> Optional[Any]:
        """Get snapshot for specific ticker"""
        try:
            endpoint = f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
            data = await self._make_request(endpoint)
            
            if data and 'ticker' in data:
                return SimpleSnapshot(data['ticker'])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get snapshot for {symbol}: {e}")
            return None
    
    async def get_recent_ipos(self, days: int = 30) -> List[str]:
        """Get recent IPOs for momentum candidates"""
        try:
            endpoint = "/vX/reference/ipos"
            params = {
                'ipo_status': 'new',  # Recently listed
                'order': 'desc',
                'limit': 100,
                'sort': 'listing_date'
            }
            
            data = await self._make_request(endpoint, params)
            
            if data and 'results' in data:
                symbols = []
                for item in data['results']:
                    symbol = item.get('ticker', '')
                    if symbol and len(symbol) <= 5:  # Filter out complex symbols
                        symbols.append(symbol)
                return symbols[:50]  # Top 50 recent IPOs
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get recent IPOs: {e}")
            return []
    
    async def list_aggs(self, symbol: str, multiplier: int, timespan: str, 
                       from_date: datetime, to_date: datetime, limit: int = 120) -> List[Any]:
        """Get aggregated bars for momentum analysis"""
        try:
            from_str = from_date.strftime('%Y-%m-%d')
            to_str = to_date.strftime('%Y-%m-%d')
            
            endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_str}/{to_str}"
            params = {'limit': limit, 'sort': 'asc'}
            
            data = await self._make_request(endpoint, params)
            
            if data and 'results' in data:
                bars = []
                for item in data['results']:
                    bar = SimpleBar(item)
                    bars.append(bar)
                return bars
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get aggregated data for {symbol}: {e}")
            return []
    
    async def get_trades(self, symbol: str, timestamp: datetime = None, limit: int = 50) -> List[Any]:
        """Get recent trades for a symbol"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            timestamp_str = timestamp.strftime('%Y-%m-%d')
            endpoint = f"/v3/trades/{symbol}"
            params = {
                'timestamp': timestamp_str,
                'limit': limit,
                'sort': 'timestamp'
            }
            
            data = await self._make_request(endpoint, params)
            
            if data and 'results' in data:
                trades = []
                for item in data['results']:
                    trade = SimpleTrade(item)
                    trades.append(trade)
                return trades
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get trades for {symbol}: {e}")
            return []
    
    async def get_gainers_losers(self, direction: str) -> List[Dict]:
        """Get top gainers or losers with details"""
        try:
            endpoint = f"/v2/snapshot/locale/us/markets/stocks/{direction}"
            data = await self._make_request(endpoint)
            
            if data and 'results' in data:
                return data['results'][:50]  # Top 50
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get {direction}: {e}")
            return []
    
    async def get_previous_close(self, symbol: str) -> Optional[float]:
        """Get previous close price for gap analysis"""
        try:
            endpoint = f"/v2/aggs/ticker/{symbol}/prev"
            data = await self._make_request(endpoint)
            
            if data and 'results' in data and len(data['results']) > 0:
                return data['results'][0].get('c')  # Close price
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get previous close for {symbol}: {e}")
            return None
    
    async def get_full_market_snapshot(self) -> List[Dict]:
        """Get full market snapshot covering 10,000+ tickers in single request"""
        try:
            endpoint = "/v2/snapshot/locale/us/markets/stocks/tickers"
            # No tickers parameter = get all tickers
            data = await self._make_request(endpoint)
            
            if data and 'tickers' in data:
                logger.info(f"Retrieved {len(data['tickers'])} tickers from full market snapshot")
                return data['tickers']
            
            logger.warning("No ticker data in full market snapshot response")
            return []
            
        except Exception as e:
            logger.error(f"Failed to get full market snapshot: {e}")
            return []
    
    async def scan_full_market_momentum(self) -> List[Dict]:
        """Scan entire market for momentum signals using full snapshot"""
        try:
            logger.info("Getting full market snapshot (10,000+ tickers)...")
            
            # Get complete market snapshot in one request
            market_data = await self.get_full_market_snapshot()
            
            if not market_data:
                logger.warning("No market data available for momentum scan")
                return []
            
            logger.info(f"Analyzing {len(market_data)} tickers for momentum signals...")
            
            momentum_candidates = []
            
            # Analyze each ticker for momentum
            for ticker_data in market_data:
                momentum_signal = self._analyze_momentum_signal(ticker_data)
                if momentum_signal:
                    momentum_candidates.append(momentum_signal)
            
            # Sort by momentum strength
            momentum_candidates.sort(key=lambda x: x.get('momentum_score', 0), reverse=True)
            
            logger.info(f"Full market momentum scan complete: {len(momentum_candidates)} candidates found from {len(market_data)} tickers")
            
            return momentum_candidates[:500]  # Return top 500 momentum candidates
            
        except Exception as e:
            logger.error(f"Full market momentum scan failed: {e}")
            return []
    
    def _analyze_momentum_signal(self, ticker_data: Dict) -> Optional[Dict]:
        """Analyze individual ticker for momentum signals"""
        try:
            symbol = ticker_data.get('ticker', '')
            day_data = ticker_data.get('day', {})
            prev_data = ticker_data.get('prevDay', {})
            
            if not symbol or not day_data or not prev_data:
                return None
            
            # Current metrics
            current_volume = day_data.get('v', 0)
            current_price = day_data.get('c', 0)
            price_change_pct = day_data.get('p', 0)
            
            # Previous day metrics
            prev_volume = prev_data.get('v', 0)
            prev_close = prev_data.get('c', 0)
            
            # Skip if missing critical data
            if current_price <= 0 or prev_close <= 0 or current_volume <= 0:
                return None
            
            # Apply basic filters
            if current_price < 5 or current_price > 300:  # Price range filter
                return None
            
            if current_volume < 100000:  # Minimum volume filter
                return None
            
            # Calculate momentum signals
            volume_ratio = current_volume / prev_volume if prev_volume > 0 else 1
            momentum_score = 0
            
            # Price momentum (weighted heavily)
            if price_change_pct > 5:  # 5%+ move
                momentum_score += 30
            elif price_change_pct > 3:  # 3%+ move
                momentum_score += 20
            elif price_change_pct > 1:  # 1%+ move
                momentum_score += 10
            
            # Volume momentum
            if volume_ratio > 3:  # 3x volume spike
                momentum_score += 25
            elif volume_ratio > 2:  # 2x volume spike
                momentum_score += 15
            elif volume_ratio > 1.5:  # 1.5x volume spike
                momentum_score += 10
            
            # Price gap analysis
            if prev_close > 0:
                gap_pct = abs(current_price - prev_close) / prev_close
                if gap_pct > 0.05:  # 5%+ gap
                    momentum_score += 15
                elif gap_pct > 0.03:  # 3%+ gap
                    momentum_score += 10
            
            # Return candidate if momentum score is significant
            if momentum_score >= 15:  # Minimum momentum threshold
                return {
                    'symbol': symbol,
                    'momentum_score': momentum_score,
                    'price_change_pct': price_change_pct,
                    'volume_ratio': volume_ratio,
                    'current_price': current_price,
                    'current_volume': current_volume
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Momentum analysis failed for ticker: {e}")
            return None


class SimpleSnapshot:
    """Simple snapshot object for easier processing"""
    
    def __init__(self, data: Dict):
        self.ticker = data.get('ticker', '')
        self.day = None
        self.lastTrade = None
        self.prevDay = None
        
        # Day data
        if 'day' in data:
            day_data = data['day']
            self.day = type('Day', (), {
                'volume': day_data.get('v', 0),
                'change': day_data.get('c', 0),
                'change_percent': day_data.get('p', 0),
                'open': day_data.get('o', 0),
                'high': day_data.get('h', 0),
                'low': day_data.get('l', 0),
                'close': day_data.get('c', 0)
            })()
        
        # Last trade
        if 'lastTrade' in data:
            trade_data = data['lastTrade']
            self.lastTrade = type('LastTrade', (), {
                'p': trade_data.get('p', 0),  # Price
                's': trade_data.get('s', 0),  # Size
                't': trade_data.get('t', 0)   # Timestamp
            })()
        
        # Previous day
        if 'prevDay' in data:
            prev_data = data['prevDay']
            self.prevDay = type('PrevDay', (), {
                'c': prev_data.get('c', 0),   # Close
                'v': prev_data.get('v', 0),   # Volume
                'o': prev_data.get('o', 0),   # Open
                'h': prev_data.get('h', 0),   # High
                'l': prev_data.get('l', 0)    # Low
            })()


class SimpleBar:
    """Simple bar object for easier processing"""
    
    def __init__(self, data: Dict):
        self.close = data.get('c', 0)
        self.high = data.get('h', 0)
        self.low = data.get('l', 0)
        self.open = data.get('o', 0)
        self.volume = data.get('v', 0)
        self.timestamp = data.get('t', 0)


class SimpleTrade:
    """Simple trade object for easier processing"""
    
    def __init__(self, data: Dict):
        self.price = data.get('price', 0)
        self.size = data.get('size', 0)
        self.timestamp = data.get('participant_timestamp', 0)
        self.exchange = data.get('exchange', 0)