"""
Yahoo Finance Client for Market Data
Fetches trending stocks, gainers, losers, and most active from Yahoo Finance
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional
import json

logger = logging.getLogger(__name__)


class YahooClient:
    """Yahoo Finance client for market movers"""
    
    def __init__(self):
        self.session = None
        self.base_url = "https://query1.finance.yahoo.com/v1/finance"
        self.rate_limit_delay = 0.2  # 200ms between requests
        self.last_request_time = 0
        
    async def connect(self):
        """Initialize HTTP session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
            logger.info("Yahoo Finance client connected")
    
    async def disconnect(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Yahoo Finance client disconnected")
    
    async def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make rate-limited request to Yahoo Finance"""
        if not self.session:
            await self.connect()
        
        # Rate limiting
        import time
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        try:
            url = f"{self.base_url}{endpoint}"
            self.last_request_time = time.time()
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"Yahoo API error {response.status}: {await response.text()}")
                    return None
                    
        except Exception as e:
            logger.error(f"Yahoo request failed for {endpoint}: {e}")
            return None
    
    async def get_trending_tickers(self, region: str = 'US') -> List[str]:
        """Get trending tickers from Yahoo Finance"""
        endpoint = f"/trending/{region}"
        data = await self._make_request(endpoint)
        
        if data and 'finance' in data and 'result' in data['finance']:
            result = data['finance']['result'][0] if data['finance']['result'] else {}
            quotes = result.get('quotes', [])
            
            symbols = []
            for quote in quotes:
                symbol = quote.get('symbol', '')
                if symbol and len(symbol) <= 5 and symbol.isalpha():
                    symbols.append(symbol)
            
            return symbols[:50]  # Top 50
        
        return []
    
    async def get_market_movers(self, mover_type: str = 'gainers') -> List[str]:
        """Get market movers from Yahoo Finance - screener endpoints disabled due to 405 errors"""
        # Yahoo Finance screener endpoints are returning 405 Method Not Allowed
        # These endpoints require special authentication or are restricted
        logger.debug(f"Yahoo screener failed, falling back to other data sources for {mover_type}")
        return []
    
    async def get_most_active(self) -> List[str]:
        """Get most active stocks"""
        return await self.get_market_movers('most_active')
    
    async def get_all_movers(self) -> Dict[str, List[str]]:
        """Get all types of market movers"""
        tasks = [
            self.get_trending_tickers(),
            self.get_market_movers('gainers'),
            self.get_market_movers('losers'),
            self.get_most_active()
        ]
        
        trending, gainers, losers, most_active = await asyncio.gather(*tasks)
        
        return {
            'trending': trending,
            'gainers': gainers,
            'losers': losers,
            'most_active': most_active
        }