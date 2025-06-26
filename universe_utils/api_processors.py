#!/usr/bin/env python3
"""
API processors for Polygon and Unusual Whales data sources.
Handles all external API interactions and data fetching.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('/home/ubuntu/moe-1/.env')

# API SDK Imports
try:
    from polygon import RESTClient as PolygonClient
    POLYGON_AVAILABLE = True
except ImportError:
    logging.warning("Polygon SDK not available")
    POLYGON_AVAILABLE = False

try:
    from unusualwhales import UnusualWhalesClient
    UW_AVAILABLE = True
except ImportError:
    logging.warning("Unusual Whales SDK not available")
    UW_AVAILABLE = False

# Local imports - updated paths
from .data_structures import UnusualWhalesData, PolygonData
# Note: SimplifiedSentimentAnalyzer is imported only when needed to avoid memory conflicts

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Production-ready cache manager for API responses and calculations.
    Reduces API calls and improves performance.
    """
    
    def __init__(self, cache_dir: Path = None):
        if cache_dir is None:
            cache_dir = Path('/home/ubuntu/moe-1/cache')
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_key(self, prefix: str, symbol: str, date: Optional[str] = None) -> str:
        """Generate cache key for data."""
        if date:
            return f"{prefix}_{symbol}_{date}"
        return f"{prefix}_{symbol}_{datetime.now().strftime('%Y%m%d')}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get full cache file path."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get(self, prefix: str, symbol: str, date: Optional[str] = None) -> Optional[Any]:
        """Retrieve data from cache if valid."""
        import pickle
        
        cache_key = self._get_cache_key(prefix, symbol, date)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            # Check if cache is still valid (24 hours)
            file_age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
            
            if file_age_hours < 24:
                try:
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                        logger.debug(f"✅ Cache hit: {cache_key}")
                        return data
                except Exception as e:
                    logger.warning(f"Cache read error for {cache_key}: {e}")
        
        return None
    
    def set(self, prefix: str, symbol: str, data: Any, date: Optional[str] = None):
        """Store data in cache."""
        import pickle
        
        cache_key = self._get_cache_key(prefix, symbol, date)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
                logger.debug(f"💾 Cached: {cache_key}")
        except Exception as e:
            logger.warning(f"Cache write error for {cache_key}: {e}")
    
    def clear_old_cache(self, max_age_days: int = 7):
        """Clear cache files older than max_age_days."""
        current_time = time.time()
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            file_age_days = (current_time - cache_file.stat().st_mtime) / (24 * 3600)
            
            if file_age_days > max_age_days:
                try:
                    cache_file.unlink()
                    logger.info(f"🗑️  Deleted old cache: {cache_file.name}")
                except Exception as e:
                    logger.warning(f"Error deleting cache {cache_file}: {e}")


class APIClients:
    """Centralized API client management with improved initialization."""
    
    def __init__(self):
        # Initialize Polygon client with proper session management
        polygon_api_key = os.getenv('POLYGON_API_KEY')
        if not polygon_api_key:
            raise ValueError("POLYGON_API_KEY not found in environment variables")
        
        if POLYGON_AVAILABLE:
            # Configure session with larger connection pool
            import requests
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=20,
                max_retries=3
            )
            session.mount('https://', adapter)
            session.mount('http://', adapter)
            
            self.polygon = PolygonClient(polygon_api_key)
            # Set the session on the client if possible
            if hasattr(self.polygon, '_session'):
                self.polygon._session = session
        else:
            self.polygon = None
        
        # Store API token for Unusual Whales instead of client
        self.uw_api_token = os.getenv('UW_API_TOKEN')
        if not self.uw_api_token:
            raise ValueError("UW_API_TOKEN not found in environment variables")
        
        logger.info("📡 API credentials loaded successfully")
    
    def test_connections(self) -> Dict[str, bool]:
        """Test API connections."""
        results = {}
        
        # Test Polygon
        if self.polygon and POLYGON_AVAILABLE:
            try:
                self.polygon.get_ticker_details("AAPL")
                results['polygon'] = True
            except Exception as e:
                logger.error(f"Polygon connection test failed: {e}")
                results['polygon'] = False
        else:
            results['polygon'] = False
        
        # Test Unusual Whales
        try:
            headers = {
                'Authorization': f'Bearer {self.uw_api_token}',
                'Content-Type': 'application/json'
            }
            # Use correct endpoint from test file
            response = requests.get(
                "https://api.unusualwhales.com/api/news/headlines",
                headers=headers,
                timeout=10
            )
            results['unusual_whales'] = response.status_code == 200
        except Exception as e:
            logger.error(f"Unusual Whales connection test failed: {e}")
            results['unusual_whales'] = False
        
        return results


class UnusualWhalesDataProcessor:
    """Production-ready Unusual Whales data processor with enhanced rate limiting."""
    
    def __init__(self, api_token: str):
        """Initialize with API token and enhanced rate limiting."""
        self.api_token = api_token
        self.base_url = "https://api.unusualwhales.com"
        self.cache = CacheManager()
        
        # Enhanced rate limiting parameters
        self._base_delay = 0.5  # Base delay between requests (500ms)
        self._last_request_time = 0
        self._consecutive_rate_limits = 0
        self._circuit_breaker_failures = 0
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_reset_time = 300  # 5 minutes
        self._circuit_breaker_opened_time = 0
        
        # Request statistics
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        
        # Session with connection pooling
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=5,  # Reduced from default
            pool_maxsize=10,     # Reduced from default
            max_retries=0        # Handle retries manually
        )
        self._session.mount('https://', adapter)
        self._session.mount('http://', adapter)
        
        logger.info("Enhanced Unusual Whales processor initialized with conservative rate limiting")
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._circuit_breaker_failures >= self._circuit_breaker_threshold:
            if time.time() - self._circuit_breaker_opened_time < self._circuit_breaker_reset_time:
                return True
            else:
                # Reset circuit breaker
                self._circuit_breaker_failures = 0
                self._circuit_breaker_opened_time = 0
                logger.info("Circuit breaker reset - resuming API requests")
                return False
        return False
    
    def _handle_rate_limit(self, attempt: int = 1) -> float:
        """Implement exponential backoff with jitter for rate limiting."""
        self._consecutive_rate_limits += 1
        
        # Exponential backoff with jitter: base * (2^attempt) + random(0, 1)
        import random
        base_delay = 2.0  # Start with 2 seconds
        max_delay = 120.0  # Cap at 2 minutes
        
        exponential_delay = base_delay * (2 ** min(attempt, 6))  # Cap exponential growth
        jitter = random.uniform(0, 1)
        delay = min(exponential_delay + jitter, max_delay)
        
        logger.warning(f"Rate limited (attempt {attempt}). Waiting {delay:.1f} seconds...")
        time.sleep(delay)
        
        return delay
    
    def _adaptive_delay(self):
        """Implement adaptive delay based on recent success rate."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        # Calculate success rate
        if self._total_requests > 0:
            success_rate = self._successful_requests / self._total_requests
        else:
            success_rate = 1.0
        
        # Adaptive delay based on success rate
        if success_rate > 0.9:
            min_delay = self._base_delay  # Normal delay
        elif success_rate > 0.7:
            min_delay = self._base_delay * 1.5  # Slightly longer
        elif success_rate > 0.5:
            min_delay = self._base_delay * 2.0  # Double delay
        else:
            min_delay = self._base_delay * 3.0  # Triple delay
        
        # Ensure minimum delay
        if time_since_last < min_delay:
            sleep_time = min_delay - time_since_last
            time.sleep(sleep_time)
    
    def _make_request(self, endpoint: str, params: Dict = None, max_retries: int = 3) -> Optional[Dict]:
        """Make HTTP request with enhanced error handling and circuit breaker."""
        # Check circuit breaker
        if self._is_circuit_breaker_open():
            logger.warning("Circuit breaker is open - skipping API request")
            return None
        
        # Implement adaptive delay
        self._adaptive_delay()
        
        for attempt in range(max_retries + 1):
            try:
                self._total_requests += 1
                
                headers = {
                    'Authorization': f'Bearer {self.api_token}',
                    'Content-Type': 'application/json',
                    'User-Agent': 'UniverseUpdater/5.0',
                    'Accept': 'application/json'
                }
                
                url = f"{self.base_url}{endpoint}"
                
                # Make request with session
                response = self._session.get(
                    url, 
                    headers=headers, 
                    params=params or {}, 
                    timeout=45,  # Increased timeout
                    stream=False
                )
                
                self._last_request_time = time.time()
                
                if response.status_code == 200:
                    self._consecutive_rate_limits = 0  # Reset on success
                    self._successful_requests += 1
                    self._circuit_breaker_failures = 0  # Reset circuit breaker
                    
                    try:
                        data = response.json()
                        # Handle both dict and list responses
                        if isinstance(data, dict):
                            return data
                        elif isinstance(data, list):
                            return {'data': data}  # Wrap list in dict
                        return data
                    except ValueError as e:
                        logger.warning(f"Invalid JSON response from {endpoint}: {e}")
                        return None
                        
                elif response.status_code == 401:
                    logger.error(f"Unauthorized access to {endpoint} - check API token")
                    self._failed_requests += 1
                    return None
                    
                elif response.status_code == 429:
                    # Rate limited - implement backoff
                    if attempt < max_retries:
                        delay = self._handle_rate_limit(attempt + 1)
                        continue
                    else:
                        logger.error(f"Max retries exceeded for rate limiting on {endpoint}")
                        self._failed_requests += 1
                        return None
                        
                elif response.status_code == 403:
                    logger.error(f"Forbidden access to {endpoint} - check permissions")
                    self._failed_requests += 1
                    return None
                    
                elif response.status_code >= 500:
                    # Server error - retry with backoff
                    if attempt < max_retries:
                        delay = 2 ** attempt  # Simple exponential backoff
                        logger.warning(f"Server error {response.status_code} for {endpoint}, retrying in {delay}s")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Server error {response.status_code} for {endpoint} after {max_retries} retries")
                        self._failed_requests += 1
                        self._circuit_breaker_failures += 1
                        return None
                        
                else:
                    logger.warning(f"API request failed for {endpoint}: {response.status_code} - {response.text[:200]}")
                    self._failed_requests += 1
                    
                    if attempt < max_retries:
                        time.sleep(1)  # Brief delay before retry
                        continue
                    else:
                        return None
                        
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout for {endpoint} (attempt {attempt + 1})")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    self._failed_requests += 1
                    self._circuit_breaker_failures += 1
                    return None
                    
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error for {endpoint}: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    self._failed_requests += 1
                    self._circuit_breaker_failures += 1
                    return None
                    
            except Exception as e:
                logger.warning(f"Unexpected error for {endpoint}: {e}")
                self._failed_requests += 1
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                else:
                    return None
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        success_rate = (self._successful_requests / self._total_requests * 100) if self._total_requests > 0 else 0
        
        return {
            'total_requests': self._total_requests,
            'successful_requests': self._successful_requests,
            'failed_requests': self._failed_requests,
            'success_rate_percent': round(success_rate, 1),
            'consecutive_rate_limits': self._consecutive_rate_limits,
            'circuit_breaker_failures': self._circuit_breaker_failures,
            'circuit_breaker_open': self._is_circuit_breaker_open()
        }
    
    def get_flow_alerts(self, symbol: str) -> List[Dict]:
        """Get recent flow alerts for symbol with caching."""
        cached_data = self.cache.get("uw_flow", symbol)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # Use correct endpoint from test file
            endpoint = f"/api/stock/{symbol}/flow-alerts"
            response_data = self._make_request(endpoint)
            
            if response_data and 'data' in response_data:
                result = response_data['data'][:10]  # Keep top 10
                self.cache.set("uw_flow", symbol, result)
                return result
            elif response_data and isinstance(response_data, list):
                result = response_data[:10]  # Keep top 10
                self.cache.set("uw_flow", symbol, result)
                return result
            else:
                return []
                
        except Exception as e:
            logger.warning(f"Error fetching flow alerts for {symbol}: {e}")
            return []
    
    def get_gamma_exposure(self, symbol: str) -> Dict[str, float]:
        """Get gamma exposure data with proper error handling."""
        cached_data = self.cache.get("uw_gamma", symbol)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # Use correct endpoint from test file
            endpoint = f"/api/stock/{symbol}/greek-exposure"
            response_data = self._make_request(endpoint)
            
            if response_data:
                # Handle the response properly whether it's a dict or wrapped list
                if 'data' in response_data:
                    data = response_data['data']
                    # If data is a list, take the first item
                    if isinstance(data, list) and len(data) > 0:
                        data = data[0]
                else:
                    data = response_data
                
                # Extract values with proper type checking
                result = {
                    'gamma_exposure': float(data.get('gex', 0)) if isinstance(data, dict) else 0,
                    'delta_exposure': float(data.get('dex', 0)) if isinstance(data, dict) else 0,
                    'vanna_exposure': float(data.get('vanna', 0)) if isinstance(data, dict) else 0,
                    'charm_exposure': float(data.get('charm', 0)) if isinstance(data, dict) else 0,
                    'max_pain': float(data.get('max_pain', 0)) if isinstance(data, dict) else 0
                }
                self.cache.set("uw_gamma", symbol, result)
                return result
        except Exception as e:
            logger.warning(f"Error fetching gamma exposure for {symbol}: {e}")
        
        return {
            'gamma_exposure': 0, 'delta_exposure': 0,
            'vanna_exposure': 0, 'charm_exposure': 0,
            'max_pain': 0
        }
    
    def get_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """Get all Unusual Whales data for a symbol."""
        data = {}
        
        # Get all data types with error handling
        data['gamma'] = self.get_gamma_exposure(symbol)
        data['flow_alerts'] = self.get_flow_alerts(symbol)
        data['dark_pool'] = self.get_dark_pool_data(symbol)
        data['congressional'] = self.get_congressional_activity(symbol)
        data['insider'] = self.get_insider_data(symbol)
        data['institutional'] = self.get_institutional_data(symbol)
        data['calendar'] = self.get_calendar_events(symbol)
        data['options'] = self.get_options_data(symbol)
        data['short'] = self.get_short_data(symbol)
        data['etf'] = self.get_etf_data(symbol)
        
        return data
    
    def get_dark_pool_data(self, symbol: str) -> Dict[str, float]:
        """Get dark pool trading data with proper error handling."""
        cached_data = self.cache.get("uw_darkpool", symbol)
        if cached_data is not None:
            return cached_data
        
        try:
            endpoint = f"/api/darkpool/{symbol}"
            params = {'date': datetime.now().strftime('%Y-%m-%d')}
            response_data = self._make_request(endpoint, params)
            
            if response_data and 'data' in response_data:
                trades = response_data['data']
                total_volume = sum(float(trade.get('size', 0)) for trade in trades)
                
                result = {
                    'dark_pool_volume_24h': total_volume,
                    'dark_pool_prints_count': len(trades),
                    'lit_dark_ratio': 0.0,  # Would need lit volume to calculate
                    'avg_dark_pool_size': total_volume / len(trades) if trades else 0
                }
                
                self.cache.set("uw_darkpool", symbol, result)
                return result
                    
        except Exception as e:
            logger.warning(f"Error fetching dark pool data for {symbol}: {e}")
        
        return {
            'dark_pool_volume_24h': 0,
            'dark_pool_prints_count': 0,
            'lit_dark_ratio': 0,
            'avg_dark_pool_size': 0
        }
    
    def get_congressional_activity(self, symbol: str) -> Dict[str, Any]:
        """Check for recent congressional activity with date handling."""
        try:
            # Use correct endpoint from test file
            endpoint = "/api/congress/recent-trades"
            response_data = self._make_request(endpoint)
            
            if response_data and 'data' in response_data:
                data = response_data['data']
                if not isinstance(data, list):
                    logger.warning(f"Congressional data is not a list for {symbol}")
                    data = []
                
                symbol_trades = []
                for trade in data:
                    if isinstance(trade, dict) and trade.get('ticker') == symbol:
                        symbol_trades.append(trade)
                
                if symbol_trades:
                    # Analyze sentiment from trades
                    buy_trades = 0
                    sell_trades = 0
                    
                    for trade in symbol_trades:
                        if not isinstance(trade, dict):
                            continue
                        transaction_type = str(trade.get('transaction_type', '')).lower()
                        if 'buy' in transaction_type:
                            buy_trades += 1
                        elif 'sell' in transaction_type:
                            sell_trades += 1
                    
                    sentiment = 'bullish' if buy_trades > sell_trades else 'bearish' if sell_trades > buy_trades else 'neutral'
                    
                    return {
                        'congressional_activity_recent': True,
                        'congressional_sentiment': sentiment,
                        'congressional_trades_count': len(symbol_trades),
                        'buy_trades': buy_trades,
                        'sell_trades': sell_trades
                    }
            elif response_data and isinstance(response_data, list):
                # Handle direct list response
                symbol_trades = []
                for trade in response_data:
                    if isinstance(trade, dict) and trade.get('ticker') == symbol:
                        symbol_trades.append(trade)
                
                if symbol_trades:
                    # Analyze sentiment from trades
                    buy_trades = 0
                    sell_trades = 0
                    
                    for trade in symbol_trades:
                        if not isinstance(trade, dict):
                            continue
                        transaction_type = str(trade.get('transaction_type', '')).lower()
                        if 'buy' in transaction_type:
                            buy_trades += 1
                        elif 'sell' in transaction_type:
                            sell_trades += 1
                    
                    sentiment = 'bullish' if buy_trades > sell_trades else 'bearish' if sell_trades > buy_trades else 'neutral'
                    
                    return {
                        'congressional_activity_recent': True,
                        'congressional_sentiment': sentiment,
                        'congressional_trades_count': len(symbol_trades),
                        'buy_trades': buy_trades,
                        'sell_trades': sell_trades
                    }
                        
        except Exception as e:
            logger.warning(f"Error fetching congressional data for {symbol}: {e}")
        
        return {
            'congressional_activity_recent': False,
            'congressional_sentiment': 'neutral',
            'congressional_trades_count': 0,
            'buy_trades': 0,
            'sell_trades': 0
        }
    
    def get_insider_data(self, symbol: str) -> Dict[str, Any]:
        """Get insider trading data with proper calculations."""
        try:
            endpoint = f"/api/insider/{symbol}"
            response_data = self._make_request(endpoint)
            
            if response_data and 'data' in response_data:
                trades = response_data['data']
                
                # Calculate buy/sell ratio
                buy_trades = sum(1 for t in trades if 'buy' in str(t.get('transaction_type', '')).lower())
                sell_trades = sum(1 for t in trades if 'sell' in str(t.get('transaction_type', '')).lower())
                
                buy_sell_ratio = buy_trades / sell_trades if sell_trades > 0 else buy_trades
                
                return {
                    'insider_transactions_30d': len(trades),
                    'insider_buy_sell_ratio': buy_sell_ratio,
                    'insider_buy_count': buy_trades,
                    'insider_sell_count': sell_trades
                }
                    
        except Exception as e:
            logger.warning(f"Error fetching insider data for {symbol}: {e}")
        
        return {
            'insider_transactions_30d': 0,
            'insider_buy_sell_ratio': 0,
            'insider_buy_count': 0,
            'insider_sell_count': 0
        }
    
    def get_institutional_data(self, symbol: str) -> Dict[str, Any]:
        """Get institutional ownership data."""
        try:
            endpoint = f"/api/institution/{symbol}/ownership"
            response_data = self._make_request(endpoint)
            
            if response_data and 'data' in response_data:
                holdings = response_data['data']
                
                # Calculate ownership changes
                total_shares = sum(float(holding.get('shares', 0)) for holding in holdings)
                
                # Get recent changes (would need historical data)
                recent_changes = {}
                
                return {
                    'institutional_ownership_changes': recent_changes,
                    'latest_13f_changes': {},
                    'total_institutional_shares': total_shares,
                    'number_of_institutions': len(holdings)
                }
                    
        except Exception as e:
            logger.warning(f"Error fetching institutional data for {symbol}: {e}")
        
        return {
            'institutional_ownership_changes': {},
            'latest_13f_changes': {},
            'total_institutional_shares': 0,
            'number_of_institutions': 0
        }
    
    def get_calendar_events(self, symbol: str) -> Dict[str, int]:
        """Get proximity to calendar events with proper date calculations."""
        calendar_data = {
            'earnings_date_proximity': 999,
            'fda_event_proximity': 999,
            'economic_event_proximity': 999,
            'dividend_ex_date_proximity': 999
        }
        
        try:
            # Check earnings calendar
            earnings_endpoint = f"/api/earnings/{symbol}"
            earnings_data = self._make_request(earnings_endpoint)
            
            if earnings_data and 'data' in earnings_data:
                # Find next earnings date
                today = datetime.now().date()
                
                for earning in earnings_data['data']:
                    if earning.get('date'):
                        earning_date = datetime.strptime(str(earning['date']), '%Y-%m-%d').date()
                        if earning_date >= today:
                            days_until = (earning_date - today).days
                            calendar_data['earnings_date_proximity'] = days_until
                            break
            
            # Check FDA calendar
            fda_endpoint = "/api/market/fda-calendar"
            fda_data = self._make_request(fda_endpoint)
            
            if fda_data and 'data' in fda_data:
                today = datetime.now().date()
                for event in fda_data['data']:
                    if event.get('ticker') == symbol and event.get('date'):
                        event_date = datetime.strptime(str(event['date']), '%Y-%m-%d').date()
                        if event_date >= today:
                            days_until = (event_date - today).days
                            calendar_data['fda_event_proximity'] = days_until
                            break
                
        except Exception as e:
            logger.warning(f"Error fetching calendar data for {symbol}: {e}")
        
        return calendar_data
    
    def get_options_data(self, symbol: str) -> Dict[str, float]:
        """Get comprehensive options analysis data."""
        try:
            # IV Rank
            iv_endpoint = f"/api/stock/{symbol}/iv-rank"
            iv_data = self._make_request(iv_endpoint)
            iv_rank = float(iv_data.get('iv_rank', 0)) if iv_data else 0
            
            # Options Volume
            vol_endpoint = f"/api/stock/{symbol}/options-volume"
            params = {'date': datetime.now().strftime('%Y-%m-%d')}
            vol_data = self._make_request(vol_endpoint, params)
            
            options_volume = 0
            call_volume = 0
            put_volume = 0
            
            if vol_data and 'data' in vol_data:
                for item in vol_data['data']:
                    volume = float(item.get('volume', 0))
                    options_volume += volume
                    
                    symbol_str = str(item.get('symbol', ''))
                    if 'C' in symbol_str:
                        call_volume += volume
                    elif 'P' in symbol_str:
                        put_volume += volume
            
            put_call_ratio = put_volume / call_volume if call_volume > 0 else 1.0
            
            # Check for unusual activity
            unusual_activity = iv_rank > 75 or put_call_ratio > 2.0 or put_call_ratio < 0.5
            
            return {
                'iv_rank': iv_rank,
                'options_volume_vs_avg': 1.0,  # Would need historical average
                'unusual_options_activity': unusual_activity,
                'put_call_ratio': put_call_ratio,
                'total_options_volume': options_volume,
                'call_volume': call_volume,
                'put_volume': put_volume
            }
                
        except Exception as e:
            logger.warning(f"Error fetching options data for {symbol}: {e}")
        
        return {
            'iv_rank': 0,
            'options_volume_vs_avg': 0,
            'unusual_options_activity': False,
            'put_call_ratio': 0,
            'total_options_volume': 0,
            'call_volume': 0,
            'put_volume': 0
        }
    
    def get_short_data(self, symbol: str) -> Dict[str, float]:
        """Get short interest data."""
        try:
            endpoint = f"/api/shorts/{symbol}/interest-float"
            response_data = self._make_request(endpoint)
            
            if response_data:
                return {
                    'short_interest_ratio': float(response_data.get('short_interest_ratio', 0)),
                    'days_to_cover': float(response_data.get('days_to_cover', 0)),
                    'short_float_percent': float(response_data.get('short_float_percent', 0)),
                }
                
        except Exception as e:
            logger.warning(f"Error fetching short data for {symbol}: {e}")
        
        return {
            'short_interest_ratio': 0,
            'days_to_cover': 0,
            'short_float_percent': 0
        }

    def get_etf_data(self, symbol: str) -> Dict[str, float]:
        """Get ETF-related data."""
        try:
            endpoint = f"/api/etfs/{symbol}/in-outflow"
            response_data = self._make_request(endpoint)
            
            if response_data:
                return {
                    'etf_inflow_outflow': float(response_data.get('net_flow', 0)),
                    'etf_rebalance_pressure': float(response_data.get('rebalance_pressure', 0))
                }
                    
        except Exception as e:
            logger.warning(f"Error fetching ETF data for {symbol}: {e}")
        
        return {
            'etf_inflow_outflow': 0,
            'etf_rebalance_pressure': 0
        }
    
    def get_news_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Get news data from Unusual Whales news feed."""
        try:
            # Use correct endpoint from test file
            endpoint = "/api/news/headlines"
            response_data = self._make_request(endpoint)
            
            symbol_news = []
            if response_data and 'data' in response_data:
                for article in response_data['data']:
                    if isinstance(article, dict):
                        # Check if article mentions this symbol
                        title = article.get('title', '')
                        content = article.get('content', '')
                        tickers = article.get('tickers', [])
                        
                        if symbol in tickers or symbol.lower() in title.lower() or symbol.lower() in content.lower():
                            symbol_news.append({
                                'title': title,
                                'content': content,
                                'published': article.get('published_at', ''),
                                'source': 'unusual_whales',
                                'sentiment': article.get('sentiment', 'neutral'),
                                'tickers': tickers
                            })
            elif response_data and isinstance(response_data, list):
                # Handle direct list response
                for article in response_data:
                    if isinstance(article, dict):
                        title = article.get('title', '')
                        content = article.get('content', '')
                        tickers = article.get('tickers', [])
                        
                        if symbol in tickers or symbol.lower() in title.lower() or symbol.lower() in content.lower():
                            symbol_news.append({
                                'title': title,
                                'content': content,
                                'published': article.get('published_at', ''),
                                'source': 'unusual_whales',
                                'sentiment': article.get('sentiment', 'neutral'),
                                'tickers': tickers
                            })
            
            return symbol_news[:10]  # Return top 10 most recent
                
        except Exception as e:
            logger.warning(f"Error fetching Unusual Whales news for {symbol}: {e}")
        
        return []


class PolygonDataProcessor:
    """Enhanced Polygon data processor with FinGPT integration."""
    
    def __init__(self, client: PolygonClient, use_fingpt: bool = True):
        self.client = client
        self.cache = CacheManager()
        self.use_fingpt = use_fingpt
        
        if self.use_fingpt:
            try:
                self.sentiment_analyzer = SimplifiedSentimentAnalyzer()
                logger.info("✅ Enhanced FinGPT sentiment analyzer initialized")
            except Exception as e:
                logger.warning(f"Could not initialize FinGPT: {e}")
                self.use_fingpt = False
                self.sentiment_analyzer = None
        else:
            self.sentiment_analyzer = None
    
    def get_historical_data(self, symbol: str, days: int = 90) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data with caching."""
        cache_key = f"polygon_historical_{symbol}_{days}d"
        cached_data = self.cache.get("polygon_hist", symbol, f"{days}d")
        
        if cached_data is not None:
            return pd.DataFrame(cached_data)
        
        if not self.client:
            logger.warning("Polygon client not available")
            return None
        
        try:
            # Add small delay to help with connection pool management
            time.sleep(0.1)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 30)  # Extra days for moving averages
            
            aggs = self.client.list_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                limit=5000
            )
            
            data = []
            for agg in aggs:
                data.append({
                    'timestamp': datetime.fromtimestamp(agg.timestamp / 1000),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume,
                    'vwap': getattr(agg, 'vwap', agg.close),
                    'transactions': getattr(agg, 'transactions', 0)
                })
            
            if len(data) > 20:
                df = pd.DataFrame(data)
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Cache the data
                self.cache.set("polygon_hist", symbol, df.to_dict(), f"{days}d")
                
                return df
                
        except Exception as e:
            logger.warning(f"Error fetching historical data for {symbol}: {e}")
        
        return None
    
    def get_enhanced_news_data(self, symbol: str) -> Dict[str, Any]:
        """Get news with advanced FinGPT sentiment analysis."""
        if not self.client:
            logger.warning("Polygon client not available")
            return self._empty_news_response()
        
        try:
            # Get news from last 24 hours
            yesterday = datetime.now() - timedelta(days=1)
            
            news = self.client.list_ticker_news(
                ticker=symbol,
                published_utc_gte=yesterday.strftime('%Y-%m-%d'),
                limit=50
            )
            
            articles = []
            for article in news:
                articles.append({
                    'title': article.title,
                    'description': getattr(article, 'description', ''),
                    'published_utc': article.published_utc,
                    'publisher': {
                        'name': article.publisher.name if hasattr(article, 'publisher') else 'Unknown'
                    },
                    'url': getattr(article, 'article_url', '')
                })
            
            if self.use_fingpt and self.sentiment_analyzer and articles:
                # Use FinGPT for advanced analysis
                news_analysis = self.sentiment_analyzer.analyze_news_batch(articles, symbol)
                
                return {
                    'news_sentiment_score': news_analysis.overall_sentiment.sentiment_score,
                    'news_count_24h': len(articles),
                    'news_relevance_score': news_analysis.overall_sentiment.confidence,
                    'sentiment_confidence': news_analysis.overall_sentiment.confidence,
                    'sentiment_label': news_analysis.overall_sentiment.sentiment_label,
                    'key_topics': news_analysis.overall_sentiment.key_topics,
                    'market_impact_score': news_analysis.overall_sentiment.market_impact_score,
                    'sentiment_momentum': news_analysis.sentiment_momentum,
                    'controversy_score': news_analysis.controversy_score,
                    'entity_sentiments': news_analysis.overall_sentiment.entity_sentiments,
                    'credibility_weighted_score': news_analysis.credibility_weighted_score,
                    'articles': articles[:5]  # Keep top 5 articles
                }
            else:
                # Fallback to basic sentiment analysis
                return self._basic_sentiment_analysis(articles, symbol)
                
        except Exception as e:
            logger.warning(f"Error fetching news for {symbol}: {e}")
        
        return self._empty_news_response()
    
    def _basic_sentiment_analysis(self, articles: List[Dict], symbol: str) -> Dict[str, Any]:
        """Basic sentiment analysis fallback."""
        news_count = len(articles)
        sentiment_scores = []
        
        for article in articles:
            # Basic keyword-based sentiment
            title = article['title'].lower()
            sentiment = 0
            
            positive_words = ['up', 'gain', 'rise', 'bull', 'positive', 'beat', 'strong', 'growth']
            negative_words = ['down', 'fall', 'drop', 'bear', 'negative', 'miss', 'weak', 'loss']
            
            sentiment += sum(1 for word in positive_words if word in title)
            sentiment -= sum(1 for word in negative_words if word in title)
            
            sentiment_scores.append(sentiment)
        
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        
        return {
            'news_sentiment_score': avg_sentiment / 3.0,  # Normalize
            'news_count_24h': news_count,
            'news_relevance_score': 0.5,
            'sentiment_confidence': 0.5,
            'sentiment_label': 'bullish' if avg_sentiment > 0 else 'bearish' if avg_sentiment < 0 else 'neutral',
            'key_topics': [],
            'market_impact_score': 0.5,
            'sentiment_momentum': 0.0,
            'controversy_score': 0.0,
            'entity_sentiments': {symbol: avg_sentiment / 3.0},
            'credibility_weighted_score': avg_sentiment / 3.0,
            'articles': articles[:5]
        }
    
    def _empty_news_response(self) -> Dict[str, Any]:
        """Return empty news response structure."""
        return {
            'news_sentiment_score': 0,
            'news_count_24h': 0,
            'news_relevance_score': 0,
            'sentiment_confidence': 0,
            'sentiment_label': 'neutral',
            'key_topics': [],
            'market_impact_score': 0,
            'sentiment_momentum': 0,
            'controversy_score': 0,
            'entity_sentiments': {},
            'credibility_weighted_score': 0,
            'articles': []
        }
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive technical indicators using pandas-ta."""
        if len(df) < 50:
            return {}
        
        try:
            import pandas_ta as ta
            from scipy import stats
            
            # Create a copy to avoid modifying original
            df_ta = df.copy()
            
            # Calculate all indicators using pandas-ta
            # Moving Averages
            df_ta.ta.sma(length=20, append=True)
            df_ta.ta.sma(length=50, append=True)
            df_ta.ta.sma(length=200, append=True)
            df_ta.ta.ema(length=12, append=True)
            df_ta.ta.ema(length=26, append=True)
            
            # VWAP
            df_ta.ta.vwap(append=True)
            
            # ATR
            df_ta.ta.atr(length=14, append=True)
            df_ta.ta.natr(length=14, append=True)
            
            # Supertrend
            supertrend_df = df_ta.ta.supertrend(length=7, multiplier=3)
            if supertrend_df is not None and not supertrend_df.empty:
                df_ta = pd.concat([df_ta, supertrend_df], axis=1)
            
            # Keltner Channels
            kc_df = df_ta.ta.kc(length=20, scalar=2)
            if kc_df is not None and not kc_df.empty:
                df_ta = pd.concat([df_ta, kc_df], axis=1)
            
            # RSI
            df_ta.ta.rsi(length=14, append=True)
            
            # MACD
            macd_df = df_ta.ta.macd(fast=12, slow=26, signal=9)
            if macd_df is not None and not macd_df.empty:
                df_ta = pd.concat([df_ta, macd_df], axis=1)
            
            # Bollinger Bands
            bbands_df = df_ta.ta.bbands(length=20, std=2)
            if bbands_df is not None and not bbands_df.empty:
                df_ta = pd.concat([df_ta, bbands_df], axis=1)
            
            # Volume indicators
            df_ta['volume_sma_20'] = df_ta['volume'].rolling(window=20).mean()
            
            # On-Balance Volume
            df_ta.ta.obv(append=True)
            
            # Money Flow Index
            df_ta.ta.mfi(length=14, append=True)
            
            # ADX
            df_ta.ta.adx(length=14, append=True)
            
            # CCI
            df_ta.ta.cci(length=20, append=True)
            
            # Stochastic
            stoch_df = df_ta.ta.stoch(high='high', low='low', close='close', k=14, d=3)
            if stoch_df is not None and not stoch_df.empty:
                df_ta = pd.concat([df_ta, stoch_df], axis=1)
            
            # Extract the latest values
            indicators = {}
            
            # Helper function to safely get last value
            def safe_get_last(df, column, default=0.0):
                if column in df and len(df) > 0 and not df[column].empty:
                    try:
                        last_val = df[column].iloc[-1]
                        return float(last_val) if pd.notna(last_val) else default
                    except (IndexError, ValueError, TypeError):
                        return default
                return default
            
            # Moving Averages
            indicators['sma_20'] = safe_get_last(df_ta, 'SMA_20')
            indicators['sma_50'] = safe_get_last(df_ta, 'SMA_50')
            indicators['sma_200'] = safe_get_last(df_ta, 'SMA_200', indicators['sma_50'])
            indicators['ema_12'] = safe_get_last(df_ta, 'EMA_12')
            indicators['ema_26'] = safe_get_last(df_ta, 'EMA_26')
            
            # VWAP
            vwap_val = safe_get_last(df_ta, 'VWAP_D')
            if vwap_val == 0.0:
                vwap_val = safe_get_last(df_ta, 'close', df_ta['close'].mean() if len(df_ta) > 0 else 0.0)
            indicators['vwap'] = vwap_val
            
            # ATR
            indicators['atr_14'] = safe_get_last(df_ta, 'ATRr_14')
            indicators['natr_14'] = safe_get_last(df_ta, 'NATR_14')
            
            # Supertrend
            indicators['supertrend'] = safe_get_last(df_ta, 'SUPERT_7_3.0')
            indicators['supertrend_direction'] = safe_get_last(df_ta, 'SUPERTd_7_3.0')
            
            # Keltner Channels
            kc_upper = safe_get_last(df_ta, 'KCUe_20_2')
            kc_lower = safe_get_last(df_ta, 'KCLe_20_2')
            kc_middle = safe_get_last(df_ta, 'KCBe_20_2')
            
            if kc_upper > 0 and kc_lower > 0:
                indicators['kc_upper'] = kc_upper
                indicators['kc_lower'] = kc_lower
                indicators['kc_middle'] = kc_middle if kc_middle > 0 else safe_get_last(df_ta, 'close')
                
                # Position within Keltner Channels
                kc_range = kc_upper - kc_lower
                if kc_range > 0:
                    current_close = safe_get_last(df_ta, 'close')
                    if current_close > 0:
                        indicators['kc_position'] = float((current_close - kc_lower) / kc_range)
                    else:
                        indicators['kc_position'] = 0.5
                else:
                    indicators['kc_position'] = 0.5
            else:
                indicators['kc_upper'] = 0.0
                indicators['kc_lower'] = 0.0
                indicators['kc_middle'] = 0.0
                indicators['kc_position'] = 0.5
            
            # Oscillators
            indicators['rsi_14'] = safe_get_last(df_ta, 'RSI_14', 50.0)
            
            # MACD
            indicators['macd'] = safe_get_last(df_ta, 'MACD_12_26_9')
            indicators['macd_signal'] = safe_get_last(df_ta, 'MACDs_12_26_9')
            indicators['macd_histogram'] = safe_get_last(df_ta, 'MACDh_12_26_9')
            
            # Bollinger Bands
            bb_upper = safe_get_last(df_ta, 'BBU_20_2')
            bb_lower = safe_get_last(df_ta, 'BBL_20_2')
            
            if bb_upper > 0 and bb_lower > 0:
                indicators['bollinger_upper'] = bb_upper
                indicators['bollinger_lower'] = bb_lower
                
                # Calculate position within bands
                bb_range = bb_upper - bb_lower
                if bb_range > 0:
                    current_close = safe_get_last(df_ta, 'close')
                    if current_close > 0:
                        indicators['bollinger_position'] = float((current_close - bb_lower) / bb_range)
                    else:
                        indicators['bollinger_position'] = 0.5
                else:
                    indicators['bollinger_position'] = 0.5
            else:
                indicators['bollinger_upper'] = 0.0
                indicators['bollinger_lower'] = 0.0
                indicators['bollinger_position'] = 0.5
            
            # Volume indicators
            volume_sma = safe_get_last(df_ta, 'volume_sma_20')
            indicators['volume_sma_20'] = volume_sma
            if volume_sma > 0:
                current_volume = safe_get_last(df_ta, 'volume')
                indicators['volume_ratio_vs_avg'] = float(current_volume / volume_sma) if current_volume > 0 else 1.0
            else:
                indicators['volume_ratio_vs_avg'] = 1.0
            
            # OBV
            obv_val = safe_get_last(df_ta, 'OBV')
            if obv_val != 0.0 and 'OBV' in df_ta and len(df_ta) >= 20:
                indicators['obv'] = obv_val
                # Calculate OBV trend
                try:
                    obv_ma = df_ta['OBV'].rolling(window=20).mean()
                    if len(obv_ma) > 0 and pd.notna(obv_ma.iloc[-1]):
                        indicators['obv_trend'] = 1 if obv_val > obv_ma.iloc[-1] else -1
                    else:
                        indicators['obv_trend'] = 0
                except (IndexError, ValueError):
                    indicators['obv_trend'] = 0
            else:
                indicators['obv'] = 0.0
                indicators['obv_trend'] = 0
            
            # Money Flow Index
            indicators['mfi_14'] = safe_get_last(df_ta, 'MFI_14', 50.0)
            
            # Additional indicators
            indicators['adx'] = safe_get_last(df_ta, 'ADX_14')
            indicators['cci'] = safe_get_last(df_ta, 'CCI_20_0.015')
            
            # Stochastic
            indicators['stoch_k'] = safe_get_last(df_ta, 'STOCHk_14_3_3', 50.0)
            indicators['stoch_d'] = safe_get_last(df_ta, 'STOCHd_14_3_3', 50.0)
            
            # Data quality
            indicators['data_quality_score'] = min(1.0, len(df) / 200)
            
            # Validate all indicators
            for key, value in indicators.items():
                if pd.isna(value) or np.isinf(value):
                    indicators[key] = 0.0
            
            return indicators
            
        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {e}")
            return {}
    
    def get_company_details(self, symbol: str) -> Dict[str, Any]:
        """Get company details from Polygon."""
        if not self.client:
            logger.warning("Polygon client not available")
            return self._empty_company_details()
        
        try:
            details = self.client.get_ticker_details(symbol)
            
            return {
                'market_cap': float(getattr(details, 'market_cap', 0)),
                'shares_outstanding': float(getattr(details, 'weighted_shares_outstanding', 0)),
                'name': getattr(details, 'name', ''),
                'sector': getattr(details, 'sector', ''),
                'industry': getattr(details, 'industry', ''),
                'employees': int(getattr(details, 'total_employees', 0)),
                'description': getattr(details, 'description', '')
            }
            
        except Exception as e:
            logger.warning(f"Error fetching company details for {symbol}: {e}")
            return self._empty_company_details()
    
    def _empty_company_details(self) -> Dict[str, Any]:
        """Return empty company details structure."""
        return {
            'market_cap': 0,
            'shares_outstanding': 0,
            'name': '',
            'sector': '',
            'industry': '',
            'employees': 0,
            'description': ''
        }
