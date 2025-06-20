"""
Test Stock Candidate Finder
Uses live Polygon API to find stocks matching our momentum trading criteria
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict
import sys
import os

# Add the trading_system directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from polygon_client import PolygonClient
from yahoo_client import YahooClient
from config import POLYGON_API_KEY, UNIVERSE_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CandidateFinder:
    """Find stock candidates using Polygon API and our criteria"""
    
    def __init__(self, api_key: str):
        self.polygon = PolygonClient(api_key)
        self.yahoo = YahooClient()
        self.criteria = UNIVERSE_CONFIG
        
    async def find_candidates(self) -> Dict:
        """Find all stock candidates matching our criteria"""
        logger.info("🔍 Starting candidate search with criteria:")
        logger.info(f"   Price Range: ${self.criteria['min_price']:.2f} - ${self.criteria['max_price']:.2f}")
        logger.info(f"   Min Volume: {self.criteria['min_daily_volume']:,} shares")
        logger.info(f"   Volume Spike: {self.criteria['volume_spike_threshold']}x threshold")
        logger.info(f"   Price Move: {self.criteria['price_move_threshold']*100}% threshold")
        
        results = {
            'candidates': [],
            'stats': {
                'total_processed': 0,
                'price_filtered': 0,
                'volume_filtered': 0,
                'gap_candidates': 0,
                'volume_spike_candidates': 0,
                'market_movers': 0,
                'recent_ipos': 0,
                'yahoo_trending': 0,
                'yahoo_gainers': 0,
                'yahoo_losers': 0,
                'yahoo_most_active': 0
            }
        }
        
        try:
            # Connect to APIs
            await self.polygon.connect()
            await self.yahoo.connect()
            
            # 0. Test API connectivity first
            logger.info("🔧 Testing API connectivity...")
            test_snapshot = await self.polygon.get_snapshot_ticker('AAPL')
            if test_snapshot:
                logger.info("✅ API connectivity confirmed with AAPL snapshot")
                current_price = self._get_current_price(test_snapshot)
                daily_volume = self._get_daily_volume(test_snapshot)
                logger.info(f"   AAPL: Price=${current_price:.2f}, Volume={daily_volume:,}")
                logger.info(f"   Has day data: {test_snapshot.day is not None}")
                logger.info(f"   Has lastTrade data: {test_snapshot.lastTrade is not None}")
                logger.info(f"   Has prevDay data: {test_snapshot.prevDay is not None}")
            else:
                logger.warning("⚠️ No data returned for AAPL - API might have issues")
            
            # 1. Get all market snapshots
            logger.info("📊 Fetching all market snapshots...")
            all_snapshots = await self.polygon.get_snapshot_all_tickers()
            results['stats']['total_processed'] = len(all_snapshots)
            logger.info(f"   Retrieved {len(all_snapshots)} snapshots")
            
            # Debug: Show sample snapshot data if available
            if all_snapshots:
                sample = all_snapshots[0]
                logger.info(f"   Sample snapshot: {sample.ticker}")
                sample_price = self._get_current_price(sample)
                sample_volume = self._get_daily_volume(sample)
                logger.info(f"   Sample price: ${sample_price:.2f}, volume: {sample_volume:,}")
            else:
                logger.warning("⚠️ No market snapshots returned - this is unusual")
            
            # 2. Get market movers
            logger.info("📈 Fetching market movers...")
            gainers = await self.polygon.get_market_movers('gainers')
            losers = await self.polygon.get_market_movers('losers')
            market_movers = set(gainers + losers)
            results['stats']['market_movers'] = len(market_movers)
            logger.info(f"   Found {len(gainers)} gainers, {len(losers)} losers")
            
            # Debug: Show sample market movers
            if gainers:
                logger.info(f"   Sample gainers: {gainers[:5]}")
            if losers:
                logger.info(f"   Sample losers: {losers[:5]}")
            
            # 3. Get recent IPOs (new momentum opportunities)
            logger.info("🏢 Fetching recent IPOs...")
            recent_ipos = await self.polygon.get_recent_ipos()
            results['stats']['recent_ipos'] = len(recent_ipos)
            logger.info(f"   Found {len(recent_ipos)} recent IPOs")
            if recent_ipos:
                logger.info(f"   Sample IPOs: {recent_ipos[:5]}")
            
            # 4. Get Yahoo Finance market movers
            logger.info("🔥 Fetching Yahoo Finance data...")
            yahoo_data = await self.yahoo.get_all_movers()
            results['stats']['yahoo_trending'] = len(yahoo_data['trending'])
            results['stats']['yahoo_gainers'] = len(yahoo_data['gainers'])
            results['stats']['yahoo_losers'] = len(yahoo_data['losers'])
            results['stats']['yahoo_most_active'] = len(yahoo_data['most_active'])
            
            logger.info(f"   Yahoo Trending: {len(yahoo_data['trending'])}")
            logger.info(f"   Yahoo Gainers: {len(yahoo_data['gainers'])}")
            logger.info(f"   Yahoo Losers: {len(yahoo_data['losers'])}")
            logger.info(f"   Yahoo Most Active: {len(yahoo_data['most_active'])}")
            
            if yahoo_data['trending']:
                logger.info(f"   Sample trending: {yahoo_data['trending'][:5]}")
            
            # 5. Filter snapshots by criteria
            logger.info("🔬 Filtering candidates...")
            candidates = set()
            
            # Add IPOs to candidates (they often have explosive momentum)
            candidates.update(recent_ipos)
            
            # Add Yahoo Finance candidates
            candidates.update(yahoo_data['trending'])
            candidates.update(yahoo_data['gainers'])
            candidates.update(yahoo_data['losers'])
            candidates.update(yahoo_data['most_active'])
            
            for snapshot in all_snapshots:
                symbol = snapshot.ticker
                
                # Basic symbol validation
                if not self._is_valid_symbol(symbol):
                    continue
                
                # Get price and volume data
                current_price = self._get_current_price(snapshot)
                daily_volume = self._get_daily_volume(snapshot)
                
                if not current_price or not daily_volume:
                    continue
                
                # Apply price filter
                if not self._meets_price_criteria(current_price):
                    results['stats']['price_filtered'] += 1
                    continue
                
                # Apply volume filter
                if not self._meets_volume_criteria(daily_volume):
                    results['stats']['volume_filtered'] += 1
                    continue
                
                # Check for price gaps
                if self._has_significant_gap(snapshot):
                    candidates.add(symbol)
                    results['stats']['gap_candidates'] += 1
                
                # Check for volume spikes
                if self._has_volume_spike(snapshot):
                    candidates.add(symbol)
                    results['stats']['volume_spike_candidates'] += 1
                
                # Include market movers that meet criteria
                if symbol in market_movers:
                    candidates.add(symbol)
            
            # Convert to list and get detailed info
            results['candidates'] = await self._get_candidate_details(list(candidates))
            
            # Log summary
            self._log_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding candidates: {e}")
            return results
        
        finally:
            await self.polygon.disconnect()
            await self.yahoo.disconnect()
    
    def _is_valid_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid for trading"""
        if not symbol:
            return False
        
        # Basic format checks
        if not (1 <= len(symbol) <= 5):
            return False
        
        # Must be alphabetic and uppercase
        if not (symbol.isalpha() and symbol.isupper()):
            return False
        
        # Filter out known problematic symbols
        invalid_symbols = {
            'TEST', 'DEMO', 'TEMP', 'NULL', 'NONE', 'N/A', 'NA',
            'USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF',  # Currencies
            'CASH', 'BOND', 'FUND', 'ETF',  # Generic terms
            'INDEX', 'FOREX', 'CRYPTO', 'OPTION', 'FUTURE'  # Other asset types
        }
        
        if symbol in invalid_symbols:
            return False
        
        # Filter out symbols with suspicious patterns
        if symbol.startswith('X') and len(symbol) >= 4:  # Often delisted
            return False
        
        if any(char.isdigit() for char in symbol):  # No numbers in symbols
            return False
        
        return True
    
    def _is_market_hours(self) -> bool:
        """Check if it's currently market hours (9:30 AM - 4:00 PM EST)"""
        from datetime import datetime, timezone, timedelta
        
        # Get current time in EST
        est = timezone(timedelta(hours=-5))  # EST is UTC-5
        now_est = datetime.now(est)
        
        # Check if it's a weekday
        if now_est.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if it's during market hours (9:30 AM - 4:00 PM EST)
        market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now_est <= market_close
    
    def _get_current_price(self, snapshot) -> float:
        """Extract current price from snapshot"""
        try:
            if snapshot.lastTrade and snapshot.lastTrade.p:
                return float(snapshot.lastTrade.p)
            elif snapshot.day and hasattr(snapshot.day, 'close'):
                return float(snapshot.day.close)
            return 0.0
        except (AttributeError, ValueError):
            return 0.0
    
    def _get_daily_volume(self, snapshot) -> int:
        """Extract daily volume from snapshot"""
        try:
            if snapshot.day and hasattr(snapshot.day, 'volume'):
                return int(snapshot.day.volume)
            return 0
        except (AttributeError, ValueError):
            return 0
    
    def _meets_price_criteria(self, price: float) -> bool:
        """Check if price meets our criteria"""
        return self.criteria['min_price'] <= price <= self.criteria['max_price']
    
    def _meets_volume_criteria(self, volume: int) -> bool:
        """Check if volume meets our criteria"""
        return volume >= self.criteria['min_daily_volume']
    
    def _has_significant_gap(self, snapshot) -> bool:
        """Check for significant price gap"""
        try:
            if not (snapshot.prevDay and snapshot.lastTrade):
                return False
            
            prev_close = float(snapshot.prevDay.c)
            current_price = float(snapshot.lastTrade.p)
            
            if prev_close <= 0:
                return False
            
            gap_pct = abs(current_price - prev_close) / prev_close
            return gap_pct >= self.criteria['price_move_threshold']
            
        except (AttributeError, ValueError, ZeroDivisionError):
            return False
    
    def _has_volume_spike(self, snapshot) -> bool:
        """Check for volume spike"""
        try:
            if not (snapshot.day and snapshot.prevDay):
                return False
            
            current_volume = int(snapshot.day.volume)
            prev_volume = int(snapshot.prevDay.v)
            
            if prev_volume <= 0:
                return False
            
            volume_ratio = current_volume / prev_volume
            return volume_ratio >= self.criteria['volume_spike_threshold']
            
        except (AttributeError, ValueError, ZeroDivisionError):
            return False
    
    async def _get_candidate_details(self, candidates: List[str]) -> List[Dict]:
        """Get detailed information for candidates"""
        detailed_candidates = []
        
        # Filter and validate symbols first
        valid_candidates = []
        for symbol in candidates:
            if self._is_valid_symbol(symbol):
                valid_candidates.append(symbol)
            else:
                logger.debug(f"Skipping invalid symbol: {symbol}")
        
        logger.info(f"📋 Getting details for {len(valid_candidates)} valid candidates...")
        logger.info(f"   First 10 candidates: {valid_candidates[:10]}")
        
        successful_lookups = 0
        failed_lookups = 0
        not_found_count = 0
        
        for i, symbol in enumerate(valid_candidates[:50]):  # Limit to prevent API overload
            try:
                if i < 5:  # Log details for first 5 symbols
                    logger.info(f"Processing symbol {i+1}/50: {symbol}")
                
                snapshot = await self.polygon.get_snapshot_ticker(symbol)
                
                if snapshot:
                    current_price = self._get_current_price(snapshot)
                    daily_volume = self._get_daily_volume(snapshot)
                    
                    if i < 5:
                        logger.info(f"   {symbol}: price={current_price}, volume={daily_volume}")
                        logger.info(f"   Has prevDay: {snapshot.prevDay is not None}")
                        if snapshot.prevDay:
                            logger.info(f"   PrevDay close: {getattr(snapshot.prevDay, 'c', 'None')}")
                    
                    # During market hours: require valid price and volume
                    # During market closure: accept candidates even with 0 current data
                    is_market_hours = self._is_market_hours()
                    
                    if is_market_hours:
                        # Strict validation during market hours
                        should_include = current_price > 0 and daily_volume > 0
                    else:
                        # Relaxed validation during market closure - accept if symbol exists in Polygon
                        should_include = True
                        # Use previous day data if available
                        if snapshot.prevDay and hasattr(snapshot.prevDay, 'c') and snapshot.prevDay.c > 0:
                            current_price = float(snapshot.prevDay.c)  # Use previous close
                        if snapshot.prevDay and hasattr(snapshot.prevDay, 'v') and snapshot.prevDay.v > 0:
                            daily_volume = int(snapshot.prevDay.v)  # Use previous volume
                    
                    if should_include:
                        # Apply price filtering even during market closure
                        if not self._meets_price_criteria(current_price):
                            if i < 5:
                                logger.info(f"   {symbol}: Price ${current_price:.2f} outside range ${self.criteria['min_price']}-${self.criteria['max_price']}")
                            failed_lookups += 1
                            continue
                        
                        # Apply volume filtering
                        if not self._meets_volume_criteria(daily_volume):
                            if i < 5:
                                logger.info(f"   {symbol}: Volume {daily_volume:,} below minimum {self.criteria['min_daily_volume']:,}")
                            failed_lookups += 1
                            continue
                        
                        # Calculate gap properly based on market hours
                        gap_pct = 0.0
                        if is_market_hours:
                            # During market hours: use current vs previous close
                            if snapshot.prevDay and snapshot.lastTrade:
                                try:
                                    prev_close = float(snapshot.prevDay.c)
                                    current_price_for_gap = float(snapshot.lastTrade.p)
                                    if prev_close > 0 and current_price_for_gap > 0:
                                        gap_pct = (current_price_for_gap - prev_close) / prev_close * 100
                                except:
                                    pass
                        else:
                            # During market closure: no gap calculation (use 0%)
                            gap_pct = 0.0
                        
                        # Calculate volume ratio properly based on market hours
                        volume_ratio = 1.0
                        if is_market_hours:
                            # During market hours: current vs previous volume
                            if snapshot.day and snapshot.prevDay:
                                try:
                                    current_vol = int(snapshot.day.volume)
                                    prev_vol = int(snapshot.prevDay.v)
                                    if prev_vol > 0 and current_vol > 0:
                                        volume_ratio = current_vol / prev_vol
                                except:
                                    pass
                        else:
                            # During market closure: use previous day's volume activity as baseline
                            volume_ratio = 1.0
                        
                        detailed_candidates.append({
                            'symbol': symbol,
                            'price': current_price,
                            'volume': daily_volume,
                            'gap_percent': gap_pct,
                            'volume_ratio': volume_ratio,
                            'meets_gap_criteria': abs(gap_pct) >= self.criteria['price_move_threshold'] * 100,
                            'meets_volume_spike': volume_ratio >= self.criteria['volume_spike_threshold']
                        })
                        successful_lookups += 1
                    else:
                        logger.debug(f"No valid data for {symbol}: price={current_price}, volume={daily_volume}")
                        failed_lookups += 1
                else:
                    if i < 5:
                        logger.info(f"   {symbol}: No snapshot data returned (not found in Polygon)")
                    not_found_count += 1
                    failed_lookups += 1
                
                # Rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                # Handle 404 and other errors gracefully
                if "404" in str(e) or "NotFound" in str(e):
                    logger.debug(f"Symbol {symbol} not found in Polygon (404) - likely delisted/invalid")
                else:
                    logger.debug(f"Failed to get details for {symbol}: {e}")
                failed_lookups += 1
                continue
        
        logger.info(f"   ✅ Successfully retrieved: {successful_lookups}")
        logger.info(f"   ❌ Failed/Invalid: {failed_lookups}")
        logger.info(f"   🔍 Not found in Polygon: {not_found_count}")
        
        # Sort by volume (descending)
        detailed_candidates.sort(key=lambda x: x['volume'], reverse=True)
        
        return detailed_candidates
    
    def _log_summary(self, results: Dict):
        """Log search summary"""
        stats = results['stats']
        candidates = results['candidates']
        
        logger.info("\n" + "="*60)
        logger.info("📊 CANDIDATE SEARCH SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Snapshots Processed: {stats['total_processed']:,}")
        logger.info(f"Market Movers Found: {stats['market_movers']:,}")
        logger.info(f"Recent IPOs: {stats['recent_ipos']:,}")
        logger.info(f"Yahoo Trending: {stats['yahoo_trending']:,}")
        logger.info(f"Yahoo Gainers: {stats['yahoo_gainers']:,}")
        logger.info(f"Yahoo Losers: {stats['yahoo_losers']:,}")
        logger.info(f"Yahoo Most Active: {stats['yahoo_most_active']:,}")
        logger.info(f"Price Filtered Out: {stats['price_filtered']:,}")
        logger.info(f"Volume Filtered Out: {stats['volume_filtered']:,}")
        logger.info(f"Gap Candidates: {stats['gap_candidates']:,}")
        logger.info(f"Volume Spike Candidates: {stats['volume_spike_candidates']:,}")
        logger.info(f"Final Candidates: {len(candidates):,}")
        logger.info("="*60)
        
        if candidates:
            logger.info("\n🎯 TOP 10 CANDIDATES:")
            logger.info("-" * 80)
            logger.info(f"{'Symbol':<8} {'Price':<10} {'Volume':<12} {'Gap%':<8} {'Vol Ratio':<10} {'Criteria'}")
            logger.info("-" * 80)
            
            for candidate in candidates[:10]:
                criteria_met = []
                if candidate['meets_gap_criteria']:
                    criteria_met.append('GAP')
                if candidate['meets_volume_spike']:
                    criteria_met.append('VOL')
                
                logger.info(
                    f"{candidate['symbol']:<8} "
                    f"${candidate['price']:<9.2f} "
                    f"{candidate['volume']:<12,} "
                    f"{candidate['gap_percent']:<7.1f}% "
                    f"{candidate['volume_ratio']:<9.1f}x "
                    f"{', '.join(criteria_met)}"
                )


async def main():
    """Main test function"""
    if not POLYGON_API_KEY:
        logger.error("❌ POLYGON_API_KEY not found in environment variables")
        return
    
    logger.info("🚀 Starting Stock Candidate Finder Test")
    logger.info(f"⏰ Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    finder = CandidateFinder(POLYGON_API_KEY)
    results = await finder.find_candidates()
    
    # Save results to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"candidate_results_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"Stock Candidate Search Results\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Candidates Found: {len(results['candidates'])}\n\n")
        
        f.write("Criteria Used:\n")
        f.write(f"- Price Range: ${UNIVERSE_CONFIG['min_price']} - ${UNIVERSE_CONFIG['max_price']}\n")
        f.write(f"- Min Volume: {UNIVERSE_CONFIG['min_daily_volume']:,} shares\n")
        f.write(f"- Gap Threshold: {UNIVERSE_CONFIG['price_move_threshold']*100}%\n")
        f.write(f"- Volume Spike: {UNIVERSE_CONFIG['volume_spike_threshold']}x\n\n")
        
        f.write("Candidates:\n")
        if results['candidates']:
            for candidate in results['candidates']:
                criteria_met = []
                if candidate['meets_gap_criteria']:
                    criteria_met.append('GAP')
                if candidate['meets_volume_spike']:
                    criteria_met.append('VOL')
                
                f.write(f"{candidate['symbol']}: ${candidate['price']:.2f}, "
                       f"Vol: {candidate['volume']:,}, Gap: {candidate['gap_percent']:.1f}%, "
                       f"VolRatio: {candidate['volume_ratio']:.1f}x, "
                       f"Criteria: {', '.join(criteria_met) if criteria_met else 'None'}\n")
        else:
            f.write("No candidates found matching criteria.\n")
    
    logger.info(f"💾 Results saved to {filename}")
    logger.info("✅ Test completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())