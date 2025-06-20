"""
Main Momentum Trading System
Pure momentum detection with 5% trailing stops
Optimized for CRCL-type explosive moves
"""

import asyncio
import time
import signal as sys_signal
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import numpy as np

from trading_system.momentum_detector import LiveMomentumDetector
from trading_system.position_manager import PositionManager
from trading_system.data_manager import PolygonStream
from trading_system.polygon_client import PolygonClient
from trading_system.alpaca_client import AlpacaClient
from trading_system.yahoo_client import YahooClient
from trading_system import config

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOGGING_CONFIG['level']),
    format=config.LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(config.LOGGING_CONFIG['file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MomentumTradingSystem:
    def __init__(self):
        # Initialize API clients
        self.polygon = PolygonClient(config.POLYGON_API_KEY)
        self.alpaca = AlpacaClient(config.ALPACA_KEY, config.ALPACA_SECRET, config.ALPACA_BASE_URL)
        self.yahoo = YahooClient()
        
        # Core components
        self.momentum_detector = LiveMomentumDetector(self.polygon)
        self.position_manager = PositionManager(self.alpaca)
        self.data_stream = PolygonStream(config.POLYGON_API_KEY)
        
        # System state
        self.universe = []
        self.last_universe_update = datetime.now()
        self.active_positions = {}
        self.running = False
        
        # Duplicate order prevention
        self.recent_orders = {}  # symbol -> timestamp of last order
        self.order_cooldown = 300  # 5 minutes cooldown between orders for same symbol
        
    async def initialize(self):
        """Initialize the trading system"""
        logger.info("🚀 Initializing Momentum Trading System...")
        
        # Train momentum model if needed
        if not self.momentum_detector.is_trained:
            logger.info("📚 Training momentum detection model...")
            await self.momentum_detector.train_model_from_historical_movers()
        
        # Build initial universe
        await self._update_universe()
        
        # Connect to APIs
        await self.data_stream.connect()
        await self.yahoo.connect()
        
        if self.universe:
            await self.data_stream.subscribe(self.universe[:1000])  # Top 1000 symbols
            logger.info(f"Subscribed to {min(len(self.universe), 1000)} symbols")
        
        # Get current positions
        self.active_positions = await self.position_manager.get_positions()
        
        # Log startup summary
        await self._log_startup_summary()
        
        logger.info("Momentum Trading System ready!")
    
    async def _log_startup_summary(self):
        """Log system startup information"""
        try:
            account = await self.alpaca.get_account()
            account_value = float(account['portfolio_value'])
            
            logger.info(f"Account Value: ${account_value:,.2f}")
            logger.info(f"Universe Size: {len(self.universe):,} momentum candidates")
            logger.info(f"Active Positions: {len(self.active_positions)}")
            logger.info(f"Entry Threshold: {config.MOMENTUM_CONFIG['entry_threshold']}, Trailing Stop: {config.RISK_PARAMS['trailing_stop_pct']*100}%")
        except Exception as e:
            logger.error(f"Failed to log startup summary: {e}")
    
    async def run(self):
        """Main trading loop"""
        logger.info("Starting momentum trading loop...")
        self.running = True
        
        # Start concurrent tasks
        tasks = [
            self._momentum_scanning_loop(),
            self._position_monitoring_loop(),
            self._universe_update_loop()
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
        finally:
            self.running = False
    
    async def _momentum_scanning_loop(self):
        """Scan universe for momentum opportunities"""
        while self.running:
            try:
                scan_start = time.time()
                
                # Get current price data from stream
                momentum_signals = []
                
                # Process universe in batches
                batch_size = 100
                for i in range(0, len(self.universe), batch_size):
                    batch = self.universe[i:i + batch_size]
                    
                    # Process batch in parallel
                    batch_tasks = []
                    for symbol in batch:
                        if symbol not in self.active_positions:
                            batch_tasks.append(self._check_momentum(symbol))
                    
                    if batch_tasks:
                        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                        
                        for result in batch_results:
                            if isinstance(result, dict) and result.get('momentum_score', 0) >= config.MOMENTUM_CONFIG['entry_threshold']:
                                momentum_signals.append(result)
                
                # Sort by momentum score to get the absolute best candidate
                momentum_signals.sort(key=lambda x: x['momentum_score'], reverse=True)
                
                # ENHANCED FILTERING: Only execute truly high-quality trades
                if momentum_signals:
                    quality_candidates = []
                    
                    # Apply multiple quality filters
                    for signal in momentum_signals:
                        if self._passes_quality_filters(signal):
                            quality_candidates.append(signal)
                    
                    if quality_candidates:
                        # Execute ONLY the single best quality candidate
                        best_signal = quality_candidates[0]
                        
                        logger.info(f"🎯 High-quality candidate: {best_signal['symbol']} "
                                   f"(score: {best_signal['momentum_score']:.0f}, "
                                   f"grade: {best_signal.get('quality_analysis', {}).get('overall_grade', 'N/A')})")
                        await self._execute_entry(best_signal)
                    else:
                        # Log why no trades were taken
                        best_signal = momentum_signals[0]
                        quality = best_signal.get('quality_analysis', {})
                        logger.info(f"❌ No quality candidates found. Best: {best_signal['symbol']} "
                                   f"(score: {best_signal['momentum_score']:.0f}, "
                                   f"grade: {quality.get('overall_grade', 'N/A')}, "
                                   f"issues: {quality.get('penalties', [])})")
                
                # Log scan performance
                scan_time = time.time() - scan_start
                if momentum_signals:
                    logger.info(f"📈 Scan: {scan_time:.1f}s | Signals: {len(momentum_signals)} | "
                               f"Top: {momentum_signals[0]['symbol']} ({momentum_signals[0]['momentum_score']:.0f})")
                
                # Wait before next scan
                await asyncio.sleep(max(1, config.REALTIME_CONFIG['scan_frequency_seconds'] - scan_time))
                
            except Exception as e:
                logger.error(f"Momentum scanning error: {e}")
                await asyncio.sleep(30)
    
    async def _check_momentum(self, symbol: str) -> Dict:
        """Check momentum for a single symbol"""
        try:
            # Get current price first for early validation
            current_price = await self._get_current_price(symbol)
            if not current_price:
                return {'symbol': symbol, 'momentum_score': 0}
            
            # Early price validation - filter out penny stocks before momentum detection
            if not self._meets_price_requirements(current_price):
                logger.debug(f"Skipping momentum check for {symbol} - price ${current_price:.2f} outside range ${config.UNIVERSE_CONFIG['min_price']}-${config.UNIVERSE_CONFIG['max_price']}")
                return {'symbol': symbol, 'momentum_score': 0}
            
            # Detect momentum using live features
            result = await self.momentum_detector.detect_momentum(symbol)
            return result
            
        except Exception as e:
            logger.debug(f"Momentum check failed for {symbol}: {e}")
            return {'symbol': symbol, 'momentum_score': 0}
    
    async def _execute_entry(self, signal: Dict) -> bool:
        """Execute momentum entry"""
        try:
            symbol = signal['symbol']
            momentum_score = signal['momentum_score']
            
            # Check for duplicate order prevention
            if await self._has_recent_order(symbol):
                logger.debug(f"Skipping {symbol} - recent order within cooldown period")
                return False
            
            # Check for existing pending orders
            if await self._has_pending_orders(symbol):
                logger.debug(f"Skipping {symbol} - already has pending orders")
                return False
            
            # Get current price
            current_price = await self._get_current_price(symbol)
            if not current_price:
                return False
            
            # Validate price meets minimum threshold
            if not self._meets_price_requirements(current_price):
                logger.debug(f"Skipping {symbol} - price ${current_price:.2f} below minimum ${config.UNIVERSE_CONFIG['min_price']}")
                return False
            
            # Check if we can afford position
            account_value = await self.alpaca.get_account_value()
            
            # Check risk limits
            if len(self.active_positions) >= config.RISK_PARAMS['max_positions']:
                logger.debug(f"Max positions reached ({config.RISK_PARAMS['max_positions']})")
                return False
            
            # Get quality grade for position sizing
            quality_analysis = signal.get('quality_analysis', {})
            quality_grade = quality_analysis.get('overall_grade', 'C')
            
            # Execute entry with quality-based sizing
            success = await self.position_manager.enter_position(
                symbol, momentum_score, current_price, account_value, quality_grade
            )
            
            if success:
                # Record the order timestamp for duplicate prevention
                self.recent_orders[symbol] = datetime.now()
                
                self.active_positions[symbol] = {
                    'entry_time': datetime.now(),
                    'entry_price': current_price,
                    'momentum_score': momentum_score
                }
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Entry execution failed for {signal.get('symbol', 'unknown')}: {e}")
            return False
    
    async def _position_monitoring_loop(self):
        """Monitor positions with periodic sync"""
        sync_counter = 0
        
        while self.running:
            try:
                sync_counter += 1
                
                # Sync with Alpaca every 10 iterations (100 seconds)
                if sync_counter % 10 == 0:
                    sync_report = await self.position_manager.sync_positions_with_alpaca()
                    
                    # Validate trailing stops every 30 iterations (300 seconds)
                    if sync_counter % 30 == 0:
                        stop_validation = await self.position_manager.validate_trailing_stops()
                        invalid_stops = [s for s, valid in stop_validation.items() if not valid]
                        if invalid_stops:
                            logger.warning(f"Invalid trailing stops detected: {invalid_stops}")
                
                if not self.active_positions:
                    await asyncio.sleep(config.REALTIME_CONFIG['update_frequency_seconds'])
                    continue
                
                # Get current prices for all positions
                current_prices = {}
                for symbol in self.active_positions.keys():
                    price = await self._get_current_price(symbol)
                    if price:
                        current_prices[symbol] = price
                
                # Check for exits
                exit_signals = await self.position_manager.check_all_positions(current_prices)
                
                # Update active positions
                for symbol in exit_signals:
                    if symbol in self.active_positions:
                        del self.active_positions[symbol]
                
                # Update position tracking
                self.active_positions = await self.position_manager.get_positions()
                
                await asyncio.sleep(config.REALTIME_CONFIG['update_frequency_seconds'])
                
            except Exception as e:
                logger.error(f"Position monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _universe_update_loop(self):
        """Update trading universe periodically"""
        while self.running:
            try:
                time_since_update = (datetime.now() - self.last_universe_update).total_seconds()
                
                if time_since_update >= config.UNIVERSE_CONFIG['refresh_interval_seconds']:
                    await self._update_universe()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Universe update error: {e}")
                await asyncio.sleep(300)
    
    async def _update_universe(self):
        """Update the trading universe with momentum candidates"""
        try:
            logger.info("🔄 Updating momentum universe...")
            
            candidates = set()
            
            # 1. Current market movers (Polygon) - ONLY GAINERS for long positions
            try:
                gainers = await self.polygon.get_market_movers('gainers')
                candidates.update(gainers[:150])  # Top 150 gainers (increased from 100)
                logger.info(f"Polygon movers: {len(gainers)} gainers added")
            except Exception as e:
                logger.warning(f"Failed to get Polygon market movers: {e}")
            
            # 2. Recent IPOs (explosive potential)
            try:
                recent_ipos = await self.polygon.get_recent_ipos()
                candidates.update(recent_ipos)
                logger.info(f"Recent IPOs: {len(recent_ipos)} found")
            except Exception as e:
                logger.warning(f"Failed to get recent IPOs: {e}")
            
            # 3. Yahoo Finance market movers - ONLY GAINERS and positive momentum
            try:
                yahoo_data = await self.yahoo.get_all_movers()
                candidates.update(yahoo_data['trending'])
                candidates.update(yahoo_data['gainers'])
                candidates.update(yahoo_data['most_active'])  # Keep most active for momentum opportunities
                logger.info(f"Yahoo data: {len(yahoo_data['trending'])} trending, "
                           f"{len(yahoo_data['gainers'])} gainers, "
                           f"{len(yahoo_data['most_active'])} most active")
            except Exception as e:
                logger.warning(f"Failed to get Yahoo Finance data: {e}")
            
            # 4. Volume spike detection
            try:
                all_snapshots = await self.polygon.get_snapshot_all_tickers()
                for snapshot in all_snapshots[:500]:  # Limit processing
                    if hasattr(snapshot, 'day') and snapshot.day:
                        if snapshot.day.volume > config.UNIVERSE_CONFIG['min_daily_volume']:
                            candidates.add(snapshot.ticker)
            except Exception as e:
                logger.warning(f"Failed to get volume spikes: {e}")
            
            # 5. Price gap detection (momentum often starts with gaps)
            try:
                for snapshot in all_snapshots[:300]:  # Check subset for gaps
                    if hasattr(snapshot, 'prevDay') and hasattr(snapshot, 'lastTrade') and snapshot.prevDay and snapshot.lastTrade:
                        prev_close = snapshot.prevDay.c
                        current_price = snapshot.lastTrade.p
                        if prev_close > 0:
                            gap_pct = abs(current_price - prev_close) / prev_close
                            if gap_pct >= config.UNIVERSE_CONFIG['price_move_threshold']:
                                candidates.add(snapshot.ticker)
            except Exception as e:
                logger.warning(f"Failed to detect price gaps: {e}")
            
            # Filter and limit universe with improved validation
            filtered_universe = []
            for symbol in candidates:
                if self._is_valid_symbol(symbol):
                    filtered_universe.append(symbol)
            
            # Limit size
            if len(filtered_universe) > config.UNIVERSE_CONFIG['max_universe_size']:
                filtered_universe = filtered_universe[:config.UNIVERSE_CONFIG['max_universe_size']]
            
            # Update universe
            old_size = len(self.universe)
            self.universe = filtered_universe
            self.last_universe_update = datetime.now()
            
            # Update data stream subscriptions
            if self.universe:
                await self.data_stream.update_subscriptions(self.universe[:1000])
            
            logger.info(f"📊 Universe updated: {old_size} → {len(self.universe)} symbols")
            
        except Exception as e:
            logger.error(f"Universe update failed: {e}")
    
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
        
        # Filter out warrants and other derivative securities
        # Warrants end with W, WS, WT, WR
        if symbol.endswith(('W', 'WS', 'WT', 'WR')):
            return False
            
        # Filter out rights (usually end with R)
        if len(symbol) >= 4 and symbol.endswith('R'):
            return False
            
        # Filter out units (usually end with U)
        if len(symbol) >= 4 and symbol.endswith('U'):
            return False
        
        # Filter out preferred shares (usually end with P, PR)
        if symbol.endswith(('P', 'PR')):
            return False
        
        # Filter out symbols with problematic patterns
        # Skip obvious test/demo symbols
        if any(keyword in symbol.upper() for keyword in ['TEST', 'DEMO', 'TEMP']):
            return False
        
        # Filter out symbols with suspicious patterns
        if symbol.startswith('X') and len(symbol) >= 4:  # Often delisted
            return False
        
        if any(char.isdigit() for char in symbol):  # No numbers in symbols
            return False
        
        return True
    
    def _passes_quality_filters(self, signal: Dict) -> bool:
        """Enhanced quality filters to ensure only the best trades"""
        try:
            symbol = signal['symbol']
            momentum_score = signal.get('momentum_score', 0)
            quality_analysis = signal.get('quality_analysis', {})
            
            # 1. MINIMUM MOMENTUM SCORE (raised threshold)
            if momentum_score < config.MOMENTUM_CONFIG['entry_threshold']:
                logger.debug(f"{symbol}: Below minimum momentum threshold ({momentum_score:.0f} < {config.MOMENTUM_CONFIG['entry_threshold']})")
                return False
            
            # 2. QUALITY GRADE FILTER
            overall_grade = quality_analysis.get('overall_grade', 'F')
            if overall_grade in ['D', 'F']:
                logger.debug(f"{symbol}: Poor quality grade ({overall_grade})")
                return False
            
            # 3. EXCESSIVE PENALTIES FILTER
            penalties = quality_analysis.get('penalties', [])
            if len(penalties) > 2:
                logger.debug(f"{symbol}: Too many quality issues ({len(penalties)} penalties)")
                return False
            
            # 4. CRITICAL PENALTY FILTERS
            critical_penalties = [
                'Extreme volume spike',
                'Choppy price action',
                'Low liquidity',
                'Analysis failed'
            ]
            
            for penalty in penalties:
                if any(critical in penalty for critical in critical_penalties):
                    logger.debug(f"{symbol}: Critical quality issue - {penalty}")
                    return False
            
            # 5. HIGH-QUALITY MOMENTUM REQUIREMENT
            # For scores below quality threshold, require Grade A
            if momentum_score < config.MOMENTUM_CONFIG.get('quality_threshold', 50):
                if overall_grade != 'A':
                    logger.debug(f"{symbol}: Moderate momentum requires Grade A (got {overall_grade})")
                    return False
            
            # 6. VOLUME CONSISTENCY CHECK
            volume_consistency = quality_analysis.get('volume_consistency', 0)
            if volume_consistency < config.MOMENTUM_CONFIG.get('min_volume_consistency', 0.7):
                logger.debug(f"{symbol}: Poor volume consistency ({volume_consistency:.2f})")
                return False
            
            # 7. TREND STRENGTH CHECK
            trend_strength = quality_analysis.get('trend_strength', 0)
            if trend_strength < config.MOMENTUM_CONFIG.get('min_trend_strength', 0.6):
                logger.debug(f"{symbol}: Weak trend strength ({trend_strength:.2f})")
                return False
            
            # 8. ENSURE POSITIVE BONUSES FOR HIGH-QUALITY TRADES
            bonuses = quality_analysis.get('bonuses', [])
            if momentum_score < 60 and len(bonuses) == 0:
                logger.debug(f"{symbol}: Moderate momentum with no quality bonuses")
                return False
            
            logger.debug(f"{symbol}: ✅ Passed all quality filters (score: {momentum_score:.0f}, grade: {overall_grade})")
            return True
            
        except Exception as e:
            logger.error(f"Quality filter error for {signal.get('symbol', 'unknown')}: {e}")
            return False
    
    def _meets_price_requirements(self, price: float) -> bool:
        """Check if price meets minimum trading requirements"""
        return (config.UNIVERSE_CONFIG['min_price'] <= price <= config.UNIVERSE_CONFIG['max_price'])
    
    async def _has_recent_order(self, symbol: str) -> bool:
        """Check if symbol has recent order within cooldown period"""
        if symbol not in self.recent_orders:
            return False
        
        last_order_time = self.recent_orders[symbol]
        time_since_order = (datetime.now() - last_order_time).total_seconds()
        
        if time_since_order >= self.order_cooldown:
            # Cooldown expired, remove from tracking
            del self.recent_orders[symbol]
            return False
        
        return True
    
    async def _has_pending_orders(self, symbol: str) -> bool:
        """Check if symbol already has pending orders in Alpaca"""
        try:
            # Get all open orders for this symbol
            open_orders = await self.alpaca.list_orders(status='open')
            
            for order in open_orders:
                if order.get('symbol') == symbol:
                    logger.debug(f"Found pending order for {symbol}: {order.get('status')}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check pending orders for {symbol}: {e}")
            return False
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol using Alpaca API"""
        try:
            # Use Alpaca's data client for current price
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            
            data_client = StockHistoricalDataClient(
                api_key=config.ALPACA_KEY,
                secret_key=config.ALPACA_SECRET
            )
            
            quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            latest_quote = data_client.get_stock_latest_quote(quote_request)
            
            if symbol in latest_quote:
                quote = latest_quote[symbol]
                # Use mid-price (average of bid/ask) for more accurate pricing
                if quote.ask_price and quote.bid_price:
                    return (float(quote.ask_price) + float(quote.bid_price)) / 2
                elif quote.ask_price:
                    return float(quote.ask_price)
                elif quote.bid_price:
                    return float(quote.bid_price)
            
            # Fallback to Polygon if Alpaca doesn't have data
            snapshot = await self.polygon.get_snapshot_ticker(symbol)
            if snapshot and snapshot.lastTrade and snapshot.lastTrade.p > 0:
                return snapshot.lastTrade.p
            elif snapshot and snapshot.prevDay and snapshot.prevDay.c > 0:
                return snapshot.prevDay.c  # Use previous close during market closure
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to get price for {symbol}: {e}")
            return None
    
    async def shutdown(self):
        """Gracefully shutdown the system"""
        logger.info("Shutting down momentum trading system...")
        
        self.running = False
        
        try:
            # Get final performance summary
            positions_summary = self.position_manager.get_position_summary()
            logger.info(f"Final summary: {positions_summary}")
            
            # Disconnect from data stream
            await self.data_stream.disconnect()
            
            # Close API connections
            await self.polygon.disconnect()
            await self.yahoo.disconnect()
            
            logger.info("Shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")


async def main():
    """Main entry point"""
    system = MomentumTradingSystem()
    
    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    
    def shutdown_handler(sig):
        logger.info(f"Received signal {sig}, shutting down...")
        asyncio.create_task(system.shutdown())
        loop.stop()
    
    for sig in (sys_signal.SIGTERM, sys_signal.SIGINT):
        sys_signal.signal(sig, lambda s, f: shutdown_handler(s))
    
    try:
        # Initialize and run
        await system.initialize()
        await system.run()
        
    except KeyboardInterrupt:
        logger.info("⌨️ Keyboard interrupt received")
        await system.shutdown()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        await system.shutdown()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)