"""
Event-Driven Momentum Trading System
Pure event-driven architecture: WebSocket → Filter → XGBoost → Execute
No scheduled loops - everything triggered by real-time events
"""

import asyncio
import time
import signal as sys_signal
from datetime import datetime, timedelta, time as dt_time
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


class EventDrivenMomentumSystem:
    def __init__(self):
        # Initialize API clients
        self.polygon = PolygonClient(config.POLYGON_API_KEY)
        self.yahoo = YahooClient()
        
        # Initialize data stream first
        self.data_stream = PolygonStream(config.POLYGON_API_KEY)
        
        # Initialize Alpaca client with data manager for real-time quotes
        self.alpaca = AlpacaClient(config.ALPACA_KEY, config.ALPACA_SECRET, config.ALPACA_BASE_URL, self.polygon, self.data_stream)
        
        # Core components
        self.momentum_detector = LiveMomentumDetector(self.polygon)
        self.position_manager = PositionManager(self.alpaca)
        
        # System state
        self.active_positions = {}
        self.running = False
        
        # Duplicate order prevention
        self.recent_orders = {}  # symbol -> timestamp of last order
        self.order_cooldown = 60  # 1 minute cooldown (reduced for speed)
        
        # Performance tracking
        self.signals_processed = 0
        self.trades_executed = 0
        self.last_trade_time = None
        
        # Historical data cache for XGBoost
        self.historical_data_cache = {}
        
    async def initialize(self):
        """Initialize the event-driven trading system"""
        logger.info("🚀 Initializing Event-Driven Momentum Trading System...")
        
        # Train model if not already trained
        if not self.momentum_detector.is_trained:
            logger.info("📚 Training momentum model on startup...")
            await self.momentum_detector.train_model_from_historical_movers()
        
        # Pre-load historical data cache for faster XGBoost analysis
        await self._preload_historical_data()
        
        # Connect to data stream
        await self.data_stream.connect()
        
        # Set up event-driven momentum callback
        self.data_stream.add_momentum_callback(self._handle_instant_momentum_signal)
        
        # Subscribe to ALL market trades
        await self.data_stream.subscribe()
        
        # Get current positions
        self.active_positions = await self.position_manager.get_positions()
        
        # Log startup summary
        await self._log_startup_summary()
        
        logger.info("✅ Event-Driven Momentum Trading System ready!")
        logger.info("🔥 Monitoring ALL market trades for instant momentum detection...")
    
    async def _preload_historical_data(self):
        """Pre-load historical data for common symbols to speed up XGBoost analysis"""
        try:
            logger.info("📊 Pre-loading historical data cache...")
            
            # Get list of active/popular symbols from multiple sources
            symbols_to_cache = set()
            
            # Get recent gainers
            try:
                gainers = await self.polygon.get_market_movers('gainers')
                symbols_to_cache.update(gainers[:100])
            except Exception as e:
                logger.warning(f"Failed to get gainers for cache: {e}")
            
            # Get Yahoo data
            try:
                yahoo_data = await self.yahoo.get_all_movers()
                symbols_to_cache.update(yahoo_data.get('gainers', [])[:50])
                symbols_to_cache.update(yahoo_data.get('most_active', [])[:50])
            except Exception as e:
                logger.warning(f"Failed to get Yahoo data for cache: {e}")
            
            # Pre-load data for top symbols
            cache_count = 0
            for symbol in list(symbols_to_cache)[:200]:  # Limit to top 200
                try:
                    # Get 3-day historical data
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=4)  # Extra day for safety
                    
                    bars = await self.polygon.list_aggs(symbol, 1, 'day', start_date, end_date, limit=5)
                    if bars and len(bars) >= 2:
                        self.historical_data_cache[symbol] = bars
                        cache_count += 1
                    
                    # Rate limiting
                    if cache_count % 10 == 0:
                        await asyncio.sleep(0.1)
                        
                except Exception as e:
                    logger.debug(f"Failed to cache data for {symbol}: {e}")
                    continue
            
            logger.info(f"📈 Cached historical data for {cache_count} symbols")
            
        except Exception as e:
            logger.error(f"Historical data cache loading failed: {e}")
    
    async def _log_startup_summary(self):
        """Log system startup information"""
        try:
            account = await self.alpaca.get_account()
            account_value = float(account['portfolio_value'])
            
            logger.info(f"💰 Account Value: ${account_value:,.2f}")
            logger.info(f"📊 Historical Cache: {len(self.historical_data_cache)} symbols")
            logger.info(f"📍 Active Positions: {len(self.active_positions)}")
            logger.info(f"⚡ Entry Threshold: {config.MOMENTUM_CONFIG['entry_threshold']}")
            logger.info(f"🛡️  Stop Loss: {config.RISK_PARAMS['stop_loss_pct']}%")
            logger.info(f"💰 Take Profit: {config.RISK_PARAMS['take_profit_pct']}%")
        except Exception as e:
            logger.error(f"Failed to log startup summary: {e}")
    
    async def run(self):
        """Main event loop - purely event-driven, no scheduled tasks"""
        logger.info("🔥 Starting Event-Driven Trading System...")
        logger.info("📡 System is now LIVE and monitoring ALL market trades...")
        self.running = True
        
        # Only run position monitoring - everything else is event-driven
        try:
            await self._position_monitoring_loop()
        except Exception as e:
            logger.error(f"Event-driven system error: {e}")
        finally:
            self.running = False
    
    async def _handle_instant_momentum_signal(self, symbol: str, momentum_data: Dict):
        """Handle real-time momentum signals with instant XGBoost analysis and execution"""
        try:
            self.signals_processed += 1
            
            # Skip if already have position
            if symbol in self.active_positions:
                return
            
            # Rate limit per symbol
            if await self._has_recent_order(symbol):
                return
            
            # Get current price - always use fresh price for execution
            current_price = await self._get_current_price(symbol)
            if not current_price:
                # Fallback to momentum data price if API fails
                current_price = momentum_data.get('current_price')
                if not current_price:
                    return
            
            # Log price sources for debugging
            momentum_price = momentum_data.get('current_price', 0)
            logger.info(f"🔍 PRICE SYNC: {symbol} | Momentum: ${momentum_price:.2f} | Current: ${current_price:.2f}")
            
            # Validate price requirements
            if not self._meets_price_requirements(current_price):
                logger.debug(f"❌ {symbol}: Price ${current_price:.2f} outside range ${config.UNIVERSE_CONFIG['min_price']}-${config.UNIVERSE_CONFIG['max_price']}")
                return
            
            # Log all symbols going through XGBoost analysis
            live_score = momentum_data.get('momentum_score', 0)
            logger.info(f"🔍 ANALYZING: {symbol} | Live Score: {live_score:.0f} | Price: ${current_price:.2f} | Volume: {momentum_data.get('volume_ratio', 1):.1f}x")
            
            # INSTANT XGBOOST ANALYSIS with historical data fusion
            xgboost_result = await self._analyze_with_xgboost(symbol, momentum_data)
            if not xgboost_result:
                logger.info(f"❌ {symbol}: XGBoost analysis failed - insufficient data")
                return
            
            # Get XGBoost score and log detailed analysis
            xgb_score = xgboost_result.get('momentum_score', 0)
            raw_xgb_score = xgboost_result.get('raw_xgboost_score', xgb_score)
            method = xgboost_result.get('method', 'unknown')
            confidence = xgboost_result.get('confidence', 0)
            
            # LOG ALL XGBOOST RESULTS (not just passing ones)
            logger.info(f"🤖 XGBOOST RESULT: {symbol}")
            logger.info(f"   📊 Live Score: {live_score:.0f} → XGBoost Score: {xgb_score:.0f} (raw: {raw_xgb_score:.0f})")
            logger.info(f"   🎯 Method: {method} | Confidence: {confidence:.3f}")
            logger.info(f"   💰 Price: ${current_price:.2f} | Volume Ratio: {momentum_data.get('volume_ratio', 1):.1f}x")
            
            # Check XGBoost score threshold
            entry_threshold = config.MOMENTUM_CONFIG['entry_threshold']
            if xgb_score < entry_threshold:
                logger.debug(f"❌ {symbol}: Score {xgb_score:.0f} below threshold {entry_threshold}")
                return
            
            # Apply quality filters
            if not self._passes_quality_filters(xgboost_result):
                quality_analysis = xgboost_result.get('quality_analysis', {})
                grade = quality_analysis.get('overall_grade', 'N/A')
                penalties = quality_analysis.get('penalties', [])
                logger.debug(f"❌ {symbol}: Failed quality filters (Grade: {grade}, Issues: {penalties})")
                return
            
            # Passed all filters - ready for execution
            quality_analysis = xgboost_result.get('quality_analysis', {})
            grade = quality_analysis.get('overall_grade', 'N/A')
            peak_score = quality_analysis.get('peak_avoidance_score', 0)
            
            logger.info(f"🎯 TRADE SIGNAL APPROVED: {symbol}")
            logger.info(f"   ✅ Final Score: {xgb_score:.0f} | Quality Grade: {grade} | Peak Avoidance: {peak_score:.2f}")
            logger.info(f"   🚀 Proceeding to execution...")
            
            # INSTANT EXECUTION with fresh price
            success = await self._execute_instant_entry(symbol, xgboost_result, current_price)
            if success:
                self.trades_executed += 1
                self.last_trade_time = datetime.now()
                logger.info(f"✅ TRADE EXECUTED: {symbol} | Total Executed: {self.trades_executed}")
            else:
                logger.info(f"❌ EXECUTION FAILED: {symbol}")
            
        except Exception as e:
            logger.error(f"Instant momentum signal handler error for {symbol}: {e}")
    
    async def _analyze_with_xgboost(self, symbol: str, live_data: Dict) -> Optional[Dict]:
        """Instant XGBoost analysis combining historical + live data"""
        try:
            # Use cached historical data if available
            if symbol in self.historical_data_cache:
                # Use cached data for speed
                historical_bars = self.historical_data_cache[symbol]
            else:
                # Quick fetch if not cached
                end_date = datetime.now()
                start_date = end_date - timedelta(days=4)
                historical_bars = await self.polygon.list_aggs(symbol, 1, 'day', start_date, end_date, limit=5)
                
                # Cache for future use
                if historical_bars:
                    self.historical_data_cache[symbol] = historical_bars
            
            # Run XGBoost analysis with fused data
            result = await self.momentum_detector.detect_momentum(symbol)
            
            if result and result.get('momentum_score', 0) > 0:
                # Enhance with live data
                result['live_data'] = live_data
                result['fusion_timestamp'] = datetime.now()
                return result
            
            return None
            
        except Exception as e:
            logger.debug(f"XGBoost analysis failed for {symbol}: {e}")
            return None
    
    async def _execute_instant_entry(self, symbol: str, signal: Dict, current_price: float) -> bool:
        """Execute trade instantly with no delays"""
        try:
            momentum_score = signal['momentum_score']
            
            # Check risk limits
            if len(self.active_positions) >= config.RISK_PARAMS['max_positions']:
                logger.debug(f"Max positions reached: {len(self.active_positions)}")
                return False
            
            # Check available cash
            available_cash = await self.alpaca.get_available_cash()
            if available_cash < config.RISK_PARAMS['min_position_size']:
                logger.debug(f"Insufficient cash: ${available_cash:,.2f}")
                return False
            
            # Get quality grade
            quality_analysis = signal.get('quality_analysis', {})
            quality_grade = quality_analysis.get('overall_grade', 'C')
            
            # INSTANT EXECUTION
            account_value = await self.alpaca.get_account_value()
            success = await self.position_manager.enter_position(
                symbol, momentum_score, current_price, account_value, quality_grade
            )
            
            if success:
                # Record order and update tracking
                self.recent_orders[symbol] = datetime.now()
                self.active_positions[symbol] = {
                    'entry_time': datetime.now(),
                    'entry_price': current_price,
                    'momentum_score': momentum_score
                }
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Instant entry execution failed for {symbol}: {e}")
            return False
    
    async def _position_monitoring_loop(self):
        """Lightweight position monitoring - only remaining scheduled component"""
        sync_counter = 0
        
        while self.running:
            try:
                sync_counter += 1
                
                # Sync with Alpaca every 10 iterations
                if sync_counter % 10 == 0:
                    await self.position_manager.sync_positions_with_alpaca()
                
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
    
    def _passes_quality_filters(self, signal: Dict) -> bool:
        """Simplified filter - rely entirely on XGBoost scores"""
        try:
            symbol = signal['symbol']
            momentum_score = signal.get('momentum_score', 0)
            
            # Only check XGBoost score threshold
            if momentum_score < config.MOMENTUM_CONFIG['entry_threshold']:
                return False
            
            # Basic trading hours check (9:30 AM - 4:00 PM EST)
            import pytz
            est = pytz.timezone('US/Eastern')
            current_time_est = datetime.now(est).time()
            if current_time_est < dt_time(9, 30) or current_time_est > dt_time(16, 0):
                logger.debug(f"❌ {symbol}: Outside trading hours EST: {current_time_est}")
                return False
            
            # TRUST XGBOOST: If it passes score threshold and trading hours, EXECUTE
            logger.info(f"🎯 TRADE APPROVED: {symbol} | Score: {momentum_score:.0f} → EXECUTING")
            return True
            
        except Exception as e:
            logger.error(f"Filter error for {signal.get('symbol', 'unknown')}: {e}")
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
            del self.recent_orders[symbol]
            return False
        
        return True
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            # Try data stream first
            stream_price = await self.data_stream.get_last_price(symbol)
            if stream_price:
                return stream_price
            
            # Fallback to Alpaca
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
                if quote.ask_price and quote.bid_price:
                    return (float(quote.ask_price) + float(quote.bid_price)) / 2
                elif quote.ask_price:
                    return float(quote.ask_price)
                elif quote.bid_price:
                    return float(quote.bid_price)
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to get price for {symbol}: {e}")
            return None
    
    def get_performance_stats(self) -> Dict:
        """Get system performance statistics"""
        stream_stats = self.data_stream.get_stats()
        return {
            'signals_processed': self.signals_processed,
            'trades_executed': self.trades_executed,
            'last_trade_time': self.last_trade_time,
            'active_positions': len(self.active_positions),
            'historical_cache_size': len(self.historical_data_cache),
            'stream_stats': stream_stats
        }
    
    async def shutdown(self):
        """Gracefully shutdown the event-driven system"""
        logger.info("Shutting down event-driven momentum trading system...")
        
        self.running = False
        
        try:
            # Get final performance stats
            stats = self.get_performance_stats()
            logger.info(f"Final Performance: {stats}")
            
            # Cancel all pending tasks first
            tasks = [task for task in asyncio.all_tasks() if not task.done()]
            if tasks:
                logger.info(f"Cancelling {len(tasks)} pending tasks...")
                for task in tasks:
                    task.cancel()
                
                # Wait briefly for tasks to cancel
                try:
                    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=2)
                except asyncio.TimeoutError:
                    logger.warning("Some tasks didn't cancel in time")
            
            # Disconnect from data stream
            try:
                await asyncio.wait_for(self.data_stream.disconnect(), timeout=3)
            except asyncio.TimeoutError:
                logger.warning("Data stream disconnect timeout")
            
            # Close API connections
            try:
                await asyncio.wait_for(self.polygon.disconnect(), timeout=2)
            except asyncio.TimeoutError:
                logger.warning("Polygon disconnect timeout")
            
            try:
                await asyncio.wait_for(self.yahoo.disconnect(), timeout=2)
            except asyncio.TimeoutError:
                logger.warning("Yahoo disconnect timeout")
            
            logger.info("Event-driven system shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")


async def main():
    """Main entry point for event-driven system"""
    system = EventDrivenMomentumSystem()
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    
    def shutdown_handler(sig):
        logger.info(f"Received signal {sig}, shutting down...")
        asyncio.create_task(system.shutdown())
        loop.stop()
    
    for sig in (sys_signal.SIGTERM, sys_signal.SIGINT):
        sys_signal.signal(sig, lambda s, f: shutdown_handler(s))
    
    try:
        # Initialize and run event-driven system
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