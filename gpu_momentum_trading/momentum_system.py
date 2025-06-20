"""
GPU-Accelerated Momentum Trading System
Main system coordinating all components
"""

import asyncio
import logging
import signal
from datetime import datetime, time as dt_time
from typing import Dict
import pytz
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from gpu_momentum_scanner import GPUMomentumScanner
from realtime_executor import MomentumExecutor
from market_data_manager import MarketDataManager
from system_config import MOMENTUM_CONFIG

logger = logging.getLogger(__name__)


class GPUMomentumSystem:
    """Main momentum trading system leveraging A10 GPU"""
    
    def __init__(self):
        # API Keys
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        self.alpaca_key = os.getenv('ALPACA_API_KEY')
        self.alpaca_secret = os.getenv('ALPACA_SECRET_KEY')
        
        # Components
        self.gpu_scanner = GPUMomentumScanner(max_symbols=15000)
        self.executor = MomentumExecutor(
            self.alpaca_key, 
            self.alpaca_secret,
            paper=True
        )
        self.data_manager = MarketDataManager(self.polygon_key, self.gpu_scanner)
        
        # System state
        self.running = False
        self.active_signals = {}
        self.execution_lock = asyncio.Lock()
        
        # Performance tracking
        self.signals_processed = 0
        self.trades_executed = 0
        
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing GPU Momentum Trading System...")
        
        # Connect data manager
        await self.data_manager.connect()
        
        # Set up momentum callback
        self.data_manager.add_momentum_callback(self._handle_momentum_signal)
        
        # Log system info
        account_value = self.executor.get_available_cash()
        logger.info(f"System initialized with ${account_value:,.2f}")
        logger.info("GPU acceleration active on NVIDIA A10")
        
        self.running = True
    
    async def _handle_momentum_signal(self, symbol: str, data: Dict):
        """Handle momentum signal from GPU scanner"""
        try:
            momentum_score = data.get('momentum_score', 0)
            price = data.get('price', 0)
            
            logger.info(f"🎯 Processing signal: {symbol} Score: {momentum_score:.1f} Price: ${price:.2f}")
            
            # Skip if score too low
            if momentum_score < MOMENTUM_CONFIG['min_score']:
                logger.info(f"❌ {symbol}: Score {momentum_score:.1f} below threshold {MOMENTUM_CONFIG['min_score']}")
                return
            
            # Skip if price outside $5-$500 range
            if price < 5.0 or price > 500.0:
                logger.info(f"❌ {symbol}: Price ${price:.2f} outside $5-$500 range")
                return
            
            # Skip if already processing
            if symbol in self.active_signals:
                logger.info(f"❌ {symbol}: Already processing")
                return
            
            # Check market hours
            if not self._is_market_hours():
                logger.info(f"❌ {symbol}: Market closed")
                return
            
            # No max position limit - rely on available cash only
            
            # Mark as processing
            self.active_signals[symbol] = datetime.now()
            self.signals_processed += 1
            
            logger.info(f"📈 {symbol}: Getting real-time quote...")
            
            # Get real-time quote - ZERO COPY from WebSocket cache
            quote = self.data_manager.get_quote(symbol)
            if not quote:
                logger.warning(f"❌ {symbol}: Failed to get quote")
                del self.active_signals[symbol]
                return
            
            bid = quote.get('bid', 0)
            ask = quote.get('ask', 0)
            
            logger.info(f"💰 {symbol}: Quote - Bid: ${bid:.2f}, Ask: ${ask:.2f}")
            
            # Validate quote
            if bid <= 0 or ask <= 0:
                logger.warning(f"❌ {symbol}: Invalid quote - Bid: ${bid:.2f}, Ask: ${ask:.2f}")
                del self.active_signals[symbol]
                return
                
            spread_pct = (ask - bid) / ask * 100
            if spread_pct > 1.0:  # Max 1% spread
                logger.warning(f"❌ {symbol}: Spread too wide {spread_pct:.2f}%")
                del self.active_signals[symbol]
                return
            
            logger.info(f"⚡ {symbol}: Executing trade - Spread: {spread_pct:.2f}%")
            
            # Execute trade
            async with self.execution_lock:
                success = await self.executor.execute_momentum_trade(
                    symbol, momentum_score, price, bid, ask
                )
                
                if success:
                    self.trades_executed += 1
                    logger.info(f"🚀 TRADE #{self.trades_executed}: {symbol} Score: {momentum_score:.0f}")
                else:
                    logger.warning(f"❌ {symbol}: Trade execution failed")
            
            # Cleanup
            del self.active_signals[symbol]
            
        except Exception as e:
            logger.error(f"Signal handler error for {symbol}: {e}")
            if symbol in self.active_signals:
                del self.active_signals[symbol]
    
    def _is_market_hours(self) -> bool:
        """Check if market is open"""
        est = pytz.timezone('US/Eastern')
        now = datetime.now(est)
        
        # Monday-Friday
        if now.weekday() >= 5:
            return False
        
        # 9:30 AM - 4:00 PM EST
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        
        return market_open <= now.time() <= market_close
    
    async def run(self):
        """Main event loop"""
        logger.info("Starting momentum detection...")
        
        while self.running:
            try:
                # Update positions every 10 seconds
                positions = await self.executor.update_positions()
                
                if positions:
                    total_pnl = sum(pos['pnl'] for pos in positions.values())
                    logger.info(f"Positions: {len(positions)} | P&L: ${total_pnl:+,.2f}")
                
                # Log performance every minute
                if self.signals_processed % 100 == 0 and self.signals_processed > 0:
                    logger.info(f"Performance: {self.signals_processed} signals, {self.trades_executed} trades")
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(30)
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down system...")
        self.running = False
        
        # Disconnect data
        await self.data_manager.disconnect()
        
        # Log final stats
        logger.info(f"Final: {self.trades_executed} trades executed")


async def main():
    """Entry point"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('momentum_trading.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create system
    system = GPUMomentumSystem()
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    
    def shutdown_handler(sig):
        logger.info(f"Received signal {sig}")
        asyncio.create_task(system.shutdown())
        loop.stop()
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, f: shutdown_handler(s))
    
    try:
        # Initialize
        await system.initialize()
        
        # Run
        await system.run()
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())