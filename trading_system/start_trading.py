"""
Entry Point for Momentum Trading System
Simple launcher for the pure momentum detection system
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add trading_system to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_system.main import MomentumTradingSystem
from trading_system import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOGGING_CONFIG['level']),
    format=config.LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(config.LOGGING_CONFIG['file']),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def validate_configuration():
    """Validate required configuration before startup"""
    errors = []
    
    # Check API keys
    if not config.POLYGON_API_KEY:
        errors.append("POLYGON_API_KEY not set in environment")
    
    if not config.ALPACA_KEY:
        errors.append("ALPACA_API_KEY not set in environment")
    
    if not config.ALPACA_SECRET:
        errors.append("ALPACA_SECRET_KEY not set in environment")
    
    # Check critical config values
    if config.ACCOUNT_SIZE <= 0:
        errors.append("ACCOUNT_SIZE must be positive")
    
    if config.MOMENTUM_CONFIG['entry_threshold'] <= 0:
        errors.append("Momentum entry threshold must be positive")
    
    if config.RISK_PARAMS['trailing_stop_pct'] <= 0:
        errors.append("Trailing stop percentage must be positive")
    
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  {error}")
        return False
    
    return True


def log_configuration():
    """Log key configuration parameters"""
    logger.info("Configuration Summary:")
    logger.info(f"   Account Size: ${config.ACCOUNT_SIZE:,}")
    logger.info(f"   Entry Threshold: {config.MOMENTUM_CONFIG['entry_threshold']}")
    logger.info(f"   Trailing Stop: {config.RISK_PARAMS['trailing_stop_pct']*100}%")
    logger.info(f"   Max Positions: {config.RISK_PARAMS['max_positions']}")
    logger.info(f"   Max Position Size: {config.RISK_PARAMS['max_position_size_pct']*100}%")
    logger.info(f"   Universe Size: {config.UNIVERSE_CONFIG['max_universe_size']:,}")
    logger.info(f"   Scan Frequency: {config.REALTIME_CONFIG['scan_frequency_seconds']}s")
    logger.info(f"   Update Frequency: {config.REALTIME_CONFIG['update_frequency_seconds']}s")


async def main():
    """Main entry point"""
    
    # Validate configuration
    if not validate_configuration():
        logger.error("Configuration validation failed. Exiting.")
        return 1
    
    # Log configuration
    log_configuration()
    
    # Initialize and run system
    system = None
    try:
        logger.info("Starting Momentum Trading System...")
        
        system = MomentumTradingSystem()
        
        # Initialize components
        await system.initialize()
        
        # Start trading
        logger.info("System initialized. Beginning momentum detection...")
        await system.run()
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        if system:
            try:
                await system.shutdown()
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
    
    logger.info("Momentum Trading System stopped")
    return 0


if __name__ == "__main__":
    # Run the system
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        sys.exit(1)