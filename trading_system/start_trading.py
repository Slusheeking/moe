"""
Entry Point for Event-Driven Momentum Trading System
Compatible with VS Code play button and direct execution
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add parent directory to Python path for VS Code compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import the modules
try:
    from trading_system.main import EventDrivenMomentumSystem
    from trading_system import config
except ImportError as e:
    print(f"Import error: {e}")
    print("Current working directory:", os.getcwd())
    print("Python path:", sys.path)
    sys.exit(1)

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
    
    if config.RISK_PARAMS['stop_loss_pct'] <= 0:
        errors.append("Stop loss percentage must be positive")
    
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  {error}")
        return False
    
    return True


def log_configuration():
    """Log key configuration parameters"""
    logger.info("Event-Driven System Configuration:")
    logger.info(f"   Account Size: ${config.ACCOUNT_SIZE:,}")
    logger.info(f"   Entry Threshold: {config.MOMENTUM_CONFIG['entry_threshold']}")
    logger.info(f"   Stop Loss: {config.RISK_PARAMS['stop_loss_pct']}%")
    logger.info(f"   Take Profit: {config.RISK_PARAMS['take_profit_pct']}%")
    logger.info(f"   Max Positions: {config.RISK_PARAMS['max_positions']}")
    logger.info(f"   Max Position Size: {config.RISK_PARAMS['max_position_size_pct']*100}%")
    logger.info(f"   Max Simultaneous Entries: {config.RISK_PARAMS['max_simultaneous_entries']}")
    logger.info(f"   Price Range: ${config.UNIVERSE_CONFIG['min_price']}-${config.UNIVERSE_CONFIG['max_price']}")
    logger.info(f"   Position Update Frequency: {config.REALTIME_CONFIG['update_frequency_seconds']}s")


async def main():
    """Main entry point for event-driven system"""
    
    # Validate configuration
    if not validate_configuration():
        logger.error("Configuration validation failed. Exiting.")
        return 1
    
    # Log configuration
    log_configuration()
    
    # Initialize and run event-driven system
    system = None
    try:
        logger.info("🚀 Starting Event-Driven Momentum Trading System...")
        logger.info("=" * 60)
        logger.info("🔥 PURE EVENT-DRIVEN MODE:")
        logger.info("   📡 WebSocket → Instant Filter → XGBoost → Execute")
        logger.info("   ⚡ No scheduled loops - everything triggered by market events")
        logger.info("   🎯 Monitoring ALL market trades for momentum opportunities")
        logger.info("   🛡️  5% Stop Loss + 10% Take Profit via Bracket Orders")
        logger.info("=" * 60)
        
        system = EventDrivenMomentumSystem()
        
        # Initialize components
        await system.initialize()
        
        # Start event-driven trading
        logger.info("🎯 System initialized. Beginning real-time momentum detection...")
        await system.run()
        
    except KeyboardInterrupt:
        logger.info("⌨️ Keyboard interrupt received. Shutting down...")
        
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
    
    logger.info("Event-Driven Momentum Trading System stopped")
    return 0


if __name__ == "__main__":
    # Check if API keys are set
    if not os.getenv('POLYGON_API_KEY'):
        print("⚠️  POLYGON_API_KEY environment variable not set!")
        print("Set it with: export POLYGON_API_KEY='your_key_here'")
    
    if not os.getenv('ALPACA_API_KEY'):
        print("⚠️  ALPACA_API_KEY environment variable not set!")
        print("Set it with: export ALPACA_API_KEY='your_key_here'")
    
    if not os.getenv('ALPACA_SECRET_KEY'):
        print("⚠️  ALPACA_SECRET_KEY environment variable not set!")
        print("Set it with: export ALPACA_SECRET_KEY='your_secret_here'")
    
    # Run the event-driven system
    try:
        print("🎮 VS Code Play Button - Starting Event-Driven Trading System...")
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        sys.exit(1)