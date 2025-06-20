#!/usr/bin/env python3
"""
Cancel Orders and Positions Script
Allows canceling orders and closing positions by symbol or all at once
Usage: python cancel_orders_positions.py --symbol QNTM
       python cancel_orders_positions.py --symbol ALL
"""

import asyncio
import argparse
import logging
import sys
from typing import List, Dict

from trading_system.alpaca_client import AlpacaClient
from trading_system import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrderPositionCanceller:
    """Utility for canceling orders and closing positions"""
    
    def __init__(self):
        self.alpaca = AlpacaClient(
            config.ALPACA_KEY, 
            config.ALPACA_SECRET, 
            config.ALPACA_BASE_URL
        )
    
    async def cancel_orders_for_symbol(self, symbol: str) -> Dict:
        """Cancel all orders for a specific symbol"""
        try:
            logger.info(f"🔍 Checking orders for {symbol}...")
            
            # Get all open orders for this symbol
            open_orders = await self.alpaca.list_orders(status='open')
            symbol_orders = [order for order in open_orders if order.get('symbol') == symbol]
            
            if not symbol_orders:
                logger.info(f"📋 No open orders found for {symbol}")
                return {'cancelled': 0, 'failed': 0}
            
            logger.info(f"📋 Found {len(symbol_orders)} open orders for {symbol}")
            
            cancelled_count = 0
            failed_count = 0
            
            for order in symbol_orders:
                order_id = order.get('id')
                order_type = order.get('order_type')
                qty = order.get('qty')
                
                logger.info(f"  Cancelling: {order_type} {qty} {symbol} (ID: {order_id})")
                
                success = await self.alpaca.cancel_order(order_id)
                if success:
                    cancelled_count += 1
                    logger.info(f"  ✅ Cancelled order {order_id}")
                else:
                    failed_count += 1
                    logger.error(f"  ❌ Failed to cancel order {order_id}")
            
            logger.info(f"🎯 {symbol} Orders: {cancelled_count} cancelled, {failed_count} failed")
            return {'cancelled': cancelled_count, 'failed': failed_count}
            
        except Exception as e:
            logger.error(f"Error cancelling orders for {symbol}: {e}")
            return {'cancelled': 0, 'failed': 1}
    
    async def cancel_all_orders(self) -> Dict:
        """Cancel all open orders"""
        try:
            logger.info("🔍 Checking all open orders...")
            
            # Use Alpaca's cancel all orders method
            cancelled_count = await self.alpaca.cancel_all_orders()
            
            logger.info(f"🎯 All Orders: {cancelled_count} cancelled")
            return {'cancelled': cancelled_count, 'failed': 0}
            
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return {'cancelled': 0, 'failed': 1}
    
    async def close_position_for_symbol(self, symbol: str) -> Dict:
        """Close position for a specific symbol"""
        try:
            logger.info(f"🔍 Checking position for {symbol}...")
            
            # Get position for this symbol
            position = await self.alpaca.get_position(symbol)
            
            if not position:
                logger.info(f"📊 No position found for {symbol}")
                return {'closed': 0, 'failed': 0}
            
            qty = int(float(position['qty']))
            current_price = float(position['current_price'])
            market_value = float(position['market_value'])
            
            logger.info(f"📊 Found position: {qty} shares of {symbol} @ ${current_price:.2f} (${market_value:,.2f})")
            
            # Place market sell order to close position
            logger.info(f"  Placing market sell order for {qty} shares...")
            
            order_result = await self.alpaca.place_order(
                symbol=symbol,
                qty=abs(qty),  # Ensure positive quantity
                side='sell',
                type='market',
                time_in_force='day'
            )
            
            if order_result:
                logger.info(f"  ✅ Market sell order placed for {symbol} (Order ID: {order_result.get('id')})")
                return {'closed': 1, 'failed': 0}
            else:
                logger.error(f"  ❌ Failed to place sell order for {symbol}")
                return {'closed': 0, 'failed': 1}
                
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return {'closed': 0, 'failed': 1}
    
    async def close_all_positions(self) -> Dict:
        """Close all open positions"""
        try:
            logger.info("🔍 Checking all positions...")
            
            # Get all positions
            positions = await self.alpaca.list_positions()
            
            if not positions:
                logger.info("📊 No positions found")
                return {'closed': 0, 'failed': 0}
            
            logger.info(f"📊 Found {len(positions)} positions")
            
            closed_count = 0
            failed_count = 0
            
            for position in positions:
                symbol = position['symbol']
                qty = int(float(position['qty']))
                current_price = float(position['current_price'])
                market_value = float(position['market_value'])
                
                logger.info(f"  Closing: {qty} shares of {symbol} @ ${current_price:.2f} (${market_value:,.2f})")
                
                # Place market sell order
                order_result = await self.alpaca.place_order(
                    symbol=symbol,
                    qty=abs(qty),
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                
                if order_result:
                    closed_count += 1
                    logger.info(f"  ✅ Market sell order placed for {symbol}")
                else:
                    failed_count += 1
                    logger.error(f"  ❌ Failed to place sell order for {symbol}")
                
                # Small delay between orders
                await asyncio.sleep(0.5)
            
            logger.info(f"🎯 All Positions: {closed_count} closed, {failed_count} failed")
            return {'closed': closed_count, 'failed': failed_count}
            
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return {'closed': 0, 'failed': 1}
    
    async def process_symbol(self, symbol: str) -> None:
        """Process cancellation/closure for specific symbol or ALL"""
        try:
            await self.alpaca.connect()
            
            if symbol.upper() == 'ALL':
                logger.info("🚨 CANCELLING ALL ORDERS AND CLOSING ALL POSITIONS")
                
                # Cancel all orders first
                order_results = await self.cancel_all_orders()
                
                # Then close all positions
                position_results = await self.close_all_positions()
                
                # Summary
                logger.info("\n" + "="*50)
                logger.info("📋 FINAL SUMMARY:")
                logger.info(f"  Orders cancelled: {order_results['cancelled']}")
                logger.info(f"  Orders failed: {order_results['failed']}")
                logger.info(f"  Positions closed: {position_results['closed']}")
                logger.info(f"  Positions failed: {position_results['failed']}")
                logger.info("="*50)
                
            else:
                logger.info(f"🚨 CANCELLING ORDERS AND CLOSING POSITION FOR {symbol}")
                
                # Cancel orders for this symbol
                order_results = await self.cancel_orders_for_symbol(symbol)
                
                # Close position for this symbol
                position_results = await self.close_position_for_symbol(symbol)
                
                # Summary
                logger.info("\n" + "="*50)
                logger.info(f"📋 FINAL SUMMARY FOR {symbol}:")
                logger.info(f"  Orders cancelled: {order_results['cancelled']}")
                logger.info(f"  Orders failed: {order_results['failed']}")
                logger.info(f"  Positions closed: {position_results['closed']}")
                logger.info(f"  Positions failed: {position_results['failed']}")
                logger.info("="*50)
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
        finally:
            await self.alpaca.disconnect()


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Cancel orders and close positions for specific symbols or all positions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cancel_orders_positions.py --symbol QNTM     # Cancel orders and close position for QNTM
  python cancel_orders_positions.py --symbol ALL      # Cancel all orders and close all positions
        """
    )
    
    parser.add_argument(
        '--symbol',
        required=True,
        help='Symbol to cancel orders/close positions for, or "ALL" for everything'
    )
    
    args = parser.parse_args()
    
    # Validate symbol
    symbol = args.symbol.upper().strip()
    if not symbol:
        logger.error("❌ Symbol cannot be empty")
        sys.exit(1)
    
    if symbol != 'ALL' and (len(symbol) > 5 or not symbol.isalpha()):
        logger.error(f"❌ Invalid symbol format: {symbol}")
        sys.exit(1)
    
    # Confirmation prompt for ALL
    if symbol == 'ALL':
        print(f"\n🚨 WARNING: This will cancel ALL orders and close ALL positions!")
        print(f"🚨 This action cannot be undone!")
        confirm = input("\nType 'YES' to confirm: ").strip()
        
        if confirm != 'YES':
            print("❌ Operation cancelled")
            sys.exit(0)
    
    # Process the request
    canceller = OrderPositionCanceller()
    await canceller.process_symbol(symbol)
    
    logger.info("✅ Operation completed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⌨️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)