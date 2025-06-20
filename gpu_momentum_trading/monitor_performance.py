#!/usr/bin/env python3
"""
Performance Monitor for GPU Momentum Trading System
Real-time dashboard for monitoring trading performance
"""

import asyncio
import os
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress alpaca debug logs
logging.getLogger('alpaca').setLevel(logging.WARNING)


class PerformanceMonitor:
    """Monitor trading performance in real-time"""
    
    def __init__(self):
        self.client = TradingClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            paper=True
        )
    
    def get_account_status(self):
        """Get current account status"""
        account = self.client.get_account()
        return {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'pattern_day_trader': account.pattern_day_trader,
            'trades_today': int(account.daytrade_count) if account.daytrade_count else 0
        }
    
    def get_positions(self):
        """Get all current positions"""
        positions = self.client.get_all_positions()
        return [{
            'symbol': pos.symbol,
            'qty': int(pos.qty),
            'avg_price': float(pos.avg_entry_price),
            'current_price': float(pos.current_price) if pos.current_price else 0,
            'market_value': float(pos.market_value),
            'pnl': float(pos.unrealized_pl),
            'pnl_pct': float(pos.unrealized_plpc) * 100
        } for pos in positions]
    
    def get_recent_orders(self, limit=10):
        """Get recent orders"""
        orders = self.client.get_orders(limit=limit)
        return [{
            'symbol': order.symbol,
            'side': order.side.value,
            'qty': int(order.qty),
            'type': order.order_type.value,
            'status': order.status.value,
            'filled_qty': int(order.filled_qty) if order.filled_qty else 0,
            'filled_price': float(order.filled_avg_price) if order.filled_avg_price else 0,
            'created': order.created_at
        } for order in orders]
    
    def display_dashboard(self):
        """Display performance dashboard"""
        os.system('clear' if os.name != 'nt' else 'cls')
        
        print("=" * 80)
        print("GPU MOMENTUM TRADING SYSTEM - PERFORMANCE MONITOR")
        print("=" * 80)
        print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Account Status
        account = self.get_account_status()
        print("ACCOUNT STATUS")
        print("-" * 40)
        print(f"Portfolio Value: ${account['portfolio_value']:,.2f}")
        print(f"Cash Available:  ${account['cash']:,.2f}")
        print(f"Buying Power:    ${account['buying_power']:,.2f}")
        print(f"Trades Today:    {account['trades_today']}")
        print()
        
        # Current Positions
        positions = self.get_positions()
        if positions:
            print("CURRENT POSITIONS")
            print("-" * 80)
            print(f"{'Symbol':<8} {'Qty':>6} {'Avg Price':>10} {'Current':>10} {'P&L $':>12} {'P&L %':>8}")
            print("-" * 80)
            
            total_pnl = 0
            for pos in sorted(positions, key=lambda x: x['pnl'], reverse=True):
                total_pnl += pos['pnl']
                pnl_color = '\033[92m' if pos['pnl'] >= 0 else '\033[91m'
                reset_color = '\033[0m'
                
                print(f"{pos['symbol']:<8} {pos['qty']:>6} "
                      f"${pos['avg_price']:>9.2f} ${pos['current_price']:>9.2f} "
                      f"{pnl_color}${pos['pnl']:>11.2f} {pos['pnl_pct']:>7.1f}%{reset_color}")
            
            print("-" * 80)
            pnl_color = '\033[92m' if total_pnl >= 0 else '\033[91m'
            print(f"{'TOTAL':<8} {'':<27} {pnl_color}${total_pnl:>11.2f}{reset_color}")
            print()
        else:
            print("No open positions")
            print()
        
        # Recent Orders
        orders = self.get_recent_orders(5)
        if orders:
            print("RECENT ORDERS")
            print("-" * 80)
            print(f"{'Time':<20} {'Symbol':<8} {'Side':<5} {'Qty':>6} {'Status':<10} {'Filled':>10}")
            print("-" * 80)
            
            for order in orders:
                time_str = order['created'].strftime('%H:%M:%S') if order['created'] else 'N/A'
                filled_str = f"${order['filled_price']:.2f}" if order['filled_price'] > 0 else '-'
                status_color = '\033[92m' if order['status'] == 'filled' else '\033[93m'
                reset_color = '\033[0m'
                
                print(f"{time_str:<20} {order['symbol']:<8} {order['side']:<5} "
                      f"{order['qty']:>6} {status_color}{order['status']:<10}{reset_color} {filled_str:>10}")
        
        print()
        print("Press Ctrl+C to exit")
    
    async def run(self):
        """Run performance monitor"""
        while True:
            try:
                self.display_dashboard()
                await asyncio.sleep(5)  # Update every 5 seconds
            except KeyboardInterrupt:
                print("\n\nMonitor stopped")
                break
            except Exception as e:
                print(f"\nError: {e}")
                await asyncio.sleep(10)


async def main():
    """Main entry point"""
    monitor = PerformanceMonitor()
    await monitor.run()


if __name__ == "__main__":
    # Check API keys
    if not os.getenv('ALPACA_API_KEY') or not os.getenv('ALPACA_SECRET_KEY'):
        print("Error: Alpaca API keys not set")
        print("Set them with:")
        print("  export ALPACA_API_KEY='your_key'")
        print("  export ALPACA_SECRET_KEY='your_secret'")
        exit(1)
    
    asyncio.run(main())