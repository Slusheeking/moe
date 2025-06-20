"""
Position Manager with 10% Trailing Stops (Alpaca minimum)
Optimized for momentum trading - ride until 10% pullback from high
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
import logging
from trading_system import config

logger = logging.getLogger(__name__)


class PositionManager:
    def __init__(self, alpaca_client):
        self.alpaca = alpaca_client
        self.positions = {}  # symbol -> position data
        
    async def get_positions(self) -> Dict:
        """Get all current positions with tracking data"""
        alpaca_positions = await self.alpaca.list_positions()
        
        current_positions = {}
        for pos in alpaca_positions:
            symbol = pos['symbol']
            
            # Create or update position tracking (simplified since Alpaca handles stops)
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'entry_time': datetime.now(),
                    'entry_price': float(pos['avg_entry_price']),
                    'quantity': float(pos['qty']),
                    'momentum_score': 0,  # Will be updated when available
                    'trailing_stop_percent': config.RISK_PARAMS['trailing_stop_pct'] * 100
                }
            
            current_positions[symbol] = {
                'symbol': symbol,
                'quantity': float(pos['qty']),
                'avg_entry_price': float(pos['avg_entry_price']),
                'current_price': float(pos.get('current_price', 0)),
                'market_value': float(pos['market_value']),
                'unrealized_pnl': float(pos['unrealized_pl']),
                'unrealized_pnl_pct': float(pos['unrealized_plpc']),
                'side': pos['side'],
                'entry_time': self.positions[symbol]['entry_time'],
                'momentum_score': self.positions[symbol].get('momentum_score', 0),
                'trailing_stop_percent': self.positions[symbol].get('trailing_stop_percent', 10.0)
            }
            
        return current_positions
    
    async def enter_position(self, symbol: str, momentum_score: float, current_price: float, account_value: float, quality_grade: str = 'C') -> bool:
        """Enter momentum position with automatic trailing stop"""
        try:
            # Calculate position size based on momentum score AND quality grade
            position_size = self._calculate_position_size(momentum_score, account_value, quality_grade)
            
            if position_size <= 0:
                logger.info(f"Position size too small for {symbol}: ${position_size} (quality: {quality_grade})")
                return False
            
            # Calculate shares
            shares = int(position_size / current_price)
            if shares < 1:
                logger.info(f"Less than 1 share for {symbol} at ${current_price}")
                return False
            
            # Get trailing stop percentage (convert from decimal to percentage)
            trail_percent = config.RISK_PARAMS['trailing_stop_pct'] * 100
            
            # Use bracket order for automatic trailing stop protection
            bracket_order = await self.alpaca.place_bracket_order(
                symbol=symbol,
                qty=shares,
                trail_percent=trail_percent  # 10% trailing stop from Alpaca
            )
            
            if not bracket_order:
                logger.error(f"Failed to place bracket order for {symbol}")
                return False
            
            # Track position (simplified since Alpaca handles trailing stop)
            self.positions[symbol] = {
                'entry_time': datetime.now(),
                'entry_price': current_price,
                'quantity': shares,
                'momentum_score': momentum_score,
                'order_id': bracket_order.get('id'),
                'trailing_stop_percent': trail_percent
            }
            
            logger.info(f"ENTERED: {symbol} @ ${current_price:.2f} | "
                       f"Shares: {shares} | Size: ${position_size:.0f} | "
                       f"Momentum: {momentum_score:.0f} | Quality: {quality_grade} | Trail: {trail_percent}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to enter position {symbol}: {e}")
            return False
    
    async def update_position(self, symbol: str, current_price: float) -> Optional[str]:
        """Update position tracking (Alpaca handles trailing stops automatically)"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Just track for logging/monitoring - Alpaca handles the trailing stop
        position['current_price'] = current_price
        position['unrealized_pnl'] = (current_price - position['entry_price']) / position['entry_price'] * 100
        
        # Log significant moves
        pnl_pct = position['unrealized_pnl']
        if pnl_pct >= 10 and pnl_pct % 5 == 0:  # Log every 5% above 10%
            logger.info(f"{symbol} unrealized gain: {pnl_pct:+.1f}% @ ${current_price:.2f}")
        
        # No manual trailing stop check needed - Alpaca handles it
        return None
    
    async def exit_position(self, symbol: str, reason: str) -> bool:
        """Exit position with market order"""
        try:
            # Cancel any open orders
            cancelled = await self.alpaca.cancel_all_orders(symbol)
            if cancelled > 0:
                logger.info(f"Cancelled {cancelled} open orders for {symbol}")
            
            # Get current position
            position = await self.alpaca.get_position(symbol)
            if not position:
                logger.warning(f"No position found for {symbol}")
                return False
            
            quantity = int(float(position['qty']))
            
            # Place market order to exit
            exit_order = await self.alpaca.place_order(
                symbol=symbol,
                qty=quantity,
                side='sell',
                type='market'
            )
            
            # Calculate profit/loss
            if symbol in self.positions:
                entry_price = self.positions[symbol]['entry_price']
                current_price = float(position.get('current_price', 0))
                pnl_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                
                logger.info(f"EXITED: {symbol} @ ${current_price:.2f} | "
                           f"Reason: {reason} | PnL: {pnl_pct:+.1f}%")
                
                # Clean up tracking
                del self.positions[symbol]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to exit position {symbol}: {e}")
            return False
    
    def _calculate_position_size(self, momentum_score: float, account_value: float, quality_grade: str = 'C') -> float:
        """Calculate position size based on momentum score AND quality grade"""
        base_size = account_value * config.RISK_PARAMS['max_position_size_pct']
        
        # CONSERVATIVE position sizing based on momentum AND quality
        # Momentum-based multiplier (more conservative)
        if momentum_score >= 80:
            momentum_multiplier = 1.2  # Reduced from 1.5x
        elif momentum_score >= 60:
            momentum_multiplier = 1.0  # Normal size
        elif momentum_score >= 40:
            momentum_multiplier = 0.8  # Smaller for moderate
        else:
            momentum_multiplier = 0.5  # Much smaller for weak
        
        # Quality-based multiplier
        quality_multipliers = {
            'A': 1.1,   # 10% bonus for excellent quality
            'B': 1.0,   # Normal size for good quality
            'C': 0.9,   # 10% reduction for average quality
            'D': 0.7,   # 30% reduction for poor quality
            'F': 0.5    # 50% reduction for failing quality
        }
        
        quality_multiplier = quality_multipliers.get(quality_grade, 0.8)
        
        # Combined multiplier
        total_multiplier = momentum_multiplier * quality_multiplier
        
        position_size = base_size * total_multiplier
        
        # Apply maximum limits
        max_position = account_value * config.RISK_PARAMS['max_position_size_pct']
        position_size = min(position_size, max_position)
        
        # Ensure minimum viable position
        min_position = config.RISK_PARAMS['min_position_size']
        if position_size < min_position:
            position_size = 0  # Don't trade if position would be too small
        
        return position_size
    
    async def check_all_positions(self, price_data: Dict[str, float]) -> List[str]:
        """Update position tracking (Alpaca auto-exits via trailing stops)"""
        closed_positions = []
        
        # Get current Alpaca positions
        alpaca_positions = await self.alpaca.list_positions()
        alpaca_symbols = {pos['symbol'] for pos in alpaca_positions}
        
        # Check for positions that were closed by Alpaca trailing stops
        for symbol in list(self.positions.keys()):
            if symbol not in alpaca_symbols:
                # Position was closed (likely by trailing stop)
                logger.info(f"Position {symbol} was closed by Alpaca (trailing stop)")
                closed_positions.append(symbol)
                del self.positions[symbol]
            elif symbol in price_data:
                # Update tracking for open positions
                current_price = price_data[symbol]
                await self.update_position(symbol, current_price)
        
        return closed_positions
    
    def get_position_summary(self) -> Dict:
        """Get summary of all positions"""
        total_positions = len(self.positions)
        total_value = 0
        avg_momentum = 0
        
        for symbol, pos in self.positions.items():
            total_value += pos.get('quantity', 0) * pos.get('highest_price', 0)
            avg_momentum += pos.get('momentum_score', 0)
        
        if total_positions > 0:
            avg_momentum /= total_positions
        
        return {
            'total_positions': total_positions,
            'total_value': total_value,
            'average_momentum_score': avg_momentum,
            'positions': list(self.positions.keys())
        }
    
    async def sync_positions_with_alpaca(self) -> Dict[str, str]:
        """Synchronize local position tracking with Alpaca's actual positions"""
        try:
            sync_report = {
                'synced': 0,
                'added': 0,
                'removed': 0,
                'errors': 0
            }
            
            # Get current Alpaca positions
            alpaca_positions = await self.alpaca.list_positions()
            alpaca_symbols = {pos['symbol']: pos for pos in alpaca_positions}
            
            # Get current orders to avoid conflicts
            open_orders = await self.alpaca.list_orders(status='open')
            pending_symbols = {order['symbol'] for order in open_orders}
            
            # Check for positions in Alpaca but not in local tracking
            for symbol, alpaca_pos in alpaca_symbols.items():
                if symbol not in self.positions:
                    # Position exists in Alpaca but not locally - add it
                    logger.warning(f"Found untracked position: {symbol}")
                    
                    self.positions[symbol] = {
                        'entry_time': datetime.now() - timedelta(hours=1),  # Approximate
                        'entry_price': float(alpaca_pos['avg_entry_price']),
                        'quantity': float(alpaca_pos['qty']),
                        'momentum_score': 0,  # Unknown
                        'highest_price': float(alpaca_pos.get('current_price', alpaca_pos['avg_entry_price'])),
                        'trailing_stop_percent': config.RISK_PARAMS['trailing_stop_pct'] * 100,
                        'synced': True,
                        'sync_reason': 'found_in_alpaca'
                    }
                    sync_report['added'] += 1
                    logger.info(f"Added untracked position: {symbol} - {alpaca_pos['qty']} shares @ ${alpaca_pos['avg_entry_price']}")
            
            # Check for positions in local tracking but not in Alpaca
            for symbol in list(self.positions.keys()):
                if symbol not in alpaca_symbols and symbol not in pending_symbols:
                    # Position closed outside our system
                    logger.warning(f"Position {symbol} was closed outside the system")
                    
                    # Try to get exit details from orders
                    exit_info = await self._get_position_exit_info(symbol)
                    
                    if exit_info:
                        logger.info(f"Exit found for {symbol}: {exit_info['side']} @ ${exit_info['price']:.2f}")
                    
                    # Remove from tracking
                    del self.positions[symbol]
                    sync_report['removed'] += 1
            
            # Verify quantities match for existing positions
            for symbol in self.positions:
                if symbol in alpaca_symbols:
                    local_qty = self.positions[symbol]['quantity']
                    alpaca_qty = float(alpaca_symbols[symbol]['qty'])
                    
                    if abs(local_qty - alpaca_qty) > 0.01:  # Small tolerance for float comparison
                        logger.warning(f"Quantity mismatch for {symbol}: Local={local_qty}, Alpaca={alpaca_qty}")
                        
                        # Update to Alpaca's quantity
                        self.positions[symbol]['quantity'] = alpaca_qty
                        self.positions[symbol]['sync_adjusted'] = True
                        sync_report['synced'] += 1
            
            # Log sync summary
            total_changes = sync_report['added'] + sync_report['removed'] + sync_report['synced']
            if total_changes > 0:
                logger.info(f"Position Sync Complete: Added={sync_report['added']}, "
                           f"Removed={sync_report['removed']}, Adjusted={sync_report['synced']}")
            
            return sync_report
            
        except Exception as e:
            logger.error(f"Position sync failed: {e}")
            return {'error': str(e)}

    async def _get_position_exit_info(self, symbol: str) -> Optional[Dict]:
        """Try to find exit information for a closed position"""
        try:
            # Get recent orders for this symbol
            all_orders = await self.alpaca.list_orders(
                status='filled',
                symbols=symbol,
                limit=50
            )
            
            # Look for sell orders
            for order in all_orders:
                if order['symbol'] == symbol and order['side'] == 'sell':
                    return {
                        'symbol': symbol,
                        'side': order['side'],
                        'price': float(order['filled_avg_price']),
                        'quantity': float(order['filled_qty']),
                        'time': order['filled_at']
                    }
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not get exit info for {symbol}: {e}")
            return None

    async def validate_trailing_stops(self) -> Dict[str, bool]:
        """Validate that all positions have proper trailing stop orders"""
        validation_results = {}
        
        try:
            # Get all open orders
            open_orders = await self.alpaca.list_orders(status='open')
            
            # Create a map of symbols to their stop orders
            stop_orders = {}
            for order in open_orders:
                if order['order_type'] == 'trailing_stop':
                    symbol = order['symbol']
                    if symbol not in stop_orders:
                        stop_orders[symbol] = []
                    stop_orders[symbol].append(order)
            
            # Check each position
            for symbol, position in self.positions.items():
                if symbol in stop_orders:
                    # Verify stop order parameters
                    for stop_order in stop_orders[symbol]:
                        trail_percent = float(stop_order.get('trail_percent', 0))
                        expected_trail = config.RISK_PARAMS['trailing_stop_pct'] * 100
                        
                        if abs(trail_percent - expected_trail) < 0.1:  # Close enough
                            validation_results[symbol] = True
                        else:
                            logger.warning(f"{symbol} has incorrect trailing stop: {trail_percent}% vs expected {expected_trail}%")
                            validation_results[symbol] = False
                else:
                    # No trailing stop found
                    logger.warning(f"{symbol} is missing trailing stop order")
                    validation_results[symbol] = False
                    
                    # Attempt to add missing trailing stop
                    await self._add_missing_trailing_stop(symbol, position)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Trailing stop validation failed: {e}")
            return {}

    async def _add_missing_trailing_stop(self, symbol: str, position: Dict) -> bool:
        """Add missing trailing stop order for a position"""
        try:
            quantity = position['quantity']
            trail_percent = config.RISK_PARAMS['trailing_stop_pct'] * 100
            
            stop_order = await self.alpaca.place_order(
                symbol=symbol,
                qty=quantity,
                side='sell',
                type='trailing_stop',
                trail_percent=trail_percent,
                time_in_force='gtc'
            )
            
            if stop_order:
                logger.info(f"Added missing trailing stop for {symbol}: {trail_percent}%")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to add trailing stop for {symbol}: {e}")
            return False

    def cleanup_old_tracking(self):
        """Clean up tracking for positions no longer held"""
        # This will be called periodically to clean up
        # positions that were closed outside the system
        pass