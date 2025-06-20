"""
Position Manager with 5% Stop Loss and 10% Take Profit
Optimized for event-driven momentum trading
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
            
            # Create or update position tracking
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'entry_time': datetime.now(),
                    'entry_price': float(pos['avg_entry_price']),
                    'quantity': float(pos['qty']),
                    'momentum_score': 0,
                    'stop_loss_pct': 5.0,  # 5% stop loss
                    'take_profit_pct': 10.0  # 10% take profit
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
                'stop_loss_pct': 5.0,
                'take_profit_pct': 10.0
            }
            
        return current_positions
    
    async def enter_position(self, symbol: str, momentum_score: float, current_price: float, account_value: float, quality_grade: str = 'C') -> bool:
        """Enter position with STRICT PRICE PROTECTION - limit orders only at precise prices"""
        try:
            # Get available cash
            available_cash = await self.alpaca.get_available_cash()
            
            if available_cash <= 0:
                logger.warning(f"No available cash for {symbol}: ${available_cash:,.2f}")
                return False
            
            # GET REAL-TIME BID/ASK DATA FOR PRECISE PRICING
            quote = await self.alpaca.get_latest_quote(symbol)
            if not quote:
                logger.error(f"❌ BLOCKED: No bid/ask data for {symbol} - cannot determine fair price")
                return False
            
            bid_price = float(quote.get('bid_price', 0))
            ask_price = float(quote.get('ask_price', 0))
            
            if bid_price <= 0 or ask_price <= 0:
                logger.error(f"❌ BLOCKED: Invalid bid/ask for {symbol}: bid=${bid_price}, ask=${ask_price}")
                return False
            
            # STRICT SPREAD CHECK - reject if spread too wide
            spread_pct = (ask_price - bid_price) / bid_price
            max_spread = config.EXECUTION_PARAMS.get('max_spread_pct', 0.01)
            
            if spread_pct > max_spread:
                logger.error(f"❌ BLOCKED: Spread too wide for {symbol}: {spread_pct:.2%} > {max_spread:.2%}")
                return False
            
            # CALCULATE EXACT LIMIT PRICE - bid + small buffer (NO MARKET ORDERS)
            price_buffer = config.EXECUTION_PARAMS.get('limit_price_buffer', 0.002)
            limit_price = round(bid_price * (1 + price_buffer), 2)
            
            # Ensure limit price doesn't exceed ask
            if limit_price > ask_price:
                limit_price = round((bid_price + ask_price) / 2, 2)  # Midpoint if buffer too large
            
            logger.info(f"🎯 PRICE PROTECTION: {symbol}")
            logger.info(f"   📊 Bid: ${bid_price:.2f} | Ask: ${ask_price:.2f} | Spread: {spread_pct:.2%}")
            logger.info(f"   ✅ LIMIT PRICE: ${limit_price:.2f} (bid + {price_buffer:.1%} buffer)")
            
            # Calculate position size
            position_size = self._calculate_position_size(momentum_score, available_cash, quality_grade)
            
            if position_size <= 0:
                logger.info(f"Position size too small for {symbol}: ${position_size}")
                return False
            
            # Calculate shares based on LIMIT PRICE (not current market price)
            shares = int(position_size / limit_price)
            if shares < 1:
                logger.info(f"Less than 1 share for {symbol} at limit ${limit_price}")
                return False
            
            # Final safety check with limit price
            total_cost = shares * limit_price
            if total_cost > available_cash:
                logger.warning(f"Trade cost ${total_cost:,.2f} exceeds available cash ${available_cash:,.2f}")
                return False
            
            # Calculate stop loss and take profit based on LIMIT PRICE
            stop_loss_price = round(limit_price * 0.95, 2)  # 5% stop loss
            take_profit_price = round(limit_price * 1.10, 2)  # 10% take profit
            
            # PLACE STRICT LIMIT ORDER ONLY - NO MARKET ORDER FALLBACK
            limit_order = await self.alpaca.place_limit_order_with_bracket(
                symbol=symbol,
                qty=shares,
                limit_price=limit_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                timeout_seconds=config.EXECUTION_PARAMS.get('order_timeout_seconds', 30)
            )
            
            if not limit_order:
                logger.error(f"❌ LIMIT ORDER FAILED: {symbol} @ ${limit_price:.2f} - NO FALLBACK")
                return False
            
            # Track position with exact limit price
            self.positions[symbol] = {
                'entry_time': datetime.now(),
                'entry_price': limit_price,  # Use limit price, not market price
                'quantity': shares,
                'momentum_score': momentum_score,
                'order_id': limit_order.get('id'),
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price,
                'stop_loss_pct': 5.0,
                'take_profit_pct': 10.0,
                'quality_grade': quality_grade,
                'bid_at_entry': bid_price,
                'ask_at_entry': ask_price,
                'spread_at_entry': spread_pct
            }
            
            logger.info(f"✅ LIMIT ORDER PLACED: {symbol}")
            logger.info(f"   📊 Shares: {shares} | Size: ${position_size:.0f}")
            logger.info(f"   💰 EXACT LIMIT: ${limit_price:.2f} (NO SLIPPAGE)")
            logger.info(f"   🛡️  Stop Loss: ${stop_loss_price:.2f} (-5%)")
            logger.info(f"   💰 Take Profit: ${take_profit_price:.2f} (+10%)")
            logger.info(f"   🎯 Momentum: {momentum_score:.0f} | Grade: {quality_grade}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to enter position {symbol}: {e}")
            return False
    
    async def update_position(self, symbol: str, current_price: float) -> Optional[str]:
        """Update position tracking (Alpaca handles bracket orders automatically)"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Update tracking data
        position['current_price'] = current_price
        entry_price = position['entry_price']
        unrealized_pnl_pct = (current_price - entry_price) / entry_price * 100
        position['unrealized_pnl_pct'] = unrealized_pnl_pct
        
        # Log significant moves
        if unrealized_pnl_pct >= 5 and unrealized_pnl_pct % 2.5 == 0:  # Log every 2.5% above 5%
            logger.info(f"📈 {symbol}: {unrealized_pnl_pct:+.1f}% @ ${current_price:.2f}")
        elif unrealized_pnl_pct <= -2.5 and unrealized_pnl_pct % 2.5 == 0:  # Log losses
            logger.info(f"📉 {symbol}: {unrealized_pnl_pct:+.1f}% @ ${current_price:.2f}")
        
        # Calculate target prices based on entry price
        stop_loss_price = position.get('stop_loss_price', entry_price * 0.95)  # 5% stop loss
        take_profit_price = position.get('take_profit_price', entry_price * 1.10)  # 10% take profit
        
        # Update position with calculated targets if not set
        if 'stop_loss_price' not in position:
            position['stop_loss_price'] = stop_loss_price
        if 'take_profit_price' not in position:
            position['take_profit_price'] = take_profit_price
        
        if current_price <= stop_loss_price * 1.01:  # Within 1% of stop loss
            logger.warning(f"⚠️ {symbol} near stop loss: ${current_price:.2f} vs ${stop_loss_price:.2f}")
        elif current_price >= take_profit_price * 0.99:  # Within 1% of take profit
            logger.info(f"🎯 {symbol} near take profit: ${current_price:.2f} vs ${take_profit_price:.2f}")
        
        return None
    
    async def exit_position(self, symbol: str, reason: str) -> bool:
        """Manual exit position (emergency use)"""
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
            
            # Calculate P&L
            if symbol in self.positions:
                entry_price = self.positions[symbol]['entry_price']
                current_price = float(position.get('current_price', 0))
                pnl_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                
                logger.info(f"🚪 MANUAL EXIT: {symbol} @ ${current_price:.2f}")
                logger.info(f"   📊 Reason: {reason} | PnL: {pnl_pct:+.1f}%")
                
                del self.positions[symbol]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to exit position {symbol}: {e}")
            return False
    
    def _calculate_position_size(self, momentum_score: float, available_cash: float, quality_grade: str = 'C') -> float:
        """Calculate AGGRESSIVE position size for event-driven momentum trading"""
        # More aggressive allocation for momentum trading
        max_simultaneous = config.RISK_PARAMS.get('max_simultaneous_entries', 5)  # Fewer positions = bigger sizes
        cash_reserve_pct = config.RISK_PARAMS.get('min_cash_reserve_pct', 0.05)  # Less cash reserve
        
        # Reserve less cash for future opportunities
        usable_cash = available_cash * (1 - cash_reserve_pct)
        
        # MUCH LARGER base allocation per position
        base_allocation_pct = 0.60 / max_simultaneous  # 60% total spread across fewer entries
        base_size = usable_cash * base_allocation_pct
        
        # AGGRESSIVE momentum-based multipliers
        if momentum_score >= 90:
            momentum_multiplier = 4.0  # Exceptional momentum - GO BIG
        elif momentum_score >= 80:
            momentum_multiplier = 3.5  # Very strong momentum
        elif momentum_score >= 70:
            momentum_multiplier = 3.0  # Strong momentum
        elif momentum_score >= 60:
            momentum_multiplier = 2.5  # Good momentum
        elif momentum_score >= 50:
            momentum_multiplier = 2.0  # Moderate momentum
        elif momentum_score >= 40:
            momentum_multiplier = 1.5  # Decent momentum
        elif momentum_score >= 25:  # Entry threshold
            momentum_multiplier = 1.0  # Base size
        else:
            momentum_multiplier = 0.5  # Weak signals
        
        # AGGRESSIVE quality-based multipliers
        quality_multipliers = {
            'A': 2.0,   # Excellent quality - GO BIG
            'B': 1.7,   # Good quality
            'C': 1.4,   # Average quality
            'D': 1.0,   # Poor quality
            'F': 0.7    # Failing quality
        }
        
        quality_multiplier = quality_multipliers.get(quality_grade, 1.0)
        
        # Combined position size
        total_multiplier = momentum_multiplier * quality_multiplier
        position_size = base_size * total_multiplier
        
        # MUCH HIGHER safety caps
        max_single_position = usable_cash * 0.40  # Max 40% per position (was 20%)
        position_size = min(position_size, max_single_position)
        
        # Higher minimum position
        min_position = config.RISK_PARAMS.get('min_position_size', 2000)  # $2K minimum (was $500)
        if position_size < min_position:
            if usable_cash >= min_position * 1.2:
                position_size = min_position
            else:
                position_size = 0
        
        logger.info(f"🚀 AGGRESSIVE POSITION SIZE: ${position_size:,.0f} (momentum: {momentum_score:.0f}, quality: {quality_grade}, multiplier: {total_multiplier:.1f}x)")
        return position_size
    
    async def check_all_positions(self, price_data: Dict[str, float]) -> List[str]:
        """Update position tracking (Alpaca auto-exits via bracket orders)"""
        closed_positions = []
        
        # Get current Alpaca positions
        alpaca_positions = await self.alpaca.list_positions()
        alpaca_symbols = {pos['symbol'] for pos in alpaca_positions}
        
        # Check for positions closed by Alpaca bracket orders
        for symbol in list(self.positions.keys()):
            if symbol not in alpaca_symbols:
                # Position was closed (by stop loss or take profit)
                await self._log_position_exit(symbol)
                closed_positions.append(symbol)
                del self.positions[symbol]
            elif symbol in price_data:
                # Update tracking for open positions
                current_price = price_data[symbol]
                await self.update_position(symbol, current_price)
        
        return closed_positions
    
    async def _log_position_exit(self, symbol: str):
        """Log details when a position is closed"""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            entry_price = position.get('entry_price', 0)
            stop_loss_price = position.get('stop_loss_price', 0)
            take_profit_price = position.get('take_profit_price', 0)
            
            # Try to determine exit reason by checking recent orders
            recent_orders = await self.alpaca.list_orders(status='filled', limit=10)
            exit_reason = "Unknown"
            exit_price = 0
            
            for order in recent_orders:
                if (order.get('symbol') == symbol and 
                    order.get('side') == 'sell' and
                    order.get('filled_at')):
                    
                    exit_price = float(order.get('filled_avg_price', 0))
                    
                    # Determine if stop loss or take profit
                    if abs(exit_price - stop_loss_price) < abs(exit_price - take_profit_price):
                        exit_reason = "Stop Loss (-5%)"
                    else:
                        exit_reason = "Take Profit (+10%)"
                    break
            
            if exit_price > 0:
                pnl_pct = (exit_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                logger.info(f"🏁 POSITION CLOSED: {symbol}")
                logger.info(f"   📊 Entry: ${entry_price:.2f} → Exit: ${exit_price:.2f}")
                logger.info(f"   📈 P&L: {pnl_pct:+.1f}% | Reason: {exit_reason}")
            else:
                logger.info(f"🏁 POSITION CLOSED: {symbol} (details unavailable)")
                
        except Exception as e:
            logger.error(f"Error logging position exit for {symbol}: {e}")
    
    def get_position_summary(self) -> Dict:
        """Get summary of all positions"""
        total_positions = len(self.positions)
        total_value = 0
        avg_momentum = 0
        
        for symbol, pos in self.positions.items():
            total_value += pos.get('quantity', 0) * pos.get('current_price', pos.get('entry_price', 0))
            avg_momentum += pos.get('momentum_score', 0)
        
        if total_positions > 0:
            avg_momentum /= total_positions
        
        return {
            'total_positions': total_positions,
            'total_value': total_value,
            'average_momentum_score': avg_momentum,
            'positions': list(self.positions.keys()),
            'stop_loss_pct': 5.0,
            'take_profit_pct': 10.0
        }
    
    async def sync_positions_with_alpaca(self) -> Dict[str, str]:
        """Synchronize local position tracking with Alpaca"""
        try:
            sync_report = {'synced': 0, 'added': 0, 'removed': 0}
            
            alpaca_positions = await self.alpaca.list_positions()
            alpaca_symbols = {pos['symbol']: pos for pos in alpaca_positions}
            
            # Check for new positions in Alpaca
            for symbol, alpaca_pos in alpaca_symbols.items():
                if symbol not in self.positions:
                    logger.warning(f"Found untracked position: {symbol}")
                    
                    self.positions[symbol] = {
                        'entry_time': datetime.now() - timedelta(hours=1),
                        'entry_price': float(alpaca_pos['avg_entry_price']),
                        'quantity': float(alpaca_pos['qty']),
                        'momentum_score': 0,
                        'stop_loss_pct': 5.0,
                        'take_profit_pct': 10.0,
                        'synced': True
                    }
                    sync_report['added'] += 1
            
            # Check for closed positions
            for symbol in list(self.positions.keys()):
                if symbol not in alpaca_symbols:
                    await self._log_position_exit(symbol)
                    del self.positions[symbol]
                    sync_report['removed'] += 1
            
            total_changes = sync_report['added'] + sync_report['removed']
            if total_changes > 0:
                logger.info(f"Position Sync: Added={sync_report['added']}, Removed={sync_report['removed']}")
            
            return sync_report
            
        except Exception as e:
            logger.error(f"Position sync failed: {e}")
            return {'error': str(e)}