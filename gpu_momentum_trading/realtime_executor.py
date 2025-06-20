"""
Real-Time Momentum Executor
Fast execution with strict risk management
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    LimitOrderRequest,
    TrailingStopOrderRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError
from system_config import MOMENTUM_CONFIG

logger = logging.getLogger(__name__)


class MomentumExecutor:
    """Ultra-fast momentum trade execution with A10 advantage"""
    
    def __init__(self, alpaca_key: str, alpaca_secret: str, paper: bool = True):
        self.client = TradingClient(
            api_key=alpaca_key,
            secret_key=alpaca_secret,
            paper=paper
        )
        self.positions = {}
        self.pending_orders = {}
        self.account_value = 25000
        self.max_position_pct = 0.20  # 20% max per position
        self.trailing_stop_pct = 5.0   # 5% trailing stop
        
    async def execute_momentum_trade(self, symbol: str, momentum_score: float,
                                   current_price: float, bid: float, ask: float) -> bool:
        """Execute momentum trade with precise limit orders"""
        try:
            logger.info(f"🎯 EXECUTOR: Starting trade execution for {symbol}")
            
            # Check if already in position
            if symbol in self.positions:
                logger.warning(f"❌ EXECUTOR: {symbol} already in positions")
                return False
            
            # Get account status
            logger.info(f"📊 EXECUTOR: Getting account status...")
            account = self.client.get_account()
            cash = float(account.cash)
            buying_power = float(account.buying_power)
            
            logger.info(f"💰 EXECUTOR: Cash: ${cash:.2f}, Buying Power: ${buying_power:.2f}")
            
            # STRICT: Only use actual cash, never margin
            if cash <= 100:  # Keep $100 minimum
                logger.warning(f"❌ EXECUTOR: Insufficient cash: ${cash:.2f}")
                return False
            
            # Calculate position size based on momentum
            position_size = self._calculate_position_size(momentum_score, cash)
            logger.info(f"📈 EXECUTOR: Position size calculated: ${position_size:.2f}")
            
            if position_size < 500:  # Minimum $500 position
                logger.warning(f"❌ EXECUTOR: Position size ${position_size:.2f} below minimum $500")
                return False
            
            # Calculate shares (round down)
            shares = int(position_size / ask)
            logger.info(f"📊 EXECUTOR: Calculated shares: {shares} @ ${ask:.2f}")
            
            if shares < 1:
                logger.warning(f"❌ EXECUTOR: Less than 1 share calculated")
                return False
            
            # Final cost check
            total_cost = shares * ask
            logger.info(f"💵 EXECUTOR: Total cost: ${total_cost:.2f}")
            
            if total_cost > cash - 100:  # Keep $100 buffer
                shares = int((cash - 100) / ask)
                logger.info(f"⚠️ EXECUTOR: Adjusted shares due to cash buffer: {shares}")
                if shares < 1:
                    logger.warning(f"❌ EXECUTOR: Adjusted shares less than 1")
                    return False
            
            # Place limit order at ask price
            logger.info(f"📝 EXECUTOR: Placing limit order: {shares} shares @ ${ask:.2f}")
            
            limit_order = LimitOrderRequest(
                symbol=symbol,
                qty=shares,
                side=OrderSide.BUY,
                limit_price=ask,
                time_in_force=TimeInForce.IOC  # Immediate or Cancel
            )
            
            order = self.client.submit_order(limit_order)
            
            if order:
                self.pending_orders[symbol] = {
                    'order_id': order.id,
                    'shares': shares,
                    'limit_price': ask,
                    'momentum_score': momentum_score,
                    'timestamp': datetime.now()
                }
                
                logger.info(f"✅ LIMIT ORDER SUBMITTED: {symbol} {shares} @ ${ask:.2f} (Score: {momentum_score:.0f}) Order ID: {order.id}")
                
                # Schedule trailing stop placement
                asyncio.create_task(self._place_trailing_stop_after_fill(symbol, shares))
                
                return True
            else:
                logger.warning(f"❌ EXECUTOR: Order submission returned None")
                return False
            
        except APIError as e:
            logger.error(f"❌ EXECUTOR: Alpaca API error for {symbol}: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ EXECUTOR: Execution failed for {symbol}: {e}")
            return False
    
    def _calculate_position_size(self, momentum_score: float, available_cash: float) -> float:
        """Calculate position size based on momentum strength"""
        # Base allocation
        base_size = available_cash * 0.10  # 10% base
        
        # Momentum multiplier (using config values)
        if momentum_score >= MOMENTUM_CONFIG['extreme_momentum']:        # Extreme momentum
            multiplier = 2.5
        elif momentum_score >= MOMENTUM_CONFIG['strong_momentum']:      # Strong momentum
            multiplier = 2.0
        elif momentum_score >= MOMENTUM_CONFIG['min_score']:      # Minimum momentum
            multiplier = 1.5
        else:
            multiplier = 1.0  # Fallback (shouldn't happen with min_score+ filter)
        
        position_size = base_size * multiplier
        
        # Cap at max position size
        max_size = available_cash * self.max_position_pct
        position_size = min(position_size, max_size)
        
        return position_size
    
    async def _place_trailing_stop_after_fill(self, symbol: str, shares: int):
        """Place trailing stop after order fills"""
        await asyncio.sleep(1)  # Brief wait for fill
        
        try:
            # Check if position exists
            position = self.client.get_open_position(symbol)
            if position:
                actual_shares = int(position.qty)
                
                # Place trailing stop
                trailing_stop = TrailingStopOrderRequest(
                    symbol=symbol,
                    qty=actual_shares,
                    side=OrderSide.SELL,
                    trail_percent=self.trailing_stop_pct,
                    time_in_force=TimeInForce.GTC
                )
                
                stop_order = self.client.submit_order(trailing_stop)
                
                if stop_order:
                    logger.info(f"✅ TRAILING STOP: {symbol} {actual_shares} @ {self.trailing_stop_pct}%")
                    
                    self.positions[symbol] = {
                        'shares': actual_shares,
                        'entry_price': float(position.avg_entry_price),
                        'stop_order_id': stop_order.id,
                        'momentum_score': self.pending_orders[symbol]['momentum_score'],
                        'entry_time': datetime.now()
                    }
                    
                    del self.pending_orders[symbol]
                    
        except Exception as e:
            logger.error(f"Failed to place trailing stop for {symbol}: {e}")
    
    async def update_positions(self) -> Dict[str, Dict]:
        """Update all positions and check for exits"""
        try:
            alpaca_positions = self.client.get_all_positions()
            current_positions = {}
            
            for pos in alpaca_positions:
                symbol = pos.symbol
                current_positions[symbol] = {
                    'shares': int(pos.qty),
                    'avg_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price) if pos.current_price else 0,
                    'pnl': float(pos.unrealized_pl),
                    'pnl_pct': float(pos.unrealized_plpc) * 100
                }
            
            # Check for closed positions
            for symbol in list(self.positions.keys()):
                if symbol not in current_positions:
                    logger.info(f"🏁 POSITION CLOSED: {symbol}")
                    del self.positions[symbol]
            
            return current_positions
            
        except Exception as e:
            logger.error(f"Failed to update positions: {e}")
            return {}
    
    def get_available_cash(self) -> float:
        """Get available cash for new positions"""
        try:
            account = self.client.get_account()
            return float(account.cash)
        except:
            return 0
    
    def get_position_count(self) -> int:
        """Get current position count"""
        try:
            positions = self.client.get_all_positions()
            return len(positions)
        except:
            return 0