"""
Alpaca Trading Client for Momentum Execution
Optimized for fast execution with 10% trailing stops (Alpaca minimum)
Uses Official Alpaca Python SDK
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    TrailingStopOrderRequest,
    StopOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
    GetOrdersRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
from alpaca.common.exceptions import APIError

logger = logging.getLogger(__name__)


class AlpacaClient:
    """Alpaca client using official SDK for momentum trading execution"""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self.api_key = api_key
        self.secret_key = secret_key
        # Determine if using paper trading
        self.paper = 'paper' in base_url.lower()
        self.client = None
        
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self):
        """Initialize Alpaca Trading Client"""
        if not self.client:
            self.client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper
            )
            logger.info("🔗 Alpaca SDK client connected")
    
    async def disconnect(self):
        """Clean up client connection"""
        if self.client:
            self.client = None
            logger.info("🔗 Alpaca SDK client disconnected")
    
    def _ensure_connected(self):
        """Ensure client is connected"""
        if not self.client:
            # For sync methods, we need to connect synchronously
            self.client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper
            )
    
    def _is_within_trading_hours(self) -> bool:
        """Check if current time is within allowed trading hours (9:40 AM - 3:50 PM EST = 14:40 - 20:50 UTC)"""
        now_utc = datetime.now(timezone.utc)
        
        # Check if weekend (Saturday = 5, Sunday = 6)
        if now_utc.weekday() >= 5:
            return False
        
        # Convert current time to minutes since midnight UTC
        current_minutes = now_utc.hour * 60 + now_utc.minute
        
        # Trading hours in UTC: 14:40 (9:40 AM EST) to 20:50 (3:50 PM EST)
        trading_start = 14 * 60 + 40  # 14:40 UTC = 880 minutes
        trading_end = 20 * 60 + 50    # 20:50 UTC = 1250 minutes
        
        return trading_start <= current_minutes <= trading_end
    
    async def _check_sufficient_cash(self, symbol: str, qty: int, estimated_price: float) -> bool:
        """Check if account has sufficient cash for the trade"""
        try:
            account = await self.get_account()
            if not account:
                logger.error("Cannot check cash - failed to get account info")
                return False
            
            available_cash = float(account.get('cash', 0))
            buying_power = float(account.get('buying_power', 0))
            
            # Estimate trade cost (price * quantity + small buffer for slippage)
            estimated_cost = estimated_price * qty * 1.01  # 1% slippage buffer
            
            # Use buying power for check (includes margin if available)
            if buying_power >= estimated_cost:
                logger.debug(f"Cash check passed: ${buying_power:,.2f} available, ${estimated_cost:,.2f} needed for {qty} {symbol}")
                return True
            else:
                logger.warning(f"Insufficient funds for {qty} {symbol}: ${buying_power:,.2f} available, ${estimated_cost:,.2f} needed")
                return False
                
        except Exception as e:
            logger.error(f"Cash check failed for {symbol}: {e}")
            return False
    
    async def get_account(self) -> Optional[Dict]:
        """Get account information"""
        try:
            self._ensure_connected()
            account = self.client.get_account()
            return {
                'id': account.id,
                'account_number': account.account_number,
                'status': account.status,
                'currency': account.currency,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'last_equity': float(account.last_equity)
            }
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return None
    
    async def get_account_value(self) -> float:
        """Get current account portfolio value"""
        try:
            account = await self.get_account()
            if account:
                return float(account.get('portfolio_value', 0))
            return 0
        except Exception as e:
            logger.error(f"Failed to get account value: {e}")
            return 0
    
    async def list_positions(self) -> List[Dict]:
        """Get all current positions"""
        try:
            self._ensure_connected()
            positions = self.client.get_all_positions()
            return [self._position_to_dict(pos) for pos in positions]
        except Exception as e:
            logger.error(f"Failed to list positions: {e}")
            return []
    
    async def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for specific symbol"""
        try:
            self._ensure_connected()
            position = self.client.get_open_position(symbol)
            return self._position_to_dict(position)
        except Exception as e:
            logger.debug(f"No position found for {symbol}")
            return None
    
    def _position_to_dict(self, position) -> Dict:
        """Convert position object to dictionary"""
        return {
            'symbol': position.symbol,
            'qty': str(position.qty),
            'side': position.side.value,
            'market_value': str(position.market_value),
            'avg_entry_price': str(position.avg_entry_price),
            'unrealized_pl': str(position.unrealized_pl),
            'unrealized_plpc': str(position.unrealized_plpc),
            'current_price': str(position.current_price) if position.current_price else '0'
        }
    
    async def place_order(self, symbol: str, qty: int, side: str,
                         type: str = 'market', limit_price: float = None,
                         stop_price: float = None, time_in_force: str = 'day',
                         trail_percent: float = None) -> Optional[Dict]:
        """Place order for momentum trading using official SDK"""
        try:
            self._ensure_connected()
            
            # Check trading hours (9:40 AM - 3:50 PM EST = 14:40 - 20:50 UTC)
            if not self._is_within_trading_hours():
                logger.warning(f"Order rejected for {symbol} - outside trading hours (14:40-20:50 UTC)")
                return None
            
            # Check cash for BUY orders only
            if side.lower() == 'buy':
                # Get estimated price for cash check
                estimated_price = limit_price if limit_price else None
                
                # If no limit price provided, get current market price
                if not estimated_price:
                    try:
                        from alpaca.data.historical import StockHistoricalDataClient
                        from alpaca.data.requests import StockLatestQuoteRequest
                        
                        data_client = StockHistoricalDataClient(
                            api_key=self.api_key,
                            secret_key=self.secret_key
                        )
                        
                        quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                        latest_quote = data_client.get_stock_latest_quote(quote_request)
                        
                        if symbol in latest_quote:
                            estimated_price = float(latest_quote[symbol].ask_price)
                        else:
                            logger.error(f"Cannot get price for cash check: {symbol}")
                            return None
                            
                    except Exception as e:
                        logger.error(f"Failed to get price for cash check {symbol}: {e}")
                        return None
                
                # Check if we have sufficient cash
                if not await self._check_sufficient_cash(symbol, qty, estimated_price):
                    return None
            
            # Convert string parameters to enums
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            time_in_force_enum = TimeInForce.DAY if time_in_force.lower() == 'day' else TimeInForce.GTC
            
            # Create appropriate order request based on type
            if type.lower() == 'market':
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=time_in_force_enum
                )
            elif type.lower() == 'limit' and limit_price:
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    limit_price=limit_price,
                    time_in_force=time_in_force_enum
                )
            elif type.lower() == 'stop' and stop_price:
                order_request = StopOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    stop_price=stop_price,
                    time_in_force=time_in_force_enum
                )
            elif type.lower() == 'trailing_stop' and trail_percent:
                order_request = TrailingStopOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    trail_percent=trail_percent / 100,  # SDK expects decimal (0.05 for 5%)
                    time_in_force=time_in_force_enum
                )
            else:
                logger.error(f"Invalid order type or missing parameters: {type}")
                return None
            
            # Submit order
            result = self.client.submit_order(order_request)
            
            logger.info(f"📋 Order placed: {side.upper()} {qty} {symbol} @ {type}")
            return self._order_to_dict(result)
                
        except APIError as e:
            logger.error(f"Alpaca API error for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to place order for {symbol}: {e}")
            return None
    
    async def place_bracket_order(self, symbol: str, qty: int, trail_percent: float = 10.0,
                                 take_profit_price: float = None) -> Optional[Dict]:
        """Place proper bracket order with stop loss using official Alpaca format"""
        try:
            self._ensure_connected()
            
            # Check trading hours (9:40 AM - 3:50 PM EST = 14:40 - 20:50 UTC)
            if not self._is_within_trading_hours():
                logger.warning(f"Bracket order rejected for {symbol} - outside trading hours (14:40-20:50 UTC)")
                return None
            
            # Get current market price to calculate stop price for trailing stop
            # For bracket orders, we need a stop_price even for trailing stops
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            
            try:
                # Get current price for stop calculation
                data_client = StockHistoricalDataClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key
                )
                
                quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                latest_quote = data_client.get_stock_latest_quote(quote_request)
                
                if symbol in latest_quote:
                    current_price = float(latest_quote[symbol].ask_price)
                    if current_price <= 0:
                        logger.error(f"Invalid price {current_price} for {symbol}")
                        return None
                    
                    # Check if we have sufficient cash for this trade
                    if not await self._check_sufficient_cash(symbol, qty, current_price):
                        return None
                    
                    # Calculate initial stop price (will trail from here)
                    initial_stop_price = round(current_price * (1 - trail_percent / 100), 2)
                    
                    # Validate stop price
                    if initial_stop_price <= 0:
                        logger.error(f"Invalid stop price {initial_stop_price} for {symbol} (current: {current_price}, trail: {trail_percent}%)")
                        return None
                        
                    # Calculate take profit if not provided (2:1 reward/risk ratio)
                    if not take_profit_price:
                        risk_amount = current_price - initial_stop_price
                        take_profit_price = round(current_price + (risk_amount * 2), 2)
                        
                else:
                    logger.error(f"No quote data available for {symbol}")
                    return None
                    
            except Exception as e:
                logger.error(f"Failed to get current price for {symbol}: {e}")
                return None
            
            # Create proper bracket order with trailing stop
            from alpaca.trading.requests import StopLossRequest, TakeProfitRequest
            
            # Validate trail_percent is within Alpaca limits (minimum 0.1% = 0.001)
            if trail_percent < 0.1:
                logger.error(f"Trail percent {trail_percent}% below Alpaca minimum 0.1%")
                return None
            
            # Create stop loss with trailing percent
            stop_loss = StopLossRequest(
                stop_price=initial_stop_price,  # Required field
                trail_percent=trail_percent / 100  # This makes it a trailing stop
            )
            
            # Always include take profit for bracket orders (Alpaca requirement)
            take_profit = TakeProfitRequest(limit_price=take_profit_price)
            
            # Create bracket order request
            bracket_order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            result = self.client.submit_order(bracket_order_request)
            
            if result:
                logger.info(f"   BRACKET ORDER PLACED: BUY {qty} {symbol}")
                logger.info(f"   Entry: Market @ ${current_price}")
                logger.info(f"   Stop Loss: {trail_percent}% trailing @ ${initial_stop_price}")
                logger.info(f"   Take Profit: @ ${take_profit_price}")
                return self._order_to_dict(result)
            else:
                logger.error(f"Bracket order failed: {qty} {symbol}")
                return None
                
        except APIError as e:
            logger.error(f"Alpaca API error for bracket order {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to place bracket order for {symbol}: {e}")
            return None
    
    def _order_to_dict(self, order) -> Dict:
        """Convert order object to dictionary"""
        return {
            'id': order.id,
            'client_order_id': order.client_order_id,
            'symbol': order.symbol,
            'qty': str(order.qty),
            'side': order.side.value,
            'order_type': order.order_type.value,
            'time_in_force': order.time_in_force.value,
            'status': order.status.value,
            'created_at': order.created_at.isoformat() if order.created_at else None
        }
    
    async def place_trailing_stop(self, symbol: str, qty: int, trail_percent: float) -> Optional[Dict]:
        """Place trailing stop order using official Alpaca format"""
        try:
            self._ensure_connected()
            
            # According to Alpaca docs, minimum trail_percent is 0.1 (10%)
            if trail_percent < 10.0:
                logger.error(f"Trail percent {trail_percent}% below Alpaca minimum 10%")
                return None
            
            # Use the correct trailing stop format from Alpaca documentation
            trailing_stop_request = TrailingStopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                trail_percent=trail_percent / 100,  # Convert to decimal (10% = 0.1)
                time_in_force=TimeInForce.DAY  # Use DAY as recommended in docs
            )
            
            result = self.client.submit_order(trailing_stop_request)
            
            logger.info(f"Trailing stop placed: {qty} {symbol} @ {trail_percent}% trail")
            return self._order_to_dict(result)
                
        except APIError as e:
            logger.error(f"Alpaca API error for trailing stop {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to place trailing stop for {symbol}: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel specific order using official SDK"""
        try:
            self._ensure_connected()
            self.client.cancel_order_by_id(order_id)
            logger.info(f"Cancelled order: {order_id}")
            return True
        except APIError as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: str = None) -> int:
        """Cancel all orders, optionally for specific symbol using official SDK"""
        try:
            self._ensure_connected()
            
            if symbol:
                # Get orders for specific symbol first, then cancel them
                get_orders_request = GetOrdersRequest(
                    status=QueryOrderStatus.OPEN,
                    symbols=[symbol]
                )
                orders = self.client.get_orders(get_orders_request)
                cancelled_count = 0
                
                for order in orders:
                    try:
                        self.client.cancel_order_by_id(order.id)
                        cancelled_count += 1
                    except:
                        continue
                        
                if cancelled_count > 0:
                    logger.info(f"Cancelled {cancelled_count} orders for {symbol}")
                return cancelled_count
            else:
                # Cancel all orders
                cancelled_orders = self.client.cancel_orders()
                cancelled_count = len(cancelled_orders) if cancelled_orders else 0
                if cancelled_count > 0:
                    logger.info(f"Cancelled {cancelled_count} orders")
                return cancelled_count
            
        except APIError as e:
            logger.error(f"Failed to cancel orders: {e}")
            return 0
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
            return 0
    
    async def list_orders(self, status: str = 'open', limit: int = 50) -> List[Dict]:
        """List orders with optional status filter using official SDK"""
        try:
            self._ensure_connected()
            
            # Convert status string to enum
            if status.lower() == 'open':
                status_enum = QueryOrderStatus.OPEN
            elif status.lower() == 'closed':
                status_enum = QueryOrderStatus.CLOSED
            else:
                status_enum = QueryOrderStatus.ALL
            
            get_orders_request = GetOrdersRequest(
                status=status_enum,
                limit=limit
            )
            
            orders = self.client.get_orders(get_orders_request)
            return [self._order_to_dict(order) for order in orders]
            
        except APIError as e:
            logger.error(f"Failed to list orders: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to list orders: {e}")
            return []
    
    async def is_market_open(self) -> bool:
        """Check if market is currently open using official SDK"""
        try:
            self._ensure_connected()
            clock = self.client.get_clock()
            return clock.is_open
        except APIError as e:
            logger.error(f"Failed to check market status: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to check market status: {e}")
            return False
    
    async def get_account_portfolio_history(self, period: str = '1D') -> Optional[Dict]:
        """Get portfolio performance history using Alpaca trading client"""
        try:
            self._ensure_connected()
            
            # Use trading client's portfolio history method
            portfolio_history = self.client.get_portfolio_history(
                period=period,
                timeframe='1H'  # Hourly timeframe for detailed tracking
            )
            
            if portfolio_history:
                return {
                    'timestamp': [t.isoformat() for t in portfolio_history.timestamp],
                    'equity': [float(e) for e in portfolio_history.equity],
                    'profit_loss': [float(pl) for pl in portfolio_history.profit_loss] if portfolio_history.profit_loss else [],
                    'profit_loss_pct': [float(plp) for plp in portfolio_history.profit_loss_pct] if portfolio_history.profit_loss_pct else [],
                    'base_value': float(portfolio_history.base_value) if portfolio_history.base_value else 0,
                    'timeframe': portfolio_history.timeframe,
                    'next_page_token': portfolio_history.next_page_token
                }
            else:
                logger.warning("No portfolio history data available")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get portfolio history: {e}")
            return None