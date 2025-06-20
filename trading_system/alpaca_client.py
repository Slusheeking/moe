"""
Alpaca Trading Client for Momentum Execution
Optimized for fast execution with 5% trailing stops
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
    
    def __init__(self, api_key: str, secret_key: str, base_url: str, polygon_client=None, data_manager=None):
        self.api_key = api_key
        self.secret_key = secret_key
        # Determine if using paper trading
        self.paper = 'paper' in base_url.lower()
        self.client = None
        self.polygon_client = polygon_client  # For fallback pricing
        self.data_manager = data_manager  # For real-time quote data
        
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
        """Check if current time is within allowed trading hours (9:30 AM - 4:00 PM ET)"""
        now_utc = datetime.now(timezone.utc)
        
        # Check if weekend (Saturday = 5, Sunday = 6)
        if now_utc.weekday() >= 5:
            return False
        
        # Convert current time to minutes since midnight UTC
        current_minutes = now_utc.hour * 60 + now_utc.minute
        
        # Determine if we're in daylight saving time (approx March-November)
        # During DST (summer): 9:30 AM - 4:00 PM EDT = 13:30 - 20:00 UTC
        # During Standard (winter): 9:30 AM - 4:00 PM EST = 14:30 - 21:00 UTC
        month = now_utc.month
        is_dst = 3 <= month <= 11  # Rough DST period (March to November)
        
        if is_dst:
            # Daylight time: EDT (UTC-4)
            trading_start = 13 * 60 + 30  # 13:30 UTC = 810 minutes
            trading_end = 20 * 60 + 0     # 20:00 UTC = 1200 minutes
        else:
            # Standard time: EST (UTC-5)
            trading_start = 14 * 60 + 30  # 14:30 UTC = 870 minutes
            trading_end = 21 * 60 + 0     # 21:00 UTC = 1260 minutes
        
        return trading_start <= current_minutes <= trading_end
    
    async def _get_current_price_with_fallback(self, symbol: str) -> Optional[float]:
        """Get current price with Polygon fallback if Alpaca fails"""
        try:
            # First try Alpaca
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            
            data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            
            quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            latest_quote = data_client.get_stock_latest_quote(quote_request)
            
            if symbol in latest_quote:
                quote = latest_quote[symbol]
                
                # Try multiple Alpaca price sources
                if quote.ask_price and float(quote.ask_price) > 0:
                    return float(quote.ask_price)
                elif quote.bid_price and float(quote.bid_price) > 0:
                    logger.debug(f"Using Alpaca bid price for {symbol}")
                    return float(quote.bid_price)
                elif (quote.ask_price and quote.bid_price and
                      float(quote.ask_price) > 0 and float(quote.bid_price) > 0):
                    mid_price = (float(quote.ask_price) + float(quote.bid_price)) / 2
                    logger.debug(f"Using Alpaca mid-price for {symbol}")
                    return mid_price
            
            # Alpaca failed, try Polygon fallback
            if self.polygon_client:
                logger.info(f"Alpaca quote failed for {symbol}, trying Polygon fallback...")
                
                # Get current snapshot from Polygon
                snapshot = await self.polygon_client.get_snapshot_ticker(symbol)
                if snapshot:
                    # Try last trade price first
                    if snapshot.lastTrade and snapshot.lastTrade.p > 0:
                        price = snapshot.lastTrade.p
                        logger.info(f"Using Polygon last trade price ${price:.2f} for {symbol}")
                        return price
                    
                    # Fall back to previous day close
                    elif snapshot.prevDay and snapshot.prevDay.c > 0:
                        price = snapshot.prevDay.c
                        logger.info(f"Using Polygon previous close ${price:.2f} for {symbol}")
                        return price
                
                # Try previous close from Polygon
                prev_close = await self.polygon_client.get_previous_close(symbol)
                if prev_close and prev_close > 0:
                    logger.info(f"Using Polygon previous close ${prev_close:.2f} for {symbol}")
                    return prev_close
            
            logger.error(f"Could not get price for {symbol} from any source")
            return None
            
        except Exception as e:
            logger.error(f"Price retrieval failed for {symbol}: {e}")
            return None
    
    async def _check_sufficient_cash(self, symbol: str, qty: int, estimated_price: float) -> bool:
        """Check if account has sufficient ACTUAL CASH for the trade"""
        try:
            account = await self.get_account()
            if not account:
                logger.error("Cannot check cash - failed to get account info")
                return False
            
            cash = float(account.get('cash', 0))
            buying_power = float(account.get('buying_power', 0))
            
            # CRITICAL FIX: Use actual cash, not buying power, to prevent over-leveraging
            if cash < 0:
                logger.warning(f"Cash check FAILED - negative cash: ${cash:,.2f}")
                return False
            
            # Estimate trade cost (price * quantity + small buffer for slippage)
            estimated_cost = estimated_price * qty * 1.01  # 1% slippage buffer
            
            # Use actual cash for check, not buying power
            if cash >= estimated_cost:
                logger.debug(f"Cash check passed: ${cash:,.2f} cash available, ${estimated_cost:,.2f} needed for {qty} {symbol}")
                return True
            else:
                logger.warning(f"Insufficient CASH for {qty} {symbol}: ${cash:,.2f} cash, ${estimated_cost:,.2f} needed (buying_power: ${buying_power:,.2f})")
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
    
    async def get_available_cash(self) -> float:
        """Get available cash for new positions (CRITICAL for position sizing)"""
        try:
            account = await self.get_account()
            if account:
                cash = float(account.get('cash', 0))
                buying_power = float(account.get('buying_power', 0))
                
                # CRITICAL FIX: If cash is negative, we can't afford new positions
                if cash < 0:
                    logger.warning(f"Negative cash balance: ${cash:,.2f} - no new positions allowed")
                    return 0
                
                # Use actual cash, not buying power, to prevent over-leveraging
                available = cash
                logger.debug(f"Available cash: ${available:,.2f} (cash: ${cash:,.2f}, buying_power: ${buying_power:,.2f})")
                return available
            return 0
        except Exception as e:
            logger.error(f"Failed to get available cash: {e}")
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
            
            # Check trading hours (9:30 AM - 4:00 PM ET)
            if not self._is_within_trading_hours():
                logger.warning(f"Order rejected for {symbol} - outside trading hours (9:30 AM - 4:00 PM ET)")
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
                    trail_percent=trail_percent,  # Send as percentage (5.0 = 5%)
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
    
    async def place_profit_bracket_order(self, symbol: str, qty: int, take_profit_price: float,
                                        trail_percent: float = 5.0, partial_exit_qty: int = None) -> Optional[Dict]:
        """Place bracket order with take profit + trailing stop using official Alpaca bracket orders"""
        try:
            self._ensure_connected()
            
            # Check trading hours (9:30 AM - 4:00 PM ET)
            if not self._is_within_trading_hours():
                logger.warning(f"Profit bracket order rejected for {symbol} - outside trading hours")
                return None
            
            # Get current price using robust fallback system
            current_price = await self._get_current_price_with_fallback(symbol)
            if not current_price or current_price <= 0:
                logger.error(f"Could not get valid price for {symbol}")
                return None
            
            # Check if we have sufficient cash for this trade
            if not await self._check_sufficient_cash(symbol, qty, current_price):
                return None
            
            # Calculate stop price (5% below current price for trailing stop)
            stop_price = round(current_price * (1 - trail_percent / 100), 2)
            
            # Validate prices according to Alpaca requirements
            if take_profit_price <= current_price:
                logger.error(f"Take profit price ${take_profit_price:.2f} must be above current price ${current_price:.2f}")
                return None
                
            if stop_price >= current_price:
                logger.error(f"Stop price ${stop_price:.2f} must be below current price ${current_price:.2f}")
                return None
                
            if stop_price >= take_profit_price:
                logger.error(f"Stop price ${stop_price:.2f} must be below take profit ${take_profit_price:.2f}")
                return None
            
            # Create bracket order using official Alpaca SDK format
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderClass
            
            bracket_order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,  # GTC for bracket orders
                order_class=OrderClass.BRACKET,
                take_profit={'limit_price': take_profit_price},
                stop_loss={'stop_price': stop_price}
            )
            
            # Submit the bracket order
            result = self.client.submit_order(bracket_order_request)
            
            if result:
                logger.info(f"✅ BRACKET ORDER PLACED: BUY {qty} {symbol} @ ${current_price:.2f}")
                logger.info(f"   💰 Take Profit: ${take_profit_price:.2f} ({((take_profit_price/current_price-1)*100):.1f}%)")
                logger.info(f"   🛡️  Stop Loss: ${stop_price:.2f} ({trail_percent}%)")
                
                return self._order_to_dict(result)
            else:
                logger.error(f"❌ Bracket order failed for {symbol}")
                return None
                
        except APIError as e:
            logger.error(f"Alpaca API error for bracket order {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to place bracket order for {symbol}: {e}")
            return None

    async def place_bracket_order_with_targets(self, symbol: str, qty: int, stop_loss_price: float, take_profit_price: float) -> Optional[Dict]:
        """Place bracket order with 5% stop loss and 10% take profit using Alpaca bracket order format"""
        try:
            self._ensure_connected()
            
            # Check trading hours
            if not self._is_within_trading_hours():
                logger.warning(f"Bracket order rejected for {symbol} - outside trading hours")
                return None
            
            # Use current price derived from stop/take profit levels (more reliable than fresh fetch)
            # Since stop_loss_price = current_price * 0.95, we can reverse calculate
            estimated_current_price = stop_loss_price / 0.95
            
            # Check cash availability using estimated price
            if not await self._check_sufficient_cash(symbol, qty, estimated_current_price):
                return None
            
            # Validate bracket order prices with basic sanity checks
            if stop_loss_price >= take_profit_price:
                logger.error(f"Stop loss ${stop_loss_price:.2f} must be below take profit ${take_profit_price:.2f}")
                return None
            
            if stop_loss_price <= 0 or take_profit_price <= 0:
                logger.error(f"Invalid prices: stop_loss=${stop_loss_price:.2f}, take_profit=${take_profit_price:.2f}")
                return None
                
            # Create bracket order using Alpaca SDK format without problematic price validation
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderClass
            
            bracket_order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,  # GTC required for bracket orders
                order_class=OrderClass.BRACKET,
                take_profit={
                    'limit_price': take_profit_price
                },
                stop_loss={
                    'stop_price': stop_loss_price,
                    'limit_price': round(stop_loss_price - 0.05, 2)  # Stop-limit order with small buffer
                }
            )
            
            # Submit the bracket order
            result = self.client.submit_order(bracket_order_request)
            
            if result:
                # Calculate P&L percentages using estimated price
                pnl_gain = ((take_profit_price / estimated_current_price) - 1) * 100
                pnl_loss = ((stop_loss_price / estimated_current_price) - 1) * 100
                
                logger.info(f"✅ BRACKET ORDER PLACED: {symbol}")
                logger.info(f"   📊 BUY {qty} shares @ ~${estimated_current_price:.2f}")
                logger.info(f"   💰 Take Profit: ${take_profit_price:.2f} (+{pnl_gain:.1f}%)")
                logger.info(f"   🛡️  Stop Loss: ${stop_loss_price:.2f} ({pnl_loss:.1f}%)")
                
                return self._order_to_dict(result)
            else:
                logger.error(f"❌ Bracket order failed for {symbol}")
                return None
                
        except APIError as e:
            logger.error(f"Alpaca API error for bracket order {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to place bracket order for {symbol}: {e}")
            return None

    async def place_multiple_bracket_orders(self, order_specs: List[Dict]) -> List[Dict]:
        """Place multiple bracket orders concurrently for multi-stock execution"""
        try:
            self._ensure_connected()
            
            # Check trading hours once for all orders
            if not self._is_within_trading_hours():
                logger.warning("Multiple bracket orders rejected - outside trading hours")
                return []
            
            logger.info(f"📋 Placing {len(order_specs)} bracket orders concurrently...")
            
            # Create tasks for concurrent order placement
            order_tasks = []
            for spec in order_specs:
                task = self.place_bracket_order_with_targets(
                    symbol=spec['symbol'],
                    qty=spec['qty'],
                    stop_loss_price=spec['stop_loss_price'],
                    take_profit_price=spec['take_profit_price']
                )
                order_tasks.append(task)
            
            # Execute all orders concurrently with timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*order_tasks, return_exceptions=True),
                    timeout=60  # 60 second timeout for all orders
                )
            except asyncio.TimeoutError:
                logger.error("Timeout placing multiple bracket orders")
                return []
            
            # Process results
            successful_orders = []
            failed_count = 0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Order {i+1} failed: {result}")
                    failed_count += 1
                elif result is not None:
                    successful_orders.append(result)
                else:
                    failed_count += 1
            
            logger.info(f"✅ Multi-order result: {len(successful_orders)} successful, {failed_count} failed")
            return successful_orders
            
        except Exception as e:
            logger.error(f"Failed to place multiple bracket orders: {e}")
            return []

    async def get_multiple_positions_status(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get status for multiple positions efficiently"""
        try:
            self._ensure_connected()
            
            # Get all positions at once
            all_positions = await self.list_positions()
            
            # Filter for requested symbols
            position_status = {}
            for symbol in symbols:
                position_status[symbol] = None
                
                # Find position for this symbol
                for pos in all_positions:
                    if pos['symbol'] == symbol:
                        position_status[symbol] = pos
                        break
            
            return position_status
            
        except Exception as e:
            logger.error(f"Failed to get multiple positions status: {e}")
            return {}

    async def get_account_utilization(self) -> Dict:
        """Get detailed account utilization for multi-stock trading"""
        try:
            account = await self.get_account()
            if not account:
                return {}
            
            positions = await self.list_positions()
            
            cash = float(account.get('cash', 0))
            buying_power = float(account.get('buying_power', 0))
            portfolio_value = float(account.get('portfolio_value', 0))
            
            # Calculate position values
            total_position_value = sum(float(pos.get('market_value', 0)) for pos in positions)
            
            # Calculate utilization ratios
            cash_utilization = (portfolio_value - cash) / portfolio_value if portfolio_value > 0 else 0
            position_count = len(positions)
            
            return {
                'cash': cash,
                'buying_power': buying_power,
                'portfolio_value': portfolio_value,
                'total_position_value': total_position_value,
                'cash_utilization_pct': cash_utilization * 100,
                'position_count': position_count,
                'available_cash_pct': (cash / portfolio_value * 100) if portfolio_value > 0 else 0,
                'positions': [{
                    'symbol': pos['symbol'],
                    'value': float(pos.get('market_value', 0)),
                    'pnl_pct': float(pos.get('unrealized_plpc', 0)) * 100
                } for pos in positions]
            }
            
        except Exception as e:
            logger.error(f"Failed to get account utilization: {e}")
            return {}

    async def place_bracket_order(self, symbol: str, qty: int, trail_percent: float = 5.0,
                                 take_profit_price: float = None) -> Optional[Dict]:
        """Place proper bracket order with stop loss using official Alpaca format"""
        try:
            self._ensure_connected()
            
            # Check trading hours (9:30 AM - 4:00 PM ET)
            if not self._is_within_trading_hours():
                logger.warning(f"Bracket order rejected for {symbol} - outside trading hours (9:30 AM - 4:00 PM ET)")
                return None
            
            # Get current price using robust fallback system
            current_price = await self._get_current_price_with_fallback(symbol)
            if not current_price or current_price <= 0:
                logger.error(f"Could not get valid price for {symbol}")
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
                
            # For momentum trading, use simple market order + separate trailing stop
            # This avoids take profit complications and lets winners run
            
            # First, place the market buy order
            market_order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            
            # Submit the market order
            buy_result = self.client.submit_order(market_order_request)
            
            if not buy_result:
                logger.error(f"Failed to place market order for {symbol}")
                return None
            
            logger.info(f"   MARKET ORDER PLACED: BUY {qty} {symbol} @ ${current_price:.2f}")
            
            # CRITICAL FIX: Wait much longer for position to settle and retry trailing stop
            logger.info(f"   Waiting for {symbol} position to settle before placing trailing stop...")
            await asyncio.sleep(5.0)  # Increased wait time
            
            # Retry trailing stop placement with better error handling
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Verify position exists before placing stop
                    position = await self.get_position(symbol)
                    if position:
                        actual_qty = int(float(position['qty']))
                        logger.info(f"   Attempt {attempt + 1}: Position confirmed, placing trailing stop for {actual_qty} shares")
                        
                        # Use the actual position quantity, not the original order quantity
                        trailing_stop_result = await self.place_trailing_stop(symbol, actual_qty, trail_percent)
                        
                        if trailing_stop_result:
                            logger.info(f"   ✅ TRAILING STOP PLACED: {trail_percent}% for {symbol}")
                            break
                        else:
                            logger.warning(f"   Attempt {attempt + 1} failed, retrying...")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2.0)
                    else:
                        logger.warning(f"   Attempt {attempt + 1}: Position not found yet, retrying...")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(3.0)
                        
                except Exception as e:
                    logger.error(f"   Attempt {attempt + 1} trailing stop error: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2.0)
            else:
                logger.error(f"   ❌ ALL TRAILING STOP ATTEMPTS FAILED for {symbol}")
            
            return self._order_to_dict(buy_result)
                
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
        """Place trailing stop order that tracks 5% from highest price (high water mark)"""
        try:
            self._ensure_connected()
            
            # Verify we actually have a position first
            position = await self.get_position(symbol)
            if not position:
                logger.error(f"Cannot place trailing stop for {symbol} - no position found")
                return None
            
            # Use actual position quantity
            actual_qty = int(float(position['qty']))
            if actual_qty <= 0:
                logger.error(f"Invalid position quantity {actual_qty} for {symbol}")
                return None
            
            logger.info(f"Placing trailing stop for {symbol}: position_qty={actual_qty}")
            
            # Debug the trail_percent value being sent
            logger.info(f"Trail percent for {symbol}: input={trail_percent}, sending to Alpaca={trail_percent}")
            
            # Validate trail_percent (Alpaca minimum is 0.05% = 0.05, we use 5.0)
            if trail_percent < 0.05:
                logger.error(f"Trail percent {trail_percent}% below Alpaca minimum 0.05%")
                return None
            
            # Create trailing stop order - tracks from HIGH WATER MARK (highest price achieved)
            # Alpaca expects trail_percent as percentage value (5.0 = 5%, not 0.05)
            trailing_stop_request = TrailingStopOrderRequest(
                symbol=symbol,
                qty=actual_qty,  # Use actual position quantity
                side=OrderSide.SELL,
                trail_percent=trail_percent,  # Send as percentage (5.0 = 5%)
                time_in_force=TimeInForce.GTC  # Good Till Cancelled for persistence
            )
            
            result = self.client.submit_order(trailing_stop_request)
            
            if result:
                logger.info(f"✅ TRAILING STOP PLACED: {actual_qty} {symbol} @ {trail_percent}% from HIGH WATER MARK")
                return self._order_to_dict(result)
            else:
                logger.error(f"❌ Trailing stop order failed for {symbol}")
                return None
                
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
    
    async def get_latest_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time bid/ask quote from live Polygon websocket stream"""
        try:
            # First try to get real-time quote from data manager (Polygon stream)
            if self.data_manager:
                live_quote = self.data_manager.get_latest_quote(symbol)
                if live_quote:
                    # Validate bid/ask data from live stream
                    bid_price = live_quote.get('bid_price')
                    ask_price = live_quote.get('ask_price')
                    
                    if bid_price and ask_price and bid_price > 0 and ask_price > 0 and bid_price < ask_price:
                        logger.debug(f"✅ Live quote: {symbol} bid=${bid_price:.2f} ask=${ask_price:.2f} spread={live_quote.get('spread_pct', 0)*100:.2f}%")
                        return {
                            'symbol': symbol,
                            'bid_price': bid_price,
                            'ask_price': ask_price,
                            'bid_size': live_quote.get('bid_size', 0),
                            'ask_size': live_quote.get('ask_size', 0),
                            'timestamp': live_quote.get('timestamp').isoformat() if live_quote.get('timestamp') else None,
                            'spread_pct': live_quote.get('spread_pct', 0)
                        }
                    else:
                        logger.warning(f"Invalid live quote data for {symbol}: bid={bid_price}, ask={ask_price}")
                else:
                    logger.warning(f"No live quote available for {symbol} from stream")
            
            # Fallback to Alpaca historical data client (less reliable for real-time)
            logger.info(f"Falling back to Alpaca quote for {symbol}")
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            
            data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            
            quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            latest_quote = data_client.get_stock_latest_quote(quote_request)
            
            if symbol in latest_quote:
                quote = latest_quote[symbol]
                
                # Validate that we have valid bid/ask prices - reject if any are None or 0
                bid_price = float(quote.bid_price) if quote.bid_price and float(quote.bid_price) > 0 else None
                ask_price = float(quote.ask_price) if quote.ask_price and float(quote.ask_price) > 0 else None
                
                if bid_price is None or ask_price is None:
                    logger.error(f"Invalid Alpaca bid/ask data for {symbol}: bid={quote.bid_price}, ask={quote.ask_price}")
                    return None
                
                return {
                    'symbol': symbol,
                    'bid_price': bid_price,
                    'ask_price': ask_price,
                    'bid_size': int(quote.bid_size) if quote.bid_size else 0,
                    'ask_size': int(quote.ask_size) if quote.ask_size else 0,
                    'timestamp': quote.timestamp.isoformat() if quote.timestamp else None
                }
            
            logger.error(f"No quote data available for {symbol} from any source")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get latest quote for {symbol}: {e}")
            return None
    
    async def place_limit_order_with_bracket(self, symbol: str, qty: int, limit_price: float, 
                                           stop_loss_price: float, take_profit_price: float,
                                           timeout_seconds: int = 30) -> Optional[Dict]:
        """Place limit order with bracket (stop loss + take profit) - STRICT PRICE PROTECTION"""
        try:
            self._ensure_connected()
            
            # Check trading hours
            if not self._is_within_trading_hours():
                logger.error(f"❌ BLOCKED: Order outside trading hours for {symbol}")
                return None
            
            # Validate all prices
            if limit_price <= 0 or stop_loss_price <= 0 or take_profit_price <= 0:
                logger.error(f"❌ BLOCKED: Invalid prices for {symbol}: limit=${limit_price}, stop=${stop_loss_price}, take=${take_profit_price}")
                return None
            
            if stop_loss_price >= limit_price:
                logger.error(f"❌ BLOCKED: Stop loss ${stop_loss_price} must be below limit ${limit_price}")
                return None
                
            if take_profit_price <= limit_price:
                logger.error(f"❌ BLOCKED: Take profit ${take_profit_price} must be above limit ${limit_price}")
                return None
            
            # Check cash before placing order
            if not await self._check_sufficient_cash(symbol, qty, limit_price):
                logger.error(f"❌ BLOCKED: Insufficient cash for {symbol}")
                return None
            
            # Create bracket limit order - NEVER market order
            from alpaca.trading.requests import LimitOrderRequest
            from alpaca.trading.enums import OrderClass
            
            bracket_limit_request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                limit_price=limit_price,  # EXACT LIMIT PRICE - NO SLIPPAGE
                time_in_force=TimeInForce.DAY,  # Day order to avoid hanging
                order_class=OrderClass.BRACKET,
                take_profit={
                    'limit_price': take_profit_price
                },
                stop_loss={
                    'stop_price': stop_loss_price,
                    'limit_price': round(stop_loss_price * 0.98, 2)  # Stop-limit with small buffer
                }
            )
            
            # Submit the bracket limit order
            logger.info(f"🎯 SUBMITTING LIMIT BRACKET: {symbol} @ ${limit_price:.2f} (EXACT PRICE)")
            result = self.client.submit_order(bracket_limit_request)
            
            if result:
                logger.info(f"✅ LIMIT BRACKET PLACED: {symbol}")
                logger.info(f"   💰 LIMIT BUY: ${limit_price:.2f} (NO SLIPPAGE ALLOWED)")
                logger.info(f"   🛡️  Stop Loss: ${stop_loss_price:.2f}")
                logger.info(f"   💰 Take Profit: ${take_profit_price:.2f}")
                logger.info(f"   ⏰ Timeout: {timeout_seconds}s")
                
                return self._order_to_dict(result)
            else:
                logger.error(f"❌ LIMIT BRACKET ORDER FAILED: {symbol}")
                return None
                
        except APIError as e:
            logger.error(f"❌ Alpaca API error for limit bracket {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to place limit bracket order for {symbol}: {e}")
            return None