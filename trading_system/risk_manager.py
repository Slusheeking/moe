"""
Risk Manager for Momentum Trading
Focused on protecting capital during explosive momentum plays
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from trading_system import config

logger = logging.getLogger(__name__)


class RiskManager:
    """Risk management for momentum trading with 5% trailing stops"""
    
    def __init__(self, alpaca_client):
        self.alpaca = alpaca_client
        self.daily_pnl = 0
        self.daily_trades = 0
        self.session_start = datetime.now()
        self.max_daily_loss_hit = False
        
    async def check_entry_risk(self, symbol: str, position_size: float, 
                              momentum_score: float, account_value: float) -> Dict:
        """Check if new position meets risk criteria"""
        try:
            # Get current positions
            positions = await self.alpaca.list_positions()
            
            # Calculate current metrics
            total_positions = len(positions)
            total_exposure = sum(float(pos['market_value']) for pos in positions)
            exposure_pct = total_exposure / account_value if account_value > 0 else 0
            
            # Check daily loss limit
            if self.max_daily_loss_hit:
                return {
                    'approved': False,
                    'reason': 'Daily loss limit reached',
                    'risk_score': 100
                }
            
            # Check maximum positions
            if total_positions >= config.RISK_PARAMS['max_positions']:
                return {
                    'approved': False,
                    'reason': f'Max positions reached ({config.RISK_PARAMS["max_positions"]})',
                    'risk_score': 90
                }
            
            # Check position size
            position_pct = position_size / account_value if account_value > 0 else 0
            max_position_pct = config.RISK_PARAMS['max_position_size_pct']
            
            if position_pct > max_position_pct:
                return {
                    'approved': False,
                    'reason': f'Position too large ({position_pct:.1%} > {max_position_pct:.1%})',
                    'risk_score': 85
                }
            
            # Check minimum position size
            if position_size < config.RISK_PARAMS['min_position_size']:
                return {
                    'approved': False,
                    'reason': f'Position too small (${position_size:.0f} < ${config.RISK_PARAMS["min_position_size"]})',
                    'risk_score': 0
                }
            
            # Check if already have position in this symbol
            existing_position = None
            for pos in positions:
                if pos['symbol'] == symbol:
                    existing_position = pos
                    break
            
            if existing_position:
                return {
                    'approved': False,
                    'reason': f'Already have position in {symbol}',
                    'risk_score': 70
                }
            
            # Calculate risk score based on momentum and exposure
            risk_score = self._calculate_entry_risk_score(
                momentum_score, position_pct, exposure_pct, total_positions
            )
            
            # Approve if risk score is acceptable
            max_risk_score = 60  # Threshold for approval
            approved = risk_score <= max_risk_score
            
            return {
                'approved': approved,
                'reason': 'Risk acceptable' if approved else f'Risk too high (score: {risk_score})',
                'risk_score': risk_score,
                'position_pct': position_pct,
                'total_exposure_pct': exposure_pct,
                'momentum_score': momentum_score
            }
            
        except Exception as e:
            logger.error(f"Risk check failed for {symbol}: {e}")
            return {
                'approved': False,
                'reason': 'Risk check error',
                'risk_score': 100
            }
    
    def _calculate_entry_risk_score(self, momentum_score: float, position_pct: float,
                                   exposure_pct: float, total_positions: int) -> float:
        """Calculate numerical risk score (0-100, lower is better)"""
        risk_score = 0
        
        # Momentum quality (lower momentum = higher risk)
        if momentum_score >= 80:
            risk_score += 5   # Very strong momentum = low risk
        elif momentum_score >= 60:
            risk_score += 15  # Strong momentum = moderate risk
        elif momentum_score >= 40:
            risk_score += 30  # Moderate momentum = higher risk
        else:
            risk_score += 50  # Weak momentum = high risk
        
        # Position sizing risk
        if position_pct > 0.10:  # >10% position
            risk_score += 20
        elif position_pct > 0.05:  # >5% position
            risk_score += 10
        
        # Portfolio exposure risk
        if exposure_pct > 0.80:  # >80% exposed
            risk_score += 25
        elif exposure_pct > 0.60:  # >60% exposed
            risk_score += 15
        elif exposure_pct > 0.40:  # >40% exposed
            risk_score += 10
        
        # Position count risk
        position_count_risk = max(0, total_positions - 10) * 3  # 3 points per position above 10
        risk_score += position_count_risk
        
        return min(risk_score, 100)
    
    async def check_daily_limits(self) -> Dict:
        """Check if daily risk limits have been breached"""
        try:
            # Calculate daily P&L
            account = await self.alpaca.get_account()
            if not account:
                return {'within_limits': False, 'reason': 'Cannot access account'}
            
            # Get daily P&L
            day_trade_buying_power = float(account.get('daytrading_buying_power', 0))
            portfolio_value = float(account.get('portfolio_value', 0))
            
            # Calculate daily loss percentage
            equity = float(account.get('equity', portfolio_value))
            daily_change = equity - config.ACCOUNT_SIZE
            daily_change_pct = daily_change / config.ACCOUNT_SIZE if config.ACCOUNT_SIZE > 0 else 0
            
            # Check daily loss limit
            max_daily_loss_pct = config.RISK_PARAMS['max_daily_loss_pct']
            if daily_change_pct <= -max_daily_loss_pct:
                self.max_daily_loss_hit = True
                logger.warning(f"Daily loss limit hit: {daily_change_pct:.1%}")
                return {
                    'within_limits': False,
                    'reason': f'Daily loss limit exceeded ({daily_change_pct:.1%})',
                    'daily_pnl_pct': daily_change_pct
                }
            
            # All limits OK
            return {
                'within_limits': True,
                'daily_pnl_pct': daily_change_pct,
                'account_value': portfolio_value
            }
            
        except Exception as e:
            logger.error(f"Daily limits check failed: {e}")
            return {'within_limits': False, 'reason': 'Limits check error'}
    
    async def validate_exit_conditions(self, symbol: str, current_price: float, 
                                     position_data: Dict) -> Dict:
        """Validate exit conditions for position"""
        try:
            entry_price = position_data.get('entry_price', 0)
            highest_price = position_data.get('highest_price', entry_price)
            trailing_stop_price = position_data.get('trailing_stop_price', 0)
            
            if entry_price <= 0 or current_price <= 0:
                return {'should_exit': False, 'reason': 'Invalid price data'}
            
            # Calculate current P&L
            current_pnl_pct = (current_price - entry_price) / entry_price
            
            # Check trailing stop (5% from high)
            trailing_stop_pct = config.RISK_PARAMS['trailing_stop_pct']
            stop_price = highest_price * (1 - trailing_stop_pct / 100)
            
            if current_price <= stop_price:
                return {
                    'should_exit': True,
                    'reason': 'trailing_stop',
                    'exit_type': 'trailing_stop',
                    'current_pnl_pct': current_pnl_pct,
                    'stop_price': stop_price
                }
            
            # Emergency stop loss (circuit breaker)
            emergency_stop_pct = 0.15  # 15% emergency stop
            if current_pnl_pct <= -emergency_stop_pct:
                return {
                    'should_exit': True,
                    'reason': 'emergency_stop',
                    'exit_type': 'emergency_stop',
                    'current_pnl_pct': current_pnl_pct
                }
            
            # No exit needed
            return {
                'should_exit': False,
                'current_pnl_pct': current_pnl_pct,
                'stop_price': stop_price,
                'distance_to_stop_pct': (current_price - stop_price) / current_price
            }
            
        except Exception as e:
            logger.error(f"Exit validation failed for {symbol}: {e}")
            return {'should_exit': False, 'reason': 'Validation error'}
    
    def get_risk_summary(self) -> Dict:
        """Get current risk summary"""
        return {
            'daily_loss_limit_hit': self.max_daily_loss_hit,
            'session_duration_hours': (datetime.now() - self.session_start).total_seconds() / 3600,
            'daily_trades': self.daily_trades,
            'max_positions': config.RISK_PARAMS['max_positions'],
            'max_position_size_pct': config.RISK_PARAMS['max_position_size_pct'],
            'trailing_stop_pct': config.RISK_PARAMS['trailing_stop_pct'],
            'max_daily_loss_pct': config.RISK_PARAMS['max_daily_loss_pct']
        }
    
    def record_trade(self, symbol: str, pnl: float):
        """Record completed trade for risk tracking"""
        self.daily_trades += 1
        self.daily_pnl += pnl
        
        logger.info(f"Trade recorded: {symbol} PnL: {pnl:+.2f} | "
                   f"Daily: {self.daily_pnl:+.2f} ({self.daily_trades} trades)")
    
    def reset_daily_metrics(self):
        """Reset daily risk metrics (call at market open)"""
        self.daily_pnl = 0
        self.daily_trades = 0
        self.max_daily_loss_hit = False
        self.session_start = datetime.now()
        logger.info("Daily risk metrics reset")