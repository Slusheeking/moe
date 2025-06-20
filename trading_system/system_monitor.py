"""
System Monitor for Momentum Trading
Real-time monitoring of momentum detection and trading performance
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

from trading_system import config

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitor momentum trading system performance and health"""
    
    def __init__(self, alpaca_client, polygon_client):
        self.alpaca = alpaca_client
        self.polygon = polygon_client
        
        # Performance tracking
        self.session_start = datetime.now()
        self.trades_today = []
        self.momentum_detections = deque(maxlen=1000)
        self.system_alerts = deque(maxlen=100)
        
        # Real-time metrics
        self.current_positions = {}
        self.account_value_history = deque(maxlen=1440)  # 24 hours of minute data
        self.momentum_scores = defaultdict(list)
        
        # System health
        self.api_errors = defaultdict(int)
        self.last_heartbeat = datetime.now()
        
    async def start_monitoring(self):
        """Start monitoring loops"""
        logger.info("📊 Starting system monitoring...")
        
        # Start monitoring tasks
        tasks = [
            self._performance_monitoring_loop(),
            self._account_monitoring_loop(),
            self._health_monitoring_loop(),
            self._alert_monitoring_loop()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _performance_monitoring_loop(self):
        """Monitor trading performance metrics"""
        while True:
            try:
                # Update position tracking
                positions = await self.alpaca.list_positions()
                current_positions = {}
                
                total_unrealized_pnl = 0
                total_market_value = 0
                
                for pos in positions:
                    symbol = pos['symbol']
                    unrealized_pnl = float(pos['unrealized_pl'])
                    market_value = float(pos['market_value'])
                    
                    current_positions[symbol] = {
                        'symbol': symbol,
                        'qty': float(pos['qty']),
                        'market_value': market_value,
                        'unrealized_pnl': unrealized_pnl,
                        'unrealized_pnl_pct': float(pos['unrealized_plpc']),
                        'avg_entry_price': float(pos['avg_entry_price']),
                        'current_price': float(pos.get('current_price', 0))
                    }
                    
                    total_unrealized_pnl += unrealized_pnl
                    total_market_value += market_value
                
                self.current_positions = current_positions
                
                # Log performance summary every 5 minutes
                if datetime.now().minute % 5 == 0:
                    await self._log_performance_summary(
                        total_unrealized_pnl, total_market_value
                    )
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                self.api_errors['performance'] += 1
                await asyncio.sleep(60)
    
    async def _account_monitoring_loop(self):
        """Monitor account value and equity changes"""
        while True:
            try:
                account = await self.alpaca.get_account()
                if account:
                    portfolio_value = float(account['portfolio_value'])
                    equity = float(account['equity'])
                    buying_power = float(account['buying_power'])
                    
                    # Store account value history
                    self.account_value_history.append({
                        'timestamp': datetime.now(),
                        'portfolio_value': portfolio_value,
                        'equity': equity,
                        'buying_power': buying_power
                    })
                    
                    # Check for significant changes
                    if len(self.account_value_history) > 1:
                        prev_value = self.account_value_history[-2]['portfolio_value']
                        change_pct = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0
                        
                        # Alert on large changes
                        if abs(change_pct) > 0.02:  # 2% change
                            await self._create_alert(
                                'ACCOUNT_CHANGE',
                                f"Account value changed {change_pct:+.1%} to ${portfolio_value:,.2f}",
                                'high' if abs(change_pct) > 0.05 else 'medium'
                            )
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Account monitoring error: {e}")
                self.api_errors['account'] += 1
                await asyncio.sleep(300)
    
    async def _health_monitoring_loop(self):
        """Monitor system health and API connectivity"""
        while True:
            try:
                # Update heartbeat
                self.last_heartbeat = datetime.now()
                
                # Check API health
                api_health = await self._check_api_health()
                
                # Check error rates
                total_errors = sum(self.api_errors.values())
                if total_errors > 50:  # High error rate
                    await self._create_alert(
                        'HIGH_ERROR_RATE',
                        f"High API error rate: {total_errors} errors",
                        'high'
                    )
                
                # Check system performance
                await self._check_system_performance()
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(120)
    
    async def _alert_monitoring_loop(self):
        """Monitor for trading alerts and opportunities"""
        while True:
            try:
                # Check for momentum opportunities
                await self._check_momentum_alerts()
                
                # Check position alerts
                await self._check_position_alerts()
                
                # Check risk alerts
                await self._check_risk_alerts()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Alert monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _check_api_health(self) -> Dict:
        """Check API connectivity and health"""
        health = {'alpaca': False, 'polygon': False}
        
        try:
            # Check Alpaca
            account = await self.alpaca.get_account()
            health['alpaca'] = account is not None
            
            # Check Polygon (simple test)
            # We'll assume it's healthy if no recent errors
            health['polygon'] = self.api_errors.get('polygon', 0) < 10
            
        except Exception as e:
            logger.error(f"API health check failed: {e}")
        
        return health
    
    async def _check_system_performance(self):
        """Check overall system performance metrics"""
        try:
            # Calculate detection rate
            recent_detections = [
                d for d in self.momentum_detections 
                if d['timestamp'] > datetime.now() - timedelta(hours=1)
            ]
            
            detection_rate = len(recent_detections)
            
            # Log performance metrics
            if detection_rate == 0:
                await self._create_alert(
                    'LOW_DETECTION_RATE',
                    "No momentum detections in past hour",
                    'medium'
                )
            
        except Exception as e:
            logger.error(f"System performance check failed: {e}")
    
    async def _check_momentum_alerts(self):
        """Check for notable momentum detections"""
        try:
            # Check for high-score momentum detections
            high_momentum_threshold = 80
            
            recent_high_momentum = [
                d for d in self.momentum_detections
                if d.get('momentum_score', 0) >= high_momentum_threshold
                and d['timestamp'] > datetime.now() - timedelta(minutes=5)
            ]
            
            for detection in recent_high_momentum:
                symbol = detection.get('symbol', 'UNKNOWN')
                score = detection.get('momentum_score', 0)
                
                await self._create_alert(
                    'HIGH_MOMENTUM',
                    f"High momentum detected: {symbol} (score: {score:.0f})",
                    'medium'
                )
            
        except Exception as e:
            logger.error(f"Momentum alert check failed: {e}")
    
    async def _check_position_alerts(self):
        """Check for position-related alerts"""
        try:
            for symbol, pos_data in self.current_positions.items():
                unrealized_pnl_pct = pos_data.get('unrealized_pnl_pct', 0)
                
                # Alert on large gains
                if unrealized_pnl_pct > 0.20:  # 20%+ gain
                    await self._create_alert(
                        'LARGE_GAIN',
                        f"{symbol}: {unrealized_pnl_pct:+.1%} unrealized gain",
                        'low'
                    )
                
                # Alert on large losses
                elif unrealized_pnl_pct < -0.10:  # 10%+ loss
                    await self._create_alert(
                        'LARGE_LOSS',
                        f"{symbol}: {unrealized_pnl_pct:+.1%} unrealized loss",
                        'high'
                    )
            
        except Exception as e:
            logger.error(f"Position alert check failed: {e}")
    
    async def _check_risk_alerts(self):
        """Check for risk-related alerts"""
        try:
            if not self.account_value_history:
                return
            
            current_value = self.account_value_history[-1]['portfolio_value']
            
            # Calculate daily P&L
            session_start_value = config.ACCOUNT_SIZE  # Starting value
            daily_pnl_pct = (current_value - session_start_value) / session_start_value
            
            # Alert on approaching daily loss limit
            max_loss_pct = config.RISK_PARAMS['max_daily_loss_pct']
            if daily_pnl_pct <= -max_loss_pct * 0.8:  # 80% of max loss
                await self._create_alert(
                    'APPROACHING_LOSS_LIMIT',
                    f"Approaching daily loss limit: {daily_pnl_pct:.1%}",
                    'high'
                )
            
        except Exception as e:
            logger.error(f"Risk alert check failed: {e}")
    
    async def _create_alert(self, alert_type: str, message: str, severity: str):
        """Create system alert"""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message,
            'severity': severity
        }
        
        self.system_alerts.append(alert)
        
        # Log based on severity
        if severity == 'high':
            logger.warning(f"{alert_type}: {message}")
        elif severity == 'medium':
            logger.info(f"{alert_type}: {message}")
        else:
            logger.debug(f"{alert_type}: {message}")
    
    async def _log_performance_summary(self, total_unrealized_pnl: float, total_market_value: float):
        """Log performance summary"""
        try:
            num_positions = len(self.current_positions)
            session_duration = (datetime.now() - self.session_start).total_seconds() / 3600
            
            # Calculate session metrics
            session_pnl_pct = 0
            if self.account_value_history:
                current_value = self.account_value_history[-1]['portfolio_value']
                session_pnl_pct = (current_value - config.ACCOUNT_SIZE) / config.ACCOUNT_SIZE * 100
            
            logger.info(f"""
MOMENTUM SYSTEM PERFORMANCE SUMMARY
{'='*60}
⏱Session Time: {session_duration:.1f} hours
Account P&L: {session_pnl_pct:+.2f}%
Positions: {num_positions} (${total_market_value:,.0f})
Unrealized P&L: ${total_unrealized_pnl:+,.2f}
Momentum Detections: {len(self.momentum_detections)} total
System Alerts: {len(self.system_alerts)} active
API Errors: {sum(self.api_errors.values())} total
            """)
            
        except Exception as e:
            logger.error(f"Performance summary logging failed: {e}")
    
    def record_momentum_detection(self, symbol: str, momentum_score: float, features: Dict = None):
        """Record momentum detection for monitoring"""
        detection = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'momentum_score': momentum_score,
            'features': features or {}
        }
        
        self.momentum_detections.append(detection)
        self.momentum_scores[symbol].append(momentum_score)
    
    def record_trade_execution(self, symbol: str, side: str, quantity: int, price: float):
        """Record trade execution"""
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price
        }
        
        self.trades_today.append(trade)
        logger.info(f"Trade recorded: {side.upper()} {quantity} {symbol} @ ${price:.2f}")
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'session_duration_hours': (datetime.now() - self.session_start).total_seconds() / 3600,
            'active_positions': len(self.current_positions),
            'momentum_detections_count': len(self.momentum_detections),
            'trades_today': len(self.trades_today),
            'system_alerts': len(self.system_alerts),
            'api_errors': dict(self.api_errors),
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'account_value': self.account_value_history[-1]['portfolio_value'] if self.account_value_history else 0
        }
    
    def get_recent_alerts(self, hours: int = 1) -> List[Dict]:
        """Get recent system alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.system_alerts 
            if alert['timestamp'] > cutoff_time
        ]
    
    def clear_old_data(self):
        """Clear old monitoring data to prevent memory buildup"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Clear old momentum detections
        self.momentum_detections = deque([
            d for d in self.momentum_detections 
            if d['timestamp'] > cutoff_time
        ], maxlen=1000)
        
        # Clear old trades
        self.trades_today = [
            t for t in self.trades_today 
            if t['timestamp'] > cutoff_time
        ]
        
        logger.debug("Cleared old monitoring data")