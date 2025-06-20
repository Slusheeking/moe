"""
Pure Momentum Trading System
Optimized for CRCL-type explosive moves with 5% trailing stops

Focus: Numerical indicators only, no news dependency
Entry: XGBoost momentum detection ≥ 40 score
Exit: 5% trailing stop from highest price
Data: Live + 3-day maximum lookback (IPO compatible)
"""

__version__ = "1.0.0"
__author__ = "Momentum Trader"
__description__ = "Pure numerical momentum detection and trading system"

# Core system components
from .momentum_detector import LiveMomentumDetector
from .position_manager import PositionManager
from .risk_manager import RiskManager
from .system_monitor import SystemMonitor
from .data_manager import PolygonStream, DataAggregator
from .polygon_client import PolygonClient
from .alpaca_client import AlpacaClient
from .main import MomentumTradingSystem

__all__ = [
    'LiveMomentumDetector',
    'PositionManager', 
    'RiskManager',
    'SystemMonitor',
    'PolygonStream',
    'DataAggregator',
    'PolygonClient',
    'AlpacaClient',
    'MomentumTradingSystem'
]

# System configuration
SYSTEM_INFO = {
    'name': 'Pure Momentum Trading System',
    'version': __version__,
    'strategy': 'Live momentum detection with XGBoost pattern recognition',
    'entry_method': 'Momentum score ≥ 40 (live + 3-day features)',
    'exit_method': '5% trailing stop from highest price',
    'data_requirements': 'Live market data + maximum 3-day lookback',
    'target_moves': 'CRCL-type explosive momentum (100%+ potential)',
    'risk_management': '5% trailing stops, daily loss limits, position sizing'
}