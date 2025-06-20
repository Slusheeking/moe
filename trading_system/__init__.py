"""
Event-Driven Momentum Trading System
Pure event-driven architecture with instant signal processing
No scheduled loops - everything triggered by real-time market events

Focus: Real-time momentum detection with instant XGBoost analysis
Entry: Event-driven momentum signals ≥ 25 score
Exit: 5% stop loss, 10% take profit via bracket orders
Data: Live WebSocket + 3-day historical fusion
"""

__version__ = "2.0.0"
__author__ = "Event-Driven Momentum Trader"
__description__ = "Event-driven momentum detection and trading system"

# Core system components
from .momentum_detector import LiveMomentumDetector
from .position_manager import PositionManager
from .risk_manager import RiskManager
from .system_monitor import SystemMonitor
from .data_manager import PolygonStream, DataAggregator
from .polygon_client import PolygonClient
from .alpaca_client import AlpacaClient
from .main import EventDrivenMomentumSystem

__all__ = [
    'LiveMomentumDetector',
    'PositionManager', 
    'RiskManager',
    'SystemMonitor',
    'PolygonStream',
    'DataAggregator',
    'PolygonClient',
    'AlpacaClient',
    'EventDrivenMomentumSystem'
]

# System configuration
SYSTEM_INFO = {
    'name': 'Event-Driven Momentum Trading System',
    'version': __version__,
    'architecture': 'Pure event-driven - WebSocket → Filter → XGBoost → Execute',
    'strategy': 'Real-time momentum detection with instant execution',
    'entry_method': 'Event-driven momentum score ≥ 25 (live + historical fusion)',
    'exit_method': '5% stop loss, 10% take profit via Alpaca bracket orders',
    'data_requirements': 'Live WebSocket trades + 3-day historical cache',
    'target_moves': 'Instant momentum detection for maximum opportunity capture',
    'risk_management': 'Automatic bracket orders, multi-stock position limits'
}