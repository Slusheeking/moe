#!/usr/bin/env python3
"""
Enhanced Universe Updater Package
Version 5.0 - Production Ready with FinGPT Integration

This package provides comprehensive financial data processing for the Russell 1000 universe.
"""

# Import main components for easy access
from .universe_calculator import CompleteUniverseCalculator, main as update_universe
from .data_structures import (
    EnhancedSymbolData, GapStatistics, OptionsBaselines,
    TechnicalBaselines, InstitutionalFootprint, SectorMarketRelationships,
    LiquidityMicrostructure, TimeBasedPatterns, UnusualWhalesData, PolygonData,
    SentimentAnalysis, NewsAnalysis
)
from .api_processors import (
    APIClients, UnusualWhalesDataProcessor, PolygonDataProcessor, CacheManager
)
from .fingpt_sentiment import SimplifiedSentimentAnalyzer
from .technical_calculations import (
    calculate_gaps_vectorized, calculate_returns_vectorized,
    calculate_true_range_vectorized, find_support_resistance_vectorized,
    calculate_volatility_profile, calculate_trend_metrics,
    calculate_market_microstructure, calculate_regime_indicators,
    calculate_gap_patterns, calculate_volume_profile,
    GPU_AVAILABLE
)

__version__ = "5.0.0"
__author__ = "Enhanced Universe Updater Team"

# Convenience function to check system status
def check_system_status():
    """Check if all required components are available."""
    import logging
    logger = logging.getLogger(__name__)
    
    status = {
        'gpu_available': GPU_AVAILABLE,
        'fingpt_available': False,
        'polygon_available': False,
        'unusual_whales_available': False
    }
    
    # Check FinGPT
    try:
        from .fingpt_sentiment import TRANSFORMERS_AVAILABLE
        status['fingpt_available'] = TRANSFORMERS_AVAILABLE
    except:
        pass
    
    # Check APIs
    try:
        from polygon import RESTClient
        status['polygon_available'] = True
    except:
        pass
    
    try:
        from unusualwhales import UnusualWhalesClient
        status['unusual_whales_available'] = True
    except:
        pass
    
    logger.info("System Status Check:")
    logger.info(f"  GPU Acceleration: {'✅' if status['gpu_available'] else '❌'}")
    logger.info(f"  FinGPT Sentiment: {'✅' if status['fingpt_available'] else '❌'}")
    logger.info(f"  Polygon API: {'✅' if status['polygon_available'] else '❌'}")
    logger.info(f"  Unusual Whales API: {'✅' if status['unusual_whales_available'] else '❌'}")
    
    return status

__all__ = [
    # Main calculator
    'CompleteUniverseCalculator',
    'update_universe',
    
    # Data structures
    'EnhancedSymbolData',
    'GapStatistics',
    'OptionsBaselines',
    'TechnicalBaselines',
    'InstitutionalFootprint',
    'SectorMarketRelationships',
    'LiquidityMicrostructure',
    'TimeBasedPatterns',
    'UnusualWhalesData',
    'PolygonData',
    'SentimentAnalysis',
    'NewsAnalysis',
    
    # API processors
    'APIClients',
    'UnusualWhalesDataProcessor',
    'PolygonDataProcessor',
    'CacheManager',
    
    # FinGPT sentiment
    'SimplifiedSentimentAnalyzer',
    
    # Technical calculations
    'calculate_gaps_vectorized',
    'calculate_returns_vectorized',
    'calculate_true_range_vectorized',
    'find_support_resistance_vectorized',
    'calculate_volatility_profile',
    'calculate_trend_metrics',
    'calculate_market_microstructure',
    'calculate_regime_indicators',
    'calculate_gap_patterns',
    'calculate_volume_profile',
    'GPU_AVAILABLE',
    
    # Utilities
    'check_system_status',
    '__version__'
]