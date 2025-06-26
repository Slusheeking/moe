#!/usr/bin/env python3
"""
Data structures for the Enhanced Universe Updater.
Contains all dataclasses used for storing processed financial data.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np


@dataclass
class GapStatistics:
    """Complete Gap Pattern Features (12 total) with validation."""
    gap_frequency_90d: float = 0.0
    avg_gap_size: float = 0.0
    gap_continuation_rate: float = 0.0
    gap_fill_rate_24h: float = 0.0
    gap_fill_rate_72h: float = 0.0
    post_earnings_gap_avg: float = 0.0
    gap_up_frequency: float = 0.0
    gap_down_frequency: float = 0.0
    largest_gap_90d: float = 0.0
    gap_timing_preferences: Dict[str, float] = field(default_factory=dict)
    sector_gap_rank_historical: int = 0
    time_since_last_significant_gap: int = 0
    gap_quality_score_baseline: float = 0.0
    
    # FinGPT Enhancement Fields
    base_gap_score: float = 0.0  # Traditional gap analysis score
    fingpt_gap_enhancement: float = 0.5  # FinGPT enhancement score
    fingpt_sentiment_strength: float = 0.0  # News sentiment impact
    fingpt_catalyst_score: float = 0.0  # Catalyst detection score
    fingpt_forecast_confidence: float = 0.0  # Daily forecast confidence
    
    def __post_init__(self):
        """Validate data after initialization."""
        # Ensure all float values are valid
        for field_name, field_type in self.__annotations__.items():
            if field_type == float:
                value = getattr(self, field_name)
                if np.isnan(value) or np.isinf(value):
                    setattr(self, field_name, 0.0)


@dataclass
class OptionsBaselines:
    """Complete Options Historical Features (8 total) with validation."""
    avg_options_volume_90d: float = 0.0
    options_to_stock_ratio: float = 0.0
    avg_put_call_ratio: float = 0.0
    options_interest_turnover: float = 0.0
    unusual_activity_frequency: float = 0.0
    avg_implied_volatility_rank: float = 0.0
    options_flow_bullish_bias: float = 0.0
    earnings_options_activity_multiplier: float = 0.0
    
    def __post_init__(self):
        """Validate data after initialization."""
        for field_name, field_type in self.__annotations__.items():
            if field_type == float:
                value = getattr(self, field_name)
                if np.isnan(value) or np.isinf(value):
                    setattr(self, field_name, 0.0)


@dataclass
class TechnicalBaselines:
    """Complete Technical Features (19 total) with validation."""
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    trend_strength_90d: float = 0.0
    mean_reversion_tendency: float = 0.0
    breakout_success_rate: float = 0.0
    momentum_persistence: float = 0.0
    atr_20d: float = 0.0
    atr_percent_20d: float = 0.0
    realized_volatility_30d: float = 0.0
    realized_volatility_90d: float = 0.0
    max_drawdown_90d: float = 0.0
    upside_capture_ratio: float = 0.0
    downside_capture_ratio: float = 0.0
    bollinger_band_position: float = 0.0
    rsi: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    volume_surge_threshold: float = 0.0
    price_shock_threshold: float = 0.0
    
    def __post_init__(self):
        """Validate data after initialization."""
        for field_name, field_type in self.__annotations__.items():
            if field_type == float:
                value = getattr(self, field_name)
                if np.isnan(value) or np.isinf(value):
                    setattr(self, field_name, 0.0)


@dataclass
class InstitutionalFootprint:
    """Complete Institutional Features (10 total) with validation."""
    institutional_ownership_pct: float = 0.0
    avg_block_trade_size: float = 0.0
    institutional_footprint_score: float = 0.0
    smart_money_flow_ratio: float = 0.0
    institutional_participation_rate: float = 0.0
    avg_transaction_size: float = 0.0
    large_trade_frequency: float = 0.0
    institutional_sentiment_score: float = 0.0
    hedge_fund_ownership_estimate: float = 0.0
    dark_pool_activity_estimate: float = 0.0
    
    def __post_init__(self):
        """Validate data after initialization."""
        for field_name, field_type in self.__annotations__.items():
            if field_type == float:
                value = getattr(self, field_name)
                if np.isnan(value) or np.isinf(value):
                    setattr(self, field_name, 0.0)


@dataclass
class SectorMarketRelationships:
    """Complete Sector & Market Features (12 total) with validation."""
    sector_beta: float = 0.0
    sector_correlation: float = 0.0
    correlation_to_spy: float = 0.0
    correlation_to_vix: float = 0.0
    sector_leader_correlation: float = 0.0
    peer_rank_volume: int = 0
    peer_rank_volatility: int = 0
    sector_relative_strength_90d: float = 0.0
    market_cap: float = 0.0
    shares_outstanding: float = 0.0
    trending_market_performance: float = 0.0
    volatile_market_performance: float = 0.0
    
    def __post_init__(self):
        """Validate data after initialization."""
        for field_name, field_type in self.__annotations__.items():
            if field_type == float:
                value = getattr(self, field_name)
                if np.isnan(value) or np.isinf(value):
                    setattr(self, field_name, 0.0)


@dataclass
class LiquidityMicrostructure:
    """Complete Liquidity Features (8 total) with validation."""
    liquidity_score: float = 0.0
    avg_spread_bps: float = 0.0
    spread_volatility: float = 0.0
    depth_score: float = 0.0
    market_impact_coefficient: float = 0.0
    transaction_frequency_score: float = 0.0
    tick_frequency_90d: float = 0.0
    avg_dollar_volume_90d: float = 0.0
    
    def __post_init__(self):
        """Validate data after initialization."""
        for field_name, field_type in self.__annotations__.items():
            if field_type == float:
                value = getattr(self, field_name)
                if np.isnan(value) or np.isinf(value):
                    setattr(self, field_name, 0.0)


@dataclass
class TimeBasedPatterns:
    """Complete Time-Based Features (10 total) with validation."""
    intraday_volatility_profile: List[float] = field(default_factory=lambda: [0.0] * 24)
    day_of_week_bias: Dict[str, float] = field(default_factory=dict)
    monthly_seasonality: List[float] = field(default_factory=lambda: [0.0] * 12)
    pre_market_activity_score: float = 0.0
    earnings_gap_sensitivity: float = 0.0
    news_flow_impact_score: float = 0.0
    analyst_revision_sensitivity: float = 0.0
    social_sentiment_correlation: float = 0.0
    event_driven_volatility: float = 0.0
    regime_classification_accuracy: float = 0.0
    
    def __post_init__(self):
        """Validate data after initialization."""
        # Ensure lists have correct length
        if len(self.intraday_volatility_profile) != 24:
            self.intraday_volatility_profile = [0.0] * 24
        if len(self.monthly_seasonality) != 12:
            self.monthly_seasonality = [0.0] * 12
            
        for field_name, field_type in self.__annotations__.items():
            if field_type == float:
                value = getattr(self, field_name)
                if np.isnan(value) or np.isinf(value):
                    setattr(self, field_name, 0.0)


@dataclass
class UnusualWhalesData:
    """Complete Unusual Whales data structure with validation."""
    # Flow & Greek Exposure
    recent_flow_alerts: List[Dict] = field(default_factory=list)
    gamma_exposure: float = 0.0
    delta_exposure: float = 0.0
    vanna_exposure: float = 0.0
    charm_exposure: float = 0.0
    max_pain_current: float = 0.0
    
    # Dark Pool Intelligence
    dark_pool_volume_24h: float = 0.0
    dark_pool_prints_count: int = 0
    lit_dark_ratio: float = 0.0
    
    # Congressional & Insider Intelligence
    congressional_activity_recent: bool = False
    congressional_sentiment: str = "neutral"
    insider_transactions_30d: int = 0
    insider_buy_sell_ratio: float = 0.0
    
    # Institutional Data
    institutional_ownership_changes: Dict = field(default_factory=dict)
    latest_13f_changes: Dict = field(default_factory=dict)
    
    # Calendar Events
    earnings_date_proximity: int = 999
    fda_event_proximity: int = 999
    economic_event_proximity: int = 999
    dividend_ex_date_proximity: int = 999
    
    # Options Intelligence
    iv_rank: float = 0.0
    options_volume_vs_avg: float = 0.0
    unusual_options_activity: bool = False
    put_call_ratio: float = 0.0
    
    # Short Interest & ETF Data
    short_interest_ratio: float = 0.0
    days_to_cover: float = 0.0
    etf_inflow_outflow: float = 0.0
    etf_rebalance_pressure: float = 0.0
    
    def __post_init__(self):
        """Validate data after initialization."""
        for field_name, field_type in self.__annotations__.items():
            if field_type == float:
                value = getattr(self, field_name)
                if np.isnan(value) or np.isinf(value):
                    setattr(self, field_name, 0.0)


@dataclass
class PolygonData:
    """Complete Polygon data structure with validation."""
    # News & Sentiment
    news_sentiment_score: float = 0.0
    news_count_24h: int = 0
    news_relevance_score: float = 0.0
    
    # Enhanced sentiment from FinGPT
    sentiment_confidence: float = 0.0
    sentiment_label: str = "neutral"
    key_topics: List[str] = field(default_factory=list)
    market_impact_score: float = 0.0
    sentiment_momentum: float = 0.0
    controversy_score: float = 0.0
    entity_sentiments: Dict[str, float] = field(default_factory=dict)
    
    # Technical Indicators
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    rsi_14: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    bollinger_upper: float = 0.0
    bollinger_lower: float = 0.0
    bollinger_position: float = 0.0
    
    # Volume Analysis
    volume_sma_20: float = 0.0
    volume_ratio_vs_avg: float = 0.0
    
    # Market Data Quality
    data_quality_score: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate data after initialization."""
        for field_name, field_type in self.__annotations__.items():
            if field_type == float:
                value = getattr(self, field_name)
                if np.isnan(value) or np.isinf(value):
                    setattr(self, field_name, 0.0)


@dataclass
class EnhancedSymbolData:
    """Complete symbol data with ALL pre-calculable features."""
    
    # Basic Identifiers
    company: str
    symbol: str
    gics_sector: str = ""
    gics_sub_industry: str = ""
    exchange: str = ""
    
    # Historical Price & Volume (90-day averages)
    avg_price_90d: float = 0.0
    avg_volume_90d: float = 0.0
    avg_dollar_volume_90d: float = 0.0
    
    # COMPLETE FEATURE SETS (ALL 75+ FEATURES)
    gap_statistics: GapStatistics = field(default_factory=GapStatistics)
    options_baselines: OptionsBaselines = field(default_factory=OptionsBaselines)
    technical_baselines: TechnicalBaselines = field(default_factory=TechnicalBaselines)
    institutional_footprint: InstitutionalFootprint = field(default_factory=InstitutionalFootprint)
    sector_market_relationships: SectorMarketRelationships = field(default_factory=SectorMarketRelationships)
    liquidity_microstructure: LiquidityMicrostructure = field(default_factory=LiquidityMicrostructure)
    time_based_patterns: TimeBasedPatterns = field(default_factory=TimeBasedPatterns)
    
    # Unusual Whales Data
    unusual_whales: UnusualWhalesData = field(default_factory=UnusualWhalesData)
    
    # Polygon Data
    polygon: PolygonData = field(default_factory=PolygonData)
    
    # Quality Scores & Rankings
    overall_trading_quality: float = 0.0
    gap_trading_suitability: float = 0.0
    universe_rank_overall: int = 0
    universe_rank_gap_trading: int = 0
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    data_completeness_score: float = 0.0
    
    def __post_init__(self):
        """Calculate data completeness score."""
        total_fields = 0
        filled_fields = 0
        
        # Check all numeric fields
        for field_name in self.__dataclass_fields__:
            field_value = getattr(self, field_name)
            
            if isinstance(field_value, (int, float)):
                total_fields += 1
                if field_value != 0 and not np.isnan(field_value):
                    filled_fields += 1
            elif hasattr(field_value, '__dataclass_fields__'):
                # Check nested dataclass fields
                for nested_field in field_value.__dataclass_fields__:
                    nested_value = getattr(field_value, nested_field)
                    if isinstance(nested_value, (int, float)):
                        total_fields += 1
                        if nested_value != 0 and not np.isnan(nested_value):
                            filled_fields += 1
        
        self.data_completeness_score = filled_fields / total_fields if total_fields > 0 else 0.0


# Sentiment Analysis Data Structures
@dataclass
class SentimentAnalysis:
    """Enhanced sentiment analysis results."""
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    sentiment_label: str  # 'bullish', 'bearish', 'neutral'
    key_topics: List[str]
    entity_sentiments: Dict[str, float]  # Company-specific sentiments
    market_impact_score: float  # Estimated market impact


@dataclass
class NewsAnalysis:
    """Complete news analysis with FinGPT."""
    overall_sentiment: SentimentAnalysis
    individual_articles: List[Dict[str, Any]]
    sentiment_momentum: float  # Change in sentiment over time
    controversy_score: float  # Disagreement in sentiment
    credibility_weighted_score: float  # Weighted by source credibility