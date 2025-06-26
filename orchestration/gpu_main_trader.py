#!/usr/bin/env python3
"""
GPU-Accelerated Gap Trading System - Main Orchestrator
Coordinates all components through the trading day with precise timing and GPU optimization.
"""

import os
import sys
import asyncio
import torch
import signal
import logging
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Any
import pytz
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all components
from config.gpu_trading_config import GPUTradingConfig
from database.connection import get_db_manager
from gpu_core.tensor_universe_manager import TensorUniverseManager
from gpu_core.parallel_data_pipeline import ParallelDataPipeline
from gpu_features.vectorized_gap_analyzer import VectorizedGapAnalyzer
from gpu_models.ensemble_continuation_filter import EnsembleContinuationFilter
from gpu_models.neural_magnitude_predictor import NeuralMagnitudePredictor
from gpu_models.attention_regime_classifier import AttentionRegimeClassifier
from gpu_risk.portfolio_optimizer import GPUPortfolioOptimizer
from gpu_execution.smart_order_router import GPUSmartOrderRouter, OrderRequest, OrderType
from gpu_monitoring.pattern_recognition_engine import GPUPatternRecognitionEngine
from gpu_learning.distributed_online_learner import GPUDistributedOnlineLearner, TradingOutcome


class GPUMainTrader:
    """Main orchestrator for GPU-accelerated gap trading system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize all trading components."""
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Initializing GPU Gap Trading System...")
        
        # Load configuration
        self.config = GPUTradingConfig()
        
        # Database connection for performance and events logging
        self.db_manager = get_db_manager()
        
        # Validate GPU setup
        self._validate_gpu_setup()
        
        # Initialize components
        self._initialize_components()
        
        # Trading state
        self.is_running = False
        self.current_phase = "INITIALIZATION"
        self.active_positions = {}
        self.pending_orders = {}
        self.daily_pnl = 0.0
        self.total_trades = 0
        
        # Performance tracking
        self.session_start_time = datetime.now(self.config.TIMEZONE)
        self.session_stats = {
            'gaps_analyzed': 0,
            'trades_executed': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Initialization complete!")
    
    def _setup_logging(self):
        """Configure logging system."""
        log_dir = Path('/home/ubuntu/gap_trading_logs')
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"gap_trader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _validate_gpu_setup(self):
        """Validate GPU configuration and availability."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This system requires GPU.")
        
        gpu_props = torch.cuda.get_device_properties(0)
        self.logger.info(f"GPU: {gpu_props.name}")
        self.logger.info(f"GPU Memory: {gpu_props.total_memory / 1e9:.1f} GB")
        self.logger.info(f"CUDA Version: {torch.version.cuda}")
        
        # Set GPU memory growth
        torch.cuda.set_per_process_memory_fraction(self.config.GPU_MEMORY_FRACTION)
    
    def _initialize_components(self):
        """Initialize all trading components in correct order."""
        self.logger.info("Initializing trading components...")
        
        # 1. Universe Manager
        self.universe_manager = TensorUniverseManager(self.config)
        self.logger.info(f"Universe loaded: {len(self.universe_manager.universe_symbols)} symbols")
        
        # 2. Data Pipeline
        self.data_pipeline = ParallelDataPipeline(self.config, self.universe_manager)
        
        # 3. Feature Engineering
        self.gap_analyzer = VectorizedGapAnalyzer(self.config, self.universe_manager, self.data_pipeline)
        
        # 4. Prediction Models
        self.continuation_filter = EnsembleContinuationFilter(self.config, self.gap_analyzer)
        self.magnitude_predictor = NeuralMagnitudePredictor(self.config, self.gap_analyzer)
        
        # 5. Market Analysis
        self.regime_classifier = AttentionRegimeClassifier(self.config, self.data_pipeline)
        
        # 6. Risk Management
        self.portfolio_optimizer = GPUPortfolioOptimizer(
            self.config, 
            self.universe_manager,
            initial_capital=100000  # Get from config or broker
        )
        
        # 7. Execution
        self.order_router = GPUSmartOrderRouter(self.config, self.data_pipeline)
        
        # 8. Monitoring
        self.pattern_engine = GPUPatternRecognitionEngine(self.config, self.data_pipeline)
        
        # 9. Learning
        self.online_learner = GPUDistributedOnlineLearner(self.config)
        
        self.logger.info("All components initialized successfully!")
    
    async def run(self):
        """Main trading loop."""
        self.is_running = True
        self.logger.info("Starting main trading loop...")
        
        try:
            while self.is_running:
                current_time = datetime.now(self.config.TIMEZONE)
                self.current_phase = self.config.get_current_trading_phase(current_time.time())
                
                # Execute phase-specific logic
                if self.current_phase == "CLOSED":
                    await self._handle_closed_phase()
                    
                elif self.current_phase == "PRE_MARKET_SCAN":
                    await self._handle_pre_market_scan()
                    
                elif self.current_phase == "FINAL_PREPARATION":
                    await self._handle_final_preparation()
                    
                elif self.current_phase == "PRE_MARKET_ENTRY":
                    await self._handle_pre_market_entry()
                    
                elif self.current_phase == "PRIMARY_ENTRY":
                    await self._handle_primary_entry()
                    
                elif self.current_phase == "POSITION_MANAGEMENT":
                    await self._handle_position_management()
                    
                elif self.current_phase == "EXIT_WINDOW":
                    await self._handle_exit_window()
                    
                elif self.current_phase == "POST_MARKET":
                    await self._handle_post_market()
                
                # Brief pause to prevent CPU spinning
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Critical error in main loop: {e}", exc_info=True)
            self.is_running = False
        
        finally:
            await self._shutdown()
    
    async def _handle_closed_phase(self):
        """Handle market closed phase."""
        # Update universe monthly
        if datetime.now().day == 1:
            self.logger.info("Monthly universe update...")
            self.universe_manager.update_universe()
        
        # Sleep until pre-market
        await asyncio.sleep(60)
    
    async def _handle_pre_market_scan(self):
        """Handle pre-market scanning phase (4:00 AM - 9:00 AM)."""
        current_time = datetime.now(self.config.TIMEZONE)
        
        # Start WebSocket data if not already running
        if not self.data_pipeline.ws_connected:
            self.logger.info("Starting real-time data streams...")
            asyncio.create_task(self.data_pipeline.start_realtime_data())
            await asyncio.sleep(5)  # Allow connections to establish
        
        # Fetch historical data for context
        if current_time.minute % 30 == 0:  # Every 30 minutes
            self.logger.info("Updating historical data...")
            self.data_pipeline.fetch_historical_data()
        
        # Periodic gap analysis
        if current_time.minute % 5 == 0:  # Every 5 minutes
            await self._analyze_pre_market_gaps()
        
        await asyncio.sleep(30)
    
    async def _analyze_pre_market_gaps(self):
        """Analyze gaps forming in pre-market."""
        self.logger.info("Analyzing pre-market gaps...")
        
        # Get gap analysis
        gap_features = self.gap_analyzer.analyze_gaps()
        
        # Get top gaps
        top_symbols, top_features = self.gap_analyzer.get_top_gaps(n=20)
        
        self.session_stats['gaps_analyzed'] = len(top_symbols)
        
        if top_symbols:
            self.logger.info(f"Found {len(top_symbols)} significant gaps")
            for i, symbol in enumerate(top_symbols[:5]):
                gap_report = self.gap_analyzer.get_gap_report(symbol)
                self.logger.info(f"  {i+1}. {symbol}: {gap_report['gap_size']:.2f}% gap, "
                               f"quality={gap_report['gap_quality_score']:.3f}")
    
    async def _handle_final_preparation(self):
        """Handle final preparation phase (9:00 AM - 9:20 AM)."""
        self.logger.info("Final preparation phase...")
        
        # Classify market regime
        regime_analysis = self.regime_classifier.classify_regime()
        self.logger.info(f"Market regime: {regime_analysis['regime']} "
                        f"(confidence: {regime_analysis['confidence']:.2%})")
        
        # Get regime adjustments
        regime_adjustments = self.regime_classifier.get_regime_adjusted_parameters()
        
        # Final gap analysis
        gap_features = self.gap_analyzer.analyze_gaps()
        top_symbols, _ = self.gap_analyzer.get_top_gaps(n=30)
        
        # Run models for filtering and prediction
        gap_opportunities = {}
        magnitude_predictions = {}
        
        for symbol in top_symbols:
            gap_report = self.gap_analyzer.get_gap_report(symbol)
            
            # Continuation filter
            symbol_features = self.gap_analyzer.gap_features[
                self.gap_analyzer.universe_symbols.index(symbol)
            ].unsqueeze(0)
            
            should_trade, cont_predictions = self.continuation_filter.predict_continuation(
                symbol_features
            )
            
            if should_trade[0]:
                gap_opportunities[symbol] = gap_report
                
                # Magnitude prediction
                mag_predictions = self.magnitude_predictor.predict_magnitude_distribution(
                    symbol_features
                )
                magnitude_predictions[symbol] = {
                    k: v[0] if torch.is_tensor(v) and v.dim() > 0 else v
                    for k, v in mag_predictions.items()
                }
        
        self.logger.info(f"Qualified opportunities: {len(gap_opportunities)}")
        
        # Portfolio optimization
        if gap_opportunities:
            optimal_positions = self.portfolio_optimizer.optimize_portfolio(
                gap_opportunities,
                magnitude_predictions,
                regime_adjustments
            )
            
            self.pending_orders = optimal_positions
            self.logger.info(f"Optimal portfolio: {len(optimal_positions)} positions")
            
            # Log top positions
            for symbol, size in sorted(optimal_positions.items(), 
                                     key=lambda x: x[1], reverse=True)[:5]:
                self.logger.info(f"  {symbol}: ${size:,.0f}")
        
        await asyncio.sleep(30)
    
    async def _handle_pre_market_entry(self):
        """Handle pre-market entry phase (9:20 AM - 9:30 AM)."""
        # Execute highest conviction trades pre-market
        if self.pending_orders:
            self.logger.info("Executing pre-market entries...")
            
            # Sort by position size (proxy for conviction)
            sorted_orders = sorted(self.pending_orders.items(), 
                                 key=lambda x: x[1], reverse=True)
            
            # Execute top 20% pre-market
            n_premarket = max(1, int(len(sorted_orders) * 0.2))
            
            for symbol, position_size in sorted_orders[:n_premarket]:
                await self._execute_trade(symbol, position_size, urgency=0.7)
                await asyncio.sleep(2)  # Space out orders
        
        await asyncio.sleep(10)
    
    async def _handle_primary_entry(self):
        """Handle primary entry window (9:30 AM - 10:00 AM)."""
        # Execute remaining orders
        for symbol, position_size in self.pending_orders.items():
            if symbol not in self.active_positions:
                await self._execute_trade(symbol, position_size, urgency=0.8)
                await asyncio.sleep(1)
        
        # Clear pending orders
        self.pending_orders.clear()
        
        # Start position monitoring
        await self._monitor_positions()
        
        await asyncio.sleep(30)
    
    async def _handle_position_management(self):
        """Handle position management phase (10:00 AM - 11:00 AM)."""
        # Continuous monitoring and adjustment
        await self._monitor_positions()
        
        # Check for new opportunities (secondary gaps)
        if datetime.now(self.config.TIMEZONE).minute % 10 == 0:
            await self._check_secondary_opportunities()
        
        await asyncio.sleep(10)
    
    async def _handle_exit_window(self):
        """Handle exit window (11:00 AM - 11:30 AM)."""
        self.logger.info("Entering exit window - closing all positions...")
        
        # Get exit recommendations
        exit_recommendations = self.pattern_engine.get_exit_recommendations()
        
        # Sort by urgency
        sorted_positions = sorted(
            self.active_positions.items(),
            key=lambda x: exit_recommendations.get(x[0], {}).get('urgency', 0),
            reverse=True
        )
        
        # Exit all positions
        for symbol, position_data in sorted_positions:
            await self._exit_position(symbol, urgency=0.9)
            await asyncio.sleep(1)
        
        await asyncio.sleep(30)
    
    async def _handle_post_market(self):
        """Handle post-market phase."""
        if self.total_trades > 0:
            self.logger.info("Post-market analysis...")
            
            # Calculate session statistics
            self._calculate_session_stats()
            
            # Save learning checkpoint
            self.online_learner.save_checkpoint(
                f"session_{datetime.now().strftime('%Y%m%d')}"
            )
            
            # Generate performance report
            self._generate_daily_report()
            
            # Persist daily performance to database
            self._persist_daily_performance()
            
            # Reset for next day
            self.daily_pnl = 0.0
            self.total_trades = 0
            self.session_stats['gaps_analyzed'] = 0
        
        # Stop data streams
        if self.data_pipeline.ws_connected:
            self.data_pipeline.stop_realtime_data()
        
        await asyncio.sleep(60)
    
    async def _execute_trade(self, symbol: str, position_size: float, urgency: float = 0.5):
        """Execute a trade through smart order router."""
        try:
            # Calculate share quantity
            current_price = self._get_current_price(symbol)
            if current_price <= 0:
                self.logger.warning(f"Invalid price for {symbol}, skipping trade")
                return
            
            quantity = int(position_size / current_price)
            if quantity < 1:
                self.logger.warning(f"Position size too small for {symbol}")
                return
            
            # Create order request
            order_request = OrderRequest(
                symbol=symbol,
                quantity=quantity,
                side='buy',
                urgency=urgency,
                max_slippage=0.3,  # 30 bps max slippage
                time_limit=30
            )
            
            self.logger.info(f"Executing: BUY {quantity} {symbol} @ ~${current_price:.2f}")
            
            # Execute through smart router
            execution_metrics = await self.order_router.execute_order(order_request)
            
            if execution_metrics.success:
                # Record position
                self.active_positions[symbol] = {
                    'quantity': execution_metrics.filled_quantity,
                    'entry_price': execution_metrics.average_price,
                    'entry_time': datetime.now(self.config.TIMEZONE),
                    'position_size': execution_metrics.filled_quantity * execution_metrics.average_price,
                    'slippage': execution_metrics.slippage
                }
                
                # Add to pattern monitoring
                self.pattern_engine.add_position(symbol, execution_metrics.average_price)
                
                self.total_trades += 1
                self.session_stats['trades_executed'] += 1
                
                self.logger.info(f"Position opened: {symbol} - "
                               f"Filled: {execution_metrics.filled_quantity} @ "
                               f"${execution_metrics.average_price:.2f} "
                               f"(slippage: {execution_metrics.slippage:.2f} bps)")
            else:
                self.logger.warning(f"Failed to open position in {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}")
    
    async def _exit_position(self, symbol: str, urgency: float = 0.7):
        """Exit a position."""
        try:
            if symbol not in self.active_positions:
                return
            
            position = self.active_positions[symbol]
            
            # Create exit order
            order_request = OrderRequest(
                symbol=symbol,
                quantity=position['quantity'],
                side='sell',
                urgency=urgency,
                max_slippage=0.5,  # Accept more slippage on exit
                time_limit=60
            )
            
            self.logger.info(f"Exiting: SELL {position['quantity']} {symbol}")
            
            # Execute exit
            execution_metrics = await self.order_router.execute_order(order_request)
            
            if execution_metrics.filled_quantity > 0:
                # Calculate P&L
                exit_value = execution_metrics.filled_quantity * execution_metrics.average_price
                entry_value = position['quantity'] * position['entry_price']
                pnl = exit_value - entry_value
                pnl_pct = (pnl / entry_value) * 100
                
                self.daily_pnl += pnl
                self.session_stats['total_pnl'] += pnl
                
                if pnl > 0:
                    self.session_stats['winning_trades'] += 1
                
                self.logger.info(f"Position closed: {symbol} - "
                               f"P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%) "
                               f"Exit: ${execution_metrics.average_price:.2f}")
                
                # Record outcome for learning
                outcome = TradingOutcome(
                    timestamp=datetime.now(self.config.TIMEZONE),
                    symbol=symbol,
                    gap_features=self.gap_analyzer.get_gap_report(symbol),
                    predicted_continuation=0.7,  # Get from model
                    predicted_magnitude=2.5,  # Get from model
                    actual_continuation=pnl > 0,
                    actual_magnitude=pnl_pct,
                    entry_price=position['entry_price'],
                    exit_price=execution_metrics.average_price,
                    position_size=position['position_size'],
                    pnl=pnl,
                    regime=self.regime_classifier.current_regime,
                    execution_slippage=execution_metrics.slippage
                )
                
                self.online_learner.record_outcome(outcome)
                
                # Remove from tracking
                del self.active_positions[symbol]
                self.pattern_engine.remove_position(symbol)
                
            else:
                self.logger.error(f"Failed to exit position in {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error exiting position {symbol}: {e}")
    
    async def _monitor_positions(self):
        """Monitor active positions for exit signals."""
        if not self.active_positions:
            return
        
        # Scan for patterns
        alerts = self.pattern_engine.scan_patterns()
        
        # Process high-urgency alerts
        for alert in alerts:
            if alert.urgency > 0.85 and alert.symbol in self.active_positions:
                self.logger.warning(f"High urgency alert for {alert.symbol}: "
                                  f"{alert.pattern_type.value} - {alert.message}")
                
                # Exit position
                await self._exit_position(alert.symbol, urgency=alert.urgency)
        
        # Check time-based stops
        current_time = datetime.now(self.config.TIMEZONE)
        for symbol, position in list(self.active_positions.items()):
            time_in_position = (current_time - position['entry_time']).total_seconds() / 60
            
            if time_in_position > self.config.INITIAL_TIME_STOP_MINUTES:
                # Check if showing profit
                current_price = self._get_current_price(symbol)
                pnl_pct = ((current_price - position['entry_price']) / 
                          position['entry_price'] * 100)
                
                if pnl_pct < 0.5:  # Less than 0.5% profit after time stop
                    self.logger.info(f"Time stop triggered for {symbol}")
                    await self._exit_position(symbol, urgency=0.8)
    
    async def _check_secondary_opportunities(self):
        """Check for secondary gap opportunities during the day."""
        # Only if we have available capital
        used_capital = sum(p['position_size'] for p in self.active_positions.values())
        available_capital = self.portfolio_optimizer.portfolio_state.total_value - used_capital
        
        if available_capital > self.config.MIN_POSITION_SIZE * 2:
            # Quick gap scan
            gap_features = self.gap_analyzer.analyze_gaps()
            top_symbols, _ = self.gap_analyzer.get_top_gaps(n=5)
            
            # Filter for symbols not already held
            new_opportunities = [s for s in top_symbols if s not in self.active_positions]
            
            if new_opportunities:
                self.logger.info(f"Found {len(new_opportunities)} secondary opportunities")
                # Could execute if very high quality
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        if symbol in self.data_pipeline.symbol_to_idx:
            idx = self.data_pipeline.symbol_to_idx[symbol]
            return self.data_pipeline.realtime_features[idx, 0].item()
        
        # Fallback to API
        try:
            quote = self.order_router.api.get_latest_quote(symbol)
            return (quote.bidprice + quote.askprice) / 2
        except:
            return 0.0
    
    def _calculate_session_stats(self):
        """Calculate session statistics."""
        if self.session_stats['trades_executed'] > 0:
            win_rate = (self.session_stats['winning_trades'] / 
                       self.session_stats['trades_executed'])
            self.session_stats['win_rate'] = win_rate
            
            # Simple Sharpe approximation
            if self.session_stats['trades_executed'] > 5:
                returns = []  # Would calculate from actual returns
                self.session_stats['sharpe_ratio'] = 0.0  # Placeholder
    
    def _generate_daily_report(self):
        """Generate comprehensive daily trading report."""
        report_time = datetime.now(self.config.TIMEZONE)
        
        report = f"""
========================================
Gap Trading Daily Report
{report_time.strftime('%Y-%m-%d %H:%M:%S')} ET
========================================

Session Statistics:
  Duration: {(report_time - self.session_start_time).total_seconds() / 3600:.1f} hours
  Gaps Analyzed: {self.session_stats['gaps_analyzed']}
  Trades Executed: {self.session_stats['trades_executed']}
  
Performance:
  Total P&L: ${self.session_stats['total_pnl']:+,.2f}
  Win Rate: {self.session_stats.get('win_rate', 0):.1%}
  Winning Trades: {self.session_stats['winning_trades']}
  Losing Trades: {self.session_stats['trades_executed'] - self.session_stats['winning_trades']}
  
Risk Metrics:
  Max Drawdown: ${self.session_stats.get('max_drawdown', 0):,.2f}
  Sharpe Ratio: {self.session_stats.get('sharpe_ratio', 0):.2f}
  
Execution Quality:
  {self.order_router.get_execution_analytics()}
  
Learning Progress:
  {self.online_learner.get_learning_report()}
  
Market Regime:
  {self.regime_classifier.regime_history[-1] if self.regime_classifier.regime_history else 'No data'}
========================================
"""
        
        self.logger.info(report)
        
        # Save to file
        report_path = Path('/home/ubuntu/gap_trading_reports')
        report_path.mkdir(exist_ok=True)
        
        report_file = report_path / f"daily_report_{report_time.strftime('%Y%m%d')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
    def _persist_daily_performance(self):
        """Persist daily performance metrics to database."""
        try:
            # Calculate additional performance metrics
            session_duration = (datetime.now(self.config.TIMEZONE) - self.session_start_time).total_seconds() / 3600
            avg_trade_pnl = self.session_stats['total_pnl'] / max(self.session_stats['trades_executed'], 1)
            
            # Build comprehensive metadata
            metadata = {
                'session_duration_hours': float(session_duration),
                'gaps_analyzed': self.session_stats['gaps_analyzed'],
                'active_positions_count': len(self.active_positions),
                'pending_orders_count': len(self.pending_orders),
                'current_phase': self.current_phase,
                'system_stats': {
                    'gpu_memory_used': torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                    'gpu_memory_cached': torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
                },
                'execution_quality': {
                    'avg_slippage': getattr(self.order_router, 'avg_slippage', 0.0),
                    'fill_rate': getattr(self.order_router, 'fill_rate', 0.0),
                },
                'learning_progress': {
                    'model_updates': len(getattr(self.online_learner, 'performance_history', [])),
                    'best_model_accuracy': max([p.get('best_accuracy', 0) for p in getattr(self.online_learner, 'performance_history', [])], default=0),
                },
                'regime_info': {
                    'current_regime': getattr(self.regime_classifier, 'regime_history', ['unknown'])[-1] if hasattr(self.regime_classifier, 'regime_history') and self.regime_classifier.regime_history else 'unknown',
                    'regime_confidence': getattr(self.regime_classifier, 'regime_confidence', 0.0),
                },
                'trader_version': '1.0'
            }
            
            # Prepare database record
            performance_record = {
                'date': datetime.now(self.config.TIMEZONE).date(),
                'total_pnl': float(self.session_stats['total_pnl']),
                'total_trades': int(self.session_stats['trades_executed']),
                'winning_trades': int(self.session_stats['winning_trades']),
                'win_rate': float(self.session_stats.get('win_rate', 0)),
                'max_drawdown': float(self.session_stats.get('max_drawdown', 0)),
                'sharpe_ratio': float(self.session_stats.get('sharpe_ratio', 0)),
                'avg_trade_pnl': float(avg_trade_pnl),
                'metadata': metadata
            }
            
            # Insert daily performance
            insert_query = """
                INSERT INTO daily_performance (
                    date, total_pnl, total_trades, winning_trades, win_rate,
                    max_drawdown, sharpe_ratio, avg_trade_pnl, metadata
                ) VALUES (
                    %(date)s, %(total_pnl)s, %(total_trades)s, %(winning_trades)s,
                    %(win_rate)s, %(max_drawdown)s, %(sharpe_ratio)s, %(avg_trade_pnl)s, %(metadata)s
                )
                ON CONFLICT (date) DO UPDATE SET
                    total_pnl = EXCLUDED.total_pnl,
                    total_trades = EXCLUDED.total_trades,
                    winning_trades = EXCLUDED.winning_trades,
                    win_rate = EXCLUDED.win_rate,
                    max_drawdown = EXCLUDED.max_drawdown,
                    sharpe_ratio = EXCLUDED.sharpe_ratio,
                    avg_trade_pnl = EXCLUDED.avg_trade_pnl,
                    metadata = EXCLUDED.metadata
            """
            
            success = self.db_manager.execute_query(insert_query, performance_record)
            if success:
                self.logger.info(f"Persisted daily performance: ${self.session_stats['total_pnl']:+,.2f} PnL, {self.session_stats['trades_executed']} trades")
            else:
                self.logger.error("Failed to persist daily performance to database")
                
        except Exception as e:
            self.logger.error(f"Error persisting daily performance to database: {e}")
            # Continue processing even if database persistence fails
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info("Shutdown signal received...")
        self.is_running = False
    
    async def _shutdown(self):
        """Graceful shutdown procedure."""
        self.logger.info("Shutting down trading system...")
        
        # Close all positions
        if self.active_positions:
            self.logger.info("Closing all positions...")
            for symbol in list(self.active_positions.keys()):
                await self._exit_position(symbol, urgency=1.0)
        
        # Stop data streams
        if hasattr(self.data_pipeline, 'ws_connected') and self.data_pipeline.ws_connected:
            self.data_pipeline.stop_realtime_data()
        
        # Save final states
        self.online_learner.save_checkpoint('shutdown')
        
        # Generate final report
        self._generate_daily_report()
        
        self.logger.info("Shutdown complete")


# Example usage when file is run directly
if __name__ == "__main__":
    print("GPU-Accelerated Gap Trading System")
    print("=" * 50)
    
    # Create and run trader
    trader = GPUMainTrader()
    
    # Run async event loop
    try:
        asyncio.run(trader.run())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"\nCritical error: {e}")
        logging.error("Critical error", exc_info=True)
