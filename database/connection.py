#!/usr/bin/env python3
"""
PostgreSQL Database Connection Manager for GPU Trading System
Provides optimized connection pooling and database operations for high-frequency trading.
"""

import os
import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor, Json
from psycopg2 import sql
from typing import Dict, List, Any, Optional, Union
from contextlib import contextmanager
import logging
from datetime import datetime, date
from decimal import Decimal
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path('/home/ubuntu/moe-1/.env')
load_dotenv(dotenv_path=env_path)

class DatabaseConnectionManager:
    """
    High-performance PostgreSQL connection manager for trading system.
    Features connection pooling, automatic retries, and trading-specific operations.
    """
    
    def __init__(self, pool_size: int = 10, max_pool_size: int = 20):
        self.logger = logging.getLogger(__name__)
        
        # Database configuration from environment
        self.config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'gap_trading'),
            'user': os.getenv('DB_USER', 'gap_trader'),
            'password': os.getenv('DB_PASSWORD'),
            'sslmode': os.getenv('DB_SSL_MODE', 'prefer')
        }
        
        if not self.config['password']:
            raise ValueError("Database password not found in environment variables")
        
        # Connection pool
        self.pool = None
        self.pool_size = pool_size
        self.max_pool_size = max_pool_size
        
        self._initialize_pool()
        
    def _initialize_pool(self):
        """Initialize the connection pool."""
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=self.pool_size,
                maxconn=self.max_pool_size,
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password'],
                sslmode=self.config['sslmode'],
                cursor_factory=RealDictCursor
            )
            self.logger.info(f"Database connection pool initialized: {self.pool_size}-{self.max_pool_size} connections")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, commit: bool = True):
        """Context manager for database cursors."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
                if commit:
                    conn.commit()
            except Exception as e:
                conn.rollback()
                raise
            finally:
                cursor.close()
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results."""
        with self.get_cursor(commit=False) as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def execute_single(self, query: str, params: tuple = None) -> Optional[Dict[str, Any]]:
        """Execute a SELECT query and return single result."""
        with self.get_cursor(commit=False) as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()
    
    def execute_insert(self, query: str, params: tuple = None) -> Optional[int]:
        """Execute an INSERT query and return the ID if available."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            try:
                return cursor.fetchone()[0] if cursor.description else None
            except (TypeError, IndexError):
                return None
    
    def execute_update(self, query: str, params: tuple = None) -> int:
        """Execute an UPDATE/DELETE query and return affected rows."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount
    
    def execute_batch(self, query: str, param_list: List[tuple]) -> None:
        """Execute batch operations for better performance."""
        with self.get_cursor() as cursor:
            cursor.executemany(query, param_list)
    
    # Trading-specific database operations
    
    def insert_universe_symbol(self, symbol: str, company_name: str = None,
                             sector: str = None, market_cap: float = None,
                             avg_volume: int = None, gap_frequency: float = None,
                             gap_quality_score: float = None, **kwargs) -> int:
        """Insert a new universe symbol with enhanced Polygon.io features."""
        query = """
        INSERT INTO universe_symbols
        (symbol, company_name, sector, market_cap, avg_volume, gap_frequency, gap_quality_score,
         options_volume, institutional_footprint, options_interest_ratio, put_call_ratio,
         liquidity_score, volatility, atr, atr_percent, features)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol) DO UPDATE SET
            company_name = EXCLUDED.company_name,
            sector = EXCLUDED.sector,
            market_cap = EXCLUDED.market_cap,
            avg_volume = EXCLUDED.avg_volume,
            gap_frequency = EXCLUDED.gap_frequency,
            gap_quality_score = EXCLUDED.gap_quality_score,
            options_volume = EXCLUDED.options_volume,
            institutional_footprint = EXCLUDED.institutional_footprint,
            options_interest_ratio = EXCLUDED.options_interest_ratio,
            put_call_ratio = EXCLUDED.put_call_ratio,
            liquidity_score = EXCLUDED.liquidity_score,
            volatility = EXCLUDED.volatility,
            atr = EXCLUDED.atr,
            atr_percent = EXCLUDED.atr_percent,
            features = EXCLUDED.features,
            last_updated = CURRENT_TIMESTAMP
        RETURNING id;
        """
        return self.execute_insert(query, (
            symbol, company_name, sector, market_cap, avg_volume, gap_frequency, gap_quality_score,
            kwargs.get('options_volume', 0), kwargs.get('institutional_footprint', 0),
            kwargs.get('options_interest_ratio', 0), kwargs.get('put_call_ratio', 0),
            kwargs.get('liquidity_score', 0), kwargs.get('volatility', 0),
            kwargs.get('atr', 0), kwargs.get('atr_percent', 0),
            Json(kwargs.get('features', {}))
        ))
    
    def insert_gap_event(self, symbol: str, timestamp: datetime, gap_size: float,
                        gap_direction: int, **kwargs) -> int:
        """Insert a new gap event with enhanced Polygon.io features."""
        query = """
        INSERT INTO gap_events
        (symbol, timestamp, gap_size, gap_direction, gap_quality_score,
         institutional_footprint, volume_surge_score, continuation_probability,
         options_volume_ratio, put_call_ratio, options_flow_score, unusual_options_score,
         previous_close, pre_market_open, market_open, gap_size_atr, pre_market_volume,
         pre_market_trades, predicted_continuation, predicted_magnitude, regime, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, timestamp) DO UPDATE SET
            gap_size = EXCLUDED.gap_size,
            gap_direction = EXCLUDED.gap_direction,
            gap_quality_score = EXCLUDED.gap_quality_score,
            institutional_footprint = EXCLUDED.institutional_footprint,
            volume_surge_score = EXCLUDED.volume_surge_score,
            continuation_probability = EXCLUDED.continuation_probability,
            options_volume_ratio = EXCLUDED.options_volume_ratio,
            put_call_ratio = EXCLUDED.put_call_ratio,
            options_flow_score = EXCLUDED.options_flow_score,
            unusual_options_score = EXCLUDED.unusual_options_score,
            previous_close = EXCLUDED.previous_close,
            pre_market_open = EXCLUDED.pre_market_open,
            market_open = EXCLUDED.market_open,
            gap_size_atr = EXCLUDED.gap_size_atr,
            pre_market_volume = EXCLUDED.pre_market_volume,
            pre_market_trades = EXCLUDED.pre_market_trades,
            predicted_continuation = EXCLUDED.predicted_continuation,
            predicted_magnitude = EXCLUDED.predicted_magnitude,
            regime = EXCLUDED.regime,
            metadata = EXCLUDED.metadata
        RETURNING id;
        """
        return self.execute_insert(query, (
            symbol, timestamp, gap_size, gap_direction,
            kwargs.get('gap_quality_score'), kwargs.get('institutional_footprint'),
            kwargs.get('volume_surge_score'), kwargs.get('continuation_probability'),
            kwargs.get('options_volume_ratio'), kwargs.get('put_call_ratio'),
            kwargs.get('options_flow_score'), kwargs.get('unusual_options_score'),
            kwargs.get('previous_close'), kwargs.get('pre_market_open'), kwargs.get('market_open'),
            kwargs.get('gap_size_atr'), kwargs.get('pre_market_volume'), kwargs.get('pre_market_trades'),
            kwargs.get('predicted_continuation'), kwargs.get('predicted_magnitude'), kwargs.get('regime'),
            Json(kwargs.get('metadata', {}))
        ))
    
    def insert_trade(self, trade_id: str, symbol: str, side: str, quantity: int,
                    entry_time: datetime, entry_price: float, gap_event_id: int = None,
                    **kwargs) -> int:
        """Insert a new trade record."""
        query = """
        INSERT INTO trades 
        (trade_id, gap_event_id, symbol, side, quantity, entry_time, entry_price,
         entry_slippage, position_size, pattern_alerts, execution_metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        return self.execute_insert(query, (
            trade_id, gap_event_id, symbol, side, quantity, entry_time, entry_price,
            kwargs.get('entry_slippage'), kwargs.get('position_size'),
            Json(kwargs.get('pattern_alerts', {})),
            Json(kwargs.get('execution_metadata', {}))
        ))
    
    def update_trade_exit(self, trade_id: str, exit_time: datetime, exit_price: float,
                         pnl: float, pnl_percent: float, exit_reason: str,
                         **kwargs) -> int:
        """Update trade with exit information."""
        query = """
        UPDATE trades SET
            exit_time = %s,
            exit_price = %s,
            exit_slippage = %s,
            pnl = %s,
            pnl_percent = %s,
            commission = %s,
            max_favorable_excursion = %s,
            max_adverse_excursion = %s,
            time_to_peak_minutes = %s,
            exit_reason = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE trade_id = %s;
        """
        return self.execute_update(query, (
            exit_time, exit_price, kwargs.get('exit_slippage'), pnl, pnl_percent,
            kwargs.get('commission'), kwargs.get('max_favorable_excursion'),
            kwargs.get('max_adverse_excursion'), kwargs.get('time_to_peak_minutes'),
            exit_reason, trade_id
        ))
    
    def get_current_positions(self) -> List[Dict[str, Any]]:
        """Get all current open positions."""
        query = "SELECT * FROM current_positions;"
        return self.execute_query(query)
    
    def get_today_performance(self) -> Dict[str, Any]:
        """Get today's trading performance."""
        query = "SELECT * FROM today_performance;"
        result = self.execute_single(query)
        return dict(result) if result else {}
    
    def insert_model_prediction(self, symbol: str, model_type: str, 
                               prediction_value: float, confidence: float,
                               features_used: dict, gap_event_id: int = None,
                               model_version: str = None) -> int:
        """Insert model prediction."""
        query = """
        INSERT INTO model_predictions 
        (prediction_time, gap_event_id, symbol, model_type, model_version,
         prediction_value, confidence, features_used)
        VALUES (CURRENT_TIMESTAMP, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        return self.execute_insert(query, (
            gap_event_id, symbol, model_type, model_version,
            prediction_value, confidence, Json(features_used)
        ))
    
    def update_prediction_outcome(self, prediction_id: int, actual_outcome: float) -> int:
        """Update prediction with actual outcome."""
        query = """
        UPDATE model_predictions SET
            actual_outcome = %s,
            prediction_error = ABS(prediction_value - %s)
        WHERE id = %s;
        """
        return self.execute_update(query, (actual_outcome, actual_outcome, prediction_id))
    
    def insert_execution_metrics(self, symbol: str, side: str, quantity: int,
                                avg_price: float, slippage_bps: float,
                                execution_time_ms: int, **kwargs) -> int:
        """Insert enhanced execution metrics with Polygon.io features."""
        query = """
        INSERT INTO execution_metrics
        (timestamp, symbol, side, quantity, avg_price, slippage_bps, execution_time_ms,
         market_impact_bps, order_type, success_rate, predicted_slippage_bps,
         actual_vs_predicted, venue, tick_spread_bps, order_book_depth, market_regime,
         volatility_at_execution, institutional_flow_score, metadata)
        VALUES (CURRENT_TIMESTAMP, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        return self.execute_insert(query, (
            symbol, side, quantity, avg_price, slippage_bps, execution_time_ms,
            kwargs.get('market_impact_bps', 0), kwargs.get('order_type', 'market'),
            kwargs.get('success_rate', 1.0), kwargs.get('predicted_slippage_bps', 0),
            kwargs.get('actual_vs_predicted', 0), kwargs.get('venue', 'alpaca'),
            kwargs.get('tick_spread_bps', 0), kwargs.get('order_book_depth', 0),
            kwargs.get('market_regime', 'unknown'), kwargs.get('volatility_at_execution', 0),
            kwargs.get('institutional_flow_score', 0), Json(kwargs.get('metadata', {}))
        ))

    def insert_daily_performance(self, date: date, total_pnl: float, total_trades: int,
                                winning_trades: int, win_rate: float, **kwargs) -> int:
        """Insert enhanced daily performance with Polygon.io metrics."""
        query = """
        INSERT INTO daily_performance
        (date, total_pnl, total_trades, winning_trades, win_rate, max_drawdown,
         sharpe_ratio, avg_trade_pnl, gaps_analyzed, options_trades, avg_options_flow_score,
         institutional_activity_score, tick_data_quality, microstructure_score, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (date) DO UPDATE SET
            total_pnl = EXCLUDED.total_pnl,
            total_trades = EXCLUDED.total_trades,
            winning_trades = EXCLUDED.winning_trades,
            win_rate = EXCLUDED.win_rate,
            max_drawdown = EXCLUDED.max_drawdown,
            sharpe_ratio = EXCLUDED.sharpe_ratio,
            avg_trade_pnl = EXCLUDED.avg_trade_pnl,
            gaps_analyzed = EXCLUDED.gaps_analyzed,
            options_trades = EXCLUDED.options_trades,
            avg_options_flow_score = EXCLUDED.avg_options_flow_score,
            institutional_activity_score = EXCLUDED.institutional_activity_score,
            tick_data_quality = EXCLUDED.tick_data_quality,
            microstructure_score = EXCLUDED.microstructure_score,
            metadata = EXCLUDED.metadata
        RETURNING id;
        """
        return self.execute_insert(query, (
            date, total_pnl, total_trades, winning_trades, win_rate,
            kwargs.get('max_drawdown', 0), kwargs.get('sharpe_ratio', 0),
            kwargs.get('avg_trade_pnl', 0), kwargs.get('gaps_analyzed', 0),
            kwargs.get('options_trades', 0), kwargs.get('avg_options_flow_score', 0),
            kwargs.get('institutional_activity_score', 0), kwargs.get('tick_data_quality', 0),
            kwargs.get('microstructure_score', 0), Json(kwargs.get('metadata', {}))
        ))

    def insert_system_event(self, event_type: str, severity: str,
                           component: str = None, message: str = None,
                           details: dict = None) -> int:
        """Insert system event/alert."""
        query = """
        INSERT INTO system_events
        (event_time, event_type, severity, component, message, details)
        VALUES (CURRENT_TIMESTAMP, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        return self.execute_insert(query, (
            event_type, severity, component, message, Json(details or {})
        ))
    
    def get_active_universe(self) -> List[Dict[str, Any]]:
        """Get all active universe symbols."""
        query = "SELECT * FROM universe_symbols WHERE is_active = true ORDER BY symbol;"
        return self.execute_query(query)
    
    def get_recent_gaps(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get recent gap events."""
        query = """
        SELECT * FROM gap_events 
        WHERE gap_date >= CURRENT_DATE - INTERVAL '%s days'
        ORDER BY gap_date DESC, gap_time DESC;
        """
        return self.execute_query(query, (days,))
    
    def calculate_daily_performance(self, target_date: date = None) -> None:
        """Calculate and store daily performance metrics."""
        if target_date is None:
            target_date = date.today()
        
        with self.get_cursor() as cursor:
            cursor.execute("SELECT calculate_daily_performance(%s);", (target_date,))
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        try:
            with self.get_cursor(commit=False) as cursor:
                # Check connection
                cursor.execute("SELECT 1;")
                
                # Check table counts
                cursor.execute("""
                SELECT 
                    (SELECT COUNT(*) FROM universe_symbols) as universe_count,
                    (SELECT COUNT(*) FROM gap_events) as gap_events_count,
                    (SELECT COUNT(*) FROM trades) as trades_count,
                    (SELECT COUNT(*) FROM trades WHERE exit_time IS NULL) as open_positions;
                """)
                stats = cursor.fetchone()
                
                return {
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'universe_symbols': stats['universe_count'],
                    'gap_events': stats['gap_events_count'],
                    'total_trades': stats['trades_count'],
                    'open_positions': stats['open_positions'],
                    'pool_info': {
                        'size': f"{self.pool_size}-{self.max_pool_size}",
                        'available': len(self.pool._pool) if hasattr(self.pool, '_pool') else 'unknown'
                    }
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def close_pool(self):
        """Close the connection pool."""
        if self.pool:
            self.pool.closeall()
            self.logger.info("Database connection pool closed")


# Global instance for easy access
_db_manager = None

def get_db_manager() -> DatabaseConnectionManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseConnectionManager()
    return _db_manager

def close_db_connections():
    """Close all database connections."""
    global _db_manager
    if _db_manager:
        _db_manager.close_pool()
        _db_manager = None


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing Database Connection Manager")
    print("=" * 50)
    
    try:
        # Initialize database manager
        db = get_db_manager()
        
        # Health check
        health = db.health_check()
        print(f"Database Health: {health['status']}")
        for key, value in health.items():
            if key != 'status':
                print(f"  {key}: {value}")
        
        # Test basic operations
        print(f"\nTesting basic operations...")
        
        # Test universe query
        universe = db.get_active_universe()
        print(f"  Active universe symbols: {len(universe)}")
        
        # Test current positions
        positions = db.get_current_positions()
        print(f"  Current open positions: {len(positions)}")
        
        # Test today's performance
        performance = db.get_today_performance()
        print(f"  Today's trades: {performance.get('total_trades', 0)}")
        
        print(f"\n✅ Database connection tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        sys.exit(1)
    
    finally:
        close_db_connections()