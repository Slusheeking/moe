# PostgreSQL Database Setup for GPU Trading System

## Overview
Successfully installed and configured PostgreSQL 17.5 for the GPU-accelerated gap trading system. The database is optimized for high-frequency trading operations with comprehensive schema for trading data, performance tracking, and real-time analytics.

## Installation Details

### PostgreSQL Version
- **Version**: PostgreSQL 17.5 (latest stable)
- **Installation Source**: Official PostgreSQL APT repository
- **Location**: `/var/lib/postgresql/17/main`

### Database Configuration
- **Database Name**: `gap_trading`
- **Database User**: `gap_trader`
- **Host**: `localhost`
- **Port**: `5432`
- **SSL Mode**: `prefer`

## Schema Overview

### Core Tables

#### 1. `universe_symbols`
- Tracks tradeable stocks in the universe
- Includes market cap, volume, gap frequency metrics
- **Primary Key**: `id (SERIAL)`
- **Unique Key**: `symbol`

#### 2. `gap_events`
- Records all detected gap trading opportunities
- Links to universe symbols
- Stores gap analysis and predictions
- **Primary Key**: `id (SERIAL)`
- **Unique Key**: `(symbol, gap_date)`

#### 3. `trades`
- Complete trade execution records
- Entry/exit prices, PnL, slippage tracking
- JSON fields for pattern alerts and metadata
- **Primary Key**: `id (SERIAL)`
- **Unique Key**: `trade_id`

#### 4. `model_predictions`
- ML model predictions and outcomes
- Performance tracking for model validation
- Feature importance storage

#### 5. `daily_performance`
- Aggregated daily trading metrics
- Sharpe ratio, win rate, profit factor
- Automatic calculation via stored procedure

#### 6. `risk_metrics`
- Portfolio heat, VaR, correlation tracking
- Real-time risk monitoring data

#### 7. `execution_metrics`
- Order execution quality analysis
- Slippage, market impact measurements

#### 8. `system_events`
- System alerts and event logging
- Component health monitoring

#### 9. `backtest_results`
- Historical strategy testing results
- Parameter optimization data

#### 10. `model_performance`
- ML model accuracy tracking
- Cross-validation results

### Views

#### `current_positions`
- Real-time view of open trading positions
- Calculates time held and gap metrics

#### `today_performance`
- Current day trading summary
- Live P&L and trade statistics

### Stored Procedures

#### `calculate_daily_performance(date)`
- Aggregates daily trading statistics
- Upserts to `daily_performance` table
- Handles win/loss ratios and profit factors

## Database Connection

### Python Integration
The system includes a comprehensive database connection manager:

**File**: `database/connection.py`

**Features**:
- Connection pooling (10-20 connections)
- Automatic retry logic
- Trading-specific helper methods
- Health monitoring
- Transaction management

**Usage Example**:
```python
from database.connection import get_db_manager

db = get_db_manager()

# Insert a trade
trade_id = db.insert_trade(
    trade_id="TRADE_001",
    symbol="AAPL", 
    side="BUY",
    quantity=100,
    entry_time=datetime.now(),
    entry_price=150.25
)

# Get current positions
positions = db.get_current_positions()

# Health check
health = db.health_check()
```

### Environment Configuration
Database credentials are stored in `.env`:

```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=gap_trading
DB_USER=gap_trader
DB_PASSWORD=gap_trading_secure_2025!
DB_SSL_MODE=prefer
```

## Performance Optimizations

### Indexes
- **Composite indexes** on frequently queried columns
- **Time-based indexes** for performance tracking
- **Symbol indexes** for fast lookups

### Connection Pooling
- Threaded connection pool (10-20 connections)
- Automatic connection recycling
- Prepared statement caching

### Data Types
- **DECIMAL** for financial precision
- **JSON** for flexible metadata storage
- **TIMESTAMP** with timezone support

## Security Features

### User Permissions
- Dedicated `gap_trader` user with minimal privileges
- No superuser access to production data
- SSL encryption enabled

### Password Security
- Strong password generation
- Environment variable storage
- No hardcoded credentials

## Monitoring & Health Checks

### Built-in Health Monitoring
```python
health = db.health_check()
# Returns:
# {
#   'status': 'healthy',
#   'universe_symbols': 0,
#   'gap_events': 0, 
#   'total_trades': 0,
#   'open_positions': 0,
#   'pool_info': {...}
# }
```

### System Events Table
All system events, errors, and alerts are logged to `system_events` table for analysis.

## Testing & Validation

### Connection Testing
```bash
cd /home/ubuntu/moe-1
python database/connection.py
```

### PostgreSQL Direct Access
```bash
# Connect as gap_trader
PGPASSWORD='gap_trading_secure_2025!' psql -h localhost -U gap_trader -d gap_trading

# Connect as postgres (admin)
sudo -u postgres psql -d gap_trading
```

### Configuration Validation
```bash
python -c "from config.gpu_trading_config import GPUTradingConfig; print(GPUTradingConfig().validate_configuration())"
```

## Integration with Trading System

### GPU Trading Config
Database settings are integrated into `config/gpu_trading_config.py`:

```python
# Database Configuration
self.DB_HOST = 'localhost'
self.DB_PORT = 5432
self.DB_NAME = 'gap_trading' 
self.DB_USER = 'gap_trader'
self.DB_PASSWORD = os.getenv('DB_PASSWORD')
```

### Real-time Data Flow
1. **Gap Detection** → `gap_events` table
2. **Trade Execution** → `trades` table  
3. **Model Predictions** → `model_predictions` table
4. **Performance Tracking** → `daily_performance` table
5. **Risk Monitoring** → `risk_metrics` table

## Maintenance

### Daily Tasks
- Daily performance calculation (automated)
- Connection pool monitoring
- Log rotation

### Weekly Tasks  
- Database statistics update
- Index maintenance
- Backup verification

### Backup Strategy
- PostgreSQL built-in `pg_dump` for logical backups
- Point-in-time recovery with WAL archiving
- Regular backup testing

## Troubleshooting

### Common Issues

**Connection Refused**:
```bash
sudo systemctl status postgresql
sudo systemctl start postgresql
```

**Permission Denied**:
```bash
sudo -u postgres psql -c "GRANT ALL ON ALL TABLES IN SCHEMA public TO gap_trader;"
```

**Pool Exhausted**:
- Check connection pool settings in `connection.py`
- Monitor long-running transactions
- Adjust pool size if needed

### Logs Location
- PostgreSQL logs: `/var/log/postgresql/`
- Application logs: Configurable via Python logging

## Next Steps

The database is now ready for:
1. ✅ **Universe Management** - Add/update tradeable symbols
2. ✅ **Gap Detection** - Store identified trading opportunities  
3. ✅ **Trade Execution** - Record all trading activity
4. ✅ **Performance Analytics** - Track strategy effectiveness
5. ✅ **Risk Management** - Monitor portfolio exposure
6. ✅ **Model Training** - Store ML predictions and outcomes

## Files Created/Modified

### New Files
- `database/schema_postgresql.sql` - PostgreSQL schema
- `database/connection.py` - Python database manager
- `database/README.md` - This documentation

### Modified Files  
- `.env` - Added database credentials
- `config/gpu_trading_config.py` - Added database configuration

---

**Setup Completed**: ✅ PostgreSQL 17.5 installation and configuration complete
**Status**: Ready for production trading system integration
**Last Updated**: June 25, 2025