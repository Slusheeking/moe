-- Gap Trading System Database Schema
-- PostgreSQL 17.5 database for storing all trading data

-- Create database (must be run separately by superuser)
-- CREATE DATABASE gap_trading;

-- Connect to gap_trading database first, then run the rest

-- Universe tracking with enhanced Polygon.io features
CREATE TABLE universe_symbols (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL UNIQUE,
    company_name VARCHAR(255),
    sector VARCHAR(50),
    market_cap DECIMAL(20, 2),
    avg_volume BIGINT,
    gap_frequency DECIMAL(5, 4),
    gap_quality_score DECIMAL(5, 4),
    -- Enhanced features for Polygon.io integration
    options_volume BIGINT DEFAULT 0,
    institutional_footprint DECIMAL(5, 4) DEFAULT 0,
    options_interest_ratio DECIMAL(5, 4) DEFAULT 0,
    put_call_ratio DECIMAL(5, 4) DEFAULT 0,
    liquidity_score DECIMAL(5, 4) DEFAULT 0,
    volatility DECIMAL(6, 2) DEFAULT 0,
    atr DECIMAL(10, 4) DEFAULT 0,
    atr_percent DECIMAL(5, 2) DEFAULT 0,
    features JSON, -- Store comprehensive feature set
    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_symbol ON universe_symbols(symbol);
CREATE INDEX idx_active ON universe_symbols(is_active);

-- Gap events with enhanced Polygon.io features
CREATE TABLE gap_events (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    gap_size DECIMAL(6, 2) NOT NULL,
    gap_direction INTEGER NOT NULL, -- 1 for up, -1 for down
    gap_quality_score DECIMAL(5, 4),
    institutional_footprint DECIMAL(5, 4),
    volume_surge_score DECIMAL(5, 4),
    continuation_probability DECIMAL(5, 4),
    -- Enhanced options features from Polygon.io
    options_volume_ratio DECIMAL(5, 4),
    put_call_ratio DECIMAL(5, 4),
    options_flow_score DECIMAL(5, 4),
    unusual_options_score DECIMAL(5, 4),
    -- Legacy fields for compatibility
    gap_date DATE GENERATED ALWAYS AS (DATE(timestamp)) STORED,
    gap_time TIMESTAMP GENERATED ALWAYS AS (timestamp) STORED,
    previous_close DECIMAL(10, 2),
    pre_market_open DECIMAL(10, 2),
    market_open DECIMAL(10, 2),
    gap_percent DECIMAL(6, 2) GENERATED ALWAYS AS (gap_size) STORED,
    gap_size_atr DECIMAL(6, 2),
    pre_market_volume BIGINT,
    pre_market_trades INTEGER,
    predicted_continuation DECIMAL(5, 4),
    predicted_magnitude DECIMAL(6, 2),
    regime VARCHAR(20),
    metadata JSON, -- Store comprehensive gap analysis data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_gap UNIQUE(symbol, timestamp),
    CONSTRAINT fk_gap_symbol FOREIGN KEY (symbol) REFERENCES universe_symbols(symbol)
);

-- Enhanced indexes for Polygon.io features
CREATE INDEX idx_gap_timestamp ON gap_events(timestamp);
CREATE INDEX idx_gap_size ON gap_events(gap_size);
CREATE INDEX idx_gap_quality ON gap_events(gap_quality_score);
CREATE INDEX idx_options_flow ON gap_events(options_flow_score);
CREATE INDEX idx_institutional ON gap_events(institutional_footprint);
CREATE INDEX idx_gap_direction ON gap_events(gap_direction);
-- Legacy compatibility indexes
CREATE INDEX idx_gap_date ON gap_events(gap_date);
CREATE INDEX idx_gap_percent ON gap_events(gap_percent);

-- Trade executions
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    trade_id VARCHAR(50) UNIQUE NOT NULL,
    gap_event_id BIGINT,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    entry_time TIMESTAMP NOT NULL,
    entry_price DECIMAL(10, 4) NOT NULL,
    entry_slippage DECIMAL(6, 4),
    exit_time TIMESTAMP,
    exit_price DECIMAL(10, 4),
    exit_slippage DECIMAL(6, 4),
    position_size DECIMAL(12, 2),
    pnl DECIMAL(12, 2),
    pnl_percent DECIMAL(6, 2),
    commission DECIMAL(8, 2),
    max_favorable_excursion DECIMAL(6, 2),
    max_adverse_excursion DECIMAL(6, 2),
    time_to_peak_minutes INTEGER,
    exit_reason VARCHAR(50),
    pattern_alerts JSON,
    execution_metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_trade_gap_event FOREIGN KEY (gap_event_id) REFERENCES gap_events(id),
    CONSTRAINT fk_trade_symbol FOREIGN KEY (symbol) REFERENCES universe_symbols(symbol)
);

CREATE INDEX idx_entry_time ON trades(entry_time);
CREATE INDEX idx_symbol_trades ON trades(symbol);
CREATE INDEX idx_pnl ON trades(pnl);

-- Create trigger for updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_trades_updated_at BEFORE UPDATE ON trades
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Model predictions log
CREATE TABLE model_predictions (
    id SERIAL PRIMARY KEY,
    prediction_time TIMESTAMP NOT NULL,
    gap_event_id BIGINT,
    symbol VARCHAR(10) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    model_version VARCHAR(20),
    prediction_value DECIMAL(10, 6),
    confidence DECIMAL(5, 4),
    features_used JSON,
    actual_outcome DECIMAL(10, 6),
    prediction_error DECIMAL(10, 6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_prediction_gap_event FOREIGN KEY (gap_event_id) REFERENCES gap_events(id)
);

CREATE INDEX idx_prediction_time ON model_predictions(prediction_time);
CREATE INDEX idx_model_type ON model_predictions(model_type);

-- Enhanced performance metrics with Polygon.io features
CREATE TABLE daily_performance (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    total_pnl DECIMAL(12, 2),
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 4),
    max_drawdown DECIMAL(10, 2),
    sharpe_ratio DECIMAL(6, 3),
    avg_trade_pnl DECIMAL(10, 2),
    -- Enhanced Polygon.io metrics
    gaps_analyzed INTEGER DEFAULT 0,
    options_trades INTEGER DEFAULT 0,
    avg_options_flow_score DECIMAL(5, 4),
    institutional_activity_score DECIMAL(5, 4),
    tick_data_quality DECIMAL(5, 4),
    microstructure_score DECIMAL(5, 4),
    -- Legacy fields for compatibility
    trading_date DATE GENERATED ALWAYS AS (date) STORED,
    gross_profit DECIMAL(12, 2),
    gross_loss DECIMAL(12, 2),
    largest_win DECIMAL(10, 2),
    largest_loss DECIMAL(10, 2),
    total_commission DECIMAL(10, 2),
    portfolio_value DECIMAL(15, 2),
    profit_factor DECIMAL(6, 3),
    average_win DECIMAL(10, 2),
    average_loss DECIMAL(10, 2),
    execution_quality DECIMAL(5, 4),
    metadata JSON, -- Store comprehensive daily metrics
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_trading_date ON daily_performance(trading_date);

-- Risk metrics tracking
CREATE TABLE risk_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    portfolio_heat DECIMAL(5, 4),
    var_95 DECIMAL(10, 2),
    cvar_95 DECIMAL(10, 2),
    correlation_exposure DECIMAL(5, 4),
    sector_concentration JSON,
    largest_position_percent DECIMAL(5, 4),
    margin_used DECIMAL(12, 2),
    buying_power DECIMAL(12, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_timestamp ON risk_metrics(timestamp);

-- Model performance tracking
CREATE TABLE model_performance (
    id SERIAL PRIMARY KEY,
    evaluation_date DATE NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    model_version VARCHAR(20),
    accuracy DECIMAL(5, 4),
    precision_score DECIMAL(5, 4),
    recall_score DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    auc_roc DECIMAL(5, 4),
    sample_count INTEGER,
    feature_importance JSON,
    hyperparameters JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_model_eval UNIQUE(evaluation_date, model_type, model_version)
);

-- System events and alerts
CREATE TABLE system_events (
    id SERIAL PRIMARY KEY,
    event_time TIMESTAMP NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    component VARCHAR(50),
    message TEXT,
    details JSON,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_event_time ON system_events(event_time);
CREATE INDEX idx_event_type ON system_events(event_type);
CREATE INDEX idx_severity ON system_events(severity);

-- Enhanced execution quality metrics for Polygon.io integration
CREATE TABLE execution_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    avg_price DECIMAL(10, 4),
    slippage_bps DECIMAL(6, 2),
    execution_time_ms INTEGER,
    market_impact_bps DECIMAL(6, 2),
    order_type VARCHAR(30),
    success_rate DECIMAL(5, 4),
    predicted_slippage_bps DECIMAL(6, 2),
    actual_vs_predicted DECIMAL(6, 2),
    venue VARCHAR(20) DEFAULT 'alpaca',
    -- Enhanced Polygon.io microstructure metrics
    tick_spread_bps DECIMAL(6, 2),
    order_book_depth DECIMAL(10, 2),
    market_regime VARCHAR(20),
    volatility_at_execution DECIMAL(6, 2),
    institutional_flow_score DECIMAL(5, 4),
    -- Legacy compatibility fields
    trade_id VARCHAR(50),
    requested_quantity INTEGER GENERATED ALWAYS AS (quantity) STORED,
    filled_quantity INTEGER GENERATED ALWAYS AS (quantity) STORED,
    requested_price DECIMAL(10, 4),
    average_fill_price DECIMAL(10, 4) GENERATED ALWAYS AS (avg_price) STORED,
    execution_time_ms_legacy INTEGER GENERATED ALWAYS AS (execution_time_ms) STORED,
    rejected_reason VARCHAR(100),
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_execution_symbol FOREIGN KEY (symbol) REFERENCES universe_symbols(symbol)
);

-- Indexes for execution metrics
CREATE INDEX idx_execution_timestamp ON execution_metrics(timestamp);
CREATE INDEX idx_execution_symbol ON execution_metrics(symbol);
CREATE INDEX idx_execution_slippage ON execution_metrics(slippage_bps);
CREATE INDEX idx_execution_venue ON execution_metrics(venue);

-- Backtesting results
CREATE TABLE backtest_results (
    id SERIAL PRIMARY KEY,
    backtest_id VARCHAR(50) UNIQUE NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital DECIMAL(12, 2),
    final_capital DECIMAL(12, 2),
    total_return DECIMAL(8, 4),
    sharpe_ratio DECIMAL(6, 3),
    max_drawdown DECIMAL(8, 4),
    win_rate DECIMAL(5, 4),
    total_trades INTEGER,
    parameters JSON,
    equity_curve JSON,
    trade_log JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create views for common queries
CREATE VIEW current_positions AS
SELECT 
    t.symbol,
    t.quantity,
    t.entry_price,
    t.entry_time,
    t.position_size,
    ge.gap_percent,
    ge.gap_quality_score,
    EXTRACT(EPOCH FROM (NOW() - t.entry_time))/60 as minutes_held
FROM trades t
JOIN gap_events ge ON t.gap_event_id = ge.id
WHERE t.exit_time IS NULL
ORDER BY t.entry_time DESC;

CREATE VIEW today_performance AS
SELECT 
    COUNT(*) as total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(pnl) as total_pnl,
    AVG(pnl_percent) as avg_return,
    MAX(pnl) as best_trade,
    MIN(pnl) as worst_trade
FROM trades
WHERE DATE(entry_time) = CURRENT_DATE;

-- Create stored procedure for daily performance calculation
CREATE OR REPLACE FUNCTION calculate_daily_performance(trading_date DATE)
RETURNS VOID AS $$
BEGIN
    INSERT INTO daily_performance (
        trading_date,
        total_trades,
        winning_trades,
        total_pnl,
        gross_profit,
        gross_loss,
        largest_win,
        largest_loss,
        win_rate,
        profit_factor,
        average_win,
        average_loss
    )
    SELECT 
        trading_date,
        COUNT(*) as total_trades,
        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
        SUM(pnl) as total_pnl,
        SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit,
        SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END) as gross_loss,
        MAX(CASE WHEN pnl > 0 THEN pnl ELSE NULL END) as largest_win,
        MIN(CASE WHEN pnl < 0 THEN pnl ELSE NULL END) as largest_loss,
        CASE WHEN COUNT(*) > 0 THEN 
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::DECIMAL / COUNT(*)
        ELSE 0 END as win_rate,
        CASE WHEN SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) > 0 THEN
            ABS(SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) / 
                SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END))
        ELSE NULL END as profit_factor,
        AVG(CASE WHEN pnl > 0 THEN pnl ELSE NULL END) as average_win,
        AVG(CASE WHEN pnl < 0 THEN pnl ELSE NULL END) as average_loss
    FROM trades
    WHERE DATE(exit_time) = trading_date
    ON CONFLICT (trading_date) DO UPDATE SET
        total_trades = EXCLUDED.total_trades,
        winning_trades = EXCLUDED.winning_trades,
        total_pnl = EXCLUDED.total_pnl,
        gross_profit = EXCLUDED.gross_profit,
        gross_loss = EXCLUDED.gross_loss,
        largest_win = EXCLUDED.largest_win,
        largest_loss = EXCLUDED.largest_loss,
        win_rate = EXCLUDED.win_rate,
        profit_factor = EXCLUDED.profit_factor,
        average_win = EXCLUDED.average_win,
        average_loss = EXCLUDED.average_loss;
END;
$$ LANGUAGE plpgsql;

-- Create composite indexes for performance
CREATE INDEX idx_trades_composite ON trades(symbol, entry_time, pnl);
CREATE INDEX idx_gaps_composite ON gap_events(gap_date, gap_percent, gap_quality_score);
CREATE INDEX idx_predictions_composite ON model_predictions(symbol, prediction_time, model_type);

-- Database User and Security Setup (Run as superuser)
-- Create database and user according to .env configuration
-- CREATE DATABASE gap_trading;
-- CREATE USER gap_trader WITH PASSWORD 'gap_trading_secure_2025!';
-- ALTER USER gap_trader CREATEDB; -- Allow creating test databases
-- GRANT ALL PRIVILEGES ON DATABASE gap_trading TO gap_trader;

-- Connect to gap_trading database, then run:
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO gap_trader;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO gap_trader;
-- GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO gap_trader;

-- Additional security for Polygon.io integration
-- GRANT USAGE ON SCHEMA public TO gap_trader;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO gap_trader;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO gap_trader;