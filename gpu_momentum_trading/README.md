# GPU-Accelerated Momentum Trading System

A high-performance momentum trading system that leverages NVIDIA A10 GPU for real-time market scanning and execution.

## Key Features

- **GPU Acceleration**: Processes 10,000+ stocks in parallel using CUDA
- **Real-Time Detection**: Sub-second momentum detection across entire market
- **Smart Execution**: Limit orders only with strict cash management
- **Risk Management**: 5% trailing stops on all positions
- **No Predictions**: Pure momentum detection and following

## System Architecture

1. **GPU Momentum Scanner** (`gpu_momentum_scanner.py`)
   - Parallel processing of price/volume data
   - Real-time momentum scoring
   - Handles 15,000+ symbols simultaneously

2. **Market Data Manager** (`market_data_manager.py`)
   - Polygon WebSocket for real-time trades
   - Market snapshots every 30 seconds
   - Efficient batching to GPU

3. **Momentum Executor** (`realtime_executor.py`)
   - Immediate-or-Cancel limit orders
   - Automatic trailing stop placement
   - Strict position sizing

4. **Main System** (`momentum_system.py`)
   - Coordinates all components
   - Handles signal generation
   - Performance tracking

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Keys
```bash
export POLYGON_API_KEY='your_polygon_key'
export ALPACA_API_KEY='your_alpaca_key'
export ALPACA_SECRET_KEY='your_alpaca_secret'
```

### 3. Verify GPU
```bash
python -c "import cupy; print(f'GPU: {cupy.cuda.runtime.getDeviceCount()} devices')"
```

## Usage

### Run the System
```bash
python run_momentum.py
```

Or make it executable:
```bash
chmod +x run_momentum.py
./run_momentum.py
```

## Configuration

Edit `system_config.py` to adjust:
- Position sizing (10-20% per position)
- Momentum thresholds (40+ score minimum)
- Risk parameters (5% trailing stop)
- Market filters ($5-300 price range)

## Momentum Scoring

The system calculates momentum based on:
- **Price Velocity** (40 points max): Recent price movement
- **Volume Spike** (30 points max): Volume vs average
- **Acceleration** (20 points max): Increasing momentum
- **Consistency** (10 points max): Consecutive positive moves

Scores above 40 trigger analysis, above 60 are strong signals.

## Risk Management

- **Cash Only**: Never uses margin
- **Position Limits**: Max 10 concurrent positions
- **Trailing Stops**: 5% from highest point
- **Spread Filter**: Max 1% bid-ask spread
- **Market Hours**: 9:30 AM - 4:00 PM EST only

## Performance Expectations

- **Win Rate**: 35-45% (by design)
- **Average Win**: +15-25%
- **Average Loss**: -5% (trailing stop)
- **Risk/Reward**: 1:3 to 1:5

## Monitoring

The system logs:
- Every momentum signal detected
- All trade executions
- Position updates every 10 seconds
- Performance stats every minute

Check `momentum_trading.log` for details.

## Emergency Controls

- **Ctrl+C**: Graceful shutdown
- **Max Positions**: Hard limit of 10
- **Cash Reserve**: Always keeps $100 minimum

## Important Notes

1. This system does NOT predict momentum - it detects and follows
2. Losses are normal and controlled by trailing stops
3. Let winners run - no fixed profit targets
4. The GPU advantage is in scanning speed, not prediction

## Troubleshooting

### GPU Not Detected
```bash
nvidia-smi  # Check GPU status
```

### WebSocket Disconnects
- System auto-reconnects
- Check internet stability

### Orders Not Filling
- Check market hours
- Verify account has settled cash
- Review bid-ask spreads

## Support

This is a momentum detection system, not a prediction system. It will have losses - the key is that winners should be larger than losers through proper position management.