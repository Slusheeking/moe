#!/bin/bash
# Setup script for GPU Momentum Trading System

echo "======================================"
echo "GPU Momentum Trading System Setup"
echo "======================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "❌ No NVIDIA GPU detected or nvidia-smi not found"
    echo "This system requires an NVIDIA GPU (A10 recommended)"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

# Check API keys
echo ""
echo "Checking API keys..."

if [ -z "$POLYGON_API_KEY" ]; then
    echo "❌ POLYGON_API_KEY not set"
    echo "Get your key from: https://polygon.io/"
else
    echo "✓ POLYGON_API_KEY found"
fi

if [ -z "$ALPACA_API_KEY" ]; then
    echo "❌ ALPACA_API_KEY not set"
    echo "Get your keys from: https://alpaca.markets/"
else
    echo "✓ ALPACA_API_KEY found"
fi

if [ -z "$ALPACA_SECRET_KEY" ]; then
    echo "❌ ALPACA_SECRET_KEY not set"
else
    echo "✓ ALPACA_SECRET_KEY found"
fi

# Create .env template
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env template..."
    cat > .env << EOF
# API Keys
POLYGON_API_KEY=your_polygon_key_here
ALPACA_API_KEY=your_alpaca_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_here

# Optional Settings
LOG_LEVEL=INFO
EOF
    echo "✓ Created .env file - please add your API keys"
fi

# Test GPU with CuPy
echo ""
echo "Testing GPU with CuPy..."
python3 -c "import cupy as cp; print(f'✓ CuPy working with {cp.cuda.runtime.getDeviceCount()} GPU(s)')" 2>/dev/null || echo "❌ CuPy test failed"

echo ""
echo "======================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Add your API keys to .env file"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Run the system: python run_momentum.py"
echo "======================================"