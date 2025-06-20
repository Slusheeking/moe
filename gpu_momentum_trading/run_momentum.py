#!/usr/bin/env python3
"""
Run GPU-Accelerated Momentum Trading System
Production entry point
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify environment
def check_environment():
    """Verify all requirements are met"""
    errors = []
    
    # Check GPU
    try:
        import cupy as cp
        gpu_count = cp.cuda.runtime.getDeviceCount()
        if gpu_count == 0:
            errors.append("No GPU detected")
        else:
            print(f"✓ GPU detected: {gpu_count} device(s)")
    except ImportError:
        errors.append("CuPy not installed: pip install cupy-cuda11x")
    except Exception as e:
        errors.append(f"GPU error: {e}")
    
    # Check API keys
    if not os.getenv('POLYGON_API_KEY'):
        errors.append("POLYGON_API_KEY not set")
    else:
        print("✓ Polygon API key found")
    
    if not os.getenv('ALPACA_API_KEY'):
        errors.append("ALPACA_API_KEY not set")
    else:
        print("✓ Alpaca API key found")
    
    if not os.getenv('ALPACA_SECRET_KEY'):
        errors.append("ALPACA_SECRET_KEY not set")
    else:
        print("✓ Alpaca secret key found")
    
    # Check dependencies
    required_packages = [
        ('websockets', 'websockets'),
        ('aiohttp', 'aiohttp'),
        ('alpaca-py', 'alpaca'),
        ('numba', 'numba'),
        ('pytz', 'pytz')
    ]
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✓ {package_name} installed")
        except ImportError:
            errors.append(f"{package_name} not installed")
    
    if errors:
        print("\n❌ Environment check failed:")
        for error in errors:
            print(f"  - {error}")
        print("\nSetup instructions:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set API keys:")
        print("   export POLYGON_API_KEY='your_key'")
        print("   export ALPACA_API_KEY='your_key'")
        print("   export ALPACA_SECRET_KEY='your_secret'")
        return False
    
    print("\n✅ Environment check passed")
    return True


def main():
    """Main entry point"""
    print("=" * 60)
    print("GPU-ACCELERATED MOMENTUM TRADING SYSTEM")
    print("=" * 60)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    print("\nStarting system...")
    
    # Import and run
    try:
        import asyncio
        from momentum_system import main as run_system
        
        asyncio.run(run_system())
        
    except KeyboardInterrupt:
        print("\n\nSystem stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()