#!/usr/bin/env python3
"""
Main script to update the Russell 1000 universe with enhanced features.
Uses the modularized universe_utils package with proper CUDA initialization.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to the path so we can import universe_utils
sys.path.append(str(Path(__file__).parent))

# CRITICAL: Initialize CUDA environment BEFORE any other imports
from cuda_init import full_cuda_initialization

# Initialize CUDA environment first
print("🔧 Initializing CUDA environment...")
cuda_status = full_cuda_initialization()

# Load environment variables
load_dotenv('/home/ubuntu/moe-1/.env')

# Configure logging AFTER CUDA initialization
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the universe updater AFTER CUDA initialization
from universe_utils import (
    CompleteUniverseCalculator, 
    check_system_status,
    update_universe
)


def main():
    """Main function to update the Russell 1000 universe."""
    print("\n" + "=" * 80)
    print("� Russell 1000 Universe Updater")
    print("�📊 Enhanced Features with FinGPT Integration")
    print("=" * 80 + "\n")
    
    # Check system status
    print("Checking system status...")
    status = check_system_status()
    print()
    
    # Verify environment variables
    required_env_vars = ['POLYGON_API_KEY', 'UW_API_TOKEN']
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please ensure they are set in /home/ubuntu/moe-1/.env")
        return 1
    
    print("✅ All required environment variables found")
    print()
    
    # Confirm before proceeding
    response = input("Do you want to update the Russell 1000 universe? (y/n): ")
    if response.lower() != 'y':
        print("Update cancelled.")
        return 0
    
    print()
    
    try:
        # Run the universe update
        update_universe()
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Update interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        print(f"\n❌ Update failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
