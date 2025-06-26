#!/usr/bin/env python3
"""
Test script to verify FinGPT loading with proper memory management.
"""

import sys
import os
sys.path.append('/home/ubuntu/moe-1')

# Import CUDA initialization first
from universe_utils.cuda_init import full_cuda_initialization

# Initialize CUDA environment
print("🚀 Initializing CUDA environment...")
cuda_status = full_cuda_initialization()

# Now test FinGPT loading
from universe_utils.fingpt_sentiment import SimpleFinGPTManager
import torch

def test_fingpt_loading():
    """Test FinGPT model loading with memory monitoring."""
    
    print("\n" + "="*80)
    print("🧪 Testing FinGPT Model Loading")
    print("="*80)
    
    # Check initial GPU memory
    if torch.cuda.is_available():
        print(f"📊 Initial GPU Memory:")
        print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"   Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print(f"   Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize FinGPT manager
    print("\n🔧 Initializing FinGPT Manager...")
    manager = SimpleFinGPTManager()
    
    # Attempt to load model
    print("\n📥 Loading FinGPT Sentiment Model...")
    success = manager.load_sentiment_model()
    
    if success:
        print("✅ FinGPT model loaded successfully!")
        
        # Check GPU memory after loading
        if torch.cuda.is_available():
            print(f"\n📊 GPU Memory After Loading:")
            print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"   Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        
        # Test sentiment analysis
        print("\n🧪 Testing Sentiment Analysis...")
        test_text = "Apple reported strong quarterly earnings, beating analyst expectations."
        result = manager.analyze_sentiment(test_text)
        
        print(f"Test Text: {test_text}")
        print(f"Result: {result}")
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        manager.cleanup()
        
        if torch.cuda.is_available():
            print(f"\n📊 GPU Memory After Cleanup:")
            print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"   Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        
    else:
        print("❌ FinGPT model failed to load")
        print("🔄 Will use fallback sentiment analysis")
    
    return success

if __name__ == "__main__":
    test_fingpt_loading()
