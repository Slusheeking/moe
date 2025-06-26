#!/usr/bin/env python3
"""
Standalone test script to verify FinGPT loading without other imports.
"""

import os
import sys
import torch
import gc
import warnings

# Set environment variables before any imports
os.environ.update({
    'CUDA_LAUNCH_BLOCKING': '0',
    'CUDA_CACHE_DISABLE': '0',
    'CUDA_MODULE_LOADING': 'LAZY',
    'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512,expandable_segments:True',
    'TORCH_CUDA_ARCH_LIST': '7.0;7.5;8.0;8.6',
    'CUDNN_DETERMINISTIC': '0',
    'CUDNN_BENCHMARK': '1',
    'TOKENIZERS_PARALLELISM': 'false',
    'TRANSFORMERS_VERBOSITY': 'error',
    'TRANSFORMERS_NO_ADVISORY_WARNINGS': '1',
    'TF_CPP_MIN_LOG_LEVEL': '3',
    'TF_ENABLE_ONEDNN_OPTS': '0',
    'HF_HUB_DISABLE_PROGRESS_BARS': '1',
    'HF_HUB_DISABLE_SYMLINKS_WARNING': '1',
    'HF_HUB_DISABLE_EXPERIMENTAL_WARNING': '1'
})

# Suppress warnings
warnings.filterwarnings('ignore')

def test_fingpt_direct():
    """Test FinGPT loading directly without other imports."""
    
    print("🚀 Direct FinGPT Loading Test")
    print("="*50)
    
    # Clear any existing CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        
        print(f"📊 Initial GPU Memory:")
        print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"   Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print(f"   Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Set conservative memory fraction
        torch.cuda.set_per_process_memory_fraction(0.5)  # Only use 50% of GPU memory
        print(f"   Memory fraction set to: 50%")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        
        print("\n📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "NousResearch/Llama-2-13b-hf",
            trust_remote_code=True,
            use_fast=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Tokenizer loaded")
        
        if torch.cuda.is_available():
            print(f"📊 Memory after tokenizer:")
            print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"   Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        
        print("\n📥 Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            "NousResearch/Llama-2-13b-hf",
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "10GB"},  # Conservative limit
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            load_in_8bit=True  # Use 8-bit quantization
        )
        
        print("✅ Base model loaded")
        
        if torch.cuda.is_available():
            print(f"📊 Memory after base model:")
            print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"   Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        
        # Try to load PEFT adapter
        peft_path = "/home/ubuntu/moe-1/FinGPT/fingpt/fingpt-sentiment_llama2-13b_lora"
        if os.path.exists(peft_path):
            print(f"\n📥 Loading PEFT adapter from {peft_path}...")
            model = PeftModel.from_pretrained(
                model,
                peft_path,
                torch_dtype=torch.float16,
                is_trainable=False
            )
            print("✅ PEFT adapter loaded")
        else:
            print(f"⚠️ PEFT adapter not found at {peft_path}")
        
        if torch.cuda.is_available():
            print(f"📊 Final memory usage:")
            print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"   Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        
        print("\n✅ FinGPT model loaded successfully!")
        
        # Test inference
        print("\n🧪 Testing inference...")
        from transformers import pipeline
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device=0 if torch.cuda.is_available() else -1
        )
        
        test_prompt = "Analyze sentiment: Apple beats earnings expectations. Sentiment:"
        result = pipe(test_prompt, max_new_tokens=10, do_sample=False)
        print(f"Test result: {result}")
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        del model, tokenizer, pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return False

if __name__ == "__main__":
    success = test_fingpt_direct()
    print(f"\n🎯 Test result: {'SUCCESS' if success else 'FAILED'}")
