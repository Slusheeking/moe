#!/usr/bin/env python3
"""
CUDA Environment Initialization Script
Properly configures CUDA environment to prevent warnings and errors.
"""

import os
import warnings
import logging

def setup_cuda_environment():
    """Setup CUDA environment variables and suppress warnings."""
    
    # Set LD_LIBRARY_PATH for BitsAndBytes
    import os
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    cuda_lib_path = '/usr/lib/x86_64-linux-gnu'
    if cuda_lib_path not in current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib_path}:{current_ld_path}" if current_ld_path else cuda_lib_path
    
    # Core CUDA environment variables
    cuda_env = {
        # CUDA Core Settings
        'CUDA_LAUNCH_BLOCKING': '0',
        'CUDA_CACHE_DISABLE': '0',
        'CUDA_MODULE_LOADING': 'LAZY',
        'CUDA_VISIBLE_DEVICES': '0',  # Use only first GPU
        
        # Memory Management - Enhanced for better memory efficiency
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128,expandable_segments:True,roundup_power2_divisions:16,garbage_collection_threshold:0.6',
        'TORCH_CUDA_ARCH_LIST': '7.0;7.5;8.0;8.6;9.0',
        
        # cuDNN Settings
        'CUDNN_DETERMINISTIC': '0',
        'CUDNN_BENCHMARK': '1',
        'CUDNN_ALLOW_TF32': '1',
        
        # cuBLAS Settings
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'CUBLASLT_LOG_LEVEL': '0',
        
        # Suppress CUDA Warnings
        'CUDA_SUPPRESS_WARNINGS': '1',
        'NCCL_SUPPRESS_WARN': '1',
        
        # TensorFlow Suppression
        'TF_CPP_MIN_LOG_LEVEL': '3',
        'TF_ENABLE_ONEDNN_OPTS': '0',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        
        # Transformers/HuggingFace Settings
        'TOKENIZERS_PARALLELISM': 'false',
        'TRANSFORMERS_VERBOSITY': 'error',
        'TRANSFORMERS_NO_ADVISORY_WARNINGS': '1',
        'HF_HUB_DISABLE_PROGRESS_BARS': '1',
        'HF_HUB_DISABLE_SYMLINKS_WARNING': '1',
        'HF_HUB_DISABLE_EXPERIMENTAL_WARNING': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1',
        
        # Accelerate Settings
        'ACCELERATE_DISABLE_RICH': '1',
        
        # PEFT Settings
        'PEFT_DISABLE_WARNINGS': '1',
        
        # Suppress Python Warnings
        'PYTHONWARNINGS': 'ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning',
    }
    
    # Apply environment variables
    for key, value in cuda_env.items():
        os.environ[key] = value
    
    # Configure Python warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', message='.*CUDA initialization.*')
    warnings.filterwarnings('ignore', message='.*cuDNN.*')
    warnings.filterwarnings('ignore', message='.*cuBLAS.*')
    warnings.filterwarnings('ignore', message='.*computation placer.*')
    warnings.filterwarnings('ignore', message='.*factory.*')
    warnings.filterwarnings('ignore', message='.*meta device.*')
    warnings.filterwarnings('ignore', message='.*device placement.*')
    
    # Configure logging to suppress CUDA messages
    cuda_loggers = [
        'torch.cuda',
        'torch.nn',
        'transformers.modeling_utils',
        'transformers.tokenization_utils_base',
        'accelerate.utils.modeling',
        'accelerate.big_modeling',
        'peft',
        'bitsandbytes'
    ]
    
    for logger_name in cuda_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False
    
    print("✅ CUDA environment configured successfully")

def initialize_torch_cuda():
    """Initialize PyTorch CUDA with proper settings."""
    try:
        import torch
        
        if torch.cuda.is_available():
            # Initialize CUDA context properly
            torch.cuda.init()
            
            # Set memory fraction to prevent OOM - very conservative
            torch.cuda.set_per_process_memory_fraction(0.6, device=0)
            
            # Enable memory optimization
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
            
            # Clear any existing cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            print(f"✅ PyTorch CUDA initialized: {torch.cuda.get_device_name(0)}")
            print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            return True
        else:
            print("⚠️ CUDA not available, using CPU")
            return False
            
    except Exception as e:
        print(f"❌ Error initializing PyTorch CUDA: {e}")
        return False

def setup_transformers_environment():
    """Setup transformers library environment."""
    try:
        # Import after CUDA setup
        import transformers
        
        # Configure transformers logging
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        
        # Disable warnings
        transformers.logging.disable_default_handler()
        
        print("✅ Transformers environment configured")
        return True
        
    except ImportError:
        print("⚠️ Transformers not available")
        return False
    except Exception as e:
        print(f"❌ Error configuring transformers: {e}")
        return False

def full_cuda_initialization():
    """Complete CUDA initialization sequence."""
    print("🚀 Initializing CUDA Environment...")
    
    # Step 1: Setup environment variables
    setup_cuda_environment()
    
    # Step 2: Initialize PyTorch CUDA
    cuda_available = initialize_torch_cuda()
    
    # Step 3: Setup transformers
    transformers_available = setup_transformers_environment()
    
    print("\n📋 Initialization Summary:")
    print(f"   CUDA Available: {'✅' if cuda_available else '❌'}")
    print(f"   Transformers: {'✅' if transformers_available else '❌'}")
    print(f"   Environment: ✅ Configured")
    
    return {
        'cuda_available': cuda_available,
        'transformers_available': transformers_available,
        'environment_configured': True
    }

if __name__ == "__main__":
    full_cuda_initialization()
