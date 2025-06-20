"""
GPU-Accelerated Momentum Scanner
Leverages NVIDIA A10 for parallel processing of 10,000+ stocks
"""

import cupy as cp
import numpy as np
from numba import cuda
import asyncio
import logging
from typing import Dict, List, Tuple
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class GPUMomentumScanner:
    """Ultra-fast momentum detection using GPU parallel processing"""
    
    def __init__(self, max_symbols: int = 15000):
        self.max_symbols = max_symbols
        self.symbol_map = {}  # symbol -> index mapping
        self.reverse_map = {}  # index -> symbol mapping
        self.active_count = 0
        
        # GPU arrays for real-time data
        self.prices = cp.zeros((max_symbols, 20), dtype=cp.float32)  # Last 20 prices
        self.volumes = cp.zeros((max_symbols, 20), dtype=cp.float32)  # Last 20 volumes
        self.timestamps = cp.zeros((max_symbols, 20), dtype=cp.float64)  # Timestamps
        self.momentum_scores = cp.zeros(max_symbols, dtype=cp.float32)  # Output scores
        
        # Price/volume tracking indices
        self.current_idx = cp.zeros(max_symbols, dtype=cp.int32)
        
        # GPU kernel compilation
        self._compile_kernels()
        
        logger.info(f"GPU Momentum Scanner initialized for {max_symbols} symbols")
    
    def _compile_kernels(self):
        """Compile CUDA kernels for momentum calculation"""
        
        # Momentum calculation kernel
        self.momentum_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void calculate_momentum(
            float* prices, float* volumes, float* scores,
            int* current_idx, int max_symbols, int history_size
        ) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= max_symbols) return;
            
            int idx = current_idx[tid];
            if (idx < 5) {  // Need at least 5 data points
                scores[tid] = 0.0f;
                return;
            }
            
            // Get price slice for this symbol
            float* symbol_prices = &prices[tid * history_size];
            float* symbol_volumes = &volumes[tid * history_size];
            
            // Calculate momentum components
            float current_price = symbol_prices[idx % history_size];
            float prev_price = symbol_prices[(idx - 1) % history_size];
            float old_price = symbol_prices[(idx - 5) % history_size];
            
            // Price velocity (1 bar and 5 bar)
            float velocity_1 = (current_price - prev_price) / prev_price;
            float velocity_5 = (current_price - old_price) / old_price;
            
            // Volume analysis
            float current_vol = symbol_volumes[idx % history_size];
            float avg_vol = 0.0f;
            for (int i = 1; i <= 5; i++) {
                avg_vol += symbol_volumes[(idx - i) % history_size];
            }
            avg_vol /= 5.0f;
            float volume_ratio = current_vol / (avg_vol + 1.0f);
            
            // Acceleration
            float accel = velocity_1 - (prev_price - symbol_prices[(idx - 2) % history_size]) / symbol_prices[(idx - 2) % history_size];
            
            // Calculate momentum score
            float score = 0.0f;
            
            // Price momentum (weighted)
            if (velocity_5 > 0.10f) score += 40.0f;  // 10%+ move
            else if (velocity_5 > 0.05f) score += 30.0f;  // 5%+ move
            else if (velocity_5 > 0.03f) score += 20.0f;  // 3%+ move
            else if (velocity_5 > 0.01f) score += 10.0f;  // 1%+ move
            
            // Volume confirmation
            if (volume_ratio > 5.0f) score += 30.0f;  // 5x volume
            else if (volume_ratio > 3.0f) score += 25.0f;  // 3x volume
            else if (volume_ratio > 2.0f) score += 20.0f;  // 2x volume
            else if (volume_ratio > 1.5f) score += 10.0f;  // 1.5x volume
            
            // Acceleration bonus
            if (accel > 0.01f) score += 20.0f;  // Accelerating
            
            // Consistency check - all recent moves positive
            int positive_moves = 0;
            for (int i = 0; i < 3; i++) {
                float p1 = symbol_prices[(idx - i) % history_size];
                float p2 = symbol_prices[(idx - i - 1) % history_size];
                if (p1 > p2) positive_moves++;
            }
            if (positive_moves >= 3) score += 10.0f;
            
            scores[tid] = score;
        }
        ''', 'calculate_momentum')
        
        # Top movers extraction kernel
        self.top_movers_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void extract_top_movers(
            float* scores, int* indices, float* top_scores, int* top_indices,
            int max_symbols, int top_n
        ) {
            extern __shared__ float shared_data[];
            float* shared_scores = shared_data;
            int* shared_indices = (int*)&shared_scores[blockDim.x];
            
            int tid = threadIdx.x;
            int gid = blockIdx.x * blockDim.x + threadIdx.x;
            
            // Load data into shared memory
            if (gid < max_symbols) {
                shared_scores[tid] = scores[gid];
                shared_indices[tid] = gid;
            } else {
                shared_scores[tid] = -1.0f;
                shared_indices[tid] = -1;
            }
            __syncthreads();
            
            // Parallel bitonic sort in shared memory
            for (int k = 2; k <= blockDim.x; k <<= 1) {
                for (int j = k >> 1; j > 0; j >>= 1) {
                    int ixj = tid ^ j;
                    if (ixj > tid) {
                        if ((tid & k) == 0) {
                            if (shared_scores[tid] < shared_scores[ixj]) {
                                float temp_score = shared_scores[tid];
                                shared_scores[tid] = shared_scores[ixj];
                                shared_scores[ixj] = temp_score;
                                
                                int temp_idx = shared_indices[tid];
                                shared_indices[tid] = shared_indices[ixj];
                                shared_indices[ixj] = temp_idx;
                            }
                        } else {
                            if (shared_scores[tid] > shared_scores[ixj]) {
                                float temp_score = shared_scores[tid];
                                shared_scores[tid] = shared_scores[ixj];
                                shared_scores[ixj] = temp_score;
                                
                                int temp_idx = shared_indices[tid];
                                shared_indices[tid] = shared_indices[ixj];
                                shared_indices[ixj] = temp_idx;
                            }
                        }
                    }
                    __syncthreads();
                }
            }
            
            // Write top results
            if (tid < top_n && blockIdx.x == 0) {
                top_scores[tid] = shared_scores[tid];
                top_indices[tid] = shared_indices[tid];
            }
        }
        ''', 'extract_top_movers')
    
    def update_price(self, symbol: str, price: float, volume: float, timestamp: float):
        """Update price data for a symbol"""
        if symbol not in self.symbol_map:
            if self.active_count >= self.max_symbols:
                return  # Skip if at capacity
            
            self.symbol_map[symbol] = self.active_count
            self.reverse_map[self.active_count] = symbol
            self.active_count += 1
        
        idx = self.symbol_map[symbol]
        current = int(self.current_idx[idx])
        pos = current % 20
        
        # Update arrays
        self.prices[idx, pos] = price
        self.volumes[idx, pos] = volume
        self.timestamps[idx, pos] = timestamp
        self.current_idx[idx] = current + 1
    
    def calculate_all_momentum(self) -> Dict[str, float]:
        """Calculate momentum for all symbols in parallel on GPU"""
        if self.active_count == 0:
            return {}
        
        # Launch kernel
        threads_per_block = 256
        blocks = (self.active_count + threads_per_block - 1) // threads_per_block
        
        self.momentum_kernel(
            (blocks,), (threads_per_block,),
            (self.prices, self.volumes, self.momentum_scores,
             self.current_idx, self.active_count, 20)
        )
        
        # Copy results back
        scores = cp.asnumpy(self.momentum_scores[:self.active_count])
        
        # Build result dict
        results = {}
        for i in range(self.active_count):
            if scores[i] > 0:
                symbol = self.reverse_map.get(i)
                if symbol:
                    results[symbol] = float(scores[i])
        
        return results
    
    def get_top_movers(self, n: int = 100) -> List[Tuple[str, float]]:
        """Get top N momentum stocks using GPU sorting"""
        if self.active_count == 0:
            return []
        
        # Calculate momentum first
        threads_per_block = 256
        blocks = (self.active_count + threads_per_block - 1) // threads_per_block
        
        self.momentum_kernel(
            (blocks,), (threads_per_block,),
            (self.prices, self.volumes, self.momentum_scores,
             self.current_idx, self.active_count, 20)
        )
        
        # Use CuPy's built-in sorting (optimized for GPU)
        scores = self.momentum_scores[:self.active_count]
        indices = cp.argsort(scores)[::-1][:n]  # Top N indices
        
        top_scores = scores[indices]
        top_indices = cp.asnumpy(indices)
        top_scores_cpu = cp.asnumpy(top_scores)
        
        # Build results
        results = []
        for i, (idx, score) in enumerate(zip(top_indices, top_scores_cpu)):
            if score > 0:
                symbol = self.reverse_map.get(idx)
                if symbol:
                    results.append((symbol, float(score)))
        
        return results
    
    def get_symbol_data(self, symbol: str) -> Dict:
        """Get current data for a specific symbol"""
        if symbol not in self.symbol_map:
            return {}
        
        idx = self.symbol_map[symbol]
        current = int(self.current_idx[idx])
        
        if current < 2:
            return {}
        
        # Get latest data
        pos = (current - 1) % 20
        prev_pos = (current - 2) % 20
        
        price = float(self.prices[idx, pos])
        prev_price = float(self.prices[idx, prev_pos])
        volume = float(self.volumes[idx, pos])
        score = float(self.momentum_scores[idx])
        
        return {
            'symbol': symbol,
            'price': price,
            'prev_price': prev_price,
            'change_pct': (price - prev_price) / prev_price * 100,
            'volume': volume,
            'momentum_score': score
        }
    
    def batch_update(self, updates: List[Dict]):
        """Batch update multiple symbols at once"""
        for update in updates:
            self.update_price(
                update['symbol'],
                update['price'],
                update['volume'],
                update.get('timestamp', datetime.now().timestamp())
            )
    
    def clear_stale_data(self, max_age_seconds: int = 300):
        """Clear symbols with no recent updates"""
        current_time = datetime.now().timestamp()
        
        for i in range(self.active_count):
            idx = int(self.current_idx[i])
            if idx > 0:
                last_pos = (idx - 1) % 20
                last_time = float(self.timestamps[i, last_pos])
                
                if current_time - last_time > max_age_seconds:
                    # Reset this slot
                    self.current_idx[i] = 0
                    self.momentum_scores[i] = 0