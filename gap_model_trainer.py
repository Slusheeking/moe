"""
GPU-Accelerated Multi-Horizon PatchTST Trading Model Trainer
Optimized for A10 GPU with RAPIDS cuDF, CuPy, PyTorch CUDA, and Numba
Implements ultra-fast PatchTST for 1-minute, 5-minute, and 10-minute directional prediction
"""

import os
import json
import warnings
import logging
from datetime import datetime, timedelta, time
import asyncio
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
import random
from dataclasses import dataclass
import pytz
from dotenv import load_dotenv

# GPU-accelerated PyTorch with optimizations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

# Enable TF32 and optimizations for A10 GPU
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed

# GPU-accelerated data processing with fallbacks
try:
    import cudf as pd
    import cupy as np
    from cudf import DataFrame
    print("🚀 Model Trainer: Using RAPIDS cuDF and CuPy for GPU acceleration!")
    GPU_AVAILABLE = True
except ImportError:
    import pandas as pd
    import numpy as np
    from pandas import DataFrame
    print("⚠️  Model Trainer: Using CPU fallback - install RAPIDS for GPU acceleration")
    GPU_AVAILABLE = False

try:
    from numba import cuda, jit
    NUMBA_AVAILABLE = cuda.is_available()
    if NUMBA_AVAILABLE:
        print(f"⚡ Model Trainer: Numba CUDA acceleration enabled for data preprocessing")
except ImportError:
    NUMBA_AVAILABLE = False

# GPU-accelerated scikit-learn alternative
try:
    from cuml.preprocessing import RobustScaler
    print("🔥 Using cuML RobustScaler for GPU-accelerated preprocessing!")
except ImportError:
    from sklearn.preprocessing import RobustScaler
    print("⚠️  Using sklearn RobustScaler - install cuML for GPU acceleration")

# Import our GPU-optimized clients
from polygon_data_client import PolygonClient, TickerFilter, calculate_gap_metrics

# Suppress GPU warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module='cudf')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable TF32 on Ampere GPUs for faster training
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@dataclass
class TrainingConfig:
    """Training configuration for multi-horizon PatchTST"""
    # Model parameters
    d_model: int = 256  # Reduced for better performance
    n_heads: int = 8
    n_layers: int = 3   # Reduced layers for faster training
    patch_length: int = 16
    stride: int = 8
    num_features: int = 46  # Updated for enterprise dataset + momentum features
    
    # Multi-horizon prediction
    prediction_horizons: List[int] = None  # [1, 5, 10] minute horizons
    num_classes: int = 3  # UP, DOWN, NEUTRAL
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    n_epochs_real: int = 30  # Training on real enterprise data
    gradient_clip: float = 1.0
    
    # Data parameters for 1-minute bars
    sequence_length: int = 150  # 2.5 hours of 1-minute data
    validation_split: float = 0.2
    
    # Enterprise dataset paths
    enterprise_data_dir: str = "data/midcap_universe"
    enterprise_metadata_path: str = "data/midcap_universe/metadata.json"
    
    # GPU optimization
    use_amp: bool = True  # Automatic mixed precision for A10
    num_workers: int = 4
    pin_memory: bool = True
    
    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = [1, 5, 10]

class EnhancedFeatureEngineering:
    """Advanced feature engineering for trading prediction"""
    
    @staticmethod
    def create_gap_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create specialized features for trading prediction"""
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Gap-specific features
        df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['gap_fill'] = ((df['low'] <= df['close'].shift(1)) & (df['overnight_gap'] > 0)) | \
                         ((df['high'] >= df['close'].shift(1)) & (df['overnight_gap'] < 0))
        
        # Intraday momentum
        df['open_to_high'] = (df['high'] - df['open']) / df['open']
        df['open_to_low'] = (df['low'] - df['open']) / df['open']
        df['close_vs_open'] = (df['close'] - df['open']) / df['open']
        
        # Volume features
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        df['dollar_volume'] = df['close'] * df['volume']
        df['volume_momentum'] = df['volume'].pct_change()
        
        # Volatility features
        df['true_range'] = df[['high', 'close']].shift(1).max(axis=1) - \
                           df[['low', 'close']].shift(1).min(axis=1)
        df['atr_14'] = df['true_range'].rolling(14).mean()
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_ratio'] = df['volatility_20'] / df['volatility_20'].rolling(60).mean()
        
        # Technical indicators optimized for gaps
        df['rsi'] = EnhancedFeatureEngineering._calculate_rsi(df['close'], 14)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # VWAP deviation
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
        
        # Time-based features
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute
            df['day_of_week'] = df.index.dayofweek
            df['is_premarket'] = ((df.index.hour >= 4) & (df.index.hour < 9)) | \
                                ((df.index.hour == 9) & (df.index.minute < 30))
            df['is_first_hour'] = (df.index.hour == 9) & (df.index.minute >= 30) | \
                                 (df.index.hour == 10) & (df.index.minute < 30)
        
        # Gap history features
        df['gap_streak'] = (df['overnight_gap'] > 0.01).astype(int).groupby(
            (df['overnight_gap'] <= 0.01).astype(int).cumsum()
        ).cumsum()
        
        # Rolling gap statistics
        df['gap_mean_5d'] = df['overnight_gap'].rolling(5).mean()
        df['gap_std_5d'] = df['overnight_gap'].rolling(5).std()
        df['gap_max_5d'] = df['overnight_gap'].rolling(5).max()
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

class EnterpriseDataset(Dataset):
    """GPU-Accelerated dataset for enterprise mid-cap universe data with multi-horizon labels"""
    
    def __init__(self, data_dir: str, config: TrainingConfig = None, train=True):
        self.config = config or TrainingConfig()
        self.data = []
        self.labels = []
        self.scaler = RobustScaler()
        self.train = train
        self.gpu_available = GPU_AVAILABLE
        
        # Load enterprise dataset with GPU acceleration
        self._load_enterprise_data(data_dir)
    
    def _load_enterprise_data(self, data_dir: str):
        """Load enterprise dataset from parquet files with GPU acceleration"""
        
        # Load metadata to understand the dataset
        metadata_path = Path(data_dir) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"🚀 Loading enterprise dataset: {metadata['schema']['total_records']} records")
        
        # Load data from partitioned parquet
        data_path = Path(data_dir) / "partitioned"
        if not data_path.exists():
            logger.error(f"Enterprise data not found at {data_path}")
            return
        
        # GPU-accelerated parquet loading
        if GPU_AVAILABLE:
            logger.info("🔥 Using GPU-accelerated parquet loading with cuDF")
            df = pd.read_parquet(data_path, engine='cudf')
        else:
            df = pd.read_parquet(data_path)
        
        logger.info(f"⚡ Loaded {len(df)} records from enterprise dataset using {'GPU' if GPU_AVAILABLE else 'CPU'}")
        
        # Define feature columns (matching our enterprise dataset + momentum features)
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'log_returns', 'overnight_gap', 'gap_fill',
            'open_to_high', 'open_to_low', 'close_vs_open',
            'volume_ma_20', 'volume_ratio', 'dollar_volume', 'volume_momentum',
            'true_range', 'atr_14', 'volatility_20', 'volatility_ratio',
            'rsi', 'rsi_oversold', 'rsi_overbought',
            'vwap', 'vwap_deviation',
            'hour', 'minute', 'day_of_week', 'is_premarket', 'is_first_hour',
            'gap_streak', 'gap_mean_5d', 'gap_std_5d', 'gap_max_5d',
            # Enhanced momentum detection features
            'price_momentum_5', 'price_momentum_20', 'volume_surge', 'volume_surge_signal',
            'price_range_20', 'close_position', 'breakout_signal',
            'relative_strength', 'momentum_strength',
            'day'  # Total: 46 features
        ]
        
        # Convert boolean columns to int
        bool_cols = ['gap_fill', 'is_premarket', 'is_first_hour', 'rsi_oversold', 'rsi_overbought']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        # Group by ticker to create sequences
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].sort_values('timestamp')
            
            if len(ticker_data) < self.config.sequence_length:
                continue
            
            # Extract features with GPU acceleration
            available_features = [col for col in feature_cols if col in ticker_data.columns]
            
            if GPU_AVAILABLE:
                # GPU-accelerated feature extraction
                features = ticker_data[available_features].fillna(0).values
                # Convert cuDF to CuPy array for faster processing
                if hasattr(features, 'get'):
                    features = features.get()  # cuDF to numpy
            else:
                features = ticker_data[available_features].fillna(0).values
            
            # GPU-accelerated feature padding
            if features.shape[1] < self.config.num_features:
                if GPU_AVAILABLE and 'cupy' in str(type(features)):
                    padding = np.zeros((features.shape[0], self.config.num_features - features.shape[1]))
                    features = np.hstack([features, padding])
                else:
                    padding = np.zeros((features.shape[0], self.config.num_features - features.shape[1]))
                    features = np.hstack([features, padding])
            elif features.shape[1] > self.config.num_features:
                features = features[:, :self.config.num_features]
            
            # Extract multi-horizon labels
            horizon_labels = {}
            for horizon in self.config.prediction_horizons:
                label_col = f'label_{horizon}m'
                if label_col in ticker_data.columns:
                    horizon_labels[horizon] = ticker_data[label_col].values
                else:
                    # Fallback: create neutral labels if column doesn't exist
                    logger.warning(f"Label column {label_col} not found, using neutral labels")
                    horizon_labels[horizon] = np.full(len(ticker_data), 2)  # NEUTRAL
            
            # Create sliding windows for training
            for i in range(len(features) - self.config.sequence_length + 1):
                sequence_features = features[i:i + self.config.sequence_length]
                
                # Get labels at the end of sequence for each horizon
                target_labels = {}
                for horizon in self.config.prediction_horizons:
                    if horizon in horizon_labels:
                        target_labels[horizon] = horizon_labels[horizon][i + self.config.sequence_length - 1]
                    else:
                        target_labels[horizon] = 2  # NEUTRAL
                
                # Skip samples with too many NEUTRAL labels for balanced training
                neutral_count = sum(1 for label in target_labels.values() if label == 2)
                if neutral_count == len(target_labels) and np.random.random() > 0.3:
                    continue
                
                self.data.append(sequence_features)
                self.labels.append(target_labels)
        
        if self.data:
            logger.info(f"Created {len(self.data)} training sequences from enterprise data")
            
            # Fit scaler on training data
            all_features = np.vstack(self.data)
            self.scaler.fit(all_features)
            self.data = [self.scaler.transform(d) for d in self.data]
            
            # Log label distribution for each horizon
            for horizon in self.config.prediction_horizons:
                horizon_counts = {}
                for label_dict in self.labels:
                    label = label_dict.get(horizon, 2)
                    horizon_counts[label] = horizon_counts.get(label, 0) + 1
                logger.info(f"Label distribution for {horizon}min: {horizon_counts}")
        else:
            logger.warning("No training sequences created from enterprise data")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.data[idx])
        
        # Create patches with overlap
        patches = features.unfold(0, self.config.patch_length, self.config.stride)
        patches = patches.reshape(patches.size(0), -1)
        
        # Convert multi-horizon labels to tensors
        labels = self.labels[idx]
        label_tensors = {}
        for horizon in self.config.prediction_horizons:
            label_tensors[f'label_{horizon}min'] = torch.LongTensor([labels.get(horizon, 2)])
        
        return {
            'patches': patches,
            **label_tensors
        }

class MultiHorizonPatchTSTModel(nn.Module):
    """Multi-horizon PatchTST model for directional price prediction"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.prediction_horizons = config.prediction_horizons
        
        # Calculate patch dimensions
        self.n_patches = (config.sequence_length - config.patch_length) // config.stride + 1
        self.patch_dim = config.patch_length * config.num_features
        
        # Patch embedding
        self.patch_embedding = nn.Linear(self.patch_dim, config.d_model)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.n_patches, config.d_model) * 0.02
        )
        
        # Transformer layers (shared backbone)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers
        )
        
        # Multi-horizon prediction heads
        self.prediction_heads = nn.ModuleDict()
        for horizon in self.prediction_horizons:
            self.prediction_heads[f'classifier_{horizon}min'] = nn.Sequential(
                nn.Linear(config.d_model, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, config.num_classes)  # UP, DOWN, NEUTRAL
            )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, patches):
        # Embed patches
        x = self.patch_embedding(patches)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]
        
        # Transform (shared backbone)
        x = self.transformer(x)
        
        # Global pooling
        x = x.mean(dim=1)
        
        # Multi-horizon predictions
        predictions = {}
        for horizon in self.prediction_horizons:
            head_name = f'classifier_{horizon}min'
            predictions[f'logits_{horizon}min'] = self.prediction_heads[head_name](x)
        
        return predictions

class MultiHorizonModelTrainer:
    """GPU-Optimized Multi-horizon PatchTST trainer for A10 GPU maximum performance"""
    
    def __init__(self, config: TrainingConfig, save_dir='models'):
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # GPU optimization setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            # A10 GPU optimizations
            torch.cuda.empty_cache()  # Clear cache
            torch.cuda.set_device(0)  # Use first GPU
            
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            logger.info(f"🚀 GPU ACCELERATION ENABLED:")
            logger.info(f"   Device: {gpu_name}")
            logger.info(f"   Memory: {gpu_memory:.2f} GB")
            logger.info(f"   CUDA Version: {torch.version.cuda}")
            logger.info(f"   TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
            logger.info(f"   cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
            
            # A10 specific optimizations
            if "A10" in gpu_name:
                logger.info("🔥 A10 GPU detected - applying specialized optimizations!")
                # Optimize batch size for A10's 24GB memory
                if config.batch_size > 64:
                    config.batch_size = min(config.batch_size, 128)  # Max batch size for A10
                    logger.info(f"   Optimized batch size for A10: {config.batch_size}")
        
        # Create model with A10 optimizations
        self.model = MultiHorizonPatchTSTModel(config)
        
        # Move to GPU with memory optimization
        self.model.to(self.device)
        if torch.cuda.is_available():
            # Compile model for A10 if PyTorch 2.0+
            if hasattr(torch, 'compile'):
                logger.info("🔥 Compiling model with PyTorch 2.0 for maximum A10 performance!")
                self.model = torch.compile(self.model, mode='max-autotune')
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate * 5,
            total_steps=config.n_epochs_real * 1000,  # Approximate
            pct_start=0.1
        )
        
        # Mixed precision training
        self.scaler = torch.amp.GradScaler('cuda') if config.use_amp else None
        
        # Loss functions for each horizon
        self.classification_losses = {}
        for horizon in config.prediction_horizons:
            # Balanced weights for UP/DOWN/NEUTRAL
            self.classification_losses[horizon] = nn.CrossEntropyLoss(
                weight=torch.tensor([1.5, 1.5, 1.0]).to(self.device)  # Weight UP/DOWN more than NEUTRAL
            )
        
        # Metrics tracking
        self.history = {
            'train_loss': [], 'val_loss': [],
            'learning_rates': []
        }
        
        # Add accuracy tracking for each horizon
        for horizon in config.prediction_horizons:
            self.history[f'train_acc_{horizon}min'] = []
            self.history[f'val_acc_{horizon}min'] = []
    
    def train_on_enterprise_data(self):
        """Train on enterprise mid-cap universe dataset"""
        logger.info("="*60)
        logger.info("Training multi-horizon PatchTST on enterprise data...")
        logger.info("="*60)
        
        # Load enterprise dataset
        dataset = EnterpriseDataset(self.config.enterprise_data_dir, self.config)
        
        if len(dataset) == 0:
            logger.error("No enterprise data available for training")
            return
        
        logger.info(f"Enterprise dataset contains {len(dataset)} training sequences")
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Save scaler from enterprise dataset
        with open(self.save_dir / 'enterprise_scaler.pkl', 'wb') as f:
            pickle.dump(dataset.scaler, f)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=True
        )
        
        # Reset learning rate scheduler for enterprise training
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate * 5,  # Lower max LR for real data
            total_steps=self.config.n_epochs_real * len(train_loader),
            pct_start=0.1
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.config.n_epochs_real):
            # Train
            train_metrics = self._train_epoch(train_loader)
            
            # Validate
            val_metrics = self._validate(val_loader)
            
            # Update history
            for key in train_metrics:
                if f'train_{key}' in self.history:
                    self.history[f'train_{key}'].append(train_metrics[key])
            for key in val_metrics:
                if f'val_{key}' in self.history:
                    self.history[f'val_{key}'].append(val_metrics[key])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log progress with multi-horizon accuracies
            log_msg = f"Epoch {epoch+1}/{self.config.n_epochs_real} - Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}"
            
            for horizon in self.config.prediction_horizons:
                if f'acc_{horizon}min' in train_metrics and f'acc_{horizon}min' in val_metrics:
                    log_msg += f", {horizon}min Acc: {train_metrics[f'acc_{horizon}min']:.1f}/{val_metrics[f'acc_{horizon}min']:.1f}%"
            
            log_msg += f", LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            logger.info(log_msg)
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_enterprise_model.pth', epoch, val_metrics)
        
        # Save final model
        self.save_checkpoint('enterprise_model_final.pth', epoch, val_metrics)

    def _train_epoch(self, dataloader):
        """Train for one epoch with multi-horizon targets"""
        self.model.train()
        
        total_loss = 0
        horizon_metrics = {horizon: {'correct': 0, 'total': 0} for horizon in self.config.prediction_horizons}
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            patches = batch['patches'].to(self.device)
            
            # Get multi-horizon targets
            targets = {}
            for horizon in self.config.prediction_horizons:
                label_key = f'label_{horizon}min'
                if label_key in batch:
                    targets[horizon] = batch[label_key].squeeze().to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.config.use_amp:
                with torch.amp.autocast('cuda'):
                    predictions = self.model(patches)
                    
                    # Calculate loss for each horizon
                    total_horizon_loss = 0
                    for horizon in self.config.prediction_horizons:
                        if horizon in targets:
                            logits_key = f'logits_{horizon}min'
                            if logits_key in predictions:
                                horizon_loss = self.classification_losses[horizon](
                                    predictions[logits_key], targets[horizon]
                                )
                                total_horizon_loss += horizon_loss
                    
                    loss = total_horizon_loss
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward/backward
                predictions = self.model(patches)
                
                # Calculate loss for each horizon
                total_horizon_loss = 0
                for horizon in self.config.prediction_horizons:
                    if horizon in targets:
                        logits_key = f'logits_{horizon}min'
                        if logits_key in predictions:
                            horizon_loss = self.classification_losses[horizon](
                                predictions[logits_key], targets[horizon]
                            )
                            total_horizon_loss += horizon_loss
                
                loss = total_horizon_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            
            # Update scheduler
            self.scheduler.step()
            
            # Track metrics for each horizon
            total_loss += loss.item()
            for horizon in self.config.prediction_horizons:
                if horizon in targets:
                    logits_key = f'logits_{horizon}min'
                    if logits_key in predictions:
                        _, predicted = predictions[logits_key].max(1)
                        horizon_metrics[horizon]['correct'] += predicted.eq(targets[horizon]).sum().item()
                        horizon_metrics[horizon]['total'] += targets[horizon].size(0)
        
        # Calculate accuracies
        result = {'loss': total_loss / len(dataloader)}
        for horizon in self.config.prediction_horizons:
            if horizon_metrics[horizon]['total'] > 0:
                acc = 100. * horizon_metrics[horizon]['correct'] / horizon_metrics[horizon]['total']
                result[f'acc_{horizon}min'] = acc
        
        return result
    
    def _validate(self, dataloader):
        """Validate model with multi-horizon targets"""
        self.model.eval()
        
        total_loss = 0
        horizon_metrics = {horizon: {'correct': 0, 'total': 0} for horizon in self.config.prediction_horizons}
        
        with torch.no_grad():
            for batch in dataloader:
                patches = batch['patches'].to(self.device)
                
                # Get multi-horizon targets
                targets = {}
                for horizon in self.config.prediction_horizons:
                    label_key = f'label_{horizon}min'
                    if label_key in batch:
                        targets[horizon] = batch[label_key].squeeze().to(self.device)
                
                if self.config.use_amp:
                    with torch.amp.autocast('cuda'):
                        predictions = self.model(patches)
                        
                        # Calculate loss for each horizon
                        total_horizon_loss = 0
                        for horizon in self.config.prediction_horizons:
                            if horizon in targets:
                                logits_key = f'logits_{horizon}min'
                                if logits_key in predictions:
                                    horizon_loss = self.classification_losses[horizon](
                                        predictions[logits_key], targets[horizon]
                                    )
                                    total_horizon_loss += horizon_loss
                        
                        loss = total_horizon_loss
                else:
                    predictions = self.model(patches)
                    
                    # Calculate loss for each horizon
                    total_horizon_loss = 0
                    for horizon in self.config.prediction_horizons:
                        if horizon in targets:
                            logits_key = f'logits_{horizon}min'
                            if logits_key in predictions:
                                horizon_loss = self.classification_losses[horizon](
                                    predictions[logits_key], targets[horizon]
                                )
                                total_horizon_loss += horizon_loss
                    
                    loss = total_horizon_loss
                
                total_loss += loss.item()
                
                # Track metrics for each horizon
                for horizon in self.config.prediction_horizons:
                    if horizon in targets:
                        logits_key = f'logits_{horizon}min'
                        if logits_key in predictions:
                            _, predicted = predictions[logits_key].max(1)
                            horizon_metrics[horizon]['correct'] += predicted.eq(targets[horizon]).sum().item()
                            horizon_metrics[horizon]['total'] += targets[horizon].size(0)
        
        # Calculate accuracies
        result = {'loss': total_loss / len(dataloader)}
        for horizon in self.config.prediction_horizons:
            if horizon_metrics[horizon]['total'] > 0:
                acc = 100. * horizon_metrics[horizon]['correct'] / horizon_metrics[horizon]['total']
                result[f'acc_{horizon}min'] = acc
        
        return result
    
    def save_checkpoint(self, filename, epoch, metrics):
        """Save model checkpoint with all necessary information"""
        save_path = self.save_dir / filename
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'metrics': metrics,
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")
        
        # Also save model config separately for easy loading
        config_path = self.save_dir / 'model_config.json'
        with open(config_path, 'w') as f:
            json.dump({
                'd_model': self.config.d_model,
                'n_heads': self.config.n_heads,
                'n_layers': self.config.n_layers,
                'patch_length': self.config.patch_length,
                'stride': self.config.stride,
                'num_features': self.config.num_features,
                'sequence_length': self.config.sequence_length,
                'prediction_horizons': self.config.prediction_horizons,
                'num_classes': self.config.num_classes
            }, f, indent=2)

def train_enterprise_model():
    """Synchronous wrapper for enterprise training"""
    
    # Configuration for enterprise training
    config = TrainingConfig(
        # Optimized for enterprise dataset
        batch_size=32,  # Smaller batch for better gradient updates
        learning_rate=1e-4,
        n_epochs_real=30,
        sequence_length=150,  # 2.5 hours of 1-minute data
        
        # Enterprise data paths
        enterprise_data_dir="data/midcap_universe",
        enterprise_metadata_path="data/midcap_universe/metadata.json"
    )
    
    # Check if enterprise data exists
    enterprise_data_path = Path(config.enterprise_data_dir) / "partitioned"
    if not enterprise_data_path.exists():
        logger.error(f"Enterprise data not found at {enterprise_data_path}")
        logger.info("Please run the data collection first: python run_data_collection.py")
        return None
    
    logger.info("="*60)
    logger.info("TRAINING MULTI-HORIZON PATCHTST MODEL")
    logger.info("="*60)
    logger.info(f"Prediction horizons: {config.prediction_horizons} minutes")
    logger.info(f"Model architecture: {config.d_model}d, {config.n_heads} heads, {config.n_layers} layers")
    
    # Initialize and train
    trainer = MultiHorizonModelTrainer(config)
    trainer.train_on_enterprise_data()
    
    # Save final results
    history_path = trainer.save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(trainer.history, f, indent=2, default=str)
    
    logger.info("\n" + "="*60)
    logger.info("MULTI-HORIZON TRAINING COMPLETE!")
    logger.info("="*60)
    
    # Log final accuracies for each horizon
    for horizon in config.prediction_horizons:
        val_acc_key = f'val_acc_{horizon}min'
        if val_acc_key in trainer.history and trainer.history[val_acc_key]:
            logger.info(f"Final {horizon}min validation accuracy: {trainer.history[val_acc_key][-1]:.2f}%")
    
    logger.info(f"\nModels saved in: {trainer.save_dir}")
    logger.info("- enterprise_model_final.pth: Final trained model")
    logger.info("- best_enterprise_model.pth: Best checkpoint")
    logger.info("- model_config.json: Model configuration")
    logger.info("- enterprise_scaler.pkl: Feature scaler")
    logger.info("- training_history.json: Training metrics")
    
    return trainer

async def main():
    """Main training pipeline for enterprise dataset"""
    
    # Configuration
    config = TrainingConfig()
    
    # Check if enterprise data exists
    enterprise_data_path = Path(config.enterprise_data_dir) / "partitioned"
    if not enterprise_data_path.exists():
        logger.error(f"Enterprise data not found at {enterprise_data_path}")
        logger.info("Please run the data collection first: python run_data_collection.py")
        return
    
    logger.info("="*60)
    logger.info("MULTI-HORIZON PATCHTST TRAINING PIPELINE")
    logger.info("="*60)
    logger.info(f"Using enterprise dataset: {config.enterprise_data_dir}")
    
    # Initialize trainer
    trainer = MultiHorizonModelTrainer(config)
    
    # Train on enterprise data
    logger.info("\nTraining multi-horizon PatchTST on enterprise data")
    trainer.train_on_enterprise_data()
    
    # Save training history
    history_path = trainer.save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(trainer.history, f, indent=2, default=str)
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("MULTI-HORIZON TRAINING COMPLETE!")
    logger.info("="*60)
    
    # Show next steps
    logger.info("\nNEXT STEPS:")
    logger.info("1. Test model: python test_system.py")
    logger.info("2. Run live scanning: python gap_scanner.py")
    logger.info("3. Monitor performance: Check training_history.json")

if __name__ == "__main__":
    # For enterprise training, use the synchronous wrapper
    train_enterprise_model()
    
    # Alternative: Use async version for full pipeline
    # asyncio.run(main())