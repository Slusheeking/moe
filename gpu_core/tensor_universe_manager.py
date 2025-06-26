#!/usr/bin/env python3
"""
GPU-Accelerated Universe Manager
Maintains trading universe as GPU-resident tensors for parallel processing.
Calculates gap tendency scores and filters stocks based on multiple criteria.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import requests
from typing import List, Dict, Tuple, Optional
import pytz
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.gpu_trading_config import GPUTradingConfig
from database.connection import get_db_manager


class TensorUniverseManager:
    """Manages trading universe using GPU-accelerated tensor operations."""
    
    def __init__(self, config: GPUTradingConfig):
        self.config = config
        self.device = config.device
        self.dtype = config.TENSOR_DTYPE
        
        # Database connection for universe persistence
        self.db_manager = get_db_manager()
        
        # Initialize universe storage
        self.universe_symbols: List[str] = []
        self.universe_data: Optional[torch.Tensor] = None
        self.feature_names: List[str] = []
        self.last_update: Optional[datetime] = None
        
        # Enhanced feature indices for Polygon integration
        self.feature_indices = {
            'price': 0,
            'volume': 1,
            'dollar_volume': 2,
            'atr': 3,
            'atr_percent': 4,
            'gap_frequency': 5,
            'avg_gap_size': 6,
            'gap_continuation_rate': 7,
            'volatility': 8,
            'liquidity_score': 9,
            'sector_strength': 10,
            'options_volume': 11,
            'institutional_footprint': 12,
            'options_interest_ratio': 13,
            'put_call_ratio': 14,
            'vwap': 15,
            'transactions': 16
        }
        self.feature_names = list(self.feature_indices.keys())
        
        # Load existing universe from database or file
        try:
            self._load_universe_from_database()
        except Exception as e:
            print(f"Could not load universe from database: {e}")
            # Fallback to file if database fails
            if self.config.RUSSELL_1000_FILE.exists():
                self._load_universe()
    
    def update_universe(self, force: bool = False) -> torch.Tensor:
        """Update trading universe if needed or forced."""
        current_time = datetime.now(self.config.TIMEZONE)
        
        # Check if update is needed
        if not force and self.last_update:
            days_since_update = (current_time - self.last_update).days
            if days_since_update < self.config.UNIVERSE_UPDATE_FREQUENCY_DAYS:
                print(f"Universe is current (updated {days_since_update} days ago)")
                return self.universe_data
        
        print("Updating trading universe...")
        
        # Get top stocks by dollar volume
        companies = self._get_top_symbols_by_volume()
        
        if not companies:
            print("No symbols found, using default set")
            companies = self._get_default_symbols()
        
        # Calculate features for each symbol
        universe_data = []
        valid_symbols = []
        
        for company in companies:
            symbol = company['symbol']
            features = self._calculate_symbol_features(symbol, company)
            if features is not None:
                universe_data.append(features)
                valid_symbols.append(symbol)
            
            # Limit size for GPU memory
            if len(valid_symbols) >= self.config.MAX_UNIVERSE_SIZE:
                break
        
        if universe_data:
            self.universe_symbols = valid_symbols
            self.universe_data = torch.tensor(universe_data, device=self.device, dtype=self.dtype)
            self.last_update = current_time
            
            # Create symbol-to-index mapping
            self.symbol_to_idx = {symbol: i for i, symbol in enumerate(self.universe_symbols)}
            
            # Save to database and file
            self._save_universe()
            
            print(f"Universe updated with {len(self.universe_symbols)} symbols")
        
        return self.universe_data
    
    def get_universe_tensor(self) -> Tuple[torch.Tensor, List[str]]:
        """Get universe data as tensor and symbol list."""
        if self.universe_data is None:
            self.update_universe()
        return self.universe_data, self.universe_symbols
    
    def get_symbol_features(self, symbol: str) -> Optional[torch.Tensor]:
        """Get features for a specific symbol."""
        if symbol in self.symbol_to_idx:
            idx = self.symbol_to_idx[symbol]
            return self.universe_data[idx]
        return None
    
    def _save_universe(self) -> None:
        """Save universe data to database and disk."""
        # Save to database first
        try:
            self._save_universe_to_database()
        except Exception as e:
            print(f"Error saving universe to database: {e}")
        
        # Convert to pandas for storage
        df = pd.DataFrame(
            self.universe_data.cpu().numpy(),
            index=self.universe_symbols,
            columns=self.feature_names
        )
        df['last_update'] = self.last_update
        
        # Universe data is updated directly in the JSON file by the universe updater
        print(f"Universe saved to database (JSON file updated separately)")
    
    def _load_universe(self) -> None:
        """Load universe data from JSON file."""
        try:
            with open(self.config.RUSSELL_1000_FILE, 'r') as f:
                universe_data = json.load(f)
            
            companies = universe_data.get('companies', [])
            if not companies:
                print(f"No companies found in {self.config.RUSSELL_1000_FILE}")
                return
            
            # Extract symbols and features
            self.universe_symbols = []
            features_list = []
            
            for company in companies:
                symbol = company.get('symbol')
                if not symbol:
                    continue
                    
                self.universe_symbols.append(symbol)
                
                # Extract features in the order defined by feature_indices
                feature_array = []
                for feature_name in self.feature_names:
                    value = company.get(feature_name, 0.0)
                    # Handle different data types
                    if isinstance(value, (int, float)):
                        feature_array.append(float(value))
                    else:
                        feature_array.append(0.0)
                
                features_list.append(feature_array)
            
            # Convert to tensor
            if features_list:
                self.universe_data = torch.tensor(
                    features_list,
                    device=self.device,
                    dtype=self.dtype
                )
                
                # Create symbol-to-index mapping
                self.symbol_to_idx = {symbol: i for i, symbol in enumerate(self.universe_symbols)}
                
                # Extract last update time from metadata
                metadata = universe_data.get('metadata', {})
                if 'last_updated' in metadata:
                    self.last_update = datetime.fromisoformat(metadata['last_updated'].replace('Z', '+00:00'))
                
                print(f"Loaded universe from {self.config.RUSSELL_1000_FILE}")
                print(f"  - {len(self.universe_symbols)} symbols")
                print(f"  - {len(self.feature_names)} features per symbol")
            else:
                print(f"No valid companies found in {self.config.RUSSELL_1000_FILE}")
            
        except Exception as e:
            print(f"Error loading universe from JSON: {e}")
    
    def _load_universe_from_database(self) -> None:
        """Load universe data from database."""
        try:
            # Get symbols from database
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Query universe symbols with their features
                cursor.execute("""
                    SELECT symbol, market_cap, average_volume, sector, 
                           gap_frequency, last_updated, features
                    FROM universe_symbols 
                    WHERE is_active = true
                    ORDER BY market_cap DESC
                """)
                
                results = cursor.fetchall()
                
                if not results:
                    print("No universe data found in database")
                    return
                
                # Extract symbols and features
                self.universe_symbols = []
                features_list = []
                
                for row in results:
                    symbol, market_cap, avg_volume, sector, gap_freq, last_updated, features_json = row
                    self.universe_symbols.append(symbol)
                    
                    # Parse features JSON or create default features
                    if features_json:
                        features = features_json
                    else:
                        # Create default features
                        features = {
                            'price': 100.0,  # Will be updated with real data
                            'volume': avg_volume,
                            'dollar_volume': avg_volume * 100.0,
                            'atr': 2.0,
                            'atr_percent': 2.0,
                            'gap_frequency': gap_freq,
                            'avg_gap_size': 2.0,
                            'gap_continuation_rate': 0.6,
                            'volatility': 20.0,
                            'liquidity_score': 0.8,
                            'sector_strength': 0.5,
                            'options_volume': avg_volume * 0.1
                        }
                    
                    # Convert to feature array
                    feature_array = [features.get(name, 0.0) for name in self.feature_names]
                    features_list.append(feature_array)
                
                # Convert to tensor
                if features_list:
                    self.universe_data = torch.tensor(
                        features_list, 
                        device=self.device, 
                        dtype=self.dtype
                    )
                    self.last_update = datetime.now(self.config.TIMEZONE)
                    
                    # Create symbol-to-index mapping
                    self.symbol_to_idx = {symbol: i for i, symbol in enumerate(self.universe_symbols)}
                    
                    print(f"Loaded {len(self.universe_symbols)} symbols from database")
                
        except Exception as e:
            print(f"Error loading universe from database: {e}")
            raise
    
    def _save_universe_to_database(self) -> None:
        """Save universe data to database."""
        if self.universe_data is None or len(self.universe_symbols) == 0:
            return
            
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Clear existing data
                cursor.execute("DELETE FROM universe_symbols")
                
                # Insert universe data
                for i, symbol in enumerate(self.universe_symbols):
                    features = self.universe_data[i].cpu().numpy()
                    
                    # Create features dictionary
                    features_dict = {name: float(features[j]) for j, name in enumerate(self.feature_names)}
                    
                    # Insert symbol with features
                    cursor.execute("""
                        INSERT INTO universe_symbols 
                        (symbol, market_cap, average_volume, sector, gap_frequency, 
                         last_updated, is_active, features)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        symbol,
                        float(features_dict.get('dollar_volume', 0)) / float(features_dict.get('price', 100)),  # market_cap approximation
                        float(features_dict.get('volume', 0)),
                        'UNKNOWN',  # Default sector
                        float(features_dict.get('gap_frequency', 0)),
                        self.last_update or datetime.now(self.config.TIMEZONE),
                        True,
                        features_dict
                    ))
                
                conn.commit()
                print(f"Saved {len(self.universe_symbols)} symbols to database")
                
        except Exception as e:
            print(f"Error saving universe to database: {e}")
            raise
    
    def _get_top_symbols_by_volume(self) -> List[Dict]:
        """Get top symbols by dollar volume from the Russell 1000 JSON file."""
        return self._get_russell_1000_data()

    def _get_russell_1000_data(self) -> List[Dict]:
        """Load company data from the Russell 1000 JSON file."""
        try:
            with open(self.config.RUSSELL_1000_FILE, 'r') as f:
                data = json.load(f)
            return data['companies']
        except Exception as e:
            print(f"Error loading Russell 1000 data: {e}")
            return []

    def _get_default_symbols(self) -> List[str]:
        """Fallback default symbol universe."""
        return [{'symbol': s} for s in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']]
    
    def _calculate_symbol_features(self, symbol: str, company_data: Dict) -> Optional[List[float]]:
        """Calculate features for a single symbol using data from the universe file."""
        try:
            features = [
                company_data.get('price', 0.0),
                company_data.get('volume', 0.0),
                company_data.get('dollar_volume', 0.0),
                company_data.get('atr', 0.0),
                company_data.get('atr_percent', 0.0),
                company_data.get('gap_frequency', 0.0),
                company_data.get('avg_gap_size', 0.0),
                company_data.get('gap_continuation_rate', 0.0),
                company_data.get('volatility', 0.0),
                company_data.get('liquidity_score', 0.0),
                company_data.get('sector_strength', 0.0),
                company_data.get('options_volume', 0.0)
            ]
            return features
        except Exception as e:
            print(f"Error calculating features for {symbol}: {e}")
            return None


# Example usage when file is run directly
if __name__ == "__main__":
    print("GPU-Accelerated Universe Manager Test")
    print("=" * 50)
    
    config = GPUTradingConfig()
    universe_manager = TensorUniverseManager(config)
    
    # Update universe
    print("\nUpdating universe...")
    universe_tensor = universe_manager.update_universe(force=True)
    
    print(f"\nUniverse loaded:")
    print(f"  Shape: {universe_tensor.shape}")
    print(f"  Device: {universe_tensor.device}")
    print(f"  Symbols: {len(universe_manager.universe_symbols)}")
    
    # Test feature access
    if universe_manager.universe_symbols:
        test_symbol = universe_manager.universe_symbols[0]
        features = universe_manager.get_symbol_features(test_symbol)
        print(f"\nTest symbol {test_symbol} features:")
        print(f"  Shape: {features.shape if features is not None else 'None'}")
        
        if features is not None:
            for i, name in enumerate(universe_manager.feature_names):
                print(f"  {name}: {features[i].item():.2f}")
