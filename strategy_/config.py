# config.py
import os
from typing import Dict, List




class TradingConfig:
    """Centralized configuration for multi-crypto trading system"""
    
    def __init__(self):
        # Load from environment variables if available
        self.API_KEY = os.getenv('BINANCE_API_KEY', 'EduyybaFGjUpSkR7q2J0HwHjHF6dB8TB5klAAUX8Ukum2Yz1jR2J8osZVXz9kxZC')
        self.API_SECRET = os.getenv('BINANCE_API_SECRET', 'QmAxhDG4QYxdrif38WyQ6uvGLv5OZvlGPIRBzdtFWry7adtRNzGFY8HlLkOSLOyY')
        self.MODEL_PATH = os.getenv('MODEL_PATH', 'models/trading_lstm_20250701_233903.pth')
        
        # Trading Mode
        self.USE_TESTNET = os.getenv('USE_TESTNET', 'False').lower() == 'true'

    def get_enhanced_config():
        """Enhanced configuration with technical analysis improvements"""
        return {
            # Required (existing)
            'api_key': 'YOUR_API_KEY',
            'api_secret': 'YOUR_API_SECRET',
            'symbol': 'ETHUSDT',
            'model_path': 'models/your_model.pth',
            
            # Technical Analysis Configuration (enhanced)
            'use_technical_levels': True,
            'lookback_candles': 200,
            'sr_merge_threshold': 0.001,
            'max_risk_technical': 0.02,
            'min_rr_ratio': 1.5,
            'level_update_interval': 300,
            'fallback_to_fixed': True,
            
            # Resistance check configuration
            'check_resistance': True,
            'resistance_proximity_threshold': 0.003,
            'resistance_buffer_pct': 0.0005,
            'allow_breakout_trades': True,
            'breakout_confirmation_pct': 0.002,
            
            # Enhanced buffer configuration
            'min_sl_distance_pct': 0.003,  # Minimum 0.3% from S/R
            'min_sl_distance_atr': 0.5,    # Minimum 0.5x ATR from S/R
            
            # Zone proximity thresholds
            'support_zone_threshold': 0.02,     # 2% from support
            'resistance_zone_threshold': 0.02,  # 2% from resistance
            'fibonacci_zone_threshold': 0.015,  # 1.5% from Fibonacci
            
            # Trading parameters (existing)
            'leverage': 20,
            'position_pct_normal': 0.02,
            'position_pct_strong': 0.05,
            'stop_loss_pct': 0.002,
            'use_dynamic_sl': True,
            'use_take_profit': True,
            'tp1_percent': 0.005,
            'tp2_percent': 0.01,
            'tp1_size_ratio': 0.5,
        }

    def get_conservative_config(self) -> Dict:
        """Conservative trading configuration - Lower risk"""
        return {
            # API Configuration
            'api_key': self.API_KEY,
            'api_secret': self.API_SECRET,
            'model_path': self.MODEL_PATH,
            'testnet': self.USE_TESTNET,
            
            # Portfolio Management - CONSERVATIVE
            'max_positions': 5,                    # Max 3 positions only
            'position_pct_per_symbol': 0.08,       # 8% per symbol (conservative)
            'total_balance_pct': 0.6,              # Use only 60% of balance
            
            # Trading Configuration - CONSERVATIVE
            'leverage': 10,                        # Lower leverage
            'stop_loss_pct': 0.015,               # 1.5% stop loss (tighter)
            'use_take_profit': True,
            'tp1_percent': 0.002,                 # 0.2% TP1 (closer)
            'tp2_percent': 0.005,                 # 0.5% TP2 (closer)
            'tp1_size_ratio': 0.6,                # 60% at TP1 (more conservative)
            
            # Screening Configuration - CONSERVATIVE
            'enable_screening': True,
            'screening_interval': 1,           # 1 hour (less frequent)
            'min_volume_24h': 10000000,          # $100M minimum (major pairs only)
            'min_signal_strength': 0.8,           # High threshold (strong signals only)
            'max_symbols': 10,                     # Top 5 only
            
            # Risk Management - CONSERVATIVE
            'max_daily_trades': 15,               # Lower trade limit
            'trading_interval': 120,              # 2 minutes (less frequent)
            
            # Filters - CONSERVATIVE
            'exclude_stable': True,
            'min_price': 0.1,                     # Higher minimum price
            'max_price': 500,                     # Lower maximum price
            'blacklist': ['BTCUSDT', '1000PEPEUSDT', '1000SHIBUSDT'],  # Exclude high volatility

            'use_technical_levels': True,
            'lookback_candles': 200,
            'sr_merge_threshold': 0.001,
            'max_risk_technical': 0.02,
            'min_rr_ratio': 1.5,
            'level_update_interval': 300,
            'fallback_to_fixed': True,

            'check_resistance': True,                      # Enable resistance checking
            'resistance_proximity_threshold': 0.003,       # 0.3% - how close is "too close"
            'resistance_buffer_pct': 0.0005,              # 0.05% buffer below resistance
            'allow_breakout_trades': True,                # Allow trades above resistance
            'breakout_confirmation_pct': 0.002,           # Need 0.2% above resistance for confirmation
            'min_sl_distance_pct': 0.004,      # Conservative: 0.4% minimal distance
            'min_sl_distance_atr': 0.7,        # Conservative: 0.7x ATR
        }
    
    def get_balanced_config(self) -> Dict:
        """Balanced trading configuration - Medium risk/reward"""
        return {
            # API Configuration
            'api_key': self.API_KEY,
            'api_secret': self.API_SECRET,
            'model_path': self.MODEL_PATH,
            'testnet': self.USE_TESTNET,
            
            # Portfolio Management - BALANCED
            'max_positions': 5,                    # Standard 5 positions
            'position_pct_per_symbol': 0.12,       # 12% per symbol
            'total_balance_pct': 0.75,             # Use 75% of balance
            
            # Trading Configuration - BALANCED
            'leverage': 30,                        # Medium leverage
            'stop_loss_pct': 0.02,                # 2% stop loss
            'use_take_profit': True,
            'tp1_percent': 0.003,                 # 0.3% TP1
            'tp2_percent': 0.008,                 # 0.8% TP2
            'tp1_size_ratio': 0.4,                # 40% at TP1
            
            # Screening Configuration - BALANCED (LOWERED THRESHOLD)
            'enable_screening': True,
            'screening_interval': 1,           # 30 minutes
            'min_volume_24h': 50000000,           # $50M minimum
            'min_signal_strength': 0.7,           # LOWERED from 0.65 to 0.3
            'max_symbols': 8,                     # Top 8 symbols
            
            # Risk Management - BALANCED
            'max_daily_trades': 25,               # Medium trade limit
            'trading_interval': 60,               # 1 minute
            
            # Filters - BALANCED
            'exclude_stable': True,
            'min_price': 0.01,
            'max_price': 1000,
            'blacklist': ['1000PEPEUSDT'],  # Exclude high volatility

            'use_technical_levels': True,
            'lookback_candles': 200,
            'sr_merge_threshold': 0.001,
            'max_risk_technical': 0.02,
            'min_rr_ratio': 1.5,
            'level_update_interval': 300,
            'fallback_to_fixed': True,    # Exclude only meme coins

            'check_resistance': True,                      # Enable resistance checking
            'resistance_proximity_threshold': 0.003,       # 0.3% - how close is "too close"
            'resistance_buffer_pct': 0.0005,              # 0.05% buffer below resistance
            'allow_breakout_trades': True,                # Allow trades above resistance
            'breakout_confirmation_pct': 0.002,           # Need 0.2% above resistance for confirmation
            'min_sl_distance_pct': 0.004,      # Conservative: 0.4% minimal distance
            'min_sl_distance_atr': 0.7,        # Conservative: 0.7x ATR
        }
    
    def get_aggressive_config(self) -> Dict:
        """Aggressive trading configuration - Higher risk/reward"""
        return {
            # API Configuration
            'api_key': self.API_KEY,
            'api_secret': self.API_SECRET,
            'model_path': self.MODEL_PATH,
            'testnet': self.USE_TESTNET,
            
            # Portfolio Management - AGGRESSIVE
            'max_positions': 8,                    # More positions
            'position_pct_per_symbol': 0.01,       # 15% per symbol
            'total_balance_pct': 0.6,              # Use 60% of balance
            
            # Trading Configuration - AGGRESSIVE
            'leverage': 30,                        # Higher leverage
            'stop_loss_pct': 0.017,               # 2.5% stop loss (wider)
            'use_take_profit': True,
            'tp1_percent': 0.02,                 # 0.5% TP1 (further)
            'tp2_percent': 0.025,                 # 1.2% TP2 (further)
            'tp1_size_ratio': 0.6,                # 30% at TP1 (let profits run)
            
            
            # Batch Processing
            'use_batch_screening': True,      # ⚡ NEW: Enable batch processing
            'batch_size': 5,                 # ⚡ NEW: 20 symbols per batch
            'max_batches':5,                 # ⚡ NEW: Up to 3 batches (60 symbols)
            'batch_delay': 2,                 # ⚡ NEW: 2 seconds between batches
            
            # Screening Configuration - AGGRESSIVE
            'enable_screening': True,
            'screening_interval': 60,            # 15 minutes (more frequent)
            'min_volume_24h': 20000000,           # $20M minimum (include smaller caps)
            'min_signal_strength': 0.7,           # Lower threshold (more signals)
            'max_symbols': 12,                    # Top 12 symbols
            'trading_interval': 30,           # ⚡ 30 seconds (was 60 seconds)
            
            # Risk Management - AGGRESSIVE
            'max_daily_trades': 50,               # Higher trade limit
            'trading_interval': 45,               # 45 seconds (more frequent)
            
            # Filters - AGGRESSIVE
            'exclude_stable': True,
            'min_price': 0.001,                   # Include micro-cap
            'max_price': 50000,                    # Include expensive coins
            'blacklist': [],  
            
            # Exclude high volatility

            'use_technical_levels': True,
            'lookback_candles': 200,
            'sr_merge_threshold': 0.01,
            'max_risk_technical': 0.02,
            'min_rr_ratio': 1.5,
            'level_update_interval': 300,
            'fallback_to_fixed': True,

            'check_resistance': True,                      # Enable resistance checking
            'resistance_proximity_threshold': 0.003,       # 0.3% - how close is "too close"
            'resistance_buffer_pct': 0.0005,              # 0.05% buffer below resistance
            'allow_breakout_trades': True,                # Allow trades above resistance
            'breakout_confirmation_pct': 0.002,           # Need 0.2% above resistance for confirmation
            'min_sl_distance_pct': 0.004,      # Conservative: 0.4% minimal distance
            'min_sl_distance_atr': 0.7,        # Conservative: 0.7x ATR             
        }
    
    def get_scalping_config(self) -> Dict:
        """Scalping configuration - High frequency, small profits"""
        return {
            # API Configuration
            'api_key': self.API_KEY,
            'api_secret': self.API_SECRET,
            'model_path': self.MODEL_PATH,
            'testnet': self.USE_TESTNET,
            
            # Portfolio Management - SCALPING
            'max_positions': 10,                   # Many small positions
            'position_pct_per_symbol': 0.08,       # 8% per symbol (smaller size)
            'total_balance_pct': 0.8,              # Use 80% of balance
            
            # Trading Configuration - SCALPING
            'leverage': 30,                        # Very high leverage
            'stop_loss_pct': 0.002,               # 0.8% stop loss (very tight)
            'use_take_profit': True,
            'tp1_percent': 0.001,                 # 0.1% TP1 (very close)
            'tp2_percent': 0.0025,                # 0.25% TP2 (very close)
            'tp1_size_ratio': 0.8,                # 80% at TP1 (quick profits)
            
            # Screening Configuration - SCALPING
            'enable_screening': True,
            'screening_interval': 1,            # 10 minutes (frequent)
            'min_volume_24h': 80000000,           # $80M (high liquidity needed)
            'min_signal_strength': 0.7,           # Medium threshold
            'max_symbols': 15,                    # Many symbols
            
            # Risk Management - SCALPING
            'max_daily_trades': 100,              # Very high trade limit
            'trading_interval': 30,               # 30 seconds (very frequent)
            
            # Filters - SCALPING
            'exclude_stable': True,
            'min_price': 0.01,                    # Avoid micro-caps
            'max_price': 1000,
            'blacklist': ['BTCUSDT'],  # Exclude high volatility

            'use_technical_levels': True,
            'lookback_candles': 200,
            'sr_merge_threshold': 0.001,
            'max_risk_technical': 0.02,
            'min_rr_ratio': 1.5,
            'level_update_interval': 300,
            'fallback_to_fixed': True,
            
            'check_resistance': True,                      # Enable resistance checking
            'resistance_proximity_threshold': 0.003,       # 0.3% - how close is "too close"
            'resistance_buffer_pct': 0.0005,              # 0.05% buffer below resistance
            'allow_breakout_trades': True,                # Allow trades above resistance
            'breakout_confirmation_pct': 0.002,           # Need 0.2% above resistance for confirmation         # Exclude BTC (too slow for scalping)
            'min_sl_distance_pct': 0.004,      # Conservative: 0.4% minimal distance
            'min_sl_distance_atr': 0.7,        # Conservative: 0.7x ATR
        }
    
    def get_swing_config(self) -> Dict:
        """Swing trading configuration - Lower frequency, bigger moves"""
        return {
            # API Configuration
            'api_key': self.API_KEY,
            'api_secret': self.API_SECRET,
            'model_path': self.MODEL_PATH,
            'testnet': self.USE_TESTNET,
            
            # Portfolio Management - SWING
            'max_positions': 3,                    # Few concentrated positions
            'position_pct_per_symbol': 0.25,       # 25% per symbol (larger size)
            'total_balance_pct': 0.75,             # Use 75% of balance
            
            # Trading Configuration - SWING
            'leverage': 5,                         # Low leverage for safety
            'stop_loss_pct': 0.05,                # 5% stop loss (wide)
            'use_take_profit': True,
            'tp1_percent': 0.02,                  # 2% TP1 (far)
            'tp2_percent': 0.05,                  # 5% TP2 (very far)
            'tp1_size_ratio': 0.2,                # 20% at TP1 (let profits run)
            
            # Screening Configuration - SWING
            'enable_screening': True,
            'screening_interval': 1,           # 2 hours (infrequent)
            'min_volume_24h': 200000000,          # $200M (major pairs only)
            'min_signal_strength': 0.85,          # Very high threshold
            'max_symbols': 5,                     # Top 5 only
            
            # Risk Management - SWING
            'max_daily_trades': 5,                # Very low trade limit
            'trading_interval': 300,              # 5 minutes (infrequent)
            
            # Filters - SWING
            'exclude_stable': True,
            'min_price': 1.0,                     # Major coins only
            'max_price': 1000,
            'blacklist': ['1000PEPEUSDT', '1000SHIBUSDT', 'DOGEUSDT'],
              
            'check_resistance': True,                      # Enable resistance checking
            'resistance_proximity_threshold': 0.003,       # 0.3% - how close is "too close"
            'resistance_buffer_pct': 0.0005,              # 0.05% buffer below resistance
            'allow_breakout_trades': True,                # Allow trades above resistance
            'breakout_confirmation_pct': 0.002,           # Need 0.2% above resistance for confirmation  # No meme coins
            'min_sl_distance_pct': 0.004,      # Conservative: 0.4% minimal distance
            'min_sl_distance_atr': 0.7,        # Conservative: 0.7x ATR
        }
    
    def get_custom_config(self, 
                         risk_level: str = "balanced",
                         max_positions: int = None,
                         leverage: int = None,
                         **kwargs) -> Dict:
        """Get custom configuration with overrides"""
        
        # Start with base config
        if risk_level.lower() == "conservative":
            config = self.get_conservative_config()
        elif risk_level.lower() == "aggressive":
            config = self.get_aggressive_config()
        elif risk_level.lower() == "scalping":
            config = self.get_scalping_config()
        elif risk_level.lower() == "swing":
            config = self.get_swing_config()
        else:
            config = self.get_balanced_config()
        
        # Apply overrides
        if max_positions is not None:
            config['max_positions'] = max_positions
        if leverage is not None:
            config['leverage'] = leverage
        
        # Apply any additional kwargs
        config.update(kwargs)
        
        return config


# Convenience functions for quick access
def get_config(strategy: str = "balanced") -> Dict:
    """Get configuration for specific strategy"""
    config_manager = TradingConfig()
    
    strategy_map = {
        "conservative": config_manager.get_conservative_config,
        "balanced": config_manager.get_balanced_config,
        "aggressive": config_manager.get_aggressive_config,
        "scalping": config_manager.get_scalping_config,
        "swing": config_manager.get_swing_config
    }
    
    return strategy_map.get(strategy.lower(), config_manager.get_balanced_config)()


def print_config_comparison():
    """Print comparison of all configurations"""
    config_manager = TradingConfig()
    
    configs = {
        "Conservative": config_manager.get_conservative_config(),
        "Balanced": config_manager.get_balanced_config(),
        "Aggressive": config_manager.get_aggressive_config(),
        "Scalping": config_manager.get_scalping_config(),
        "Swing": config_manager.get_swing_config()
    }
    
    print("📊 TRADING STRATEGY COMPARISON")
    print("=" * 80)
    
    # Key metrics to compare
    metrics = [
        ('max_positions', 'Max Positions'),
        ('position_pct_per_symbol', 'Position Size %'),
        ('leverage', 'Leverage'),
        ('stop_loss_pct', 'Stop Loss %'),
        ('tp1_percent', 'Take Profit 1 %'),
        ('tp2_percent', 'Take Profit 2 %'),
        ('min_signal_strength', 'Min Signal'),
        ('max_daily_trades', 'Daily Trade Limit'),
        ('trading_interval', 'Trading Interval (s)')
    ]
    
    # Print header
    print(f"{'Metric':<20}", end="")
    for name in configs.keys():
        print(f"{name:>12}", end="")
    print()
    print("-" * 80)
    
    # Print each metric
    for key, label in metrics:
        print(f"{label:<20}", end="")
        for config in configs.values():
            value = config.get(key, 'N/A')
            if key.endswith('_pct'):
                value = f"{value*100:.1f}%" if isinstance(value, (int, float)) else str(value)
            elif key == 'position_pct_per_symbol':
                value = f"{value*100:.0f}%" if isinstance(value, (int, float)) else str(value)
            print(f"{str(value):>12}", end="")
        print()
    
    print("\n💡 STRATEGY RECOMMENDATIONS:")
    print("Conservative: New traders, small accounts, risk-averse")
    print("Balanced:     Most traders, medium accounts, moderate risk")
    print("Aggressive:   Experienced traders, larger accounts, high risk tolerance")
    print("Scalping:     Very experienced, high-speed trading, small profits")
    print("Swing:        Patient traders, trend following, larger moves")


# Usage examples
if __name__ == "__main__":
    # Print comparison
    print_config_comparison()
    
    print("\n\n🚀 USAGE EXAMPLES:")
    print("=" * 50)
    
    # Example 1: Get balanced config
    print("1. Balanced Strategy:")
    config = get_config("balanced")
    print(f"   Max Positions: {config['max_positions']}")
    print(f"   Leverage: {config['leverage']}x")
    print(f"   Position Size: {config['position_pct_per_symbol']*100:.0f}%")
    
    # Example 2: Custom config
    print("\n2. Custom Configuration:")
    config_manager = TradingConfig()
    custom_config = config_manager.get_custom_config(
        risk_level="aggressive",
        max_positions=3,           # Override: Fewer positions
        leverage=25,               # Override: Medium leverage
        min_signal_strength=0.75   # Override: Higher threshold
    )
    print(f"   Max Positions: {custom_config['max_positions']}")
    print(f"   Leverage: {custom_config['leverage']}x")
    print(f"   Min Signal: {custom_config['min_signal_strength']}")
    
    # Example 3: Environment variables
    print("\n3. Environment Variables:")
    print("   Set these to override defaults:")
    print("   export BINANCE_API_KEY='your_api_key'")
    print("   export BINANCE_API_SECRET='your_api_secret'")
    print("   export MODEL_PATH='path/to/your/model.pth'")
    print("   export USE_TESTNET='true'  # For testing")