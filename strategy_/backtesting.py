import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime
import os
import sys

# === PyTorch 2.6+ Compatibility Fix ===
import torch
import torch.serialization
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings

# Setup safe globals untuk PyTorch 2.6+ (handle numpy deprecation)
safe_globals_list = [StandardScaler, np.ndarray, np.dtype]

# Handle numpy.core deprecation gracefully
try:
    # Try new numpy._core first (recommended)
    safe_globals_list.append(np._core.multiarray.scalar)
except AttributeError:
    try:
        # Fallback to old numpy.core (with warning suppression)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            safe_globals_list.append(np.core.multiarray.scalar)
    except AttributeError:
        # If both fail, continue without it
        pass

torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
print("✅ PyTorch 2.6+ compatibility enabled (numpy deprecation handled)")

# Add path untuk import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from library.data_utils import DataLoader
from strategy_.features import create_lstm_features
from strategy_.model import TradingModel,ModelConfig


# ==========================================
# 🎛️ BACKTESTING CONFIGURATION CENTER
# ==========================================
class BacktestConfig:
    """Centralized backtesting configuration untuk easy adjustment"""
    
    # === TRADING SETUP ===
    INITIAL_BALANCE = 10000
    TRADING_FEE = 0.001   # Lower fees
    SPREAD = 0.0003        # Tighter spread
    LEVERAGE = 100              # 1:100 leverage
    STOP_LOSS_PCT = 0.002        # 2% default stop loss
    
    # === DATA SETTINGS ===
    SYMBOL = 'btc'
    DATA_LIMIT = 10000
    
    # === SIGNAL THRESHOLDS (SYNC DENGAN ModelConfig) ===
    STRONG_BUY = 0.9
    BUY = ModelConfig.BUY_THRESHOLD        # 0.4 (dari ModelConfig)
    NEUTRAL_HIGH = ModelConfig.NEUTRAL_HIGH # 0.4
    NEUTRAL_LOW = ModelConfig.NEUTRAL_LOW   # -0.4
    SELL = ModelConfig.SELL_THRESHOLD      # -0.4
    STRONG_SELL = -0.9
    
    # === POSITION SIZING ===
    POSITION_MULTIPLIER_STRONG = 0.05    # Full position for strong signals
    POSITION_MULTIPLIER_NORMAL = 0.02    # 60% position for normal signals
    POSITION_MULTIPLIER_NEUTRAL = 0.0   # No position for neutral
    
    # === STOP LOSS SETTINGS ===
    USE_DYNAMIC_SL = True       # Use dynamic stop loss based on volatility
    MIN_SL_PCT = 0.005          # Minimum 0.5% stop loss
    MAX_SL_PCT = 0.05           # Maximum 5% stop loss
    SL_LOOKBACK = 10            # Lookback period for volatility calculation
    SL_MULTIPLIER = 2           # Volatility multiplier for stop loss
    
    # === MODEL COMPATIBILITY ===
    FORCE_MODEL_COMPATIBILITY = True   # Force compatibility with different model architectures
    DEFAULT_SEQ_LEN = 120               # Default sequence length for fallback
    DEFAULT_HIDDEN_SIZE = 64            # Default hidden size for fallback
    
    @classmethod
    def print_config(cls):
        """Print current backtesting configuration"""
        print("\n🎛️ BACKTESTING CONFIGURATION:")
        print("=" * 50)
        print(f"Trading Setup:")
        print(f"  Initial Balance: ${cls.INITIAL_BALANCE:,.2f}")
        print(f"  Trading Fee: {cls.TRADING_FEE*100:.2f}%")
        print(f"  Spread: {cls.SPREAD*100:.3f}%")
        print(f"  Leverage: 1:{cls.LEVERAGE}")
        print(f"  Stop Loss: {cls.STOP_LOSS_PCT*100:.1f}%")
        print(f"\nData Settings:")
        print(f"  Symbol: {cls.SYMBOL}")
        print(f"  Data Limit: {cls.DATA_LIMIT}")
        print(f"\nSignal Thresholds:")
        print(f"  Strong Buy: >{cls.STRONG_BUY}")
        print(f"  Buy: {cls.BUY} to {cls.STRONG_BUY}")
        print(f"  Neutral: {cls.NEUTRAL_LOW} to {cls.NEUTRAL_HIGH}")
        print(f"  Sell: {cls.STRONG_SELL} to {cls.SELL}")
        print(f"  Strong Sell: <{cls.STRONG_SELL}")
        print(f"\nPosition Sizing:")
        print(f"  Strong: {cls.POSITION_MULTIPLIER_STRONG*100:.0f}%")
        print(f"  Normal: {cls.POSITION_MULTIPLIER_NORMAL*100:.0f}%")
        print(f"  Neutral: {cls.POSITION_MULTIPLIER_NEUTRAL*100:.0f}%")
        print(f"\nStop Loss Settings:")
        print(f"  Dynamic SL: {cls.USE_DYNAMIC_SL}")
        print(f"  Min SL: {cls.MIN_SL_PCT*100:.1f}%")
        print(f"  Max SL: {cls.MAX_SL_PCT*100:.1f}%")
        print(f"  Lookback: {cls.SL_LOOKBACK}")
        print(f"  Multiplier: {cls.SL_MULTIPLIER}x")
        print("=" * 50)


class SimpleBacktest:
    def __init__(self, initial_balance=None, trading_fee=None, spread=None, leverage=None, stop_loss_pct=None):
        # Use config defaults if not provided
        self.initial_balance = initial_balance or BacktestConfig.INITIAL_BALANCE
        self.balance = self.initial_balance
        self.position = 0  # 1 = long, -1 = short, 0 = no position
        self.trades = []
        self.equity_curve = []
        
        # Trading costs from config
        self.trading_fee = trading_fee or BacktestConfig.TRADING_FEE
        self.spread = spread or BacktestConfig.SPREAD
        self.total_fees_paid = 0  # Track total fees
        
        # Leverage settings from config
        self.leverage = leverage or BacktestConfig.LEVERAGE
        self.margin_requirement = 1.0 / self.leverage  # 1% margin for 1:100 leverage
        self.liquidation_threshold = 0.8  # 80% of margin (0.8% loss triggers liquidation)
        self.funding_rate = 0.0001  # 0.01% funding fee per position per candle
        self.total_funding_paid = 0
        
        # Position tracking
        self.position_size = 0  # Actual position size (balance * leverage)
        self.used_margin = 0    # Margin currently used
        self.free_margin = self.initial_balance  # Available margin
        
        # 5-Level Decision thresholds from config
        self.STRONG_BUY = BacktestConfig.STRONG_BUY
        self.BUY = BacktestConfig.BUY
        self.NEUTRAL_HIGH = BacktestConfig.NEUTRAL_HIGH
        self.NEUTRAL_LOW = BacktestConfig.NEUTRAL_LOW
        self.SELL = BacktestConfig.SELL
        self.STRONG_SELL = BacktestConfig.STRONG_SELL
        
        # Position sizing based on signal strength from config
        self.position_multipliers = {
            'strong': BacktestConfig.POSITION_MULTIPLIER_STRONG,
            'normal': BacktestConfig.POSITION_MULTIPLIER_NORMAL,
            'neutral': BacktestConfig.POSITION_MULTIPLIER_NEUTRAL
        }
        
        # Track position strength
        self.position_strength = 0
        
        # STOP LOSS SETTINGS from config
        self.stop_loss_pct = stop_loss_pct or BacktestConfig.STOP_LOSS_PCT
        self.use_dynamic_sl = BacktestConfig.USE_DYNAMIC_SL
        self.min_sl_pct = BacktestConfig.MIN_SL_PCT
        self.max_sl_pct = BacktestConfig.MAX_SL_PCT
        
        # Stop loss tracking
        self.stop_loss_level = 0
        self.total_stop_losses = 0
        self.stop_loss_trades = []
        
        # Print configuration
        BacktestConfig.print_config()
        
        print(f"\n💰 Applied Trading Setup:")
        print(f"   - Initial Balance: ${self.initial_balance:,.2f}")
        print(f"   - Leverage: 1:{self.leverage}")
        print(f"   - Margin Requirement: {self.margin_requirement*100:.2f}%") 
        print(f"   - Liquidation Threshold: {self.liquidation_threshold*100:.1f}% loss")
        print(f"   - Trading Fee: {self.trading_fee*100:.2f}%")
        print(f"   - Spread: {self.spread*100:.3f}%")
        print(f"   - Stop Loss: {self.stop_loss_pct*100:.1f}% (Dynamic: {self.use_dynamic_sl})")
        print(f"   - Decision Levels: Strong Buy={self.STRONG_BUY}, Buy={self.BUY}, Sell={self.SELL}, Strong Sell={self.STRONG_SELL}")
        
    def calculate_dynamic_stop_loss(self, df, current_idx, lookback=None):
        """
        Calculate dynamic stop loss based on recent volatility
        Similar to logic in label creation
        """
        if not self.use_dynamic_sl:
            return self.stop_loss_pct
        
        # Use config lookback if not provided
        lookback = lookback or BacktestConfig.SL_LOOKBACK
            
        # Get recent candles for volatility calculation
        start_idx = max(0, current_idx - lookback)
        
        percentages = []
        for i in range(start_idx + 1, current_idx + 1):
            prev_close = df['close'].iloc[i-1]
            curr_close = df['close'].iloc[i]
            
            if prev_close != 0:
                pct_change = abs((curr_close - prev_close) / prev_close)
                percentages.append(pct_change)
        
        # Calculate average volatility
        if percentages:
            avg_volatility = np.mean(percentages)
            # Use config multiplier for volatility as stop loss
            dynamic_sl = avg_volatility * BacktestConfig.SL_MULTIPLIER
            # Clamp between min and max from config
            dynamic_sl = max(self.min_sl_pct, min(dynamic_sl, self.max_sl_pct))
            return dynamic_sl
        else:
            return self.stop_loss_pct
    
    def load_latest_model(self):
        """Enhanced model loading with compatibility fix"""
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        # Get latest model file
        model_files = sorted(
            [f for f in os.listdir(models_dir) if f.endswith('.pth')],
            key=lambda f: os.path.getmtime(os.path.join(models_dir, f)),
            reverse=True
        )
        
        if not model_files:
            raise FileNotFoundError("No model found!")
        
        model_path = os.path.join(models_dir, model_files[0])
        print(f"📂 Loading: {model_files[0]}")
        
        # Load checkpoint with error handling
        try:
            checkpoint = torch.load(model_path, weights_only=False)
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}")
            if BacktestConfig.FORCE_MODEL_COMPATIBILITY:
                print("🔧 Attempting compatibility mode...")
                # Try with map_location
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            else:
                raise
        
        # Enhanced model creation with compatibility
        try:
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                print(f"📋 Found model config: {config}")
                
                # Try to create model with saved config
                model = TradingModel(
                    seq_len=config.get('seq_len', BacktestConfig.DEFAULT_SEQ_LEN),
                    hidden_size=config.get('hidden_size', BacktestConfig.DEFAULT_HIDDEN_SIZE)
                )
            else:
                print("⚠️ No model config found, using defaults")
                # Default fallback with config values
                model = TradingModel(
                    seq_len=BacktestConfig.DEFAULT_SEQ_LEN, 
                    hidden_size=BacktestConfig.DEFAULT_HIDDEN_SIZE
                )
            
            # Load using built-in method with error handling
            try:
                model.load_model(model_path)
                print("✅ Model loaded successfully!")
            except Exception as load_error:
                print(f"⚠️ Model loading error: {load_error}")
                
                if BacktestConfig.FORCE_MODEL_COMPATIBILITY:
                    print("🔧 Attempting compatibility loading...")
                    
                    # Try to load state dict manually with partial loading
                    try:
                        state_dict = checkpoint['model_state']
                        model_state = model.model.state_dict()
                        
                        # Filter out incompatible keys
                        filtered_state_dict = {}
                        skipped_keys = []
                        
                        for key, value in state_dict.items():
                            if key in model_state and model_state[key].shape == value.shape:
                                filtered_state_dict[key] = value
                            else:
                                skipped_keys.append(key)
                        
                        if skipped_keys:
                            print(f"⚠️ Skipped incompatible keys: {skipped_keys}")
                        
                        # Load filtered state dict
                        model.model.load_state_dict(filtered_state_dict, strict=False)
                        
                        # Load other components
                        if 'scaler' in checkpoint:
                            model.scaler = checkpoint['scaler']
                        if 'seq_len' in checkpoint:
                            model.seq_len = checkpoint['seq_len']
                            
                        print("✅ Model loaded with compatibility mode!")
                        
                    except Exception as compat_error:
                        print(f"❌ Compatibility loading failed: {compat_error}")
                        print("🔄 Creating new model with default architecture...")
                        
                        # Last resort: create completely new model
                        model = TradingModel(
                            seq_len=BacktestConfig.DEFAULT_SEQ_LEN,
                            hidden_size=BacktestConfig.DEFAULT_HIDDEN_SIZE
                        )
                        print("⚠️ Using fresh model - predictions may be random!")
                else:
                    raise load_error
            
        except Exception as e:
            print(f"❌ Model creation failed: {e}")
            if BacktestConfig.FORCE_MODEL_COMPATIBILITY:
                print("🔄 Creating fallback model...")
                model = TradingModel(
                    seq_len=BacktestConfig.DEFAULT_SEQ_LEN,
                    hidden_size=BacktestConfig.DEFAULT_HIDDEN_SIZE
                )
                print("⚠️ Using fallback model - predictions may be random!")
            else:
                raise
        
        return model
    
    def prepare_data(self, symbol=None, limit=None):
        """Load dan prepare data dengan detailed tracking using config"""
        symbol = symbol or BacktestConfig.SYMBOL
        limit = limit or BacktestConfig.DATA_LIMIT
        
        print(f"\n📊 Loading data for {symbol}...")
        print(f"🎯 Target limit: {limit}")
        
        try:
            # Load data
            loader = DataLoader()
            multi_data = loader.load_data(
                symbol=symbol,
                timeframes=['1m', '5m', '30m', '1h'],
                limit=limit,
                auto_align=True,
                alignment_mode='current_only'
            )
            
            # Check data setelah loading
            for tf, data in multi_data.items():
                print(f"   📈 {tf}: {len(data)} candles")
            
            raw_data_original = multi_data['1m'].copy()
            print(f"🔍 Raw 1m data: {len(raw_data_original)} candles")
            
            # Create features
            print("🔧 Creating features...")
            features_df = create_lstm_features(multi_data)
            print(f"🔍 After feature creation: {len(features_df)} samples")
            
            # Get 1m data for candlestick (align dengan features)
            raw_data = multi_data['1m'].copy()
            
            # Align lengths untuk debugging
            min_len = min(len(features_df), len(raw_data))
            if len(features_df) != len(raw_data):
                print(f"⚠️  Length mismatch - Features: {len(features_df)}, Raw: {len(raw_data)}")
                print(f"🔧 Using aligned length: {min_len}")
                
                # Align data
                features_df = features_df.iloc[:min_len]
                raw_data = raw_data.iloc[:min_len]
            
            print(f"✅ Final data prepared - Features: {len(features_df)}, Raw: {len(raw_data)}")
            
            # Data reduction summary
            reduction_pct = ((limit - len(features_df)) / limit) * 100
            print(f"📉 Data reduction: {reduction_pct:.1f}% ({limit} → {len(features_df)})")
            
            return features_df, raw_data
            
        except Exception as e:
            print(f"❌ Error preparing data: {e}")
            raise
    
    def run_backtest(self, predictions, raw_data, model=None):
        """UPDATED: Handle predictions without timestamp index + FIXED LEVERAGE + FIXED WIN RATE BUG"""
        print("\n🚀 Running backtest with CORRECTED leverage calculations and WIN RATE fix...")
        
        # Debug data lengths
        print(f"🔍 Input data - Predictions: {len(predictions)}, Raw: {len(raw_data)}")
        
        # FIXED: Get sequence length from model or use default
        if model is not None:
            seq_len = model.seq_len
        else:
            seq_len = BacktestConfig.DEFAULT_SEQ_LEN
            print(f"⚠️ Using default seq_len: {seq_len}")
        
        print(f"🔧 Using sequence length: {seq_len}")
        
        # Initialize raw_data columns
        raw_data = raw_data.copy()  # Avoid modifying original
        raw_data['signal'] = 0.0
        raw_data['position'] = 0
        raw_data['exit_reason'] = ''
        raw_data['signal_strength'] = ''
        raw_data['stop_loss'] = 0.0
        
        # SAFE SIGNAL ALIGNMENT
        # Calculate how many predictions we can actually use
        max_signals = min(len(predictions), len(raw_data) - seq_len)
        
        print(f"🔧 Can align {max_signals} signals with {len(raw_data)} raw data points")
        
        aligned_signals = 0
        for i in range(max_signals):
            target_idx = i + seq_len
            if target_idx < len(raw_data):
                raw_data.iloc[target_idx, raw_data.columns.get_loc('signal')] = predictions['decision'].iloc[i]
                aligned_signals += 1
        
        print(f"✅ Successfully aligned {aligned_signals} signals")
        
        # Validation check
        non_zero_signals = (raw_data['signal'] != 0).sum()
        print(f"🔍 Non-zero signals in raw_data: {non_zero_signals}")
        
        if non_zero_signals == 0:
            print("❌ WARNING: No signals aligned! Returning empty backtest.")
            return raw_data
        
        # Signal distribution analysis
        actual_signals = raw_data['signal'].values
        non_zero_mask = actual_signals != 0
        
        if non_zero_mask.sum() > 0:
            non_zero_signals_values = actual_signals[non_zero_mask]
            
            print(f"\n🔍 Signal Distribution Analysis:")
            strong_buy = (non_zero_signals_values > self.STRONG_BUY).sum()
            buy = ((non_zero_signals_values > self.BUY) & (non_zero_signals_values <= self.STRONG_BUY)).sum()
            neutral = ((non_zero_signals_values >= self.NEUTRAL_LOW) & (non_zero_signals_values <= self.NEUTRAL_HIGH)).sum()
            sell = ((non_zero_signals_values >= self.STRONG_SELL) & (non_zero_signals_values < self.SELL)).sum()
            strong_sell = (non_zero_signals_values < self.STRONG_SELL).sum()
            
            total_signals = len(non_zero_signals_values)
            print(f"   Strong Buy (>{self.STRONG_BUY}): {strong_buy} ({strong_buy/total_signals*100:.1f}%)")
            print(f"   Buy ({self.BUY} to {self.STRONG_BUY}): {buy} ({buy/total_signals*100:.1f}%)")
            print(f"   Neutral ({self.NEUTRAL_LOW} to {self.NEUTRAL_HIGH}): {neutral} ({neutral/total_signals*100:.1f}%)")
            print(f"   Sell ({self.STRONG_SELL} to {self.SELL}): {sell} ({sell/total_signals*100:.1f}%)")
            print(f"   Strong Sell (<{self.STRONG_SELL}): {strong_sell} ({strong_sell/total_signals*100:.1f}%)")
            
            print(f"   Signal range: [{non_zero_signals_values.min():.3f}, {non_zero_signals_values.max():.3f}]")
        
        # Trading variables
        entry_price = 0
        entry_idx = 0
        
        # FIXED: Proper position tracking variables
        self.position_value = 0      # Actual position value in USD
        self.position_size_btc = 0   # Position size in BTC
        self.margin_used = 0         # Margin currently in use
        
        # Add liquidation tracking
        self.liquidations = []
        self.total_liquidations = 0
        
        print("\n💹 Executing trades...")
        
        # Start from seq_len to ensure valid signals
        start_idx = max(1, seq_len + 1)
        trades_attempted = 0
        
        for i in range(start_idx, len(raw_data)):
            current_price = raw_data['close'].iloc[i]
            signal = raw_data['signal'].iloc[i]
            
            # Skip if no signal
            if signal == 0:
                continue
                
            trades_attempted += 1
            
            # Update equity curve
            self.equity_curve.append(self.balance)
            
            # === STOP LOSS & LIQUIDATION CHECK FIRST ===
            if self.position != 0:
                if self.position == 1:  # LONG position
                    current_loss_pct = (entry_price - current_price) / entry_price
                    
                    # FIXED: Proper liquidation check
                    position_loss_dollars = self.position_value * current_loss_pct
                    
                    # Liquidation when loss exceeds 80% of margin
                    if position_loss_dollars >= (self.margin_used * 0.8):
                        # LIQUIDATION!
                        print(f"💀 LIQUIDATION! Loss: ${position_loss_dollars:.2f} exceeds 80% of margin ${self.margin_used:.2f}")
                        
                        # Lose the entire margin
                        self.balance -= self.margin_used
                        
                        self.trades.append({
                            'type': 'LONG',
                            'entry': entry_price,
                            'exit': current_price,
                            'profit_pct': -100,  # Lost entire margin
                            'profit_dollars': -self.margin_used,
                            'duration': i - entry_idx,
                            'exit_reason': 'LIQUIDATION',
                            'position_size': self.position_strength,
                            'position_value': self.position_value,
                            'leverage': self.leverage,
                            'fees_paid': 0
                        })
                        
                        self.liquidations.append(self.trades[-1])
                        self.total_liquidations += 1
                        
                        # Reset position
                        self.position = 0
                        self.position_strength = 0
                        self.position_value = 0
                        self.position_size_btc = 0
                        self.margin_used = 0
                        self.free_margin = self.balance
                        entry_price = 0
                        self.stop_loss_level = 0
                        
                        continue
                    
                    # Regular stop loss check
                    if current_loss_pct >= self.stop_loss_level:
                        # STOP LOSS TRIGGERED
                        exit_price = current_price * (1 - self.spread)
                        
                        # FIXED: Proper PnL calculation
                        price_change_pct = (exit_price - entry_price) / entry_price
                        position_pnl_dollars = self.position_value * price_change_pct
                        
                        # Fee calculation
                        exit_value = self.position_size_btc * exit_price
                        exit_fee = exit_value * self.trading_fee
                        
                        # Funding fee (per 8 hours)
                        hours_held = (i - entry_idx) / 60  # Assuming 1m candles
                        funding_periods = int(hours_held / 8)
                        funding_fee = self.position_value * self.funding_rate * funding_periods
                        self.total_funding_paid += funding_fee
                        
                        # Update balance
                        net_pnl = position_pnl_dollars - exit_fee - funding_fee
                        self.balance += net_pnl
                        self.total_fees_paid += exit_fee
                        self.total_stop_losses += 1
                        
                        # FIXED: Calculate return BEFORE releasing margin
                        margin_used_temp = self.margin_used  # Store for calculation
                        return_on_margin = (net_pnl / margin_used_temp) * 100 if margin_used_temp > 0 else 0
                        
                        trade_info = {
                            'type': 'LONG',
                            'entry': entry_price,
                            'exit': exit_price,
                            'profit_pct': return_on_margin,
                            'profit_dollars': net_pnl,
                            'duration': i - entry_idx,
                            'exit_reason': 'STOP_LOSS',
                            'position_size': self.position_strength,
                            'position_value': self.position_value,
                            'leverage': self.leverage,
                            'fees_paid': exit_fee + funding_fee,
                            'stop_loss_pct': self.stop_loss_level * 100
                        }
                        
                        self.trades.append(trade_info)
                        self.stop_loss_trades.append(trade_info)
                        
                        print(f"🛑 STOP LOSS - Close LONG at ${exit_price:.2f}")
                        print(f"   💰 PnL: ${net_pnl:+.2f} ({return_on_margin:+.2f}% on margin)")
                        
                        # Release margin AFTER calculation
                        self.free_margin = self.balance
                        self.margin_used = 0
                        
                        # Reset position
                        self.position = 0
                        self.position_strength = 0
                        self.position_value = 0
                        self.position_size_btc = 0
                        entry_price = 0
                        self.stop_loss_level = 0
                        
                        continue
                        
                elif self.position == -1:  # SHORT position
                    current_loss_pct = (current_price - entry_price) / entry_price
                    
                    # FIXED: Proper liquidation check for SHORT
                    position_loss_dollars = self.position_value * current_loss_pct
                    
                    # Liquidation when loss exceeds 80% of margin
                    if position_loss_dollars >= (self.margin_used * 0.8):
                        # LIQUIDATION!
                        print(f"💀 LIQUIDATION! Loss: ${position_loss_dollars:.2f} exceeds 80% of margin ${self.margin_used:.2f}")
                        
                        # Lose the entire margin
                        self.balance -= self.margin_used
                        
                        self.trades.append({
                            'type': 'SHORT',
                            'entry': entry_price,
                            'exit': current_price,
                            'profit_pct': -100,  # Lost entire margin
                            'profit_dollars': -self.margin_used,
                            'duration': i - entry_idx,
                            'exit_reason': 'LIQUIDATION',
                            'position_size': self.position_strength,
                            'position_value': self.position_value,
                            'leverage': self.leverage,
                            'fees_paid': 0
                        })
                        
                        self.liquidations.append(self.trades[-1])
                        self.total_liquidations += 1
                        
                        # Reset position
                        self.position = 0
                        self.position_strength = 0
                        self.position_value = 0
                        self.position_size_btc = 0
                        self.margin_used = 0
                        self.free_margin = self.balance
                        entry_price = 0
                        self.stop_loss_level = 0
                        
                        continue
                    
                    # Regular stop loss check
                    if current_loss_pct >= self.stop_loss_level:
                        # STOP LOSS TRIGGERED
                        exit_price = current_price * (1 + self.spread)
                        
                        # FIXED: Proper PnL calculation for SHORT
                        price_change_pct = (entry_price - exit_price) / entry_price
                        position_pnl_dollars = self.position_value * price_change_pct
                        
                        # Fee calculation
                        exit_value = self.position_size_btc * exit_price
                        exit_fee = exit_value * self.trading_fee
                        
                        # Funding fee (per 8 hours)
                        hours_held = (i - entry_idx) / 60  # Assuming 1m candles
                        funding_periods = int(hours_held / 8)
                        funding_fee = self.position_value * self.funding_rate * funding_periods
                        self.total_funding_paid += funding_fee
                        
                        # Update balance
                        net_pnl = position_pnl_dollars - exit_fee - funding_fee
                        self.balance += net_pnl
                        self.total_fees_paid += exit_fee
                        self.total_stop_losses += 1
                        
                        # FIXED: Calculate return BEFORE releasing margin
                        margin_used_temp = self.margin_used
                        return_on_margin = (net_pnl / margin_used_temp) * 100 if margin_used_temp > 0 else 0
                        
                        trade_info = {
                            'type': 'SHORT',
                            'entry': entry_price,
                            'exit': exit_price,
                            'profit_pct': return_on_margin,
                            'profit_dollars': net_pnl,
                            'duration': i - entry_idx,
                            'exit_reason': 'STOP_LOSS',
                            'position_size': self.position_strength,
                            'position_value': self.position_value,
                            'leverage': self.leverage,
                            'fees_paid': exit_fee + funding_fee,
                            'stop_loss_pct': self.stop_loss_level * 100
                        }
                        
                        self.trades.append(trade_info)
                        self.stop_loss_trades.append(trade_info)
                        
                        print(f"🛑 STOP LOSS - Close SHORT at ${exit_price:.2f}")
                        print(f"   💰 PnL: ${net_pnl:+.2f} ({return_on_margin:+.2f}% on margin)")
                        
                        # Release margin AFTER calculation
                        self.free_margin = self.balance
                        self.margin_used = 0
                        
                        # Reset position
                        self.position = 0
                        self.position_strength = 0
                        self.position_value = 0
                        self.position_size_btc = 0
                        entry_price = 0
                        self.stop_loss_level = 0
                        
                        continue
            
            # === SIGNAL-BASED TRADING ===
            if self.position == 0:
                if signal > self.STRONG_BUY:  # Strong Buy
                    position_size = self.position_multipliers['strong']
                    entry_cost = current_price * (1 + self.spread)
                    
                    # FIXED: Proper leverage calculations
                    margin_used = self.balance * position_size
                    position_value = margin_used * self.leverage
                    position_size_btc = position_value / entry_cost
                    
                    # Check margin
                    if margin_used > self.free_margin:
                        continue
                    
                    # Fee from position VALUE
                    fee_cost = position_value * self.trading_fee
                    
                    self.stop_loss_level = self.calculate_dynamic_stop_loss(raw_data, i)
                    
                    self.position = 1
                    self.position_strength = position_size
                    self.margin_used = margin_used
                    self.position_value = position_value
                    self.position_size_btc = position_size_btc
                    self.free_margin = self.balance - margin_used
                    entry_price = entry_cost
                    entry_idx = i
                    self.total_fees_paid += fee_cost
                    
                    print(f"🔵 STRONG BUY at ${entry_cost:.2f} (Signal: {signal:.3f})")
                    print(f"   💰 Position Value: ${position_value:,.2f} (Margin: ${margin_used:.2f})")
                    print(f"   📊 Position Size: {position_size_btc:.6f} BTC")
                    
                elif signal > self.BUY:  # Normal Buy
                    position_size = self.position_multipliers['normal']
                    entry_cost = current_price * (1 + self.spread)
                    
                    # FIXED: Proper leverage calculations
                    margin_used = self.balance * position_size
                    position_value = margin_used * self.leverage
                    position_size_btc = position_value / entry_cost
                    
                    # Check margin
                    if margin_used > self.free_margin:
                        continue
                    
                    # Fee from position VALUE
                    fee_cost = position_value * self.trading_fee
                    
                    self.stop_loss_level = self.calculate_dynamic_stop_loss(raw_data, i)
                    
                    self.position = 1
                    self.position_strength = position_size
                    self.margin_used = margin_used
                    self.position_value = position_value
                    self.position_size_btc = position_size_btc
                    self.free_margin = self.balance - margin_used
                    entry_price = entry_cost
                    entry_idx = i
                    self.total_fees_paid += fee_cost
                    
                    print(f"🔵 BUY at ${entry_cost:.2f} (Signal: {signal:.3f})")
                    print(f"   💰 Position Value: ${position_value:,.2f} (Margin: ${margin_used:.2f})")
                    print(f"   📊 Position Size: {position_size_btc:.6f} BTC")
                    
                elif signal < self.STRONG_SELL:  # Strong Sell
                    position_size = self.position_multipliers['strong']
                    entry_cost = current_price * (1 - self.spread)
                    
                    # FIXED: Proper leverage calculations
                    margin_used = self.balance * position_size
                    position_value = margin_used * self.leverage
                    position_size_btc = position_value / entry_cost
                    
                    # Check margin
                    if margin_used > self.free_margin:
                        continue
                    
                    # Fee from position VALUE
                    fee_cost = position_value * self.trading_fee
                    
                    self.stop_loss_level = self.calculate_dynamic_stop_loss(raw_data, i)
                    
                    self.position = -1
                    self.position_strength = position_size
                    self.margin_used = margin_used
                    self.position_value = position_value
                    self.position_size_btc = position_size_btc
                    self.free_margin = self.balance - margin_used
                    entry_price = entry_cost
                    entry_idx = i
                    self.total_fees_paid += fee_cost
                    
                    print(f"🔴 STRONG SELL at ${entry_cost:.2f} (Signal: {signal:.3f})")
                    print(f"   💰 Position Value: ${position_value:,.2f} (Margin: ${margin_used:.2f})")
                    print(f"   📊 Position Size: {position_size_btc:.6f} BTC")
                    
                elif signal < self.SELL:  # Normal Sell
                    position_size = self.position_multipliers['normal']
                    entry_cost = current_price * (1 - self.spread)
                    
                    # FIXED: Proper leverage calculations
                    margin_used = self.balance * position_size
                    position_value = margin_used * self.leverage
                    position_size_btc = position_value / entry_cost
                    
                    # Check margin
                    if margin_used > self.free_margin:
                        continue
                    
                    # Fee from position VALUE
                    fee_cost = position_value * self.trading_fee
                    
                    self.stop_loss_level = self.calculate_dynamic_stop_loss(raw_data, i)
                    
                    self.position = -1
                    self.position_strength = position_size
                    self.margin_used = margin_used
                    self.position_value = position_value
                    self.position_size_btc = position_size_btc
                    self.free_margin = self.balance - margin_used
                    entry_price = entry_cost
                    entry_idx = i
                    self.total_fees_paid += fee_cost
                    
                    print(f"🔴 SELL at ${entry_cost:.2f} (Signal: {signal:.3f})")
                    print(f"   💰 Position Value: ${position_value:,.2f} (Margin: ${margin_used:.2f})")
                    print(f"   📊 Position Size: {position_size_btc:.6f} BTC")
            
            # === EXIT LOGIC ===
            elif self.position == 1:  # In LONG position
                if signal < self.STRONG_SELL:  # Strong reversal
                    # Exit long and enter short
                    exit_price = current_price * (1 - self.spread)
                    
                    # FIXED: Proper PnL calculation
                    price_change_pct = (exit_price - entry_price) / entry_price
                    position_pnl_dollars = self.position_value * price_change_pct
                    
                    # Fee calculation
                    exit_value = self.position_size_btc * exit_price
                    exit_fee = exit_value * self.trading_fee
                    
                    # Funding fee (per 8 hours)
                    hours_held = (i - entry_idx) / 60  # Assuming 1m candles
                    funding_periods = int(hours_held / 8)
                    funding_fee = self.position_value * self.funding_rate * funding_periods
                    self.total_funding_paid += funding_fee
                    
                    # Update balance
                    net_pnl = position_pnl_dollars - exit_fee - funding_fee
                    self.balance += net_pnl
                    self.total_fees_paid += exit_fee
                    
                    # FIXED: Calculate return BEFORE releasing margin
                    margin_used_temp = self.margin_used
                    return_on_margin = (net_pnl / margin_used_temp) * 100 if margin_used_temp > 0 else 0
                    
                    self.trades.append({
                        'type': 'LONG',
                        'entry': entry_price,
                        'exit': exit_price,
                        'profit_pct': return_on_margin,
                        'profit_dollars': net_pnl,
                        'duration': i - entry_idx,
                        'exit_reason': 'STRONG_REVERSAL',
                        'position_size': self.position_strength,
                        'position_value': self.position_value,
                        'leverage': self.leverage,
                        'fees_paid': exit_fee + funding_fee
                    })
                    
                    print(f"✅ Close LONG at ${exit_price:.2f}")
                    print(f"   📈 Price Change: {price_change_pct*100:+.2f}%")
                    print(f"   💰 PnL: ${net_pnl:+.2f} ({return_on_margin:+.2f}% on margin)")
                    
                    # Release margin
                    self.free_margin = self.balance
                    self.margin_used = 0
                    
                    # Enter short immediately
                    position_size = self.position_multipliers['strong']
                    entry_cost = current_price * (1 - self.spread)
                    
                    # Leverage calculations for new position
                    margin_used = self.balance * position_size
                    position_value = margin_used * self.leverage
                    position_size_btc = position_value / entry_cost
                    
                    if margin_used <= self.free_margin:
                        entry_fee = position_value * self.trading_fee
                        
                        self.stop_loss_level = self.calculate_dynamic_stop_loss(raw_data, i)
                        
                        self.position = -1
                        self.position_strength = position_size
                        self.margin_used = margin_used
                        self.position_value = position_value
                        self.position_size_btc = position_size_btc
                        self.free_margin = self.balance - margin_used
                        entry_price = entry_cost
                        entry_idx = i
                        self.total_fees_paid += entry_fee
                        
                        print(f"🔴 STRONG SELL at ${entry_cost:.2f} (Signal: {signal:.3f})")
                        print(f"   💰 Position Value: ${position_value:,.2f} (Margin: ${margin_used:.2f})")
                    
                elif signal < self.SELL:  # Normal sell - exit only
                    exit_price = current_price * (1 - self.spread)
                    
                    # FIXED: Proper PnL calculation
                    price_change_pct = (exit_price - entry_price) / entry_price
                    position_pnl_dollars = self.position_value * price_change_pct
                    
                    # Fee calculation
                    exit_value = self.position_size_btc * exit_price
                    exit_fee = exit_value * self.trading_fee
                    
                    # Funding fee (per 8 hours)
                    hours_held = (i - entry_idx) / 60  # Assuming 1m candles
                    funding_periods = int(hours_held / 8)
                    funding_fee = self.position_value * self.funding_rate * funding_periods
                    self.total_funding_paid += funding_fee
                    
                    # Update balance
                    net_pnl = position_pnl_dollars - exit_fee - funding_fee
                    self.balance += net_pnl
                    self.total_fees_paid += exit_fee
                    
                    # FIXED: Calculate return BEFORE releasing margin
                    margin_used_temp = self.margin_used
                    return_on_margin = (net_pnl / margin_used_temp) * 100 if margin_used_temp > 0 else 0
                    
                    self.trades.append({
                        'type': 'LONG',
                        'entry': entry_price,
                        'exit': exit_price,
                        'profit_pct': return_on_margin,
                        'profit_dollars': net_pnl,
                        'duration': i - entry_idx,
                        'exit_reason': 'SELL_SIGNAL',
                        'position_size': self.position_strength,
                        'position_value': self.position_value,
                        'leverage': self.leverage,
                        'fees_paid': exit_fee + funding_fee
                    })
                    
                    print(f"✅ Close LONG at ${exit_price:.2f}")
                    print(f"   📈 Price Change: {price_change_pct*100:+.2f}%")
                    print(f"   💰 PnL: ${net_pnl:+.2f} ({return_on_margin:+.2f}% on margin)")
                    
                    # Release margin
                    self.free_margin = self.balance
                    self.margin_used = 0
                    
                    # Reset position
                    self.position = 0
                    self.position_strength = 0
                    self.position_value = 0
                    self.position_size_btc = 0
                    entry_price = 0
                    self.stop_loss_level = 0
            
            elif self.position == -1:  # In SHORT position
                if signal > self.STRONG_BUY:  # Strong reversal
                    # Exit short and enter long
                    exit_price = current_price * (1 + self.spread)
                    
                    # FIXED: Proper PnL calculation for SHORT
                    price_change_pct = (entry_price - exit_price) / entry_price
                    position_pnl_dollars = self.position_value * price_change_pct
                    
                    # Fee calculation
                    exit_value = self.position_size_btc * exit_price
                    exit_fee = exit_value * self.trading_fee
                    
                    # Funding fee (per 8 hours)
                    hours_held = (i - entry_idx) / 60  # Assuming 1m candles
                    funding_periods = int(hours_held / 8)
                    funding_fee = self.position_value * self.funding_rate * funding_periods
                    self.total_funding_paid += funding_fee
                    
                    # Update balance
                    net_pnl = position_pnl_dollars - exit_fee - funding_fee
                    self.balance += net_pnl
                    self.total_fees_paid += exit_fee
                    
                    # FIXED: Calculate return BEFORE releasing margin
                    margin_used_temp = self.margin_used
                    return_on_margin = (net_pnl / margin_used_temp) * 100 if margin_used_temp > 0 else 0
                    
                    self.trades.append({
                        'type': 'SHORT',
                        'entry': entry_price,
                        'exit': exit_price,
                        'profit_pct': return_on_margin,
                        'profit_dollars': net_pnl,
                        'duration': i - entry_idx,
                        'exit_reason': 'STRONG_REVERSAL',
                        'position_size': self.position_strength,
                        'position_value': self.position_value,
                        'leverage': self.leverage,
                        'fees_paid': exit_fee + funding_fee
                    })
                    
                    print(f"✅ Close SHORT at ${exit_price:.2f}")
                    print(f"   📈 Price Change: {price_change_pct*100:+.2f}%")
                    print(f"   💰 PnL: ${net_pnl:+.2f} ({return_on_margin:+.2f}% on margin)")
                    
                    # Release margin
                    self.free_margin = self.balance
                    self.margin_used = 0
                    
                    # Enter long immediately
                    position_size = self.position_multipliers['strong']
                    entry_cost = current_price * (1 + self.spread)
                    
                    # Leverage calculations for new position
                    margin_used = self.balance * position_size
                    position_value = margin_used * self.leverage
                    position_size_btc = position_value / entry_cost
                    
                    if margin_used <= self.free_margin:
                        entry_fee = position_value * self.trading_fee
                        
                        self.stop_loss_level = self.calculate_dynamic_stop_loss(raw_data, i)
                        
                        self.position = 1
                        self.position_strength = position_size
                        self.margin_used = margin_used
                        self.position_value = position_value
                        self.position_size_btc = position_size_btc
                        self.free_margin = self.balance - margin_used
                        entry_price = entry_cost
                        entry_idx = i
                        self.total_fees_paid += entry_fee
                        
                        print(f"🔵 STRONG BUY at ${entry_cost:.2f} (Signal: {signal:.3f})")
                        print(f"   💰 Position Value: ${position_value:,.2f} (Margin: ${margin_used:.2f})")
                    
                elif signal > self.BUY:  # Normal buy - exit only
                    exit_price = current_price * (1 + self.spread)
                    
                    # FIXED: Proper PnL calculation for SHORT
                    price_change_pct = (entry_price - exit_price) / entry_price
                    position_pnl_dollars = self.position_value * price_change_pct
                    
                    # Fee calculation
                    exit_value = self.position_size_btc * exit_price
                    exit_fee = exit_value * self.trading_fee
                    
                    # Funding fee (per 8 hours)
                    hours_held = (i - entry_idx) / 60  # Assuming 1m candles
                    funding_periods = int(hours_held / 8)
                    funding_fee = self.position_value * self.funding_rate * funding_periods
                    self.total_funding_paid += funding_fee
                    
                    # Update balance
                    net_pnl = position_pnl_dollars - exit_fee - funding_fee
                    self.balance += net_pnl
                    self.total_fees_paid += exit_fee
                    
                    # FIXED: Calculate return BEFORE releasing margin
                    margin_used_temp = self.margin_used
                    return_on_margin = (net_pnl / margin_used_temp) * 100 if margin_used_temp > 0 else 0
                    
                    self.trades.append({
                        'type': 'SHORT',
                        'entry': entry_price,
                        'exit': exit_price,
                        'profit_pct': return_on_margin,
                        'profit_dollars': net_pnl,
                        'duration': i - entry_idx,
                        'exit_reason': 'BUY_SIGNAL',
                        'position_size': self.position_strength,
                        'position_value': self.position_value,
                        'leverage': self.leverage,
                        'fees_paid': exit_fee + funding_fee
                    })
                    
                    print(f"✅ Close SHORT at ${exit_price:.2f}")
                    print(f"   📈 Price Change: {price_change_pct*100:+.2f}%")
                    print(f"   💰 PnL: ${net_pnl:+.2f} ({return_on_margin:+.2f}% on margin)")
                    
                    # Release margin
                    self.free_margin = self.balance
                    self.margin_used = 0
                    
                    # Reset position
                    self.position = 0
                    self.position_strength = 0
                    self.position_value = 0
                    self.position_size_btc = 0
                    entry_price = 0
                    self.stop_loss_level = 0
        
        print(f"\n🏁 Backtest completed:")
        print(f"   📊 Data points: {len(raw_data)}")
        print(f"   🎯 Trades attempted: {trades_attempted}")
        print(f"   ✅ Trades executed: {len(self.trades)}")
        print(f"   🛑 Stop losses: {self.total_stop_losses}")
        print(f"   💀 Liquidations: {self.total_liquidations}")
        
        return raw_data
    
    def plot_results(self, data_with_signals):
        """Plot candlestick with 5-level signals and equity curve"""
        print("\n📊 Creating visualizations...")
        
        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Plot 1: Price with signals
            ax1.plot(data_with_signals.index, data_with_signals['close'], 'k-', linewidth=1, label='Price')
            
            # Separate signals by strength
            strong_buy_signals = data_with_signals[data_with_signals['signal'] > self.STRONG_BUY]
            buy_signals = data_with_signals[(data_with_signals['signal'] > self.BUY) & 
                                          (data_with_signals['signal'] <= self.STRONG_BUY)]
            sell_signals = data_with_signals[(data_with_signals['signal'] >= self.STRONG_SELL) & 
                                            (data_with_signals['signal'] < self.SELL)]
            strong_sell_signals = data_with_signals[data_with_signals['signal'] < self.STRONG_SELL]
            
            # Mark stop loss exits
            stop_loss_exits = data_with_signals[data_with_signals['exit_reason'] == 'STOP_LOSS']
            
            # Plot with different markers/colors
            if len(strong_buy_signals) > 0:
                ax1.scatter(strong_buy_signals.index, strong_buy_signals['close'], 
                           color='darkgreen', marker='^', s=100, label='Strong Buy', alpha=0.9)
            
            if len(buy_signals) > 0:
                ax1.scatter(buy_signals.index, buy_signals['close'], 
                           color='lightgreen', marker='^', s=60, label='Buy', alpha=0.7)
            
            if len(sell_signals) > 0:
                ax1.scatter(sell_signals.index, sell_signals['close'], 
                           color='lightcoral', marker='v', s=60, label='Sell', alpha=0.7)
            
            if len(strong_sell_signals) > 0:
                ax1.scatter(strong_sell_signals.index, strong_sell_signals['close'], 
                           color='darkred', marker='v', s=100, label='Strong Sell', alpha=0.9)
            
            # Mark stop loss exits with X
            if len(stop_loss_exits) > 0:
                ax1.scatter(stop_loss_exits.index, stop_loss_exits['close'], 
                           color='red', marker='x', s=150, label='Stop Loss', alpha=1.0, linewidths=3)
            
            ax1.set_title('Price Chart with 5-Level Trading Signals + Stop Loss')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Equity Curve
            if self.equity_curve:
                ax2.plot(self.equity_curve, 'b-', linewidth=2, label='Portfolio Value')
                ax2.axhline(y=self.initial_balance, color='gray', linestyle='--', 
                           alpha=0.7, label='Initial Balance')
                
                # Mark stop loss points on equity curve
                if self.stop_loss_trades:
                    sl_indices = []
                    sl_values = []
                    for trade in self.stop_loss_trades:
                        # Find approximate index in equity curve
                        for j, t in enumerate(self.trades):
                            if t == trade:
                                idx = sum([t['duration'] for t in self.trades[:j+1]])
                                if idx < len(self.equity_curve):
                                    sl_indices.append(idx)
                                    sl_values.append(self.equity_curve[idx])
                                break
                    
                    if sl_indices:
                        ax2.scatter(sl_indices, sl_values, color='red', marker='x', 
                                   s=100, label=f'Stop Loss ({len(sl_indices)})', zorder=5)
                
            ax2.set_title('Equity Curve')
            ax2.set_ylabel('Balance ($)')
            ax2.set_xlabel('Time')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"⚠️  Error creating plots: {e}")
            print("📊 Continuing without visualization...")
        
    def print_summary(self):
        """Print trading summary with stop loss analysis"""
        print("\n" + "="*50)
        print("📈 LEVERAGED BACKTEST SUMMARY (5-LEVEL + STOP LOSS)")
        print("="*50)
        
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${self.balance:,.2f}")
        print(f"Total Fees Paid: ${self.total_fees_paid:,.2f}")
        print(f"Total Funding Paid: ${self.total_funding_paid:,.2f}")
        print(f"Leverage Used: 1:{self.leverage}")
        
        # Calculate returns
        total_return = (self.balance / self.initial_balance - 1) * 100
        total_costs = self.total_fees_paid + self.total_funding_paid
        gross_return = ((self.balance + total_costs) / self.initial_balance - 1) * 100
        cost_impact = gross_return - total_return
        
        print(f"Net Return: {total_return:+.2f}%")
        print(f"Gross Return (before costs): {gross_return:+.2f}%")
        print(f"Total Cost Impact: -{cost_impact:.2f}%")
        print(f"Total Trades: {len(self.trades)}")
        
        if self.trades:
            wins = [t for t in self.trades if t['profit_pct'] > 0]
            losses = [t for t in self.trades if t['profit_pct'] <= 0]
            
            win_rate = len(wins) / len(self.trades) * 100
            print(f"Win Rate: {win_rate:.1f}% ({len(wins)}/{len(self.trades)})")
            
            if wins:
                avg_win = np.mean([t['profit_pct'] for t in wins])
                max_win = max([t['profit_pct'] for t in wins])
                print(f"Average Win: {avg_win:.2f}%")
                print(f"Best Trade: {max_win:.2f}%")
            
            if losses:
                avg_loss = np.mean([t['profit_pct'] for t in losses])
                max_loss = min([t['profit_pct'] for t in losses])
                print(f"Average Loss: {avg_loss:.2f}%")
                print(f"Worst Trade: {max_loss:.2f}%")
            
            # STOP LOSS ANALYSIS
            print("\n" + "-"*30)
            print("🛑 STOP LOSS ANALYSIS")
            print("-"*30)
            
            print(f"Total Stop Losses: {self.total_stop_losses}")
            print(f"Stop Loss Rate: {self.total_stop_losses/len(self.trades)*100:.1f}%")
            
            if self.stop_loss_trades:
                sl_losses = [t['profit_pct'] for t in self.stop_loss_trades]
                avg_sl_loss = np.mean(sl_losses)
                max_sl_loss = min(sl_losses)
                
                print(f"Average SL Loss: {avg_sl_loss:.2f}%")
                print(f"Worst SL Loss: {max_sl_loss:.2f}%")
                
                # Dynamic vs Fixed SL analysis
                if self.use_dynamic_sl:
                    sl_percentages = [t.get('stop_loss_pct', 0) for t in self.stop_loss_trades]
                    if sl_percentages:
                        print(f"Avg Dynamic SL%: {np.mean(sl_percentages):.2f}%")
                        print(f"Min SL%: {min(sl_percentages):.2f}%")
                        print(f"Max SL%: {max(sl_percentages):.2f}%")
            
            print("\n" + "-"*30)
            print("📊 POSITION SIZING ANALYSIS")
            print("-"*30)

            # Use actual position multipliers from config
            strong_positions = [t for t in self.trades if t.get('position_size', 0) == self.position_multipliers['strong']]
            normal_positions = [t for t in self.trades if t.get('position_size', 0) == self.position_multipliers['normal']]

            print(f"Strong Positions ({self.position_multipliers['strong']*100:.0f}%): {len(strong_positions)} ({len(strong_positions)/len(self.trades)*100:.1f}%)")
            print(f"Normal Positions ({self.position_multipliers['normal']*100:.0f}%): {len(normal_positions)} ({len(normal_positions)/len(self.trades)*100:.1f}%)")

            if strong_positions:
                strong_win_rate = len([t for t in strong_positions if t['profit_pct'] > 0]) / len(strong_positions) * 100
                strong_sl_rate = len([t for t in strong_positions if t.get('exit_reason') == 'STOP_LOSS']) / len(strong_positions) * 100
                print(f"Strong Position Win Rate: {strong_win_rate:.1f}%")
                print(f"Strong Position SL Rate: {strong_sl_rate:.1f}%")
                
            if normal_positions:
                normal_win_rate = len([t for t in normal_positions if t['profit_pct'] > 0]) / len(normal_positions) * 100
                normal_sl_rate = len([t for t in normal_positions if t.get('exit_reason') == 'STOP_LOSS']) / len(normal_positions) * 100
                print(f"Normal Position Win Rate: {normal_win_rate:.1f}%")
                print(f"Normal Position SL Rate: {normal_sl_rate:.1f}%")

            # TAMBAHAN: Debug untuk memastikan position_size tersimpan
            if self.trades:
                print("\n📍 Position Size Distribution:")
                position_sizes = {}
                for trade in self.trades:
                    size = trade.get('position_size', 0)
                    position_sizes[size] = position_sizes.get(size, 0) + 1
                
                for size, count in sorted(position_sizes.items()):
                    print(f"   Size {size*100:.0f}%: {count} trades ({count/len(self.trades)*100:.1f}%)")
            
            # Exit reason analysis
            print("\n" + "-"*30)
            print("📊 EXIT REASON ANALYSIS")
            print("-"*30)
            
            exit_reasons = [t.get('exit_reason', 'UNKNOWN') for t in self.trades]
            strong_reversal = len([r for r in exit_reasons if r == 'STRONG_REVERSAL'])
            sell_signal = len([r for r in exit_reasons if r == 'SELL_SIGNAL'])
            buy_signal = len([r for r in exit_reasons if r == 'BUY_SIGNAL'])
            stop_loss = len([r for r in exit_reasons if r == 'STOP_LOSS'])
            
            print(f"Strong Reversal: {strong_reversal} ({strong_reversal/len(self.trades)*100:.1f}%)")
            print(f"Sell Signal Exit: {sell_signal} ({sell_signal/len(self.trades)*100:.1f}%)")
            print(f"Buy Signal Exit: {buy_signal} ({buy_signal/len(self.trades)*100:.1f}%)")
            print(f"Stop Loss Exit: {stop_loss} ({stop_loss/len(self.trades)*100:.1f}%)")
            
            # Trade type analysis
            print("\n" + "-"*30)
            print("📊 TRADE TYPE ANALYSIS")
            print("-"*30)
            
            long_trades = [t for t in self.trades if t['type'] == 'LONG']
            short_trades = [t for t in self.trades if t['type'] == 'SHORT']
            
            print(f"Long Trades: {len(long_trades)} ({len(long_trades)/len(self.trades)*100:.1f}%)")
            print(f"Short Trades: {len(short_trades)} ({len(short_trades)/len(self.trades)*100:.1f}%)")
            
            if long_trades:
                long_win_rate = len([t for t in long_trades if t['profit_pct'] > 0]) / len(long_trades) * 100
                long_sl_rate = len([t for t in long_trades if t.get('exit_reason') == 'STOP_LOSS']) / len(long_trades) * 100
                print(f"Long Win Rate: {long_win_rate:.1f}%")
                print(f"Long SL Rate: {long_sl_rate:.1f}%")
                
            if short_trades:
                short_win_rate = len([t for t in short_trades if t['profit_pct'] > 0]) / len(short_trades) * 100
                short_sl_rate = len([t for t in short_trades if t.get('exit_reason') == 'STOP_LOSS']) / len(short_trades) * 100
                print(f"Short Win Rate: {short_win_rate:.1f}%")
                print(f"Short SL Rate: {short_sl_rate:.1f}%")
        
        print("="*50)


def main():
    """Main backtest function with configuration"""
    print("🚀 Starting Simple Backtest with 5-Level Strategy + STOP LOSS")
    print("=" * 50)
    
    try:
        # Initialize backtester dengan config defaults
        backtester = SimpleBacktest()
        
        # Load latest model with compatibility
        model = backtester.load_latest_model()
        
        # Prepare data using config
        features, raw_data = backtester.prepare_data()
        
        # Get predictions
        print("\n🎯 Getting model predictions...")
        predictions = model.predict(features)
        print(f"✅ Generated {len(predictions)} predictions")
        print(f"🔍 Prediction vs Features: {len(predictions)}/{len(features)} ({len(predictions)/len(features)*100:.1f}%)")
        
        # Run backtest
        data_with_signals = backtester.run_backtest(predictions, raw_data, model)
        
        # Print summary
        backtester.print_summary()
        
        # Plot results
        backtester.plot_results(data_with_signals)
        
        print("\n🎉 Backtest completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\n❌ File Error: {e}")
        print("💡 Make sure you have:")
        print("   - Trained models in the 'models' folder")
        print("   - Data access configured properly")
        
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"\n❌ Model Architecture Mismatch: {e}")
            print("💡 This usually means:")
            print("   - Model was trained with different hidden_size")
            print("   - Try retraining the model or check saved model parameters")
        else:
            print(f"\n❌ Runtime Error: {e}")
            
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        print("\n🔍 Error details:")
        traceback.print_exc()


# ==========================================
# 🎛️ QUICK BACKTESTING PRESETS
# ==========================================

class BacktestPresets:
    """Predefined backtesting configurations for different scenarios"""
    
    @staticmethod
    def conservative_trading():
        """Conservative trading settings"""
        BacktestConfig.TRADING_FEE = 0.0005
        BacktestConfig.SPREAD = 0.0003
        BacktestConfig.LEVERAGE = 50
        BacktestConfig.STOP_LOSS_PCT = 0.015
        BacktestConfig.POSITION_MULTIPLIER_STRONG = 0.8
        BacktestConfig.POSITION_MULTIPLIER_NORMAL = 0.4
    
    @staticmethod
    def aggressive_trading():
        """Aggressive trading settings"""
        BacktestConfig.TRADING_FEE = 0.001
        BacktestConfig.SPREAD = 0.0007
        BacktestConfig.LEVERAGE = 200
        BacktestConfig.STOP_LOSS_PCT = 0.025
        BacktestConfig.POSITION_MULTIPLIER_STRONG = 1.0
        BacktestConfig.POSITION_MULTIPLIER_NORMAL = 0.8
    
    @staticmethod
    def quick_test():
        """Quick testing with small dataset"""
        BacktestConfig.DATA_LIMIT = 500
        BacktestConfig.INITIAL_BALANCE = 5000
        BacktestConfig.LEVERAGE = 100
    
    @staticmethod
    def comprehensive_test():
        """Comprehensive testing with large dataset"""
        BacktestConfig.DATA_LIMIT = 5000
        BacktestConfig.INITIAL_BALANCE = 50000
        BacktestConfig.LEVERAGE = 100


if __name__ == "__main__":
    # Quick setup examples:
    # BacktestPresets.conservative_trading()    # Conservative settings
    # BacktestPresets.aggressive_trading()      # Aggressive settings
    # BacktestPresets.quick_test()              # Quick testing
    # BacktestPresets.comprehensive_test()      # Comprehensive testing
    
    # Or manual configuration:
    # BacktestConfig.INITIAL_BALANCE = 20000
    # BacktestConfig.LEVERAGE = 150
    # BacktestConfig.STRONG_BUY = 0.7
    
    main()