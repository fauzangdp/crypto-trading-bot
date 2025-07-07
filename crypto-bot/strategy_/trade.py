# strategy_/trade.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import time
from datetime import datetime
import tensorflow as tf  # Add this import

from library.data_utils import DataLoader, prepare_multi_timeframe_data
from library.indicators import TechnicalIndicators as TI
from library.model_utils import ModelManager
from strategy_.features import create_all_features

class FuturesTrader:
    """Simple futures trading bot"""
    
    def __init__(self, model_path, symbol='BTCUSDT', leverage=10):
        """
        Initialize futures trader
        
        Args:
            model_path: path to saved model
            symbol: trading pair
            leverage: futures leverage (default 10x)
        """
        self.symbol = symbol
        self.leverage = leverage
        self.position = None
        self.model = None
        self.load_model(model_path)
        
        # Trading parameters
        self.min_confidence = 0.7
        self.risk_per_trade = 0.02  # 2% risk
        self.capital = 1000  # Starting capital
        
        # Update: Multi-timeframe config
        self.timeframes = ['1m', '5m', '1h', '4h']
        
    def load_model(self, model_path):
        """Load trained model"""
        print(f"Loading model from: {model_path}")
        
        # Alternative fix: Skip Lambda layer issue
        try:
            # Load model architecture as JSON first
            import json
            model_file = os.path.join(model_path, 'model.h5')
            
            # Try to load with h5py and reconstruct
            import h5py
            
            # Load weights
            with h5py.File(model_file, 'r') as f:
                # Build model manually if Lambda fails
                from tensorflow.keras import layers, Model
                
                # Recreate model architecture (adjust based on your actual model)
                inputs = layers.Input(shape=(30, None))  # Dynamic feature size
                
                # LSTM layers
                x = layers.LSTM(256, return_sequences=True)(inputs)
                x = layers.Dropout(0.2)(x)
                x = layers.LSTM(128, return_sequences=True)(x)  
                x = layers.Dropout(0.2)(x)
                
                # Simple attention (avoid Lambda)
                x = layers.GlobalAveragePooling1D()(x)  # Replace Lambda with this
                
                # Dense layers
                x = layers.Dense(64, activation='relu')(x)
                x = layers.Dropout(0.2)(x)
                outputs = layers.Dense(8)(x)
                
                # Create model
                self.model = Model(inputs=inputs, outputs=outputs)
                
                # Load weights from h5 file
                self.model.load_weights(model_file, by_name=True, skip_mismatch=True)
            
            # Compile model
            self.model.compile(
                optimizer='adam',
                loss=self._directional_loss,
                metrics=['mae', 'mse']
            )
            
            # Load config
            with open(os.path.join(model_path, 'config.json'), 'r') as f:
                config = json.load(f)
            
            print("Model loaded successfully (rebuilt architecture)")
            print(f"Model input shape: {config.get('input_shape', 'Unknown')}")
            
        except Exception as e:
            # Fallback: try normal load
            print("Trying standard load...")
            model_file = os.path.join(model_path, 'model.h5')
            self.model = tf.keras.models.load_model(model_file, compile=False)
            self.model.compile(
                optimizer='adam',
                loss=self._directional_loss,
                metrics=['mae', 'mse']
            )
            print("Model loaded with standard method")
    
    def _directional_loss(self, y_true, y_pred):
        """Directional loss function for model compilation"""
        # Extract direction (first output)
        direction_true = y_true[:, 0]
        direction_pred = y_pred[:, 0]
        
        # Direction loss (MSE)
        direction_loss = tf.keras.losses.mse(direction_true, direction_pred)
        
        # Other outputs loss
        other_loss = tf.keras.losses.mse(y_true[:, 1:], y_pred[:, 1:])
        
        # Weight direction more
        return direction_loss * 2.0 + other_loss
    
    def get_latest_data(self, lookback=100):
        """
        Get latest multi-timeframe data
        In real trading, this would connect to exchange API
        """
        # Update: Load multi-timeframe data
        loader = DataLoader()
        mtf_data = {}
        
        for tf in self.timeframes:
            try:
                df = loader.load_ohlcv(self.symbol, tf)
                # Get appropriate lookback for each timeframe
                if tf == '1m':
                    mtf_data[tf] = df.tail(lookback)
                elif tf == '5m':
                    mtf_data[tf] = df.tail(lookback // 5 + 10)
                elif tf == '1h':
                    mtf_data[tf] = df.tail(lookback // 60 + 5)
                elif tf == '4h':
                    mtf_data[tf] = df.tail(lookback // 240 + 2)
            except:
                print(f"Warning: Could not load {tf} data")
                mtf_data[tf] = None
        
        return mtf_data
    
    def generate_signal(self, mtf_data):
        """Generate trading signal from multi-timeframe data"""
        # Update: Create multi-timeframe features
        features_df = create_all_features(self.symbol, self.timeframes)
        
        # Create sequences
        loader = DataLoader()
        sequences = loader.create_sequences(features_df, sequence_length=30)
        
        if len(sequences) == 0:
            return None
        
        # Get latest sequence
        latest_X = sequences[-1:] 
        
        # Predict
        prediction = self.model.predict(latest_X, verbose=0)[0]
        
        # Parse prediction
        signal = {
            'direction': prediction[0],
            'confidence': prediction[1],
            'tp1': prediction[2],
            'tp2': prediction[3],
            'tp3': prediction[4],
            'sl': prediction[5],
            'timing': int(prediction[6]),
            'risk': prediction[7]
        }
        
        return signal
    
    def calculate_position_size(self, stop_loss_distance):
        """
        Calculate position size for futures
        
        Args:
            stop_loss_distance: SL distance in price
            
        Returns:
            position size (contracts)
        """
        # Risk amount
        risk_amount = self.capital * self.risk_per_trade
        
        # Position size with leverage
        position_size = (risk_amount / stop_loss_distance) * self.leverage
        
        return position_size
    
    def place_order(self, signal, current_price, atr):
        """
        Place futures order
        In real trading, this would use exchange API
        """
        # Check if we should trade
        if signal['confidence'] < self.min_confidence:
            print(f"Low confidence: {signal['confidence']:.2f}. Skip trade.")
            return
        
        if abs(signal['direction']) < 0.3:
            print("No clear direction. Skip trade.")
            return
        
        # Determine trade type
        if signal['direction'] > 0:
            trade_type = 'LONG'
            entry = current_price
            tp1 = entry + (signal['tp1'] * atr)
            tp2 = entry + (signal['tp2'] * atr)
            tp3 = entry + (signal['tp3'] * atr)
            sl = entry - (signal['sl'] * atr)
        else:
            trade_type = 'SHORT'
            entry = current_price
            tp1 = entry - (signal['tp1'] * atr)
            tp2 = entry - (signal['tp2'] * atr)
            tp3 = entry - (signal['tp3'] * atr)
            sl = entry + (signal['sl'] * atr)
        
        # Calculate position size
        sl_distance = abs(entry - sl)
        position_size = self.calculate_position_size(sl_distance)
        
        # Create position
        self.position = {
            'type': trade_type,
            'entry': entry,
            'size': position_size,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'sl': sl,
            'status': 'OPEN',
            'entry_time': datetime.now()
        }
        
        print(f"\n=== {trade_type} Order Placed ===")
        print(f"Entry: {entry:.2f}")
        print(f"Size: {position_size:.4f} contracts")
        print(f"Leverage: {self.leverage}x")
        print(f"TP1: {tp1:.2f} ({abs(tp1-entry):.2f} points)")
        print(f"TP2: {tp2:.2f} ({abs(tp2-entry):.2f} points)")
        print(f"TP3: {tp3:.2f} ({abs(tp3-entry):.2f} points)")
        print(f"SL: {sl:.2f} ({sl_distance:.2f} points)")
        print(f"Risk: ${self.capital * self.risk_per_trade:.2f}")
    
    def check_position(self, current_price):
        """Check and manage open position"""
        if not self.position or self.position['status'] == 'CLOSED':
            return
        
        pos = self.position
        
        if pos['type'] == 'LONG':
            # Check stop loss
            if current_price <= pos['sl']:
                self.close_position(current_price, 'STOP_LOSS')
                return
            
            # Check take profits
            if current_price >= pos['tp1'] and pos['status'] == 'OPEN':
                print(f"TP1 hit at {current_price:.2f}")
                # Close 50% and move SL to breakeven
                pos['size'] *= 0.5
                pos['sl'] = pos['entry']
                pos['status'] = 'TP1_HIT'
            
            elif current_price >= pos['tp2'] and pos['status'] == 'TP1_HIT':
                print(f"TP2 hit at {current_price:.2f}")
                # Close another 50% of remaining
                pos['size'] *= 0.5
                pos['status'] = 'TP2_HIT'
            
            elif current_price >= pos['tp3']:
                self.close_position(current_price, 'TP3_HIT')
        
        else:  # SHORT
            # Check stop loss
            if current_price >= pos['sl']:
                self.close_position(current_price, 'STOP_LOSS')
                return
            
            # Check take profits
            if current_price <= pos['tp1'] and pos['status'] == 'OPEN':
                print(f"TP1 hit at {current_price:.2f}")
                pos['size'] *= 0.5
                pos['sl'] = pos['entry']
                pos['status'] = 'TP1_HIT'
            
            elif current_price <= pos['tp2'] and pos['status'] == 'TP1_HIT':
                print(f"TP2 hit at {current_price:.2f}")
                pos['size'] *= 0.5
                pos['status'] = 'TP2_HIT'
            
            elif current_price <= pos['tp3']:
                self.close_position(current_price, 'TP3_HIT')
    
    def close_position(self, exit_price, reason):
        """Close position and calculate PnL"""
        if not self.position:
            return
        
        pos = self.position
        
        # Calculate PnL
        if pos['type'] == 'LONG':
            pnl_points = exit_price - pos['entry']
        else:  # SHORT
            pnl_points = pos['entry'] - exit_price
        
        pnl_percentage = (pnl_points / pos['entry']) * 100 * self.leverage
        pnl_dollar = self.capital * (pnl_percentage / 100)
        
        # Update capital
        self.capital += pnl_dollar
        
        print(f"\n=== Position Closed: {reason} ===")
        print(f"Exit: {exit_price:.2f}")
        print(f"PnL: {pnl_points:.2f} points ({pnl_percentage:.2f}%)")
        print(f"PnL $: ${pnl_dollar:.2f}")
        print(f"New Capital: ${self.capital:.2f}")
        
        # Clear position
        self.position['status'] = 'CLOSED'
    
    def run(self, iterations=100, delay=5):
        """
        Run trading bot
        
        Args:
            iterations: number of iterations (for demo)
            delay: seconds between checks
        """
        print(f"\n=== Starting Futures Trading Bot ===")
        print(f"Symbol: {self.symbol}")
        print(f"Leverage: {self.leverage}x")
        print(f"Capital: ${self.capital}")
        print(f"Risk per trade: {self.risk_per_trade*100}%")
        print(f"Timeframes: {', '.join(self.timeframes)}")  # Update: Show timeframes
        
        for i in range(iterations):
            try:
                # Update: Get multi-timeframe data
                mtf_data = self.get_latest_data()
                
                # Get current price from 1m data
                if '1m' in mtf_data and mtf_data['1m'] is not None:
                    current_price = mtf_data['1m']['close'].iloc[-1]
                    
                    # Calculate ATR from 1m data
                    ti = TI()
                    atr = ti.atr(
                        mtf_data['1m']['high'], 
                        mtf_data['1m']['low'], 
                        mtf_data['1m']['close']
                    ).iloc[-1]
                else:
                    print("Error: No 1m data available")
                    continue
                
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Price: {current_price:.2f}")
                
                # Check existing position
                if self.position and self.position['status'] != 'CLOSED':
                    self.check_position(current_price)
                else:
                    # Generate new signal with multi-timeframe data
                    signal = self.generate_signal(mtf_data)
                    if signal:
                        print(f"Signal: {signal['direction']:.2f}, Confidence: {signal['confidence']:.2f}")
                        self.place_order(signal, current_price, atr)
                
                # Wait before next check
                if i < iterations - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"Error: {str(e)}")
                time.sleep(delay)
        
        print(f"\n=== Trading Complete ===")
        print(f"Final Capital: ${self.capital:.2f}")
        print(f"Total Return: {((self.capital - 1000) / 1000) * 100:.2f}%")


# Quick start functions
def start_trading(model_path, symbol='BTCUSDT', leverage=10):
    """Start trading with saved model"""
    trader = FuturesTrader(model_path, symbol, leverage)
    trader.run(iterations=100, delay=5)


def demo_trading():
    """Demo trading with dummy model path"""
    # Use latest model in models directory
    import glob
    models = glob.glob('models/**/model.h5', recursive=True)
    if not models:
        print("No trained models found!")
        return
    
    latest_model = max(models, key=os.path.getctime)
    model_dir = os.path.dirname(latest_model)
    
    print(f"Using model: {model_dir}")
    start_trading(model_dir)


if __name__ == "__main__":
    # Run demo
    demo_trading()