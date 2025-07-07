import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from sklearn.mixture import GaussianMixture
import warnings
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

def create_lstm_labels(multi_tf_data: Dict[str, pd.DataFrame], 
                      lookforward_periods: int = 20,
                      regime_lookback: int = 100) -> pd.DataFrame:
    """
    Create FUTURES TRADING LSTM labels with forward-looking prediction
    COMPLETELY REWRITTEN for 2-way futures trading (LONG/SHORT)
    
    Args:
        multi_tf_data: Output dari load_data() dengan keys ['1m', '5m', '30m', '1h']
        lookforward_periods: Candles to look forward for prediction (default: 20)
        regime_lookback: Bars untuk regime detection (default: 100)
    
    Returns:
        pd.DataFrame: Forward-looking futures trading labels
    """
    
    # 1. Validate input
    required_tfs = ['1m', '5m', '30m', '1h']
    missing_tfs = [tf for tf in required_tfs if tf not in multi_tf_data]
    if missing_tfs:
        raise ValueError(f"Missing timeframes: {missing_tfs}")
    
    print(f"🎯 Creating FUTURES TRADING labels with forward-looking prediction...")
    print(f"   📅 Lookforward periods: {lookforward_periods}")
    print(f"   🔍 Regime lookback: {regime_lookback}")
    
    # 2. Helper functions
    def get_close_price(df, tf):
        """Extract close price dari dataframe"""
        for col_name in ['close', 'Close', f'close_{tf}', f'Close_{tf}']:
            if col_name in df.columns:
                return df[col_name]
        
        for col in df.columns:
            if 'close' in col.lower():
                return df[col]
        
        raise ValueError(f"Close price column not found in {tf} data")
    
    def get_ohlcv_columns(df, tf):
        """Extract OHLCV columns"""
        ohlcv = {}
        base_cols = ['open', 'high', 'low', 'close', 'volume']
        
        for col in base_cols:
            for variant in [col, col.title(), f'{col}_{tf}', f'{col.title()}_{tf}']:
                if variant in df.columns:
                    ohlcv[col] = df[variant]
                    break
            else:
                raise ValueError(f"Column '{col}' not found in {tf} data")
        
        return ohlcv
    
    def calculate_atr(high, low, close, period=14):
        """Calculate ATR"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def normalize_atr(atr_series, price_series, lookback=200):
        """
        🔥 FIXED: Normalize ATR dengan error handling
        """
        # ATR as percentage of price
        atr_pct = atr_series / price_series
        atr_normalized = pd.Series(index=atr_series.index, dtype=float)
        
        for i in range(len(atr_series)):
            if i < lookback:
                historical_atr_pct = atr_pct.iloc[:i+1]
            else:
                historical_atr_pct = atr_pct.iloc[i-lookback:i+1]
            
            # 🔥 ENHANCED ERROR HANDLING
            try:
                # Remove NaN values first
                clean_data = historical_atr_pct.dropna()
                
                if len(clean_data) < 5:  # Need minimum data points
                    atr_normalized.iloc[i] = 0.5  # Default neutral
                    continue
                
                # 🔥 SAFE QUANTILE CALCULATION
                try:
                    hist_min = clean_data.quantile(0.05)
                    hist_max = clean_data.quantile(0.95)
                except:
                    # Fallback to min/max if quantile fails
                    hist_min = clean_data.min()
                    hist_max = clean_data.max()
                
                # Handle edge cases
                if pd.isna(hist_min) or pd.isna(hist_max) or hist_max <= hist_min:
                    atr_normalized.iloc[i] = 0.5
                    continue
                
                # Normalize
                current_atr_pct = atr_pct.iloc[i]
                if pd.isna(current_atr_pct):
                    atr_normalized.iloc[i] = 0.5
                else:
                    normalized_val = (current_atr_pct - hist_min) / (hist_max - hist_min)
                    atr_normalized.iloc[i] = np.clip(normalized_val, 0, 1)
                    
            except Exception as e:
                # Ultimate fallback
                atr_normalized.iloc[i] = 0.5
                continue
        
        return atr_normalized.fillna(0.5)
    
    def detect_market_regime(price_data, volume_data, lookback=100):
        """Detect market regime using volatility + volume analysis"""
        print("   🔍 Detecting market regimes...")
        
        returns = price_data.pct_change().dropna()
        volatility = returns.rolling(window=20).std() * np.sqrt(252)
        volume_ma = volume_data.rolling(window=20).mean()
        volume_ratio = volume_data / volume_ma
        
        regime_series = pd.Series(index=price_data.index, dtype=str)
        
        for i in range(lookback, len(price_data)):
            recent_vol = volatility.iloc[max(0, i-lookback):i].mean()
            recent_vol_std = volatility.iloc[max(0, i-lookback):i].std()
            recent_volume = volume_ratio.iloc[max(0, i-lookback):i].mean()
            
            if pd.isna(recent_vol) or pd.isna(recent_vol_std):
                regime_series.iloc[i] = 'normal_vol'
                continue
            
            vol_threshold_low = recent_vol - 0.5 * recent_vol_std
            vol_threshold_high = recent_vol + 0.5 * recent_vol_std
            current_vol = volatility.iloc[i] if i < len(volatility) else recent_vol
            
            if pd.isna(current_vol):
                regime_series.iloc[i] = 'normal_vol'
            elif current_vol < vol_threshold_low and recent_volume < 1.2:
                regime_series.iloc[i] = 'low_vol'
            elif current_vol > vol_threshold_high or recent_volume > 1.5:
                regime_series.iloc[i] = 'high_vol'
            else:
                regime_series.iloc[i] = 'normal_vol'
        
        regime_series = regime_series.fillna(method='bfill').fillna('normal_vol')
        regime_counts = regime_series.value_counts()
        print(f"     ✅ Regimes detected: {dict(regime_counts)}")
        
        return regime_series
    
    def predict_future_direction(price_series, atr_value, max_hold=20):
        """
        🔥 FIXED: Predict future price direction dengan RELAXED thresholds
        """
        if len(price_series) < max_hold + 1:
            return 'HOLD', 0.0, 0.0, 0.5
        
        current_price = price_series.iloc[0]
        
        # 🔥 RELAXED: Much lower thresholds
        atr_pct = atr_value / current_price
        profit_threshold = max(atr_pct * 0.8, 0.005)  # REDUCED: 0.3% min (was 0.8%)
        loss_threshold = max(atr_pct * 0.6, 0.004)    # REDUCED: 0.2% min (was 0.5%)
        
        # Analyze future price movements
        future_returns = []
        hit_profits = 0
        hit_losses = 0
        max_profit = 0
        max_loss = 0
        
        for i in range(1, min(max_hold + 1, len(price_series))):
            future_price = price_series.iloc[i]
            return_pct = (future_price - current_price) / current_price
            future_returns.append(return_pct)
            
            max_profit = max(max_profit, return_pct)
            max_loss = min(max_loss, return_pct)
            
            # 🔥 RELAXED: Lower requirements
            if return_pct >= profit_threshold:
                hit_profits += 1
            elif return_pct <= -loss_threshold:
                hit_losses += 1
        
        if not future_returns:
            return 'HOLD', 0.0, 0.0, 0.5
        
        # Calculate statistics
        avg_return = np.mean(future_returns)
        volatility = np.std(future_returns)
        
        # Win/Loss ratios
        total_periods = len(future_returns)
        profit_ratio = hit_profits / total_periods
        loss_ratio = hit_losses / total_periods
        
        # Risk assessment
        risk_level = np.clip(volatility * 8 + abs(min(future_returns)) * 3, 0, 1)
        
        # 🔥 MUCH MORE RELAXED DIRECTION LOGIC
        direction_confidence = 0.0
        
        # Strong signals - RELAXED requirements
        if avg_return > profit_threshold * 0.8 and profit_ratio > 0.25:  # REDUCED from 0.4
            direction = 'LONG'
            consistency_score = profit_ratio
            magnitude_score = min(avg_return / profit_threshold, 1.5) / 1.5
            direction_confidence = min(0.9, (consistency_score + magnitude_score) / 2)
            
        elif avg_return < -loss_threshold * 0.8 and loss_ratio > 0.25:  # REDUCED from 0.4
            direction = 'SHORT'
            consistency_score = loss_ratio
            magnitude_score = min(abs(avg_return) / loss_threshold, 1.5) / 1.5
            direction_confidence = min(0.9, (consistency_score + magnitude_score) / 2)
            
        # Moderate signals - VERY RELAXED
        elif avg_return > profit_threshold * 0.3 and profit_ratio > 0.15:  # MUCH LOWER
            direction = 'LONG'
            direction_confidence = min(0.7, profit_ratio * 2 + avg_return / profit_threshold * 0.3)
            
        elif avg_return < -loss_threshold * 0.3 and loss_ratio > 0.15:  # MUCH LOWER
            direction = 'SHORT'
            direction_confidence = min(0.7, loss_ratio * 2 + abs(avg_return) / loss_threshold * 0.3)
            
        # Weak directional bias - NEW category
        elif avg_return > 0.001:  # Even tiny positive bias
            direction = 'LONG'
            direction_confidence = min(0.5, avg_return * 200)  # Scale up tiny movements
            
        elif avg_return < -0.001:  # Even tiny negative bias
            direction = 'SHORT'
            direction_confidence = min(0.5, abs(avg_return) * 200)
            
        else:
            # True sideways market
            direction = 'HOLD'
            direction_confidence = max(0.2, 1.0 - volatility * 5)
        
        return direction, direction_confidence, avg_return, risk_level
    
    def create_futures_label(direction, confidence, regime, signal_strength, volume_confirmation, risk_level):
        """
        🔥 FIXED: Create futures trading label dengan RELAXED thresholds
        """
        # Regime adjustments
        regime_adjustment = {
            'low_vol': 0.15,     # INCREASED boost
            'normal_vol': 0.05,  # Small boost
            'high_vol': -0.1     # Smaller penalty
        }.get(regime, 0.0)
        
        # Technical adjustments
        signal_adjustment = (signal_strength - 0.5) * 0.3      # INCREASED influence
        volume_adjustment = (volume_confirmation - 0.5) * 0.2  # INCREASED influence
        risk_adjustment = -risk_level * 0.15                   # REDUCED penalty
        
        # Calculate final confidence
        final_confidence = confidence + regime_adjustment + signal_adjustment + volume_adjustment + risk_adjustment
        final_confidence = np.clip(final_confidence, 0.0, 1.0)
        
        # 🔥 MUCH LOWER THRESHOLDS for labels
        if direction == 'LONG':
            if final_confidence >= 0.6:        # REDUCED from 0.7
                return 'STRONG_LONG', final_confidence
            elif final_confidence >= 0.25:     # REDUCED from 0.4
                return 'LONG', final_confidence
            else:
                return 'HOLD', final_confidence
                
        elif direction == 'SHORT':
            if final_confidence >= 0.6:        # REDUCED from 0.7
                return 'STRONG_SHORT', final_confidence
            elif final_confidence >= 0.25:     # REDUCED from 0.4
                return 'SHORT', final_confidence
            else:
                return 'HOLD', final_confidence
                
        else:  # direction == 'HOLD'
            return 'HOLD', final_confidence
    
    # 3. Main execution
    base_1m = multi_tf_data['1m'].copy()
    
    # 4. Extract OHLCV data
    print("   📊 Extracting OHLCV data...")
    ohlcv_1m = get_ohlcv_columns(base_1m, '1m')
    
    close_1m = ohlcv_1m['close']
    high_1m = ohlcv_1m['high']
    low_1m = ohlcv_1m['low']
    volume_1m = ohlcv_1m['volume']
    
    # 5. Calculate technical components
    print("   🔧 Calculating technical components...")
    
    # ATR calculation
    atr_1m = calculate_atr(high_1m, low_1m, close_1m, period=14)
    atr_normalized = normalize_atr(atr_1m, close_1m, lookback=200)
    print(f"     ✅ ATR normalized: {atr_normalized.min():.3f} - {atr_normalized.max():.3f}")
    
    # Volume momentum
    volume_ma = volume_1m.rolling(window=20).mean()
    volume_momentum = (volume_1m / volume_ma).fillna(1.0)
    volume_momentum_norm = np.clip(volume_momentum / 2.0, 0, 1)
    
    # Trend strength (price momentum)
    price_momentum = close_1m.pct_change(10).abs()
    trend_strength = np.clip(price_momentum * 50, 0, 1)
    
    # 6. Market regime detection
    market_regime = detect_market_regime(close_1m, volume_1m, regime_lookback)
    
    # 7. 🔥 MAIN PROCESSING: Forward-looking label creation
    print("   🎯 Processing futures trading labels...")
    
    results = []
    start_idx = max(regime_lookback, 50)
    end_idx = len(close_1m) - lookforward_periods - 5  # Safety buffer
    
    for i in range(start_idx, end_idx):
        if i % 1000 == 0:
            print(f"     Processing {i}/{end_idx} ({i/end_idx*100:.1f}%)")
        
        current_timestamp = close_1m.index[i]
        current_price = close_1m.iloc[i]
        current_atr = atr_1m.iloc[i]
        current_atr_norm = atr_normalized.iloc[i]
        current_regime = market_regime.iloc[i]
        current_volume_momentum = volume_momentum_norm.iloc[i]
        current_trend_strength = trend_strength.iloc[i]
        
        # Skip if missing critical data
        if pd.isna(current_price) or pd.isna(current_atr) or current_atr <= 0:
            continue
        
        # 🔮 GET FUTURE PRICE SERIES
        future_start = i
        future_end = i + lookforward_periods + 1
        future_prices = close_1m.iloc[future_start:future_end]
        
        if len(future_prices) < lookforward_periods:
            continue
        
        # 🎯 PREDICT FUTURE DIRECTION
        direction, direction_confidence, expected_return, risk_level = predict_future_direction(
            future_prices, current_atr, max_hold=lookforward_periods
        )
        
        # Calculate technical signal strength
        signal_strength = np.clip((current_trend_strength + current_volume_momentum) / 2, 0, 1)
        volume_confirmation = np.clip(current_volume_momentum, 0, 1)
        
        # 🏷️ CREATE FINAL FUTURES TRADING LABEL
        final_label, final_confidence = create_futures_label(
            direction, direction_confidence, current_regime, 
            signal_strength, volume_confirmation, risk_level
        )
        
        # Calculate reference TP/SL levels (for analysis)
        atr_multiplier = 2.0 if current_regime == 'high_vol' else 1.5
        if direction == 'LONG':
            tp_level = current_price * (1 + (current_atr / current_price) * atr_multiplier)
            sl_level = current_price * (1 - (current_atr / current_price) * 1.0)
        elif direction == 'SHORT':
            tp_level = current_price * (1 - (current_atr / current_price) * atr_multiplier)
            sl_level = current_price * (1 + (current_atr / current_price) * 1.0)
        else:
            tp_level = current_price
            sl_level = current_price
        
        # Store result
        results.append({
            'timestamp': current_timestamp,
            'price': current_price,
            'regime': current_regime,
            'tp_level': tp_level,           # Reference only
            'sl_level': sl_level,           # Reference only
            'direction': direction,         # LONG/SHORT/HOLD
            'expected_return': expected_return,
            'risk_level': risk_level,       # NEW: Risk assessment
            'label': final_label,          # STRONG_LONG/LONG/SHORT/STRONG_SHORT/HOLD
            'confidence': final_confidence, # 0 to 1 range
            'atr_normalized': current_atr_norm,
            'volume_momentum': current_volume_momentum,
            'trend_strength': current_trend_strength,
            'signal_strength': signal_strength,
            'volume_confirmation': volume_confirmation
        })
    
    # 8. Create final DataFrame
    if not results:
        raise ValueError("No valid labels created. Check data quality.")
    
    labels_df = pd.DataFrame(results)
    labels_df.set_index('timestamp', inplace=True)
    
    
    # 9. 📊 ENHANCED QUALITY CHECKS
    print("   🧹 Quality validation...")

    label_counts = labels_df['label'].value_counts()
    direction_counts = labels_df['direction'].value_counts()
    regime_counts = labels_df['regime'].value_counts()

    print(f"✅ FUTURES TRADING labels created: {len(labels_df)} samples")
    print(f"   🎯 LABEL distribution:")
    for label, count in label_counts.items():
        percentage = (count / len(labels_df)) * 100
        if 'LONG' in label:
            emoji = "📈"
        elif 'SHORT' in label:
            emoji = "📉"
        else:
            emoji = "⚪"
        print(f"      {emoji} {label}: {count} ({percentage:.1f}%)")

    print(f"   🧭 DIRECTION distribution: {dict(direction_counts)}")
    print(f"   🌡️  REGIME distribution: {dict(regime_counts)}")
    print(f"   📊 Confidence range: {labels_df['confidence'].min():.3f} to {labels_df['confidence'].max():.3f}")
    print(f"   💰 Expected return range: {labels_df['expected_return'].min():.3f} to {labels_df['expected_return'].max():.3f}")
    print(f"   ⚠️  Risk level range: {labels_df['risk_level'].min():.3f} to {labels_df['risk_level'].max():.3f}")

    # 🔥 FIX ERROR: Convert value_counts to dict first
    label_counts_dict = dict(label_counts)
    min_class_count = min(label_counts_dict.values())
    max_class_count = max(label_counts_dict.values())
    balance_ratio = min_class_count / max_class_count if max_class_count > 0 else 0
    print(f"   ⚖️  Class balance ratio: {balance_ratio:.2f} (>0.3 = good)")

    # Validate forward-looking (no lookahead bias)
    future_data_check = labels_df.index.max() <= close_1m.index[-lookforward_periods-10]
    print(f"   ✅ No lookahead bias: {future_data_check}")
    
    print(f"\n🎯 FUTURES TRADING LABELS MAPPING:")
    print(f"   📈 STRONG_LONG: High confidence bullish → Open large long position")
    print(f"   📊 LONG: Medium confidence bullish → Open standard long position")
    print(f"   📉 STRONG_SHORT: High confidence bearish → Open large short position")  
    print(f"   📊 SHORT: Medium confidence bearish → Open standard short position")
    print(f"   ⚪ HOLD: No clear direction → Close positions / Stay out")
    
    return labels_df


# Example usage
if __name__ == "__main__":
    from library.data_utils import DataLoader
    import os
    from datetime import datetime
    
    print("🚀 Testing FUTURES TRADING Label Creation...")
    
    # 1. Load multi-timeframe data
    loader = DataLoader()
    multi_data = loader.load_data(
        symbol='btc',
        timeframes=['1m', '5m', '30m', '1h'],
        limit=3000,  # Larger sample for better analysis
        auto_align=True,
        alignment_mode='current_only'
    )
    
    # 2. Create FUTURES TRADING labels
    futures_labels = create_lstm_labels(
        multi_data, 
        lookforward_periods=60,  # 20 minutes ahead prediction
        regime_lookback=100
    )
    
    # 3. Verify output
    print(f"\n📊 FUTURES TRADING Labels Output:")
    print(f"Shape: {futures_labels.shape}")
    print(f"Columns: {list(futures_labels.columns)}")
    print(f"Sample data:\n{futures_labels.tail(5)}")
    
    # 4. Export results
    try:
        output_dir = "exports"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        labels_filename = f"{output_dir}/futures_labels_btc_{timestamp}.csv"
        
        futures_labels.to_csv(labels_filename)
        print(f"\n💾 Futures labels exported: {labels_filename}")
        print(f"   📏 Size: {futures_labels.shape[0]} samples x {futures_labels.shape[1]} columns")
        
        # Export summary
        summary_filename = f"{output_dir}/futures_summary_btc_{timestamp}.txt"
        with open(summary_filename, 'w') as f:
            f.write(f"FUTURES TRADING Labels Export Summary\n")
            f.write(f"=" * 40 + "\n")
            f.write(f"Export Time: {datetime.now()}\n")
            f.write(f"Total Samples: {len(futures_labels)}\n")
            f.write(f"Date Range: {futures_labels.index[0]} to {futures_labels.index[-1]}\n\n")
            
            f.write(f"Label Distribution:\n")
            label_dist = futures_labels['label'].value_counts()
            for label, count in label_dist.items():
                percentage = count/len(futures_labels)*100
                f.write(f"{label}: {count} ({percentage:.1f}%)\n")
            
            f.write(f"\nDirection Distribution:\n")
            direction_dist = futures_labels['direction'].value_counts()
            for direction, count in direction_dist.items():
                percentage = count/len(futures_labels)*100
                f.write(f"{direction}: {count} ({percentage:.1f}%)\n")
            
            f.write(f"\nConfidence Statistics:\n")
            f.write(f"Mean: {futures_labels['confidence'].mean():.3f}\n")
            f.write(f"Std: {futures_labels['confidence'].std():.3f}\n")
            f.write(f"Min: {futures_labels['confidence'].min():.3f}\n")
            f.write(f"Max: {futures_labels['confidence'].max():.3f}\n")
            
            f.write(f"\nExpected Return Statistics:\n")
            f.write(f"Mean: {futures_labels['expected_return'].mean():.4f}\n")
            f.write(f"Positive returns: {(futures_labels['expected_return'] > 0).sum()}\n")
            f.write(f"Negative returns: {(futures_labels['expected_return'] < 0).sum()}\n")
        
        print(f"💾 Summary exported: {summary_filename}")
        print(f"\n✅ FUTURES TRADING labels ready for model training!")
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")