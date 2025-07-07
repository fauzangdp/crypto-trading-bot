import pandas as pd
import numpy as np
from typing import Dict
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from library.indicators import TechnicalIndicators
from visualize_features import LSTMFeaturesVisualizer
from library.data_utils import DataLoader
from datetime import datetime

def create_lstm_features(multi_tf_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create LSTM features from multi-timeframe data dengan real-time logic
    """
    
    # 1. Validate input
    required_tfs = ['1m', '5m', '30m', '1h']
    missing_tfs = [tf for tf in required_tfs if tf not in multi_tf_data]
    if missing_tfs:
        raise ValueError(f"Missing timeframes: {missing_tfs}")
    
    print(f"🔧 Creating LSTM features with real-time logic...")
    
    # 2. Helper functions (same as before)
    def get_close_price(df, tf):
        """Extract close price dari dataframe"""
        for col_name in ['close', 'Close', f'close_{tf}', f'Close_{tf}']:
            if col_name in df.columns:
                return df[col_name]
        
        for col in df.columns:
            if 'close' in col.lower():
                return df[col]
        
        raise ValueError(f"Close price column not found in {tf} data")
    
    def get_basic_ohlcv(df, tf):
        """Extract basic OHLCV data"""
        ohlcv = {}
        base_cols = ['open', 'high', 'low', 'close', 'volume']
        
        for col in base_cols:
            for variant in [col, col.title(), f'{col}_{tf}', f'{col.title()}_{tf}']:
                if variant in df.columns:
                    ohlcv[col.title()] = df[variant]
                    break
            else:
                raise ValueError(f"Column '{col}' not found in {tf} data")
        
        return pd.DataFrame(ohlcv, index=df.index)
    
    def apply_realtime_logic(df_tf, tf, close_1m_series):
        """Apply real-time close price update logic WITH proper alignment"""
        if tf == '1m':
            return df_tf  # No changes needed for 1m
        
        print(f"   🔄 Applying real-time logic untuk {tf}...")
        
        # DEBUG: Print BEFORE update
        print(f"     🔍 BEFORE update - Original {tf} data points: {len(df_tf)}")
        print(f"     🔍 Close sample: {df_tf['Close'].iloc[:3].values}")
        
        # Create new dataframe with 1-minute frequency index
        # Use the same index as close_1m_series to ensure alignment
        updated_df = pd.DataFrame(index=close_1m_series.index)
        
        # First, forward fill the original timeframe data to 1-minute intervals
        # This preserves the OHLV structure from original timeframe
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df_tf.columns:
                # Reindex to 1-minute frequency and forward fill
                updated_df[col] = df_tf[col].reindex(close_1m_series.index, method='ffill')
        
        # Now update the close price with real-time 1m close
        # This is the key: every minute gets the actual 1m close price
        updated_df['Close'] = close_1m_series.values
        
        # Update High/Low if the new close exceeds the boundaries
        # This maintains OHLV integrity
        for i in range(len(updated_df)):
            current_close = updated_df['Close'].iloc[i]
            
            # Update High if close is higher
            if pd.notna(current_close) and pd.notna(updated_df['High'].iloc[i]):
                if current_close > updated_df['High'].iloc[i]:
                    updated_df.loc[updated_df.index[i], 'High'] = current_close
            
            # Update Low if close is lower
            if pd.notna(current_close) and pd.notna(updated_df['Low'].iloc[i]):
                if current_close < updated_df['Low'].iloc[i]:
                    updated_df.loc[updated_df.index[i], 'Low'] = current_close
        
        # For any remaining NaN values (shouldn't happen if data is complete)
        for col in ['Open', 'High', 'Low', 'Volume']:
            if col in updated_df.columns:
                updated_df[col] = updated_df[col].ffill()
        
        # DEBUG: Print AFTER update
        print(f"     🔍 AFTER update - Data points: {len(updated_df)} (1-minute intervals)")
        print(f"     🔍 Close sample: {updated_df['Close'].iloc[:10].values}")
        print(f"     🔍 Unique close values in first 20 rows: {updated_df['Close'].iloc[:20].nunique()}")
        
        # Verify update worked
        print(f"     ✅ {tf}: Converted to 1-minute data with real-time close prices")
        
        return updated_df
    # 3. Get base data dari 1m timeframe
    base_1m = multi_tf_data['1m'].copy()
    features_df = pd.DataFrame(index=base_1m.index)
    
    # 4. Get 1m close prices untuk update logic
    close_1m = get_close_price(base_1m, '1m')
    print(f"   📊 Base 1m data: {len(base_1m)} candles")
    print(f"   🔍 1m close sample: {close_1m.iloc[:3].values}")
    
    # 5. Process each timeframe dengan real-time logic + DEBUG
    
    # 1H timeframe (2 features)
    print("   📊 1H indicators...")
    ohlcv_1h = get_basic_ohlcv(multi_tf_data['1h'], '1h')
    ohlcv_1h_updated = apply_realtime_logic(ohlcv_1h, '1h', close_1m)
    
    # DEBUG: Check data going into TechnicalIndicators
    print(f"     🔍 Data into TI - Close sample: {ohlcv_1h_updated['Close'].iloc[:3].values}")
    
    ti_1h = TechnicalIndicators(ohlcv_1h_updated)
    features_df['rsi_norm_1h'] = ti_1h.rsi_norm
    features_df['macd_norm_1h'] = ti_1h.macd_norm
    features_df['rsi_norm_1h'] = features_df['rsi_norm_1h'].rolling(window=60, min_periods=1).mean()
    features_df['macd_norm_1h'] = features_df['macd_norm_1h'].rolling(window=60, min_periods=1).mean()
    
    # DEBUG: Check output indicators
    print(f"     🔍 RSI output sample: {ti_1h.rsi_norm.iloc[:3].values}")
    
    # 30M timeframe (2 features)
    print("   📊 30M indicators...")
    ohlcv_30m = get_basic_ohlcv(multi_tf_data['30m'], '30m')
    ohlcv_30m_updated = apply_realtime_logic(ohlcv_30m, '30m', close_1m)
    
    ti_30m = TechnicalIndicators(ohlcv_30m_updated)
    features_df['rsi_norm_30m'] = ti_30m.rsi_norm
    features_df['bb_position_30m'] = ti_30m.bb_position
    features_df['rsi_norm_30m'] = features_df['rsi_norm_30m'].rolling(window=30, min_periods=1).mean()
    features_df['bb_position_30m'] = features_df['bb_position_30m'].rolling(window=30, min_periods=1).mean()
    
    # 5M timeframe (2 features)
    print("   📊 5M indicators...")
    ohlcv_5m = get_basic_ohlcv(multi_tf_data['5m'], '5m')
    ohlcv_5m_updated = apply_realtime_logic(ohlcv_5m, '5m', close_1m)
    
    ti_5m = TechnicalIndicators(ohlcv_5m_updated)
    features_df['macd_norm_5m'] = ti_5m.macd_norm
    features_df['adx_norm_5m'] = ti_5m.adx_norm
    features_df['macd_norm_5m'] = features_df['macd_norm_5m'].rolling(window=5, min_periods=1).mean()
    features_df['adx_norm_5m'] = features_df['adx_norm_5m'].rolling(window=5, min_periods=1).mean()
    
    # 6. Derived features (1 feature)
    print("   🔬 Derived features...")
    
    # Momentum convergence (MACD 1h+5m)
    features_df['momentum_convergence'] = (
        features_df['macd_norm_1h'] * 0.6 + 
        features_df['macd_norm_5m'] * 0.4
    )
    
    # 7. Final cleanup
    print("   🧹 Cleanup...")
    
    # Replace infinite values
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    
    # Forward fill untuk continuity (fixed deprecation warning)
    features_df = features_df.ffill().bfill().fillna(0.5)
    
    # Ensure all values are dalam 0-1 range
    features_df = features_df.clip(0, 1)
    
    # 8. Reorder columns (FIXED: 7 features total)
    final_features = [
        'rsi_norm_1h', 'macd_norm_1h',           # 2 features
        'rsi_norm_30m', 'bb_position_30m',       # 2 features
        'macd_norm_5m', 'adx_norm_5m',           # 2 features
        'momentum_convergence'                   # 1 feature
    ]
    
    features_df = features_df[final_features]
    
    # 9. Quality check dengan additional debug
    nan_count = features_df.isna().sum().sum()
    zero_count = (features_df == 0).sum().sum()
    half_count = (features_df == 0.5).sum().sum()
    
    print(f"✅ Features created: {features_df.shape[0]} samples x {features_df.shape[1]} features")
    print(f"   📊 Data quality: NaN={nan_count}, Zeros={zero_count}, Halves={half_count}")
    print(f"   📈 Range: {features_df.min().min():.3f} - {features_df.max().max():.3f}")
    
    # DEBUG: Check if features are dynamic
    print(f"\n🔍 Feature Dynamics Check:")
    for col in final_features:
        unique_vals = features_df[col].nunique()
        print(f"   {col}: {unique_vals} unique values")
    
    return features_df

# Example usage:
if __name__ == "__main__":
    
    
    # 1. Load multi-timeframe data
    loader = DataLoader()
    multi_data = loader.load_data(
        symbol='btc',
        timeframes=['1m', '5m', '30m', '1h'],
        limit=1000,
        auto_align=True,
        alignment_mode='current_only'
    )
    
    # 2. Create LSTM features
    lstm_features = create_lstm_features(multi_data)
    
    # 3. Verify output
    print(f"\n📊 Final Output:")
    print(f"Shape: {lstm_features.shape}")
    print(f"Features: {list(lstm_features.columns)}")
    print(f"Sample data:\n{lstm_features.tail(19)}")
    print("\n📊 Memulai Visualisasi...")

    visualizer = LSTMFeaturesVisualizer(multi_data, lstm_features)

    # VERIFIKASI DULU
    print("\n🔍 STEP 1: Verifikasi Features")
    visualizer.verify_features(sample_size=20)

    # Cek distribusi
    print("\n📊 STEP 2: Cek Distribusi Features")
    visualizer.plot_feature_distribution()

    # Baru visualisasi
    print("\n📈 STEP 3: Visualisasi Time Series")
    visualizer.plot_all(last_n_candles=300)
    
    # 4. Export to CSV
    try:
        # Create output directory if not exists
        output_dir = "exports"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export LSTM features
        features_filename = f"{output_dir}/lstm_features_btc_{timestamp}.csv"
        lstm_features.to_csv(features_filename)
        print(f"\n💾 LSTM Features exported: {features_filename}")
        print(f"   📏 Size: {lstm_features.shape[0]} rows x {lstm_features.shape[1]} columns")
        
        # Export raw data for reference (optional)
        for tf, df in multi_data.items():
            raw_filename = f"{output_dir}/raw_data_{tf}_btc_{timestamp}.csv"
            df.to_csv(raw_filename)
            print(f"💾 Raw {tf} data exported: {raw_filename}")
            print(f"   📏 Size: {df.shape[0]} rows x {df.shape[1]} columns")
        
        # Export summary info
        summary_filename = f"{output_dir}/summary_btc_{timestamp}.txt"
        with open(summary_filename, 'w') as f:
            f.write(f"LSTM Features Export Summary\n")
            f.write(f"=" * 30 + "\n")
            f.write(f"Export Time: {datetime.now()}\n")
            f.write(f"Symbol: BTC\n")
            f.write(f"Timeframes: {list(multi_data.keys())}\n")
            f.write(f"Data Limit: 1000 per TF\n")
            f.write(f"Alignment Mode: current_only\n\n")
            
            f.write(f"LSTM Features:\n")
            f.write(f"Shape: {lstm_features.shape}\n")
            f.write(f"Features: {list(lstm_features.columns)}\n")
            f.write(f"Date Range: {lstm_features.index[0]} to {lstm_features.index[-1]}\n\n")
            
            f.write(f"Feature Statistics:\n")
            f.write(f"Min: {lstm_features.min().min():.4f}\n")
            f.write(f"Max: {lstm_features.max().max():.4f}\n")
            f.write(f"Mean: {lstm_features.mean().mean():.4f}\n")
            f.write(f"NaN Count: {lstm_features.isna().sum().sum()}\n")
        
        print(f"💾 Summary exported: {summary_filename}")
        print(f"\n✅ All files exported successfully to '{output_dir}/' folder!")
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")