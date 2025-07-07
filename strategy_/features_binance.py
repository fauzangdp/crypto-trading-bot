import pandas as pd
import numpy as np
from typing import Dict,Optional
import sys
import os
import warnings
import asyncio
import time 

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from library.indicators import TechnicalIndicators
from library.binance_connector import generate_trading_features
from datetime import datetime

from binance.client import Client

warnings.filterwarnings("ignore", category=DeprecationWarning, module="binance.helpers")

if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def create_lstm_features(features_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create LSTM features from aligned trading data
    
    Args:
        features_data: DataFrame from generate_trading_features() with columns:
                      open_1m, high_1m, low_1m, close_1m, volume_1m,
                      open_5m, high_5m, low_5m, close_5m, volume_5m, etc.
    
    Returns:
        pd.DataFrame: 7 normalized features for LSTM model
    """
    
    print(f"🔧 Creating LSTM features...")
    
    # Helper function to extract OHLCV for specific timeframe
    def extract_ohlcv(data: pd.DataFrame, tf: str) -> pd.DataFrame:
        """Extract OHLCV columns for specific timeframe"""
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Build column names
        if tf == '1m':
            cols = {col: f'{col}_{tf}' for col in ohlcv_cols}
        else:
            cols = {col: f'{col}_{tf}' for col in ohlcv_cols}
        
        # Extract and rename to standard format
        ohlcv_df = pd.DataFrame(index=data.index)
        for std_name, col_name in cols.items():
            if col_name in data.columns:
                ohlcv_df[std_name.title()] = data[col_name]
            else:
                raise ValueError(f"Column {col_name} not found in data")
        
        return ohlcv_df
    
    # Initialize features dataframe
    features_df = pd.DataFrame(index=features_data.index)
    
    # Process each timeframe
    print("   📊 Calculating indicators...")
    
    # 1H timeframe (2 features)
    ohlcv_1h = extract_ohlcv(features_data, '1h')
    ti_1h = TechnicalIndicators(ohlcv_1h)
    features_df['rsi_norm_1h'] = ti_1h.rsi_norm
    features_df['macd_norm_1h'] = ti_1h.macd_norm
    features_df['rsi_norm_1h'] = features_df['rsi_norm_1h'].rolling(window=60, min_periods=1).mean()
    features_df['macd_norm_1h'] = features_df['macd_norm_1h'].rolling(window=60, min_periods=1).mean()
    
    
    # 30M timeframe (2 features)
    ohlcv_30m = extract_ohlcv(features_data, '30m')
    ti_30m = TechnicalIndicators(ohlcv_30m)
    features_df['rsi_norm_30m'] = ti_30m.rsi_norm
    features_df['bb_position_30m'] = ti_30m.bb_position
    features_df['rsi_norm_30m'] = features_df['rsi_norm_30m'].rolling(window=30, min_periods=1).mean()
    features_df['bb_position_30m'] = features_df['bb_position_30m'].rolling(window=30, min_periods=1).mean()
    
    # 5M timeframe (2 features)
    ohlcv_5m = extract_ohlcv(features_data, '5m')
    ti_5m = TechnicalIndicators(ohlcv_5m)
    features_df['macd_norm_5m'] = ti_5m.macd_norm
    features_df['adx_norm_5m'] = ti_5m.adx_norm
    features_df['macd_norm_5m'] = features_df['macd_norm_5m'].rolling(window=5, min_periods=1).mean()
    features_df['adx_norm_5m'] = features_df['adx_norm_5m'].rolling(window=5, min_periods=1).mean()
    
    # Derived features (1 feature)
    features_df['momentum_convergence'] = (
        features_df['macd_norm_1h'] * 0.6 + 
        features_df['macd_norm_5m'] * 0.4
    )
    
    # Final cleanup
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.ffill().bfill().fillna(0.5)
    features_df = features_df.clip(0, 1)
    
    # Reorder columns (maintain same order)
    final_features = [
        'rsi_norm_1h', 'macd_norm_1h',
        'rsi_norm_30m', 'bb_position_30m',
        'macd_norm_5m', 'adx_norm_5m',
        'momentum_convergence'
    ]
    
    features_df = features_df[final_features]
    
    # Quality check
    print(f"✅ Features created: {features_df.shape[0]} samples x {features_df.shape[1]} features")
    print(f"   📈 Range: {features_df.min().min():.3f} - {features_df.max().max():.3f}")
    
    return features_df


# Convenience function for live trading
def get_live_lstm_features(symbol: str = 'ETHUSDT', limit: int = 200, client: Optional[Client] = None) -> pd.DataFrame:
    """
    Get LSTM features from live Binance data - OPTIMIZED WITH SHARED CONNECTION
    
    Args:
        symbol: Trading symbol
        limit: Number of candles to fetch
        client: Optional shared Binance client (untuk avoid multiple connections)
    
    Returns:
        pd.DataFrame: 7 LSTM features ready for model input
    """
    
    # 🔧 USE SHARED CLIENT IF PROVIDED
    if client is not None:
        # Use the shared client - don't create new connection
        trading_features = generate_trading_features_with_client(
            symbol=symbol,
            client=client,  # Pass shared client
            data_limit=limit
        )
    else:
        # Fallback to original method (for backward compatibility)
        trading_features = generate_trading_features(
            symbol=symbol,
            api_key="EduyybaFGjUpSkR7q2J0HwHjHF6dB8TB5klAAUX8Ukum2Yz1jR2J8osZVXz9kxZC",
            api_secret="QmAxhDG4QYxdrif38WyQ6uvGLv5OZvlGPIRBzdtFWry7adtRNzGFY8HlLkOSLOyY",
            data_limit=limit
        )
    
    # Create LSTM features
    lstm_features = create_lstm_features(trading_features)
    
    return lstm_features

def generate_trading_features_with_client(symbol: str, client: Client, data_limit: int = 200) -> pd.DataFrame:
    """Generate trading features dengan shared client - COMPLETE VERSION"""
    try:
        print(f"📊 Generating features for {symbol}...")
        
        # Initialize results dict untuk multi-timeframe
        timeframes = ['1m', '5m', '30m', '1h']
        all_data = {}
        
        # Get data untuk setiap timeframe
        for tf in timeframes:
            try:
                klines = client.futures_klines(
                    symbol=symbol,
                    interval=tf,
                    limit=data_limit
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Convert types
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Rename columns dengan timeframe suffix
                df_renamed = pd.DataFrame(index=df.index)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df_renamed[f'{col}_{tf}'] = df[col]
                
                all_data[tf] = df_renamed
                
                # Small delay untuk rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error getting {tf} data for {symbol}: {e}")
                continue
        
        if not all_data:
            raise Exception("No data retrieved for any timeframe")
        
        # Merge all timeframes ke common timestamps (gunakan 5m sebagai base)
        if '5m' in all_data:
            base_df = all_data['5m']
        else:
            base_df = list(all_data.values())[0]
        
        # Merge semua timeframes
        merged_df = base_df.copy()
        
        for tf, df in all_data.items():
            if tf == '5m':
                continue  # Skip base
            
            # Reindex untuk align dengan base timestamps
            df_reindexed = df.reindex(base_df.index, method='ffill')
            
            # Merge columns
            for col in df.columns:
                merged_df[col] = df_reindexed[col]
        
        # Clean up
        merged_df = merged_df.ffill().bfill().dropna()
        
        print(f"✅ Created features: {merged_df.shape[1]} columns, {merged_df.shape[0]} rows")
        print(f"   📈 Range: {merged_df.min().min():.3f} - {merged_df.max().max():.3f}")
        
        return merged_df
        
    except Exception as e:
        print(f"Error generating features for {symbol}: {e}")
        # Fallback to original method
        return generate_trading_features(
            symbol=symbol,
            api_key="EduyybaFGjUpSkR7q2J0HwHjHF6dB8TB5klAAUX8Ukum2Yz1jR2J8osZVXz9kxZC",
            api_secret="QmAxhDG4QYxdrif38WyQ6uvGLv5OZvlGPIRBzdtFWry7adtRNzGFY8HlLkOSLOyY",
            data_limit=data_limit
        )


def visualize_lstm_features(symbol: str = 'ETHUSDT', limit: int = 200):
    """
    Visualize candlestick 1m and LSTM features
    
    Args:
        symbol: Trading symbol
        limit: Number of candles to display
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    
    print(f"📊 Generating visualization for {symbol}...")
    
    # Get live LSTM features
    lstm_features = get_live_lstm_features(symbol, limit)
    
    # Get raw trading data for candlestick
    trading_data = generate_trading_features(
        symbol=symbol,
        api_key="EduyybaFGjUpSkR7q2J0HwHjHF6dB8TB5klAAUX8Ukum2Yz1jR2J8osZVXz9kxZC",
        api_secret="QmAxhDG4QYxdrif38WyQ6uvGLv5OZvlGPIRBzdtFWry7adtRNzGFY8HlLkOSLOyY",
        data_limit=limit
    )

    lstm_features = create_lstm_features(trading_data)
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    
    # Use last 100 candles for clarity
    display_limit = min(100, len(lstm_features))
    
    # Subplot 1: Candlestick 1m
    ax1 = plt.subplot(2, 1, 1)
    
    # Plot candlestick
    for i in range(len(trading_data) - display_limit, len(trading_data)):
        idx = trading_data.index[i]
        o = trading_data['open_1m'].iloc[i]
        h = trading_data['high_1m'].iloc[i]
        l = trading_data['low_1m'].iloc[i]
        c = trading_data['close_1m'].iloc[i]
        
        color = 'green' if c >= o else 'red'
        
        # Draw candle body
        ax1.add_patch(Rectangle((mdates.date2num(idx) - 0.0003, min(o, c)),
                               0.0006, abs(c - o),
                               facecolor=color, edgecolor=color, alpha=0.8))
        # Draw wick
        ax1.plot([mdates.date2num(idx), mdates.date2num(idx)], [l, h],
                color=color, linewidth=1)
    
    ax1.set_title(f'{symbol} - 1 Minute Candlestick', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Subplot 2: LSTM Features
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    
    # Plot each feature
    features_to_plot = lstm_features.tail(display_limit)
    time_index = features_to_plot.index
    
    # Define colors for each feature
    colors = ['blue', 'cyan', 'green', 'lime', 'red', 'orange', 'purple']
    
    for i, (col, color) in enumerate(zip(features_to_plot.columns, colors)):
        ax2.plot(time_index, features_to_plot[col], 
                label=col, color=color, linewidth=1.5, alpha=0.8)
    
    # Add horizontal lines at 0 and 1
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    
    ax2.set_title('LSTM Features (Normalized 0-1)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Feature Value', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', ncol=4, fontsize=9)
    ax2.set_ylim(-0.1, 1.1)
    
    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Add info text
    fig.text(0.02, 0.02, 
             f'Data points: {len(lstm_features)} | '
             f'Features: {len(lstm_features.columns)} | '
             f'Time range: {lstm_features.index[0].strftime("%Y-%m-%d %H:%M")} - '
             f'{lstm_features.index[-1].strftime("%H:%M")}',
             fontsize=10, alpha=0.7)
    
    plt.show()
    
    # Print verification info
    print(f"\n✅ Verification:")
    print(f"1. Total features: {len(lstm_features.columns)} - {list(lstm_features.columns)}")
    print(f"2. Data alignment: {len(trading_data)} candles → {len(lstm_features)} feature rows")
    print(f"3. Latest timestamp: {lstm_features.index[-1]}")
    print(f"4. Feature ranges:")
    for col in lstm_features.columns:
        print(f"   - {col}: [{lstm_features[col].min():.3f}, {lstm_features[col].max():.3f}]")
    
    return lstm_features


# Quick test function
def test_lstm_features(symbol: str = 'ETHUSDT'):
    """Quick test to verify LSTM features generation"""
    print(f"🧪 Testing LSTM features for {symbol}...")
    
    # Get features
    features = get_live_lstm_features(symbol, limit=50)
    
    # Basic checks
    print(f"\n✅ Shape: {features.shape}")
    print(f"✅ All values in [0,1]?: {(features.min().min() >= 0) and (features.max().max() <= 1)}")
    print(f"✅ Any NaN?: {features.isna().sum().sum() == 0}")
    print(f"✅ Sample:\n{features.tail(3)}")

    print("\n📊 Last 5 values for each feature:")
    for col in features.columns:
        last_5 = features[col].tail(5).values
        print(f"{col}: {last_5}")
    
    # Visualize
    visualize_lstm_features(symbol, limit=200)

    
    return features




# Example usage
if __name__ == "__main__":
    # Method 1: From pre-generated data
    from library.binance_connector import generate_trading_features
    
    # Get trading data
    #trading_data = generate_trading_features(
    #    symbol='BTCUSDT',
    #    api_key="your_api_key",
    #    api_secret="your_api_secret",
    #    data_limit=500
    #)
    
    # Create LSTM features
    #lstm_features = create_lstm_features(trading_data)
    
    #print(f"\n📊 Final Output:")
    #print(f"Shape: {lstm_features.shape}")
    #print(f"Features: {list(lstm_features.columns)}")
    #print(f"Sample:\n{lstm_features.tail(5)}")
    
    # Method 2: Quick live features
    #live_features = get_live_lstm_features('ETHUSDT', limit=300)
    #print(f"\n📊 Live Features: {live_features.shape}")

    
    
    # Export if needed
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #filename = f"lstm_features_{timestamp}.csv"
    #lstm_features.to_csv(filename)
    #print(f"\n💾 Features exported to {filename}")


    test_lstm_features('BTCUSDT')