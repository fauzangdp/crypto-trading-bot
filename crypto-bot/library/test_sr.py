# ==================== SUPPORT & RESISTANCE TESTING ====================

# Import dari kedua file
from data_utils import DataLoader
from indicators import SupportResistance
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np

def test_support_resistance(symbol='btc', timeframes='1m', limit=1000, 
                           left_bars=15, right_bars=15, volume_threshold=25.0):
    """
    Test Support & Resistance dengan parameter yang sama seperti load_data
    
    Args:
        symbol (str): Symbol trading
        timeframes (str/list): Single atau multiple timeframes
        limit (int): Jumlah candles
        left_bars (int): Left bars untuk pivot detection
        right_bars (int): Right bars untuk pivot detection
        volume_threshold (float): Volume threshold untuk break detection
    """
    print(f"🎯 Testing Support & Resistance")
    print(f"   Symbol: {symbol}")
    print(f"   Timeframes: {timeframes}")
    print(f"   Limit: {limit}")
    print(f"   S&R Params: left={left_bars}, right={right_bars}, vol_thresh={volume_threshold}")
    print("=" * 60)
    
    # Step 1: Load data menggunakan DataLoader
    print(f"\n📊 Step 1: Loading data...")
    loader = DataLoader()
    data = loader.load_data(symbol=symbol, timeframes=timeframes, limit=limit, alignment_mode='current_only')
    
    # Step 2: Initialize Support & Resistance
    print(f"\n🔍 Step 2: Analyzing Support & Resistance...")
    sr = SupportResistance(left_bars=left_bars, right_bars=right_bars, volume_threshold=volume_threshold)
    
    # Step 3: Get S&R levels untuk setiap timeframe
    sr_results = {}
    
    for tf_name, df in data.items():
        print(f"\n   📈 Analyzing {tf_name}...")
        
        # Prepare columns (handle multi-timeframe suffix)
        if f'high_{tf_name}' in df.columns:
            # Multi-timeframe data
            high_col = f'high_{tf_name}'
            low_col = f'low_{tf_name}'
            close_col = f'close_{tf_name}'
            volume_col = f'volume_{tf_name}'
        else:
            # Single timeframe data
            high_col = 'high'
            low_col = 'low'
            close_col = 'close'
            volume_col = 'volume'
        
        # Get S&R levels
        levels = sr.get_levels(
            high=df[high_col],
            low=df[low_col], 
            volume=df[volume_col],
            close=df[close_col]
        )
        
        sr_results[tf_name] = {
            'levels': levels,
            'data': df,
            'columns': {
                'high': high_col,
                'low': low_col,
                'close': close_col,
                'volume': volume_col
            }
        }
        
        # Show results
        print(f"      ✅ {tf_name} S&R Results:")
        print(f"         Support: {levels['nearest_support']:.2f} (distance: {levels['distance_to_support']:.2f}%)")
        print(f"         Resistance: {levels['nearest_resistance']:.2f} (distance: {levels['distance_to_resistance']:.2f}%)")
    
    # Step 4: Create visualization
    print(f"\n🎨 Step 3: Creating visualization...")
    
    if len(data) == 1:
        # Single timeframe visualization
        visualize_single_timeframe(sr_results, symbol)
    else:
        # Multi-timeframe visualization
        visualize_multi_timeframe(sr_results, symbol)
    
    return sr_results

def visualize_single_timeframe(sr_results, symbol):
    """
    Visualisasi untuk single timeframe dengan candlestick + S&R lines
    """
    tf_name = list(sr_results.keys())[0]
    result = sr_results[tf_name]
    
    df = result['data']
    levels = result['levels']
    cols = result['columns']
    
    print(f"      📊 Single timeframe visualization: {tf_name}")
    
    # Prepare OHLCV data untuk mplfinance
    chart_data = pd.DataFrame({
        'open': df[cols['high']],  # Using high as open for simplicity
        'high': df[cols['high']],
        'low': df[cols['low']],
        'close': df[cols['close']],
        'volume': df[cols['volume']]
    }, index=df.index)
    
    # Create horizontal lines untuk S&R
    support_line = levels['nearest_support']
    resistance_line = levels['nearest_resistance']
    
    # Plot dengan mplfinance
    fig, axes = mpf.plot(
        chart_data,
        type='candle',
        style='charles',
        title=f'{symbol.upper()} - {tf_name} with Support & Resistance',
        ylabel='Price',
        volume=True,
        figsize=(15, 10),
        returnfig=True,
        hlines=dict(hlines=[support_line, resistance_line], 
                   colors=['green', 'red'], 
                   linestyle='--', 
                   linewidths=2)
    )
    
    # Add legend untuk S&R lines
    axes[0].axhline(y=support_line, color='green', linestyle='--', linewidth=2, 
                   label=f'Support: {support_line:.2f}')
    axes[0].axhline(y=resistance_line, color='red', linestyle='--', linewidth=2, 
                   label=f'Resistance: {resistance_line:.2f}')
    axes[0].legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"         ✅ Single TF chart created with S&R lines")

def visualize_multi_timeframe(sr_results, symbol):
    """
    Visualisasi untuk multi-timeframe dengan candlestick + multiple S&R lines
    """
    print(f"      📊 Multi-timeframe visualization: {list(sr_results.keys())}")
    
    # Get base timeframe (first one) untuk candlestick
    base_tf = list(sr_results.keys())[0]
    base_result = sr_results[base_tf]
    base_df = base_result['data']
    base_cols = base_result['columns']
    
    # Prepare candlestick data
    chart_data = pd.DataFrame({
        'open': base_df[base_cols['high']],  # Using high as open
        'high': base_df[base_cols['high']],
        'low': base_df[base_cols['low']],
        'close': base_df[base_cols['close']],
        'volume': base_df[base_cols['volume']]
    }, index=base_df.index)
    
    # Create figure manually untuk more control
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), 
                                  gridspec_kw={'height_ratios': [4, 1]})
    
    # Plot candlestick using matplotlib
    for i in range(len(chart_data)):
        date = chart_data.index[i]
        open_price = chart_data['open'].iloc[i]
        high_price = chart_data['high'].iloc[i]
        low_price = chart_data['low'].iloc[i]
        close_price = chart_data['close'].iloc[i]
        
        # Color candle
        color = 'green' if close_price >= open_price else 'red'
        
        # Plot wick
        ax1.plot([date, date], [low_price, high_price], color='black', linewidth=0.5)
        
        # Plot body (simplified)
        body_height = abs(close_price - open_price)
        if i % 5 == 0:  # Show every 5th candle untuk clarity
            ax1.bar(date, body_height, bottom=min(open_price, close_price), 
                   color=color, alpha=0.7, width=pd.Timedelta(minutes=1))
    
    # Colors untuk different timeframes
    colors = ['green', 'red', 'blue', 'orange', 'purple']
    
    # Plot S&R lines untuk each timeframe
    for i, (tf_name, result) in enumerate(sr_results.items()):
        levels = result['levels']
        color = colors[i % len(colors)]
        
        support = levels['nearest_support']
        resistance = levels['nearest_resistance']
        
        # Plot horizontal lines
        ax1.axhline(y=support, color=color, linestyle='--', linewidth=2, alpha=0.8,
                   label=f'{tf_name} Support: {support:.2f}')
        ax1.axhline(y=resistance, color=color, linestyle='-', linewidth=2, alpha=0.8,
                   label=f'{tf_name} Resistance: {resistance:.2f}')
    
    ax1.set_title(f'{symbol.upper()} - Multi-Timeframe Support & Resistance')
    ax1.set_ylabel('Price')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot volume
    ax2.bar(chart_data.index, chart_data['volume'], alpha=0.6, color='blue')
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Time')
    
    plt.tight_layout()
    plt.show()
    
    print(f"         ✅ Multi-TF chart created with overlaid S&R lines")

def test_support_resistance_comprehensive():
    """
    Comprehensive testing dengan berbagai parameter
    """
    print("🚀 COMPREHENSIVE SUPPORT & RESISTANCE TESTING")
    print("=" * 80)
    
    # Test cases
    test_cases = [
        {
            'desc': 'BTC 1m - Default Parameters',
            'params': {'symbol': 'btc', 'timeframes': '1m', 'limit': 500}
        },
        {
            'desc': 'BTC 5m - Sensitive S&R',
            'params': {'symbol': 'btc', 'timeframes': '5m', 'limit': 300, 'left_bars': 10, 'right_bars': 10}
        },
        {
            'desc': 'BTC Multi-TF - Conservative S&R',
            'params': {'symbol': 'btc', 'timeframes': ['1m', '5m'], 'limit': 1000, 'left_bars': 20, 'right_bars': 20}
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['desc']}")
        print("-" * 50)
        
        try:
            result = test_support_resistance(**test_case['params'])
            results.append(result)
            print(f"   ✅ Test {i} completed successfully")
            
        except Exception as e:
            print(f"   ❌ Test {i} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 COMPREHENSIVE TESTING COMPLETED!")
    print(f"✅ {len(results)}/{len(test_cases)} tests successful")
    
    return results

def quick_test():
    """Quick test untuk debugging"""
    print("⚡ QUICK SUPPORT & RESISTANCE TEST")
    print("=" * 40)
    
    try:
        # Simple single timeframe test
        result = test_support_resistance(
            symbol='btc', 
            timeframes='1m', 
            limit=200,
            left_bars=10,
            right_bars=10
        )
        
        print(f"\n✅ Quick test completed!")
        return result
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function untuk testing"""
    print("🎯 CHOOSE TESTING MODE:")
    print("1️⃣ quick_test() - Simple single timeframe")
    print("2️⃣ test_support_resistance_comprehensive() - Full testing")
    print("3️⃣ Custom test")
    
    # Uncomment the test you want:
    
    # Option 1: Quick test
    #quick_test()
    
    # Option 2: Comprehensive test  
    #test_support_resistance_comprehensive()
    
    # Option 3: Custom test
    test_support_resistance(symbol='btc', timeframes=['5m','30m','1h'], limit=150)

if __name__ == "__main__":
    main()