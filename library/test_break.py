import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from data_utils import DataLoader
from indicators import SupportResistance
import time

import importlib
import indicators
importlib.reload(indicators)

print(f"Indicators file location: {indicators.__file__}")


sr_test = SupportResistance()
print(f"Class methods: {dir(sr_test)}")
print(f"ATR periods: {getattr(sr_test, 'trend_atr_period', 'NOT FOUND')}")

def test_support_resistance_breaks(symbol='btc', timeframes='1m', limit=1000, 
                                 left_bars=12, right_bars=12, volume_threshold=20.0):
    """
    Test Support & Resistance breaks detection dengan visualization
    
    Args:
        symbol (str): Symbol trading ('btc', 'eth', dll)
        timeframes (str/list): Single TF ('1m') atau Multi TF (['1m', '5m', '30m'])
        limit (int): Jumlah candles per timeframe (default: 1000)
        left_bars (int): Left bars untuk pivot detection (default: 15)
        right_bars (int): Right bars untuk pivot detection (default: 15)
        volume_threshold (float): Volume threshold untuk break confirmation (default: 25.0)
    
    Returns:
        Dict: Results dengan data, levels, dan breaks untuk setiap timeframe
    """
    print(f"🧪 Testing Support & Resistance Breaks Detection")
    print(f"   Symbol: {symbol}")
    print(f"   Timeframes: {timeframes}")
    print(f"   Limit: {limit}")
    print(f"   S&R Params: left={left_bars}, right={right_bars}, vol_threshold={volume_threshold}")
    print("=" * 70)
    
    try:
        # Step 1: Load data menggunakan DataLoader
        print(f"\n📊 Loading data...")
        start_time = time.time()
        
        loader = DataLoader()
        data = loader.load_data(
            symbol=symbol,
            timeframes=timeframes,
            limit=limit,
            auto_align=True
        )
        
        end_time = time.time()
        print(f"✅ Data loaded in {end_time - start_time:.2f} seconds")
        
        # Step 2: Process Support & Resistance untuk setiap timeframe
        print(f"\n🔧 Processing Support & Resistance...")
        results = {}
        
        for tf_name, df in data.items():
            print(f"\n   📈 Processing {tf_name}...")
            
            # Prepare data untuk SupportResistance
            sr_data = _prepare_sr_data(df, tf_name)
            
            if sr_data is None:
                print(f"   ❌ {tf_name}: Cannot process - missing OHLCV data")
                continue
            
            # Initialize SupportResistance dengan custom parameters
            sr = SupportResistance(
                left_bars=left_bars,
                right_bars=right_bars,
                volume_threshold=volume_threshold
            )
            
            # Get support/resistance levels
            levels = sr.get_levels(
                high=sr_data['high'],
                low=sr_data['low'],
                volume=sr_data['volume'],
                close=sr_data['close']
            )
            
            # Detect breaks - Use fixed class method
            breaks = sr.detect_breaks(
                close=sr_data['close'],
                volume=sr_data['volume'],  # Now pass volume parameter
                levels_data=levels
            )
            
            # Get pivot points untuk visualization
            pivot_highs = sr.find_pivot_highs(sr_data['high'])
            pivot_lows = sr.find_pivot_lows(sr_data['low'])
            
            results[tf_name] = {
                'raw_data': df,
                'sr_data': sr_data,
                'levels': levels,
                'breaks': breaks,
                'pivot_highs': pivot_highs,
                'pivot_lows': pivot_lows,
                'sr_instance': sr
            }
            
            # Show summary
            print(f"   ✅ {tf_name} Results:")
            print(f"      Support: {levels['nearest_support']:.2f} (distance: {levels['distance_to_support']:.2f}%)")
            print(f"      Resistance: {levels['nearest_resistance']:.2f} (distance: {levels['distance_to_resistance']:.2f}%)")
            print(f"      Pivot Highs: {len(pivot_highs)} found")
            print(f"      Pivot Lows: {len(pivot_lows)} found")
            print(f"      Breaks Detected: {len(breaks)}")
            
            # Show break details
            if breaks:
                print(f"      🔥 Break Details:")
                for i, break_event in enumerate(breaks):
                    break_type = break_event['type']
                    timestamp = break_event['timestamp']
                    price = break_event['price']
                    confirmed = "✅ CONFIRMED" if break_event['confirmed'] else "⚠️ UNCONFIRMED"
                    print(f"         {i+1}. {break_type.upper()} at {timestamp} | Price: {price:.2f} | {confirmed}")
        
        # Step 3: Create visualization
        num_timeframes = len(results)
        
        if num_timeframes == 1:
            print(f"\n🎨 Creating single timeframe S&R visualization...")
            _create_single_tf_sr_visual(results, symbol)
        else:
            print(f"\n📊 Creating multi-timeframe S&R visualization...")
            _create_multi_tf_sr_visual(results, symbol)
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def _prepare_sr_data(df, tf_name):
    """
    Prepare data untuk SupportResistance class
    """
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    prepared_data = {}
    
    for col in required_cols:
        # Try different variations
        col_options = [
            col,  # exact
            f'{col}_{tf_name}',  # with suffix
            col.title(),  # capitalized
            f'{col.title()}_{tf_name}'  # capitalized with suffix
        ]
        
        found = False
        for col_option in col_options:
            if col_option in df.columns:
                prepared_data[col] = df[col_option]
                found = True
                break
        
        # Search dalam columns
        if not found:
            for df_col in df.columns:
                if col in df_col.lower():
                    prepared_data[col] = df[df_col]
                    found = True
                    break
        
        if not found:
            print(f"      ⚠️ Missing {col} column")
            return None
    
    # Convert to Series dengan proper index
    sr_data = {}
    for col, data in prepared_data.items():
        sr_data[col] = pd.Series(data.values, index=df.index)
    
    return sr_data

def _create_single_tf_sr_visual(results, symbol):
    """
    Single timeframe: Candlestick + Support/Resistance lines + Break points
    """
    tf_name = list(results.keys())[0]
    result = results[tf_name]
    
    raw_df = result['raw_data']
    sr_data = result['sr_data']
    levels = result['levels']
    breaks = result['breaks']
    pivot_highs = result['pivot_highs']
    pivot_lows = result['pivot_lows']
    
    print(f"   📊 Single timeframe S&R chart: {tf_name}")
    
    # Prepare candlestick data
    chart_data = pd.DataFrame({
        'Open': sr_data['open'],
        'High': sr_data['high'],
        'Low': sr_data['low'],
        'Close': sr_data['close'],
        'Volume': sr_data['volume']
    }, index=sr_data['close'].index)
    
    chart_data = chart_data.dropna()
    
    # Create figure dengan subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                  gridspec_kw={'height_ratios': [4, 1]})
    
    # Plot candlestick manually
    _plot_candlestick_manual(ax1, chart_data, f'{symbol.upper()} - {tf_name} Support & Resistance')
    
    # Plot support/resistance lines
    nearest_support = levels['nearest_support']
    nearest_resistance = levels['nearest_resistance']
    
    ax1.axhline(y=nearest_support, color='green', linestyle='-', linewidth=2, 
               alpha=0.8, label=f'Support: {nearest_support:.2f}')
    ax1.axhline(y=nearest_resistance, color='red', linestyle='-', linewidth=2,
               alpha=0.8, label=f'Resistance: {nearest_resistance:.2f}')
    
    # Plot all pivot points sebagai background
    if len(pivot_highs) > 0:
        for timestamp, price in pivot_highs.items():
            ax1.axhline(y=price, color='red', linestyle='--', linewidth=1, 
                       alpha=0.3)
    
    if len(pivot_lows) > 0:
        for timestamp, price in pivot_lows.items():
            ax1.axhline(y=price, color='green', linestyle='--', linewidth=1,
                       alpha=0.3)
    
    # Plot break points
    support_breaks = [b for b in breaks if b['type'] == 'support_break']
    resistance_breaks = [b for b in breaks if b['type'] == 'resistance_break']
    
    if support_breaks:
        for break_event in support_breaks:
            timestamp = break_event['timestamp']
            price = break_event['price']
            confirmed = break_event['confirmed']
            
            marker = 'v' if confirmed else 'x'
            color = 'darkred' if confirmed else 'orange'
            size = 100 if confirmed else 60
            
            ax1.scatter([timestamp], [price], marker=marker, s=size, 
                       color=color, edgecolors='black', linewidths=1,
                       label=f'Support Break {"(Confirmed)" if confirmed else "(Unconfirmed)"}' if break_event == support_breaks[0] else "")
    
    if resistance_breaks:
        for break_event in resistance_breaks:
            timestamp = break_event['timestamp']
            price = break_event['price']
            confirmed = break_event['confirmed']
            
            marker = '^' if confirmed else 'x'
            color = 'darkgreen' if confirmed else 'orange'
            size = 100 if confirmed else 60
            
            ax1.scatter([timestamp], [price], marker=marker, s=size,
                       color=color, edgecolors='black', linewidths=1,
                       label=f'Resistance Break {"(Confirmed)" if confirmed else "(Unconfirmed)"}' if break_event == resistance_breaks[0] else "")
    
    ax1.set_ylabel('Price')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot volume
    ax2.bar(chart_data.index, chart_data['Volume'], alpha=0.6, color='blue')
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Time')
    ax2.grid(True, alpha=0.3)
    
    # Add break count info
    total_breaks = len(breaks)
    confirmed_breaks = len([b for b in breaks if b['confirmed']])
    
    info_text = f'Total Breaks: {total_breaks}\nConfirmed: {confirmed_breaks}\nUnconfirmed: {total_breaks - confirmed_breaks}'
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
            fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print(f"   ✅ Single timeframe S&R chart created")
    print(f"   📊 Summary: {total_breaks} breaks detected ({confirmed_breaks} confirmed)")

def _create_multi_tf_sr_visual(results, symbol):
    """
    Multi timeframe: Candlestick dari TF pertama + S&R overlay dari semua TF
    """
    timeframes = list(results.keys())
    base_tf = timeframes[0]  # TF pertama untuk candlestick
    
    print(f"   📊 Multi-timeframe S&R chart")
    print(f"   📈 Candlestick base: {base_tf}")
    print(f"   🎯 S&R timeframes: {timeframes}")
    
    # Get base timeframe data untuk candlestick
    base_result = results[base_tf]
    base_sr_data = base_result['sr_data']
    
    # Prepare candlestick data
    chart_data = pd.DataFrame({
        'Open': base_sr_data['open'],
        'High': base_sr_data['high'],
        'Low': base_sr_data['low'],
        'Close': base_sr_data['close'],
        'Volume': base_sr_data['volume']
    }, index=base_sr_data['close'].index)
    
    chart_data = chart_data.dropna()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12),
                                  gridspec_kw={'height_ratios': [4, 1]})
    
    # Plot candlestick
    _plot_candlestick_manual(ax1, chart_data, 
                           f'{symbol.upper()} - Multi-TF Support & Resistance (Base: {base_tf})')
    
    # Colors untuk different timeframes
    tf_colors = {
        '1m': 'red',
        '3m': 'orange',
        '5m': 'green',
        '15m': 'blue',
        '30m': 'purple',
        '1h': 'brown',
        '4h': 'pink',
        '1d': 'gray'
    }
    fallback_colors = ['black', 'cyan', 'magenta', 'yellow']
    
    # Print color mapping
    print(f"\n   🎨 COLOR MAPPING:")
    for tf in timeframes:
        color = tf_colors.get(tf, fallback_colors[0])
        print(f"      {tf}: {color.upper()}")
    
    # Plot S&R levels dan breaks untuk each timeframe
    all_breaks = []
    
    for tf_name in timeframes:
        result = results[tf_name]
        levels = result['levels']
        breaks = result['breaks']
        color = tf_colors.get(tf_name, fallback_colors[0])
        
        # Plot nearest support/resistance
        nearest_support = levels['nearest_support']
        nearest_resistance = levels['nearest_resistance']
        
        ax1.axhline(y=nearest_support, color=color, linestyle='-', linewidth=2,
                   alpha=0.8, label=f'{tf_name} Support: {nearest_support:.2f}')
        ax1.axhline(y=nearest_resistance, color=color, linestyle='--', linewidth=2,
                   alpha=0.8, label=f'{tf_name} Resistance: {nearest_resistance:.2f}')
        
        # Plot break points
        for break_event in breaks:
            timestamp = break_event['timestamp']
            price = break_event['price']
            break_type = break_event['type']
            confirmed = break_event['confirmed']
            
            if break_type == 'support_break':
                marker = 'v' if confirmed else 'x'
                edge_color = color
            else:
                marker = '^' if confirmed else 'x'
                edge_color = color
            
            face_color = color if confirmed else 'white'
            size = 80 if confirmed else 50
            
            ax1.scatter([timestamp], [price], marker=marker, s=size,
                       color=face_color, edgecolors=edge_color, linewidths=2,
                       alpha=0.8)
        
        all_breaks.extend(breaks)
        
        # Summary per timeframe
        tf_confirmed = len([b for b in breaks if b['confirmed']])
        print(f"      {tf_name}: {len(breaks)} breaks ({tf_confirmed} confirmed)")
    
    ax1.set_ylabel('Price')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot volume
    ax2.bar(chart_data.index, chart_data['Volume'], alpha=0.6, color='blue')
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Time')
    ax2.grid(True, alpha=0.3)
    
    # Add summary info
    total_breaks = len(all_breaks)
    total_confirmed = len([b for b in all_breaks if b['confirmed']])
    
    info_text = f'Multi-TF Summary:\nTotal Breaks: {total_breaks}\nConfirmed: {total_confirmed}\nTimeframes: {len(timeframes)}'
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
            fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print(f"   ✅ Multi-timeframe S&R chart created")
    print(f"   📊 Overall: {total_breaks} breaks across all timeframes ({total_confirmed} confirmed)")

def _plot_candlestick_manual(ax, chart_data, title):
    """
    Plot candlestick chart manually pada given axis
    """
    # Sample data for performance if too many candles
    if len(chart_data) > 500:
        step = len(chart_data) // 500
        sampled_data = chart_data.iloc[::step]
    else:
        sampled_data = chart_data
    
    for i in range(len(sampled_data)):
        date = sampled_data.index[i]
        open_price = sampled_data['Open'].iloc[i]
        high_price = sampled_data['High'].iloc[i]
        low_price = sampled_data['Low'].iloc[i]
        close_price = sampled_data['Close'].iloc[i]
        
        # Color
        color = 'green' if close_price >= open_price else 'red'
        
        # Plot wick
        ax.plot([date, date], [low_price, high_price], color='black', linewidth=0.8)
        
        # Plot body
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)
        
        # Calculate width
        if len(sampled_data) > 1:
            time_diff = sampled_data.index[1] - sampled_data.index[0]
            width = time_diff * 0.8
        else:
            width = pd.Timedelta(minutes=1)
        
        ax.bar(date, body_height, bottom=body_bottom,
               color=color, alpha=0.8, width=width)
    
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

# ==================== USAGE EXAMPLES ====================

def main():
    """Main function dengan contoh usage"""
    print("🚀 TESTING SUPPORT & RESISTANCE BREAKS DETECTION")
    print("=" * 80)
    
    # Test 1: Single timeframe dengan default parameters
    print("\n🧪 Test 1: Single Timeframe - Default Parameters")
    test_support_resistance_breaks(
        symbol='btc',
        timeframes='1m',
        limit=500
    )
    
    # Test 2: Single timeframe dengan custom parameters
    print("\n🧪 Test 2: Single Timeframe - Custom Parameters")
    test_support_resistance_breaks(
        symbol='btc',
        timeframes='5m',
        limit=300,
        left_bars=10,
        right_bars=10,
        volume_threshold=20.0
    )
    
    # Test 3: Multi timeframe
    print("\n🧪 Test 3: Multi Timeframe")
    test_support_resistance_breaks(
        symbol='btc',
        timeframes=['1m', '5m', '30m'],
        limit=800,
        left_bars=15,
        right_bars=15,
        volume_threshold=25.0
    )
    
    print(f"\n✅ All S&R break tests completed!")

def automated_ab_testing():
    """
    Automated A/B testing dengan berbagai parameter combinations
    """
    print("🚀 AUTOMATED A/B TESTING")
    print("=" * 60)
    
    # Test scenarios
    test_scenarios = [
        {"name": "Ultra Aggressive", "left": 5, "right": 5, "vol": 10.0},
        {"name": "Aggressive", "left": 8, "right": 8, "vol": 15.0},
        {"name": "Balanced", "left": 12, "right": 12, "vol": 20.0},        # NEW: More realistic default
        {"name": "Conservative", "left": 18, "right": 18, "vol": 30.0},
        {"name": "Ultra Conservative", "left": 25, "right": 25, "vol": 40.0}
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🧪 Test {i}: {scenario['name']}")
        print(f"   Parameters: left={scenario['left']}, right={scenario['right']}, vol_threshold={scenario['vol']}")
        
        test_support_resistance_breaks(
            symbol='btc',
            timeframes='1m',
            limit=1500,
            left_bars=scenario['left'],
            right_bars=scenario['right'],
            volume_threshold=scenario['vol']
        )
        
        input("Press Enter untuk continue ke test berikutnya...")

# Panggil function ini untuk automated testing
if __name__ == "__main__":
    #automated_ab_testing()  # ← UBAH dari main()
    main()