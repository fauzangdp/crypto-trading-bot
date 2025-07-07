import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from data_utils import DataLoader
from indicators import TechnicalIndicators
import time

def test_indicators(symbol='btc', timeframes='1m', limit=1000, indicators='all'):
    """
    Test Technical Indicators dengan visualization yang sederhana
    
    Args:
        symbol (str): Symbol trading ('btc', 'eth', dll)
        timeframes (str/list): Single TF ('1m') atau Multi TF (['1m', '5m', '1h'])
        limit (int): Jumlah candles per timeframe (default: 1000)
        indicators (str/list): 'all' untuk semua, atau list specific ['rsi', 'macd', 'bb']
                              Available: 'stoch', 'bb', 'rsi', 'macd', 'adx', 'mfi'
    
    Returns:
        Dict: Results dengan data dan indicator values
    """
    print(f"🧪 Testing Technical Indicators")
    print(f"   Symbol: {symbol}")
    print(f"   Timeframes: {timeframes}")
    print(f"   Limit: {limit}")
    print(f"   Indicators: {indicators}")
    print("=" * 60)
    
    try:
        # Step 1: Load data menggunakan DataLoader
        print(f"\n📊 Loading data...")
        start_time = time.time()
        
        loader = DataLoader()
        data = loader.load_data(
            symbol=symbol,
            timeframes=timeframes,
            limit=limit,
            auto_align=True,
            alignment_mode='current_only'
        )
        
        end_time = time.time()
        print(f"✅ Data loaded in {end_time - start_time:.2f} seconds")
        
        # Step 2: Process indicators untuk setiap timeframe
        print(f"\n🔧 Processing indicators...")
        results = {}
        
        for tf_name, df in data.items():
            print(f"\n   📈 Processing {tf_name}...")
            
            # Prepare data untuk TechnicalIndicators
            indicator_df = _prepare_indicator_data(df, tf_name)
            
            if indicator_df is None:
                print(f"   ❌ {tf_name}: Cannot process - missing OHLCV data")
                continue
            
            # Initialize TechnicalIndicators
            ti = TechnicalIndicators(indicator_df)
            
            # Get selected indicators
            indicator_data = _get_selected_indicators(ti, indicators)
            
            results[tf_name] = {
                'raw_data': df,
                'indicator_data': indicator_df,
                'indicators': indicator_data,
                'ti_instance': ti
            }
            
            # Show indicator summary
            print(f"   ✅ {tf_name}: {len(indicator_data.columns)} indicators calculated")
            for col in indicator_data.columns:
                valid_count = indicator_data[col].notna().sum()
                coverage = (valid_count / len(indicator_data)) * 100
                print(f"      {col}: {coverage:.1f}% coverage")
        
        # Step 3: Create visualization
        num_timeframes = len(results)
        
        if num_timeframes == 1:
            print(f"\n🎨 Creating single timeframe visualization...")
            _create_single_tf_indicators_visual(results, symbol, indicators)
        else:
            print(f"\n📊 Creating multi-timeframe indicators visualization...")
            _create_multi_tf_indicators_visual(results, symbol, indicators)
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def _prepare_indicator_data(df, tf_name):
    """
    Prepare data untuk TechnicalIndicators class (need OHLCV with capital letters)
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
                prepared_data[col.title()] = df[col_option]
                found = True
                break
        
        # Search dalam columns
        if not found:
            for df_col in df.columns:
                if col in df_col.lower():
                    prepared_data[col.title()] = df[df_col]
                    found = True
                    break
        
        if not found:
            print(f"      ⚠️ Missing {col} column")
            return None
    
    indicator_df = pd.DataFrame(prepared_data, index=df.index).dropna()
    return indicator_df if len(indicator_df) > 0 else None

def _get_selected_indicators(ti, indicators):
    """
    Get selected indicators dari TechnicalIndicators instance
    """
    # Available indicators mapping
    available_indicators = {
        'stoch': ('Stochastic K', lambda: ti.stochastic_k_norm),
        'bb': ('Bollinger Band Position', lambda: ti.bb_position),
        'rsi': ('RSI', lambda: ti.rsi_norm),
        'macd': ('MACD', lambda: ti.macd_norm),
        'adx': ('ADX', lambda: ti.adx_norm),
        'mfi': ('MFI', lambda: ti.mfi_norm)
    }
    
    # Determine which indicators to calculate
    if indicators == 'all':
        selected = list(available_indicators.keys())
    elif isinstance(indicators, str):
        selected = [indicators] if indicators in available_indicators else []
    else:
        selected = [ind for ind in indicators if ind in available_indicators]
    
    # Calculate selected indicators
    indicator_data = pd.DataFrame(index=ti.data.index)
    
    for ind_key in selected:
        if ind_key in available_indicators:
            ind_name, ind_func = available_indicators[ind_key]
            try:
                indicator_data[ind_name] = ind_func()
            except Exception as e:
                print(f"      ⚠️ Failed to calculate {ind_name}: {e}")
    
    return indicator_data

def _create_single_tf_indicators_visual(results, symbol, indicators):
    """
    Single timeframe: Candlestick + Indicators subplots
    """
    tf_name = list(results.keys())[0]
    result = results[tf_name]
    
    raw_df = result['raw_data']
    indicator_df = result['indicator_data']
    indicators_data = result['indicators']
    
    print(f"   📊 Single timeframe chart: {tf_name}")
    print(f"   📈 Indicators: {list(indicators_data.columns)}")
    
    # Prepare candlestick data
    chart_data = pd.DataFrame({
        'Open': indicator_df['Open'],
        'High': indicator_df['High'],
        'Low': indicator_df['Low'],
        'Close': indicator_df['Close'],
        'Volume': indicator_df['Volume']
    }, index=indicator_df.index)
    
    # Create subplots: candlestick + indicators
    num_indicators = len(indicators_data.columns)
    fig_height = 8 + (num_indicators * 2)  # Dynamic height
    
    fig, axes = plt.subplots(2 + num_indicators, 1, figsize=(16, fig_height),
                            gridspec_kw={'height_ratios': [4, 1] + [1.5] * num_indicators})
    
    # Plot candlestick manually pada first subplot
    _plot_candlestick_manual(axes[0], chart_data, f'{symbol.upper()} - {tf_name}')
    
    # Plot volume pada second subplot
    axes[1].bar(chart_data.index, chart_data['Volume'], alpha=0.6, color='blue')
    axes[1].set_ylabel('Volume')
    axes[1].grid(True, alpha=0.3)
    
    # Plot each indicator
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
    
    for i, (ind_name, ind_data) in enumerate(indicators_data.items()):
        ax = axes[2 + i]
        color = colors[i % len(colors)]
        
        ax.plot(ind_data.index, ind_data, color=color, linewidth=2, label=ind_name)
        ax.set_ylabel(ind_name)
        ax.set_ylim(0, 1)  # All indicators normalized 0-1
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add horizontal reference lines
        ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5)
        ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.show()
    
    print(f"   ✅ Single timeframe indicators chart created")

def _create_multi_tf_indicators_visual(results, symbol, indicators):
    """
    Multi timeframe: Candlestick dari TF pertama + Indicators overlay dari semua TF
    """
    timeframes = list(results.keys())
    base_tf = timeframes[0]  # TF pertama untuk candlestick
    
    print(f"   📊 Multi-timeframe chart")
    print(f"   📈 Candlestick base: {base_tf}")
    print(f"   🎯 Indicator timeframes: {timeframes}")
    
    # Get base timeframe data untuk candlestick
    base_result = results[base_tf]
    base_indicator_df = base_result['indicator_data']
    
    # Prepare candlestick data
    chart_data = pd.DataFrame({
        'Open': base_indicator_df['Open'],
        'High': base_indicator_df['High'],
        'Low': base_indicator_df['Low'],
        'Close': base_indicator_df['Close'],
        'Volume': base_indicator_df['Volume']
    }, index=base_indicator_df.index)
    
    # Get unique indicators across all timeframes
    all_indicators = set()
    for result in results.values():
        all_indicators.update(result['indicators'].columns)
    all_indicators = sorted(list(all_indicators))
    
    # Create subplots
    num_indicators = len(all_indicators)
    fig_height = 8 + (num_indicators * 2.5)
    
    fig, axes = plt.subplots(2 + num_indicators, 1, figsize=(18, fig_height),
                            gridspec_kw={'height_ratios': [4, 1] + [2] * num_indicators})
    
    # Plot candlestick
    _plot_candlestick_manual(axes[0], chart_data, 
                           f'{symbol.upper()} - Multi-TF Indicators (Base: {base_tf})')
    
    # Plot volume
    axes[1].bar(chart_data.index, chart_data['Volume'], alpha=0.6, color='blue')
    axes[1].set_ylabel('Volume')
    axes[1].grid(True, alpha=0.3)
    
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
    
    # Plot each indicator with multi-timeframe overlay
    for i, ind_name in enumerate(all_indicators):
        ax = axes[2 + i]
        
        # Plot indicator dari setiap timeframe
        for tf_name in timeframes:
            if ind_name in results[tf_name]['indicators'].columns:
                ind_data = results[tf_name]['indicators'][ind_name]
                color = tf_colors.get(tf_name, fallback_colors[0])
                
                ax.plot(ind_data.index, ind_data, 
                       color=color, linewidth=2, alpha=0.8,
                       label=f'{tf_name} {ind_name}')
        
        ax.set_ylabel(ind_name)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Reference lines
        ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.3)
        ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5)
        ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.3)
    
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.show()
    
    print(f"   ✅ Multi-timeframe indicators chart created")

def _plot_candlestick_manual(ax, chart_data, title):
    """
    Plot candlestick chart manually pada given axis
    """
    # Sample data untuk performance (max 500 candles)
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
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)

# ==================== USAGE EXAMPLES ====================

def main():
    """Main function dengan contoh usage"""
    print("🚀 TESTING TECHNICAL INDICATORS")
    print("=" * 70)
    
    # Test 1: Single timeframe, all indicators
    print("\n🧪 Test 1: Single Timeframe - All Indicators")
    test_indicators(
        symbol='btc',
        timeframes='1m',
        limit=500,
        indicators='all'
    )
    
    # Test 2: Single timeframe, specific indicators
    print("\n🧪 Test 2: Single Timeframe - Specific Indicators")
    test_indicators(
        symbol='btc',
        timeframes='5m',
        limit=300,
        indicators=['rsi', 'macd', 'bb']
    )
    
    # Test 3: Multi timeframe, all indicators
    print("\n🧪 Test 3: Multi Timeframe - All Indicators")
    test_indicators(
        symbol='btc',
        timeframes=['1m', '5m','30m', '1h'],
        limit=500,
        indicators='all'
    )
    
    print(f"\n✅ All tests completed!")

if __name__ == "__main__":
    main()