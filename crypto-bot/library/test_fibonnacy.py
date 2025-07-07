

# ==================== FIBONACCI TREND TEST - SIMPLIFIED ====================

# Import dari kedua file
from data_utils import DataLoader
from indicators import FibonacciTrend
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np

def test_fibonacci_trend(symbol='btc', timeframes='1m', limit=1000):
    """
    Simple Fibonacci Trend Test - Parameter fibonacci sudah fix di class
    
    Args:
        symbol (str): Symbol trading
        timeframes (str/list): Single atau multiple timeframes
        limit (int): Jumlah candles
    """
    print(f"🎯 Testing Fibonacci Trend")
    print(f"   Symbol: {symbol}")
    print(f"   Timeframes: {timeframes}")
    print(f"   Limit: {limit}")
    print("=" * 60)
    
    # Step 1: Load data menggunakan DataLoader
    print(f"\n📊 Loading data...")
    loader = DataLoader()
    data = loader.load_data(symbol=symbol, timeframes=timeframes, limit=limit, alignment_mode='current_only')
    
    # Step 2: Analyze Fibonacci untuk setiap timeframe
    print(f"\n🔍 Analyzing Fibonacci Trend...")
    
    fib_results = {}
    
    for tf_name, df in data.items():
        print(f"\n   📈 Analyzing {tf_name}...")
        
        # SIMPLIFIED: No suffix handling needed with current_only mode!
        # Just use the DataFrame as-is
        fib_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # Initialize FibonacciTrend dengan default parameters
        fib_debug = FibonacciTrend()  # Parameter sudah optimal di class
        
        # Analyze fibonacci trend
        fib_analysis = fib_debug.analyze_fibonacci_trend_debug(fib_df)
        
        fib_results[tf_name] = {
            'analysis': fib_analysis,
            'data': df
        }
        
        # Show results
        current_fib = fib_analysis['current_fib']
        if current_fib['active']:
            trend_name = "UPTREND" if current_fib['trend_direction'] == 1 else "DOWNTREND"
            fib_range = current_fib['fib_high'] - current_fib['fib_low']
            
            print(f"      ✅ {tf_name} Fibonacci Results:")
            print(f"         Trend: {trend_name}")
            print(f"         Swing High: {current_fib['fib_high']:.2f} (bar {current_fib['fib_high_idx']})")
            print(f"         Swing Low: {current_fib['fib_low']:.2f} (bar {current_fib['fib_low_idx']})")
            print(f"         Range: {fib_range:.2f}")
            
            if current_fib['levels']:
                print(f"         🎯 Key Fibonacci Levels:")
                for level_name in ['0', '236', '382', '500', '618', '786', '100']:
                    if level_name in current_fib['levels']:
                        price = current_fib['levels'][level_name]
                        percentage = {'0': '0%', '236': '23.6%', '382': '38.2%', '500': '50%', '618': '61.8%', '786': '78.6%', '100': '100%'}[level_name]
                        print(f"           {percentage}: {price:.2f}")
        else:
            print(f"      ⚠️ {tf_name}: No active fibonacci trend detected")
    
    # Step 3: Create visualization
    print(f"\n🎨 Creating visualization...")
    
    if len(data) == 1:
        # Single timeframe visualization
        visualize_fibonacci_single(fib_results, symbol)
    else:
        # Multi-timeframe visualization  
        visualize_fibonacci_multi(fib_results, symbol)
    
    return fib_results

# Replace these two functions in your test_fibonacci.py file

def visualize_fibonacci_single(fib_results, symbol):
    """
    FIXED: Simple single timeframe visualization dengan safer index handling
    """
    tf_name = list(fib_results.keys())[0]
    result = fib_results[tf_name]
    
    df = result['data']
    analysis = result['analysis']
    
    print(f"      📊 Creating {tf_name} Fibonacci chart...")
    
    # SIMPLIFIED: Direct column access
    chart_data = pd.DataFrame({
        'Open': df['open'],
        'High': df['high'],
        'Low': df['low'],
        'Close': df['close'],
        'Volume': df['volume']
    }, index=df.index)
    
    # Drop any NaN rows
    chart_data = chart_data.dropna()
    
    current_fib = analysis['current_fib']
    
    if not current_fib['active'] or not current_fib['levels']:
        print(f"         ⚠️ No active fibonacci levels to plot")
        
        # Still show basic chart without fibonacci
        mpf.plot(chart_data, type='candle', style='charles',
                title=f'{symbol.upper()} - {tf_name} (No Fibonacci)',
                ylabel='Price', volume=True, figsize=(15, 8))
        return
    
    # Get fibonacci levels untuk horizontal lines
    fib_levels = []
    fib_colors = []
    
    level_colors = {
        '0': 'blue',
        '236': 'green', 
        '382': 'orange',
        '500': 'red',
        '618': 'purple',
        '786': 'brown',
        '100': 'blue'
    }
    
    # Add fibonacci levels in order
    for level_name in ['0', '236', '382', '500', '618', '786', '100']:
        if level_name in current_fib['levels']:
            price = current_fib['levels'][level_name]
            fib_levels.append(price)
            fib_colors.append(level_colors[level_name])
    
    print(f"         📏 Plotting {len(fib_levels)} fibonacci levels")
    
    # Get trend info
    trend_name = "UPTREND" if current_fib['trend_direction'] == 1 else "DOWNTREND"
    fib_range = current_fib['fib_high'] - current_fib['fib_low']
    
    # FIXED: Safer supertrend handling
    supertrend_full = analysis['supertrend']
    
    # Ensure supertrend matches chart_data index
    if len(supertrend_full) > 0:
        # Try direct loc first
        try:
            supertrend = supertrend_full.loc[chart_data.index]
        except KeyError:
            # If fails, use reindex
            supertrend = supertrend_full.reindex(chart_data.index, method='ffill')
            
            # If still has issues, use intersection
            if supertrend.isna().all():
                common_idx = chart_data.index.intersection(supertrend_full.index)
                if len(common_idx) > 0:
                    supertrend = pd.Series(index=chart_data.index, dtype=float)
                    supertrend.loc[common_idx] = supertrend_full.loc[common_idx]
                    supertrend = supertrend.fillna(method='ffill')
                else:
                    print(f"         ⚠️ Could not align supertrend data")
                    supertrend = None
    else:
        supertrend = None
    
    # Build plot kwargs
    kwargs = {
        'type': 'candle',
        'style': 'charles',
        'title': f'{symbol.upper()} - {tf_name} Fibonacci Analysis\n{trend_name} | Range: {fib_range:.1f} | {current_fib["fib_low"]:.1f} - {current_fib["fib_high"]:.1f}',
        'ylabel': 'Price',
        'volume': True,
        'figsize': (16, 10)
    }
    
    # Add supertrend if available
    if supertrend is not None and not supertrend.isna().all():
        addplot = mpf.make_addplot(supertrend, color='black', width=2)
        kwargs['addplot'] = addplot
    
    # Add fibonacci levels
    if fib_levels:
        kwargs['hlines'] = dict(
            hlines=fib_levels, 
            colors=fib_colors, 
            linestyle='-', 
            linewidths=1.5,
            alpha=0.8
        )
    
    # Plot
    mpf.plot(chart_data, **kwargs)
    
    print(f"         ✅ {tf_name} Fibonacci chart created successfully")


def visualize_fibonacci_multi(fib_results, symbol):
    """
    FIXED: Multi-timeframe visualization dengan proper index handling
    """
    timeframes = list(fib_results.keys())
    
    # Always use shortest timeframe untuk candlestick base
    base_tf = get_shortest_timeframe(timeframes)
    print(f"      📊 Multi-timeframe chart: {timeframes}")
    print(f"      📈 Candlestick base: {base_tf} (shortest timeframe)")
    
    base_result = fib_results[base_tf]
    base_df = base_result['data']
    
    # SIMPLIFIED: Direct column access
    chart_data = pd.DataFrame({
        'Open': base_df['open'],
        'High': base_df['high'],
        'Low': base_df['low'],
        'Close': base_df['close'],
        'Volume': base_df['volume']
    }, index=base_df.index)
    
    chart_data = chart_data.dropna()
    
    # Color mapping untuk setiap timeframe
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
    
    # Fallback colors if timeframe not in mapping
    fallback_colors = ['black', 'cyan', 'magenta', 'yellow']
    
    # Print COLOR MAPPING untuk user
    print(f"\n      🎨 COLOR MAPPING:")
    for tf_name in timeframes:
        color = tf_colors.get(tf_name, fallback_colors[len([t for t in timeframes if t not in tf_colors]) % len(fallback_colors)])
        print(f"         {tf_name}: {color.upper()} lines")
    print()
    
    # Collect fibonacci levels dengan clear mapping
    all_hlines = []
    all_colors = []
    addplots = []
    
    for tf_name in timeframes:
        result = fib_results[tf_name]
        analysis = result['analysis']
        current_fib = analysis['current_fib']
        
        # Get color untuk timeframe ini
        color_base = tf_colors.get(tf_name, fallback_colors[0])
        
        # FIXED: Handle index mismatch untuk supertrend
        supertrend_full = analysis['supertrend']
        
        if tf_name == base_tf:
            # Base timeframe - direct match
            supertrend = supertrend_full.loc[chart_data.index]
        else:
            # Other timeframes - need to reindex or interpolate
            # Option 1: Reindex dengan forward fill
            supertrend = supertrend_full.reindex(chart_data.index, method='ffill')
            
            # Option 2: If reindex fails, use intersection
            if supertrend.isna().all():
                # Find common timestamps
                common_idx = chart_data.index.intersection(supertrend_full.index)
                if len(common_idx) > 0:
                    # Use only common timestamps
                    supertrend = pd.Series(index=chart_data.index, dtype=float)
                    supertrend.loc[common_idx] = supertrend_full.loc[common_idx]
                    # Forward fill the gaps
                    supertrend = supertrend.fillna(method='ffill')
                else:
                    # No common timestamps - interpolate
                    print(f"         ⚠️ {tf_name}: No common timestamps, using interpolation")
                    # Create empty series and fill with nearest values
                    supertrend = pd.Series(index=chart_data.index, dtype=float)
                    for idx in chart_data.index:
                        # Find nearest timestamp in supertrend_full
                        if len(supertrend_full) > 0:
                            time_diffs = abs(supertrend_full.index - idx)
                            nearest_idx = time_diffs.argmin()
                            supertrend.loc[idx] = supertrend_full.iloc[nearest_idx]
        
        # Check if we have valid supertrend data
        if not supertrend.isna().all():
            addplots.append(mpf.make_addplot(supertrend, color=color_base, width=2, alpha=0.8))
        else:
            print(f"         ⚠️ {tf_name}: Could not align supertrend data")
        
        # Add fibonacci levels
        if current_fib['active'] and current_fib['levels']:
            trend_name = "UP" if current_fib['trend_direction'] == 1 else "DOWN"
            fib_range = current_fib['fib_high'] - current_fib['fib_low']
            
            print(f"         {tf_name} ({color_base.upper()}): {trend_name}TREND")
            print(f"           Range: {fib_range:.1f} | {current_fib['fib_low']:.1f} - {current_fib['fib_high']:.1f}")
            
            # Add key fibonacci levels dengan price info
            key_levels = ['236', '382', '500', '618', '786']
            for level_name in key_levels:
                if level_name in current_fib['levels']:
                    price = current_fib['levels'][level_name]
                    all_hlines.append(price)
                    all_colors.append(color_base)
                    percentage = {'236': '23.6%', '382': '38.2%', '500': '50%', '618': '61.8%', '786': '78.6%'}[level_name]
                    print(f"           {percentage}: {price:.2f}")
            print()
        else:
            print(f"         {tf_name} ({color_base.upper()}): No fibonacci levels")
    
    print(f"         📏 Total fibonacci lines plotted: {len(all_hlines)}")
    
    # Create title dengan color info
    color_info = " | ".join([f"{tf}={tf_colors.get(tf, 'black').upper()}" for tf in timeframes])
    
    # Plot multi-timeframe chart
    kwargs = {
        'type': 'candle',
        'style': 'charles',
        'title': f'{symbol.upper()} - Multi-Timeframe Fibonacci\nBase: {base_tf} | Colors: {color_info}',
        'ylabel': 'Price',
        'volume': True,
        'figsize': (18, 12)
    }
    
    # Add plots if available
    if addplots:
        kwargs['addplot'] = addplots
    
    # Add horizontal lines if available
    if all_hlines:
        kwargs['hlines'] = dict(
            hlines=all_hlines, 
            colors=all_colors, 
            linestyle='-', 
            linewidths=2,
            alpha=0.7
        )
    
    mpf.plot(chart_data, **kwargs)
    
    print(f"         ✅ Multi-timeframe chart created with color mapping!")

def get_shortest_timeframe(timeframes):
    """Get shortest timeframe untuk candlestick base"""
    tf_order = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '3d', '1w']
    
    for tf in tf_order:
        if tf in timeframes:
            return tf
    
    return timeframes[0]  # fallback



def main():
    """Simple main function untuk testing"""
    print("🚀 FIBONACCI TREND TESTING")
    print("=" * 40)
    
    try:
        # Simple test - only specify data parameters
        result = test_fibonacci_trend(
            symbol='btc', 
            timeframes=['30m','1h'],  # or ['1m', '1h'] for multi-timeframe
            limit=200
        )
        
        print(f"\n✅ Fibonacci trend test completed successfully!")
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()