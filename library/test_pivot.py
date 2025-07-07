import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import os
from datetime import datetime, timedelta
from data_utils import DataLoader
from indicators import SupportResistance


def filter_valid_pivots(pivot_highs, pivot_lows, df, lookforward=150):
    """
    Filter pivot points berdasarkan validitas
    
    Rules:
    - Pivot low invalid jika dalam 30 candle kedepan ada pivot low baru tanpa pivot high
    - Pivot high invalid jika dalam 30 candle kedepan ada pivot high baru tanpa pivot low
    
    Args:
        pivot_highs (pd.Series): Pivot highs dari deteksi awal
        pivot_lows (pd.Series): Pivot lows dari deteksi awal
        df (pd.DataFrame): Data OHLC original
        lookforward (int): Jumlah candle untuk check kedepan (default: 30)
    
    Returns:
        tuple: (filtered_pivot_highs, filtered_pivot_lows)
    """
    # Convert to list untuk easier manipulation
    high_times = list(pivot_highs.index)
    low_times = list(pivot_lows.index)
    
    # Lists untuk valid pivots
    valid_highs = {}
    valid_lows = {}
    
    # Filter pivot lows
    for i, low_time in enumerate(low_times):
        is_valid = True
        low_idx = df.index.get_loc(low_time)
        
        # Check next 30 candles
        end_idx = min(low_idx + lookforward, len(df) - 1)
        
        # Find next pivot low dalam range
        next_low_found = False
        pivot_high_between = False
        
        for j in range(i + 1, len(low_times)):
            next_low_time = low_times[j]
            next_low_idx = df.index.get_loc(next_low_time)
            
            # Check if within lookforward range
            if next_low_idx <= end_idx:
                next_low_found = True
                
                # Check if ada pivot high between current and next low
                for high_time in high_times:
                    high_idx = df.index.get_loc(high_time)
                    if low_idx < high_idx < next_low_idx:
                        pivot_high_between = True
                        break
                
                # If next low found without high between, current low invalid
                if not pivot_high_between:
                    is_valid = False
                break
        
        if is_valid:
            valid_lows[low_time] = pivot_lows[low_time]
    
    # Filter pivot highs
    for i, high_time in enumerate(high_times):
        is_valid = True
        high_idx = df.index.get_loc(high_time)
        
        # Check next 30 candles
        end_idx = min(high_idx + lookforward, len(df) - 1)
        
        # Find next pivot high dalam range
        next_high_found = False
        pivot_low_between = False
        
        for j in range(i + 1, len(high_times)):
            next_high_time = high_times[j]
            next_high_idx = df.index.get_loc(next_high_time)
            
            # Check if within lookforward range
            if next_high_idx <= end_idx:
                next_high_found = True
                
                # Check if ada pivot low between current and next high
                for low_time in low_times:
                    low_idx = df.index.get_loc(low_time)
                    if high_idx < low_idx < next_high_idx:
                        pivot_low_between = True
                        break
                
                # If next high found without low between, current high invalid
                if not pivot_low_between:
                    is_valid = False
                break
        
        if is_valid:
            valid_highs[high_time] = pivot_highs[high_time]
    
    # Convert back to Series
    filtered_highs = pd.Series(valid_highs)
    filtered_lows = pd.Series(valid_lows)
    
    return filtered_highs, filtered_lows


def calculate_stop_loss(pivot_highs, pivot_lows, df):
    """
    Calculate stop loss untuk setiap pivot point dengan minimal distance
    
    Minimal Distance Logic:
    1. Ambil 10 candle sebelum pivot
    2. Hitung percentage change tiap candle (absolute value) 
    3. Rata-rata percentage = minimal distance dari pivot ke stop loss
    4. Jika ada pivot sebelumnya:
       - Hitung midpoint antara current dan previous pivot
       - Gunakan midpoint atau minimal distance (mana yang lebih jauh)
    5. Jika tidak ada pivot sebelumnya:
       - Gunakan minimal distance
    
    Args:
        pivot_highs (pd.Series): Valid pivot highs
        pivot_lows (pd.Series): Valid pivot lows
        df (pd.DataFrame): Data OHLC original
    
    Returns:
        tuple: (stop_loss_for_lows, stop_loss_for_highs)
    """
    stop_loss_lows = {}
    stop_loss_highs = {}
    
    # Convert to list untuk easier manipulation
    high_times = list(pivot_highs.index)
    low_times = list(pivot_lows.index)
    
    # Function to calculate average percentage change
    def calculate_avg_percentage(pivot_idx, df, lookback=10):
        """
        Calculate average percentage change dari lookback candles
        
        Steps:
        1. Ambil 10 candle sebelum pivot
        2. Hitung percentage change tiap candle (absolute value)
        3. Return rata-rata percentage
        """
        start_idx = max(0, pivot_idx - lookback)
        
        percentages = []
        for i in range(start_idx + 1, pivot_idx + 1):
            # Calculate percentage change dari close ke close
            prev_close = df['close'].iloc[i-1]
            curr_close = df['close'].iloc[i]
            
            # Avoid division by zero
            if prev_close != 0:
                pct_change = abs((curr_close - prev_close) / prev_close)
                percentages.append(pct_change)
        
        # Return average percentage (atau default 0.5% jika tidak ada data)
        return np.mean(percentages) if percentages else 0.005
    
    # Calculate stop loss untuk pivot lows
    for i, current_low_time in enumerate(low_times):
        current_low_price = pivot_lows[current_low_time]
        pivot_idx = df.index.get_loc(current_low_time)
        
        # Calculate minimal distance based on average percentage
        avg_pct = calculate_avg_percentage(pivot_idx, df)
        minimal_distance = current_low_price * avg_pct
        
        # Find previous pivot lows
        prev_lows = []
        for j in range(i):
            prev_low_time = low_times[j]
            prev_low_price = pivot_lows[prev_low_time]
            # Only consider lower lows
            if prev_low_price < current_low_price:
                prev_lows.append((prev_low_time, prev_low_price))
        
        if prev_lows:
            # Find nearest (highest) among previous lower lows
            nearest_low = max(prev_lows, key=lambda x: x[1])
            nearest_low_price = nearest_low[1]
            
            # Calculate midpoint
            midpoint = (current_low_price + nearest_low_price) / 2
            
            # Check if distance is sufficient
            distance_to_pivot = current_low_price - midpoint
            
            if distance_to_pivot < minimal_distance:
                # Use minimal distance
                stop_loss_price = current_low_price - minimal_distance
            else:
                # Use midpoint
                stop_loss_price = midpoint
        else:
            # No previous low found, use minimal distance
            stop_loss_price = current_low_price - minimal_distance
        
        stop_loss_lows[current_low_time] = stop_loss_price
    
    # Calculate stop loss untuk pivot highs
    for i, current_high_time in enumerate(high_times):
        current_high_price = pivot_highs[current_high_time]
        pivot_idx = df.index.get_loc(current_high_time)
        
        # Calculate minimal distance based on average percentage
        avg_pct = calculate_avg_percentage(pivot_idx, df)
        minimal_distance = current_high_price * avg_pct
        
        # Find previous pivot highs
        prev_highs = []
        for j in range(i):
            prev_high_time = high_times[j]
            prev_high_price = pivot_highs[prev_high_time]
            # Only consider higher highs
            if prev_high_price > current_high_price:
                prev_highs.append((prev_high_time, prev_high_price))
        
        if prev_highs:
            # Find nearest (lowest) among previous higher highs
            nearest_high = min(prev_highs, key=lambda x: x[1])
            nearest_high_price = nearest_high[1]
            
            # Calculate midpoint
            midpoint = (current_high_price + nearest_high_price) / 2
            
            # Check if distance is sufficient
            distance_to_pivot = midpoint - current_high_price
            
            if distance_to_pivot < minimal_distance:
                # Use minimal distance
                stop_loss_price = current_high_price + minimal_distance
            else:
                # Use midpoint
                stop_loss_price = midpoint
        else:
            # No previous high found, use minimal distance
            stop_loss_price = current_high_price + minimal_distance
        
        stop_loss_highs[current_high_time] = stop_loss_price
    
    # Convert to Series
    sl_lows = pd.Series(stop_loss_lows)
    sl_highs = pd.Series(stop_loss_highs)
    
    return sl_lows, sl_highs


def export_to_csv(df, pivot_highs, pivot_lows, stop_loss_lows, stop_loss_highs, timeframe='1m'):
    """
    Export pivot points to CSV with proper transition handling
    """
    # Initialize array with zeros
    decision_array = [0.0] * len(df)
    
    # Step 1: Collect all pivots with their indices
    all_pivots = []
    
    # Debug: Check for duplicate timestamps
    print("\n🔍 Checking pivot data integrity...")
    
    # Add pivot lows (1.0)
    pivot_low_indices = []
    for timestamp in pivot_lows.index:
        if timestamp in df.index:
            idx = df.index.get_loc(timestamp)
            all_pivots.append((idx, timestamp, 1.0))
            pivot_low_indices.append(idx)
    
    # Add pivot highs (-1.0)
    pivot_high_indices = []
    for timestamp in pivot_highs.index:
        if timestamp in df.index:
            idx = df.index.get_loc(timestamp)
            all_pivots.append((idx, timestamp, -1.0))
            pivot_high_indices.append(idx)
    
    # Sort by index
    all_pivots.sort(key=lambda x: x[0])
    
    print(f"\n📊 Pivot Analysis:")
    print(f"   Total Pivot Lows: {len(pivot_lows)}")
    print(f"   Total Pivot Highs: {len(pivot_highs)}")
    print(f"   Total Pivots: {len(all_pivots)}")
    
    # Check minimum distance between consecutive pivots
    if len(all_pivots) > 1:
        distances = []
        for i in range(1, len(all_pivots)):
            dist = all_pivots[i][0] - all_pivots[i-1][0]
            distances.append(dist)
        
        print(f"\n📏 Distance Analysis:")
        print(f"   Minimum distance: {min(distances)} candles")
        print(f"   Maximum distance: {max(distances)} candles")
        print(f"   Average distance: {sum(distances)/len(distances):.1f} candles")
        
        # Show pivots with very small distances
        small_distances = [(i, d) for i, d in enumerate(distances) if d < 5]
        if small_distances:
            print(f"\n⚠️ Found {len(small_distances)} pivot pairs with distance < 5:")
            for i, dist in small_distances[:5]:  # Show first 5
                p1 = all_pivots[i]
                p2 = all_pivots[i+1]
                print(f"   {p1[1].strftime('%H:%M:%S')} ({p1[2]}) -> {p2[1].strftime('%H:%M:%S')} ({p2[2]}): {dist} candles")
    
    # Step 2: Set all pivot values
    for idx, timestamp, value in all_pivots:
        decision_array[idx] = value
    
    # Step 3: Create transitions between pivots
    same_pivot_transitions = 0
    different_pivot_transitions = 0
    
    for i in range(len(all_pivots) - 1):
        curr_idx, curr_time, curr_val = all_pivots[i]
        next_idx, next_time, next_val = all_pivots[i + 1]
        
        distance = next_idx - curr_idx
        
        # Create transition for any distance > 1
        if distance > 1:
            # Check if same pivot type (both 1 or both -1)
            if curr_val == next_val:
                same_pivot_transitions += 1
                # Transition through 0: pivot -> 0 -> pivot
                mid_point = curr_idx + distance // 2
                
                # First half: from current pivot to 0
                for j in range(curr_idx + 1, mid_point):
                    progress = (j - curr_idx) / (mid_point - curr_idx)
                    transition_value = curr_val * (1 - progress)
                    decision_array[j] = round(transition_value, 3)
                
                # Middle point: 0
                if mid_point < next_idx:
                    decision_array[mid_point] = 0.0
                
                # Second half: from 0 to next pivot
                for j in range(mid_point + 1, next_idx):
                    progress = (j - mid_point) / (next_idx - mid_point)
                    transition_value = next_val * progress
                    decision_array[j] = round(transition_value, 3)
                
                # Debug first few same-type transitions
                if same_pivot_transitions <= 3:
                    print(f"\n   Same-type transition {same_pivot_transitions}:")
                    print(f"   {curr_time.strftime('%H:%M:%S')} ({curr_val}) -> 0 -> {next_time.strftime('%H:%M:%S')} ({next_val})")
                    print(f"   Distance: {distance} candles, zero at index {mid_point}")
            else:
                different_pivot_transitions += 1
                # Different pivot types: direct linear transition
                for j in range(curr_idx + 1, next_idx):
                    progress = (j - curr_idx) / distance
                    transition_value = curr_val + (next_val - curr_val) * progress
                    decision_array[j] = round(transition_value, 3)
    
    print(f"\n📊 Transition Summary:")
    print(f"   Same-type transitions (through 0): {same_pivot_transitions}")
    print(f"   Different-type transitions (direct): {different_pivot_transitions}")
    print(f"   Total transitions: {same_pivot_transitions + different_pivot_transitions}")
    
    # Create DataFrame
    export_df = pd.DataFrame({
        'decision': decision_array
    }, index=df.index)
    
    # Save
    start_timestamp = df.index[0].strftime('%Y%m%d_%H%M%S')
    filename = f"label_{timeframe}_{start_timestamp}.csv"
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    save_dir = os.path.join(parent_dir, 'database_learning')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    filepath = os.path.join(save_dir, filename)
    export_df.to_csv(filepath, index=True)
    
    # Final verification
    print(f"\n✅ Export Complete: {filename}")
    
    # Count consecutive duplicates
    consecutive_count = 0
    max_consecutive = 0
    last_val = None
    
    for val in decision_array:
        if val in [1.0, -1.0]:
            if val == last_val:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 1
                last_val = val
        else:
            consecutive_count = 0
            last_val = None
    
    if max_consecutive > 1:
        print(f"\n⚠️ WARNING: Found maximum {max_consecutive} consecutive pivot values!")
        print("   This indicates a problem with pivot detection or data processing.")
    
    # Value distribution
    value_counts = {}
    for val in decision_array:
        if val == 1.0:
            key = "1.0 (Pivot Low)"
        elif val == -1.0:
            key = "-1.0 (Pivot High)"
        elif val == 0.0:
            key = "0.0 (Neutral)"
        else:
            key = "Transition"
        value_counts[key] = value_counts.get(key, 0) + 1
    
    print(f"\n📊 Value Distribution:")
    for key, count in sorted(value_counts.items()):
        print(f"   {key}: {count}")
    
    return filepath

def alternative_export_method(df, pivot_highs, pivot_lows, stop_loss_lows, stop_loss_highs, timeframe='1m'):
    """
    Alternative export method dengan string-based timestamp matching
    """
    print(f"\n🔄 ALTERNATIVE: String-based timestamp matching...")
    
    # Copy dataframe
    export_df = df.copy()
    export_df['decision'] = 0
    export_df['SL'] = 0.0
    
    # Convert all timestamps to strings for exact matching
    df_timestamps = set(df.index.strftime('%Y-%m-%d %H:%M:%S'))
    
    pivot_low_matches = 0
    pivot_high_matches = 0
    
    # Process pivot lows dengan string matching
    for timestamp, pivot_price in pivot_lows.items():
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        if timestamp_str in df_timestamps:
            # Find exact match dalam dataframe
            matching_rows = df.index[df.index.strftime('%Y-%m-%d %H:%M:%S') == timestamp_str]
            
            if len(matching_rows) > 0:
                match_timestamp = matching_rows[0]
                export_df.loc[match_timestamp, 'decision'] = 1
                pivot_low_matches += 1
                
                # Set SL jika ada
                if timestamp in stop_loss_lows:
                    sl_price = stop_loss_lows[timestamp]
                    sl_percentage = abs(pivot_price - sl_price) / pivot_price
                    export_df.loc[match_timestamp, 'SL'] = sl_percentage
    
    # Process pivot highs dengan string matching
    for timestamp, pivot_price in pivot_highs.items():
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        if timestamp_str in df_timestamps:
            # Find exact match dalam dataframe
            matching_rows = df.index[df.index.strftime('%Y-%m-%d %H:%M:%S') == timestamp_str]
            
            if len(matching_rows) > 0:
                match_timestamp = matching_rows[0]
                export_df.loc[match_timestamp, 'decision'] = -1
                pivot_high_matches += 1
                
                # Set SL jika ada
                if timestamp in stop_loss_highs:
                    sl_price = stop_loss_highs[timestamp]
                    sl_percentage = abs(pivot_price - sl_price) / pivot_price
                    export_df.loc[match_timestamp, 'SL'] = sl_percentage
    
    print(f"   ✅ Alternative method - Lows matched: {pivot_low_matches}")
    print(f"   ✅ Alternative method - Highs matched: {pivot_high_matches}")
    
    # Save alternative version
    start_timestamp = df.index[0].strftime('%Y%m%d_%H%M%S')
    filename = f"label_{timeframe}_{start_timestamp}_alternative.csv"
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    save_dir = os.path.join(parent_dir, 'database_learning')
    filepath = os.path.join(save_dir, filename)
    
    export_df.to_csv(filepath, index=True)
    
    # Report final distribution
    final_distribution = export_df['decision'].value_counts().sort_index()
    print(f"\n📊 Alternative method results:")
    for decision, count in final_distribution.items():
        decision_name = {-1: "Pivot High", 0: "No Pivot", 1: "Pivot Low"}.get(decision, f"Unknown")
        print(f"   Decision {decision} ({decision_name}): {count}")
    
    return filepath


    
def visualize_pivots(symbol='btc', limit=5500, left_bars=50, right_bars=50):
    """
    Function untuk visualisasi pivot high dan low pada candlestick chart
    
    Args:
        symbol (str): Symbol trading (default: 'btc')
        limit (int): Jumlah candles yang ditampilkan (default: 500)
        left_bars (int): Jumlah bars kiri untuk deteksi pivot (default: 10)
        right_bars (int): Jumlah bars kanan untuk deteksi pivot (default: 10)
    """
    print(f"🚀 Starting Pivot Visualization for {symbol.upper()}")
    print(f"   Requested candles: {limit} (1-minute timeframe)")
    print(f"   Estimated duration: ~{limit} minutes ({limit/60:.1f} hours)")
    print(f"   Data selection: Most recent {limit} candles")
    print("=" * 50)
    
    # Initialize variables
    raw_high_count = 0
    raw_low_count = 0
    stop_loss_lows = pd.Series()
    stop_loss_highs = pd.Series()
    csv_path = None
    
    try:
        # Step 1: Load data 1 menit
        print("\n📊 Step 1: Loading 1-minute data...")
        loader = DataLoader()
        data = loader.load_data(
            symbol=symbol,
            timeframes='1m',
            limit=limit,
            auto_align=False
        )
        
        # Ambil dataframe 1m
        df = data['1m'].copy()
        print(f"✅ Loaded {len(df)} candles")
        
        # Verify timeframe
        if len(df) > 1:
            time_diff = (df.index[1] - df.index[0]).total_seconds() / 60
            print(f"✅ Verified timeframe: {time_diff:.1f} minutes between candles")
        
        # Show preview of data range
        print(f"\n📋 Data Preview:")
        print(f"   First candle: {df.index[0].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Last candle:  {df.index[-1].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Duration: {(df.index[-1] - df.index[0]).total_seconds() / 3600:.2f} hours")
        
        # Step 2: Deteksi pivot high dan low
        print("\n🔍 Step 2: Detecting pivot highs and lows...")
        sr = SupportResistance(left_bars=left_bars, right_bars=right_bars)
        
        pivot_highs = sr.find_pivot_highs(df['high'])
        pivot_lows = sr.find_pivot_lows(df['low'])
        
        # Store raw counts
        raw_high_count = len(pivot_highs)
        raw_low_count = len(pivot_lows)
        
        print(f"✅ Found {raw_high_count} raw pivot highs")
        print(f"✅ Found {raw_low_count} raw pivot lows")
        
        # Step 2.5: Filter valid pivots
        print("\n🔎 Step 2.5: Filtering valid pivots...")
        filtered_highs, filtered_lows = filter_valid_pivots(pivot_highs, pivot_lows, df)
        
        print(f"✅ Valid pivot highs: {len(filtered_highs)} (filtered {raw_high_count - len(filtered_highs)})")
        print(f"✅ Valid pivot lows: {len(filtered_lows)} (filtered {raw_low_count - len(filtered_lows)})")
        
        # Use filtered pivots for visualization
        pivot_highs = filtered_highs
        pivot_lows = filtered_lows
        
        # Step 2.6: Calculate stop loss levels
        print("\n⚡ Step 2.6: Calculating stop loss levels...")
        stop_loss_lows, stop_loss_highs = calculate_stop_loss(pivot_highs, pivot_lows, df)
        
        print(f"✅ Stop loss for pivot lows: {len(stop_loss_lows)}")
        print(f"✅ Stop loss for pivot highs: {len(stop_loss_highs)}")
        
        # Step 3: Visualisasi dengan Decision Indicator UPDATED
        print("\n📈 Step 3: Creating visualization with decision indicator...")
        
        # Prepare data untuk mplfinance
        ohlc_data = df[['open', 'high', 'low', 'close', 'volume']].copy()
        ohlc_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Create full series untuk pivot points (dengan NaN untuk non-pivot)
        pivot_high_series = pd.Series(index=df.index, dtype=float)
        pivot_low_series = pd.Series(index=df.index, dtype=float)
        stop_loss_low_series = pd.Series(index=df.index, dtype=float)
        stop_loss_high_series = pd.Series(index=df.index, dtype=float)
        
        # Fill pivot values
        for timestamp, price in pivot_highs.items():
            if timestamp in pivot_high_series.index:
                pivot_high_series[timestamp] = price
                
        for timestamp, price in pivot_lows.items():
            if timestamp in pivot_low_series.index:
                pivot_low_series[timestamp] = price
        
        # Fill stop loss values
        for timestamp, price in stop_loss_lows.items():
            if timestamp in stop_loss_low_series.index:
                stop_loss_low_series[timestamp] = price
                
        for timestamp, price in stop_loss_highs.items():
            if timestamp in stop_loss_high_series.index:
                stop_loss_high_series[timestamp] = price
        
        # CREATE DECISION SERIES DENGAN LOGIC YANG SAMA SEPERTI CSV EXPORT
        decision_series = pd.Series(index=df.index, data=0.0)
        
        # Collect ALL pivots
        all_pivots = []
        
        for timestamp in pivot_lows.index:
            if timestamp in df.index:
                idx = df.index.get_loc(timestamp)
                all_pivots.append((idx, 1.0))
        
        for timestamp in pivot_highs.index:
            if timestamp in df.index:
                idx = df.index.get_loc(timestamp)
                all_pivots.append((idx, -1.0))
        
        # Sort by index
        all_pivots.sort(key=lambda x: x[0])
        
        # AGGRESSIVE FILTERING - sama seperti di export_to_csv
        # Use ALL pivots without filtering (same as export_to_csv)
        print(f"   Using all {len(all_pivots)} pivots for decision indicator")

        # Set all pivot values first
        for idx, val in all_pivots:
            decision_series.iloc[idx] = val

        # Create transitions between pivots (same logic as export_to_csv)
        for i in range(len(all_pivots) - 1):
            curr_idx, curr_val = all_pivots[i]
            next_idx, next_val = all_pivots[i + 1]
            
            distance = next_idx - curr_idx
            
            if distance > 1:
                # Check if same pivot type
                if curr_val == next_val:
                    # Transition through 0
                    mid_point = curr_idx + distance // 2
                    
                    # First half: pivot to 0
                    for j in range(curr_idx + 1, mid_point):
                        progress = (j - curr_idx) / (mid_point - curr_idx)
                        value = curr_val * (1 - progress)
                        decision_series.iloc[j] = round(value, 3)
                    
                    # Middle: 0
                    if mid_point < next_idx:
                        decision_series.iloc[mid_point] = 0.0
                    
                    # Second half: 0 to pivot
                    for j in range(mid_point + 1, next_idx):
                        progress = (j - mid_point) / (next_idx - mid_point)
                        value = next_val * progress
                        decision_series.iloc[j] = round(value, 3)
                else:
                    # Different types: direct transition
                    for j in range(curr_idx + 1, next_idx):
                        progress = (j - curr_idx) / distance
                        value = curr_val + (next_val - curr_val) * progress
                        decision_series.iloc[j] = round(value, 3)
                
        # Create markers list
        all_markers = []
        
        # Add pivot high markers if any exist
        if pivot_high_series.notna().any():
            all_markers.append(
                mpf.make_addplot(
                    pivot_high_series,
                    type='scatter',
                    markersize=100,
                    marker='o',
                    color='green',
                    alpha=0.8,
                    secondary_y=False
                )
            )
        
        # Add pivot low markers if any exist
        if pivot_low_series.notna().any():
            all_markers.append(
                mpf.make_addplot(
                    pivot_low_series,
                    type='scatter',
                    markersize=100,
                    marker='o',
                    color='blue',
                    alpha=0.8,
                    secondary_y=False
                )
            )
        
        # Add stop loss markers for pivot lows
        if stop_loss_low_series.notna().any():
            all_markers.append(
                mpf.make_addplot(
                    stop_loss_low_series,
                    type='scatter',
                    markersize=80,
                    marker='o',
                    color='red',
                    alpha=0.8,
                    secondary_y=False
                )
            )
        
        # Add stop loss markers for pivot highs
        if stop_loss_high_series.notna().any():
            all_markers.append(
                mpf.make_addplot(
                    stop_loss_high_series,
                    type='scatter',
                    markersize=80,
                    marker='o',
                    color='red',
                    alpha=0.8,
                    secondary_y=False
                )
            )
        
        # Add decision indicator as panel below
        all_markers.append(
            mpf.make_addplot(
                decision_series,
                panel=2,  # Panel baru di bawah volume
                color='purple',
                ylabel='Decision',
                type='line',
                width=2,
                alpha=0.8
            )
        )
        
        # Add horizontal lines for decision levels
        all_markers.append(
            mpf.make_addplot(
                pd.Series(index=df.index, data=1.0),  # Buy level
                panel=2,
                color='green',
                linestyle='--',
                width=1,
                alpha=0.5
            )
        )
        
        all_markers.append(
            mpf.make_addplot(
                pd.Series(index=df.index, data=-1.0),  # Sell level
                panel=2,
                color='red',
                linestyle='--',
                width=1,
                alpha=0.5
            )
        )
        
        all_markers.append(
            mpf.make_addplot(
                pd.Series(index=df.index, data=0.0),  # Neutral level
                panel=2,
                color='black',
                linestyle='-',
                width=1,
                alpha=0.5
            )
        )
        
        # Create candlestick chart dengan decision indicator
        if all_markers:
            # Create detailed title with date range
            start_str = df.index[0].strftime('%Y-%m-%d %H:%M')
            end_str = df.index[-1].strftime('%Y-%m-%d %H:%M')
            duration_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
            
            mpf.plot(
                ohlc_data,
                type='candle',
                style='charles',
                title=f'{symbol.upper()} - 1m with Filtered Pivot Points & Decision Indicator\n'
                      f'Green = Valid High ({len(pivot_highs)}), Blue = Valid Low ({len(pivot_lows)}), '
                      f'Red = Stop Loss ({len(stop_loss_lows) + len(stop_loss_highs)})\n'
                      f'Period: {start_str} to {end_str} ({len(df)} candles, {duration_hours:.1f} hours)',
                ylabel='Price',
                volume=True,
                addplot=all_markers,
                figsize=(16, 12),  # Lebih tinggi untuk accommodate decision panel
                tight_layout=True,
                panel_ratios=(3, 1, 1),  # Price, Volume, Decision
                ylabel_lower='Decision\n(-1=Sell, 0=Neutral, 1=Buy)'
            )
        else:
            # Jika tidak ada pivot, tampilkan chart biasa
            start_str = df.index[0].strftime('%Y-%m-%d %H:%M')
            end_str = df.index[-1].strftime('%Y-%m-%d %H:%M')
            
            mpf.plot(
                ohlc_data,
                type='candle',
                style='charles',
                title=f'{symbol.upper()} - 1m (No pivots found)\n'
                      f'Period: {start_str} to {end_str} ({len(df)} candles)',
                ylabel='Price',
                volume=True,
                figsize=(16, 12),
                tight_layout=True
            )
        
        print("✅ Visualization completed!")
        
        # Step 4: Export to CSV
        print("\n📊 Step 4: Exporting to CSV...")
        try:
            csv_path = export_to_csv(df, pivot_highs, pivot_lows, stop_loss_lows, stop_loss_highs, timeframe='1m')
        except Exception as e:
            print(f"⚠️ Warning: Could not export CSV: {e}")
            csv_path = None
        
        # Print summary
        print("\n📊 Summary:")
        print(f"   Symbol: {symbol.upper()}")
        print(f"   Timeframe: 1m")
        print(f"   Total Candles: {len(df)}")
        print(f"   Valid Pivot Highs: {len(pivot_highs)} (from {raw_high_count} raw)")
        print(f"   Valid Pivot Lows: {len(pivot_lows)} (from {raw_low_count} raw)")
        print(f"   Decision Pivots Used: {len(all_pivots)}")  # <-- GUNAKAN all_pivots
        
        return {
            'data': df,
            'pivot_highs': pivot_highs,
            'pivot_lows': pivot_lows,
            'stop_loss_lows': stop_loss_lows,
            'stop_loss_highs': stop_loss_highs,
            'raw_high_count': raw_high_count,
            'raw_low_count': raw_low_count,
            'csv_path': csv_path,
            'time_range': {
                'start': df.index[0],
                'end': df.index[-1],
                'duration_hours': (df.index[-1] - df.index[0]).total_seconds() / 3600,
                'candle_count': len(df)
            }
        }
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function untuk testing"""
    print("🎯 PIVOT HIGH/LOW VISUALIZATION TEST WITH STOP LOSS & CSV EXPORT")
    print("=" * 60)
    print("\n📌 Color Legend:")
    print("   🟢 Green = Valid Pivot High")
    print("   🔵 Blue  = Valid Pivot Low")
    print("   🔴 Red   = Stop Loss Level")
    print("\n📏 Stop Loss Logic:")
    print("   - Minimal distance = avg % change of 10 candles before pivot")
    print("   - If previous pivot exists: use midpoint or minimal distance (whichever is farther)")
    print("   - If no previous pivot: use minimal distance from current pivot")
    print("\n💾 CSV Export:")
    print("   - Saved to: database_learning/label_<timeframe>_<timestamp>.csv")
    print("   - Decision: 1 (pivot low), 0 (no pivot), -1 (pivot high)")
    print("   - SL: percentage distance from pivot to stop loss")
    print("\n⏱️ Time Information:")
    print("   - Timeframe: 1 minute")
    print("   - Default: 500 candles = ~8.3 hours of data")
    print("   - Data loads from most recent available")
    
    # Show expected time range based on current time
    now = datetime.now()
    expected_start = now - timedelta(minutes=500)
    print(f"\n   Expected approximate range (if run now):")
    print(f"   - From: {expected_start.strftime('%Y-%m-%d %H:%M')} ")
    print(f"   - To:   {now.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # Test dengan default parameters
    result = visualize_pivots(
        symbol='btc',
        limit=10000,
        left_bars=50,
        right_bars=50
    )
    
    print("\n💡 Tips:")
    print("   - limit=60   → 1 hour of data")
    print("   - limit=500  → ~8.3 hours of data")
    print("   - limit=1440 → 24 hours (1 day) of data")
    print("   - limit=10080 → 1 week of data")
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    main()