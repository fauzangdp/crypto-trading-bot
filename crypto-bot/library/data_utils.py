import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, List, Optional, Union
import time
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re

class DataLoader:
    """
    Clean Data Loader Library untuk Trading Data
    3 Core Methods + 1 Convenience Method
    """
    
    def __init__(self, base_path='database_learning'):
        """
        Initialize DataLoader
        
        Args:
            base_path (str): Path ke database directory
        """
        self.base_path = base_path
        self.available_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    
    # ==================== 3 CORE METHODS (All logic built-in) ====================
    
    def search_and_load_symbol(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        CORE METHOD 1: Search symbol dan load semua timeframe yang tersedia
        All search + load logic built-in (no helper methods)
        
        Args:
            symbol (str): Symbol name atau pattern (e.g., 'btc', 'BTCUSDT', 'eth')
                         - Jika <=4 char: pattern search
                         - Jika >4 char: exact match
        
        Returns:
            Dict[str, pd.DataFrame]: {timeframe: dataframe} untuk semua TF yang tersedia
        """
        print(f"🔍 Searching and loading symbol: {symbol}")
        
        # ============= PATTERN SEARCH LOGIC (built-in) =============
        base_symbol = None
        if len(symbol) <= 4:
            print(f"🔎 Pattern search for: {symbol}")
            # Search dalam timeframe pertama yang tersedia
            for tf in self.available_timeframes:
                tf_path = os.path.join(self.base_path, tf)
                if not os.path.exists(tf_path):
                    continue
                
                # Search pattern
                search_pattern = f"*{symbol.upper()}*"
                matches = glob.glob(os.path.join(tf_path, search_pattern))
                
                if matches:
                    # Extract base symbol dari filename
                    filename = os.path.basename(matches[0])
                    base_name = os.path.splitext(filename)[0]
                    
                    # Parse: btc_futures_1m_20250625_001433 -> btc_futures
                    parts = base_name.split('_')
                    
                    # Find timeframe position
                    for i, part in enumerate(parts):
                        if part in self.available_timeframes:
                            # Base symbol = everything before timeframe
                            base_symbol = '_'.join(parts[:i])
                            print(f"🔍 Pattern '{symbol}' found -> base symbol: '{base_symbol}'")
                            break
                    
                    if base_symbol:
                        break
                    # Fallback: return full base name
                    base_symbol = base_name
                    break
        else:
            # Exact symbol
            print(f"📄 Exact symbol search: {symbol}")
            base_symbol = symbol
        
        if not base_symbol:
            raise FileNotFoundError(f"No symbol found matching: {symbol}")
        
        print(f"🎯 Base symbol identified: {base_symbol}")
        
        # ============= LOAD ALL TIMEFRAMES LOGIC (built-in) =============
        symbol_data = {}
        
        for tf in self.available_timeframes:
            tf_path = os.path.join(self.base_path, tf)
            
            if not os.path.exists(tf_path):
                continue
            
            # Build pattern untuk file search
            pattern = os.path.join(tf_path, f"{base_symbol}_{tf}_*.csv")
            matches = glob.glob(pattern)
            
            if matches:
                # Ambil file terbaru
                latest_file = max(matches, key=os.path.getctime)
                filename = os.path.basename(latest_file)
                file_path = os.path.join(tf_path, filename)
                
                try:
                    # ============= CSV LOADING LOGIC (built-in) =============
                    df = pd.read_csv(file_path)
                    
                    # Validate required columns
                    required_columns = ['open', 'high', 'low', 'close', 'volume']
                    missing_cols = [col for col in required_columns if col not in df.columns]
                    
                    if missing_cols:
                        print(f"⚠️  Missing columns in {filename}: {missing_cols}")
                        continue
                    
                    # Auto-detect timestamp column
                    timestamp_cols = ['timestamp', 'time', 'open_time', 'datetime', 'Date', 'date']
                    timestamp_col = None
                    
                    for col in timestamp_cols:
                        if col in df.columns:
                            timestamp_col = col
                            break
                    
                    # Convert to datetime index if found
                    if timestamp_col:
                        try:
                            df[timestamp_col] = df[timestamp_col].str.replace('.', ':')
                            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                            df = df.set_index(timestamp_col)
                            df = df.sort_index()
                        except Exception as e:
                            print(f"⚠️  Timestamp conversion failed: {e}")
                    
                    if len(df) > 0:
                        symbol_data[tf] = df
                        print(f"✅ Loaded {tf}: {len(df)} candles ({filename})")
                    else:
                        print(f"⚠️  {tf}: Empty data ({filename})")
                        
                except Exception as e:
                    print(f"❌ Error loading {tf}: {e}")
            else:
                print(f"🔍 {tf}: No files found")
        
        if not symbol_data:
            raise FileNotFoundError(f"No data files found for symbol: {base_symbol}")
        
        print(f"📊 Total timeframes loaded: {len(symbol_data)} {list(symbol_data.keys())}")
        return symbol_data
    
    def select_timeframes(self, symbol_data: Dict[str, pd.DataFrame], 
                         timeframes: Union[str, List[str]]) -> Dict[str, pd.DataFrame]:
        """
        CORE METHOD 2: Pilih timeframe mana yang ingin diambil
        All selection logic built-in (no helper methods)
        
        Args:
            symbol_data (Dict): Output dari search_and_load_symbol
            timeframes (str atau List[str]): Timeframe yang ingin diambil
                                           - Single: '1m' atau '5m'
                                           - Multiple: ['1m', '5m'] atau ['1m', '5m', '30m', '1h']
        
        Returns:
            Dict[str, pd.DataFrame]: Filtered data sesuai timeframes yang diminta
        """
        # ============= SELECTION LOGIC (built-in) =============
        # Convert ke list jika input string
        if isinstance(timeframes, str):
            timeframes = [timeframes]
        
        print(f"📋 Selecting timeframes: {timeframes}")
        
        # Validate timeframes tersedia
        available_tfs = list(symbol_data.keys())
        selected_data = {}
        
        for tf in timeframes:
            if tf in symbol_data:
                selected_data[tf] = symbol_data[tf].copy()
                print(f"✅ Selected {tf}: {len(selected_data[tf])} candles")
            else:
                print(f"❌ {tf}: Not available. Available: {available_tfs}")
        
        if not selected_data:
            raise ValueError(f"None of requested timeframes {timeframes} are available. Available: {available_tfs}")
        
        print(f"📊 Final selection: {len(selected_data)} timeframes {list(selected_data.keys())}")
        return selected_data
    
    def validate_and_align(self, selected_data: Dict[str, pd.DataFrame], 
                        base_timeframe: str = None,
                        alignment_mode: str = 'full') -> Dict[str, pd.DataFrame]:
        """
        CORE METHOD 3: Validate dan align timeframes jika perlu
        All alignment logic built-in (no helper methods)
        
        Args:
            selected_data (Dict): Output dari select_timeframes
            base_timeframe (str): Base timeframe untuk alignment (default: shortest timeframe)
            alignment_mode (str): Alignment strategy - NEW PARAMETER
                - 'full': Full alignment to base timeframe (current behavior, default)
                - 'current_only': Only align current/latest timestamp, preserve historical
                - 'none': No alignment, return as-is
        
        Returns:
            Dict[str, pd.DataFrame]: Aligned data based on selected mode
        """
        num_timeframes = len(selected_data)
        timeframes = list(selected_data.keys())
        
        print(f"🔧 Validating {num_timeframes} timeframe(s): {timeframes}")
        print(f"📐 Alignment mode: {alignment_mode}")
        
        # ============= VALIDATION LOGIC (built-in) =============
        # Single timeframe: no alignment needed
        if num_timeframes == 1:
            print("✅ Single timeframe detected - no alignment needed")
            return selected_data
        
        # Mode 'none': return as-is
        if alignment_mode == 'none':
            print("⏭️  Alignment mode 'none' - returning data as-is")
            return selected_data
        
        # Multiple timeframes: need alignment based on mode
        print(f"⚙️  Multiple timeframes detected - alignment mode: {alignment_mode}")
        
        # Auto-select base timeframe (shortest interval)
        if base_timeframe is None:
            tf_order = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
            for tf in tf_order:
                if tf in timeframes:
                    base_timeframe = tf
                    break
        
        if base_timeframe not in selected_data:
            raise ValueError(f"Base timeframe {base_timeframe} not in selected data")
        
        print(f"🎯 Using base timeframe: {base_timeframe}")
        
        # ============= ALIGNMENT MODE: CURRENT_ONLY =============
        if alignment_mode == 'current_only':
            print(f"🔄 Current-only alignment mode - preserving historical data")
            
            aligned_data = {}
            base_df = selected_data[base_timeframe].copy()
            
            # Keep base as-is
            aligned_data[base_timeframe] = base_df
            
            # Check if datetime index
            has_datetime_index = isinstance(base_df.index[0], (pd.Timestamp, pd.DatetimeIndex))
            
            if not has_datetime_index:
                print("⚠️  No datetime index - returning data as-is (cannot align current)")
                return selected_data
            
            # Get current (latest) timestamp from base
            base_current = base_df.index[-1]
            print(f"📍 Base current timestamp: {base_current}")
            
            # For each other timeframe
            for tf, df in selected_data.items():
                if tf == base_timeframe:
                    continue
                
                # Copy original data (preserve historical)
                aligned_df = df.copy()
                
                # Find the current candle for this timeframe
                if tf == '5m':
                    aligned_current = base_current.floor('5min')
                elif tf == '15m':
                    aligned_current = base_current.floor('15min')
                elif tf == '30m':
                    aligned_current = base_current.floor('30min')
                elif tf == '1h':
                    aligned_current = base_current.floor('1h')
                elif tf == '4h':
                    aligned_current = base_current.floor('4h')
                elif tf == '1d':
                    aligned_current = base_current.floor('1d')
                else:
                    aligned_current = base_current
                
                print(f"📍 {tf} aligned current: {aligned_current}")
                
                # Ensure the aligned current exists in the data
                if aligned_current not in aligned_df.index:
                    # Find nearest past timestamp
                    valid_timestamps = aligned_df.index[aligned_df.index <= aligned_current]
                    if len(valid_timestamps) > 0:
                        nearest_current = valid_timestamps[-1]
                        print(f"   ↳ Using nearest: {nearest_current}")
                    else:
                        print(f"   ⚠️ No valid current timestamp found for {tf}")
                
                # Store original data with metadata
                aligned_data[tf] = aligned_df
                
                # Report coverage
                time_coverage = (aligned_df.index[-1] - aligned_df.index[0])
                print(f"✅ {tf}: {len(aligned_df)} candles, coverage: {time_coverage}")
            
            return aligned_data
        
        # ============= ALIGNMENT MODE: FULL (Original behavior) =============
        elif alignment_mode == 'full':
            base_df = selected_data[base_timeframe].copy()
            aligned_data = {base_timeframe: base_df}
            
            print(f"🔄 Full alignment mode - aligning to base timeframe: {base_timeframe}")
            print(f"📏 Base data shape: {base_df.shape}")
            
            # Check if datetime index
            has_datetime_index = isinstance(base_df.index[0], (pd.Timestamp, pd.DatetimeIndex))
            
            if not has_datetime_index:
                print("⚠️  No datetime index - using simple sequential alignment")
                
                # ========= SIMPLE ALIGNMENT LOGIC (built-in) =========
                for tf, df in selected_data.items():
                    if tf == base_timeframe:
                        continue
                    
                    aligned_df = pd.DataFrame(index=base_df.index)
                    
                    for col in df.columns:
                        if len(df) >= len(base_df):
                            aligned_df[f'{col}_{tf}'] = df[col].tail(len(base_df)).values
                        else:
                            # Forward fill shorter series
                            temp_series = pd.Series(df[col].values, index=range(len(df)))
                            expanded = temp_series.reindex(range(len(base_df)), method='ffill')
                            aligned_df[f'{col}_{tf}'] = expanded.values
                    
                    aligned_data[tf] = aligned_df
            else:
                # ========= DATETIME-BASED ALIGNMENT LOGIC (built-in) =========
                print("✓ Using proper datetime-based alignment")
                
                for tf, df in selected_data.items():
                    if tf == base_timeframe:
                        continue
                    
                    print(f"🔄 Aligning {tf}...")
                    aligned_df = pd.DataFrame(index=base_df.index)
                    
                    # Skip problematic columns
                    skip_columns = ['close_time', 'ignore']
                    
                    # Align each column
                    for col in df.columns:
                        if col in skip_columns:
                            continue  # Skip non-numeric columns
                        
                        # Detect appropriate dtype
                        if df[col].dtype in ['datetime64[ns]', 'object']:
                            aligned_series = pd.Series(index=base_df.index, dtype='object')
                        else:
                            aligned_series = pd.Series(index=base_df.index, dtype='float64')
                        
                        for base_timestamp in base_df.index:
                            # ======= FLOOR TIMESTAMP LOGIC (built-in) =======
                            if tf == '5m':
                                floor_timestamp = base_timestamp.floor('5min')
                            elif tf == '15m':
                                floor_timestamp = base_timestamp.floor('15min')
                            elif tf == '30m':
                                floor_timestamp = base_timestamp.floor('30min')
                            elif tf == '1h':
                                floor_timestamp = base_timestamp.floor('1h')
                            elif tf == '4h':
                                floor_timestamp = base_timestamp.floor('4h')
                            elif tf == '1d':
                                floor_timestamp = base_timestamp.floor('1d')
                            else:
                                floor_timestamp = base_timestamp.floor('1min')
                            
                            # Find corresponding value
                            if floor_timestamp in df.index:
                                aligned_series[base_timestamp] = df.loc[floor_timestamp, col]
                            else:
                                # Find nearest past value
                                valid_timestamps = df.index[df.index <= floor_timestamp]
                                if len(valid_timestamps) > 0:
                                    nearest_timestamp = valid_timestamps[-1]
                                    aligned_series[base_timestamp] = df.loc[nearest_timestamp, col]
                                else:
                                    aligned_series[base_timestamp] = np.nan
                        
                        # Add column with suffix
                        aligned_df[f'{col}_{tf}'] = aligned_series
                    
                    aligned_data[tf] = aligned_df
                    
                    # Report coverage
                    coverage = aligned_df.count().sum() / (len(aligned_df) * len(aligned_df.columns)) if len(aligned_df) > 0 else 0
                    print(f"✅ {tf} aligned: {coverage:.1%} data coverage")
            
            # ============= VALIDATION RESULTS (built-in) =============
            print(f"\n📊 Alignment Validation:")
            print("=" * 40)
            
            # Use first timeframe as base reference
            base_tf = list(aligned_data.keys())[0]
            base_len = len(aligned_data[base_tf])
            
            # Check lengths
            for tf, df in aligned_data.items():
                if alignment_mode == 'current_only':
                    # For current_only mode, show actual candle count
                    print(f"{tf:>4}: {len(df):>6} candles")
                else:
                    # For full mode, check alignment
                    coverage = df.count().sum() / (len(df) * len(df.columns)) if len(df) > 0 else 0
                    status = "✅ OK" if len(df) == base_len else "❌ MISMATCH"
                    print(f"{tf:>4}: {len(df):>6} rows | {coverage:>5.1%} coverage | {status}")
            
            print("=" * 40)
            
            return aligned_data
        
        else:
            raise ValueError(f"Invalid alignment_mode: {alignment_mode}. Must be 'full', 'current_only', or 'none'")
        
    # ==================== 1 CONVENIENCE METHOD ====================
    
    def load_data(self, 
                symbol: str, 
                timeframes: Union[str, List[str]] = '1m',
                limit: int = 1000,
                auto_align: bool = True,
                alignment_mode: str = 'full',
                date_filter: Union[None, Dict, str] = None) -> Dict[str, pd.DataFrame]:
        """
        CONVENIENCE METHOD: Uses the 3 core methods above
        One-liner untuk quick usage
        
        Args:
            symbol (str): Symbol atau pattern ('btc', 'BTCUSDT')
            timeframes (str atau List[str]): Single TF atau multiple ['1m', '5m']  
            limit (int): Jumlah candles yang diambil per timeframe (default: 1000)
            auto_align (bool): Auto align jika multi-timeframe (default: True)
            alignment_mode (str): Alignment strategy (default: 'full')
                - 'full': Full alignment to base timeframe (original behavior)
                - 'current_only': Only align latest timestamp, preserve historical
                - 'none': No alignment at all
            date_filter (None, Dict, str): Filter by date range or period (default: None)
                - None: No date filtering (current behavior)
                - Dict: {'start': '2024-01-01', 'end': '2024-01-31'}
                - str: Period like '7d', '24h', '1M'
            
        Returns:
            Dict[str, pd.DataFrame]: Ready-to-use data (aligned based on mode)
        """
        print(f"🚀 CONVENIENCE METHOD - Loading data with parameters:")
        print(f"   Symbol: {symbol}")
        print(f"   Timeframes: {timeframes}")
        print(f"   Limit: {limit}")
        print(f"   Auto-align: {auto_align}")
        print(f"   Alignment mode: {alignment_mode}")
        print(f"   Date filter: {date_filter}")  # NEW
        
        # ============= USE CORE METHODS =============
        # Step 1: Search and load symbol (using core method)
        print(f"\n📡 Step 1: Using search_and_load_symbol()...")
        all_data = self.search_and_load_symbol(symbol)
        
        # Step 2: Select timeframes (using core method)
        print(f"\n🎯 Step 2: Using select_timeframes()...")
        selected_data = self.select_timeframes(all_data, timeframes)
        
        # ============= NEW: Apply date filter (built-in) =============
        if date_filter is not None:
            print(f"\n📅 Applying date filter: {date_filter}")
            
            # Parse date filter to get start and end dates
            start_date = None
            end_date = None
            
            if isinstance(date_filter, dict):
                # Dictionary format
                if 'start' in date_filter:
                    start_date = pd.to_datetime(date_filter['start'])
                if 'end' in date_filter:
                    end_date = pd.to_datetime(date_filter['end'])
                    
            elif isinstance(date_filter, str):
                # Period string format (e.g., '7d', '24h', '1M')
                pattern = r'(\d+)([dhMmw])'
                match = re.match(pattern, date_filter)
                
                if match:
                    amount = int(match.group(1))
                    unit = match.group(2)
                    
                    # Calculate from current time
                    end_date = pd.Timestamp.now()
                    
                    if unit == 'd':  # days
                        start_date = end_date - timedelta(days=amount)
                    elif unit == 'h':  # hours
                        start_date = end_date - timedelta(hours=amount)
                    elif unit == 'm':  # minutes
                        start_date = end_date - timedelta(minutes=amount)
                    elif unit == 'M':  # months (approximate)
                        start_date = end_date - timedelta(days=amount*30)
                    elif unit == 'w':  # weeks
                        start_date = end_date - timedelta(weeks=amount)
                else:
                    raise ValueError(f"Invalid period format: {date_filter}")
            
            print(f"📅 Date filter parsed: Start={start_date}, End={end_date}")
            
            # Apply filter to each timeframe
            filtered_data = {}
            
            for tf, df in selected_data.items():
                # Check if dataframe has datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    print(f"⚠️  {tf}: No datetime index, skipping date filter")
                    filtered_data[tf] = df
                    continue
                
                # Apply date filter
                original_len = len(df)
                
                if start_date and end_date:
                    filtered_df = df.loc[start_date:end_date]
                elif start_date:
                    filtered_df = df.loc[start_date:]
                elif end_date:
                    filtered_df = df.loc[:end_date]
                else:
                    filtered_df = df
                
                filtered_data[tf] = filtered_df
                print(f"✅ {tf}: {original_len} → {len(filtered_df)} candles "
                    f"({len(filtered_df)/original_len*100:.1f}% retained)")
            
            selected_data = filtered_data
        
        # Apply limit if specified
        if limit:
            print(f"\n✂️  Applying limit: {limit} candles per timeframe")
            for tf in selected_data:
                if len(selected_data[tf]) > limit:
                    selected_data[tf] = selected_data[tf].tail(limit)
                    print(f"   {tf}: Limited to {len(selected_data[tf])} candles")
        
        # Step 3: Validate and align (using core method)
        if auto_align:
            print(f"\n🔧 Step 3: Using validate_and_align() with mode '{alignment_mode}'...")
            final_data = self.validate_and_align(
                selected_data, 
                alignment_mode=alignment_mode
            )
        else:
            print(f"\n⏭️  Step 3: Skipping alignment (auto_align=False)")
            final_data = selected_data
        
        print(f"\n✅ CONVENIENCE METHOD COMPLETED!")
        print(f"📊 Final result: {len(final_data)} timeframes ready")
        
        return final_data

# ==================== SIMPLE TESTING + CANDLESTICK ====================


def test_load_data_visual(symbol='btc', timeframes='1m', limit=1000, show_alignment=True, date_filter=None):
    """
    Test load_data method dengan visualization yang sederhana
    
    Args:
        symbol (str): Symbol trading ('btc', 'eth', dll)
        timeframes (str/list): Single TF ('1m') atau Multiple TF (['1m', '5m', '1h'])
        limit (int): Jumlah candles per timeframe (default: 1000)
        show_alignment (bool): Show alignment check untuk multi-TF (default: True)
    
    Returns:
        Dict[str, pd.DataFrame]: Loaded data hasil dari load_data()
    """
    print(f"🧪 Testing load_data() method")
    print(f"   Symbol: {symbol}")
    print(f"   Timeframes: {timeframes}")
    print(f"   Limit: {limit}")
    print("=" * 50)
    
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
            date_filter=date_filter
        )
        
        end_time = time.time()
        print(f"✅ Data loaded in {end_time - start_time:.2f} seconds")
        
        # Step 2: Analyze loaded data
        print(f"\n📈 Data Analysis:")
        for tf_name, df in data.items():
            print(f"   {tf_name}: {df.shape} | Index: {type(df.index[0]).__name__}")
            if len(df) > 0:
                print(f"      Range: {df.index[0]} → {df.index[-1]}")
                print(f"      Columns: {list(df.columns)}")
        
        # Step 3: Create visualization berdasarkan jumlah timeframes
        num_timeframes = len(data)
        
        if num_timeframes == 1:
            print(f"\n🕯️ Creating single timeframe visualization...")
            _create_single_tf_visual(data, symbol)
        else:
            print(f"\n📊 Creating multi-timeframe alignment visualization...")
            _create_multi_tf_visual(data, symbol, show_alignment)
        
        return data
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def _create_single_tf_visual(data, symbol):
    """
    Visual Type 1: Single timeframe candlestick chart
    """
    tf_name = list(data.keys())[0]
    df = data[tf_name]
    
    print(f"   📊 Single timeframe chart: {tf_name}")
    
    # Prepare OHLCV data
    chart_data = _prepare_ohlcv_data(df, tf_name)
    
    if chart_data is None:
        print(f"   ❌ Cannot create chart - missing OHLCV data")
        return
    
    try:
        # Create candlestick dengan mplfinance
        mpf.plot(
            chart_data,
            type='candle',
            style='charles',
            title=f'{symbol.upper()} - {tf_name} ({len(chart_data)} candles)',
            ylabel='Price',
            volume=True,
            figsize=(14, 8),
            tight_layout=True
        )
        
        print(f"   ✅ Single timeframe chart created successfully")
        
    except Exception as e:
        print(f"   ❌ Chart creation failed: {e}")

def _create_multi_tf_visual(data, symbol, show_alignment=True):
    """
    Visual Type 2: Multi-timeframe alignment visualization
    """
    timeframes = list(data.keys())
    print(f"   📊 Multi-timeframe chart: {timeframes}")
    
    # Get base timeframe (shortest/first)
    base_tf = timeframes[0]
    base_df = data[base_tf]
    
    print(f"   🎯 Base timeframe: {base_tf}")
    
    # Create figure dengan 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                  gridspec_kw={'height_ratios': [4, 1]})
    
    # Colors untuk different timeframes
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Plot base timeframe sebagai candlestick background
    base_close = _get_close_column(base_df, base_tf)
    if base_close is not None:
        ax1.plot(base_df.index, base_close, 
                color='black', linewidth=1, alpha=0.5, 
                label=f'{base_tf} Price (Base)')
    
    alignment_check = []
    
    # Plot each timeframe dengan alignment markers
    for i, (tf_name, df) in enumerate(data.items()):
        color = colors[i % len(colors)]
        
        # Get close price column
        close_data = _get_close_column(df, tf_name)
        if close_data is None:
            print(f"   ⚠️ {tf_name}: No close price found")
            continue
        
        # Plot line untuk timeframe ini
        ax1.plot(df.index, close_data, 
                color=color, linewidth=2, alpha=0.8,
                label=f'{tf_name} Close')
        
        if show_alignment and tf_name != base_tf:
            # Show alignment markers (every N points)
            step = max(1, len(df) // 30)  # Max 30 markers
            marker_indices = range(0, len(df), step)
            
            ax1.scatter(df.index[marker_indices], 
                       close_data.iloc[marker_indices],
                       color=color, marker='o', s=30, 
                       alpha=0.7, edgecolors='white',
                       label=f'{tf_name} Alignment Points')
        
        # Check alignment quality
        non_null_pct = (close_data.notna().sum() / len(close_data)) * 100
        alignment_check.append(f"{tf_name}: {non_null_pct:.1f}% data coverage")
        
        print(f"   ✅ {tf_name}: {non_null_pct:.1f}% data coverage")
    
    # Chart formatting
    ax1.set_title(f'{symbol.upper()} - Multi-Timeframe Alignment Check\n' + 
                  f'Timeframes: {", ".join(timeframes)}')
    ax1.set_ylabel('Price')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Volume subplot (dari base timeframe)
    volume_data = _get_volume_column(base_df, base_tf)
    if volume_data is not None:
        ax2.bar(base_df.index, volume_data, 
               width=pd.Timedelta(minutes=1), alpha=0.6, 
               color='blue', label=f'{base_tf} Volume')
        ax2.set_ylabel('Volume')
        ax2.legend()
    else:
        # Text placeholder jika no volume
        ax2.text(0.5, 0.5, 'No Volume Data Available', 
                transform=ax2.transAxes, ha='center', va='center',
                fontsize=12, alpha=0.5)
    
    ax2.set_xlabel('Time')
    ax2.grid(True, alpha=0.3)
    
    # Add alignment summary text
    alignment_text = '\n'.join(alignment_check)
    ax1.text(0.02, 0.98, f'Alignment Check:\n{alignment_text}', 
            transform=ax1.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print(f"   ✅ Multi-timeframe alignment chart created")
    print(f"   📍 Markers show alignment points between timeframes")

def _prepare_ohlcv_data(df, tf_name):
    """
    Prepare OHLCV data untuk mplfinance dari single atau multi-timeframe dataframe
    """
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    chart_data = {}
    
    for col in required_cols:
        # Try exact column name first
        if col in df.columns:
            chart_data[col.title()] = df[col]
        # Try with timeframe suffix
        elif f'{col}_{tf_name}' in df.columns:
            chart_data[col.title()] = df[f'{col}_{tf_name}']
        # Try finding any column containing the name
        else:
            found = False
            for df_col in df.columns:
                if col in df_col.lower():
                    chart_data[col.title()] = df[df_col]
                    found = True
                    break
            
            if not found:
                print(f"      ⚠️ Missing {col} column")
                return None
    
    chart_df = pd.DataFrame(chart_data, index=df.index).dropna()
    return chart_df if len(chart_df) > 0 else None

def _get_close_column(df, tf_name):
    """
    Get close price column dari dataframe (handle suffix)
    """
    # Try different variations
    close_options = ['close', f'close_{tf_name}', 'Close', f'Close_{tf_name}']
    
    for col_name in close_options:
        if col_name in df.columns:
            return df[col_name]
    
    # Search dalam semua columns
    for col in df.columns:
        if 'close' in col.lower():
            return df[col]
    
    return None

def _get_volume_column(df, tf_name):
    """
    Get volume column dari dataframe (handle suffix) 
    """
    # Try different variations
    volume_options = ['volume', f'volume_{tf_name}', 'Volume', f'Volume_{tf_name}']
    
    for col_name in volume_options:
        if col_name in df.columns:
            return df[col_name]
    
    # Search dalam semua columns
    for col in df.columns:
        if 'volume' in col.lower():
            return df[col]
    
    return None

# ==================== SIMPLE USAGE EXAMPLES ====================

def main():
    """Main function dengan contoh usage"""
    print("🚀 TESTING LOAD_DATA VISUAL FUNCTION")
    print("=" * 60)
    
    # Test 1: Single timeframe
    print("\n🧪 Test 1: Single Timeframe")
    test_load_data_visual(
        symbol='btc',
        timeframes='1m',
        limit=500
    )
    
    # Test 2: Multi timeframe
    print("\n🧪 Test 2: Multi Timeframe (Alignment Check)")
    test_load_data_visual(
        symbol='btc',
        timeframes=['1m', '5m', '1h'],
        limit=1000,
        show_alignment=True
    )
    
    # Test 3: With date filter (NEW)
    print("\n🧪 Test 3: With Date Filter (Last 7 days)")
    test_load_data_visual(
        symbol='btc',
        timeframes='1h',
        limit=500,
        date_filter='7d'
    )
    
    # Test 4: With date range (NEW)
    print("\n🧪 Test 4: With Date Range")
    test_load_data_visual(
        symbol='btc',
        timeframes=['5m', '15m'],
        limit=1000,
        date_filter={
            'start': '2024-12-20',
            'end': '2024-12-25'
        }
    )
    
    print(f"\n✅ Testing completed!")

if __name__ == "__main__":
    main()
