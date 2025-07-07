import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import json
import os

class BinanceFuturesDataFetcher:
    def __init__(self):
        self.base_url = "https://fapi.binance.com"
        self.symbol = "BTCUSDT"
        
    def get_klines(self, interval, limit=1500, start_time=None, end_time=None):
        """
        Mengambil data kline dari Binance Futures API
        
        Args:
            interval: '1m', '5m', '30m', '1h', dll
            limit: maksimal 1500 per request
            start_time: timestamp mulai (ms)
            end_time: timestamp akhir (ms)
        """
        endpoint = f"{self.base_url}/fapi/v1/klines"
        
        params = {
            'symbol': self.symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None
    
    def get_multiple_klines(self, interval, total_candles=10000):
        """
        Mengambil data kline dalam jumlah besar dengan pagination
        DIMULAI DARI DATA TERLAMA (chronological order)
        
        Args:
            interval: timeframe ('1m', '5m', '30m', '1h')
            total_candles: jumlah total candle yang diinginkan
        """
        all_data = []
        limit_per_request = 1500  # Maksimal per request Binance
        
        # PERBAIKAN: Hitung start_time berdasarkan interval untuk mendapatkan data terlama dulu
        interval_ms = self._get_interval_ms(interval)
        # Mundur dari sekarang sebanyak total_candles * interval
        now = int(datetime.now().timestamp() * 1000)
        start_time = now - (total_candles * interval_ms)
        
        print(f"📅 Fetching {total_candles} candles for {interval} starting from {datetime.fromtimestamp(start_time/1000)}")
        
        current_start = start_time
        
        while len(all_data) < total_candles:
            remaining = total_candles - len(all_data)
            current_limit = min(limit_per_request, remaining)
            
            # PERBAIKAN: Gunakan startTime (bukan endTime) untuk urutan chronological
            data = self.get_klines(
                interval=interval,
                limit=current_limit,
                start_time=current_start
            )
            
            if not data:
                print("❌ No more data available")
                break
            
            # PERBAIKAN: Remove duplicates berdasarkan timestamp
            for candle in data:
                candle_time = candle[0]  # Open time
                
                # Check if this timestamp already exists
                if not any(existing[0] == candle_time for existing in all_data):
                    all_data.append(candle)
                
                # Break jika sudah cukup
                if len(all_data) >= total_candles:
                    break
            
            if len(all_data) >= total_candles:
                break
            
            # Update start_time untuk request berikutnya
            if data:
                last_candle_time = data[-1][0]  # Last candle open time
                current_start = last_candle_time + interval_ms  # Next interval
            
            print(f"📊 Fetched {len(data)} candles for {interval}. Total unique: {len(all_data)}")
            
            # Rate limiting
            time.sleep(0.1)
        
        # PERBAIKAN: Sort final data by timestamp (ascending) untuk memastikan urutan chronological
        all_data.sort(key=lambda x: x[0])
        
        return all_data[:total_candles]
    
    def _get_interval_ms(self, interval):
        """
        Convert interval string ke milliseconds
        """
        interval_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        return interval_map.get(interval, 60 * 1000)
    
    def format_data(self, raw_data):
        """
        Format data mentah menjadi DataFrame pandas
        HANYA 6 KOLOM: timestamp, open, high, low, close, volume
        """
        if not raw_data:
            return None
        
        # PERBAIKAN: Ambil hanya 6 kolom yang dibutuhkan dari 12 kolom Binance API
        formatted_data = []
        for candle in raw_data:
            formatted_candle = [
                candle[0],  # 0: Open time (timestamp)
                candle[1],  # 1: Open
                candle[2],  # 2: High
                candle[3],  # 3: Low
                candle[4],  # 4: Close
                candle[5],  # 5: Volume
            ]
            formatted_data.append(formatted_candle)
        
        # PERBAIKAN: Kolom sesuai permintaan (hanya 6 kolom)
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # FIX: Gunakan formatted_data (bukan raw_data)
        df = pd.DataFrame(formatted_data, columns=columns)
        
        # PERBAIKAN: Konversi tipe data yang benar
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # PERBAIKAN: Konversi timestamp ke datetime dan set sebagai index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # PERBAIKAN: Sort by index untuk memastikan chronological order
        df.sort_index(ascending=True, inplace=True)
        
        # PERBAIKAN: Remove any duplicate timestamps
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    
    def fetch_all_timeframes(self, candles_per_timeframe=10000):
        """
        Mengambil data untuk semua timeframe yang diminta
        """
        timeframes = ['1m', '5m', '30m', '1h']
        results = {}
        
        for tf in timeframes:
            print(f"\n🔄 Fetching {candles_per_timeframe} candles for {tf} timeframe...")
            raw_data = self.get_multiple_klines(tf, candles_per_timeframe)
            
            if raw_data:
                formatted_data = self.format_data(raw_data)
                if formatted_data is not None and len(formatted_data) > 0:
                    results[tf] = formatted_data
                    
                    # Validation info
                    print(f"✅ Successfully processed {len(formatted_data)} candles for {tf}")
                    print(f"📅 Date range: {formatted_data.index[0]} to {formatted_data.index[-1]}")
                    print(f"📊 Sample data shape: {formatted_data.shape}")
                    
                    # Check for chronological order
                    is_sorted = formatted_data.index.is_monotonic_increasing
                    print(f"🔍 Chronological order: {'✅ CORRECT' if is_sorted else '❌ WRONG'}")
                    
                    # Check for duplicates
                    has_duplicates = formatted_data.index.duplicated().any()
                    print(f"🔍 Duplicate timestamps: {'❌ FOUND' if has_duplicates else '✅ NONE'}")
                    
                else:
                    print(f"❌ Failed to format data for {tf}")
            else:
                print(f"❌ Failed to fetch raw data for {tf}")
        
        return results
    
    def save_to_csv(self, data_dict, base_folder="database_learning"):
        """
        Simpan data ke folder sesuai timeframe
        """
        for timeframe, df in data_dict.items():
            if df is not None:
                # Buat folder jika belum ada
                folder_path = os.path.join(base_folder, timeframe)
                os.makedirs(folder_path, exist_ok=True)
                
                # PERBAIKAN: Simpan dengan timestamp di nama file (format konsisten)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(folder_path, f"btc_futures_{timeframe}_{timestamp}.csv")
                
                # Save dengan index (timestamp)
                df.to_csv(filename)
                print(f"💾 Data {timeframe} saved to {filename}")
                print(f"   📝 Shape: {df.shape}")
                print(f"   📅 Range: {df.index[0]} to {df.index[-1]}")
    
    def validate_data_quality(self, df, timeframe):
        """
        Validasi kualitas data
        """
        print(f"\n🔍 Data Quality Check for {timeframe}:")
        print("-" * 40)
        
        # Check chronological order
        is_sorted = df.index.is_monotonic_increasing
        print(f"⏰ Chronological order: {'✅ PASS' if is_sorted else '❌ FAIL'}")
        
        # Check duplicates
        has_duplicates = df.index.duplicated().any()
        duplicate_count = df.index.duplicated().sum()
        print(f"🔁 Duplicate timestamps: {'❌ ' + str(duplicate_count) + ' found' if has_duplicates else '✅ NONE'}")
        
        # Check missing values
        missing_values = df.isnull().sum().sum()
        print(f"❓ Missing values: {'❌ ' + str(missing_values) + ' found' if missing_values > 0 else '✅ NONE'}")
        
        # Check data consistency (OHLC logic)
        consistency_errors = 0
        for idx, row in df.iterrows():
            if not (row['low'] <= row['open'] <= row['high'] and 
                    row['low'] <= row['close'] <= row['high']):
                consistency_errors += 1
        
        print(f"📊 OHLC consistency: {'❌ ' + str(consistency_errors) + ' errors' if consistency_errors > 0 else '✅ PASS'}")
        
        # Time interval consistency
        if len(df) > 1:
            expected_interval = self._get_interval_ms(timeframe) / 1000  # Convert to seconds
            time_diffs = df.index.to_series().diff().dt.total_seconds().dropna()
            irregular_intervals = (time_diffs != expected_interval).sum()
            print(f"⏱️  Interval consistency: {'❌ ' + str(irregular_intervals) + ' irregular' if irregular_intervals > 0 else '✅ PASS'}")
        
        return {
            'is_sorted': is_sorted,
            'has_duplicates': has_duplicates,
            'missing_values': missing_values,
            'consistency_errors': consistency_errors
        }

def create_folder_structure(base_folder="database_learning"):
    """
    Buat struktur folder otomatis
    """
    timeframes = ['1m', '5m', '30m', '1h']
    for tf in timeframes:
        folder_path = os.path.join(base_folder, tf)
        os.makedirs(folder_path, exist_ok=True)
    
    print(f"✅ Folder structure created in {base_folder}/")

def main():
    """
    Main function dengan error handling dan validation
    """
    print("🚀 Bitcoin Futures Data Fetcher (FIXED VERSION)")
    print("=" * 60)
    
    # Buat struktur folder dulu
    create_folder_structure()
    
    # Inisialisasi fetcher
    fetcher = BinanceFuturesDataFetcher()
    
    # Ambil data untuk semua timeframe
    print("\n📡 Fetching Bitcoin Futures data from Binance...")
    all_data = fetcher.fetch_all_timeframes(candles_per_timeframe=10000)
    
    # Validasi setiap dataset
    print("\n" + "="*60)
    print("📋 DATA QUALITY VALIDATION")
    print("="*60)
    
    for timeframe, df in all_data.items():
        if df is not None:
            fetcher.validate_data_quality(df, timeframe)
    
    # Tampilkan info data
    print("\n" + "="*60)
    print("📊 DATA SUMMARY")
    print("="*60)
    
    for timeframe, df in all_data.items():
        if df is not None:
            print(f"\n📈 {timeframe.upper()} Data:")
            print(f"   Shape: {df.shape}")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
            print(f"   Columns: {list(df.columns)}")
            print("\n   First 3 rows:")
            print(df.head(3).to_string())
            print("\n   Last 3 rows:")
            print(df.tail(3).to_string())
            
    # Simpan ke CSV
    print(f"\n💾 Saving data to CSV files...")
    fetcher.save_to_csv(all_data)
    
    return all_data

def get_single_timeframe(interval='1h', candles=5000):
    """
    Fungsi sederhana untuk mengambil data satu timeframe dengan validation
    """
    print(f"📡 Fetching {candles} candles for {interval}...")
    
    fetcher = BinanceFuturesDataFetcher()
    raw_data = fetcher.get_multiple_klines(interval, candles)
    
    if raw_data:
        df = fetcher.format_data(raw_data)
        if df is not None and len(df) > 0:
            print(f"✅ Successfully fetched {len(df)} candles")
            print(f"📅 Date range: {df.index[0]} to {df.index[-1]}")
            
            # Quick validation
            fetcher.validate_data_quality(df, interval)
            
            # Auto save ke folder yang sesuai
            folder_path = os.path.join("database_learning", interval)
            os.makedirs(folder_path, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(folder_path, f"btc_futures_{interval}_{timestamp}.csv")
            df.to_csv(filename)
            print(f"💾 Data saved to {filename}")
            
            return df
        else:
            print("❌ Failed to format data")
            return None
    else:
        print("❌ Failed to fetch raw data")
        return None

if __name__ == "__main__":
    try:
        # Jalankan script utama
        data = main()
        
        print("\n🎉 Script completed successfully!")
        print(f"📂 Files saved in: database_learning/")
        print("\n📋 Summary:")
        if data:
            for tf, df in data.items():
                if df is not None:
                    print(f"   {tf}: {len(df)} candles ({df.index[0].strftime('%Y-%m-%d %H:%M')} to {df.index[-1].strftime('%Y-%m-%d %H:%M')})")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()