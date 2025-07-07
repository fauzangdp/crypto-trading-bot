import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

class LSTMFeaturesVisualizer:
    """Simple visualizer untuk LSTM features"""
    
    def __init__(self, multi_tf_data: Dict[str, pd.DataFrame], lstm_features: pd.DataFrame):
        self.multi_tf_data = multi_tf_data
        self.lstm_features = lstm_features
        self.ohlcv_1m = self._prepare_ohlcv_data()
        
    def _prepare_ohlcv_data(self):
        """Siapkan data OHLCV dari 1m timeframe"""
        df_1m = self.multi_tf_data['1m'].copy()
        
        # Cari kolom OHLCV
        ohlcv_cols = {}
        for col in ['open', 'high', 'low', 'close', 'volume']:
            for variant in [col, col.title()]:
                if variant in df_1m.columns:
                    ohlcv_cols[col.title()] = df_1m[variant]
                    break
        
        return pd.DataFrame(ohlcv_cols, index=df_1m.index)
    
    def plot_all(self, last_n_candles=200):
        """Plot candlestick dan semua features vertikal sejajar"""
        # Ambil data terakhir saja
        ohlcv = self.ohlcv_1m.tail(last_n_candles)
        features = self.lstm_features.tail(last_n_candles)
        
        # Setup figure dengan subplots vertikal
        # 1 candlestick + 7 features = 8 plots total
        fig, axes = plt.subplots(8, 1, figsize=(14, 16), sharex=True)
        
        # 1. Candlestick chart (paling atas)
        ax_candle = axes[0]
        
        # Plot candlestick
        for idx in range(len(ohlcv)):
            row = ohlcv.iloc[idx]
            color = 'g' if row['Close'] >= row['Open'] else 'r'
            
            height = abs(row['Close'] - row['Open'])
            bottom = min(row['Close'], row['Open'])
            ax_candle.bar(idx, height, bottom=bottom, color=color, width=0.8, alpha=0.8)
            ax_candle.plot([idx, idx], [row['Low'], row['High']], color='black', linewidth=0.5)
        
        ax_candle.set_title(f'BTC/USDT 1m Candlestick (Last {last_n_candles} candles)', fontsize=12)
        ax_candle.grid(True, alpha=0.3)
        ax_candle.set_ylabel('Price ($)', fontsize=10)
        
        # 2. Features plots (sejajar vertikal)
        feature_names = features.columns.tolist()
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
        
        for i, feature in enumerate(feature_names):
            ax = axes[i+1]  # Mulai dari index 1 (setelah candlestick)
            
            # Plot feature
            ax.plot(features[feature].values, color=colors[i], linewidth=1.2)
            ax.fill_between(range(len(features[feature])), 0, features[feature].values, 
                        alpha=0.2, color=colors[i])
            
            # Garis referensi
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axhline(y=1, color='black', linestyle='-', alpha=0.3)
            
            # Setup
            ax.set_title(feature, fontsize=10, loc='left', pad=2)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.set_ylabel('Value', fontsize=9)
            
            # Remove x-axis labels kecuali yang paling bawah
            if i < len(feature_names) - 1:
                ax.tick_params(labelbottom=False)
        
        # X-axis label hanya di paling bawah
        axes[-1].set_xlabel('Candle Index', fontsize=10)
        
        # Adjust spacing
        plt.subplots_adjust(hspace=0.1)  # Kurangi jarak antar plot
        
        # Overall title
        fig.suptitle('LSTM Features Visualization (Aligned by Time)', fontsize=14, y=0.995)
        
        plt.tight_layout()
        plt.show()
      
    def plot_single_feature(self, feature_name, last_n_candles=200):
        """Plot single feature dengan candlestick"""
        if feature_name not in self.lstm_features.columns:
            print(f"❌ Feature '{feature_name}' tidak ditemukan!")
            return
        
        # Data
        ohlcv = self.ohlcv_1m.tail(last_n_candles)
        feature_data = self.lstm_features[feature_name].tail(last_n_candles)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                       gridspec_kw={'height_ratios': [2, 1]})
        
        # Candlestick
        for idx in range(len(ohlcv)):
            row = ohlcv.iloc[idx]
            color = 'g' if row['Close'] >= row['Open'] else 'r'
            
            height = abs(row['Close'] - row['Open'])
            bottom = min(row['Close'], row['Open'])
            ax1.bar(idx, height, bottom=bottom, color=color, width=0.6, alpha=0.8)
            ax1.plot([idx, idx], [row['Low'], row['High']], color='black', linewidth=0.5)
        
        ax1.set_title(f'BTC/USDT 1m vs {feature_name}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel('Price')
        
        # Feature plot
        ax2.plot(feature_data.values, color='blue', linewidth=2, label=feature_name)
        ax2.fill_between(range(len(feature_data)), 0, feature_data.values, alpha=0.3)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Middle (0.5)')
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Candle Index')
        ax2.set_ylabel('Feature Value')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    def verify_features(self, sample_size=10):
        """Verifikasi bahwa features yang divisualisasikan adalah output yang benar"""
        print("\n🔍 VERIFIKASI OUTPUT FEATURES")
        print("=" * 60)
        
        # 1. Cek shape
        print(f"📊 Shape Features: {self.lstm_features.shape}")
        print(f"   - Jumlah samples: {self.lstm_features.shape[0]}")
        print(f"   - Jumlah features: {self.lstm_features.shape[1]}")
        
        # 2. Cek columns
        print(f"\n📋 Nama Features:")
        for i, col in enumerate(self.lstm_features.columns):
            print(f"   {i+1}. {col}")
        
        # 3. Cek range nilai (harus 0-1)
        print(f"\n📈 Range Nilai per Feature:")
        for col in self.lstm_features.columns:
            min_val = self.lstm_features[col].min()
            max_val = self.lstm_features[col].max()
            mean_val = self.lstm_features[col].mean()
            print(f"   {col:.<25} Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}")
        
        # 4. Sample data
        print(f"\n📝 Sample Data (Last {sample_size} rows):")
        print(self.lstm_features.tail(sample_size).round(4))
        
        # 5. Cek alignment dengan timeframe 1m
        print(f"\n⏰ Time Alignment Check:")
        print(f"   Features Index Start: {self.lstm_features.index[0]}")
        print(f"   Features Index End: {self.lstm_features.index[-1]}")
        print(f"   1m Data Index Start: {self.ohlcv_1m.index[0]}")
        print(f"   1m Data Index End: {self.ohlcv_1m.index[-1]}")
        
        # 6. Verifikasi tidak ada NaN
        nan_count = self.lstm_features.isna().sum().sum()
        print(f"\n✅ Data Quality:")
        print(f"   NaN Count: {nan_count}")
        print(f"   All values in [0,1]: {((self.lstm_features >= 0) & (self.lstm_features <= 1)).all().all()}")
        
        return True

    def plot_feature_distribution(self):
        """Plot distribusi nilai setiap feature"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, col in enumerate(self.lstm_features.columns):
            ax = axes[i]
            
            # Histogram
            ax.hist(self.lstm_features[col], bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(x=0.5, color='red', linestyle='--', label='Middle (0.5)')
            ax.set_title(f'{col} Distribution', fontsize=10)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.set_xlim(-0.1, 1.1)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplot
        if len(self.lstm_features.columns) < 8:
            axes[-1].set_visible(False)
        
        plt.suptitle('Feature Value Distributions', fontsize=14)
        plt.tight_layout()
        plt.show()