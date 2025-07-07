import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List, Tuple, Optional


class TechnicalIndicators:
    """
    Technical Indicators Class untuk LSTM Trading
    Semua output sudah normalized ke range 0-1 untuk LSTM optimization
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize dengan OHLCV data
        
        Args:
            data (pd.DataFrame): DataFrame dengan columns ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        self.data = data.copy()
        self.high = data['High']
        self.low = data['Low'] 
        self.close = data['Close']
        self.volume = data['Volume']
        
    @property
    def stochastic_k_norm(self, period: int = 14, smooth_k: int = 3) -> pd.Series:
        """
        Stochastic %K Normalized (0-1)
        
        Args:
            period (int): Period untuk %K calculation (default: 14)
            smooth_k (int): Smoothing period untuk %K (default: 3)
            
        Returns:
            pd.Series: Stochastic %K normalized (0-1)
        """
        # Calculate %K
        lowest_low = self.low.rolling(window=period).min()
        highest_high = self.high.rolling(window=period).max()
        
        k_percent = 100 * (self.close - lowest_low) / (highest_high - lowest_low)
        
        # Smooth %K
        k_smooth = k_percent.rolling(window=smooth_k).mean()
        
        # Normalize to 0-1
        k_normalized = k_smooth / 100
        
        return k_normalized.fillna(0)
    
    @property  
    def bb_position(self, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """
        Bollinger Band Position (0-1)
        0 = At lower band, 0.5 = At middle band, 1 = At upper band
        
        Args:
            period (int): Period untuk moving average (default: 20)
            std_dev (float): Standard deviation multiplier (default: 2.0)
            
        Returns:
            pd.Series: BB Position normalized (0-1)
        """
        # Calculate Bollinger Bands
        sma = self.close.rolling(window=period).mean()
        std = self.close.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        # Calculate position (0-1)
        bb_position = (self.close - lower_band) / (upper_band - lower_band)
        
        # Clip to 0-1 range (handle price outside bands)
        bb_position = bb_position.clip(0, 1)
        
        return bb_position.fillna(0.5)
    
    @property
    def rsi_norm(self, period: int = 14) -> pd.Series:
        """
        RSI Normalized (0-1)
        
        Args:
            period (int): Period untuk RSI calculation (default: 14)
            
        Returns:
            pd.Series: RSI normalized (0-1)
        """
        # Calculate price changes
        delta = self.close.diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Normalize to 0-1
        rsi_normalized = rsi / 100
        
        return rsi_normalized.fillna(0.5)
    
    @property
    def macd_norm(self, fast: int = 12, slow: int = 26, signal: int = 9, 
                  normalization_period: int = 50) -> pd.Series:
        """
        MACD Normalized (0-1)
        Using histogram normalization with rolling min/max
        
        Args:
            fast (int): Fast EMA period (default: 12)
            slow (int): Slow EMA period (default: 26)  
            signal (int): Signal line EMA period (default: 9)
            normalization_period (int): Period untuk min/max normalization (default: 50)
            
        Returns:
            pd.Series: MACD Histogram normalized (0-1)
        """
        # Calculate MACD
        ema_fast = self.close.ewm(span=fast).mean()
        ema_slow = self.close.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        # Normalize histogram using rolling min/max
        rolling_min = histogram.rolling(window=normalization_period).min()
        rolling_max = histogram.rolling(window=normalization_period).max()
        
        # Avoid division by zero
        range_diff = rolling_max - rolling_min
        range_diff = range_diff.replace(0, 1)
        
        macd_normalized = (histogram - rolling_min) / range_diff
        
        # Clip to 0-1 and fill NaN with 0.5 (neutral)
        macd_normalized = macd_normalized.clip(0, 1)
        
        return macd_normalized.fillna(0.5)
    
    @property
    def adx_norm(self, period: int = 14) -> pd.Series:
        """
        ADX (Average Directional Index) Normalized (0-1)
        
        Args:
            period (int): Period untuk ADX calculation (default: 14)
            
        Returns:
            pd.Series: ADX normalized (0-1)
        """
        # Calculate True Range
        tr1 = self.high - self.low
        tr2 = abs(self.high - self.close.shift(1))
        tr3 = abs(self.low - self.close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        plus_dm = self.high.diff()
        minus_dm = self.low.diff() * -1
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Make sure only one DM is positive when both are positive
        plus_dm[(plus_dm > 0) & (minus_dm > 0)] = 0
        minus_dm[(plus_dm > 0) & (minus_dm > 0)] = 0
        
        # Calculate smoothed values
        atr = true_range.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        # Normalize to 0-1
        adx_normalized = adx / 100
        
        return adx_normalized.fillna(0)
    
    @property
    def mfi_norm(self, period: int = 14) -> pd.Series:
        """
        MFI (Money Flow Index) Normalized (0-1)
        
        Args:
            period (int): Period untuk MFI calculation (default: 14)
            
        Returns:
            pd.Series: MFI normalized (0-1)
        """
        # Calculate Typical Price
        typical_price = (self.high + self.low + self.close) / 3
        
        # Calculate Money Flow
        money_flow = typical_price * self.volume
        
        # Determine positive and negative money flow
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        # Calculate Money Flow Ratio
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        # Avoid division by zero
        money_flow_ratio = positive_mf / (negative_mf + 1e-10)
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        # Normalize to 0-1
        mfi_normalized = mfi / 100
        
        return mfi_normalized.fillna(0.5)
    
    def get_all_indicators(self, 
                          stoch_period: int = 14, stoch_smooth: int = 3,
                          bb_period: int = 20, bb_std: float = 2.0,
                          rsi_period: int = 14,
                          macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                          adx_period: int = 14,
                          mfi_period: int = 14) -> pd.DataFrame:
        """
        Get all indicators in one DataFrame
        
        Returns:
            pd.DataFrame: DataFrame dengan semua indicators normalized (0-1)
        """
        # Temporary store parameters
        result = pd.DataFrame(index=self.data.index)
        
        # Calculate each indicator with custom parameters
        result['Stochastic_K_norm'] = self._calculate_stochastic(stoch_period, stoch_smooth)
        result['BB_Position'] = self._calculate_bb_position(bb_period, bb_std)
        result['RSI_norm'] = self._calculate_rsi(rsi_period)
        result['MACD_norm'] = self._calculate_macd(macd_fast, macd_slow, macd_signal)
        result['ADX_norm'] = self._calculate_adx(adx_period)
        result['MFI_norm'] = self._calculate_mfi(mfi_period)
        
        return result
    
    # Helper methods for custom parameters
    def _calculate_stochastic(self, period: int, smooth_k: int) -> pd.Series:
        lowest_low = self.low.rolling(window=period).min()
        highest_high = self.high.rolling(window=period).max()
        k_percent = 100 * (self.close - lowest_low) / (highest_high - lowest_low)
        k_smooth = k_percent.rolling(window=smooth_k).mean()
        return (k_smooth / 100).fillna(0)
    
    def _calculate_bb_position(self, period: int, std_dev: float) -> pd.Series:
        sma = self.close.rolling(window=period).mean()
        std = self.close.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        bb_position = (self.close - lower_band) / (upper_band - lower_band)
        return bb_position.clip(0, 1).fillna(0.5)
    
    def _calculate_rsi(self, period: int) -> pd.Series:
        delta = self.close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return (rsi / 100).fillna(0.5)
    
    def _calculate_macd(self, fast: int, slow: int, signal: int) -> pd.Series:
        ema_fast = self.close.ewm(span=fast).mean()
        ema_slow = self.close.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        rolling_min = histogram.rolling(window=50).min()
        rolling_max = histogram.rolling(window=50).max()
        range_diff = rolling_max - rolling_min
        range_diff = range_diff.replace(0, 1)
        
        macd_normalized = (histogram - rolling_min) / range_diff
        return macd_normalized.clip(0, 1).fillna(0.5)
    
    def _calculate_adx(self, period: int) -> pd.Series:
        tr1 = self.high - self.low
        tr2 = abs(self.high - self.close.shift(1))
        tr3 = abs(self.low - self.close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        plus_dm = self.high.diff()
        minus_dm = self.low.diff() * -1
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        atr = true_range.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return (adx / 100).fillna(0)
    
    def _calculate_mfi(self, period: int) -> pd.Series:
        typical_price = (self.high + self.low + self.close) / 3
        money_flow = typical_price * self.volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        money_flow_ratio = positive_mf / (negative_mf + 1e-10)
        mfi = 100 - (100 / (1 + money_flow_ratio))
        return (mfi / 100).fillna(0.5)

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class SupportResistance:
    """
    Improved LuxAlgo/ChartPrime-based Support & Resistance Detection Library
    
    Enhanced with:
    - Multi-ATR validation system
    - Dynamic level updates
    - Trend-aware break detection
    - Real-time level strength tracking
    - ATR-based minimum break distance
    
    Returns same output format untuk compatibility dengan existing test code.
    """
    
    def __init__(self, left_bars: int = 15, right_bars: int = 15, volume_threshold: float = 25.0):
        """
        Initialize improved Support & Resistance detector
        
        Args:
            left_bars (int): Jumlah bars sebelum pivot untuk validation (default: 15)
            right_bars (int): Jumlah bars setelah pivot untuk validation (default: 15)  
            volume_threshold (float): Threshold volume oscillator untuk break confirmation (default: 25.0)
        """
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.volume_threshold = volume_threshold
        
        # Multi-ATR system (inspired by ChartPrime)
        self.trend_atr_period = 25      # For trend detection (dari Pine Script)
        self.level_atr_period = 200     # For level strength validation (dari Pine Script)
        
        # Dynamic level tracking
        self.current_trend = None
        self.trend_start_idx = None
        self.dynamic_support = None
        self.dynamic_resistance = None
        
        # Level strength tracking
        self.level_touches = {}
        self.level_ages = {}
        
        # Trend tracking variables (dari Pine Script logic)
        self.trend_high = None
        self.trend_low = None
        self.trend_high_idx = None
        self.trend_low_idx = None
    
    def calculate_multi_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict:
        """
        Calculate multiple ATR periods untuk different purposes
        
        Returns:
            Dict: {'trend_atr': Series, 'level_atr': Series}
        """
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Multiple ATR calculations
        trend_atr = true_range.rolling(window=self.trend_atr_period).mean()
        level_atr = true_range.rolling(window=self.level_atr_period).mean()
        
        return {
            'trend_atr': trend_atr,
            'level_atr': level_atr
        }
    
    def detect_trend_changes(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Detect trend changes using improved supertrend-like logic (dari Pine Script)
        
        Returns:
            pd.Series: Trend direction (1 for up, -1 for down)
        """
        atr_data = self.calculate_multi_atr(high, low, close)
        trend_atr = atr_data['trend_atr']
        
        # Supertrend calculation dengan factor dari Pine Script
        hl2 = (high + low) / 2
        factor = 4.0  # Factor dari Pine Script
        
        # Calculate basic bands
        basic_upper = hl2 + (factor * trend_atr)
        basic_lower = hl2 - (factor * trend_atr)
        
        # Final bands dengan persistence
        final_upper = pd.Series(index=close.index, dtype=float)
        final_lower = pd.Series(index=close.index, dtype=float)
        
        for i in range(len(close)):
            if pd.isna(trend_atr.iloc[i]):
                if i > 0:
                    final_upper.iloc[i] = final_upper.iloc[i-1]
                    final_lower.iloc[i] = final_lower.iloc[i-1]
                continue
            
            # Upper band logic
            if i == 0:
                final_upper.iloc[i] = basic_upper.iloc[i]
            else:
                if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
                    final_upper.iloc[i] = basic_upper.iloc[i]
                else:
                    final_upper.iloc[i] = final_upper.iloc[i-1]
            
            # Lower band logic
            if i == 0:
                final_lower.iloc[i] = basic_lower.iloc[i]
            else:
                if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
                    final_lower.iloc[i] = basic_lower.iloc[i]
                else:
                    final_lower.iloc[i] = final_lower.iloc[i-1]
        
        # Trend direction calculation dengan improved logic
        trend = pd.Series(index=close.index, dtype=int)
        supertrend = pd.Series(index=close.index, dtype=float)
        
        for i in range(len(close)):
            if pd.isna(final_upper.iloc[i]) or pd.isna(final_lower.iloc[i]):
                trend.iloc[i] = trend.iloc[i-1] if i > 0 else 1
                continue
            
            if i == 0:
                trend.iloc[i] = 1 if close.iloc[i] > final_upper.iloc[i] else -1
            else:
                # Trend change logic dari Pine Script
                if trend.iloc[i-1] == 1:
                    trend.iloc[i] = 1 if close.iloc[i] > final_lower.iloc[i] else -1
                else:
                    trend.iloc[i] = -1 if close.iloc[i] < final_upper.iloc[i] else 1
            
            # Set supertrend value
            supertrend.iloc[i] = final_lower.iloc[i] if trend.iloc[i] == 1 else final_upper.iloc[i]
        
        # Update dynamic levels when trend changes
        self._update_dynamic_levels(high, low, trend)
        
        return trend.fillna(1)
    
    def _update_dynamic_levels(self, high: pd.Series, low: pd.Series, trend: pd.Series):
        """
        Update dynamic support/resistance levels during trend (dari Pine Script logic)
        """
        for i in range(1, len(trend)):
            # Trend change detection
            if trend.iloc[i] != trend.iloc[i-1]:
                if trend.iloc[i] == 1:  # Changed to uptrend
                    # Reset tracking for new uptrend
                    self.trend_low = low.iloc[i]
                    self.trend_low_idx = i
                    self.trend_high = None
                else:  # Changed to downtrend
                    # Reset tracking for new downtrend
                    self.trend_high = high.iloc[i]
                    self.trend_high_idx = i
                    self.trend_low = None
            else:
                # Update levels during trend
                if trend.iloc[i] == 1:  # During uptrend
                    if self.trend_low is None or low.iloc[i] < self.trend_low:
                        self.trend_low = low.iloc[i]
                        self.trend_low_idx = i
                else:  # During downtrend
                    if self.trend_high is None or high.iloc[i] > self.trend_high:
                        self.trend_high = high.iloc[i]
                        self.trend_high_idx = i
    
    def find_pivot_highs(self, highs: pd.Series) -> pd.Series:
        """
        Simple pivot highs detection (Pine Script style)
        
        Args:
            highs (pd.Series): Series of high prices dengan datetime index
            
        Returns:
            pd.Series: Series berisi pivot highs, NaN untuk non-pivot points
        """
        pivot_highs = pd.Series(index=highs.index, dtype=float)
        
        for i in range(self.left_bars, len(highs) - self.right_bars):
            current_high = highs.iloc[i]
            
            # Check if current high is greater than all highs on the left
            is_highest_left = all(highs.iloc[i-j] < current_high for j in range(1, self.left_bars + 1))
            
            # Check if current high is greater than all highs on the right
            is_highest_right = all(highs.iloc[i+j] < current_high for j in range(1, self.right_bars + 1))
            
            # If both conditions are true, it's a pivot high
            if is_highest_left and is_highest_right:
                pivot_highs.iloc[i] = current_high
        
        return pivot_highs.dropna()
    
    def find_pivot_lows(self, lows: pd.Series) -> pd.Series:
        """
        Simple pivot lows detection (Pine Script style)
        
        Args:
            lows (pd.Series): Series of low prices dengan datetime index
            
        Returns:
            pd.Series: Series berisi pivot lows, NaN untuk non-pivot points
        """
        pivot_lows = pd.Series(index=lows.index, dtype=float)
        
        for i in range(self.left_bars, len(lows) - self.right_bars):
            current_low = lows.iloc[i]
            
            # Check if current low is less than all lows on the left
            is_lowest_left = all(lows.iloc[i-j] > current_low for j in range(1, self.left_bars + 1))
            
            # Check if current low is less than all lows on the right
            is_lowest_right = all(lows.iloc[i+j] > current_low for j in range(1, self.right_bars + 1))
            
            # If both conditions are true, it's a pivot low
            if is_lowest_left and is_lowest_right:
                pivot_lows.iloc[i] = current_low
        
        return pivot_lows.dropna()
    
    def calculate_level_strength(self, level: float, high: pd.Series, low: pd.Series, 
                               atr: pd.Series) -> float:
        """
        Calculate level strength berdasarkan touches dan age
        
        Args:
            level (float): Support/resistance level
            high/low (pd.Series): Price data
            atr (pd.Series): ATR for tolerance calculation
            
        Returns:
            float: Level strength score (0-1)
        """
        if len(atr.dropna()) == 0:
            return 0.5
        
        avg_atr = atr.dropna().mean()
        tolerance = avg_atr * 0.3  # Tighter tolerance untuk better accuracy
        
        # Count touches dengan weighted scoring
        touches = 0
        perfect_touches = 0
        
        for i in range(len(high)):
            # Check if price touched the level
            high_distance = abs(high.iloc[i] - level)
            low_distance = abs(low.iloc[i] - level)
            
            if high_distance <= tolerance:
                touches += 1
                if high_distance <= tolerance * 0.5:  # Perfect touch
                    perfect_touches += 1
            elif low_distance <= tolerance:
                touches += 1
                if low_distance <= tolerance * 0.5:  # Perfect touch
                    perfect_touches += 1
        
        # Weighted touch strength
        touch_score = touches / 10.0  # Normalize to max 10 touches
        perfect_score = perfect_touches / 5.0  # Perfect touches more valuable
        
        # Combined score dengan weighting
        strength = (touch_score * 0.6 + perfect_score * 0.4)
        
        return min(strength, 1.0)
    
    def calculate_volume_oscillator(self, volume: pd.Series) -> pd.Series:
        """
        FIXED: Calculate volume oscillator dengan proper error handling
        
        Args:
            volume (pd.Series): Series of volume data (numeric)
            
        Returns:
            pd.Series: Volume oscillator dalam persentage
        """
        # Validate input
        if not isinstance(volume, pd.Series):
            raise ValueError("Volume must be pandas Series")
        
        if volume.dtype not in [np.float64, np.int64, np.float32, np.int32]:
            raise ValueError(f"Volume must be numeric, got {volume.dtype}")
        
        # Calculate EMAs dengan error handling
        try:
            ema5 = volume.ewm(span=5, adjust=False).mean()
            ema10 = volume.ewm(span=10, adjust=False).mean()
        except Exception as e:
            # Fallback to simple moving average
            ema5 = volume.rolling(window=5).mean()
            ema10 = volume.rolling(window=10).mean()
        
        # Calculate oscillator dengan division by zero protection
        ema10_safe = ema10.replace(0, np.nan)
        volume_osc = 100 * (ema5 - ema10) / ema10_safe
        
        return volume_osc.fillna(0)
    
    def get_levels(self, high: pd.Series, low: pd.Series, volume: pd.Series, 
                   close: pd.Series) -> Dict:
        """
        Enhanced support and resistance levels dengan dynamic updates
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices  
            volume (pd.Series): Volume data
            close (pd.Series): Close prices
            
        Returns:
            Dict: Same format as original untuk compatibility
        """
        # Calculate multi-ATR
        atr_data = self.calculate_multi_atr(high, low, close)
        level_atr = atr_data['level_atr']
        
        # Detect trend untuk context
        trend = self.detect_trend_changes(high, low, close)
        current_trend = trend.iloc[-1]
        
        # Find pivots dengan enhanced detection
        pivot_highs = self.find_pivot_highs(high)
        pivot_lows = self.find_pivot_lows(low)
        
        # Include dynamic levels from trend tracking
        if self.trend_high is not None and len(pivot_highs) > 0:
            # Add dynamic resistance if not already in pivots
            if self.trend_high not in pivot_highs.values:
                pivot_highs = pd.concat([pivot_highs, 
                                       pd.Series([self.trend_high], 
                                               index=[high.index[self.trend_high_idx]])])
        
        if self.trend_low is not None and len(pivot_lows) > 0:
            # Add dynamic support if not already in pivots
            if self.trend_low not in pivot_lows.values:
                pivot_lows = pd.concat([pivot_lows, 
                                      pd.Series([self.trend_low], 
                                            index=[low.index[self.trend_low_idx]])])
        
        # Enhanced fallback dengan trend context
        if len(pivot_highs) == 0:
            # Use trend-aware lookback dengan Pine Script inspired logic
            if current_trend == 1:
                lookback = min(100, len(high))
                recent_high = high.tail(lookback).max()
            else:
                lookback = min(50, len(high))
                recent_high = high.tail(lookback).max()
            pivot_highs = pd.Series([recent_high], index=[high.index[-1]])
            
        if len(pivot_lows) == 0:
            if current_trend == -1:
                lookback = min(100, len(low))
                recent_low = low.tail(lookback).min()
            else:
                lookback = min(50, len(low))
                recent_low = low.tail(lookback).min()
            pivot_lows = pd.Series([recent_low], index=[low.index[-1]])
        
        current_price = close.iloc[-1]
        
        # Enhanced nearest level selection dengan strength weighting
        supports_below = [s for s in pivot_lows.values if s <= current_price]
        resistances_above = [r for r in pivot_highs.values if r >= current_price]
        
        if supports_below:
            # Weight by recency, strength, and trend alignment
            support_scores = []
            for support in supports_below:
                strength = self.calculate_level_strength(support, high, low, level_atr)
                distance_factor = 1 - abs(current_price - support) / current_price
                
                # Trend alignment bonus
                trend_bonus = 0.2 if current_trend == 1 else 0
                
                # Recency bonus (dynamic levels get bonus)
                recency_bonus = 0.3 if (self.trend_low is not None and 
                                      abs(support - self.trend_low) < level_atr.iloc[-1] * 0.1) else 0
                
                score = strength * 0.4 + distance_factor * 0.3 + trend_bonus + recency_bonus
                support_scores.append((support, score))
            
            nearest_support = max(support_scores, key=lambda x: x[1])[0]
        else:
            nearest_support = min(pivot_lows.values)
        
        if resistances_above:
            # Weight by recency, strength, and trend alignment
            resistance_scores = []
            for resistance in resistances_above:
                strength = self.calculate_level_strength(resistance, high, low, level_atr)
                distance_factor = 1 - abs(resistance - current_price) / current_price
                
                # Trend alignment bonus
                trend_bonus = 0.2 if current_trend == -1 else 0
                
                # Recency bonus (dynamic levels get bonus)
                recency_bonus = 0.3 if (self.trend_high is not None and 
                                      abs(resistance - self.trend_high) < level_atr.iloc[-1] * 0.1) else 0
                
                score = strength * 0.4 + distance_factor * 0.3 + trend_bonus + recency_bonus
                resistance_scores.append((resistance, score))
            
            nearest_resistance = min(resistance_scores, key=lambda x: x[1])[0]
        else:
            nearest_resistance = max(pivot_highs.values)
        
        # Calculate distances (same format as original)
        distance_to_support = ((current_price - nearest_support) / current_price) * 100
        distance_to_resistance = ((nearest_resistance - current_price) / current_price) * 100
        
        return {
            'distance_to_support': distance_to_support,
            'distance_to_resistance': distance_to_resistance,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance
        }
    
    def detect_breaks(self, close: pd.Series, volume: pd.Series, levels_data: Dict) -> List[Dict]:
        """
        Enhanced break detection dengan multi-factor validation
        
        Args:
            close (pd.Series): Close prices
            volume (pd.Series): Volume data (FIXED: proper parameter)
            levels_data (Dict): Output dari get_levels()
            
        Returns:
            List[Dict]: Same format as original untuk compatibility
        """
        # Calculate volume oscillator (FIXED)
        volume_osc = self.calculate_volume_oscillator(volume)
        
        # Calculate ATR untuk minimum break distance
        high = close  # Simplified for break detection
        low = close
        atr_data = self.calculate_multi_atr(high, low, close)
        level_atr = atr_data['level_atr']
        
        # Get trend context
        trend = self.detect_trend_changes(high, low, close)
        
        nearest_support = levels_data['nearest_support']
        nearest_resistance = levels_data['nearest_resistance']
        
        breaks = []
        
        for i in range(1, len(close)):
            current_close = close.iloc[i]
            prev_close = close.iloc[i-1]
            current_timestamp = close.index[i]
            current_trend = trend.iloc[i] if i < len(trend) else 1
            
            # Get volume oscillator value
            vol_osc_now = volume_osc.iloc[i] if i < len(volume_osc) else 0
            
            # ATR-based minimum break distance (tighter requirement)
            current_atr = level_atr.iloc[i] if i < len(level_atr) else 0
            min_break_distance = current_atr * 0.3 if not pd.isna(current_atr) else 0
            
            # Enhanced support break detection
            if prev_close >= nearest_support and current_close < nearest_support:
                break_distance = abs(current_close - nearest_support)
                
                # Multi-factor confirmation dengan adjusted weights
                volume_confirmed = vol_osc_now > self.volume_threshold
                distance_confirmed = break_distance >= min_break_distance
                trend_aligned = current_trend == -1  # Support break in downtrend
                
                # Check sustained break (close must stay below support)
                sustained_break = current_close < (nearest_support - min_break_distance * 0.5)
                
                # Enhanced confirmation scoring
                confirmation_score = 0
                if volume_confirmed:
                    confirmation_score += 1
                if distance_confirmed:
                    confirmation_score += 0.8
                if trend_aligned:
                    confirmation_score += 0.7
                if sustained_break:
                    confirmation_score += 0.5
                
                final_confirmed = confirmation_score >= 2.0  # Higher threshold
                
                breaks.append({
                    'type': 'support_break',
                    'timestamp': current_timestamp,
                    'price': current_close,
                    'level': nearest_support,
                    'volume_osc': vol_osc_now,
                    'confirmed': final_confirmed
                })
            
            # Enhanced resistance break detection
            if prev_close <= nearest_resistance and current_close > nearest_resistance:
                break_distance = abs(current_close - nearest_resistance)
                
                # Multi-factor confirmation dengan adjusted weights
                volume_confirmed = vol_osc_now > self.volume_threshold
                distance_confirmed = break_distance >= min_break_distance
                trend_aligned = current_trend == 1  # Resistance break in uptrend
                
                # Check sustained break (close must stay above resistance)
                sustained_break = current_close > (nearest_resistance + min_break_distance * 0.5)
                
                # Enhanced confirmation scoring
                confirmation_score = 0
                if volume_confirmed:
                    confirmation_score += 1
                if distance_confirmed:
                    confirmation_score += 0.8
                if trend_aligned:
                    confirmation_score += 0.7
                if sustained_break:
                    confirmation_score += 0.5
                
                final_confirmed = confirmation_score >= 2.0  # Higher threshold
                
                breaks.append({
                    'type': 'resistance_break',
                    'timestamp': current_timestamp,
                    'price': current_close,
                    'level': nearest_resistance,
                    'volume_osc': vol_osc_now,
                    'confirmed': final_confirmed
                })
        
        return breaks

class FibonacciTrend:
    """
    Exact copy of original FibonacciTrendDebug logic with TechnicalIndicators bug fixed
    """
    
    def __init__(self, trend_factor=4.0, atr_period=25, extend_bars=15):
        self.trend_factor = trend_factor
        self.atr_period = atr_period
        self.extend_bars = extend_bars
        
        # Fibonacci levels (exact same)
        self.fib_levels = {
            '0': 0.0,
            '236': 0.236,
            '382': 0.382,
            '500': 0.500,
            '618': 0.618,
            '786': 0.786,
            '100': 1.0
        }
        
        # Storage (exact same)
        self.current_fib = {
            'trend_direction': None,
            'fib_high': None,
            'fib_low': None,
            'fib_high_idx': None,
            'fib_low_idx': None,
            'levels': {},
            'active': False,
            'trend_start_idx': None,
            'debug_info': []
        }
        
        self.debug_logs = []
    
    def log_debug(self, message):
        """Add debug message"""
        self.debug_logs.append(message)
        print(f"  🔍 {message}")
    
    def atr(self, high, low, close, period):
        """ATR calculation - exact same as TechnicalIndicators"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def calculate_supertrend(self, high, low, close):
        """Calculate Supertrend - FIXED NaN Bug"""
        self.log_debug("Calculating Supertrend...")
        
        # Calculate ATR
        atr = self.atr(high, low, close, period=self.atr_period)
        
        # Debug ATR values
        atr_valid = atr.dropna()
        if len(atr_valid) > 0:
            self.log_debug(f"ATR stats: min={atr_valid.min():.2f}, max={atr_valid.max():.2f}, mean={atr_valid.mean():.2f}")
        else:
            self.log_debug("❌ ATR calculation failed - all NaN values")
            return pd.Series(index=close.index, dtype=float), pd.Series(index=close.index, dtype=int)
        
        # Calculate basic upper and lower bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (self.trend_factor * atr)
        lower_band = hl2 - (self.trend_factor * atr)
        
        # Debug bands
        self.log_debug(f"Price range: {close.min():.2f} - {close.max():.2f}")
        upper_valid = upper_band.dropna()
        lower_valid = lower_band.dropna()
        
        if len(upper_valid) > 0 and len(lower_valid) > 0:
            self.log_debug(f"Upper band range: {upper_valid.min():.2f} - {upper_valid.max():.2f}")
            self.log_debug(f"Lower band range: {lower_valid.min():.2f} - {lower_valid.max():.2f}")
        
        # Initialize
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)
        
        # Find first valid ATR index
        first_valid_idx = atr.first_valid_index()
        if first_valid_idx is None:
            self.log_debug("❌ No valid ATR values found")
            return supertrend, direction
        
        first_valid_pos = close.index.get_loc(first_valid_idx)
        
        # Initialize from first valid ATR position
        supertrend.iloc[first_valid_pos] = upper_band.iloc[first_valid_pos]
        direction.iloc[first_valid_pos] = 1
        
        # Debug first values
        self.log_debug(f"Initial (idx {first_valid_pos}): price={close.iloc[first_valid_pos]:.2f}, supertrend={supertrend.iloc[first_valid_pos]:.2f}, direction={direction.iloc[first_valid_pos]}")
        
        direction_changes = 0
        
        # Calculate supertrend starting from first valid position
        for i in range(first_valid_pos + 1, len(close)):
            # Skip if ATR is NaN
            if pd.isna(atr.iloc[i]) or pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
                # Copy previous values
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
                continue
            
            current_upper = upper_band.iloc[i]
            current_lower = lower_band.iloc[i]
            prev_close = close.iloc[i-1]
            current_close = close.iloc[i]
            prev_supertrend = supertrend.iloc[i-1]
            prev_direction = direction.iloc[i-1]
            
            # Skip if previous values are NaN
            if pd.isna(prev_supertrend) or pd.isna(prev_direction):
                supertrend.iloc[i] = current_upper
                direction.iloc[i] = 1
                continue
            
            # Update bands
            if current_upper < prev_supertrend or prev_close > prev_supertrend:
                final_upper = current_upper
            else:
                final_upper = prev_supertrend
                
            if current_lower > prev_supertrend or prev_close < prev_supertrend:
                final_lower = current_lower
            else:
                final_lower = prev_supertrend
            
            # Determine trend direction
            if prev_direction == 1 and current_close < final_lower:
                direction.iloc[i] = -1
                supertrend.iloc[i] = final_upper
                direction_changes += 1
                self.log_debug(f"TREND CHANGE {direction_changes} at bar {i}: UP→DOWN, price={current_close:.2f}, lower_band={final_lower:.2f}")
            elif prev_direction == -1 and current_close > final_upper:
                direction.iloc[i] = 1
                supertrend.iloc[i] = final_lower
                direction_changes += 1
                self.log_debug(f"TREND CHANGE {direction_changes} at bar {i}: DOWN→UP, price={current_close:.2f}, upper_band={final_upper:.2f}")
            else:
                direction.iloc[i] = prev_direction
                if direction.iloc[i] == 1:
                    supertrend.iloc[i] = final_lower
                else:
                    supertrend.iloc[i] = final_upper
            
            # Debug every 100 bars
            if i % 100 == 0:
                self.log_debug(f"Bar {i}: price={current_close:.2f}, supertrend={supertrend.iloc[i]:.2f}, direction={direction.iloc[i]}, changes={direction_changes}")
        
        # Count trend changes
        trend_changes = (direction != direction.shift(1)).sum()
        self.log_debug(f"Supertrend calculated: {trend_changes} trend changes (manual count: {direction_changes})")
        
        # Final direction stats
        up_count = (direction == 1).sum()
        down_count = (direction == -1).sum()
        nan_count = direction.isna().sum()
        self.log_debug(f"Direction distribution: UP={up_count}, DOWN={down_count}, NaN={nan_count}")
        
        return supertrend, direction
    
    def update_fibonacci_levels(self, current_idx, high, low, close, direction, atr):
        """EXACT COPY of original fibonacci update logic"""
        if current_idx <= 0:
            return
            
        current_high = high.iloc[current_idx]
        current_low = low.iloc[current_idx]
        current_close = close.iloc[current_idx]
        current_atr = atr.iloc[current_idx] if current_idx < len(atr) else atr.iloc[-1]
        trend = direction.iloc[current_idx]
        prev_trend = direction.iloc[current_idx-1]
        
        # Check if trend changed
        trend_changed = trend != prev_trend
        
        if trend_changed:
            trend_name = "UPTREND" if trend == 1 else "DOWNTREND"
            self.log_debug(f"TREND CHANGE at bar {current_idx}: {prev_trend} → {trend} ({trend_name})")
            self.log_debug(f"  Price at change: H:{current_high:.2f} L:{current_low:.2f} C:{current_close:.2f}")
            
            # Reset fibonacci state
            self.current_fib = {
                'trend_direction': trend,
                'fib_high': current_high,
                'fib_low': current_low,
                'fib_high_idx': current_idx,
                'fib_low_idx': current_idx,
                'trend_start_idx': current_idx,
                'levels': {},
                'active': True,
                'debug_info': [f"Trend started at {current_idx}: {trend_name}"]
            }
            
            self.log_debug(f"  Initial Fib Range: {current_low:.2f} - {current_high:.2f}")
            
        elif self.current_fib['active'] and trend == prev_trend:
            # Update swing points during trend continuation
            old_high = self.current_fib['fib_high']
            old_low = self.current_fib['fib_low']
            
            high_updated = False
            low_updated = False
            
            # Update high
            if current_high > self.current_fib['fib_high']:
                self.current_fib['fib_high'] = current_high
                self.current_fib['fib_high_idx'] = current_idx
                high_updated = True
                self.log_debug(f"  NEW HIGH at {current_idx}: {old_high:.2f} → {current_high:.2f}")
            
            # Update low
            if current_low < self.current_fib['fib_low']:
                self.current_fib['fib_low'] = current_low
                self.current_fib['fib_low_idx'] = current_idx
                low_updated = True
                self.log_debug(f"  NEW LOW at {current_idx}: {old_low:.2f} → {current_low:.2f}")
            
            # Log range update
            if high_updated or low_updated:
                new_range = self.current_fib['fib_high'] - self.current_fib['fib_low']
                self.log_debug(f"  Updated Range: {self.current_fib['fib_low']:.2f} - {self.current_fib['fib_high']:.2f} (Size: {new_range:.2f})")
        
        # Calculate fibonacci levels
        if self.current_fib['active']:
            levels = self._calculate_fib_prices_debug(trend, current_idx)
    
    def _calculate_fib_prices_debug(self, trend_direction, current_idx):
        """EXACT COPY of original fibonacci calculation"""
        fib_high = self.current_fib['fib_high']
        fib_low = self.current_fib['fib_low']
        
        if pd.isna(fib_high) or pd.isna(fib_low):
            self.log_debug(f"  ❌ Invalid fib range: high={fib_high}, low={fib_low}")
            return {}
        
        fib_range = fib_high - fib_low
        
        if fib_range <= 0:
            self.log_debug(f"  ❌ Invalid fib range: {fib_range:.2f} (high={fib_high:.2f}, low={fib_low:.2f})")
            return {}
        
        self.log_debug(f"  Calculating Fib for trend {trend_direction} at bar {current_idx}")
        self.log_debug(f"    Swing High: {fib_high:.2f} (bar {self.current_fib['fib_high_idx']})")
        self.log_debug(f"    Swing Low: {fib_low:.2f} (bar {self.current_fib['fib_low_idx']})")
        self.log_debug(f"    Range: {fib_range:.2f}")
        
        # EXACT ORIGINAL LOGIC: Proper fibonacci assignment
        if trend_direction == 1:  # UPTREND
            # For UPTREND: 0% = swing LOW, 100% = swing HIGH
            val_0 = fib_low     # 0% level (start of move)
            val_100 = fib_high  # 100% level (end of move)
            self.log_debug(f"    UPTREND: 0%={val_0:.2f} (LOW), 100%={val_100:.2f} (HIGH)")
        else:  # DOWNTREND
            # For DOWNTREND: 0% = swing HIGH, 100% = swing LOW  
            val_0 = fib_high    # 0% level (start of move)
            val_100 = fib_low   # 100% level (end of move)
            self.log_debug(f"    DOWNTREND: 0%={val_0:.2f} (HIGH), 100%={val_100:.2f} (LOW)")
        
        # Calculate each fibonacci level
        levels = {}
        for name, percentage in self.fib_levels.items():
            if name == '0':
                levels[name] = val_0
            elif name == '100':
                levels[name] = val_100
            else:
                # Linear interpolation between 0% and 100%
                fib_price = val_0 + (val_100 - val_0) * percentage
                levels[name] = fib_price
        
        # Log all calculated levels
        self.log_debug("    Calculated Fibonacci Levels:")
        for name in ['0', '236', '382', '500', '618', '786', '100']:
            if name in levels:
                percentage = self.fib_levels.get(name, 0.0)
                price = levels[name]
                self.log_debug(f"      {percentage:.3f} ({name}): {price:.2f}")
        
        self.current_fib['levels'] = levels
        return levels
    
    def analyze_fibonacci_trend_debug(self, df):
        """EXACT COPY of original analysis method"""
        print(f"\n🎯 Fibonacci Trend DEBUG Analysis:")
        print("=" * 50)
        
        # Calculate Supertrend
        supertrend, direction = self.calculate_supertrend(df['high'], df['low'], df['close'])
        
        # Calculate ATR
        atr = self.atr(df['high'], df['low'], df['close'], period=200)
        
        # Process each bar with detailed logging
        print(f"\n📊 Processing {len(df)} bars...")
        
        for i in range(1, len(df)):
            # Only log every 50 bars to avoid spam
            if i % 50 == 0:
                print(f"\n--- Processing bar {i}/{len(df)} ---")
            
            self.update_fibonacci_levels(i, df['high'], df['low'], df['close'], direction, atr)
        
        # Final summary
        print(f"\n📈 FINAL FIBONACCI STATE:")
        print("-" * 30)
        if self.current_fib['active']:
            trend_name = "UPTREND" if self.current_fib['trend_direction'] == 1 else "DOWNTREND"
            print(f"Trend: {trend_name}")
            print(f"Swing High: {self.current_fib['fib_high']:.2f} (bar {self.current_fib['fib_high_idx']})")
            print(f"Swing Low: {self.current_fib['fib_low']:.2f} (bar {self.current_fib['fib_low_idx']})")
            print(f"Range: {self.current_fib['fib_high'] - self.current_fib['fib_low']:.2f}")
            
            if self.current_fib['levels']:
                print(f"\nFibonacci Levels:")
                for name in ['0', '236', '382', '500', '618', '786', '100']:
                    if name in self.current_fib['levels']:
                        percentage = self.fib_levels.get(name, 0.0)
                        price = self.current_fib['levels'][name]
                        print(f"  {percentage:.3f}: {price:.2f}")
        
        return {
            'supertrend': supertrend,
            'direction': direction,
            'atr': atr,
            'current_fib': self.current_fib,
            'debug_logs': self.debug_logs
        }
# Example Usage
if __name__ == "__main__":
    # Sample data creation (replace with your actual data)
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    np.random.seed(42)
    
    # Generate sample OHLCV data
    close_prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)
    high_prices = close_prices + np.random.rand(1000) * 2
    low_prices = close_prices - np.random.rand(1000) * 2
    open_prices = close_prices + np.random.randn(1000) * 0.5
    volume = np.random.randint(1000000, 10000000, 1000)
    
    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    }).set_index('Date')
    
    # Initialize Technical Indicators
    ti = TechnicalIndicators(sample_data)
    
    # Get individual indicators
    print("Individual Indicators:")
    print(f"Stochastic K (last 5): {ti.stochastic_k_norm.tail()}")
    print(f"BB Position (last 5): {ti.bb_position.tail()}")
    print(f"RSI (last 5): {ti.rsi_norm.tail()}")
    print(f"MACD (last 5): {ti.macd_norm.tail()}")
    print(f"ADX (last 5): {ti.adx_norm.tail()}")
    print(f"MFI (last 5): {ti.mfi_norm.tail()}")
    
    # Get all indicators at once
    all_indicators = ti.get_all_indicators()
    print(f"\nAll Indicators Shape: {all_indicators.shape}")
    print(f"\nAll Indicators (last 5 rows):")
    print(all_indicators.tail())
    
    # Check for NaN values
    print(f"\nNaN count per indicator:")
    print(all_indicators.isnull().sum())
    
    # Value range check (should all be 0-1)
    print(f"\nValue ranges (min-max):")
    for col in all_indicators.columns:
        print(f"{col}: {all_indicators[col].min():.3f} - {all_indicators[col].max():.3f}")