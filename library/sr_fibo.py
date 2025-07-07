# technical_levels.py
"""
Technical Analysis Library for Dynamic Stop Loss and Take Profit
Includes Support/Resistance Detection and Fibonacci Calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SupportResistanceAnalyzer:
    """Analyze and detect support/resistance levels from price data"""
    
    def __init__(self, lookback_period: int = 100, merge_threshold: float = 0.001):
        """
        Initialize S/R Analyzer
        
        Args:
            lookback_period: Number of candles to analyze
            merge_threshold: Percentage threshold to merge nearby levels (0.001 = 0.1%)
        """
        self.lookback_period = lookback_period
        self.merge_threshold = merge_threshold

        self.min_sl_distance_pct = 0.005  # Minimal 0.3% dari S/R
        self.min_sl_distance_atr = 0.5    # Minimal 0.5x ATR dari S/R

        
        
    def detect_pivot_points(self, df: pd.DataFrame, window: int = 5) -> Dict[str, List[float]]:
        """
        Detect pivot highs and lows
        
        Args:
            df: DataFrame with OHLC data
            window: Number of candles on each side to confirm pivot
            
        Returns:
            Dict with 'highs' and 'lows' lists
        """
        pivots = {'highs': [], 'lows': []}
        
        if len(df) < window * 2 + 1:
            return pivots
            
        # Detect pivot highs
        for i in range(window, len(df) - window):
            high = df['high'].iloc[i]
            
            # Check if it's a pivot high
            is_pivot_high = True
            for j in range(i - window, i + window + 1):
                if j != i and df['high'].iloc[j] >= high:
                    is_pivot_high = False
                    break
                    
            if is_pivot_high:
                pivots['highs'].append(high)
                
        # Detect pivot lows
        for i in range(window, len(df) - window):
            low = df['low'].iloc[i]
            
            # Check if it's a pivot low
            is_pivot_low = True
            for j in range(i - window, i + window + 1):
                if j != i and df['low'].iloc[j] <= low:
                    is_pivot_low = False
                    break
                    
            if is_pivot_low:
                pivots['lows'].append(low)
                
        return pivots
    
    def find_volume_nodes(self, df: pd.DataFrame, bins: int = 20) -> List[float]:
        """
        Find price levels with high volume (Volume Profile)
        
        Args:
            df: DataFrame with OHLC and volume data
            bins: Number of price bins for volume distribution
            
        Returns:
            List of price levels with high volume
        """
        if 'volume' not in df.columns or len(df) < bins:
            return []
            
        # Create price bins
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_bins = np.linspace(price_min, price_max, bins + 1)
        
        # Calculate volume at each price level
        volume_profile = np.zeros(bins)
        
        for _, row in df.iterrows():
            # Distribute volume across the candle's range
            candle_low = row['low']
            candle_high = row['high']
            candle_volume = row['volume']
            
            # Find which bins this candle spans
            low_bin = np.searchsorted(price_bins, candle_low, side='left')
            high_bin = np.searchsorted(price_bins, candle_high, side='right')
            
            # Distribute volume evenly across bins
            if high_bin > low_bin:
                volume_per_bin = candle_volume / (high_bin - low_bin)
                for i in range(max(0, low_bin), min(bins, high_bin)):
                    volume_profile[i] += volume_per_bin
        
        # Find peaks in volume profile (high volume nodes)
        mean_volume = volume_profile.mean()
        std_volume = volume_profile.std()
        threshold = mean_volume + std_volume
        
        high_volume_levels = []
        for i, vol in enumerate(volume_profile):
            if vol > threshold:
                # Use the middle of the bin as the level
                level = (price_bins[i] + price_bins[i + 1]) / 2
                high_volume_levels.append(level)
                
        return high_volume_levels
    
    def get_historical_levels(self, df: pd.DataFrame, min_touches: int = 3) -> List[Tuple[float, int]]:
        """
        Find levels that have been tested multiple times
        
        Args:
            df: DataFrame with OHLC data
            min_touches: Minimum number of touches to consider a level
            
        Returns:
            List of (price_level, touch_count) tuples
        """
        levels = []
        
        # Get all potential levels from highs and lows
        price_points = []
        price_points.extend(df['high'].tolist())
        price_points.extend(df['low'].tolist())
        
        # Round prices to reduce noise
        decimal_places = 5  # Adjust based on asset
        price_points = [round(p, decimal_places) for p in price_points]
        
        # Count touches for each level
        from collections import Counter
        level_counts = Counter(price_points)
        
        # Filter levels with minimum touches
        for level, count in level_counts.items():
            if count >= min_touches:
                levels.append((level, count))
                
        # Sort by touch count (strongest first)
        levels.sort(key=lambda x: x[1], reverse=True)
        
        return levels
    
    def calculate_strength(self, level: float, df: pd.DataFrame, 
                     tolerance: float = 0.0005) -> float:
        """Calculate strength of a S/R level with better error handling"""
        try:
            if len(df) < 2:
                return 0.0
                
            touches = 0
            bounces = 0
            volume_at_level = 0
            
            level_high = level * (1 + tolerance)
            level_low = level * (1 - tolerance)
            
            for i in range(1, len(df)):
                try:
                    row = df.iloc[i]
                    prev_row = df.iloc[i-1]
                    
                    # Check if price touched the level
                    if (level_low <= row['high'] <= level_high or 
                        level_low <= row['low'] <= level_high):
                        touches += 1
                        
                        # Check if price bounced off the level
                        if 'close' in prev_row and 'close' in row:
                            if prev_row['close'] > level and row['low'] <= level and row['close'] > level:
                                bounces += 1  # Bounce from below
                            elif prev_row['close'] < level and row['high'] >= level and row['close'] < level:
                                bounces += 1  # Bounce from above
                        
                        if 'volume' in row:
                            volume_at_level += row['volume']
                except Exception as e:
                    self.logger.debug(f"Error processing row {i}: {e}")
                    continue
            
            # Calculate strength components with bounds
            touch_score = min(touches / 10, 1.0)  # Normalize to 0-1
            bounce_score = min(bounces / 5, 1.0)   # Normalize to 0-1
            recency_score = 0
            
            # Check if level was tested recently
            try:
                recent_data = df.tail(20)
                for _, row in recent_data.iterrows():
                    if level_low <= row['high'] <= level_high or level_low <= row['low'] <= level_high:
                        recency_score = 1
                        break
            except:
                pass
            
            # Weighted strength score
            strength = (touch_score * 0.4 + bounce_score * 0.4 + recency_score * 0.2)
            
            return min(max(strength, 0.0), 1.0)  # Ensure between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Error calculating strength for level {level}: {e}")
            return 0.5  # Default medium strength
    
    def merge_nearby_levels(self, levels: List[float]) -> List[float]:
        """
        Merge levels that are too close to each other
        
        Args:
            levels: List of price levels
            
        Returns:
            Merged list of levels
        """
        if not levels:
            return []
            
        levels_sorted = sorted(levels)
        merged = [levels_sorted[0]]
        
        for level in levels_sorted[1:]:
            # Check if close to the last merged level
            if abs(level - merged[-1]) / merged[-1] > self.merge_threshold:
                merged.append(level)
            else:
                # Update the last level with weighted average
                merged[-1] = (merged[-1] + level) / 2
                
        return merged
    
    def get_all_levels(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Get all support and resistance levels with their strengths
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dict with 'support' and 'resistance' lists
        """
        current_price = df['close'].iloc[-1]
        
        # Get levels from different methods
        pivots = self.detect_pivot_points(df)
        volume_levels = self.find_volume_nodes(df)
        historical = self.get_historical_levels(df)
        
        # Combine all levels
        all_levels = []
        all_levels.extend(pivots['highs'])
        all_levels.extend(pivots['lows'])
        all_levels.extend(volume_levels)
        all_levels.extend([level for level, _ in historical])
        
        # Remove duplicates and merge nearby
        unique_levels = list(set(all_levels))
        merged_levels = self.merge_nearby_levels(unique_levels)
        
        # Calculate strength for each level
        levels_with_strength = []
        for level in merged_levels:
            strength = self.calculate_strength(level, df)
            levels_with_strength.append({
                'price': level,
                'strength': strength,
                'distance_pct': abs(level - current_price) / current_price * 100
            })
        
        # Separate into support and resistance
        support = [l for l in levels_with_strength if l['price'] < current_price]
        resistance = [l for l in levels_with_strength if l['price'] > current_price]
        
        # Sort by distance from current price
        support.sort(key=lambda x: x['price'], reverse=True)  # Nearest first
        resistance.sort(key=lambda x: x['price'])  # Nearest first
        
        return {
            'support': support,
            'resistance': resistance,
            'current_price': current_price
        }


class FibonacciCalculator:
    """Calculate Fibonacci retracement and extension levels"""
    
    def __init__(self):
        # Standard Fibonacci ratios
        self.retracement_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.extension_ratios = [1.272, 1.618, 2.0, 2.618]
        
    def find_swing_points(self, df: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
        """
        Find recent swing high and low
        
        Args:
            df: DataFrame with OHLC data
            lookback: Number of candles to look back
            
        Returns:
            Dict with 'swing_high' and 'swing_low'
        """
        if len(df) < lookback:
            lookback = len(df)
            
        recent_data = df.tail(lookback)
        
        swing_high = recent_data['high'].max()
        swing_low = recent_data['low'].min()
        
        # Find the actual candles for more precision
        high_idx = recent_data['high'].idxmax()
        low_idx = recent_data['low'].idxmin()
        
        return {
            'swing_high': swing_high,
            'swing_low': swing_low,
            'high_index': high_idx,
            'low_index': low_idx,
            'is_uptrend': high_idx > low_idx  # High came after low
        }
    
    def calculate_retracement(self, swing_high: float, swing_low: float) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels
        
        Args:
            swing_high: Swing high price
            swing_low: Swing low price
            
        Returns:
            Dict of retracement levels
        """
        diff = swing_high - swing_low
        levels = {}
        
        # Add 0% and 100% levels
        levels['0.0'] = swing_low
        levels['1.0'] = swing_high
        
        # Calculate retracement levels
        for ratio in self.retracement_ratios:
            level = swing_high - (diff * ratio)
            levels[str(ratio)] = level
            
        return levels
    
    def calculate_extension(self, swing_high: float, swing_low: float, 
                          is_uptrend: bool = True) -> Dict[str, float]:
        """
        Calculate Fibonacci extension levels
        
        Args:
            swing_high: Swing high price
            swing_low: Swing low price
            is_uptrend: True for uptrend, False for downtrend
            
        Returns:
            Dict of extension levels
        """
        diff = swing_high - swing_low
        levels = {}
        
        if is_uptrend:
            # Extensions above swing high
            for ratio in self.extension_ratios:
                level = swing_low + (diff * ratio)
                levels[str(ratio)] = level
        else:
            # Extensions below swing low
            for ratio in self.extension_ratios:
                level = swing_high - (diff * ratio)
                levels[str(ratio)] = level
                
        return levels
    
    def get_confluence_zones(self, levels_list: List[Dict[str, float]], 
                           tolerance: float = 0.001) -> List[Dict]:
        """
        Find zones where multiple Fibonacci levels confluence
        
        Args:
            levels_list: List of Fibonacci level dictionaries
            tolerance: Percentage tolerance for confluence (0.001 = 0.1%)
            
        Returns:
            List of confluence zones with strength
        """
        # Collect all levels
        all_levels = []
        for levels in levels_list:
            for ratio, price in levels.items():
                all_levels.append(price)
                
        # Find confluences
        confluence_zones = []
        checked = set()
        
        for i, level1 in enumerate(all_levels):
            if i in checked:
                continue
                
            zone_levels = [level1]
            checked.add(i)
            
            # Find other levels within tolerance
            for j, level2 in enumerate(all_levels):
                if j != i and j not in checked:
                    if abs(level2 - level1) / level1 <= tolerance:
                        zone_levels.append(level2)
                        checked.add(j)
                        
            # Create confluence zone if multiple levels found
            if len(zone_levels) > 1:
                confluence_zones.append({
                    'price': np.mean(zone_levels),
                    'strength': len(zone_levels),
                    'levels': zone_levels
                })
                
        # Sort by strength
        confluence_zones.sort(key=lambda x: x['strength'], reverse=True)
        
        return confluence_zones
        
    def get_all_fibonacci_levels(self, df: pd.DataFrame, 
                               multiple_swings: bool = True) -> Dict:
        """
        Get all Fibonacci levels from recent price action
        
        Args:
            df: DataFrame with OHLC data
            multiple_swings: Analyze multiple swing points
            
        Returns:
            Dict with retracements, extensions, and confluences
        """
        results = {
            'retracements': {},
            'extensions': {},
            'confluences': [],
            'current_price': df['close'].iloc[-1]
        }
        
        # Find primary swing points
        primary_swing = self.find_swing_points(df, lookback=50)
        
        # Calculate primary levels
        primary_retracement = self.calculate_retracement(
            primary_swing['swing_high'], 
            primary_swing['swing_low']
        )
        primary_extension = self.calculate_extension(
            primary_swing['swing_high'], 
            primary_swing['swing_low'],
            primary_swing['is_uptrend']
        )
        
        results['retracements']['primary'] = primary_retracement
        results['extensions']['primary'] = primary_extension
        
        # Find additional swings if requested
        if multiple_swings and len(df) > 100:
            # Secondary swing (longer timeframe)
            secondary_swing = self.find_swing_points(df, lookback=100)
            secondary_retracement = self.calculate_retracement(
                secondary_swing['swing_high'], 
                secondary_swing['swing_low']
            )
            secondary_extension = self.calculate_extension(
                secondary_swing['swing_high'], 
                secondary_swing['swing_low'],
                secondary_swing['is_uptrend']
            )
            
            results['retracements']['secondary'] = secondary_retracement
            results['extensions']['secondary'] = secondary_extension
            
            # Find confluences
            results['confluences'] = self.get_confluence_zones([
                primary_retracement,
                primary_extension,
                secondary_retracement,
                secondary_extension
            ])
        
        return results


class DynamicLevelSelector:
    """Select optimal SL and TP levels based on S/R and Fibonacci"""
    
    def __init__(self, max_risk_pct: float = 0.02, min_rr_ratio: float = 1.5):
        """
        Initialize level selector
        
        Args:
            max_risk_pct: Maximum risk per trade (0.02 = 2%)
            min_rr_ratio: Minimum risk/reward ratio
        """
        self.max_risk_pct = max_risk_pct
        self.min_rr_ratio = min_rr_ratio
        self.sr_analyzer = SupportResistanceAnalyzer()
        self.fib_calc = FibonacciCalculator()
        self.logger = logging.getLogger(__name__)


    def get_buffer_analysis(self, sr_levels: Dict, volatility: float) -> Dict:
        """
        Analyze buffer requirements for all S/R levels
        
        Returns:
            Dict with buffer analysis for each level
        """
        analysis = {
            'volatility': volatility,
            'volatility_category': 'normal',
            'supports': [],
            'resistances': []
        }
        
        # Categorize volatility
        if volatility < 0.005:
            analysis['volatility_category'] = 'low'
        elif volatility > 0.02:
            analysis['volatility_category'] = 'high'
        elif volatility > 0.03:
            analysis['volatility_category'] = 'extreme'
        
        # Analyze supports
        for support in sr_levels.get('support', [])[:5]:
            buffer = self.calculate_dynamic_buffer(support['strength'], volatility)
            analysis['supports'].append({
                'price': support['price'],
                'strength': support['strength'],
                'buffer_pct': buffer * 100,
                'buffer_price': support['price'] * buffer
            })
        
        # Analyze resistances
        for resistance in sr_levels.get('resistance', [])[:5]:
            buffer = self.calculate_dynamic_buffer(resistance['strength'], volatility)
            analysis['resistances'].append({
                'price': resistance['price'],
                'strength': resistance['strength'],
                'buffer_pct': buffer * 100,
                'buffer_price': resistance['price'] * buffer
            })
        
        return analysis

    def calculate_dynamic_buffer(self, sr_strength: float, volatility: float) -> float:
        """
        Enhanced buffer calculation with better volatility handling
        
        Args:
            sr_strength: Strength of S/R level (0.0 to 1.0)
            volatility: Current volatility (ATR as percentage)
        
        Returns:
            Buffer percentage (e.g., 0.003 = 0.3%)
        """
        # Enhanced base buffer configuration
        base_buffer = 0.003  # Increased from 0.002 to 0.3% base
        min_buffer = 0.002   # Increased from 0.001 to 0.2% minimum
        max_buffer = 0.015   # Increased from 0.01 to 1.5% maximum
        
        # Enhanced volatility categories
        if volatility < 0.005:  # Low volatility
            vol_category = "low"
            vol_factor = 0.8
        elif volatility < 0.01:  # Normal volatility
            vol_category = "normal"
            vol_factor = 1.0
        elif volatility < 0.02:  # Medium volatility
            vol_category = "medium"
            vol_factor = 1.5
        elif volatility < 0.03:  # High volatility
            vol_category = "high"
            vol_factor = 2.0
        else:  # Extreme volatility
            vol_category = "extreme"
            vol_factor = 3.0
        
        # Enhanced strength adjustment
        # Strong S/R (0.8-1.0) = smaller buffer
        # Medium S/R (0.5-0.8) = normal buffer
        # Weak S/R (0-0.5) = larger buffer
        if sr_strength >= 0.8:
            strength_factor = 0.8
        elif sr_strength >= 0.5:
            strength_factor = 1.0
        else:
            strength_factor = 1.5
        
        # Calculate final buffer
        buffer = base_buffer * vol_factor * strength_factor
        
        # Apply minimum based on volatility category
        if vol_category == "high" or vol_category == "extreme":
            buffer = max(buffer, base_buffer * 1.5)  # Ensure larger buffer in high vol
        
        # Apply limits
        buffer = max(buffer, min_buffer)
        buffer = min(buffer, max_buffer)
        
        # Log calculation details
        if hasattr(self, 'logger'):
            self.logger.debug(f"Buffer calculation: volatility={volatility:.4f} ({vol_category}), "
                            f"strength={sr_strength:.2f}, vol_factor={vol_factor:.1f}, "
                            f"strength_factor={strength_factor:.1f}, final_buffer={buffer:.4f}")
        
        return buffer
        
    def select_stop_loss(self, entry_price: float, position_type: str,
                    sr_levels: Dict, fib_levels: Dict,
                    volatility: float = 0.01) -> Dict:
        """
        Enhanced stop loss selection with Fibonacci 0.618 rule
        
        Args:
            entry_price: Entry price
            position_type: 'LONG' or 'SHORT'
            sr_levels: Support/Resistance levels
            fib_levels: Fibonacci levels
            volatility: Current volatility (ATR as percentage)
            
        Returns:
            Dict with SL info
        """
        candidates = []
        
        # Configuration
        min_sl_distance_pct = getattr(self, 'min_sl_distance_pct', 0.003)
        min_sl_distance_atr = getattr(self, 'min_sl_distance_atr', 0.5)
        
        # Calculate minimal distances
        min_distance_price = entry_price * min_sl_distance_pct
        min_distance_atr = entry_price * volatility * min_sl_distance_atr
        minimal_distance = max(min_distance_price, min_distance_atr)
        
        # For Fibonacci 0.618 rule
        fib_618_price = None
        if 'retracements' in fib_levels and 'primary' in fib_levels['retracements']:
            if '0.618' in fib_levels['retracements']['primary']:
                fib_618_price = fib_levels['retracements']['primary']['0.618']
        
        if position_type == 'LONG':
            # Check Fibonacci 0.618 rule first
            if fib_618_price and fib_618_price < entry_price:
                # Calculate buffer for Fibonacci level
                fib_buffer = self.calculate_dynamic_buffer(0.8, volatility)  # High strength for Fib
                sl_price = fib_618_price * (1 - fib_buffer)
                
                # Ensure minimal distance
                if entry_price - sl_price < minimal_distance:
                    sl_price = entry_price - minimal_distance
                
                risk_pct = (entry_price - sl_price) / entry_price
                
                if risk_pct <= self.max_risk_pct:
                    candidates.append({
                        'price': sl_price,
                        'type': 'fib_0.618_rule',
                        'strength': 0.9,  # High priority
                        'risk_pct': risk_pct,
                        'sr_price': fib_618_price,
                        'buffer_pct': fib_buffer,
                        'distance_from_entry': entry_price - sl_price,
                        'priority': 1  # Highest priority
                    })
                    
                    if hasattr(self, 'logger'):
                        self.logger.info(f"📐 Fibonacci 0.618 rule applied: SL below ${fib_618_price:.5f}")
            
            # Then check support levels
            supports = sr_levels.get('support', [])
            for support in supports[:5]:
                sr_price = support['price']
                sr_strength = support['strength']
                
                # Skip if support too close to entry
                distance_to_entry = entry_price - sr_price
                if distance_to_entry < minimal_distance:
                    continue
                
                # Enhanced buffer calculation
                buffer = self.calculate_dynamic_buffer(sr_strength, volatility)
                
                # Place SL below support with buffer
                sl_price = sr_price * (1 - buffer)
                
                # Ensure minimal distance from entry
                if entry_price - sl_price < minimal_distance:
                    sl_price = entry_price - minimal_distance
                
                risk_pct = (entry_price - sl_price) / entry_price
                
                if risk_pct <= self.max_risk_pct:
                    candidates.append({
                        'price': sl_price,
                        'type': 'support',
                        'strength': support['strength'],
                        'risk_pct': risk_pct,
                        'sr_price': sr_price,
                        'buffer_pct': buffer,
                        'distance_from_entry': entry_price - sl_price,
                        'priority': 2  # Second priority
                    })
            
            # Additional Fibonacci levels (lower priority)
            for fib_type in ['primary', 'secondary']:
                if fib_type in fib_levels.get('retracements', {}):
                    fib_retrace = fib_levels['retracements'][fib_type]
                    
                    # Check other retracements
                    for ratio in ['0.786', '0.5', '0.382']:
                        if ratio in fib_retrace and ratio != '0.618':  # Skip 0.618 as already handled
                            fib_price = fib_retrace[ratio]
                            if fib_price < entry_price:
                                if entry_price - fib_price < minimal_distance:
                                    continue
                                
                                buffer = self.calculate_dynamic_buffer(0.7, volatility)
                                sl_price = fib_price * (1 - buffer)
                                
                                if entry_price - sl_price < minimal_distance:
                                    sl_price = entry_price - minimal_distance
                                
                                risk_pct = (entry_price - sl_price) / entry_price
                                
                                if risk_pct <= self.max_risk_pct:
                                    candidates.append({
                                        'price': sl_price,
                                        'type': f'fib_{ratio}',
                                        'strength': 0.7,
                                        'risk_pct': risk_pct,
                                        'distance_from_entry': entry_price - sl_price,
                                        'priority': 3  # Lower priority
                                    })
        
        else:  # SHORT
            # For SHORT positions, similar logic but reversed
            if fib_618_price and fib_618_price > entry_price:
                fib_buffer = self.calculate_dynamic_buffer(0.8, volatility)
                sl_price = fib_618_price * (1 + fib_buffer)
                
                if sl_price - entry_price < minimal_distance:
                    sl_price = entry_price + minimal_distance
                
                risk_pct = (sl_price - entry_price) / entry_price
                
                if risk_pct <= self.max_risk_pct:
                    candidates.append({
                        'price': sl_price,
                        'type': 'fib_0.618_rule',
                        'strength': 0.9,
                        'risk_pct': risk_pct,
                        'sr_price': fib_618_price,
                        'buffer_pct': fib_buffer,
                        'distance_from_entry': sl_price - entry_price,
                        'priority': 1
                    })
            
            # Check resistance levels
            resistances = sr_levels.get('resistance', [])
            for resistance in resistances[:5]:
                sr_price = resistance['price']
                sr_strength = resistance['strength']
                
                distance_to_entry = sr_price - entry_price
                if distance_to_entry < minimal_distance:
                    continue
                
                buffer = self.calculate_dynamic_buffer(sr_strength, volatility)
                sl_price = sr_price * (1 + buffer)
                
                if sl_price - entry_price < minimal_distance:
                    sl_price = entry_price + minimal_distance
                
                risk_pct = (sl_price - entry_price) / entry_price
                
                if risk_pct <= self.max_risk_pct:
                    candidates.append({
                        'price': sl_price,
                        'type': 'resistance',
                        'strength': resistance['strength'],
                        'risk_pct': risk_pct,
                        'sr_price': sr_price,
                        'buffer_pct': buffer,
                        'distance_from_entry': sl_price - entry_price,
                        'priority': 2
                    })
        
        # Select best candidate with priority consideration
        if candidates:
            # Sort by priority first, then by strength, then by risk
            candidates.sort(key=lambda x: (x['priority'], -x['strength'], x['risk_pct']))
            selected = candidates[0]
            
            if hasattr(self, 'logger'):
                self.logger.info(f"🎯 SL Selected: {selected['type']} @ ${selected['price']:.5f}")
                self.logger.info(f"   Risk: {selected['risk_pct']*100:.2f}%")
                self.logger.info(f"   Distance from entry: ${selected['distance_from_entry']:.5f}")
                if 'sr_price' in selected:
                    self.logger.info(f"   Technical level: ${selected['sr_price']:.5f}")
                    self.logger.info(f"   Buffer: {selected['buffer_pct']*100:.2f}%")
        else:
            # Fallback to fixed percentage
            if position_type == 'LONG':
                default_sl = entry_price * (1 - self.max_risk_pct)
                min_dist_sl = entry_price - minimal_distance
                sl_price = max(default_sl, min_dist_sl)
            else:
                default_sl = entry_price * (1 + self.max_risk_pct)
                min_dist_sl = entry_price + minimal_distance
                sl_price = min(default_sl, min_dist_sl)
                
            risk_pct = abs(sl_price - entry_price) / entry_price
            
            selected = {
                'price': sl_price,
                'type': 'fixed_pct',
                'strength': 0.5,
                'risk_pct': risk_pct,
                'distance_from_entry': abs(sl_price - entry_price),
                'priority': 99
            }
            
            if hasattr(self, 'logger'):
                self.logger.warning(f"⚠️ Using fallback SL @ ${sl_price:.5f} (no suitable technical levels)")
        
        return selected
    
    def select_take_profits(self, entry_price: float, position_type: str,
                          stop_loss: float, sr_levels: Dict, 
                          fib_levels: Dict) -> Dict:
        """
        Select TP1 and TP2 levels
        
        Args:
            entry_price: Entry price
            position_type: 'LONG' or 'SHORT'
            stop_loss: Selected stop loss price
            sr_levels: Support/Resistance levels
            fib_levels: Fibonacci levels
            
        Returns:
            Dict with TP1 and TP2 info
        """
        risk_amount = abs(entry_price - stop_loss)
        tp_candidates = []
        
        if position_type == 'LONG':
            # Look for resistance levels above entry
            resistances = sr_levels.get('resistance', [])
            
            for resistance in resistances[:10]:
                tp_price = resistance['price']
                reward = tp_price - entry_price
                rr_ratio = reward / risk_amount
                
                if rr_ratio >= 0.5:  # At least 0.5:1 for TP1
                    tp_candidates.append({
                        'price': tp_price,
                        'type': 'resistance',
                        'strength': resistance['strength'],
                        'rr_ratio': rr_ratio
                    })
            
            # Check Fibonacci extensions
            for fib_type in ['primary', 'secondary']:
                if fib_type in fib_levels.get('extensions', {}):
                    fib_ext = fib_levels['extensions'][fib_type]
                    
                    for ratio in ['1.272', '1.618', '2.0']:
                        if ratio in fib_ext:
                            fib_price = fib_ext[ratio]
                            if fib_price > entry_price:
                                reward = fib_price - entry_price
                                rr_ratio = reward / risk_amount
                                
                                tp_candidates.append({
                                    'price': fib_price,
                                    'type': f'fib_ext_{ratio}',
                                    'strength': 0.8,
                                    'rr_ratio': rr_ratio
                                })
        
        else:  # SHORT
            # Look for support levels below entry
            supports = sr_levels.get('support', [])
            
            for support in supports[:10]:
                tp_price = support['price']
                reward = entry_price - tp_price
                rr_ratio = reward / risk_amount
                
                if rr_ratio >= 0.5:
                    tp_candidates.append({
                        'price': tp_price,
                        'type': 'support',
                        'strength': support['strength'],
                        'rr_ratio': rr_ratio
                    })
        
        # Select TP1 and TP2
        result = {'tp1': None, 'tp2': None}
        
        if tp_candidates:
            # Sort by RR ratio
            tp_candidates.sort(key=lambda x: x['rr_ratio'])
            
            # TP1: First good level (1:1 to 1.5:1)
            for tp in tp_candidates:
                if 0.8 <= tp['rr_ratio'] <= 1.5:
                    result['tp1'] = tp
                    break
            
            # If no TP1 in ideal range, take the nearest
            if not result['tp1'] and tp_candidates:
                result['tp1'] = tp_candidates[0]
            
            # TP2: Look for 2:1 or better
            for tp in tp_candidates:
                if tp['rr_ratio'] >= 2.0:
                    result['tp2'] = tp
                    break
            
            # If no TP2 found, use extension
            if not result['tp2'] and result['tp1']:
                if position_type == 'LONG':
                    tp2_price = entry_price + (risk_amount * 2.5)
                else:
                    tp2_price = entry_price - (risk_amount * 2.5)
                    
                result['tp2'] = {
                    'price': tp2_price,
                    'type': 'rr_extension',
                    'strength': 0.6,
                    'rr_ratio': 2.5
                }
        
        # Fallback if no levels found
        if not result['tp1']:
            if position_type == 'LONG':
                tp1_price = entry_price + (risk_amount * 1.0)
            else:
                tp1_price = entry_price - (risk_amount * 1.0)
                
            result['tp1'] = {
                'price': tp1_price,
                'type': 'fixed_rr',
                'strength': 0.5,
                'rr_ratio': 1.0
            }
        
        if not result['tp2']:
            if position_type == 'LONG':
                tp2_price = entry_price + (risk_amount * 2.0)
            else:
                tp2_price = entry_price - (risk_amount * 2.0)
                
            result['tp2'] = {
                'price': tp2_price,
                'type': 'fixed_rr',
                'strength': 0.5,
                'rr_ratio': 2.0
            }
        
        return result
    
    def validate_risk_reward(self, entry: float, stop_loss: float,
                           take_profits: Dict) -> Dict:
        """
        Validate risk/reward ratios
        
        Args:
            entry: Entry price
            stop_loss: Stop loss price
            take_profits: Dict with TP1 and TP2
            
        Returns:
            Validation results
        """
        risk = abs(entry - stop_loss)
        
        validation = {
            'valid': True,
            'risk_amount': risk,
            'risk_pct': risk / entry * 100,
            'issues': []
        }
        
        # Check risk percentage
        if validation['risk_pct'] > self.max_risk_pct * 100:
            validation['valid'] = False
            validation['issues'].append(f"Risk too high: {validation['risk_pct']:.1f}%")
        
        # Check TP1
        if take_profits['tp1']:
            tp1_rr = take_profits['tp1']['rr_ratio']
            if tp1_rr < 0.5:
                validation['issues'].append(f"TP1 RR too low: {tp1_rr:.1f}")
        
        # Check TP2
        if take_profits['tp2']:
            tp2_rr = take_profits['tp2']['rr_ratio']
            if tp2_rr < self.min_rr_ratio:
                validation['issues'].append(f"TP2 RR below minimum: {tp2_rr:.1f}")
        
        # Overall RR check
        avg_rr = (take_profits['tp1']['rr_ratio'] + take_profits['tp2']['rr_ratio']) / 2
        validation['avg_rr_ratio'] = avg_rr
        
        if avg_rr < self.min_rr_ratio:
            validation['valid'] = False
            validation['issues'].append(f"Average RR below minimum: {avg_rr:.1f}")
        
        return validation
    
    def adjust_for_volatility(self, levels: Dict, atr_pct: float) -> Dict:
        """
        Adjust levels based on current volatility
        
        Args:
            levels: Dict with SL and TP levels
            atr_pct: ATR as percentage of price
            
        Returns:
            Adjusted levels
        """
        adjusted = levels.copy()
        
        # High volatility adjustments
        if atr_pct > 0.02:  # High volatility (>2% ATR)
            volatility_factor = min(atr_pct / 0.01, 2.0)  # Cap at 2x
            
            # Widen stop loss
            if 'stop_loss' in adjusted:
                sl_distance = abs(adjusted['entry'] - adjusted['stop_loss']['price'])
                if adjusted['position_type'] == 'LONG':
                    adjusted['stop_loss']['price'] = adjusted['entry'] - (sl_distance * volatility_factor)
                else:
                    adjusted['stop_loss']['price'] = adjusted['entry'] + (sl_distance * volatility_factor)
            
            # Adjust take profits proportionally
            if 'take_profits' in adjusted:
                for tp_key in ['tp1', 'tp2']:
                    if adjusted['take_profits'][tp_key]:
                        tp_distance = abs(adjusted['entry'] - adjusted['take_profits'][tp_key]['price'])
                        if adjusted['position_type'] == 'LONG':
                            adjusted['take_profits'][tp_key]['price'] = adjusted['entry'] + (tp_distance * volatility_factor)
                        else:
                            adjusted['take_profits'][tp_key]['price'] = adjusted['entry'] - (tp_distance * volatility_factor)
        
        return adjusted
    
    def get_optimal_levels(self, df: pd.DataFrame, entry_price: float,
                          position_type: str, atr_pct: Optional[float] = None) -> Dict:
        """
        Main method to get all optimal levels
        
        Args:
            df: DataFrame with OHLC data
            entry_price: Entry price
            position_type: 'LONG' or 'SHORT'
            atr_pct: ATR as percentage (will calculate if None)
            
        Returns:
            Complete level analysis
        """
        # Calculate ATR if not provided
        if atr_pct is None:
            atr = self._calculate_atr(df)
            atr_pct = atr / df['close'].iloc[-1]
        
        # Get S/R levels
        sr_levels = self.sr_analyzer.get_all_levels(df)
        
        # Get Fibonacci levels
        fib_levels = self.fib_calc.get_all_fibonacci_levels(df)
        
        # Select stop loss
        stop_loss = self.select_stop_loss(entry_price, position_type, 
                                         sr_levels, fib_levels, atr_pct)
        
        # Select take profits
        take_profits = self.select_take_profits(entry_price, position_type,
                                               stop_loss['price'], sr_levels, fib_levels)
        
        # Package results
        results = {
            'entry': entry_price,
            'position_type': position_type,
            'stop_loss': stop_loss,
            'take_profits': take_profits,
            'sr_levels': sr_levels,
            'fib_levels': fib_levels,
            'atr_pct': atr_pct
        }
        
        # Validate
        validation = self.validate_risk_reward(entry_price, stop_loss['price'], take_profits)
        results['validation'] = validation
        
        # Adjust for volatility
        if validation['valid']:
            results = self.adjust_for_volatility(results, atr_pct)
        
        return results
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range with better error handling"""
        try:
            if len(df) < period + 1:
                self.logger.warning(f"Insufficient data for ATR calculation: {len(df)} < {period + 1}")
                return 0.01  # Default 1% volatility
            
            # Ensure we have required columns
            required_cols = ['high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"Missing required columns for ATR. Available: {df.columns.tolist()}")
                return 0.01
            
            # Calculate True Range components
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Initialize TR array
            tr = np.zeros(len(df))
            
            # First value is just high-low
            tr[0] = high[0] - low[0]
            
            # Calculate remaining TR values
            for i in range(1, len(df)):
                hl = high[i] - low[i]
                hc = abs(high[i] - close[i-1])
                lc = abs(low[i] - close[i-1])
                tr[i] = max(hl, hc, lc)
            
            # Calculate ATR using exponential moving average
            atr = np.zeros(len(df))
            atr[:period] = np.nan
            
            # Initial ATR is simple average
            atr[period-1] = np.mean(tr[:period])
            
            # Calculate remaining ATR values
            multiplier = 2.0 / (period + 1)
            for i in range(period, len(df)):
                atr[i] = (tr[i] - atr[i-1]) * multiplier + atr[i-1]
            
            # Get final ATR value
            final_atr = atr[-1]
            if np.isnan(final_atr) or final_atr <= 0:
                self.logger.warning("Invalid ATR calculated, using default")
                return 0.01
                
            return final_atr
            
        except Exception as e:
            self.logger.error(f"ATR calculation error: {e}")
            import traceback
            traceback.print_exc()
            return 0.01  # Default 1% volatility


# Helper function for easy usage
def analyze_levels(df: pd.DataFrame, entry_price: float, 
                  position_type: str = 'LONG') -> Dict:
    """
    Quick function to analyze and get optimal levels
    
    Args:
        df: DataFrame with OHLC data
        entry_price: Entry price
        position_type: 'LONG' or 'SHORT'
        
    Returns:
        Dict with all level analysis
    """
    selector = DynamicLevelSelector()
    return selector.get_optimal_levels(df, entry_price, position_type)


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=200, freq='5T')
    sample_data = pd.DataFrame({
        'open': np.random.randn(200).cumsum() + 100,
        'high': np.random.randn(200).cumsum() + 101,
        'low': np.random.randn(200).cumsum() + 99,
        'close': np.random.randn(200).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    # Analyze levels
    current_price = sample_data['close'].iloc[-1]
    analysis = analyze_levels(sample_data, current_price, 'LONG')
    
    print(f"Entry: ${current_price:.2f}")
    print(f"Stop Loss: ${analysis['stop_loss']['price']:.2f} "
          f"({analysis['stop_loss']['type']}, "
          f"Risk: {analysis['stop_loss']['risk_pct']*100:.1f}%)")
    print(f"TP1: ${analysis['take_profits']['tp1']['price']:.2f} "
          f"(RR: {analysis['take_profits']['tp1']['rr_ratio']:.1f})")
    print(f"TP2: ${analysis['take_profits']['tp2']['price']:.2f} "
          f"(RR: {analysis['take_profits']['tp2']['rr_ratio']:.1f})")