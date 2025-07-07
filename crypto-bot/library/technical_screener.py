# library/technical_screener.py
"""
Technical Analysis Library for Crypto Screening
Integrates with existing S/R and Fibonacci analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import time

# Import existing S/R and Fibonacci analyzer
from .sr_fibo import SupportResistanceAnalyzer, FibonacciCalculator, DynamicLevelSelector

logger = logging.getLogger(__name__)


class TechnicalScreener:
    """Technical analysis for crypto screening with multi-timeframe support"""
    
    def __init__(self):
        # Initialize analyzers
        self.sr_analyzer = SupportResistanceAnalyzer(lookback_period=200, merge_threshold=0.001)
        self.fib_calc = FibonacciCalculator()
        
        # Cache for performance
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    def analyze_symbol(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """
        Complete technical analysis for a symbol
        
        Args:
            symbol: Trading symbol
            data_dict: Dictionary with timeframes as keys ('1m', '5m', '30m')
        
        Returns:
            Dict with complete technical analysis
        """
        try:
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'trend_30m': self._analyze_trend(data_dict.get('30m')),
                'setup_5m': self._analyze_setup(data_dict.get('5m')),
                'confirmation_1m': self._analyze_confirmation(data_dict.get('1m')),
                'sr_levels': self._get_multi_tf_sr_levels(data_dict),
                'fibonacci': self._analyze_fibonacci(data_dict),
                'volume_profile': self._analyze_volume_profile(data_dict.get('5m')),
                'market_structure': self._analyze_market_structure(data_dict.get('30m')),
                'technical_score': 0.0
            }
            
            # Calculate technical score
            result['technical_score'] = self._calculate_technical_score(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict:
        """Analyze 30M trend context"""
        if df is None or df.empty:
            return {'bias': 'NEUTRAL', 'strength': 0.0}
        
        try:
            # EMA analysis
            ema_20 = df['close'].ewm(span=20).mean()
            ema_50 = df['close'].ewm(span=50).mean()
            ema_200 = df['close'].ewm(span=200).mean()
            
            current_price = df['close'].iloc[-1]
            
            # Trend bias
            if current_price > ema_20.iloc[-1] > ema_50.iloc[-1] > ema_200.iloc[-1]:
                bias = 'BULLISH'
                strength = 0.8
            elif current_price < ema_20.iloc[-1] < ema_50.iloc[-1] < ema_200.iloc[-1]:
                bias = 'BEARISH' 
                strength = 0.8
            else:
                bias = 'NEUTRAL'
                strength = 0.3
            
            # RSI trend
            rsi = self._calculate_rsi(df, 14)
            rsi_trend = 'UP' if rsi.iloc[-1] > rsi.iloc[-5] else 'DOWN'
            
            # MACD trend
            macd_line, signal_line = self._calculate_macd(df)
            macd_trend = 'BULLISH' if macd_line.iloc[-1] > signal_line.iloc[-1] else 'BEARISH'
            
            return {
                'bias': bias,
                'strength': strength,
                'ema_alignment': bias,
                'rsi': rsi.iloc[-1],
                'rsi_trend': rsi_trend,
                'macd_trend': macd_trend,
                'price_vs_ema200': ((current_price - ema_200.iloc[-1]) / ema_200.iloc[-1]) * 100
            }
            
        except Exception as e:
            logger.error(f"Trend analysis error: {e}")
            return {'bias': 'NEUTRAL', 'strength': 0.0}
    
    def _analyze_setup(self, df: pd.DataFrame) -> Dict:
        """Analyze 5M setup quality"""
        if df is None or df.empty:
            return {'quality': 'POOR', 'score': 0.0}
        
        try:
            # RSI extremes
            rsi = self._calculate_rsi(df, 14)
            current_rsi = rsi.iloc[-1]
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df)
            current_price = df['close'].iloc[-1]
            bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            
            # Volume analysis
            volume_sma = df['volume'].rolling(20).mean()
            volume_ratio = df['volume'].iloc[-1] / volume_sma.iloc[-1]
            
            # Setup scoring
            setup_score = 0.0
            setup_type = None
            
            # Long setup conditions
            if current_rsi < 30 and bb_position < 0.2:
                setup_type = 'OVERSOLD_BOUNCE'
                setup_score = 0.8
            elif current_rsi < 40 and volume_ratio > 1.5 and bb_position < 0.3:
                setup_type = 'VOLUME_REVERSAL_LONG'
                setup_score = 0.7
            
            # Short setup conditions  
            elif current_rsi > 70 and bb_position > 0.8:
                setup_type = 'OVERBOUGHT_REVERSAL'
                setup_score = 0.8
            elif current_rsi > 60 and volume_ratio > 1.5 and bb_position > 0.7:
                setup_type = 'VOLUME_REVERSAL_SHORT'
                setup_score = 0.7
            
            # Breakout setups
            elif bb_position > 1.0 and volume_ratio > 2.0:
                setup_type = 'BULLISH_BREAKOUT'
                setup_score = 0.6
            elif bb_position < 0.0 and volume_ratio > 2.0:
                setup_type = 'BEARISH_BREAKOUT'
                setup_score = 0.6
            else:
                setup_type = 'NO_SETUP'
                setup_score = 0.2
            
            quality = 'EXCELLENT' if setup_score >= 0.7 else 'GOOD' if setup_score >= 0.5 else 'POOR'
            
            return {
                'quality': quality,
                'score': setup_score,
                'type': setup_type,
                'rsi': current_rsi,
                'bb_position': bb_position,
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            logger.error(f"Setup analysis error: {e}")
            return {'quality': 'POOR', 'score': 0.0}
    
    def _analyze_confirmation(self, df: pd.DataFrame) -> Dict:
        """Analyze 1M technical confirmation"""
        if df is None or df.empty:
            return {'status': 'NEUTRAL', 'strength': 0.0}
        
        try:
            # Ultra-fast RSI
            rsi_5 = self._calculate_rsi(df, 5)
            rsi_change = rsi_5.iloc[-1] - rsi_5.iloc[-3]
            
            # Price action
            last_3_closes = df['close'].iloc[-3:].values
            price_momentum = (last_3_closes[-1] - last_3_closes[0]) / last_3_closes[0] * 100
            
            # Volume spike
            volume_spike = df['volume'].iloc[-1] / df['volume'].iloc[-10:].mean()
            
            # Micro structure
            last_high = df['high'].iloc[-5:].max()
            last_low = df['low'].iloc[-5:].min()
            current_price = df['close'].iloc[-1]
            micro_position = (current_price - last_low) / (last_high - last_low) if last_high != last_low else 0.5
            
            # Confirmation logic
            if price_momentum > 0.1 and rsi_change > 5 and volume_spike > 1.5:
                status = 'STRONG_BULLISH'
                strength = 0.8
            elif price_momentum > 0.05 and rsi_change > 0:
                status = 'BULLISH'
                strength = 0.6
            elif price_momentum < -0.1 and rsi_change < -5 and volume_spike > 1.5:
                status = 'STRONG_BEARISH'
                strength = 0.8
            elif price_momentum < -0.05 and rsi_change < 0:
                status = 'BEARISH'
                strength = 0.6
            else:
                status = 'NEUTRAL'
                strength = 0.3
            
            return {
                'status': status,
                'strength': strength,
                'price_momentum': price_momentum,
                'rsi_momentum': rsi_change,
                'volume_spike': volume_spike,
                'micro_position': micro_position
            }
            
        except Exception as e:
            logger.error(f"Confirmation analysis error: {e}")
            return {'status': 'NEUTRAL', 'strength': 0.0}
    
    def _get_multi_tf_sr_levels(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """Get S/R levels from multiple timeframes"""
        sr_levels = {
            '30m': {'support': [], 'resistance': []},
            '5m': {'support': [], 'resistance': []},
            '1m': {'support': [], 'resistance': []}
        }
        
        current_price = None
        
        for tf, df in data_dict.items():
            if df is not None and not df.empty:
                if current_price is None:
                    current_price = df['close'].iloc[-1]
                
                # Get S/R levels using existing analyzer
                levels = self.sr_analyzer.get_all_levels(df)
                
                # Take top 3 supports and resistances
                sr_levels[tf]['support'] = levels['support'][:3]
                sr_levels[tf]['resistance'] = levels['resistance'][:3]
        
        # Find confluence levels (appear in multiple timeframes)
        confluence_levels = self._find_confluence_levels(sr_levels, current_price)
        
        return {
            'levels_by_timeframe': sr_levels,
            'confluence_levels': confluence_levels,
            'current_price': current_price
        }
    
    def _find_confluence_levels(self, sr_levels: Dict, current_price: float) -> Dict:
        """Find S/R levels that appear across multiple timeframes"""
        confluence_threshold = 0.002  # 0.2% price difference
        
        all_supports = []
        all_resistances = []
        
        # Collect all levels with timeframe info
        for tf, levels in sr_levels.items():
            for sup in levels['support']:
                if sup:
                    all_supports.append({'price': sup['price'], 'tf': tf, 'strength': sup['strength']})
            for res in levels['resistance']:
                if res:
                    all_resistances.append({'price': res['price'], 'tf': tf, 'strength': res['strength']})
        
        # Find confluences
        confluence_supports = []
        confluence_resistances = []
        
        # Check supports
        processed = set()
        for i, sup1 in enumerate(all_supports):
            if i in processed:
                continue
            
            cluster = [sup1]
            processed.add(i)
            
            for j, sup2 in enumerate(all_supports[i+1:], i+1):
                if abs(sup1['price'] - sup2['price']) / sup1['price'] < confluence_threshold:
                    cluster.append(sup2)
                    processed.add(j)
            
            if len(cluster) > 1:
                avg_price = np.mean([s['price'] for s in cluster])
                avg_strength = np.mean([s['strength'] for s in cluster])
                timeframes = list(set([s['tf'] for s in cluster]))
                
                confluence_supports.append({
                    'price': avg_price,
                    'strength': avg_strength * len(timeframes),  # Multiply by TF count
                    'timeframes': timeframes,
                    'distance_pct': abs(avg_price - current_price) / current_price * 100
                })
        
        # Similar for resistances
        processed = set()
        for i, res1 in enumerate(all_resistances):
            if i in processed:
                continue
            
            cluster = [res1]
            processed.add(i)
            
            for j, res2 in enumerate(all_resistances[i+1:], i+1):
                if abs(res1['price'] - res2['price']) / res1['price'] < confluence_threshold:
                    cluster.append(res2)
                    processed.add(j)
            
            if len(cluster) > 1:
                avg_price = np.mean([r['price'] for r in cluster])
                avg_strength = np.mean([r['strength'] for r in cluster])
                timeframes = list(set([r['tf'] for r in cluster]))
                
                confluence_resistances.append({
                    'price': avg_price,
                    'strength': avg_strength * len(timeframes),
                    'timeframes': timeframes,
                    'distance_pct': abs(avg_price - current_price) / current_price * 100
                })
        
        # Sort by strength
        confluence_supports.sort(key=lambda x: x['strength'], reverse=True)
        confluence_resistances.sort(key=lambda x: x['strength'], reverse=True)
        
        return {
            'support': confluence_supports[:3],  # Top 3
            'resistance': confluence_resistances[:3]
        }
    
    def _analyze_fibonacci(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze Fibonacci levels"""
        try:
            # Use 30m for major swings
            df_30m = data_dict.get('30m')
            if df_30m is None or df_30m.empty:
                return {}
            
            # Get Fibonacci levels
            fib_data = self.fib_calc.get_all_fibonacci_levels(df_30m, multiple_swings=False)
            
            current_price = df_30m['close'].iloc[-1]
            
            # Find nearest Fibonacci levels
            nearest_fib = None
            min_distance = float('inf')
            
            if 'retracements' in fib_data and 'primary' in fib_data['retracements']:
                for ratio, level in fib_data['retracements']['primary'].items():
                    distance = abs(current_price - level)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_fib = {
                            'level': level,
                            'ratio': ratio,
                            'distance_pct': (distance / current_price) * 100,
                            'type': 'retracement'
                        }
            
            # Check if near golden ratio
            golden_ratios = ['0.382', '0.5', '0.618']
            near_golden = False
            
            if nearest_fib and nearest_fib['ratio'] in golden_ratios and nearest_fib['distance_pct'] < 1.0:
                near_golden = True
            
            return {
                'nearest_level': nearest_fib,
                'near_golden_ratio': near_golden,
                'retracements': fib_data.get('retracements', {}).get('primary', {}),
                'extensions': fib_data.get('extensions', {}).get('primary', {})
            }
            
        except Exception as e:
            logger.error(f"Fibonacci analysis error: {e}")
            return {}
    
    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Analyze volume distribution at price levels"""
        if df is None or df.empty:
            return {}
        
        try:
            # Calculate volume at price levels
            bins = 30
            price_min = df['low'].min()
            price_max = df['high'].max()
            price_bins = np.linspace(price_min, price_max, bins + 1)
            
            volume_profile = np.zeros(bins)
            
            for _, row in df.iterrows():
                candle_low = row['low']
                candle_high = row['high']
                candle_volume = row['volume']
                
                # Find bins this candle spans
                low_bin = np.searchsorted(price_bins, candle_low, side='left')
                high_bin = np.searchsorted(price_bins, candle_high, side='right')
                
                # Distribute volume
                if high_bin > low_bin:
                    volume_per_bin = candle_volume / (high_bin - low_bin)
                    for i in range(max(0, low_bin), min(bins, high_bin)):
                        volume_profile[i] += volume_per_bin
            
            # Find high volume nodes (HVN)
            mean_vol = volume_profile.mean()
            std_vol = volume_profile.std()
            hvn_threshold = mean_vol + std_vol
            
            high_volume_nodes = []
            for i, vol in enumerate(volume_profile):
                if vol > hvn_threshold:
                    price_level = (price_bins[i] + price_bins[i + 1]) / 2
                    high_volume_nodes.append({
                        'price': price_level,
                        'volume': vol,
                        'strength': vol / volume_profile.max()
                    })
            
            # Point of Control (POC) - highest volume price
            poc_index = volume_profile.argmax()
            poc_price = (price_bins[poc_index] + price_bins[poc_index + 1]) / 2
            
            return {
                'poc': poc_price,
                'high_volume_nodes': high_volume_nodes[:5],  # Top 5
                'volume_distribution': {
                    'above_poc': volume_profile[poc_index+1:].sum() / volume_profile.sum(),
                    'below_poc': volume_profile[:poc_index].sum() / volume_profile.sum()
                }
            }
            
        except Exception as e:
            logger.error(f"Volume profile error: {e}")
            return {}
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze market structure (trend, range, breakout)"""
        if df is None or df.empty:
            return {'structure': 'UNDEFINED', 'strength': 0.0}
        
        try:
            # Find swing highs and lows
            window = 10
            highs = []
            lows = []
            
            for i in range(window, len(df) - window):
                # Swing high
                if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
                    highs.append({'index': i, 'price': df['high'].iloc[i]})
                
                # Swing low
                if df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
                    lows.append({'index': i, 'price': df['low'].iloc[i]})
            
            if len(highs) < 2 or len(lows) < 2:
                return {'structure': 'UNDEFINED', 'strength': 0.0}
            
            # Check for trend structure
            last_highs = [h['price'] for h in highs[-3:]]
            last_lows = [l['price'] for l in lows[-3:]]
            
            # Uptrend: Higher highs and higher lows
            if (len(last_highs) >= 2 and all(last_highs[i] < last_highs[i+1] for i in range(len(last_highs)-1)) and
                len(last_lows) >= 2 and all(last_lows[i] < last_lows[i+1] for i in range(len(last_lows)-1))):
                structure = 'UPTREND'
                strength = 0.8
            
            # Downtrend: Lower highs and lower lows
            elif (len(last_highs) >= 2 and all(last_highs[i] > last_highs[i+1] for i in range(len(last_highs)-1)) and
                  len(last_lows) >= 2 and all(last_lows[i] > last_lows[i+1] for i in range(len(last_lows)-1))):
                structure = 'DOWNTREND'
                strength = 0.8
            
            # Range-bound
            else:
                high_range = max(last_highs) - min(last_highs)
                low_range = max(last_lows) - min(last_lows)
                avg_price = df['close'].mean()
                
                if high_range / avg_price < 0.02 and low_range / avg_price < 0.02:
                    structure = 'RANGE'
                    strength = 0.6
                else:
                    structure = 'CHOPPY'
                    strength = 0.3
            
            return {
                'structure': structure,
                'strength': strength,
                'swing_highs': highs[-3:],
                'swing_lows': lows[-3:]
            }
            
        except Exception as e:
            logger.error(f"Market structure error: {e}")
            return {'structure': 'UNDEFINED', 'strength': 0.0}
    
    def _calculate_technical_score(self, analysis: Dict) -> float:
        """Calculate overall technical score"""
        score = 0.0
        
        # Trend score (30%)
        trend = analysis.get('trend_30m', {})
        if trend.get('bias') == 'BULLISH':
            score += 0.3 * trend.get('strength', 0)
        elif trend.get('bias') == 'BEARISH':
            score += -0.3 * trend.get('strength', 0)
        
        # Setup score (30%)
        setup = analysis.get('setup_5m', {})
        setup_score = setup.get('score', 0)
        if 'LONG' in setup.get('type', '') or 'BULLISH' in setup.get('type', ''):
            score += 0.3 * setup_score
        elif 'SHORT' in setup.get('type', '') or 'BEARISH' in setup.get('type', ''):
            score += -0.3 * setup_score
        
        # Confirmation score (20%)
        confirm = analysis.get('confirmation_1m', {})
        if 'BULLISH' in confirm.get('status', ''):
            score += 0.2 * confirm.get('strength', 0)
        elif 'BEARISH' in confirm.get('status', ''):
            score += -0.2 * confirm.get('strength', 0)
        
        # Market structure (10%)
        structure = analysis.get('market_structure', {})
        if structure.get('structure') == 'UPTREND':
            score += 0.1 * structure.get('strength', 0)
        elif structure.get('structure') == 'DOWNTREND':
            score += -0.1 * structure.get('strength', 0)
        
        # Fibonacci bonus (10%)
        fib = analysis.get('fibonacci', {})
        if fib.get('near_golden_ratio'):
            score += 0.1 if score > 0 else -0.1
        
        return np.clip(score, -1.0, 1.0)
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        return macd_line, signal_line
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower


class MarketRegimeDetector:
    """Detect overall market conditions"""
    
    def __init__(self):
        self.btc_weight = 0.6  # BTC influence on market
        
    def detect_regime(self, btc_data: pd.DataFrame, market_breadth: Dict) -> Dict:
        """
        Detect current market regime
        
        Returns:
            Dict with regime type and recommendations
        """
        try:
            # BTC volatility
            btc_volatility = self._calculate_volatility(btc_data)
            
            # BTC trend
            btc_trend = self._get_btc_trend(btc_data)
            
            # Market breadth (from screening data)
            advancing_ratio = market_breadth.get('advancing_ratio', 0.5)
            
            # Regime detection
            if btc_volatility > 0.03:  # High volatility
                regime = 'VOLATILE'
                recommendations = {
                    'reduce_position_size': True,
                    'prefer_strong_signals': True,
                    'avoid_breakouts': True
                }
            elif btc_trend == 'STRONG_UP' and advancing_ratio > 0.7:
                regime = 'RISK_ON'
                recommendations = {
                    'prefer_longs': True,
                    'increase_position_size': True,
                    'follow_momentum': True
                }
            elif btc_trend == 'STRONG_DOWN' and advancing_ratio < 0.3:
                regime = 'RISK_OFF'
                recommendations = {
                    'prefer_shorts': True,
                    'reduce_position_size': True,
                    'avoid_weak_signals': True
                }
            elif btc_volatility < 0.01:
                regime = 'QUIET'
                recommendations = {
                    'look_for_breakouts': True,
                    'normal_position_size': True,
                    'both_directions': True
                }
            else:
                regime = 'NEUTRAL'
                recommendations = {
                    'normal_position_size': True,
                    'both_directions': True,
                    'standard_filters': True
                }
            
            return {
                'regime': regime,
                'btc_volatility': btc_volatility,
                'btc_trend': btc_trend,
                'market_breadth': advancing_ratio,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Regime detection error: {e}")
            return {
                'regime': 'NEUTRAL',
                'recommendations': {'standard_filters': True}
            }
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate ATR-based volatility"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        return atr / df['close'].iloc[-1]
    
    def _get_btc_trend(self, df: pd.DataFrame) -> str:
        """Determine BTC trend strength"""
        ema_20 = df['close'].ewm(span=20).mean()
        ema_50 = df['close'].ewm(span=50).mean()
        
        current_price = df['close'].iloc[-1]
        
        if current_price > ema_20.iloc[-1] * 1.02 and ema_20.iloc[-1] > ema_50.iloc[-1]:
            return 'STRONG_UP'
        elif current_price < ema_20.iloc[-1] * 0.98 and ema_20.iloc[-1] < ema_50.iloc[-1]:
            return 'STRONG_DOWN'
        else:
            return 'NEUTRAL'