# crypto_screener.py - OPTIMIZED VERSION
# Optimization notes:
# 1. Centralized klines cache to prevent duplicate API calls
# 2. Extended ticker cache TTL to 5 minutes
# 3. Pre-fetching strategy for batch processing
# 4. Removed unnecessary time.sleep delays when using cached data
# 5. Reduced batch sizes and max batches to limit API usage

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
import os
import sys
DELAY_PER_REQUEST = 1
# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model import LiveTradingModel
from features_binance import get_live_lstm_features
from library.binance_connector import BinanceTrader
from library.technical_screener import TechnicalScreener, MarketRegimeDetector
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="binance")


class CryptoScreener:
    """Enhanced Cryptocurrency Screener with Technical Analysis"""
    
    def __init__(self, config: Dict):
        self.api_key = config['api_key']
        self.api_secret = config['api_secret']
        self.model_path = config['model_path']
        self.trader = BinanceTrader(self.api_key, self.api_secret, testnet=False)
        self.model = LiveTradingModel(self.model_path)
        
        # Enhanced screening parameters
        self.min_volume_24h = config.get('min_volume_24h', 50000000)
        self.min_price = config.get('min_price', 0.01)
        self.max_price = config.get('max_price', 1000)
        self.exclude_stable = config.get('exclude_stable', True)
        self.batch_size = config.get('batch_size', 20)
        self.max_batches = config.get('max_batches', 3)
        
        # NEW: Enhanced filtering parameters
        self.min_change_24h = config.get('min_change_24h', 3.0)  # Minimum 3% change
        self.max_change_24h = config.get('max_change_24h', 50.0)  # Maximum 50% change (avoid pumps)
        self.min_trades_24h = config.get('min_trades_24h', 10000)  # Minimum trade count
        self.max_spread_pct = config.get('max_spread_pct', 0.5)  # Maximum 0.5% spread
        
        # Signal thresholds
        self.min_signal_strength = config.get('min_signal_strength', 0.7)
        self.max_symbols = config.get('max_symbols', 10)
        
        # NEW: Technical analysis
        self.technical_screener = TechnicalScreener()
        self.market_regime_detector = MarketRegimeDetector()
        
        # NEW: Enable technical analysis
        self.use_technical_analysis = config.get('use_technical_analysis', True)
        self.technical_weight = config.get('technical_weight', 0.4)  # 40% weight to technical
        
        # Cache
        self.all_symbols = []
        self.filtered_symbols = []
        self.last_screening_time = 0
        self.screening_interval = 3600
        
        # Performance cache - OPTIMIZED
        self.ticker_cache = {}
        self.klines_cache = {}
        self.ticker_cache_time = 0
        self.ticker_cache_ttl = 60  # Increased to 5 minutes from 1 minute
        
        # NEW: Centralized klines cache to prevent duplicate API calls
        self.klines_cache = {}  # {symbol: {timeframe: (data, timestamp)}}
        self.klines_cache_ttl = 60  # 5 minutes cache
        
        self.logger = logging.getLogger(__name__)
    
    def get_tradeable_symbols(self) -> List[str]:
        """Get all tradeable USDT perpetual futures symbols - UNCHANGED"""
        try:
            self.logger.info("📋 Getting exchange info...")
            exchange_info = self.trader.client.futures_exchange_info()
            symbols = []
            
            for symbol_info in exchange_info['symbols']:
                symbol = symbol_info['symbol']
                
                if (symbol.endswith('USDT') and 
                    symbol_info['status'] == 'TRADING' and
                    symbol_info['contractType'] == 'PERPETUAL'):
                    
                    symbols.append(symbol)
            
            if self.exclude_stable:
                stable_keywords = ['USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDD', 'FDUSD']
                initial_count = len(symbols)
                symbols = [s for s in symbols if not any(stable in s.replace('USDT', '') for stable in stable_keywords)]
                self.logger.info(f"🚫 Excluded {initial_count - len(symbols)} stablecoins")
            
            symbols = sorted(symbols)
            self.logger.info(f"✅ Found {len(symbols)} tradeable symbols")
            
            return symbols
            
        except Exception as e:
            self.logger.error(f"❌ Error getting symbols: {e}")
            return self._get_fallback_symbols()
    
    def _get_fallback_symbols(self) -> List[str]:
        """Fallback symbols if API fails"""
        return [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
            'XRPUSDT', 'DOTUSDT', 'AVAXUSDT', 'LINKUSDT', 'MATICUSDT'
        ]
    
    def get_24h_ticker_data(self, symbols: List[str]) -> Dict:
        """Get 24h ticker data with caching - ENHANCED"""
        try:
            current_time = time.time()
            
            # Check cache
            if (self.ticker_cache and 
                current_time - self.ticker_cache_time < self.ticker_cache_ttl):
                self.logger.info("📊 Using cached ticker data")
                return self.ticker_cache
            
            # Get fresh data
            self.logger.info("📊 Fetching fresh ticker data...")
            tickers = self.trader.client.futures_ticker()
            ticker_dict = {}
            
            for ticker in tickers:
                symbol = ticker['symbol']
                if symbol in symbols:
                    ticker_dict[symbol] = {
                        'price': float(ticker['lastPrice']),
                        'volume': float(ticker['quoteVolume']),
                        'change_pct': float(ticker['priceChangePercent']),
                        'count': int(ticker['count']),
                        'high': float(ticker['highPrice']),
                        'low': float(ticker['lowPrice']),
                        'bid': float(ticker['bidPrice']) if 'bidPrice' in ticker else 0,
                        'ask': float(ticker['askPrice']) if 'askPrice' in ticker else 0,
                        'weighted_avg_price': float(ticker['weightedAvgPrice'])
                    }
            
            # Update cache
            self.ticker_cache = ticker_dict
            self.ticker_cache_time = current_time
            
            return ticker_dict
            
        except Exception as e:
            self.logger.error(f"Error getting ticker data: {e}")
            return {}
    
    def filter_symbols_by_criteria(self, symbols: List[str]) -> List[str]:
        """Enhanced filtering with 24h change priority to reduce API calls"""
        try:
            self.logger.info(f"🔍 Enhanced filtering of {len(symbols)} symbols...")
            
            # Get ticker data
            ticker_data = self.get_24h_ticker_data(symbols)
            
            if not ticker_data:
                self.logger.warning("⚠️ No ticker data received")
                return symbols[:20]
            
            # First pass: Basic filters
            basic_filtered = []
            filter_stats = {
                'total': len(symbols),
                'has_data': 0,
                'volume_filter': 0,
                'price_filter': 0,
                'trades_filter': 0,
                'spread_filter': 0,
                'basic_filtered': 0,
                'final': 0
            }
            
            for symbol in symbols:
                if symbol not in ticker_data:
                    continue
                
                filter_stats['has_data'] += 1
                data = ticker_data[symbol]
                
                # Extract values
                price = data['price']
                volume = data['volume']
                trade_count = data['count']
                bid = data['bid']
                ask = data['ask']
                
                # Basic filters only (no change filter yet)
                # Filter 1: Volume filter
                if volume < self.min_volume_24h:
                    filter_stats['volume_filter'] += 1
                    continue
                
                # Filter 2: Price filter
                if not (self.min_price <= price <= self.max_price):
                    filter_stats['price_filter'] += 1
                    continue
                
                # Filter 3: Trade count filter (liquidity check)
                if trade_count < self.min_trades_24h:
                    filter_stats['trades_filter'] += 1
                    continue
                
                # Filter 4: Spread filter
                if bid > 0 and ask > 0:
                    spread_pct = ((ask - bid) / bid) * 100
                    if spread_pct > self.max_spread_pct:
                        filter_stats['spread_filter'] += 1
                        continue
                
                basic_filtered.append(symbol)
                filter_stats['basic_filtered'] += 1
            
            # NEW: Priority filtering by 24h change
            self.logger.info(f"📊 Basic filtering complete: {len(basic_filtered)} symbols passed")
            self.logger.info("🎯 Now filtering by 24h change (top 10 gainers + top 10 losers)...")
            
            # Sort by 24h change
            symbols_with_change = []
            for symbol in basic_filtered:
                change_pct = ticker_data[symbol]['change_pct']
                # Exclude extreme changes (likely pumps/dumps)
                if abs(change_pct) <= self.max_change_24h:
                    symbols_with_change.append((symbol, change_pct))
            
            # Sort by change percentage
            symbols_with_change.sort(key=lambda x: x[1], reverse=True)
            
            # Get top gainers and losers
            top_gainers = []
            top_losers = []
            
            # Get top 10 gainers (positive change)
            for symbol, change in symbols_with_change:
                if change > self.min_change_24h and len(top_gainers) < 10:
                    top_gainers.append(symbol)
            
            # Get top 10 losers (negative change)
            for symbol, change in reversed(symbols_with_change):
                if change < -self.min_change_24h and len(top_losers) < 10:
                    top_losers.append(symbol)
            
            # Combine gainers and losers
            final_symbols = top_gainers + top_losers
            filter_stats['final'] = len(final_symbols)
            
            # Log results
            self.logger.info(f"📈 Top {len(top_gainers)} Gainers:")
            for i, symbol in enumerate(top_gainers):
                data = ticker_data[symbol]
                self.logger.info(
                    f"   {i+1:2d}. {symbol:12s}: Change={data['change_pct']:+.1f}%, "
                    f"Vol=${data['volume']/1e6:.1f}M"
                )
            
            self.logger.info(f"📉 Top {len(top_losers)} Losers:")
            for i, symbol in enumerate(top_losers):
                data = ticker_data[symbol]
                self.logger.info(
                    f"   {i+1:2d}. {symbol:12s}: Change={data['change_pct']:+.1f}%, "
                    f"Vol=${data['volume']/1e6:.1f}M"
                )
            
            # Final stats
            self.logger.info(f"📊 Final Filtering Results:")
            self.logger.info(f"   Total symbols: {filter_stats['total']}")
            self.logger.info(f"   Basic filtered: {filter_stats['basic_filtered']}")
            self.logger.info(f"   Top gainers: {len(top_gainers)}")
            self.logger.info(f"   Top losers: {len(top_losers)}")
            self.logger.info(f"   ✅ Final for technical analysis: {filter_stats['final']}")
            
            return final_symbols
            
        except Exception as e:
            self.logger.error(f"❌ Error filtering symbols: {e}")
            import traceback
            traceback.print_exc()
            return symbols[:20]

    def get_multi_timeframe_data(self, symbol: str, limit: int = 200) -> Dict[str, pd.DataFrame]:
        """Get data for multiple timeframes - OPTIMIZED WITH CACHE"""
        try:
            current_time = time.time()
            timeframes = ['1m', '5m', '30m']
            multi_tf_data = {}
            needs_delay = False  # Track if we made any API calls
            
            # Initialize symbol cache if not exists
            if symbol not in self.klines_cache:
                self.klines_cache[symbol] = {}
            
            for tf in timeframes:
                # Check cache first
                if tf in self.klines_cache[symbol]:
                    cached_data, cache_time = self.klines_cache[symbol][tf]
                    if current_time - cache_time < self.klines_cache_ttl:
                        multi_tf_data[tf] = cached_data
                        continue
                
                # Fetch new data if not in cache or expired
                klines = self.trader.client.futures_klines(
                    symbol=symbol,
                    interval=tf,
                    limit=limit
                )
                needs_delay = True  # We made an API call
                
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                # Convert types
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Cache the data
                self.klines_cache[symbol][tf] = (df.copy(), current_time)
                multi_tf_data[tf] = df
            
            return multi_tf_data
            
        except Exception as e:
            self.logger.error(f"Error getting multi-timeframe data for {symbol}: {e}")
            return {}
    
    def get_signal_with_technical(self, symbol: str) -> Optional[Dict]:
        """Get signal enhanced with technical analysis - OPTIMIZED"""
        try:
            # Get model signal (model will handle its own caching internally)
            model_signal = self.model.get_signal(symbol, client=self.trader.client)
            
            if model_signal is None:
                return None
            
            if not self.use_technical_analysis:
                return model_signal
            
            # Get multi-timeframe data (will use cache if available)
            multi_tf_data = self.get_multi_timeframe_data(symbol)
            
            if not multi_tf_data:
                self.logger.warning(f"No multi-timeframe data for {symbol}")
                return model_signal
            
            # Run technical analysis
            technical_analysis = self.technical_screener.analyze_symbol(symbol, multi_tf_data)
            
            if technical_analysis is None:
                return model_signal
            
            # Enhance signal with technical analysis
            enhanced_signal = self._enhance_signal_with_technical(model_signal, technical_analysis)
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"Error getting enhanced signal for {symbol}: {e}")
            return None
    
    def _enhance_signal_with_technical(self, model_signal: Dict, technical: Dict) -> Dict:
        """Combine model signal with technical analysis"""
        try:
            original_decision = model_signal['decision']
            technical_score = technical.get('technical_score', 0)
            
            # Calculate S/R proximity bonus/penalty
            sr_adjustment = self._calculate_sr_adjustment(
                original_decision,
                technical.get('sr_levels', {}),
                technical.get('fibonacci', {})
            )
            
            # Weight combination
            model_weight = 1.0 - self.technical_weight
            tech_weight = self.technical_weight
            
            # Combine scores
            if np.sign(original_decision) == np.sign(technical_score):
                # Same direction - reinforce
                enhanced_decision = (
                    original_decision * model_weight + 
                    technical_score * tech_weight +
                    sr_adjustment * 0.2
                )
            else:
                # Opposite direction - reduce confidence
                enhanced_decision = (
                    original_decision * model_weight + 
                    technical_score * tech_weight * 0.5  # Reduce technical influence
                )
            
            # Apply bounds
            enhanced_decision = np.clip(enhanced_decision, -1.0, 1.0)
            
            # Update signal
            enhanced_signal = model_signal.copy()
            enhanced_signal['decision'] = enhanced_decision
            enhanced_signal['original_decision'] = original_decision
            enhanced_signal['technical_score'] = technical_score
            enhanced_signal['sr_adjustment'] = sr_adjustment
            enhanced_signal['confidence'] = abs(enhanced_decision)
            
            # Add technical details
            enhanced_signal['technical_analysis'] = {
                'trend_30m': technical.get('trend_30m', {}),
                'setup_5m': technical.get('setup_5m', {}),
                'confirmation_1m': technical.get('confirmation_1m', {}),
                'near_support': self._is_near_support(technical.get('sr_levels', {})),
                'near_resistance': self._is_near_resistance(technical.get('sr_levels', {})),
                'near_fibonacci': technical.get('fibonacci', {}).get('near_golden_ratio', False)
            }
            
            # Update action based on enhanced decision
            if enhanced_decision > 0.7:
                enhanced_signal['action'] = 'BUY'
            elif enhanced_decision < -0.7:
                enhanced_signal['action'] = 'SELL'
            else:
                enhanced_signal['action'] = 'HOLD'
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"Error enhancing signal: {e}")
            return model_signal
    
    def _calculate_sr_adjustment(self, decision: float, sr_levels: Dict, fibonacci: Dict) -> float:
        """Calculate adjustment based on S/R proximity"""
        adjustment = 0.0
        
        current_price = sr_levels.get('current_price', 0)
        if current_price == 0:
            return 0.0
        
        # Check confluence levels
        confluence = sr_levels.get('confluence_levels', {})
        
        if decision > 0:  # LONG signal
            # Check support proximity
            supports = confluence.get('support', [])
            if supports:
                nearest_support = supports[0]
                distance_pct = nearest_support.get('distance_pct', 100)
                
                if distance_pct < 1.0:  # Very close to support
                    adjustment += 0.2
                elif distance_pct < 2.0:  # Near support
                    adjustment += 0.1
            
            # Check resistance distance
            resistances = confluence.get('resistance', [])
            if resistances:
                nearest_resistance = resistances[0]
                distance_pct = nearest_resistance.get('distance_pct', 0)
                
                if distance_pct < 1.0:  # Too close to resistance
                    adjustment -= 0.3
                elif distance_pct > 3.0:  # Good distance from resistance
                    adjustment += 0.1
        
        else:  # SHORT signal
            # Opposite logic
            resistances = confluence.get('resistance', [])
            if resistances:
                nearest_resistance = resistances[0]
                distance_pct = nearest_resistance.get('distance_pct', 100)
                
                if distance_pct < 1.0:  # Very close to resistance
                    adjustment += 0.2
                elif distance_pct < 2.0:  # Near resistance
                    adjustment += 0.1
            
            supports = confluence.get('support', [])
            if supports:
                nearest_support = supports[0]
                distance_pct = nearest_support.get('distance_pct', 0)
                
                if distance_pct < 1.0:  # Too close to support
                    adjustment -= 0.3
                elif distance_pct > 3.0:  # Good distance from support
                    adjustment += 0.1
        
        # Fibonacci bonus
        if fibonacci.get('near_golden_ratio'):
            adjustment += 0.1 if decision > 0 else -0.1
        
        return adjustment
    
    def _is_near_support(self, sr_levels: Dict) -> bool:
        """Check if price is near support"""
        confluence = sr_levels.get('confluence_levels', {})
        supports = confluence.get('support', [])
        
        if supports and supports[0].get('distance_pct', 100) < 1.5:
            return True
        return False
    
    def _is_near_resistance(self, sr_levels: Dict) -> bool:
        """Check if price is near resistance"""
        confluence = sr_levels.get('confluence_levels', {})
        resistances = confluence.get('resistance', [])
        
        if resistances and resistances[0].get('distance_pct', 100) < 1.5:
            return True
        return False
    
    def get_signal_for_symbol(self, symbol: str) -> Optional[Dict]:
        """Get trading signal for a single symbol - UPDATED to use technical"""
        try:
            if self.use_technical_analysis:
                signal = self.get_signal_with_technical(symbol)
            else:
                signal = self.model.get_signal(symbol, client=self.trader.client)
            
            if signal is None:
                return None
            
            # Add metadata
            signal['symbol'] = symbol
            signal['timestamp'] = time.time()
            signal['generated_at'] = datetime.now().isoformat()
            
            return signal
            
        except Exception as e:
            self.logger.warning(f"Error getting signal for {symbol}: {e}")
            return None
    
    def get_signals_parallel(self, symbols: List[str]) -> List[Dict]:
        """Get signals for multiple symbols - OPTIMIZED WITH BATCHING"""
        self.logger.info(f"🤖 Getting enhanced signals for {len(symbols)} symbols...")
        
        signals = []
        success_count = 0
        error_count = 0
        
        # Pre-fetch all klines data in one go to optimize API calls
        self.logger.info("📊 Pre-fetching klines data for all symbols...")
        
        # Process in smaller sub-batches for pre-fetching
        pre_fetch_batch_size = 5
        api_calls_made = 0
        
        for i in range(0, len(symbols), pre_fetch_batch_size):
            batch = symbols[i:i+pre_fetch_batch_size]
            
            for symbol in batch:
                try:
                    # Check if data already cached
                    current_time = time.time()
                    is_cached = (symbol in self.klines_cache and 
                               all(tf in self.klines_cache[symbol] and 
                                   current_time - self.klines_cache[symbol][tf][1] < self.klines_cache_ttl 
                                   for tf in ['1m', '5m', '30m']))
                    
                    if not is_cached:
                        # This will cache the data
                        self.get_multi_timeframe_data(symbol)
                        api_calls_made += 1
                        # Small delay only for actual API calls
                        if api_calls_made % 5 == 0:  # Delay every 5 API calls
                            time.sleep(0.2)
                except Exception as e:
                    self.logger.warning(f"Pre-fetch error for {symbol}: {e}")
        
        if api_calls_made > 0:
            self.logger.info(f"📊 Pre-fetched data for {api_calls_made} symbols (others were cached)")
        
        # Now process signals with cached data - NO DELAYS NEEDED
        self.logger.info("🤖 Processing signals with cached data...")
        
        for symbol in symbols:
            try:
                # No delay needed - using cached data
                signal = self.get_signal_for_symbol(symbol)
                if signal is not None:
                    signals.append(signal)
                    success_count += 1
                else:
                    error_count += 1
                
            except Exception as e:
                error_count += 1
                self.logger.warning(f"❌ {symbol}: {str(e)[:50]}")
        
        self.logger.info(f"📊 Signal Results: {success_count} success, {error_count} errors")

        if signals:
            print(f"\n📋 Batch Processing Result:")
            for sig in signals:
                status = "✅" if abs(sig['decision']) > 0.7 else "❌"
                print(f"   {status} {sig['symbol']:10s} = {sig['decision']:+.4f}")
        
        return signals
    
    def rank_signals(self, signals: List[Dict]) -> List[Dict]:
        """Rank signals by strength and quality - ENHANCED WITH VALIDATION"""
        try:
            # Filter by minimum strength
            strong_signals = []

            TRADING_BUY_THRESHOLD = 0.7  # Same as live_trading.py
            
            for signal in signals:
                decision = abs(signal.get('decision', 0))

                if abs(decision) > TRADING_BUY_THRESHOLD:
                    # Existing validation code...
                    if self._validate_signal_will_trade(signal):
                        strong_signals.append(signal)
                
                # Check if technical analysis agrees
                if self.use_technical_analysis:
                    tech_analysis = signal.get('technical_analysis', {})
                    
                    # Quality checks
                    if signal['decision'] > 0:  # Long signal
                        # Avoid long near resistance
                        if tech_analysis.get('near_resistance') and not tech_analysis.get('near_support'):
                            self.logger.info(f"⚠️ {signal['symbol']}: LONG rejected - near resistance")
                            continue
                            
                        # Prefer long near support
                        if tech_analysis.get('near_support'):
                            signal['quality_bonus'] = 0.1
                    else:  # Short signal
                        # Avoid short near support
                        if tech_analysis.get('near_support') and not tech_analysis.get('near_resistance'):
                            self.logger.info(f"⚠️ {signal['symbol']}: SHORT rejected - near support")
                            continue
                            
                        # Prefer short near resistance
                        if tech_analysis.get('near_resistance'):
                            signal['quality_bonus'] = 0.1
                
                # ENHANCED: Use stricter threshold untuk ensure akan trade
                # Trading threshold di live_trading.py adalah > 0.7 dan < -0.7
                # Jadi kita filter yang >= 0.74 untuk ada buffer
                trading_threshold = max(0.73, self.min_signal_strength)
                
                if decision >= trading_threshold:
                    # Additional validation untuk memastikan akan trade
                    if self._validate_signal_will_trade(signal):
                        strong_signals.append(signal)
                    else:
                        self.logger.info(f"⚠️ {signal['symbol']}: Signal {signal['decision']:.3f} won't execute - failed validation")
            
            # If no strong signals, lower threshold slightly but still ensure tradeable
            if not strong_signals and signals:
                fallback_threshold = max(0.72, self.min_signal_strength - 0.05)  # Slight buffer above 0.7
                self.logger.info(f"📊 No signals passed, trying fallback threshold: {fallback_threshold}")
                
                for signal in signals:
                    if abs(signal['decision']) >= fallback_threshold and self._validate_signal_will_trade(signal):
                        strong_signals.append(signal)
            
            # Sort by combined score
            def get_ranking_score(signal):
                base_score = abs(signal['decision'])
                quality_bonus = signal.get('quality_bonus', 0)
                
                # Technical agreement bonus
                if 'technical_score' in signal:
                    if np.sign(signal['decision']) == np.sign(signal['technical_score']):
                        agreement_bonus = 0.1
                    else:
                        agreement_bonus = -0.1
                else:
                    agreement_bonus = 0
                
                return base_score + quality_bonus + agreement_bonus
            
            ranked = sorted(strong_signals, key=get_ranking_score, reverse=True)
            
            # Limit to max symbols
            ranked = ranked[:self.max_symbols]
            
            if ranked:
                self.logger.info(f"🏆 Ranked signals: {len(ranked)} qualified & validated")
                for i, signal in enumerate(ranked):
                    direction = "LONG" if signal['decision'] > 0 else "SHORT"
                    score = get_ranking_score(signal)
                    
                    # Determine if will actually trade
                    will_trade, trade_action = self._get_trade_action(signal['decision'])
                    trade_status = "✅" if will_trade else "⚠️"
                    
                    tech_info = ""
                    if self.use_technical_analysis and 'technical_analysis' in signal:
                        ta = signal['technical_analysis']
                        tech_info = f" | 30m: {ta['trend_30m'].get('bias', 'N/A')}"
                    
                    self.logger.info(
                        f"  {i+1}. {signal['symbol']}: {signal['decision']:+.3f} "
                        f"({direction}) Score: {score:.3f}{tech_info} {trade_status} {trade_action}"
                    )
            else:
                self.logger.warning("⚠️ No signals passed validation for trading!")
            
            return ranked
            
        except Exception as e:
            self.logger.error(f"Error ranking signals: {e}")
            return signals[:self.max_symbols]

    def _validate_signal_will_trade(self, signal: Dict) -> bool:
        """Validate if signal will actually execute a trade"""
        decision = signal.get('decision', 0)
        
        # Check absolute threshold - must be clearly above trading threshold
        # Trading uses > 0.7 and < -0.7, so we need clear margin
        if abs(decision) <= 0.7:
            return False
        
        # Check technical conflicts if enabled
        if self.use_technical_analysis and 'technical_analysis' in signal:
            ta = signal['technical_analysis']
            
            # For LONG signals
            if decision > 0:
                # Skip if near resistance without support
                if ta.get('near_resistance') and not ta.get('near_support'):
                    # Unless it's a very strong signal
                    if abs(decision) < 0.85:
                        return False
            
            # For SHORT signals  
            elif decision < 0:
                # Skip if near support without resistance
                if ta.get('near_support') and not ta.get('near_resistance'):
                    # Unless it's a very strong signal
                    if abs(decision) < 0.85:
                        return False
        
        return True

    def _get_trade_action(self, decision: float) -> Tuple[bool, str]:
        """Determine exact trade action based on decision value"""
        # Replicate exact logic from live_trading.py
        STRONG_BUY = 0.9
        BUY = 0.7
        SELL = -0.7
        STRONG_SELL = -0.9
        
        if decision > STRONG_BUY:
            return True, "STRONG BUY"
        elif decision > BUY:
            return True, "BUY"
        elif decision < STRONG_SELL:
            return True, "STRONG SELL"
        elif decision < SELL:
            return True, "SELL"
        else:
            return False, "NEUTRAL"
    
    def detect_market_regime(self) -> Dict:
        """Detect overall market regime - OPTIMIZED"""
        try:
            # Get BTC data (will use cache if available)
            btc_data = self.get_multi_timeframe_data('BTCUSDT', limit=100)
            
            if not btc_data or '30m' not in btc_data:
                return {'regime': 'NEUTRAL', 'recommendations': {}}
            
            # Use cached ticker data for market breadth
            ticker_data = self.ticker_cache  # Use existing cache
            
            if not ticker_data:
                # Fallback: get fresh data for top symbols only
                top_symbols = self.all_symbols[:50]  # Reduced from 100
                ticker_data = self.get_24h_ticker_data(top_symbols)
            
            advancing = sum(1 for d in ticker_data.values() if d.get('change_pct', 0) > 0)
            declining = sum(1 for d in ticker_data.values() if d.get('change_pct', 0) < 0)
            total = advancing + declining
            
            market_breadth = {
                'advancing_ratio': advancing / total if total > 0 else 0.5
            }
            
            # Detect regime
            regime = self.market_regime_detector.detect_regime(
                btc_data['30m'],
                market_breadth
            )
            
            self.logger.info(f"🌍 Market Regime: {regime['regime']}")
            self.logger.info(f"   BTC Volatility: {regime['btc_volatility']:.3f}")
            self.logger.info(f"   Market Breadth: {regime['market_breadth']:.2f}")
            
            return regime
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return {'regime': 'NEUTRAL', 'recommendations': {}}
    
    def screen_crypto(self) -> List[Dict]:
        """Enhanced screening with batch processing - OPTIMIZED"""
        self.logger.info("🔍 Starting optimized crypto screening...")
        start_time = time.time()
        
        try:
            # Clear old cache
            self._clear_old_cache()
            
            # Step 1: Get all tradeable symbols
            self.logger.info("📋 Step 1: Getting tradeable symbols...")
            if (not self.all_symbols or 
                time.time() - self.last_screening_time > self.screening_interval):
                
                self.all_symbols = self.get_tradeable_symbols()
                self.last_screening_time = time.time()
                
                if not self.all_symbols:
                    self.logger.error("❌ No symbols found!")
                    return []
            
            self.logger.info(f"✅ Using {len(self.all_symbols)} tradeable symbols")
            
            # Step 2: Get ticker data for ALL symbols (cached for 5 mins)
            self.logger.info("📊 Step 2: Getting 24h ticker data...")
            ticker_data = self.get_24h_ticker_data(self.all_symbols)
            
            if not ticker_data:
                self.logger.warning("⚠️ No ticker data received")
                return []
            
            # Step 3: Sort ALL symbols by 24h change
            self.logger.info("🎯 Step 3: Sorting all symbols by 24h change...")
            
            # Apply basic filters first
            qualified_symbols = []
            for symbol in self.all_symbols:
                if symbol not in ticker_data:
                    continue
                    
                data = ticker_data[symbol]
                
                # Basic quality checks
                if (data['volume'] >= self.min_volume_24h and
                    self.min_price <= data['price'] <= self.max_price and
                    data['count'] >= self.min_trades_24h):
                    
                    change_pct = data['change_pct']
                    # Only include symbols with meaningful movement
                    if self.min_change_24h <= abs(change_pct) <= self.max_change_24h:
                        qualified_symbols.append((symbol, change_pct))
            
            # Sort by absolute change
            qualified_symbols.sort(key=lambda x: abs(x[1]), reverse=True)
            self.logger.info(f"📊 Found {len(qualified_symbols)} qualified symbols")
            
            # Step 4: Process in batches with optimized API usage
            batch_size = min(self.batch_size, 15)  # Reduced batch size
            max_batches = min(self.max_batches, 2)  # Reduced max batches

            all_signals = []
            
            for batch_num in range(max_batches):
                # Get batch symbols
                start_idx = batch_num * batch_size
                end_idx = start_idx + batch_size
                
                if start_idx >= len(qualified_symbols):
                    self.logger.info(f"📊 No more symbols to process")
                    break
                    
                batch_symbols = [s[0] for s in qualified_symbols[start_idx:end_idx]]
                
                self.logger.info(f"🔍 Processing batch {batch_num + 1} ({len(batch_symbols)} symbols)...")
                
                # Log batch details
                for i, symbol in enumerate(batch_symbols[:5]):
                    change = ticker_data[symbol]['change_pct']
                    vol = ticker_data[symbol]['volume']
                    self.logger.info(f"   {i+1}. {symbol}: {change:+.1f}%, Vol=${vol/1e6:.1f}M")
                
                # Get signals for this batch (with pre-fetched data)
                batch_signals = self.get_signals_parallel(batch_symbols)
                
                if batch_signals:
                    # Filter by signal strength
                    strong_signals = [s for s in batch_signals if abs(s['decision']) >= self.min_signal_strength]
                    
                    if strong_signals:
                        self.logger.info(f"✅ Found {len(strong_signals)} strong signals in batch {batch_num + 1}")
                        all_signals.extend(strong_signals)
                        
                        # If we have enough signals, stop
                        if len(all_signals) >= self.max_symbols:
                            self.logger.info(f"🎯 Reached target of {self.max_symbols} signals")
                            break
                    else:
                        self.logger.info(f"⚠️ Batch {batch_num + 1}: No signals met threshold")
                else:
                    self.logger.info(f"⚠️ Batch {batch_num + 1}: No valid signals")
                
                # Brief pause between batches only if more batches to process
                if batch_num < max_batches - 1 and len(all_signals) < self.max_symbols:
                    self.logger.info("⏳ Brief pause before next batch...")
                    time.sleep(0.5)  # Reduced from 1 second
            
            # Step 5: Sort all signals by strength and select top
            if all_signals:
                all_signals.sort(key=lambda x: abs(x['decision']), reverse=True)

                print("\n" + "="*70)
                print("📊 SCREENING THRESHOLD CHECK")
                print("="*70)
                
                # Categorize ALL signals
                above_threshold = []
                near_threshold = []
                below_threshold = []
                
                for sig in all_signals[:20]:  # Show max 20 untuk readability
                    score = abs(sig['decision'])
                    if score > 0.7:
                        above_threshold.append(sig)
                    elif score >= 0.6:
                        near_threshold.append(sig)
                    else:
                        below_threshold.append(sig)
                
                # Show results
                if above_threshold:
                    print(f"\n✅ ABOVE THRESHOLD (>0.7): {len(above_threshold)} symbols")
                    for s in above_threshold:
                        direction = "LONG" if s['decision'] > 0 else "SHORT"
                        print(f"   {s['symbol']:12s} | Score: {s['decision']:+.4f} | {direction}")
                
                if near_threshold:
                    print(f"\n⚠️  NEAR THRESHOLD (0.6-0.7): {len(near_threshold)} symbols")
                    for s in near_threshold:
                        direction = "LONG" if s['decision'] > 0 else "SHORT"
                        print(f"   {s['symbol']:12s} | Score: {s['decision']:+.4f} | {direction} ❌")
                
                if below_threshold:
                    print(f"\n❌ BELOW THRESHOLD (<0.6): {len(below_threshold)} symbols")
                    # Just show count, not details
                
                print("="*70)

                top_signals = all_signals[:self.max_symbols]
                
                elapsed = time.time() - start_time
                self.logger.info(f"✅ Screening completed in {elapsed:.1f}s")
                self.logger.info(f"📊 Found {len(top_signals)} signals from {len(qualified_symbols)} qualified symbols")
                
                return top_signals
            else:
                elapsed = time.time() - start_time
                self.logger.info(f"⚠️ No signals found after {batch_num + 1} batches in {elapsed:.1f}s")
                return []
            
        except Exception as e:
            self.logger.error(f"❌ Error in screening: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _clear_old_cache(self):
        """Clear old cached data to prevent memory issues"""
        current_time = time.time()
        
        # Clear old klines cache
        symbols_to_remove = []
        for symbol, timeframes in self.klines_cache.items():
            for tf, (data, cache_time) in list(timeframes.items()):
                if current_time - cache_time > self.klines_cache_ttl:
                    del timeframes[tf]
            if not timeframes:
                symbols_to_remove.append(symbol)
        
        for symbol in symbols_to_remove:
            del self.klines_cache[symbol]
        
        if symbols_to_remove:
            self.logger.info(f"🗑️ Cleared cache for {len(symbols_to_remove)} symbols")
        
    def _will_signal_trade(self, signal: Dict) -> bool:
        """Quick check if signal will trade"""
        decision = signal.get('decision', 0)
        
        # Basic threshold check
        if abs(decision) <= 0.7:
            return False
        
        # Resistance/support check
        if 'technical_analysis' in signal:
            ta = signal['technical_analysis']
            if decision > 0 and ta.get('near_resistance') and abs(decision) < 0.7:
                return False
            if decision < 0 and ta.get('near_support') and abs(decision) < 0.7:
                return False
        
        return True
    
    def get_quick_signals(self, symbols: List[str]) -> List[Dict]:
        """Quick signal check for specific symbols - UNCHANGED"""
        try:
            signals = self.get_signals_parallel(symbols)
            return self.rank_signals(signals)
        except Exception as e:
            self.logger.error(f"Quick signals error: {e}")
            return []

   

class PortfolioManager:
    """Manage multiple crypto positions - UNCHANGED"""
    
    def __init__(self, config: Dict):
        self.total_balance_pct = config.get('total_balance_pct', 0.8)
        self.max_positions = config.get('max_positions', 5)
        self.position_pct = config.get('position_pct_per_symbol', 0.15)
        
        self.active_symbols = set()
        self.symbol_allocations = {}
        
        self.logger = logging.getLogger(__name__)
    
    def can_open_position(self, symbol: str) -> bool:
        """Check if we can open a new position"""
        if symbol in self.active_symbols:
            return True
        
        if len(self.active_symbols) >= self.max_positions:
            return False
        
        return True
    
    def add_symbol(self, symbol: str):
        """Add symbol to active tracking"""
        self.active_symbols.add(symbol)
        if symbol not in self.symbol_allocations:
            self.symbol_allocations[symbol] = self.position_pct
    
    def remove_symbol(self, symbol: str):
        """Remove symbol from active tracking"""
        self.active_symbols.discard(symbol)
        self.symbol_allocations.pop(symbol, None)
    
    def get_allocation(self, symbol: str) -> float:
        """Get position allocation for symbol"""
        return self.symbol_allocations.get(symbol, self.position_pct)
    
    def get_status(self) -> Dict:
        """Get portfolio status"""
        return {
            'active_symbols': list(self.active_symbols),
            'position_count': len(self.active_symbols),
            'max_positions': self.max_positions,
            'total_allocation': sum(self.symbol_allocations.values()),
            'available_slots': self.max_positions - len(self.active_symbols)
        }


def test_enhanced_screener():
    """Test the enhanced screener"""
    config = {
        'api_key': 'EduyybaFGjUpSkR7q2J0HwHjHF6dB8TB5klAAUX8Ukum2Yz1jR2J8osZVXz9kxZC',
        'api_secret': 'QmAxhDG4QYxdrif38WyQ6uvGLv5OZvlGPIRBzdtFWry7adtRNzGFY8HlLkOSLOyY',
        'model_path': 'models/trading_lstm_20250701_233903.pth',
        
        # Enhanced filtering
        'min_volume_24h': 30000000,
        'min_change_24h': 3.0,
        'max_change_24h': 50.0,
        'min_trades_24h': 10000,
        'max_spread_pct': 0.5,
        
        # Technical analysis
        'use_technical_analysis': True,
        'technical_weight': 0.4,
        
        # Signal thresholds
        'min_signal_strength': 0.5,
        'max_symbols': 8
    }
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
    
    print("🚀 TESTING ENHANCED CRYPTO SCREENER")
    print("=" * 50)
    
    screener = CryptoScreener(config)
    
    # Test connection
    print("🔌 Testing Binance connection...")
    if not screener.trader.connect():
        print("❌ Failed to connect to Binance")
        return
    print("✅ Connected successfully")
    
    try:
        # Run full screening
        print("\n🔍 Running enhanced screening...")
        top_signals = screener.screen_crypto()
        
        if top_signals:
            print(f"\n🏆 TOP {len(top_signals)} TRADING OPPORTUNITIES:")
            print("=" * 80)
            
            for i, signal in enumerate(top_signals):
                direction = "🟢 LONG" if signal['decision'] > 0 else "🔴 SHORT"
                
                print(f"\n{i+1}. {signal['symbol']:12s}")
                print(f"   Direction: {direction}")
                print(f"   Model Signal: {signal.get('original_decision', signal['decision']):+.3f}")
                print(f"   Technical Score: {signal.get('technical_score', 0):+.3f}")
                print(f"   Enhanced Signal: {signal['decision']:+.3f}")
                print(f"   Confidence: {signal.get('confidence', abs(signal['decision']))*100:.0f}%")
                
                if 'technical_analysis' in signal:
                    ta = signal['technical_analysis']
                    print(f"   Technical Analysis:")
                    print(f"      30m Trend: {ta['trend_30m'].get('bias', 'N/A')}")
                    print(f"      5m Setup: {ta['setup_5m'].get('quality', 'N/A')}")
                    print(f"      Near Support: {'Yes' if ta.get('near_support') else 'No'}")
                    print(f"      Near Resistance: {'Yes' if ta.get('near_resistance') else 'No'}")
        else:
            print("❌ No signals found")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        screener.trader.disconnect()
        print("\n✅ Test completed!")


if __name__ == "__main__":
    test_enhanced_screener()