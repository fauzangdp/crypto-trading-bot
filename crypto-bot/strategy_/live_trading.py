# live_trading.py
import time
from datetime import datetime
from typing import Dict, Optional
import os
import sys
import threading
import pandas as pd 

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model import LiveTradingModel
from library.binance_connector import BinanceTrader
import logging
from library.sr_fibo import DynamicLevelSelector, SupportResistanceAnalyzer, FibonacciCalculator

class LiveScalpingTrader:
    """Simple Live Scalping Trader based on backtesting logic"""
    
    def __init__(self, config: Dict):
        """Initialize with configuration"""
        # API Configuration
        self.api_key = config['api_key']
        self.api_secret = config['api_secret']
        self.symbol = config['symbol']

        # Logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Trading Configuration
        self.leverage = config.get('leverage', 20)
        self.position_pct = config.get('position_pct', 0.02)  # 2% default
        self.stop_loss_pct = config.get('stop_loss_pct', 0.002)  # 0.2% default
        self.use_dynamic_sl = config.get('use_dynamic_sl', True)

        self.max_pyramids = 5  # Maximum pyramid levels
        self.pyramid_count = 0  # Track current pyramid level
        
        # Take Profit Configuration
        self.use_take_profit = config.get('use_take_profit', True)
        self.tp1_percent = config.get('tp1_percent', 0.005)  # 0.5% default
        self.tp2_percent = config.get('tp2_percent', 0.01)   # 1% default
        self.tp1_size_ratio = config.get('tp1_size_ratio', 0.5)  # 50% position at TP1
        
        self.sl_moved_to_breakeven = False  # Track if SL moved to BE
        self.highest_profit_price = 0       # Track highest price reached
        self.tp1_price = 0                  # Store TP1 target price
        self.trailing_active = False        # Trailing protection active



        # Decision Thresholds (from backtesting)
        self.STRONG_BUY = 0.9
        self.BUY = 0.7
        self.SELL = -0.7
        self.STRONG_SELL = -0.9
        
        # Position Multipliers (from backtesting)
        self.position_multipliers = {
            'strong': 0.05,  # 5% for strong signals
            'normal': 0.02,  # 2% for normal signals
        }
        
        # Override with config if provided
        if 'position_pct_strong' in config:
            self.position_multipliers['strong'] = config['position_pct_strong']
        if 'position_pct_normal' in config:
            self.position_multipliers['normal'] = config['position_pct_normal']
        
        # Initialize components
        self.trader = BinanceTrader(self.api_key, self.api_secret, testnet=False)
        self.model = LiveTradingModel(config.get('model_path'))
        
        # State tracking
        self.in_position = False
        self.position_side = None  # 'LONG' or 'SHORT'
        self.entry_price = 0
        self.stop_loss_price = 0
        self.position_size = 0
        self.stop_loss_order_id = None
        
        # TP State Tracking
        self.tp1_order_id = None
        self.tp2_order_id = None
        self.tp1_hit = False
        self.tp2_hit = False
        self.original_position_size = 0  # Track full position size


        self.use_technical_levels = config.get('use_technical_levels', False)  # Default OFF for safety
        self.lookback_candles = config.get('lookback_candles', 200)
        self.sr_merge_threshold = config.get('sr_merge_threshold', 0.001)
        self.max_risk_technical = config.get('max_risk_technical', 0.02)
        self.min_rr_ratio = config.get('min_rr_ratio', 1.5)
        self.level_update_interval = config.get('level_update_interval', 300)  # 5 minutes
        self.fallback_to_fixed = config.get('fallback_to_fixed', True)
        self.check_resistance = config.get('check_resistance', True)
        self.resistance_proximity_threshold = config.get('resistance_proximity_threshold', 0.003)  # 0.3%
        self.resistance_buffer_pct = config.get('resistance_buffer_pct', 0.0005)  # 0.05% buffer
        self.allow_breakout_trades = config.get('allow_breakout_trades', True)
        self.breakout_confirmation_pct = config.get('breakout_confirmation_pct', 0.002)

        if self.check_resistance:
            self.sr_analyzer = SupportResistanceAnalyzer(
                lookback_period=self.lookback_candles,
                merge_threshold=self.sr_merge_threshold
            )

        # Initialize technical analyzer (ADD THIS)
        if self.use_technical_levels:
            try:
                self.level_selector = DynamicLevelSelector(
                    max_risk_pct=self.max_risk_technical,
                    min_rr_ratio=self.min_rr_ratio
                )
                self.logger.info("✅ Technical level analyzer initialized")
            except Exception as e:
                self.logger.error(f"❌ Failed to init technical analyzer: {e}")
                self.use_technical_levels = False
                self.level_selector = None
        else:
            self.level_selector = None

        # Technical analysis cache (ADD THIS)
        self.technical_cache = {
            'last_update': 0,
            'historical_data': None,
            'sr_levels': None,
            'fib_levels': None,
            'calculated_stops': {}
        }

        # State for technical levels (ADD THIS)
        self.technical_sl_price = 0
        self.technical_tp1_price = 0
        self.technical_tp2_price = 0
        self.using_technical_levels = False  # Flag to track if using technical
        
        # Thread for cleanup
        self.cleanup_thread = None
        

        self.pyramid_gap_timeframe = config.get('pyramid_gap_timeframe', '5m')
        self.pyramid_base_gap = config.get('pyramid_candle_gap', 2)
        self.enable_dynamic_gap = config.get('dynamic_pyramid_gap', True)
        self.pyramid_gap_multipliers = config.get('pyramid_gap_multipliers', {
            'volatility_high': 1.5,
            'volatility_low': 0.8,
            'level_factor': 0.3,
            'performance_factor': 0.2
        })

        self.last_entry_timestamp = 0
        self.pyramid_history = []
        self.candles_since_entry = 0
        
        self.logger.info(f"🔧 Pyramid Timing Config:")
        self.logger.info(f"   Base gap: {self.pyramid_base_gap} candles")
        self.logger.info(f"   Timeframe: {self.pyramid_gap_timeframe}")
        self.logger.info(f"   Dynamic gap: {'Enabled' if self.enable_dynamic_gap else 'Disabled'}")

        # Tambahkan di __init__:
        self.dust_threshold_usd = config.get('dust_threshold_usd', 0.1)  # $10 default
        self.auto_clean_dust = config.get('auto_clean_dust', True)       # Auto clean dust positions

    def calculate_candles_passed(self) -> int:
        """Calculate how many candles have passed since last entry"""
        try:
            if self.last_entry_timestamp == 0:
                # No previous entry, return large number to allow trading
                return 999
            
            current_time = time.time()
            time_elapsed = current_time - self.last_entry_timestamp
            
            # Convert timeframe to seconds
            timeframe_map = {
                '1m': 60,
                '5m': 300,
                '15m': 900,
                '30m': 1800,
                '1h': 3600
            }
            
            candle_seconds = timeframe_map.get(self.pyramid_gap_timeframe, 300)  # Default 5m
            candles_passed = int(time_elapsed / candle_seconds)
            
            self.logger.debug(f"⏱️ Candles passed: {candles_passed} ({time_elapsed:.0f}s / {candle_seconds}s per candle)")
            
            return candles_passed
            
        except Exception as e:
            self.logger.error(f"Error calculating candles: {e}")
            return 0
    
    def is_dust_position(self, position_size: float, price: float) -> bool:
        """Check if position is dust (too small to trade)"""
        try:
            # Calculate position value
            position_value = position_size * price
            
            # Dust threshold: $5 or 0.1% of minimum notional
            dust_threshold = max(5.0, 10.0)  # $5-10 minimum
            
            if position_value < dust_threshold:
                self.logger.warning(f"⚠️ Dust position detected: {position_size} @ ${price:.5f} = ${position_value:.2f}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Dust check error: {e}")
            return False

    def cleanup_dust_position(self, force: bool = False):
        """Clean up dust positions that are too small"""
        try:
            # Get actual position from exchange
            actual_position = self.trader.get_position(self.symbol)
            
            if actual_position['size'] == 0:
                # No position, ensure state is clean
                if self.in_position:
                    self.logger.warning("⚠️ State shows position but exchange shows none")
                    self.cleanup_and_reset_state("No Position - Dust Cleanup")
                return True
                
            current_price = actual_position.get('mark_price', self.trader.get_current_price(self.symbol))
            
            # Check if it's dust
            if self.is_dust_position(actual_position['size'], current_price) or force:
                self.logger.warning(f"🧹 Cleaning dust position: {actual_position['size']} {self.symbol}")
                
                # Get minimum trade size
                exchange_info = self.trader.get_exchange_info(self.symbol)
                min_qty = float(exchange_info.get('min_qty', 0.001))
                step_size = float(exchange_info.get('step_size', 0.001))
                
                # If position is below minimum, we might need special handling
                if actual_position['size'] < min_qty:
                    self.logger.error(f"❌ Position {actual_position['size']} below minimum {min_qty}")
                    
                    # Try market order anyway
                    try:
                        if actual_position['side'] == 'LONG':
                            self.trader.client.futures_create_order(
                                symbol=self.symbol,
                                side='SELL',
                                type='MARKET',
                                quantity=str(actual_position['size']),
                                reduceOnly=True
                            )
                        else:
                            self.trader.client.futures_create_order(
                                symbol=self.symbol,
                                side='BUY',
                                type='MARKET',
                                quantity=str(actual_position['size']),
                                reduceOnly=True
                            )
                        
                        self.logger.info("✅ Dust cleanup order placed")
                        time.sleep(1.0)
                        
                    except Exception as e:
                        self.logger.error(f"❌ Dust cleanup failed: {e}")
                        # Last resort: accept the dust loss
                        self.logger.warning("⚠️ Accepting dust loss, resetting state")
                else:
                    # Normal close
                    if actual_position['side'] == 'LONG':
                        self.trader.market_sell(self.symbol, actual_position['size'])
                    else:
                        self.trader.market_buy(self.symbol, actual_position['size'])
                        
                    time.sleep(1.0)
                
                # Clean up state regardless
                self.cleanup_and_reset_state("Dust Position Cleaned")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Dust cleanup error: {e}")
            return False

    def can_pyramid_now(self, signal_strength: str) -> bool:
        """Check if pyramid is allowed based on candle gap and conditions"""
        try:
            # Calculate candles passed
            candles_passed = self.calculate_candles_passed()
            
            # Get required gap
            required_gap = self.get_dynamic_pyramid_gap()
            
            # Check if gap requirement met
            if candles_passed < required_gap:
                self.logger.warning(f"⏳ Pyramid delayed: {candles_passed} candles < {required_gap} required")
                self.logger.info(f"   Need to wait {required_gap - candles_passed} more candles")
                return False
            
            # Additional checks for quality pyramid
            current_price = self.trader.get_current_price(self.symbol)
            
            # Check if price still favorable
            if self.position_side == 'LONG':
                if current_price < self.entry_price:
                    self.logger.warning(f"⚠️ Price below average entry, delaying pyramid")
                    return False
            else:  # SHORT
                if current_price > self.entry_price:
                    self.logger.warning(f"⚠️ Price above average entry, delaying pyramid")
                    return False
            
            # Check if strong signal can override
            if signal_strength == 'strong' and candles_passed >= max(1, required_gap // 2):
                self.logger.info(f"💪 Strong signal override: allowing pyramid after {candles_passed} candles")
                return True
            
            self.logger.info(f"✅ Pyramid allowed: {candles_passed} candles passed (>= {required_gap})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking pyramid timing: {e}")
            return False  # Safe default

    def get_dynamic_pyramid_gap(self) -> int:
        """Calculate dynamic candle gap based on market conditions"""
        try:
            # Base gap
            base_gap = self.pyramid_base_gap
            
            if not self.enable_dynamic_gap:
                return base_gap
            
            # Factor 1: Pyramid Level (higher level = longer gap)
            level_factor = 1.0 + (self.pyramid_count / self.max_pyramids)
            
            # Factor 2: Volatility (higher volatility = longer gap)
            volatility = self.get_market_volatility()
            volatility_factor = 1.0
            
            if volatility > 0.02:  # High volatility (>2% ATR)
                volatility_factor = 1.5
            elif volatility > 0.015:  # Medium-high
                volatility_factor = 1.3
            elif volatility < 0.005:  # Low volatility
                volatility_factor = 0.8
            
            # Factor 3: Recent pyramid performance
            performance_factor = 1.0
            if len(self.pyramid_history) > 0:
                # Check if recent pyramids were profitable
                recent_pnl = self._calculate_pyramid_performance()
                if recent_pnl < -0.005:  # Recent pyramids losing
                    performance_factor = 1.5  # Wait longer
                elif recent_pnl > 0.01:  # Recent pyramids winning
                    performance_factor = 0.8  # Can be more aggressive
            
            # Calculate final gap
            dynamic_gap = int(base_gap * level_factor * volatility_factor * performance_factor)
            
            # Apply limits
            min_gap = 1
            max_gap = base_gap * 3  # Max 3x base gap
            dynamic_gap = max(min_gap, min(dynamic_gap, max_gap))
            
            self.logger.debug(f"🎯 Dynamic gap: {dynamic_gap} (base={base_gap}, "
                            f"level={level_factor:.1f}, vol={volatility_factor:.1f}, "
                            f"perf={performance_factor:.1f})")
            
            return dynamic_gap
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic gap: {e}")
            return self.pyramid_base_gap  # Fallback to base

    def update_pyramid_tracking(self, entry_price: float):
        """Update pyramid tracking after successful entry"""
        try:
            current_time = time.time()
            
            # Update timestamp
            self.last_entry_timestamp = current_time
            
            # Add to history
            pyramid_entry = {
                'timestamp': current_time,
                'price': entry_price,
                'level': self.pyramid_count,
                'size': self.position_size,
                'side': self.position_side
            }
            
            self.pyramid_history.append(pyramid_entry)
            
            # Keep history limited (last 10 entries)
            if len(self.pyramid_history) > 10:
                self.pyramid_history.pop(0)
            
            # Log tracking update
            self.logger.info(f"📊 Pyramid tracking updated:")
            self.logger.info(f"   Level: {self.pyramid_count}")
            self.logger.info(f"   Entry time: {datetime.fromtimestamp(current_time).strftime('%H:%M:%S')}")
            self.logger.info(f"   Next pyramid allowed after: {self.get_dynamic_pyramid_gap()} candles")
            
        except Exception as e:
            self.logger.error(f"Error updating pyramid tracking: {e}")

    def get_market_volatility(self) -> float:
        """Get current market volatility (ATR as percentage)"""
        try:
            # Try to get from technical cache first
            if hasattr(self, 'technical_cache') and self.technical_cache.get('historical_data') is not None:
                df = self.technical_cache['historical_data']
                atr_pct = self._calculate_atr_percentage(df)
                return atr_pct
            
            # Fallback: Simple volatility calculation from recent prices
            symbol = self.symbol
            klines = self.trader.client.futures_klines(
                symbol=symbol,
                interval='5m',
                limit=20
            )
            
            # Calculate simple volatility
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            closes = [float(k[4]) for k in klines]
            
            # ATR-like calculation
            ranges = []
            for i in range(1, len(highs)):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1])
                )
                ranges.append(tr)
            
            if ranges and closes[-1] > 0:
                avg_range = sum(ranges) / len(ranges)
                volatility = avg_range / closes[-1]
                return volatility
            
            return 0.01  # Default 1% volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return 0.01  # Safe default

    def _calculate_pyramid_performance(self) -> float:
        """Calculate recent pyramid performance for dynamic adjustments"""
        try:
            if len(self.pyramid_history) < 2:
                return 0.0
            
            # Get last 3 pyramids
            recent_pyramids = self.pyramid_history[-3:]
            current_price = self.trader.get_current_price(self.symbol)
            
            total_pnl = 0
            for pyramid in recent_pyramids:
                if self.position_side == 'LONG':
                    pnl = (current_price - pyramid['price']) / pyramid['price']
                else:  # SHORT
                    pnl = (pyramid['price'] - current_price) / pyramid['price']
                total_pnl += pnl
            
            avg_pnl = total_pnl / len(recent_pyramids)
            return avg_pnl
            
        except Exception as e:
            self.logger.error(f"Error calculating pyramid performance: {e}")
            return 0.0

    def check_resistance_proximity(self, current_price: float, signal_type: str) -> bool:
        """
        Check if price is too close to resistance for buy signals
        Returns: True if safe to trade, False if should skip
        """
        # Only check for BUY signals
        if signal_type not in ['BUY', 'STRONG BUY'] or not self.check_resistance:
            return True
        
        try:
            # Get historical data (use cache if available)
            df = self.fetch_historical_data()
            if df is None or df.empty:
                self.logger.warning("⚠️ No data for resistance check, allowing trade")
                return True
            
            # Get S/R levels using existing analyzer
            sr_data = self.sr_analyzer.get_all_levels(df)
            resistances = sr_data.get('resistance', [])
            
            if not resistances:
                self.logger.info("📊 No resistance levels found, allowing trade")
                return True
            
            # Sort by distance (nearest first)
            resistances.sort(key=lambda x: x['distance_pct'])
            
            # Check nearest resistance
            nearest_resistance = resistances[0]
            distance_pct = nearest_resistance['distance_pct'] / 100  # Convert to decimal
            
            # Check if we're above all resistances (breakout scenario)
            if current_price > nearest_resistance['price']:
                if self.allow_breakout_trades:
                    # Check if we have enough confirmation above resistance
                    breakout_distance = (current_price - nearest_resistance['price']) / nearest_resistance['price']
                    
                    if breakout_distance >= self.breakout_confirmation_pct:
                        self.logger.info(f"🚀 CONFIRMED BREAKOUT! Price {breakout_distance*100:.2f}% above resistance")
                        return True
                    else:
                        self.logger.warning(f"⚠️ False breakout risk - only {breakout_distance*100:.2f}% above resistance")
                        return False
                else:
                    self.logger.info("📈 Price above resistance but breakout trades disabled")
                    return False
            
            # Check proximity to resistance
            if distance_pct <= self.resistance_proximity_threshold:
                # Special handling for STRONG signals
                if signal_type == 'STRONG BUY' and distance_pct > self.resistance_proximity_threshold * 0.5:
                    self.logger.warning(f"⚠️ Near resistance but STRONG BUY signal - proceeding with caution")
                    return True
                
                self.logger.warning(f"🚫 TOO CLOSE TO RESISTANCE - SKIP BUY")
                self.logger.info(f"   Current: ${current_price:.5f}")
                self.logger.info(f"   Resistance: ${nearest_resistance['price']:.5f} (Strength: {nearest_resistance['strength']:.2f})")
                self.logger.info(f"   Distance: {distance_pct*100:.2f}% < Threshold: {self.resistance_proximity_threshold*100:.2f}%")
                
                # Log next 2 resistances for context
                if len(resistances) > 1:
                    self.logger.info(f"   Next resistances: ${resistances[1]['price']:.5f} ({resistances[1]['distance_pct']:.1f}%)")
                
                return False
            else:
                self.logger.info(f"✅ Safe distance from resistance: {distance_pct*100:.2f}%")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Resistance check error: {e}")
            import traceback
            traceback.print_exc()
            return True  # Allow trade on error
        
    def display_key_levels(self):
        """Display current key S/R and Fibonacci levels"""
        try:
            df = self.fetch_historical_data()
            if df is None:
                return
            
            current_price = self.trader.get_current_price(self.symbol)
            
            # Get S/R levels
            sr_data = self.sr_analyzer.get_all_levels(df)
            
            # Get Fibonacci levels
            fib_calc = FibonacciCalculator()
            fib_data = fib_calc.get_all_fibonacci_levels(df)
            
            self.logger.info(f"\n📊 KEY LEVELS for {self.symbol} @ ${current_price:.5f}")
            self.logger.info("=" * 50)
            
            # Display nearest resistances
            resistances = sr_data.get('resistance', [])[:3]
            if resistances:
                self.logger.info("🔴 RESISTANCES:")
                for r in resistances:
                    status = "🚨" if r['distance_pct'] < self.resistance_proximity_threshold * 100 else "✓"
                    self.logger.info(f"   {status} ${r['price']:.5f} (+{r['distance_pct']:.2f}%, Strength: {r['strength']:.1f})")
            
            # Display nearest supports  
            supports = sr_data.get('support', [])[:3]
            if supports:
                self.logger.info("🟢 SUPPORTS:")
                for s in supports:
                    self.logger.info(f"   ${s['price']:.5f} (-{s['distance_pct']:.2f}%, Strength: {s['strength']:.1f})")
            
            # Display key Fibonacci levels
            if 'retracements' in fib_data and 'primary' in fib_data['retracements']:
                fib_levels = fib_data['retracements']['primary']
                self.logger.info("📐 FIBONACCI:")
                for ratio in ['0.382', '0.5', '0.618']:
                    if ratio in fib_levels:
                        level = fib_levels[ratio]
                        distance = ((level - current_price) / current_price) * 100
                        self.logger.info(f"   Fib {ratio}: ${level:.5f} ({distance:+.2f}%)")
            
            self.logger.info("=" * 50)
            
        except Exception as e:
            self.logger.error(f"Display levels error: {e}")

    def fetch_historical_data(self, force_update: bool = False) -> pd.DataFrame:
        """Fetch historical OHLC data for technical analysis"""
        try:
            current_time = time.time()
            
            # Check cache
            if not force_update and self.technical_cache['historical_data'] is not None:
                if current_time - self.technical_cache['last_update'] < self.level_update_interval:
                    return self.technical_cache['historical_data']
            
            self.logger.info(f"📊 Fetching {self.lookback_candles} candles for technical analysis...")
            
            # Get multi-timeframe data
            klines = self.trader.client.futures_klines(
                symbol=self.symbol,
                interval='5m',  # 5-minute candles
                limit=self.lookback_candles
            )
            
            # Convert to DataFrame
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
            
            # Update cache
            self.technical_cache['historical_data'] = df
            self.technical_cache['last_update'] = current_time
            
            self.logger.info(f"✅ Fetched {len(df)} candles")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching historical data: {e}")
            return None

    def analyze_technical_zone(self, current_price: float, signal_strength: float) -> Dict:
        """
        Analyze current price position relative to technical zones
        
        Args:
            current_price: Current market price
            signal_strength: Model signal strength (-1 to 1)
        
        Returns:
            Dict with zone analysis and trading recommendation
        """
        try:
            # Get technical data
            df = self.fetch_historical_data()
            if df is None or df.empty:
                return {
                    'zone': 'unknown',
                    'near_support': False,
                    'near_resistance': False,
                    'near_fibonacci': False,
                    'recommendation': 'neutral',
                    'confidence': 0.5
                }
            
            # Get S/R levels
            sr_data = self.sr_analyzer.get_all_levels(df)
            
            # Get Fibonacci levels
            fib_calc = FibonacciCalculator()
            fib_data = fib_calc.get_all_fibonacci_levels(df)
            
            # Analysis results
            analysis = {
                'zone': 'neutral',
                'near_support': False,
                'near_resistance': False,
                'near_fibonacci': False,
                'nearest_support': None,
                'nearest_resistance': None,
                'nearest_fibonacci': None,
                'recommendation': 'neutral',
                'confidence': 0.5
            }
            
            # Check proximity to support (2% threshold)
            supports = sr_data.get('support', [])
            if supports:
                nearest_support = supports[0]
                distance_to_support = (current_price - nearest_support['price']) / current_price
                if distance_to_support <= 0.02:  # Within 2% of support
                    analysis['near_support'] = True
                    analysis['nearest_support'] = nearest_support
                    analysis['zone'] = 'support_zone'
            
            # Check proximity to resistance (2% threshold)
            resistances = sr_data.get('resistance', [])
            if resistances:
                nearest_resistance = resistances[0]
                distance_to_resistance = (nearest_resistance['price'] - current_price) / current_price
                if distance_to_resistance <= 0.02:  # Within 2% of resistance
                    analysis['near_resistance'] = True
                    analysis['nearest_resistance'] = nearest_resistance
                    analysis['zone'] = 'resistance_zone'
            
            # Check Fibonacci golden zones (0.618, 0.5, 0.382)
            golden_ratios = ['0.618', '0.5', '0.382']
            min_fib_distance = float('inf')
            nearest_fib = None
            
            if 'retracements' in fib_data and 'primary' in fib_data['retracements']:
                fib_levels = fib_data['retracements']['primary']
                for ratio in golden_ratios:
                    if ratio in fib_levels:
                        fib_price = fib_levels[ratio]
                        distance = abs(current_price - fib_price) / current_price
                        if distance < min_fib_distance:
                            min_fib_distance = distance
                            nearest_fib = {'ratio': ratio, 'price': fib_price, 'distance': distance}
            
            if nearest_fib and nearest_fib['distance'] <= 0.015:  # Within 1.5% of Fibonacci
                analysis['near_fibonacci'] = True
                analysis['nearest_fibonacci'] = nearest_fib
                if analysis['zone'] == 'neutral':
                    analysis['zone'] = 'fibonacci_zone'
            
            # Generate trading recommendation based on zone + signal
            if signal_strength > 0.7:  # Strong BUY signal
                if analysis['near_support'] or analysis['near_fibonacci']:
                    analysis['recommendation'] = 'strong_buy'
                    analysis['confidence'] = 0.9
                elif analysis['near_resistance']:
                    analysis['recommendation'] = 'wait'  # Wait for breakout confirmation
                    analysis['confidence'] = 0.4
                else:
                    analysis['recommendation'] = 'buy'
                    analysis['confidence'] = 0.7
                    
            elif signal_strength > 0.3:  # Normal BUY signal
                if analysis['near_support'] or analysis['near_fibonacci']:
                    analysis['recommendation'] = 'buy'
                    analysis['confidence'] = 0.8
                elif analysis['near_resistance']:
                    analysis['recommendation'] = 'neutral'
                    analysis['confidence'] = 0.3
                else:
                    analysis['recommendation'] = 'neutral'
                    analysis['confidence'] = 0.5
                    
            elif signal_strength < -0.7:  # Strong SELL signal
                if analysis['near_resistance'] or analysis['near_fibonacci']:
                    analysis['recommendation'] = 'strong_sell'
                    analysis['confidence'] = 0.9
                elif analysis['near_support']:
                    analysis['recommendation'] = 'wait'  # Wait for breakdown confirmation
                    analysis['confidence'] = 0.4
                else:
                    analysis['recommendation'] = 'sell'
                    analysis['confidence'] = 0.7
                    
            elif signal_strength < -0.3:  # Normal SELL signal
                if analysis['near_resistance'] or analysis['near_fibonacci']:
                    analysis['recommendation'] = 'sell'
                    analysis['confidence'] = 0.8
                elif analysis['near_support']:
                    analysis['recommendation'] = 'neutral'
                    analysis['confidence'] = 0.3
                else:
                    analysis['recommendation'] = 'neutral'
                    analysis['confidence'] = 0.5
            
            # Log analysis
            self.logger.info(f"🎯 Technical Zone Analysis:")
            self.logger.info(f"   Zone: {analysis['zone']}")
            self.logger.info(f"   Near Support: {analysis['near_support']}")
            self.logger.info(f"   Near Resistance: {analysis['near_resistance']}")
            self.logger.info(f"   Near Fibonacci: {analysis['near_fibonacci']}")
            self.logger.info(f"   Recommendation: {analysis['recommendation']} (confidence: {analysis['confidence']:.1f})")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Technical zone analysis error: {e}")
            return {
                'zone': 'error',
                'near_support': False,
                'near_resistance': False,
                'near_fibonacci': False,
                'recommendation': 'neutral',
                'confidence': 0.0
            }

    def calculate_technical_stops(self, entry_price: float, position_type: str) -> Dict:
        """Calculate stop loss and take profit using technical analysis"""
        try:
            if not self.use_technical_levels or not self.level_selector:
                return None
                
            # Get historical data
            df = self.fetch_historical_data()
            if df is None or df.empty:
                self.logger.warning("⚠️ No historical data for technical analysis")
                return None
            
            # Calculate ATR for volatility
            atr = self._calculate_atr_percentage(df)
            
            # Get optimal levels
            self.logger.info("🎯 Calculating technical levels...")
            analysis = self.level_selector.get_optimal_levels(
                df=df,
                entry_price=entry_price,
                position_type=position_type,
                atr_pct=atr
            )
            
            # Validate results
            if not analysis or not analysis.get('validation', {}).get('valid', False):
                self.logger.warning("⚠️ Technical analysis invalid or no good levels found")
                if analysis and analysis.get('validation'):
                    for issue in analysis['validation'].get('issues', []):
                        self.logger.warning(f"   - {issue}")
                return None
            
            # Extract levels
            result = {
                'stop_loss': analysis['stop_loss']['price'],
                'stop_loss_type': analysis['stop_loss']['type'],
                'stop_loss_risk': analysis['stop_loss']['risk_pct'],
                'tp1': analysis['take_profits']['tp1']['price'],
                'tp1_type': analysis['take_profits']['tp1']['type'],
                'tp1_rr': analysis['take_profits']['tp1']['rr_ratio'],
                'tp2': analysis['take_profits']['tp2']['price'],
                'tp2_type': analysis['take_profits']['tp2']['type'],
                'tp2_rr': analysis['take_profits']['tp2']['rr_ratio'],
                'avg_rr': analysis['validation']['avg_rr_ratio']
            }
            
            # Log analysis
            self.logger.info(f"📊 Technical Analysis Results:")
            self.logger.info(f"   Entry: ${entry_price:.5f}")
            self.logger.info(f"   SL: ${result['stop_loss']:.5f} ({result['stop_loss_type']}, Risk: {result['stop_loss_risk']*100:.1f}%)")
            self.logger.info(f"   TP1: ${result['tp1']:.5f} ({result['tp1_type']}, RR: {result['tp1_rr']:.1f})")
            self.logger.info(f"   TP2: ${result['tp2']:.5f} ({result['tp2_type']}, RR: {result['tp2_rr']:.1f})")
            self.logger.info(f"   Avg R:R: {result['avg_rr']:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating technical stops: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_atr_percentage(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR as percentage of price"""
        try:
            high = df['high']
            low = df['low']
            close = df['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            
            current_price = df['close'].iloc[-1]
            atr_pct = atr / current_price
            
            return atr_pct
            
        except Exception as e:
            self.logger.error(f"ATR calculation error: {e}")
            return 0.01  # Default 1% volatility
        
    def format_price_for_api(self, price, symbol=None):
        """UNIVERSAL price formatter - NO SCIENTIFIC NOTATION"""
        try:
            from decimal import Decimal, ROUND_DOWN
            
            if symbol is None:
                symbol = self.symbol
                
            # Get exchange info
            exchange_info = self.trader.get_exchange_info(symbol)
            tick_size = float(exchange_info.get('tick_size', 1e-06))
            
            # ✅ USE DECIMAL ARITHMETIC for perfect precision
            price_decimal = Decimal(str(price))
            tick_decimal = Decimal(str(tick_size))
            
            # Round to tick_size using Decimal
            steps = price_decimal / tick_decimal
            rounded_steps = steps.quantize(Decimal('1'), rounding=ROUND_DOWN)
            formatted_decimal = rounded_steps * tick_decimal
            
            # ✅ DYNAMIC decimal calculation
            if tick_size >= 1:
                decimals = 0
            else:
                # Handle scientific notation in tick_size
                if tick_size < 0.001:
                    tick_str = f"{tick_size:.15f}".rstrip('0')
                else:
                    tick_str = str(tick_size)
                
                if '.' in tick_str:
                    decimals = len(tick_str.split('.')[1])
                else:
                    decimals = 0
            
            # ✅ FORCE DECIMAL FORMAT (NO SCIENTIFIC)
            if decimals == 0:
                result = str(int(formatted_decimal))
            else:
                result = f"{float(formatted_decimal):.{decimals}f}"
            
            # ✅ VALIDATION: Ensure no scientific notation
            if 'e' in result.lower() or 'E' in result:
                # Emergency fallback for very small numbers
                if float(formatted_decimal) < 0.001:
                    result = f"{float(formatted_decimal):.12f}".rstrip('0')
                    if result.endswith('.'):
                        result += '0'
                else:
                    result = str(float(formatted_decimal))
            
            self.logger.info(f"🔧 Price formatted: {price} → {result} (tick: {tick_size})")
            return result
            
        except Exception as e:
            self.logger.error(f"Price formatting error: {e}")
            # ✅ SAFE FALLBACK - NEVER SCIENTIFIC
            result = f"{price:.12f}".rstrip('0')
            if result.endswith('.'):
                result += '0'
            return result

    def cancel_order_with_verification(self, order_id: int, max_retries: int = 3) -> bool:
        """Cancel order with verification and retry"""
        if not order_id:
            return True
            
        for attempt in range(max_retries):
            try:
                # Attempt to cancel
                self.trader.cancel_order(self.symbol, order_id)
                
                # Wait for processing
                time.sleep(0.5)
                
                # Verify cancellation
                open_orders = self.trader.get_open_orders(self.symbol)
                if not any(order['orderId'] == order_id for order in open_orders):
                    self.logger.info(f"✅ Order {order_id} verified cancelled")
                    return True
                else:
                    self.logger.warning(f"⚠️ Order {order_id} still active, retry {attempt+1}/{max_retries}")
                    
            except Exception as e:
                self.logger.error(f"❌ Cancel attempt {attempt+1} failed for {order_id}: {e}")
                
        return False
        
    def background_cleanup(self):
        """Run cleanup in background thread"""
        try:
            time.sleep(2.0)  # Wait 2 seconds
            open_orders = self.trader.get_open_orders(self.symbol)
            for order in open_orders:
                if order['type'] in ['STOP_MARKET', 'STOP_LOSS']:
                    try:
                        self.trader.cancel_order(self.symbol, order['orderId'])
                        self.logger.info(f"🧹 Cleaned orphan order {order['orderId']}")
                    except:
                        pass
        except:
            pass
    
    def cleanup_orphan_orders(self):
        """Clean up any orphan orders before new trade"""
        try:
            open_orders = self.trader.get_open_orders(self.symbol)
            
            if not self.in_position and open_orders:
                self.logger.warning(f"⚠️ Found {len(open_orders)} orphan orders, cleaning up...")
                
                for order in open_orders:
                    try:
                        self.trader.cancel_order(self.symbol, order['orderId'])
                        self.logger.info(f"🧹 Cancelled orphan order {order['orderId']} ({order['type']})")
                    except Exception as e:
                        self.logger.error(f"Failed to cancel orphan {order['orderId']}: {e}")
                        
                # Wait a bit for cancellations to process
                if open_orders:
                    time.sleep(0.5)
                    
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def calculate_dynamic_stop_loss(self, current_price: float) -> float:
        """Calculate dynamic stop loss (simplified for live trading)"""
        if not self.use_dynamic_sl:
            return self.stop_loss_pct
        
        # For live trading, use a simple volatility estimate
        # In production, you might want to track recent price movements
        return min(0.05, max(0.005, self.stop_loss_pct))  # Clamp between 0.5% and 5%
    
    def check_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss is hit"""
        if not self.in_position:
            return False
        
        if self.position_side == 'LONG':
            return current_price <= self.stop_loss_price
        else:  # SHORT
            return current_price >= self.stop_loss_price
    
    def log_trading_rules(self):
        """Log trading rules untuk clarity"""
        self.logger.info("📋 TRADING RULES:")
        self.logger.info(f"   STRONG BUY:  decision > {self.STRONG_BUY}")
        self.logger.info(f"   BUY:         decision > {self.BUY}")
        self.logger.info(f"   NEUTRAL:     {self.SELL} to {self.BUY}")
        self.logger.info(f"   SELL:        decision < {self.SELL}")
        self.logger.info(f"   STRONG SELL: decision < {self.STRONG_SELL}")
        self.logger.info("")
        self.logger.info("📊 POSITION SIZES:")
        self.logger.info(f"   Normal Signal: {self.position_multipliers['normal']*100:.1f}% of balance")
        self.logger.info(f"   Strong Signal: {self.position_multipliers['strong']*100:.1f}% of balance")
        self.logger.info(f"   Stop Loss: {self.stop_loss_pct*100:.1f}%")
        self.logger.info(f"   Max Pyramids: {self.max_pyramids}")
        if self.use_take_profit:
            self.logger.info("")
            self.logger.info("🎯 TAKE PROFIT:")
            self.logger.info(f"   TP1: {self.tp1_percent*100:.1f}% ({self.tp1_size_ratio*100:.0f}% of position)")
            self.logger.info(f"   TP2: {self.tp2_percent*100:.1f}% ({(1-self.tp1_size_ratio)*100:.0f}% of position)")
    
        if self.use_technical_levels:
            self.logger.info("")
            self.logger.info("📊 TECHNICAL ANALYSIS:")
            self.logger.info(f"   Enabled: Yes")
            self.logger.info(f"   Lookback: {self.lookback_candles} candles")
            self.logger.info(f"   Max Risk: {self.max_risk_technical*100:.1f}%")
            self.logger.info(f"   Min R:R: {self.min_rr_ratio}")
            self.logger.info(f"   Fallback: {'Yes' if self.fallback_to_fixed else 'No'}")

    def move_sl_to_breakeven(self):
        """Move stop loss to breakeven - FIXED VERSION WITH VERIFICATION"""
        if self.sl_moved_to_breakeven:
            return  # Already moved
        
        try:
            self.logger.info("🎯 Moving stop loss to BREAKEVEN...")
            
            # Step 1: Wait for TP1 order to fully settle
            time.sleep(2.0)  # Increased wait time
            
            # Step 2: Get ACTUAL current position size from exchange
            actual_position = self.trader.get_position(self.symbol)
            
            if actual_position['size'] == 0:
                self.logger.error("❌ No position found, cannot move SL")
                self.cleanup_and_reset_state("No Position After TP1")
                return
                
            current_size = actual_position['size']
            self.logger.info(f"📊 Current position size after TP1: {current_size}")
            
            # Update tracked size to match actual
            self.sync_position_info_only() 
            current_size = self.position_size
            
            # Step 3: Cancel old stop loss with verification
            if self.stop_loss_order_id:
                # Get all open orders to verify
                open_orders = self.trader.get_open_orders(self.symbol)
                sl_exists = any(o['orderId'] == self.stop_loss_order_id for o in open_orders)
                
                if sl_exists:
                    for attempt in range(3):
                        try:
                            self.trader.cancel_order(self.symbol, self.stop_loss_order_id)
                            self.logger.info(f"✅ Old SL cancel attempt {attempt+1}")
                            time.sleep(1.5)  # Wait for cancellation
                            
                            # Verify cancellation
                            open_orders = self.trader.get_open_orders(self.symbol)
                            if not any(o['orderId'] == self.stop_loss_order_id for o in open_orders):
                                self.logger.info("✅ Old SL successfully cancelled")
                                break
                            else:
                                self.logger.warning(f"⚠️ Old SL still exists, attempt {attempt+1}/3")
                        except Exception as e:
                            self.logger.warning(f"⚠️ Cancel attempt {attempt+1} failed: {e}")
                            
            # Step 4: Calculate breakeven price with small buffer
            fee_buffer = 0.0003  # 0.03% buffer for fees
            if self.position_side == 'LONG':
                breakeven_price = self.entry_price * (1 + fee_buffer)
            else:  # SHORT
                breakeven_price = self.entry_price * (1 - fee_buffer)
            
            # Format price properly
            formatted_price = self.format_price_for_api(breakeven_price)
            
            # Step 5: Place new stop loss with ACTUAL current size
            max_attempts = 3
            sl_placed = False
            
            for attempt in range(max_attempts):
                try:
                    # Use integer for quantity
                    quantity = int(current_size)
                    
                    self.logger.info(f"📝 Placing BE SL: {quantity} @ ${formatted_price}")
                    
                    sl_order = self.trader.place_stop_loss_order(
                        symbol=self.symbol,
                        side=self.position_side,
                        quantity=quantity,
                        stop_price=formatted_price
                    )
                    
                    self.stop_loss_order_id = sl_order['order_id']
                    self.stop_loss_price = breakeven_price
                    
                    # Wait and verify
                    time.sleep(1.5)
                    
                    # Verify the new SL exists
                    open_orders = self.trader.get_open_orders(self.symbol)
                    sl_verified = any(o['orderId'] == sl_order['order_id'] for o in open_orders)
                    
                    if sl_verified:
                        self.sl_moved_to_breakeven = True
                        sl_placed = True
                        self.logger.info(f"✅ SL moved to BREAKEVEN @ ${formatted_price}")
                        self.logger.info(f"   Size: {quantity} (verified)")
                        self.logger.info(f"   Order ID: {sl_order['order_id']}")
                        break
                    else:
                        self.logger.warning(f"⚠️ SL not found in open orders, attempt {attempt+1}/{max_attempts}")
                        
                except Exception as e:
                    self.logger.error(f"❌ Attempt {attempt+1} failed: {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(1.5)
            
            # If failed to place breakeven SL, place emergency SL
            if not sl_placed:
                self.logger.error("❌ Failed to move SL to breakeven, placing emergency SL")
                try:
                    # Emergency SL at 1% loss from current price
                    current_price = self.trader.get_current_price(self.symbol)
                    if self.position_side == 'LONG':
                        emergency_sl = current_price * 0.99
                    else:
                        emergency_sl = current_price * 1.01
                        
                    formatted_emergency = self.format_price_for_api(emergency_sl)
                    sl_order = self.trader.place_stop_loss_order(
                        symbol=self.symbol,
                        side=self.position_side,
                        quantity=int(current_size),
                        stop_price=formatted_emergency
                    )
                    self.stop_loss_order_id = sl_order['order_id']
                    self.logger.warning(f"⚠️ Emergency SL placed @ ${formatted_emergency}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Emergency SL also failed: {e}")
                    # Last resort: close position
                    self.close_position("Failed to set stop loss")
                        
        except Exception as e:
            self.logger.error(f"❌ Move SL to breakeven error: {e}")
            import traceback
            traceback.print_exc()

    def place_stop_loss_order(self, symbol: str, side: str, quantity: float, stop_price) -> Dict:
        """ENHANCED stop loss order - WITH REDUCEONLY AND DUST HANDLING"""
        try:
            # Get exchange info
            exchange_info = self.get_exchange_info(symbol)
            tick_size = float(exchange_info.get('tick_size', 0.00001))
            step_size = float(exchange_info.get('step_size', 0.001))
            min_qty = float(exchange_info.get('min_qty', 0.001))
            
            # ✅ ENSURE QUANTITY IS NOT DUST
            if quantity < min_qty:
                logging.warning(f"⚠️ Stop loss quantity {quantity} below minimum {min_qty}")
                # Round up to minimum
                quantity = min_qty
            
            # ✅ UNIVERSAL PRICE FORMATTING
            if tick_size >= 1:
                decimals = 0
            else:
                tick_str = f"{tick_size:.15f}".rstrip('0')
                decimals = len(tick_str.split('.')[1]) if '.' in tick_str else 0

            # ✅ UNIVERSAL PRICE FORMATTING
            if isinstance(stop_price, str):
                # Already formatted from live_trading.py - validate only
                formatted_stop_price = stop_price
                try:
                    # Verify it's valid number
                    test_float = float(stop_price)
                except ValueError:
                    raise ValueError(f"Invalid stop price format: {stop_price}")
            else:
                # Format raw float using SAME logic as live_trading.py
                from decimal import Decimal, ROUND_DOWN
                
                price_decimal = Decimal(str(stop_price))
                tick_decimal = Decimal(str(tick_size))
                
                steps = price_decimal / tick_decimal
                rounded_steps = steps.quantize(Decimal('1'), rounding=ROUND_DOWN)
                formatted_decimal = rounded_steps * tick_decimal
                
                # Use pre-calculated decimals
                if decimals == 0:
                    formatted_stop_price = str(int(formatted_decimal))
                else:
                    formatted_stop_price = f"{float(formatted_decimal):.{decimals}f}"
            
            # ✅ CRITICAL: SCIENTIFIC NOTATION CHECK
            if 'e' in str(formatted_stop_price).lower() or 'E' in str(formatted_stop_price):
                raise ValueError(f"Scientific notation not allowed: {formatted_stop_price}")
            
            # ✅ QUANTITY FORMATTING WITH STEP SIZE
            from decimal import Decimal, ROUND_DOWN
            
            qty_decimal = Decimal(str(quantity))
            step_decimal = Decimal(str(step_size))
            
            # Round to step size
            steps = qty_decimal / step_decimal
            rounded_steps = steps.quantize(Decimal('1'), rounding=ROUND_DOWN)
            formatted_quantity_decimal = rounded_steps * step_decimal
            
            # Format quantity
            if step_size >= 1:
                formatted_quantity = str(int(formatted_quantity_decimal))
            else:
                # Calculate decimal places for quantity
                step_str = f"{step_size:.15f}".rstrip('0')
                qty_decimals = len(step_str.split('.')[1]) if '.' in step_str else 0
                formatted_quantity = f"{float(formatted_quantity_decimal):.{qty_decimals}f}"
            
            # ✅ VALIDATION LOG
            logging.info(f"🔧 Stop Loss API Call:")
            logging.info(f"   Symbol: {symbol}")
            logging.info(f"   Stop Price: {formatted_stop_price} (type: {type(formatted_stop_price)})")
            logging.info(f"   Quantity: {formatted_quantity}")
            logging.info(f"   Tick Size: {tick_size}")
            logging.info(f"   Step Size: {step_size}")
            
            # Validate current price logic
            current_price = self.get_current_price(symbol)
            order_side = 'SELL' if side == 'LONG' else 'BUY'
            
            if side == 'LONG':
                if float(formatted_stop_price) >= current_price:
                    logging.warning(f"⚠️ Stop price {formatted_stop_price} >= current {current_price}, adjusting...")
                    adjusted_price = current_price * 0.99
                    formatted_stop_price = f"{adjusted_price:.{decimals}f}" if tick_size < 1 else str(int(adjusted_price))
            else:  # SHORT
                if float(formatted_stop_price) <= current_price:
                    logging.warning(f"⚠️ Stop price {formatted_stop_price} <= current {current_price}, adjusting...")
                    adjusted_price = current_price * 1.01
                    formatted_stop_price = f"{adjusted_price:.{decimals}f}" if tick_size < 1 else str(int(adjusted_price))
            
            # ✅ API CALL WITH STRING PARAMETERS AND REDUCEONLY
            order = self.client.futures_create_order(
                symbol=symbol,
                side=order_side,
                type='STOP_MARKET',
                stopPrice=str(formatted_stop_price),  # ✅ FORCE STRING
                quantity=str(formatted_quantity),     # ✅ FORCE STRING
                timeInForce='GTC',
                reduceOnly=True,                      # ✅ CRITICAL: Ensure it only reduces position
                priceProtect=True,
                workingType='MARK_PRICE'
            )
            
            logging.info(f"✅ Stop Loss placed: {order_side} {formatted_quantity} @ ${formatted_stop_price}")
            
            return {
                'order_id': order['orderId'],
                'stop_price': formatted_stop_price,
                'quantity': formatted_quantity,
                'side': order_side,
                'status': order['status']
            }
            
        except Exception as e:
            logging.error(f"❌ Enhanced stop loss error: {e}")
            logging.error(f"   Symbol: {symbol}")
            logging.error(f"   Side: {side}")
            logging.error(f"   Quantity: {quantity}")
            logging.error(f"   Stop Price Input: {stop_price}")
            raise

    def place_take_profit_orders(self):
        """Place TP1 and TP2 orders - WITH TECHNICAL LEVELS SUPPORT"""
        if not self.use_take_profit or not self.in_position:
            return
        time.sleep(1.0)
        self.sync_state_with_exchange()
        self.sync_position_info_only() 
        try:
            # Get exchange info ONCE
            actual_position = self.trader.get_position(self.symbol)
            if actual_position['size'] == 0:
                self.logger.error("❌ No position found for TP orders!")
                return
            
            actual_size = actual_position['size']
            if actual_size != self.position_size:
                self.logger.warning(f"⚠️ Position size mismatch: tracked={self.position_size}, actual={actual_size}")
                self.position_size = actual_size  # Update to actual

            exchange_info = self.trader.get_exchange_info(self.symbol)
            step_size = float(exchange_info.get('step_size', 1.0))
            tick_size = float(exchange_info.get('tick_size', 1e-06))
            
            # Determine TP prices based on technical or fixed
            if self.using_technical_levels and self.technical_tp1_price > 0 and self.technical_tp2_price > 0:
                # Use technical levels
                tp1_price = self.technical_tp1_price
                tp2_price = self.technical_tp2_price
                self.logger.info("🎯 Using TECHNICAL levels for take profits")
            else:
                # Fallback to fixed percentages
                current_price = self.entry_price
                if self.position_side == 'LONG':
                    tp1_price = current_price * (1 + self.tp1_percent)
                    tp2_price = current_price * (1 + self.tp2_percent)
                else:  # SHORT
                    tp1_price = current_price * (1 - self.tp1_percent)
                    tp2_price = current_price * (1 - self.tp2_percent)
                self.logger.info("📊 Using FIXED percentages for take profits")
            
            # Determine order side
            order_side = 'SELL' if self.position_side == 'LONG' else 'BUY'
            
            # Store TP1 price for trailing
            self.tp1_price = tp1_price
            
            # FORMAT PRICES WITH HELPER
            tp1_price_str = self.format_price_for_api(tp1_price)
            tp2_price_str = self.format_price_for_api(tp2_price)
            
            # Calculate raw quantities
            tp1_quantity_raw = self.position_size * self.tp1_size_ratio
            tp2_quantity_raw = self.position_size * (1 - self.tp1_size_ratio)
            
            # PROPER QUANTITY FORMATTING BASED ON STEP_SIZE
            def round_to_step_size(quantity, step_size):
                """Round quantity to proper step_size"""
                if step_size >= 1.0:
                    # For step_size like 1.0, convert to int
                    return int(round(quantity / step_size) * step_size)
                else:
                    # For step_size like 0.1, 0.01, etc.
                    multiplier = 1 / step_size
                    return round(round(quantity * multiplier) / multiplier, 10)
            
            tp1_quantity = round_to_step_size(tp1_quantity_raw, step_size)
            tp2_quantity = round_to_step_size(tp2_quantity_raw, step_size)
            
            # Validate minimum quantities
            min_qty = float(exchange_info.get('min_qty', step_size))
            if tp1_quantity < min_qty or tp2_quantity < min_qty:
                self.logger.warning(f"⚠️ TP quantities too small: TP1={tp1_quantity}, TP2={tp2_quantity}, min={min_qty}")
                return
            
            # Log for debug
            self.logger.info(f"🎯 TP Setup: step_size={step_size}, tick_size={tick_size}")
            self.logger.info(f"🎯 TP Final Quantities: TP1={tp1_quantity}, TP2={tp2_quantity}")
            self.logger.info(f"🎯 TP Orders: TP1={tp1_quantity}@{tp1_price_str}, TP2={tp2_quantity}@{tp2_price_str}")
            
            # Place TP1 order
            try:
                tp1_order = self.trader.client.futures_create_order(
                    symbol=self.symbol,
                    side=order_side,
                    type='LIMIT',
                    quantity=str(tp1_quantity),  # Convert to string
                    price=tp1_price_str,
                    timeInForce='GTC',
                    reduceOnly=True
                )
                self.tp1_order_id = tp1_order['orderId']
                self.logger.info(f"✅ TP1 placed: {tp1_quantity} @ {tp1_price_str}")
                
                time.sleep(0.3)
                
            except Exception as e:
                self.logger.error(f"❌ TP1 placement failed: {e}")
                self.logger.error(f"   TP1 params: qty={tp1_quantity}, price={tp1_price_str}")
                self.tp1_order_id = None

            # Place TP2 order
            try:
                tp2_order = self.trader.client.futures_create_order(
                    symbol=self.symbol,
                    side=order_side,
                    type='LIMIT',
                    quantity=str(tp2_quantity),  # Convert to string
                    price=tp2_price_str,
                    timeInForce='GTC',
                    reduceOnly=True
                )
                self.tp2_order_id = tp2_order['orderId']
                self.logger.info(f"✅ TP2 placed: {tp2_quantity} @ {tp2_price_str}")
                
            except Exception as e:
                self.logger.error(f"❌ TP2 placement failed: {e}")
                self.logger.error(f"   TP2 params: qty={tp2_quantity}, price={tp2_price_str}")
                self.tp2_order_id = None
                
        except Exception as e:
            self.logger.error(f"⚠️ Error placing take profits: {e}")
            self.tp1_order_id = None
            self.tp2_order_id = None

    def check_trailing_profit_protection(self, current_price: float):
        """Check if profit dropped 70% from TP1 level - NEW METHOD"""
        if not self.in_position or not self.trailing_active:
            return False
        
        # Calculate current profit
        if self.position_side == 'LONG':
            current_profit_pct = (current_price - self.entry_price) / self.entry_price
            tp1_profit_pct = self.tp1_percent
        else:  # SHORT
            current_profit_pct = (self.entry_price - current_price) / self.entry_price
            tp1_profit_pct = self.tp1_percent
        
        # Check if profit dropped to 70% of TP1 target
        profit_threshold = tp1_profit_pct * 0.7  # 70% of TP1 profit
        
        if current_profit_pct < profit_threshold and current_profit_pct > 0:
            self.logger.warning(f"⚠️ Profit dropped below 70% of TP1!")
            self.logger.info(f"   Current profit: {current_profit_pct*100:.2f}%")
            self.logger.info(f"   Threshold: {profit_threshold*100:.2f}% (70% of TP1)")
            return True
        
        # Update highest profit tracking
        if current_profit_pct > 0:
            if self.position_side == 'LONG' and current_price > self.highest_profit_price:
                self.highest_profit_price = current_price
            elif self.position_side == 'SHORT' and (self.highest_profit_price == 0 or current_price < self.highest_profit_price):
                self.highest_profit_price = current_price
        
        return False

    def check_take_profit_status(self):
        """Check if any TP has been hit - ENHANCED VERSION WITH VERIFICATION"""
        if not self.in_position or not self.use_take_profit:
            return
        
        try:
            open_orders = self.trader.get_open_orders(self.symbol)
            open_order_ids = [o['orderId'] for o in open_orders]
            
            # Check TP1
            if self.tp1_order_id and self.tp1_order_id not in open_order_ids and not self.tp1_hit:
                self.tp1_hit = True
                self.logger.info(f"💰 TP1 HIT! Closed {self.tp1_size_ratio*100:.0f}% of position")
                
                # Wait for TP1 to fully settle
                time.sleep(2.0)
                
                # Get actual position size after TP1
                actual_position = self.trader.get_position(self.symbol)
                
                if actual_position['size'] > 0:
                    # Update tracked position size to actual
                    old_size = self.position_size
                    self.position_size = actual_position['size']
                    self.logger.info(f"📊 Position size updated: {old_size} → {self.position_size}")
                    
                    # Move SL to breakeven
                    self.move_sl_to_breakeven()
                    
                    # Double verification - check SL is in place after 2 seconds
                    time.sleep(2.0)
                    open_orders = self.trader.get_open_orders(self.symbol)
                    has_sl = any(o['type'] in ['STOP_MARKET', 'STOP_LOSS'] for o in open_orders)
                    
                    if has_sl:
                        self.logger.info("✅ Breakeven SL verified in place")
                        # Count SL orders
                        sl_count = sum(1 for o in open_orders if o['type'] in ['STOP_MARKET', 'STOP_LOSS'])
                        if sl_count > 1:
                            self.logger.warning(f"⚠️ Multiple SL orders found: {sl_count}")
                    else:
                        self.logger.error("❌ WARNING: No SL found after move to breakeven!")
                        # Try once more with emergency handling
                        self.logger.info("🔄 Attempting emergency SL placement...")
                        
                        # Get fresh position data
                        fresh_position = self.trader.get_position(self.symbol)
                        if fresh_position['size'] > 0:
                            try:
                                # Place emergency SL at entry price
                                emergency_sl = self.format_price_for_api(self.entry_price)
                                sl_order = self.trader.place_stop_loss_order(
                                    symbol=self.symbol,
                                    side=self.position_side,
                                    quantity=int(fresh_position['size']),
                                    stop_price=emergency_sl
                                )
                                self.stop_loss_order_id = sl_order['order_id']
                                self.logger.info(f"✅ Emergency SL placed @ ${emergency_sl}")
                            except Exception as e:
                                self.logger.error(f"❌ Emergency SL failed: {e}")
                                self.close_position("No Stop Loss Protection")
                    
                    # Activate trailing protection
                    self.trailing_active = True
                    self.logger.info("🎯 Trailing profit protection ACTIVATED")
                    
                else:
                    self.logger.warning("⚠️ No position found after TP1 hit")
                    self.cleanup_and_reset_state("TP1 Complete - No Position")
            
            # Check TP2
            if self.tp2_order_id and self.tp2_order_id not in open_order_ids and not self.tp2_hit:
                self.tp2_hit = True
                self.logger.info(f"💰 TP2 HIT! Position fully closed")
                
                # Wait for TP2 to settle
                time.sleep(1.5)
                
                # Verify position is fully closed
                final_position = self.trader.get_position(self.symbol)
                if final_position['size'] > 0:
                    self.logger.warning(f"⚠️ Position still open after TP2: {final_position['size']}")
                    # Force close remaining
                    self.close_position("TP2 Cleanup")
                else:
                    # Proper cleanup when position fully closed
                    self.cleanup_and_reset_state("TP2 Hit - Position Closed")
                    
        except Exception as e:
            self.logger.error(f"⚠️ Error checking TP status: {e}")
            import traceback
            traceback.print_exc()

    def cleanup_and_reset_state(self, reason: str = "Position Closed"):
        """Centralized cleanup with DOUBLE VERIFICATION"""

        self.technical_sl_price = 0
        self.technical_tp1_price = 0
        self.technical_tp2_price = 0
        self.using_technical_levels = False
        try:
            self.logger.info(f"🧹 Cleaning up: {reason}")
            
            # STEP 1: Cancel tracked orders
            cleanup_successful = True
            orders_to_cancel = []
            
            if self.stop_loss_order_id:
                orders_to_cancel.append(('Stop Loss', self.stop_loss_order_id))
            if self.tp1_order_id and not self.tp1_hit:
                orders_to_cancel.append(('TP1', self.tp1_order_id))
            if self.tp2_order_id and not self.tp2_hit:
                orders_to_cancel.append(('TP2', self.tp2_order_id))
            
            # Cancel each with verification
            for order_type, order_id in orders_to_cancel:
                if not self.cancel_order_with_verification(order_id, max_retries=3):
                    self.logger.error(f"❌ Failed to cancel {order_type} order {order_id}")
                    cleanup_successful = False
                if len(orders_to_cancel) > 1:
                    time.sleep(0.2)
            
            if orders_to_cancel:
                time.sleep(1.0)
            
            # STEP 2: Double check - Get ALL open orders
            self.logger.info("🔍 Double checking for any remaining orders...")
            remaining_orders = self.trader.get_open_orders(self.symbol)
            
            if remaining_orders:
                self.logger.warning(f"⚠️ Found {len(remaining_orders)} remaining orders after cleanup!")
                
                # Cancel ALL remaining orders
                for order in remaining_orders:
                    try:
                        self.trader.cancel_order(self.symbol, order['orderId'])
                        self.logger.info(f"🧹 Force cancelled: {order['type']} order {order['orderId']}")
                        time.sleep(0.2)
                    except Exception as e:
                        self.logger.error(f"Failed to force cancel {order['orderId']}: {e}")
                
                # Wait and verify again
                time.sleep(1.0)
                
                # FINAL CHECK
                final_check = self.trader.get_open_orders(self.symbol)
                if final_check:
                    self.logger.error(f"❌ CRITICAL: {len(final_check)} orders still remain!")
                    # Log details for debugging
                    for order in final_check:
                        self.logger.error(f"   Stuck order: {order['type']} {order['orderId']} "
                                        f"Side: {order['side']} Qty: {order['origQty']}")
                else:
                    self.logger.info("✅ All orders successfully cleaned")
            else:
                self.logger.info("✅ No remaining orders found")
            
            # STEP 3: Reset ALL state variables
            self.in_position = False
            self.position_side = None
            self.entry_price = 0
            self.stop_loss_price = 0
            self.position_size = 0
            self.stop_loss_order_id = None
            self.pyramid_count = 0
            
            # Reset TP state
            self.tp1_order_id = None
            self.tp2_order_id = None
            self.tp1_hit = False
            self.tp2_hit = False
            self.original_position_size = 0
            
            # Reset trailing state
            self.sl_moved_to_breakeven = False
            self.highest_profit_price = 0
            self.tp1_price = 0
            self.trailing_active = False
            
            # 🆕 ADD: Reset pyramid timing state
            self.last_entry_timestamp = 0
            self.pyramid_history = []
            self.candles_since_entry = 0
            
            self.logger.info(f"✅ Cleanup completed: {reason}")
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup error: {e}")
            import traceback
            traceback.print_exc()

    def place_stop_loss_order_with_verification(self, symbol: str, side: str, quantity: float, stop_price: str) -> Dict:
        """Place stop loss with verification - NEW METHOD"""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                # Place order
                sl_order = self.trader.place_stop_loss_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    stop_price=stop_price
                )
                
                # Wait for processing
                time.sleep(0.5)
                
                # Verify order exists
                open_orders = self.trader.get_open_orders(symbol)
                if any(o['orderId'] == sl_order['order_id'] for o in open_orders):
                    self.logger.info(f"✅ Stop loss verified placed: {sl_order['order_id']}")
                    return sl_order
                else:
                    self.logger.warning(f"⚠️ Stop loss not found in open orders, attempt {attempt+1}")
                    
            except Exception as e:
                self.logger.error(f"❌ SL placement attempt {attempt+1} failed: {e}")
                
            time.sleep(1.0)
        
        raise Exception("Failed to place and verify stop loss order")

    def execute_trade(self, action: str, signal_strength: str):
        """Execute trade with COMPLETE VERIFICATION"""
        try:
            # 🔧 ADD: Pre-trade cleanup check
            if not self.in_position:
                if not self.pre_trade_cleanup():
                    self.logger.error("❌ Pre-trade cleanup failed, aborting trade")
                    return
            
            # Get current state
            balance = self.trader.get_balance()
            current_price = self.trader.get_current_price(self.symbol)
            
            # Calculate new position size
            position_pct = self.position_multipliers.get(signal_strength, self.position_pct)
            new_position_size = self.trader.calculate_position_size(
                balance=balance,
                percent=position_pct,
                leverage=self.leverage,
                price=current_price,
                symbol=self.symbol
            )
            
            # Execute based on action
            if action == 'BUY':
                # Check if adding to existing LONG position
                if self.position_side == 'LONG':
                    # CHECK PYRAMID LIMIT
                    if self.pyramid_count >= self.max_pyramids:
                        self.logger.warning(f"⚠️ Max pyramid level ({self.max_pyramids}) reached, skipping add")
                        return
                    
                    # 🆕 ADD: Check pyramid timing
                    if not self.can_pyramid_now(signal_strength):
                        self.logger.info(f"⏳ Pyramid conditions not met, waiting...")
                        return
                    
                    # PYRAMIDING - Add to existing position
                    self.logger.info(f"🔷 ADDING to LONG position (Pyramid #{self.pyramid_count + 1})")
                    
                    # Cancel old stop loss and TPs first with verification
                    orders_to_cancel = []
                    if self.stop_loss_order_id:
                        orders_to_cancel.append(('SL', self.stop_loss_order_id))
                    if not self.tp1_hit and self.tp1_order_id:
                        orders_to_cancel.append(('TP1', self.tp1_order_id))
                    if not self.tp2_hit and self.tp2_order_id:
                        orders_to_cancel.append(('TP2', self.tp2_order_id))
                    
                    for order_type, order_id in orders_to_cancel:
                        if self.cancel_order_with_verification(order_id):
                            self.logger.info(f"✅ Cancelled {order_type} order")
                        else:
                            self.logger.warning(f"⚠️ Failed to verify cancel {order_type}")
                    
                    self.stop_loss_order_id = None
                    self.tp1_order_id = None
                    self.tp2_order_id = None
                    
                    # Execute additional buy
                    order = self.trader.market_buy(self.symbol, new_position_size)

                    time.sleep(1.0)  # Increased from 0.5
                    actual_position = self.trader.get_position(self.symbol)
                    if actual_position['size'] == 0:
                        self.logger.error("❌ Position not detected after order!")
                        # Try wait more
                        time.sleep(0.4)
                        actual_position = self.trader.get_position(self.symbol)

                    self.logger.info("⏳ Waiting for position settlement...")
                    add_price = order['fill_price']

                    time.sleep(0.5)
                    self.logger.info("⏳ Waiting for position settlement...")

                    self.sync_state_with_exchange()
                    time.sleep(0.3)
                    
                    # Update average entry price
                    old_value = self.entry_price * self.position_size
                    new_value = add_price * new_position_size
                    self.position_size += new_position_size
                    self.entry_price = (old_value + new_value) / self.position_size
                    
                    # Increment pyramid count
                    self.pyramid_count += 1
                    
                    self.logger.info(f"🔵 PYRAMID {signal_strength.upper()} BUY @ ${add_price:.5f}")
                    self.logger.info(f"   Added: {new_position_size} | Total: {self.position_size}")
                    self.logger.info(f"   Avg Entry: ${self.entry_price:.5f}")
                    self.logger.info(f"   Pyramid Level: {self.pyramid_count}/{self.max_pyramids}")
                    
                    # 🆕 ADD: Update pyramid tracking
                    self.update_pyramid_tracking(add_price)
                    
                else:
                    # New position or flip from SHORT
                    order = self.trader.market_buy(self.symbol, new_position_size)
                    time.sleep(1.0)
                    self.logger.info("⏳ Waiting for position settlement...")

                    actual_position = self.trader.get_position(self.symbol)
                    if actual_position['size'] == 0:
                        self.logger.error("❌ Position not detected after order!")
                        # Try wait more
                        time.sleep(0.4)
                        actual_position = self.trader.get_position(self.symbol)

                    self.entry_price = order['fill_price']
                    
                    if self.entry_price == 0:
                        self.entry_price = current_price
                    
                    time.sleep(0.5)
                    self.logger.info("⏳ Waiting for position settlement...")
                    self.sync_position_info_only()
                    time.sleep(0.3)
                    
                    self.in_position = True
                    self.position_side = 'LONG'
                    self.position_size = new_position_size
                    self.original_position_size = new_position_size
                    self.pyramid_count = 1
                    self.tp1_hit = False
                    self.tp2_hit = False
                    self.sl_moved_to_breakeven = False
                    self.highest_profit_price = current_price
                    self.trailing_active = False
                    
                    self.logger.info(f"🔵 {signal_strength.upper()} BUY @ ${self.entry_price:.5f}")
                    self.logger.info(f"   Position: {new_position_size}")
                    self.logger.info(f"   Pyramid Level: {self.pyramid_count}/{self.max_pyramids}")
                    
                    # 🆕 ADD: Initialize tracking for new position
                    self.last_entry_timestamp = time.time()
                    self.pyramid_history = [{
                        'timestamp': self.last_entry_timestamp,
                        'price': self.entry_price,
                        'level': 1,
                        'size': new_position_size,
                        'side': 'LONG'
                    }]
                
                # Update stop loss for total position
                if self.use_technical_levels:
                    # Try technical analysis first
                    technical_levels = self.calculate_technical_stops(self.entry_price, 'LONG')  # or 'SHORT' for sell
                    
                    if technical_levels and technical_levels['stop_loss_risk'] <= self.max_risk_technical:
                        # Use technical levels
                        self.stop_loss_price = technical_levels['stop_loss']
                        self.technical_sl_price = technical_levels['stop_loss']
                        self.technical_tp1_price = technical_levels['tp1']
                        self.technical_tp2_price = technical_levels['tp2']
                        self.using_technical_levels = True
                        self.logger.info(f"🎯 Using TECHNICAL stop loss @ ${self.stop_loss_price:.5f}")
                    else:
                        # Fallback to dynamic/fixed
                        if technical_levels:
                            self.logger.warning(f"⚠️ Technical SL risk too high ({technical_levels['stop_loss_risk']*100:.1f}%), using fixed")
                        sl_pct = self.calculate_dynamic_stop_loss(current_price)
                        self.stop_loss_price = self.entry_price * (1 - sl_pct)  # Adjust sign for SHORT
                        self.using_technical_levels = False
                        self.logger.info(f"📊 Using FIXED stop loss @ ${self.stop_loss_price:.5f}")
                else:
                    # Standard calculation
                    sl_pct = self.calculate_dynamic_stop_loss(current_price)
                    self.stop_loss_price = self.entry_price * (1 - sl_pct)  # Adjust sign for SHORT
                    self.using_technical_levels = False
                
                # Clean any existing stop orders before placing new one
                existing_stops = [o for o in self.trader.get_open_orders(self.symbol) 
                                if o['type'] in ['STOP_MARKET', 'STOP_LOSS']]
                for stop_order in existing_stops:
                    try:
                        self.trader.cancel_order(self.symbol, stop_order['orderId'])
                        self.logger.info(f"🧹 Cleaned existing stop order {stop_order['orderId']}")
                    except:
                        pass
                
                # 🔧 PLACE STOP LOSS WITH VERIFICATION
                try:
                    time.sleep(0.7)
                    formatted_stop_price = self.format_price_for_api(self.stop_loss_price)
                    
                    # Use verified placement
                    sl_order = self.place_stop_loss_order_with_verification(
                        symbol=self.symbol,
                        side='LONG',
                        quantity=int(self.position_size),
                        stop_price=formatted_stop_price
                    )
                    self.stop_loss_order_id = sl_order['order_id']
                    self.logger.info(f"🛡️ Stop Loss verified placed @ ${formatted_stop_price}")
                    
                except Exception as e:
                    self.logger.error(f"❌ CRITICAL: Failed to place stop loss! {e}")
                    # Emergency: Close position if no SL
                    self.close_position("Failed Stop Loss")
                    return
                
                # Place take profit orders
                if self.use_take_profit:
                    try:
                        time.sleep(2.0)
                        self.place_take_profit_orders()
                        
                        # Verify protective orders
                        protective_status = []
                        if self.stop_loss_order_id:
                            protective_status.append("SL✅")
                        else:
                            protective_status.append("SL❌")
                            
                        if self.tp1_order_id:
                            protective_status.append("TP1✅")
                        else:
                            protective_status.append("TP1❌")
                            
                        if self.tp2_order_id:
                            protective_status.append("TP2✅")
                        else:
                            protective_status.append("TP2❌")
                            
                        self.logger.info(f"🛡️ Protective Orders: {' | '.join(protective_status)}")
                        
                        # Alert if missing critical protection
                        if not self.stop_loss_order_id:
                            self.logger.error(f"🚨 CRITICAL: {self.symbol} has NO STOP LOSS!")
                            self.close_position("No Stop Loss Protection")
                            
                    except Exception as e:
                        self.logger.error(f"❌ Take profit failed: {e}")
                        self.tp1_order_id = None
                        self.tp2_order_id = None
                        
            elif action == 'SELL':
                # Check if adding to existing SHORT position
                if self.position_side == 'SHORT':
                    # CHECK PYRAMID LIMIT
                    if self.pyramid_count >= self.max_pyramids:
                        self.logger.warning(f"⚠️ Max pyramid level ({self.max_pyramids}) reached, skipping add")
                        return
                    
                    # 🆕 ADD: Check pyramid timing
                    if not self.can_pyramid_now(signal_strength):
                        self.logger.info(f"⏳ Pyramid conditions not met, waiting...")
                        return
                    
                    # PYRAMIDING - Add to existing position
                    self.logger.info(f"🔷 ADDING to SHORT position (Pyramid #{self.pyramid_count + 1})")
                    
                    # Cancel old stop loss and TPs first with verification
                    orders_to_cancel = []
                    if self.stop_loss_order_id:
                        orders_to_cancel.append(('SL', self.stop_loss_order_id))
                    if not self.tp1_hit and self.tp1_order_id:
                        orders_to_cancel.append(('TP1', self.tp1_order_id))
                    if not self.tp2_hit and self.tp2_order_id:
                        orders_to_cancel.append(('TP2', self.tp2_order_id))
                    
                    for order_type, order_id in orders_to_cancel:
                        if self.cancel_order_with_verification(order_id):
                            self.logger.info(f"✅ Cancelled {order_type} order")
                        else:
                            self.logger.warning(f"⚠️ Failed to verify cancel {order_type}")
                    
                    self.stop_loss_order_id = None
                    self.tp1_order_id = None
                    self.tp2_order_id = None
                    
                    # Execute additional sell
                    order = self.trader.market_sell(self.symbol, new_position_size)
                    add_price = order['fill_price']
                    
                    time.sleep(0.5)
                    self.logger.info("⏳ Waiting for position settlement...")
                    self.sync_position_info_only()
                    time.sleep(0.3)
                    
                    # Update average entry price
                    old_value = self.entry_price * self.position_size
                    new_value = add_price * new_position_size
                    self.position_size += new_position_size
                    self.entry_price = (old_value + new_value) / self.position_size
                    
                    # Increment pyramid count
                    self.pyramid_count += 1
                    
                    self.logger.info(f"🔴 PYRAMID {signal_strength.upper()} SELL @ ${add_price:.5f}")
                    self.logger.info(f"   Added: {new_position_size} | Total: {self.position_size}")
                    self.logger.info(f"   Avg Entry: ${self.entry_price:.5f}")
                    self.logger.info(f"   Pyramid Level: {self.pyramid_count}/{self.max_pyramids}")
                    
                    # 🆕 ADD: Update tracking after SHORT pyramid
                    self.update_pyramid_tracking(add_price)
                    
                else:
                    # New position or flip from LONG
                    order = self.trader.market_sell(self.symbol, new_position_size)
                    self.entry_price = order['fill_price']
                    
                    if self.entry_price == 0:
                        self.entry_price = current_price
                    
                    time.sleep(0.5)
                    self.logger.info("⏳ Waiting for position settlement...")
                    self.sync_state_with_exchange()
                    time.sleep(0.3)
                    
                    self.in_position = True
                    self.position_side = 'SHORT'
                    self.position_size = new_position_size
                    self.original_position_size = new_position_size
                    self.pyramid_count = 1
                    self.tp1_hit = False
                    self.tp2_hit = False
                    self.sl_moved_to_breakeven = False
                    self.highest_profit_price = current_price
                    self.trailing_active = False
                    
                    self.logger.info(f"🔴 {signal_strength.upper()} SELL @ ${self.entry_price:.5f}")
                    self.logger.info(f"   Position: {new_position_size}")
                    self.logger.info(f"   Pyramid Level: {self.pyramid_count}/{self.max_pyramids}")
                    
                    # 🆕 ADD: Initialize tracking
                    self.last_entry_timestamp = time.time()
                    self.pyramid_history = [{
                        'timestamp': self.last_entry_timestamp,
                        'price': self.entry_price,
                        'level': 1,
                        'size': new_position_size,
                        'side': 'SHORT'
                    }]
                
                # Update stop loss for total position
                if self.use_technical_levels:
                    # Try technical analysis first
                    technical_levels = self.calculate_technical_stops(self.entry_price, 'SHORT')  # or 'SHORT' for sell
                    
                    if technical_levels and technical_levels['stop_loss_risk'] <= self.max_risk_technical:
                        # Use technical levels
                        self.stop_loss_price = technical_levels['stop_loss']
                        self.technical_sl_price = technical_levels['stop_loss']
                        self.technical_tp1_price = technical_levels['tp1']
                        self.technical_tp2_price = technical_levels['tp2']
                        self.using_technical_levels = True
                        self.logger.info(f"🎯 Using TECHNICAL stop loss @ ${self.stop_loss_price:.5f}")
                    else:
                        # Fallback to dynamic/fixed
                        if technical_levels:
                            self.logger.warning(f"⚠️ Technical SL risk too high ({technical_levels['stop_loss_risk']*100:.1f}%), using fixed")
                        sl_pct = self.calculate_dynamic_stop_loss(current_price)
                        self.stop_loss_price = self.entry_price * (1 + sl_pct)  # Adjust sign for SHORT
                        self.using_technical_levels = False
                        self.logger.info(f"📊 Using FIXED stop loss @ ${self.stop_loss_price:.5f}")
                else:
                    # Standard calculation
                    sl_pct = self.calculate_dynamic_stop_loss(current_price)
                    self.stop_loss_price = self.entry_price * (1 + sl_pct)  # Adjust sign for SHORT
                    self.using_technical_levels = False
                
                # Clean any existing stop orders before placing new one
                existing_stops = [o for o in self.trader.get_open_orders(self.symbol) 
                                if o['type'] in ['STOP_MARKET', 'STOP_LOSS']]
                for stop_order in existing_stops:
                    try:
                        self.trader.cancel_order(self.symbol, stop_order['orderId'])
                        self.logger.info(f"🧹 Cleaned existing stop order {stop_order['orderId']}")
                    except:
                        pass
                
                # 🔧 PLACE STOP LOSS WITH VERIFICATION
                try:
                    time.sleep(0.3)
                    formatted_stop_price = self.format_price_for_api(self.stop_loss_price)
                    
                    # Use verified placement
                    sl_order = self.place_stop_loss_order_with_verification(
                        symbol=self.symbol,
                        side='SHORT',
                        quantity=int(self.position_size),
                        stop_price=formatted_stop_price
                    )
                    self.stop_loss_order_id = sl_order['order_id']
                    self.logger.info(f"🛡️ Stop Loss verified placed @ ${formatted_stop_price}")
                    
                except Exception as e:
                    self.logger.error(f"❌ CRITICAL: Failed to place stop loss! {e}")
                    # Emergency: Close position if no SL
                    self.close_position("Failed Stop Loss")
                    return
                
                # Place take profit orders
                if self.use_take_profit:
                    try:
                        time.sleep(0.3)
                        self.place_take_profit_orders()
                        
                        # Verify protective orders
                        protective_status = []
                        if self.stop_loss_order_id:
                            protective_status.append("SL✅")
                        else:
                            protective_status.append("SL❌")
                            
                        if self.tp1_order_id:
                            protective_status.append("TP1✅")
                        else:
                            protective_status.append("TP1❌")
                            
                        if self.tp2_order_id:
                            protective_status.append("TP2✅")
                        else:
                            protective_status.append("TP2❌")
                            
                        self.logger.info(f"🛡️ Protective Orders: {' | '.join(protective_status)}")
                        
                        # Alert if missing critical protection
                        if not self.stop_loss_order_id:
                            self.logger.error(f"🚨 CRITICAL: {self.symbol} has NO STOP LOSS!")
                            self.close_position("No Stop Loss Protection")
                            
                    except Exception as e:
                        self.logger.error(f"❌ Take profit failed: {e}")
                        self.tp1_order_id = None
                        self.tp2_order_id = None
                        
        except Exception as e:
            self.logger.error(f"❌ Trade execution error: {e}")
            import traceback
            traceback.print_exc()

    def pre_trade_cleanup(self):
        """Ensure clean state before any new trade - NEW METHOD"""
        try:
            # Check if we have position
            actual_position = self.trader.get_position(self.symbol)
            
            # If no position, clean ALL orders
            if actual_position['size'] == 0:
                open_orders = self.trader.get_open_orders(self.symbol)
                if open_orders:
                    self.logger.warning(f"⚠️ Pre-trade: Found {len(open_orders)} orphan orders!")
                    
                    for order in open_orders:
                        try:
                            self.trader.cancel_order(self.symbol, order['orderId'])
                            self.logger.info(f"🧹 Pre-trade cleanup: Cancelled {order['type']} {order['orderId']}")
                            time.sleep(0.2)
                        except:
                            pass
                    
                    # Wait and verify
                    time.sleep(0.5)
                    
                    # Final check
                    remaining = self.trader.get_open_orders(self.symbol)
                    if remaining:
                        self.logger.error(f"❌ Pre-trade: {len(remaining)} orders still remain!")
                        return False
                        
            return True
            
        except Exception as e:
            self.logger.error(f"Pre-trade cleanup error: {e}")
            return False
    
    def health_check(self):
        """Periodic health check - ENHANCED WITH DUST DETECTION"""
        try:
            # Get actual state
            actual_position = self.trader.get_position(self.symbol)
            open_orders = self.trader.get_open_orders(self.symbol)
            
            # Check for dust position
            if actual_position['size'] > 0:
                current_price = actual_position.get('mark_price', self.trader.get_current_price(self.symbol))
                
                if self.is_dust_position(actual_position['size'], current_price):
                    self.logger.warning(f"⚠️ Dust position found in health check: {actual_position['size']}")
                    
                    # Check if there's a stop loss for this dust
                    has_stop_loss = any(o['type'] in ['STOP_MARKET', 'STOP_LOSS'] for o in open_orders)
                    
                    if not has_stop_loss:
                        # No stop loss on dust, just clean it
                        self.logger.info("🧹 Cleaning unprotected dust position")
                        self.cleanup_dust_position(force=True)
                        return
                    else:
                        # Has stop loss but it's dust, cancel orders and clean
                        self.logger.info("🧹 Cleaning dust position with orders")
                        
                        # Cancel all orders
                        for order in open_orders:
                            try:
                                self.trader.cancel_order(self.symbol, order['orderId'])
                                time.sleep(0.1)
                            except:
                                pass
                        
                        # Clean dust
                        time.sleep(0.5)
                        self.cleanup_dust_position(force=True)
                        return
            
            # Original health check logic continues...
            # Check 1: Position without stop loss
            if actual_position['size'] > 0 and self.in_position:
                has_stop_loss = any(o['type'] in ['STOP_MARKET', 'STOP_LOSS'] 
                                for o in open_orders)
                
                if not has_stop_loss:
                    self.logger.error("❌ CRITICAL: Position without stop loss detected!")
                    self.logger.info("🛡️ Placing emergency stop loss...")
                    
                    # Emergency stop loss
                    emergency_sl_pct = 0.05  # 5% emergency SL
                    if self.position_side == 'LONG':
                        emergency_sl_price = self.entry_price * (1 - emergency_sl_pct)
                    else:
                        emergency_sl_price = self.entry_price * (1 + emergency_sl_pct)
                    
                    try:
                        formatted_price = self.format_price_for_api(emergency_sl_price)
                        sl_order = self.place_stop_loss_order_with_verification(
                            symbol=self.symbol,
                            side=self.position_side,
                            quantity=int(actual_position['size']),
                            stop_price=formatted_price
                        )
                        self.stop_loss_order_id = sl_order['order_id']
                        self.logger.info("✅ Emergency stop loss placed")
                    except:
                        self.logger.error("❌ Failed to place emergency SL, closing position!")
                        self.close_position("Emergency - No Stop Loss")
            
            # Check 2: Orders without position
            elif actual_position['size'] == 0 and open_orders:
                self.logger.warning(f"⚠️ Health check: {len(open_orders)} orphan orders found!")
                self.cleanup_and_reset_state("Health Check - Orphan Orders")
            
            # Check 3: State mismatch
            if (actual_position['size'] > 0 and not self.in_position) or \
            (actual_position['size'] == 0 and self.in_position):
                self.logger.warning("⚠️ State mismatch detected!")
                self.sync_state_with_exchange()
                
        except Exception as e:
            self.logger.error(f"Health check error: {e}")

    def cleanup_all_orders_for_symbol(self, max_attempts: int = 3):
        """Force cleanup ALL orders for this symbol"""
        for attempt in range(max_attempts):
            try:
                open_orders = self.trader.get_open_orders(self.symbol)
                
                if not open_orders:
                    self.logger.info("✅ No open orders to clean")
                    return True
                
                self.logger.warning(f"🧹 Found {len(open_orders)} orphan orders, cleaning...")
                
                for order in open_orders:
                    try:
                        self.trader.cancel_order(self.symbol, order['orderId'])
                        self.logger.info(f"   Cancelled {order['type']} order {order['orderId']}")
                    except Exception as e:
                        self.logger.debug(f"   Failed to cancel {order['orderId']}: {e}")
                
                # Wait and verify
                time.sleep(1.0)
                
                # Check again
                remaining = self.trader.get_open_orders(self.symbol)
                if not remaining:
                    self.logger.info("✅ All orders cleaned successfully")
                    return True
                
            except Exception as e:
                self.logger.error(f"Cleanup attempt {attempt+1} failed: {e}")
            
            time.sleep(1.0)
        
        return False

    def close_position(self, reason: str = "Signal"):
        """Close current position with PROPER SIZE VERIFICATION"""
        if not self.in_position:
            return
        
        try:
            # CRITICAL: Get ACTUAL position size from exchange
            actual_position = self.trader.get_position(self.symbol)
            
            if actual_position['size'] == 0:
                self.logger.warning("⚠️ No actual position found, cleaning up state")
                self.cleanup_and_reset_state(reason)
                return
                
            # Use ACTUAL size, not tracked size
            actual_size = actual_position['size']
            actual_side = actual_position['side']
            
            # Log if there's a mismatch
            if actual_size != self.position_size:
                self.logger.warning(f"⚠️ Position size mismatch! Tracked: {self.position_size}, Actual: {actual_size}")
            
            if actual_side != self.position_side:
                self.logger.warning(f"⚠️ Position side mismatch! Tracked: {self.position_side}, Actual: {actual_side}")
            
            # Close position with ACTUAL size
            if actual_side == 'LONG':
                order = self.trader.market_sell(self.symbol, actual_size)  # Use actual_size
            else:  # SHORT
                order = self.trader.market_buy(self.symbol, actual_size)   # Use actual_size
            
            exit_price = order['fill_price']
            
            # Calculate P&L using actual entry price if available
            entry_price = actual_position.get('entry_price', self.entry_price)
            if actual_side == 'LONG':
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            else:  # SHORT
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100
            
            self.logger.info(f"✅ Closed {actual_side} @ ${exit_price:.5f}")
            self.logger.info(f"   Size: {actual_size} (actual from exchange)")
            self.logger.info(f"   Entry: ${entry_price:.5f}")
            self.logger.info(f"   P&L: {pnl_pct:+.2f}% | Reason: {reason}")
            
            # Wait for position to be fully closed
            time.sleep(1.5)
            
            # Verify position is closed
            verify_position = self.trader.get_position(self.symbol)
            if verify_position['size'] > 0:
                self.logger.error(f"❌ Position still open after close! Remaining: {verify_position['size']}")
                # Try one more time
                if verify_position['side'] == 'LONG':
                    self.trader.market_sell(self.symbol, verify_position['size'])
                else:
                    self.trader.market_buy(self.symbol, verify_position['size'])
                time.sleep(1.0)
            
            # Use centralized cleanup
            self.cleanup_and_reset_state(reason)
            
        except Exception as e:
            self.logger.error(f"❌ Position close error: {e}")
            import traceback
            traceback.print_exc()
            # Force cleanup on error
            self.cleanup_and_reset_state(f"{reason} (Error)")

    def sync_state_with_exchange(self):
        """Sync internal state with actual exchange state - ENHANCED WITH DUST HANDLING"""
        try:
            # Get real position
            actual_position = self.trader.get_position(self.symbol)
            
            # Get real orders
            actual_orders = self.trader.get_open_orders(self.symbol)
            
            # Check for dust position first
            if actual_position['size'] > 0:
                current_price = actual_position.get('mark_price', self.trader.get_current_price(self.symbol))
                
                if self.is_dust_position(actual_position['size'], current_price):
                    self.logger.warning(f"⚠️ Dust position detected during sync: {actual_position['size']}")
                    
                    # Clean it up
                    if self.cleanup_dust_position():
                        return  # State already cleaned
            
            # Sync position state
            if actual_position['size'] == 0:
                if self.in_position:
                    self.logger.warning("⚠️ State mismatch: Internal shows position but exchange shows none")
                    # Use centralized cleanup
                    self.cleanup_and_reset_state("State Sync - No Position")
                    
            else:
                # Update internal state to match exchange
                self.in_position = True
                self.position_side = actual_position['side']
                self.position_size = actual_position['size']
                self.entry_price = actual_position['entry_price']
                
                # 🆕 ADD: Set conservative pyramid timing after sync
                if self.last_entry_timestamp == 0:
                    # Assume position was just opened, prevent immediate pyramid
                    self.last_entry_timestamp = time.time()
                    self.logger.info("⚠️ Pyramid timing reset due to sync - will wait before allowing pyramid")
                    
                    # Initialize basic history
                    self.pyramid_history = [{
                        'timestamp': self.last_entry_timestamp,
                        'price': self.entry_price,
                        'level': 1,
                        'size': self.position_size,
                        'side': self.position_side
                    }]
                    
                    # Estimate pyramid count (conservative)
                    self.pyramid_count = 1
            
        except Exception as e:
            self.logger.error(f"State sync error: {e}")

    def sync_position_info_only(self):
        """Sync only position size/side without resetting order state"""
        try:
            # Get actual position from exchange
            actual_position = self.trader.get_position(self.symbol)
            
            if actual_position['size'] > 0:
                # Only update position data
                old_size = self.position_size
                self.position_size = actual_position['size']
                self.position_side = actual_position['side']
                
                # Update entry price if needed (for average price after pyramid)
                if actual_position.get('entry_price', 0) > 0:
                    self.entry_price = actual_position['entry_price']
                
                # Log if size changed
                if old_size != self.position_size:
                    self.logger.info(f"📊 Position size updated: {old_size} → {self.position_size}")
                    
                # IMPORTANT: Don't touch these!
                # - self.stop_loss_order_id
                # - self.tp1_order_id
                # - self.tp2_order_id
                # - self.tp1_hit
                # - self.tp2_hit
                # - self.sl_moved_to_breakeven
                
            elif actual_position['size'] == 0 and self.in_position:
                # Position closed unexpectedly
                self.logger.warning("⚠️ Position closed but state shows open!")
                # Still don't reset order IDs - let cleanup_and_reset_state handle it
                
        except Exception as e:
            self.logger.error(f"Position info sync error: {e}")

    def process_signal(self, signal: Dict):
        """Enhanced signal processing with technical zone analysis"""
        decision = signal['decision']
        
        # Get current price
        current_price = self.trader.get_current_price(self.symbol)
        
        # Analyze technical zones
        zone_analysis = self.analyze_technical_zone(current_price, decision)
        
        # Log signal interpretation
        if decision > self.STRONG_BUY:
            signal_type = "STRONG BUY"
        elif decision > self.BUY:
            signal_type = "BUY"
        elif decision < self.STRONG_SELL:
            signal_type = "STRONG SELL"
        elif decision < self.SELL:
            signal_type = "SELL"
        else:
            signal_type = "NEUTRAL"
        
        self.logger.info(f"📈 Signal: {signal_type} (decision: {decision:.3f})")
        self.logger.info(f"🎯 Zone: {zone_analysis['zone']} | Recommendation: {zone_analysis['recommendation']}")
        
        # Display key levels periodically
        if not hasattr(self, '_last_levels_display') or time.time() - self._last_levels_display > 300:
            self.display_key_levels()
            self._last_levels_display = time.time()
        
        # Check stop loss first (existing code)
        if self.in_position:
            # ... (existing stop loss check code)
            pass
        
        # Enhanced decision logic based on signal + technical analysis
        effective_signal = signal_type
        
        # Override signal based on technical zone analysis
        if zone_analysis['recommendation'] == 'strong_buy' and decision > self.BUY:
            effective_signal = "STRONG BUY"
            self.logger.info(f"📊 Signal upgraded to STRONG BUY due to technical zone")
        elif zone_analysis['recommendation'] == 'strong_sell' and decision < self.SELL:
            effective_signal = "STRONG SELL"
            self.logger.info(f"📊 Signal upgraded to STRONG SELL due to technical zone")
        elif zone_analysis['recommendation'] == 'wait':
            effective_signal = "NEUTRAL"
            self.logger.warning(f"⏳ Signal downgraded to NEUTRAL - waiting for better setup")
        elif zone_analysis['recommendation'] == 'neutral' and zone_analysis['confidence'] < 0.5:
            effective_signal = "NEUTRAL"
            self.logger.info(f"📊 Weak setup - staying neutral")
        
        # Execute based on effective signal
        if effective_signal == "STRONG BUY":
            if self.position_side == 'SHORT':
                self.close_position("Flip to Long")
                if not self.in_position and self.check_resistance:
                    if not self.check_resistance_proximity(current_price, signal_type):
                        self.logger.info("📊 Strong buy rejected due to resistance")
                        return
                self.execute_trade('BUY', 'strong')
            else:
                if self.in_position and self.position_side == 'LONG' and self.check_resistance:
                    if not self.check_resistance_proximity(current_price, signal_type):
                        self.logger.warning("⚠️ Skipping pyramid - too close to resistance")
                        return
                self.execute_trade('BUY', 'strong')
                
        elif effective_signal == "BUY":
            if self.position_side == 'SHORT':
                self.close_position("Buy Signal")
            elif not self.in_position:
                if self.check_resistance:
                    if not self.check_resistance_proximity(current_price, signal_type):
                        self.logger.info("📊 Buy signal rejected due to resistance")
                        return
                self.execute_trade('BUY', 'normal')
                
        elif effective_signal == "STRONG SELL":
            if self.position_side == 'LONG':
                self.close_position("Flip to Short")
                self.execute_trade('SELL', 'strong')
            else:
                self.execute_trade('SELL', 'strong')
                
        elif effective_signal == "SELL":
            if self.position_side == 'LONG':
                self.close_position("Sell Signal")
            elif not self.in_position:
                self.execute_trade('SELL', 'normal')
        
        # For NEUTRAL, do nothing (existing positions maintain)
    
    def run(self, interval_seconds: int = 60):
        """Main trading loop with COMPLETE SAFETY CHECKS"""
        self.logger.info(f"🚀 Starting Live Trader - ENHANCED SAFETY VERSION")
        self.logger.info(f"   Symbol: {self.symbol} | Leverage: {self.leverage}x")
        
        self.log_trading_rules()

        # Connect to exchange
        if not self.trader.connect():
            self.logger.error("Failed to connect to Binance")
            return
        
        # Set leverage
        self.trader.set_leverage(self.symbol, self.leverage)
        
        # Initial sync
        self.sync_state_with_exchange()
        
        # 🔧 ADD: Initial cleanup
        if not self.in_position:
            self.logger.info("🧹 Initial cleanup check...")
            self.pre_trade_cleanup()
        
        sync_counter = 0
        health_check_counter = 0
        
        try:
            while True:
                try:
                    # Periodic state sync every 10 cycles
                    sync_counter += 1
                    if sync_counter >= 10:
                        self.sync_state_with_exchange()
                        sync_counter = 0
                    
                    # 🔧 ADD: Health check every 5 cycles
                    health_check_counter += 1
                    if health_check_counter >= 5:
                        self.health_check()
                        health_check_counter = 0
                    if not self.in_position:
                        try:
                            orphan_orders = self.trader.get_open_orders(self.symbol)
                            if orphan_orders:
                                self.logger.warning(f"🧹 Found {len(orphan_orders)} orphan orders")
                                for order in orphan_orders:
                                    try:
                                        self.trader.cancel_order(self.symbol, order['orderId'])
                                        self.logger.info(f"✅ Cleaned orphan {order['type']}")
                                    except:
                                        pass
                        except:
                            pass
                    
                    if self.in_position:
                        # Quick dust check
                        try:
                            actual_pos = self.trader.get_position(self.symbol)
                            if actual_pos['size'] > 0:
                                current_price = self.trader.get_current_price(self.symbol)
                                if self.is_dust_position(actual_pos['size'], current_price):
                                    self.logger.warning("⚠️ Dust position detected in main loop")
                                    self.cleanup_dust_position()
                                    continue  # Skip to next cycle
                        except:
                            pass


                    # Check TP status if in position
                    if self.in_position:
                        self.check_take_profit_status()
                        
                        # Check trailing profit protection
                        current_price = self.trader.get_current_price(self.symbol)
                        if self.check_trailing_profit_protection(current_price):
                            self.logger.warning("📉 TRAILING STOP TRIGGERED!")
                            self.close_position("Trailing Profit Protection (70% of TP1)")
                            continue

                    # Get signal and execute immediately
                    signal = self.model.get_signal(self.symbol)
                    signal['timestamp'] = time.time()
                    
                    self.logger.info(f"\n📊 Signal: {signal['action']} ({signal['decision']:.3f})")
                    
                    # Execute immediately - NO DELAY
                    self.process_signal(signal)
                    
                    # Show position status
                    if self.in_position:
                        current_price = self.trader.get_current_price(self.symbol)
                        self.logger.info(f"💼 Position: {self.position_side} @ ${current_price:.2f}")
                        
                        # 🔧 ADD: Verify stop loss still exists
                        open_orders = self.trader.get_open_orders(self.symbol)
                        has_stop_loss = any(o['orderId'] == self.stop_loss_order_id for o in open_orders)
                        
                        if not has_stop_loss and self.stop_loss_order_id:
                            self.logger.warning("⚠️ Stop loss order not found in open orders!")
                            # Try to place new stop loss
                            try:
                                formatted_price = self.format_price_for_api(self.stop_loss_price)
                                sl_order = self.place_stop_loss_order_with_verification(
                                    symbol=self.symbol,
                                    side=self.position_side,
                                    quantity=int(self.position_size),
                                    stop_price=formatted_price
                                )
                                self.stop_loss_order_id = sl_order['order_id']
                                self.logger.info("✅ Stop loss re-placed successfully")
                            except:
                                self.logger.error("❌ Failed to re-place stop loss!")
                                self.close_position("Missing Stop Loss")
                    
                    # Wait for next cycle
                    time.sleep(interval_seconds)
                        
                except KeyboardInterrupt:
                    self.logger.info("\n⏹️ Stopping...")
                    break
                except Exception as e:
                    self.logger.error(f"❌ Loop error: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # 🔧 ADD: Emergency sync after error
                    try:
                        self.sync_state_with_exchange()
                    except:
                        pass
                    
                    time.sleep(interval_seconds)
        
        finally:
            self.logger.info("🧹 Final cleanup before shutdown...")
            
            # Close any open position
            if self.in_position:
                self.close_position("Shutdown")
            
            # 🔧 ADD: Final cleanup of any remaining orders
            try:
                remaining_orders = self.trader.get_open_orders(self.symbol)
                if remaining_orders:
                    self.logger.warning(f"⚠️ Found {len(remaining_orders)} orders at shutdown")
                    for order in remaining_orders:
                        try:
                            self.trader.cancel_order(self.symbol, order['orderId'])
                            self.logger.info(f"🧹 Cancelled {order['type']} order at shutdown")
                        except:
                            pass
            except:
                pass
            
            self.trader.disconnect()
            self.logger.info("✅ Trader stopped and cleaned up successfully")


# Configuration template
def get_default_config():
    """Get default configuration template"""
    return {
        # Required
        'api_key': 'YOUR_API_KEY',
        'api_secret': 'YOUR_API_SECRET',
        'symbol': 'ETHUSDT',
        'model_path': 'models/your_model.pth',
        
        # Optional (with defaults)
        'leverage': 20,
        'position_pct_normal': 0.02,  # 2% for normal signals
        'position_pct_strong': 0.05,  # 5% for strong signals
        'stop_loss_pct': 0.002,       # 0.2% default
        'use_dynamic_sl': True,       # Dynamic stop loss
        
        # Take Profit Configuration
        'use_take_profit': True,
        'tp1_percent': 0.005,         # 0.5% for TP1
        'tp2_percent': 0.01,          # 1% for TP2
        'tp1_size_ratio': 0.5,        # Close 50% at TP1
    }


# Usage example
if __name__ == "__main__":
    # Configure your settings
    config = {
        'api_key': 'EduyybaFGjUpSkR7q2J0HwHjHF6dB8TB5klAAUX8Ukum2Yz1jR2J8osZVXz9kxZC',
        'api_secret': 'QmAxhDG4QYxdrif38WyQ6uvGLv5OZvlGPIRBzdtFWry7adtRNzGFY8HlLkOSLOyY',
        'symbol': 'HUUSDT',
        'model_path': 'models/trading_lstm_20250630_042222.pth',
        'leverage': 50,
        'position_pct_normal': 0.04,  # 1% for normal
        'position_pct_strong': 0.1,   # 3% for strong
        'stop_loss_pct': 0.003,       # 0.3%
        
        # Take Profit Configuration
        'use_take_profit': True,
        'tp1_percent': 0.003,         # 0.3% TP1
        'tp2_percent': 0.008,         # 0.8% TP2
        'tp1_size_ratio': 0.6,        # 60% position at TP1, 40% at TP2

        # Technical Analysis Configuration (NEW)
        'use_technical_levels': True,      # Enable technical S/R and Fibo
        'lookback_candles': 200,           # 200 x 5min = ~16 hours data
        'max_risk_technical': 0.02,        # Max 2% risk with technical
        'min_rr_ratio': 1.5,               # Min 1:1.5 risk/reward
        'fallback_to_fixed': True,         # Use fixed if no good levels
    }
    
    # Create and run trader
    trader = LiveScalpingTrader(config)
    
    # Run with 30 second intervals for scalping
    trader.run(interval_seconds=30)