"""
Binance Trading Library - Simple Version
File: binance_lib.py
"""

import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
import time
from typing import Dict, List, Optional
import logging

import asyncio
import warnings

# Suppress asyncio deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="binance.helpers")

# Set event loop policy for Windows
if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class BinanceTrader:
    """Simple Binance Trading Library"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """Initialize trader"""
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client = None
        self.connected = False
        
        # Trading parameters
        self.trading_fee = 0.001  # 0.1%
        self.spread = 0.0003      # 0.03%
        
    def universal_price_formatter(price, tick_size):
        """
        Universal price formatter untuk ALL Binance price ranges
        Handles: 0.000001 to 1,000,000+ dengan perfect precision
        ZERO scientific notation risk
        """
        from decimal import Decimal, ROUND_DOWN
        
        try:
            # Step 1: Decimal arithmetic for perfect precision
            price_decimal = Decimal(str(price))
            tick_decimal = Decimal(str(tick_size))
            
            # Step 2: Round to tick_size
            steps = price_decimal / tick_decimal
            rounded_steps = steps.quantize(Decimal('1'), rounding=ROUND_DOWN)
            formatted_decimal = rounded_steps * tick_decimal
            
            # Step 3: Dynamic decimal places
            if tick_size >= 1:
                decimal_places = 0
            else:
                # Count decimal places from tick_size
                tick_str = str(tick_decimal)
                if 'E' in tick_str or 'e' in tick_str:
                    # Handle scientific notation in tick_size
                    tick_float = float(tick_decimal)
                    tick_str = f"{tick_float:.15f}".rstrip('0')
                
                if '.' in tick_str:
                    decimal_places = len(tick_str.split('.')[1])
                else:
                    decimal_places = 0
            
            # Step 4: Format (GUARANTEED NO SCIENTIFIC)
            if decimal_places == 0:
                result = str(int(formatted_decimal))
            else:
                result = f"{float(formatted_decimal):.{decimal_places}f}"
            
            # Step 5: Final validation
            if 'e' in result.lower() or 'E' in result:
                # Emergency fallback
                if float(formatted_decimal) < 0.001:
                    result = f"{float(formatted_decimal):.12f}".rstrip('0')
                    if result.endswith('.'):
                        result += '0'
                else:
                    result = str(float(formatted_decimal))
            
            return result
            
        except Exception as e:
            # Ultimate fallback
            safe_result = f"{price:.8f}".rstrip('0')
            if safe_result.endswith('.'):
                safe_result += '0'
            return safe_result

    
    # === CONNECTION METHODS ===
    def connect(self) -> bool:
        """Connect to Binance"""
        try:
            logging.info(f"🔌 Connecting to Binance {'Testnet' if self.testnet else 'Live'}...")
            
            # Create client
            self.client = Client(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet
            )
            
            # Test connection
            self.client.ping()
            self.connected = True
            
            logging.info("✅ Connected successfully!")
            return True
            
        except Exception as e:
            logging.error(f"❌ Connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from Binance"""
        self.client = None
        self.connected = False
        logging.info("🔌 Disconnected from Binance")
        return True
    
    def test_connection(self) -> bool:
        """Test if connection is alive"""
        if not self.client:
            return False
        
        try:
            self.client.ping()
            return True
        except:
            return False
    
    # === DATA METHODS ===
    def get_multi_timeframe_data(self, symbol: str, timeframes: List[str], limit: int = 200) -> Dict:
        """
        Get klines for multiple timeframes
        
        Args:
            symbol: 'BTCUSDT'
            timeframes: ['1m', '5m', '30m', '1h']
            limit: number of candles
        """
        if not self.connected:
            raise Exception("Not connected to Binance")
            
        logging.info(f"📊 Fetching {symbol} data for {timeframes}...")
        
        multi_tf_data = {}
        
        for tf in timeframes:
            try:
                # Get klines
                klines = self.client.futures_klines(
                    symbol=symbol,
                    interval=tf,
                    limit=limit
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                # Convert types
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                
                # Set timestamp index
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp')
                
                multi_tf_data[tf] = df
                logging.info(f"✅ {tf}: {len(df)} candles")
                
            except Exception as e:
                logging.error(f"❌ Error fetching {tf}: {e}")
                
        return multi_tf_data
    
    def create_trading_features(self, multi_tf_data: Dict) -> pd.DataFrame:
        """
        Create features from multi-timeframe data with proper time alignment
        and OHLC integrity check
        
        Output per minute:
        - 1m: OHLC dari 1m 
        - 5m/30m/1h: OHL dari timeframe tersebut (adjusted), Close dari 1m
        """
        # Get base 1m data
        if '1m' not in multi_tf_data:
            raise ValueError("Need 1m timeframe as base")
        
        base_df = multi_tf_data['1m'].copy()
        features = pd.DataFrame(index=base_df.index)
        
        # Add 1m OHLC (keep original)
        features['open_1m'] = base_df['open']
        features['high_1m'] = base_df['high']
        features['low_1m'] = base_df['low']
        features['close_1m'] = base_df['close']
        features['volume_1m'] = base_df['volume']
        
        # Process each higher timeframe
        for tf in ['5m', '30m', '1h']:
            if tf not in multi_tf_data:
                continue
                
            tf_df = multi_tf_data[tf]
            
            # Create empty columns
            features[f'open_{tf}'] = np.nan
            features[f'high_{tf}'] = np.nan
            features[f'low_{tf}'] = np.nan
            features[f'close_{tf}'] = base_df['close']  # Always use 1m close
            features[f'volume_{tf}'] = np.nan
            
            # Align each 1m timestamp to appropriate timeframe
            for timestamp in base_df.index:
                # Floor timestamp based on timeframe
                if tf == '5m':
                    floor_ts = timestamp.floor('5min')
                elif tf == '30m':
                    floor_ts = timestamp.floor('30min')
                elif tf == '1h':
                    floor_ts = timestamp.floor('1h')
                
                # Find matching candle in higher timeframe
                if floor_ts in tf_df.index:
                    features.loc[timestamp, f'open_{tf}'] = tf_df.loc[floor_ts, 'open']
                    features.loc[timestamp, f'high_{tf}'] = tf_df.loc[floor_ts, 'high']
                    features.loc[timestamp, f'low_{tf}'] = tf_df.loc[floor_ts, 'low']
                    features.loc[timestamp, f'volume_{tf}'] = tf_df.loc[floor_ts, 'volume']
                else:
                    # Find nearest previous candle
                    valid_times = tf_df.index[tf_df.index <= floor_ts]
                    if len(valid_times) > 0:
                        nearest_ts = valid_times[-1]
                        features.loc[timestamp, f'open_{tf}'] = tf_df.loc[nearest_ts, 'open']
                        features.loc[timestamp, f'high_{tf}'] = tf_df.loc[nearest_ts, 'high']
                        features.loc[timestamp, f'low_{tf}'] = tf_df.loc[nearest_ts, 'low']
                        features.loc[timestamp, f'volume_{tf}'] = tf_df.loc[nearest_ts, 'volume']
        
        # Forward fill any remaining NaN values
        features = features.ffill()
        
        # OHLC Integrity Check - Update High/Low if Close exceeds boundaries
        for tf in ['5m', '30m', '1h']:
            if f'high_{tf}' in features.columns:
                # Update High if Close is higher
                mask_high = features[f'close_{tf}'] > features[f'high_{tf}']
                features.loc[mask_high, f'high_{tf}'] = features.loc[mask_high, f'close_{tf}']
                
                # Update Low if Close is lower
                mask_low = features[f'close_{tf}'] < features[f'low_{tf}']
                features.loc[mask_low, f'low_{tf}'] = features.loc[mask_low, f'close_{tf}']
                
                # Log adjustments
                high_adjusted = mask_high.sum()
                low_adjusted = mask_low.sum()
                if high_adjusted > 0 or low_adjusted > 0:
                    logging.info(f"   {tf}: Adjusted {high_adjusted} highs, {low_adjusted} lows for OHLC integrity")
        
        logging.info(f"✅ Created features: {features.shape[1]} columns, {len(features)} rows")
        return features

    def format_price(self, symbol: str, price: float) -> float:
        """Format price according to symbol's tick size"""
        try:
            exchange_info = self.get_exchange_info(symbol)
            tick_size = exchange_info.get('tick_size', 0.00001)
            
            from decimal import Decimal, ROUND_DOWN
            d = Decimal(str(tick_size))
            return float(Decimal(str(price)).quantize(d, rounding=ROUND_DOWN))
        except:
            return round(price, 5)  # Default 5 decimal places

    # === ACCOUNT METHODS ===
    def get_account_info(self) -> Dict:
        """Get account information"""
        if not self.connected:
            raise Exception("Not connected to Binance")
            
        try:
            account = self.client.futures_account()
            return {
                'total_balance': float(account['totalWalletBalance']),
                'available_balance': float(account['availableBalance']),
                'total_unrealized_pnl': float(account['totalUnrealizedProfit']),
                'total_margin_balance': float(account['totalMarginBalance'])
            }
        except Exception as e:
            logging.error(f"Error getting account info: {e}")
            return {}
    
    def get_balance(self) -> float:
        """Get USDT balance"""
        account_info = self.get_account_info()
        return account_info.get('available_balance', 0.0)
    
    def get_position(self, symbol: str) -> Dict:
        """Get current position for symbol"""
        if not self.connected:
            raise Exception("Not connected to Binance")
            
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            
            for pos in positions:
                if pos['symbol'] == symbol:
                    position_amt = float(pos['positionAmt'])
                    
                    # Safe get with fallback values
                    current_leverage = int(pos.get('leverage', 1))
                    entry_price = float(pos.get('entryPrice', 0))
                    mark_price = float(pos.get('markPrice', 0))
                    unrealized_pnl = float(pos.get('unRealizedProfit', 0))  # Note: unRealizedProfit not unrealizedProfit
                    
                    if position_amt == 0:
                        return {
                            'size': 0, 
                            'side': None,
                            'leverage': current_leverage,
                            'entry_price': 0,
                            'mark_price': mark_price,
                            'unrealized_pnl': 0
                        }
                    
                    return {
                        'size': abs(position_amt),
                        'side': 'LONG' if position_amt > 0 else 'SHORT',
                        'entry_price': entry_price,
                        'mark_price': mark_price,
                        'unrealized_pnl': unrealized_pnl,
                        'margin_used': float(pos.get('initialMargin', 0)),
                        'leverage': current_leverage
                    }
                    
            return {'size': 0, 'side': None, 'leverage': 1, 'entry_price': 0, 'mark_price': 0, 'unrealized_pnl': 0}
            
        except Exception as e:
            logging.error(f"Error getting position: {e}")
            # Debug: print actual response structure
            try:
                positions = self.client.futures_position_information(symbol=symbol)
                logging.error(f"Position response keys: {list(positions[0].keys()) if positions else 'Empty'}")
            except:
                pass
            return {'size': 0, 'side': None, 'leverage': 1, 'entry_price': 0, 'mark_price': 0, 'unrealized_pnl': 0}
    
    def get_leverage(self, symbol: str) -> int:
        """Get current leverage for symbol"""
        try:
            # ✅ BENAR - Pakai futures_account
            account = self.client.futures_account()
            for pos in account['positions']:
                if pos['symbol'] == symbol:
                    return int(pos.get('leverage', 1))
            return 1
        except Exception as e:
            logging.error(f"Error getting leverage: {e}")
            return 1
        
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for symbol"""
        if not self.connected:
            raise Exception("Not connected to Binance")
            
        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            logging.info(f"✅ Leverage set to {leverage}x for {symbol}")
            return True
        except Exception as e:
            logging.error(f"Error setting leverage: {e}")
            return False
    
    # === TRADING METHODS ===
    def market_buy(self, symbol: str, quantity: float) -> Dict:
        """Place market buy order"""
        if not self.connected:
            raise Exception("Not connected to Binance")
            
        try:
            # Place order
            order = self.client.futures_create_order(
                symbol=symbol,
                side='BUY',
                type='MARKET',
                quantity=quantity
            )
            
            # Get actual fill price - market order should be filled immediately
            order_id = order['orderId']
            
            # Wait a bit for order to be processed
            time.sleep(0.5)
            
            # Get order details with fill price
            order_detail = self.client.futures_get_order(
                symbol=symbol,
                orderId=order_id
            )
            
            fill_price = float(order_detail['avgPrice'])
            
            # If still 0, use current market price as fallback
            if fill_price == 0:
                fill_price = self.get_current_price(symbol)
                
            fee = quantity * fill_price * self.trading_fee
            
            logging.info(f"🔵 MARKET BUY: {quantity} {symbol} @ ${fill_price}")
            
            return {
                'order_id': order_id,
                'fill_price': fill_price,
                'quantity': quantity,
                'fee': fee,
                'status': order_detail['status']
            }
            
        except Exception as e:
            logging.error(f"Error market buy: {e}")
            raise
    
    def market_sell(self, symbol: str, quantity: float) -> Dict:
        """Place market sell order"""
        if not self.connected:
            raise Exception("Not connected to Binance")
            
        try:
            # Place order
            order = self.client.futures_create_order(
                symbol=symbol,
                side='SELL',
                type='MARKET',
                quantity=quantity
            )
            
            # Get actual fill price
            order_id = order['orderId']
            
            # Wait a bit for order to be processed
            time.sleep(0.5)
            
            # Get order details with fill price
            order_detail = self.client.futures_get_order(
                symbol=symbol,
                orderId=order_id
            )
            
            fill_price = float(order_detail['avgPrice'])
            
            # If still 0, use current market price as fallback
            if fill_price == 0:
                fill_price = self.get_current_price(symbol)
                
            fee = quantity * fill_price * self.trading_fee
            
            logging.info(f"🔴 MARKET SELL: {quantity} {symbol} @ ${fill_price}")
            
            return {
                'order_id': order_id,
                'fill_price': fill_price,
                'quantity': quantity,
                'fee': fee,
                'status': order_detail['status']
            }
            
        except Exception as e:
            logging.error(f"Error market sell: {e}")
            raise
    
    # === MARKET DATA METHODS ===
    def get_current_price(self, symbol: str) -> float:
        """Get current mark price"""
        if not self.connected:
            raise Exception("Not connected to Binance")
            
        try:
            ticker = self.client.futures_mark_price(symbol=symbol)
            return float(ticker['markPrice'])
        except Exception as e:
            logging.error(f"Error getting price: {e}")
            return 0.0
    
    def get_24hr_stats(self, symbol: str) -> Dict:
        """Get 24hr statistics"""
        if not self.connected:
            raise Exception("Not connected to Binance")
            
        try:
            stats = self.client.futures_ticker(symbol=symbol)
            return {
                'high': float(stats['highPrice']),
                'low': float(stats['lowPrice']),
                'volume': float(stats['volume']),
                'price_change_percent': float(stats['priceChangePercent'])
            }
        except Exception as e:
            logging.error(f"Error getting 24hr stats: {e}")
            return {}
    
    # === UTILS ===
    def calculate_position_size(self, balance: float, percent: float, leverage: int, price: float, symbol: str = None) -> float:
        """
        Calculate position size in coin amount based on percentage of balance
        
        Args:
            balance: Account balance in USDT
            percent: Percentage of balance to use as MARGIN
            leverage: Leverage to use
            price: Current coin price
            symbol: Trading symbol
        
        Returns:
            Coin amount to trade (properly formatted)
        """
        # Step 1: Calculate margin
        margin = balance * percent
        
        # Step 2: Calculate total position value
        position_value = margin * leverage
        
        # Step 3: Convert to coin amount
        coin_amount = position_value / price
        
        # Step 4: Get exchange info for precision
        if symbol:
            exchange_info = self.get_exchange_info(symbol)
        else:
            logging.warning("No symbol provided, using default precision")
            exchange_info = {
                'step_size': 0.001,
                'min_notional': 10.0
            }
        
        step_size = exchange_info.get('step_size', 0.001)
        min_notional = exchange_info.get('min_notional', 10.0)
        
        # Step 5: CRITICAL - Format quantity correctly
        # If step_size = 1, quantity must be integer
        if step_size >= 1:
            # Round to nearest integer
            coin_amount = int(coin_amount)
        else:
            # Round to proper decimal places
            from decimal import Decimal, ROUND_DOWN
            
            # Convert to string to avoid float precision issues
            step_str = f"{step_size:.10f}".rstrip('0')
            decimal_places = len(step_str.split('.')[-1]) if '.' in step_str else 0
            
            # Round down to avoid exceeding balance
            coin_amount = round(coin_amount - (0.1 ** decimal_places), decimal_places)
            
            # Alternative method using Decimal for precision
            d = Decimal(str(step_size))
            coin_amount = float(Decimal(str(coin_amount)).quantize(d, rounding=ROUND_DOWN))
        
        # Step 6: Ensure minimum notional value
        if coin_amount * price < min_notional:
            coin_amount = (min_notional / price) * 1.1  # Add 10% buffer
            
            # Re-apply step size formatting
            if step_size >= 1:
                coin_amount = int(coin_amount) + 1  # Round up for minimum
            else:
                d = Decimal(str(step_size))
                coin_amount = float(Decimal(str(coin_amount)).quantize(d, rounding=ROUND_DOWN))
        
        # Step 7: Final validation - ensure it's a valid number
        if step_size >= 1:
            coin_amount = int(coin_amount)  # Force integer for whole number coins
        
        # Log for debugging
        logging.info(f"Position calculation for {symbol}:")
        logging.info(f"  Balance: ${balance:.2f}")
        logging.info(f"  Margin: ${margin:.2f} ({percent*100:.1f}%)")
        logging.info(f"  Position Value: ${position_value:.2f}")
        logging.info(f"  Price: ${price:.2f}")
        logging.info(f"  Step Size: {step_size}")
        logging.info(f"  Coin Amount: {coin_amount} (type: {type(coin_amount).__name__})")
        
        return coin_amount
    
    def get_exchange_info(self, symbol: str) -> Dict:
        """Get exchange trading rules"""
        if not self.connected:
            raise Exception("Not connected to Binance")
            
        try:
            info = self.client.futures_exchange_info()
            
            for s in info['symbols']:
                if s['symbol'] == symbol:
                    # Find LOT_SIZE filter
                    lot_size_filter = None
                    notional_filter = None
                    price_filter = None
                    
                    for f in s['filters']:
                        if f['filterType'] == 'LOT_SIZE':
                            lot_size_filter = f
                        elif f['filterType'] == 'MIN_NOTIONAL':
                            notional_filter = f
                        elif f['filterType'] == 'PRICE_FILTER':
                            price_filter = f
                    
                    if not lot_size_filter:
                        logging.error(f"No LOT_SIZE filter found for {symbol}")
                        return {}
                    
                    result = {
                        'min_qty': float(lot_size_filter.get('minQty', 0)),
                        'max_qty': float(lot_size_filter.get('maxQty', 0)),
                        'step_size': float(lot_size_filter.get('stepSize', 0.001)),
                    }
                    
                    if price_filter:
                        result['tick_size'] = float(price_filter.get('tickSize', 0.00001))
                    
                    if notional_filter:
                        result['min_notional'] = float(notional_filter.get('notional', 10))
                    
                    logging.info(f"Exchange info for {symbol}: {result}")
                    return result
                    
            logging.error(f"Symbol {symbol} not found in exchange info")
            return {}
            
        except Exception as e:
            logging.error(f"Error getting exchange info: {e}")
            return {}

    def place_stop_loss_order(self, symbol: str, side: str, quantity: float, stop_price) -> Dict:
        """ENHANCED stop loss order - NO SCIENTIFIC NOTATION"""
        try:
            # Get exchange info
            exchange_info = self.get_exchange_info(symbol)
            tick_size = float(exchange_info.get('tick_size', 0.00001))
            
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
            
            # ✅ QUANTITY FORMATTING
            step_size = float(exchange_info.get('step_size', 0.001))
            if step_size >= 1:
                formatted_quantity = str(int(quantity))
            else:
                formatted_quantity = str(quantity)
            
            # ✅ VALIDATION LOG
            logging.info(f"🔧 Stop Loss API Call:")
            logging.info(f"   Symbol: {symbol}")
            logging.info(f"   Stop Price: {formatted_stop_price} (type: {type(formatted_stop_price)})")
            logging.info(f"   Quantity: {formatted_quantity}")
            logging.info(f"   Tick Size: {tick_size}")
            
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
            
            # ✅ API CALL WITH STRING PARAMETERS
            order = self.client.futures_create_order(
                symbol=symbol,
                side=order_side,
                type='STOP_MARKET',
                stopPrice=str(formatted_stop_price),  # ✅ FORCE STRING
                quantity=str(formatted_quantity),     # ✅ FORCE STRING
                timeInForce='GTC',
                reduceOnly=True,
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

    def cancel_order(self, symbol: str, order_id: int) -> bool:
        """Cancel an order"""
        try:
            self.client.futures_cancel_order(
                symbol=symbol,
                orderId=order_id
            )
            logging.info(f"❌ Order {order_id} cancelled")
            return True
        except Exception as e:
            logging.error(f"Error cancelling order: {e}")
            return False

    def get_open_orders(self, symbol: str) -> List[Dict]:
        """Get all open orders for symbol"""
        try:
            orders = self.client.futures_get_open_orders(symbol=symbol)
            return orders
        except Exception as e:
            logging.error(f"Error getting open orders: {e}")
            return []

# ========== TEST CONNECTION ==========
def test_binance_connection():
    """Test function untuk cek connection"""
    
    print("🧪 BINANCE CONNECTION TEST")
    print("=" * 50)
    
    # ========== CONFIGURATION ==========
    # API Keys
    API_KEY = "EduyybaFGjUpSkR7q2J0HwHjHF6dB8TB5klAAUX8Ukum2Yz1jR2J8osZVXz9kxZC"
    API_SECRET = "QmAxhDG4QYxdrif38WyQ6uvGLv5OZvlGPIRBzdtFWry7adtRNzGFY8HlLkOSLOyY"
    
    # Trading Configuration
    SYMBOL = 'ETHUSDT'           # Symbol to trade
    TRADE_PERCENTAGE = 0.04      # 1% of balance (0.01 = 1%, 0.05 = 5%)
    LEVERAGE = 20                # Leverage to use
    # ===================================
    
    # Create trader instance
    trader = BinanceTrader(API_KEY, API_SECRET, testnet=False)
    
    # Test 1: Connection
    print("\n1. Testing connection...")
    if not trader.connect():
        print("❌ Connection failed!")
        return
    print("✅ Connected to Binance")
    
    # Test 2: Get account & market info
    print("\n2. Getting account & market info...")
    balance = trader.get_balance()
    price = trader.get_current_price(SYMBOL)
    leverage = trader.get_leverage(SYMBOL)
    
    print(f"✅ Account Balance: ${balance:.2f}")
    print(f"✅ {SYMBOL} Price: ${price:.2f}")
    print(f"✅ Current Leverage: {leverage}x")
    
    # Test 3: Calculate position
    print("\n3. Position calculation...")
    position_size = trader.calculate_position_size(balance, TRADE_PERCENTAGE, LEVERAGE, price)
    position_value = position_size * price
    margin_required = position_value / LEVERAGE
    
    print(f"📊 Trading Configuration:")
    print(f"   Symbol: {SYMBOL}")
    print(f"   Percentage: {TRADE_PERCENTAGE*100}%")
    print(f"   Leverage: {LEVERAGE}x")
    print(f"\n📊 Position Details:")
    print(f"   Position Size: {position_size:.4f} {SYMBOL.replace('USDT', '')}")
    print(f"   Position Value: ${position_value:.2f}")
    print(f"   Margin Required: ${margin_required:.2f}")
    
    # Test 4: Live trading test
    print(f"\n4. Live Trading Test")
    print("⚠️  WARNING: This will execute REAL trades!")
    print("   Press Enter to continue or Ctrl+C to exit")
    
    try:
        input()
        
        # BUY
        print(f"\n📘 Executing BUY {position_size:.4f} {SYMBOL}...")
        buy_order = trader.market_buy(SYMBOL, position_size)
        print(f"✅ Bought at ${buy_order['fill_price']}")
        
        # Wait and check position
        print("\n⏱️  Holding for 5 seconds...")
        time.sleep(15)
        
        position = trader.get_position(SYMBOL)
        print(f"📊 Current PNL: ${position.get('unrealized_pnl', 0):.2f}")
        
        # SELL
        print(f"\n📕 Executing SELL {position_size:.4f} {SYMBOL}...")
        sell_order = trader.market_sell(SYMBOL, position_size)
        print(f"✅ Sold at ${sell_order['fill_price']}")
        
        # Results
        pnl = (sell_order['fill_price'] - buy_order['fill_price']) * position_size
        fees = buy_order['fee'] + sell_order['fee']
        net_pnl = pnl - fees
        roi = (net_pnl / margin_required) * 100
        
        print(f"\n💰 RESULTS:")
        print(f"   Gross P&L: ${pnl:.4f}")
        print(f"   Fees: ${fees:.4f}")
        print(f"   Net P&L: ${net_pnl:.4f}")
        print(f"   ROI: {roi:.2f}%")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        # Emergency close any open position
        pos = trader.get_position(SYMBOL)
        if pos['size'] > 0:
            print("⚠️  Closing open position...")
            if pos['side'] == 'LONG':
                trader.market_sell(SYMBOL, pos['size'])
            else:
                trader.market_buy(SYMBOL, pos['size'])
    
    # Disconnect
    trader.disconnect()
    print("\n✅ Test completed!")

def test_trading_features_alignment():
    """Test function untuk cek alignment create_trading_features dengan visualisasi"""
    
    print("🧪 TESTING TRADING FEATURES ALIGNMENT")
    print("=" * 50)
    
    # ========== CONFIGURATION ==========
    # API Keys
    API_KEY = "EduyybaFGjUpSkR7q2J0HwHjHF6dB8TB5klAAUX8Ukum2Yz1jR2J8osZVXz9kxZC"
    API_SECRET = "QmAxhDG4QYxdrif38WyQ6uvGLv5OZvlGPIRBzdtFWry7adtRNzGFY8HlLkOSLOyY"
    
    # Test Configuration
    SYMBOL = 'ETHUSDT'          # Symbol to test
    DATA_LIMIT = 100            # Number of candles to fetch
    TIMEFRAMES = ['1m', '5m', '30m', '1h']  # Timeframes to test
    # ===================================
    
    # Create trader and connect
    trader = BinanceTrader(API_KEY, API_SECRET, testnet=False)
    
    print(f"\n📊 Test Configuration:")
    print(f"   Symbol: {SYMBOL}")
    print(f"   Data Limit: {DATA_LIMIT} candles")
    print(f"   Timeframes: {TIMEFRAMES}")
    
    # Connect
    if not trader.connect():
        print("❌ Connection failed!")
        return
    
    try:
        # Step 1: Get multi-timeframe data
        print(f"\n1. Fetching multi-timeframe data...")
        multi_tf_data = trader.get_multi_timeframe_data(SYMBOL, TIMEFRAMES, limit=DATA_LIMIT)
        
        # Step 2: Create features
        print(f"\n2. Creating trading features...")
        features = trader.create_trading_features(multi_tf_data)
        
        # Step 3: Verify alignment
        print(f"\n3. Verifying alignment...")
        print(f"   Features shape: {features.shape}")
        print(f"   Columns: {list(features.columns)}")
        
        # Check close alignment
        print(f"\n   Checking close price alignment:")
        for tf in ['5m', '30m', '1h']:
            if f'close_{tf}' in features.columns:
                matches = (features['close_1m'] == features[f'close_{tf}']).all()
                print(f"   - close_{tf} == close_1m: {'✅ YES' if matches else '❌ NO'}")
        
        # Step 3.5: Verify OHLC Integrity
        print(f"\n3.5. Verifying OHLC integrity...")
        integrity_issues = 0
        
        for tf in ['5m', '30m', '1h']:
            if all(col in features.columns for col in [f'high_{tf}', f'low_{tf}', f'close_{tf}']):
                # Check High >= Close
                high_violations = (features[f'close_{tf}'] > features[f'high_{tf}']).sum()
                
                # Check Low <= Close  
                low_violations = (features[f'close_{tf}'] < features[f'low_{tf}']).sum()
                
                # Count adjustments made
                high_adjusted = (features[f'close_{tf}'] == features[f'high_{tf}']).sum()
                low_adjusted = (features[f'close_{tf}'] == features[f'low_{tf}']).sum()
                
                print(f"\n   {tf} Integrity Check:")
                print(f"   - Close > High violations: {high_violations}")
                print(f"   - Close < Low violations: {low_violations}")
                print(f"   - Highs adjusted: {high_adjusted}")
                print(f"   - Lows adjusted: {low_adjusted}")
                
                if high_violations == 0 and low_violations == 0:
                    print(f"   ✅ OHLC integrity maintained")
                else:
                    print(f"   ❌ OHLC integrity issues found!")
                    integrity_issues += high_violations + low_violations
        
        print(f"\n   Total integrity issues: {integrity_issues}")
        
        # Step 4: Visualize
        print(f"\n4. Creating visualization...")
        
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import Rectangle
        
        fig, axes = plt.subplots(len(TIMEFRAMES), 1, figsize=(14, 3*len(TIMEFRAMES)), sharex=True)
        if len(TIMEFRAMES) == 1:
            axes = [axes]
        
        # Plot each timeframe
        for idx, tf in enumerate(TIMEFRAMES):
            ax = axes[idx]
            
            if tf == '1m':
                # Plot 1m data directly
                df = multi_tf_data['1m'].tail(50)  # Show last 50 for clarity
                
                # Candlestick plot
                for i in range(len(df)):
                    t = df.index[i]
                    o = df['open'].iloc[i]
                    h = df['high'].iloc[i]
                    l = df['low'].iloc[i]
                    c = df['close'].iloc[i]
                    
                    color = 'green' if c >= o else 'red'
                    # Body
                    ax.add_patch(Rectangle((mdates.date2num(t)-0.0003, min(o,c)), 
                                         0.0006, abs(c-o), 
                                         facecolor=color, edgecolor=color, alpha=0.8))
                    # Wick
                    ax.plot([mdates.date2num(t), mdates.date2num(t)], [l, h], 
                           color=color, linewidth=1)
                
                ax.set_title(f'{tf} - Original Data', fontsize=12, fontweight='bold')
                
            else:
                # Plot higher timeframe with 1m close overlay
                df_tf = multi_tf_data[tf].tail(20)  # Less candles for higher TF
                df_features = features.tail(50)
                
                # Plot original timeframe candles
                for i in range(len(df_tf)):
                    t = df_tf.index[i]
                    o = df_tf['open'].iloc[i]
                    h = df_tf['high'].iloc[i]
                    l = df_tf['low'].iloc[i]
                    c = df_tf['close'].iloc[i]
                    
                    # Determine candle width based on timeframe
                    if tf == '5m':
                        width = 0.003
                    elif tf == '30m':
                        width = 0.018
                    elif tf == '1h':
                        width = 0.036
                    
                    color = 'green' if c >= o else 'red'
                    # Body with transparency
                    ax.add_patch(Rectangle((mdates.date2num(t)-width/2, min(o,c)), 
                                         width, abs(c-o), 
                                         facecolor=color, edgecolor=color, alpha=0.3))
                    # Wick
                    ax.plot([mdates.date2num(t), mdates.date2num(t)], [l, h], 
                           color=color, linewidth=2, alpha=0.5)
                
                # Overlay 1m close as line
                ax.plot(df_features.index, df_features['close_1m'], 
                       'blue', linewidth=1.5, label='1m Close', alpha=0.8)
                
                # Overlay close_tf (should match 1m close)
                ax.plot(df_features.index, df_features[f'close_{tf}'], 
                       'orange', linewidth=1, linestyle='--', 
                       label=f'{tf} Close (from features)', alpha=0.8)
                
                # Mark adjusted highs and lows
                if f'high_{tf}' in df_features.columns and f'low_{tf}' in df_features.columns:
                    # Adjusted highs (where close == high)
                    adjusted_highs = df_features[f'close_{tf}'] == df_features[f'high_{tf}']
                    if adjusted_highs.any():
                        ax.scatter(df_features.index[adjusted_highs], 
                                  df_features[f'high_{tf}'][adjusted_highs],
                                  color='yellow', marker='^', s=50, 
                                  label='Adjusted High', zorder=5, edgecolors='black')
                    
                    # Adjusted lows (where close == low)
                    adjusted_lows = df_features[f'close_{tf}'] == df_features[f'low_{tf}']
                    if adjusted_lows.any():
                        ax.scatter(df_features.index[adjusted_lows], 
                                  df_features[f'low_{tf}'][adjusted_lows],
                                  color='yellow', marker='v', s=50, 
                                  label='Adjusted Low', zorder=5, edgecolors='black')
                
                ax.set_title(f'{tf} - OHL from {tf} + Close from 1m (with integrity check)', 
                           fontsize=12, fontweight='bold')
                ax.legend(loc='upper left', fontsize=8)
            
            # Formatting
            ax.grid(True, alpha=0.3)
            ax.set_ylabel('Price', fontsize=10)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        
        # Final formatting
        axes[-1].set_xlabel('Time', fontsize=10)
        plt.xticks(rotation=45)
        fig.suptitle(f'{SYMBOL} - Multi-Timeframe Alignment Test with OHLC Integrity\n'
                    f'Blue line = 1m Close | Orange dashed = Close from features | Yellow markers = Adjusted H/L',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Step 5: Show sample data
        print(f"\n5. Sample feature data (last 5 rows):")
        cols_to_show = []
        for tf in ['1m', '5m', '30m', '1h']:
            if tf == '1m':
                cols_to_show.extend(['open_1m', 'high_1m', 'low_1m', 'close_1m'])
            else:
                cols_to_show.extend([f'open_{tf}', f'high_{tf}', f'low_{tf}', f'close_{tf}'])
        
        # Filter existing columns
        cols_to_show = [col for col in cols_to_show if col in features.columns]
        print(features[cols_to_show].tail(5).round(2))
        
        # Step 6: Alignment statistics
        print(f"\n6. Alignment Statistics:")
        for tf in ['5m', '30m', '1h']:
            if all(col in features.columns for col in [f'open_{tf}', f'high_{tf}', f'low_{tf}']):
                # Count unique values (should change over time)
                n_unique_open = features[f'open_{tf}'].nunique()
                n_unique_high = features[f'high_{tf}'].nunique()
                n_unique_low = features[f'low_{tf}'].nunique()
                
                print(f"\n   {tf} Timeframe:")
                print(f"   - Unique open values: {n_unique_open}")
                print(f"   - Unique high values: {n_unique_high}")
                print(f"   - Unique low values: {n_unique_low}")
                
                # Check if values change appropriately
                if tf == '5m':
                    expected_changes = DATA_LIMIT // 5
                elif tf == '30m':
                    expected_changes = DATA_LIMIT // 30
                elif tf == '1h':
                    expected_changes = DATA_LIMIT // 60
                    
                print(f"   - Expected ~{expected_changes} different values")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        trader.disconnect()
        print("\n✅ Test completed!")


def generate_trading_features(symbol: str, api_key: str, api_secret: str, 
                            data_limit: int = 200, timeframes: List[str] = None) -> pd.DataFrame:
    """
    Generate trading features from multi-timeframe data
    
    Args:
        symbol: Trading symbol (e.g., 'ETHUSDT')
        api_key: Binance API key
        api_secret: Binance API secret
        data_limit: Number of candles to fetch per timeframe
        timeframes: List of timeframes (default: ['1m', '5m', '30m', '1h'])
    
    Returns:
        pd.DataFrame: Features with OHLCV for each timeframe, aligned to 1m
    """
    
    # Default timeframes
    if timeframes is None:
        timeframes = ['1m', '5m', '30m', '1h']
    
    # Create trader and connect
    trader = BinanceTrader(api_key, api_secret, testnet=False)
    
    print(f"📊 Generating features for {symbol}...")
    
    # Connect
    if not trader.connect():
        raise ConnectionError("Failed to connect to Binance")
    
    try:
        # Get multi-timeframe data
        multi_tf_data = trader.get_multi_timeframe_data(symbol, timeframes, limit=data_limit)
        
        # Create features with alignment and OHLC integrity
        features = trader.create_trading_features(multi_tf_data)
        
        # Basic validation
        if features.empty:
            raise ValueError("No features generated")
        
        # Check for any NaN values
        nan_count = features.isna().sum().sum()
        if nan_count > 0:
            print(f"⚠️  Warning: {nan_count} NaN values found, will be forward filled")
            features = features.ffill().bfill()
        
        print(f"✅ Features generated: {features.shape[0]} rows x {features.shape[1]} columns")
        
        return features
        
    except Exception as e:
        print(f"❌ Error generating features: {e}")
        raise
    
    finally:
        trader.disconnect()


# Convenience function for quick use
def get_realtime_features(symbol: str = 'ETHUSDT', limit: int = 200) -> pd.DataFrame:
    """
    Quick function to get realtime features with default API keys
    
    Args:
        symbol: Trading symbol (default: 'ETHUSDT')
        limit: Number of candles (default: 200)
    
    Returns:
        pd.DataFrame: Trading features aligned to 1m
    """
    # Default API keys (should be in env variables in production)
    API_KEY = "EduyybaFGjUpSkR7q2J0HwHjHF6dB8TB5klAAUX8Ukum2Yz1jR2J8osZVXz9kxZC"
    API_SECRET = "QmAxhDG4QYxdrif38WyQ6uvGLv5OZvlGPIRBzdtFWry7adtRNzGFY8HlLkOSLOyY"
    
    return generate_trading_features(
        symbol=symbol,
        api_key=API_KEY,
        api_secret=API_SECRET,
        data_limit=limit
    )



if __name__ == "__main__":
    # ⚠️ SECURITY WARNING ⚠️
    print("⚠️  IMPORTANT: Never put API keys directly in code!")
    print("   Use environment variables or secure config files")
    print("   Example: API_KEY = os.environ.get('BINANCE_API_KEY')")
    print("")
    
    # Run test
    #test_binance_connection()

    #test_trading_features_alignment()

    #features = generate_trading_features(
    #    symbol='BTCUSDT',
    #    api_key="your_api_key",
    #    api_secret="your_api_secret",
    #    data_limit=100,
    #    timeframes=['1m', '5m', '30m']
    #)
    
    # Method 2: Quick use with defaults
    features = get_realtime_features('ETHUSDT', limit=200)
    
    # Use features for your model
    print(f"\nFeatures ready for use:")
    print(f"Shape: {features.shape}")
    print(f"Columns: {list(features.columns)}")
    print(f"Latest data:\n{features.tail(3)}")