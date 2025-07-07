# multi_crypto_trader.py - FIXED VERSION WITH CONFLICT RESOLUTION
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import os
import sys
from datetime import datetime


# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from live_trading import LiveScalpingTrader
from crypto_screener import CryptoScreener, PortfolioManager
from model import LiveTradingModel
from library.binance_connector import BinanceTrader
from library.sr_fibo import DynamicLevelSelector


class MultiCryptoTrader:
    """Multi-cryptocurrency trading system - ENHANCED WITH CONFLICT RESOLUTION"""
    
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.WARNING)
        
        # Handler untuk console output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Kemudian lanjutkan dengan inisialisasi lainnya
        self.config = config or {}
        self.pairs = []
        self.traders = {}
        self.is_running = False
        logging.basicConfig(level=logging.WARNING) 

        logging.getLogger('multi_trading_backend').setLevel(logging.WARNING)
        logging.getLogger('multi_crypto_trader').setLevel(logging.WARNING)
        logging.getLogger('model').setLevel(logging.ERROR)


        self.config = config
        


        


        # Initialize screener
        self.screener = CryptoScreener(config)
        
        # Initialize portfolio manager
        self.portfolio = PortfolioManager(config)
        
        # Individual traders for each symbol
        self.traders = {}  # symbol -> LiveScalpingTrader
        
        # SOLUTION 1: Execution Lock - Only one trade at a time
        self.execution_lock = threading.Lock()
        self.execution_queue = []  # Queue for pending executions
        
        # SOLUTION 2: Balance Tracking
        self.allocated_balance = {}  # symbol -> allocated amount
        self.total_allocated = 0
        
        # SOLUTION 3: Symbol State Tracking
        self.symbol_states = {}  # symbol -> {'executing': bool, 'last_trade_time': timestamp}
        
        # Control flags
        self.is_running = False
        self.screening_enabled = config.get('enable_screening', True)
        
        # Timing
        self.screening_interval = config.get('screening_interval', 1800)  # 30 minutes
        self.trading_interval = config.get('trading_interval', 60)  # 1 minute
        self.last_screening = time.time()

        next_screening_time = self.last_screening + self.screening_interval
        next_screening_minutes = self.screening_interval / 60
        next_screening_str = datetime.fromtimestamp(next_screening_time).strftime('%H:%M:%S')
        
        print("\n" + "="*70)
        print(f"⏰ SCREENING SCHEDULE INFO")
        print("="*70)
        print(f"✅ Screening completed at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"⏱️  Next screening in: {next_screening_minutes:.0f} minutes")
        print(f"🕐 Next screening at: {next_screening_str}")
        print(f"💤 System will sleep and monitor active positions")
        print("="*70 + "\n")
        
        self.logger.info(f"✅ Screening completed - Next in {next_screening_minutes:.0f} minutes")
        
        # SOLUTION 4: Execution Delays
        self.min_execution_delay = 5  # Minimum 5 seconds between trades
        self.last_execution_time = 0
        
        # Risk management
        self.max_daily_trades = config.get('max_daily_trades', 50)
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()
        
        # Blacklist for problematic symbols
        self.blacklist = set(config.get('blacklist', []))
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=10)

        self.max_executions_per_screening = config.get('max_executions_per_screening', 2)
        self.current_cycle_executions = 0
        self.last_screening_cycle_time = 0
        self.screening_cycle_history = [] 
        
        # Initialization logging
        self.logger.info("🔧 MultiCryptoTrader initialized - ENHANCED VERSION")
        self.logger.info(f"   Screening enabled: {self.screening_enabled}")
        self.logger.info(f"   Screening interval: {self.screening_interval}s")
        self.logger.info(f"   Trading interval: {self.trading_interval}s")
        self.logger.info(f"   Min execution delay: {self.min_execution_delay}s")
        
    def reset_daily_counters(self):
        """Reset daily counters"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_trade_count = 0
            self.last_reset_date = current_date
            self.logger.info("🔄 Daily counters reset")
    
    def can_trade_more(self) -> bool:
        """Check if we can execute more trades today"""
        self.reset_daily_counters()
        return self.daily_trade_count < self.max_daily_trades
    
    def get_available_balance_for_symbol(self, symbol: str) -> float:
        """SOLUTION: Central balance management"""
        try:
            # Get total balance
            total_balance = self.screener.trader.get_balance()
            
            # Calculate total allocated
            total_allocated = sum(self.allocated_balance.values())
            
            # Available balance
            available = total_balance - total_allocated
            
            # Max allocation per symbol
            max_per_symbol = total_balance * self.portfolio.get_allocation(symbol)
            
            # Return minimum of available and max allocation
            return min(available, max_per_symbol)
            
        except Exception as e:
            self.logger.error(f"Error calculating available balance: {e}")
            return 0
    
    def allocate_balance(self, symbol: str, amount: float) -> bool:
        """Reserve balance for a symbol"""
        with self.execution_lock:
            available = self.get_available_balance_for_symbol(symbol)
            
            if available >= amount:
                self.allocated_balance[symbol] = amount
                self.logger.info(f"💰 Allocated ${amount:.2f} for {symbol}")
                return True
            else:
                self.logger.warning(f"❌ Insufficient balance for {symbol}: need ${amount:.2f}, available ${available:.2f}")
                return False
    
    def release_balance(self, symbol: str):
        """Release allocated balance"""
        with self.execution_lock:
            if symbol in self.allocated_balance:
                amount = self.allocated_balance.pop(symbol)
                self.logger.info(f"💰 Released ${amount:.2f} from {symbol}")
    
    def create_trader_for_symbol(self, symbol: str) -> LiveScalpingTrader:
        """Create a new trader instance for a symbol"""
        try:
            self.logger.info(f"🔧 Creating trader for {symbol}...")
            
            # Modify config for this symbol
            trader_config = self.config.copy()
            trader_config['symbol'] = symbol
            
            # Copy technical analysis settings
            technical_keys = [
                'use_technical_levels', 'lookback_candles', 
                'sr_merge_threshold', 'max_risk_technical',
                'min_rr_ratio', 'level_update_interval', 
                'fallback_to_fixed'
            ]
            for key in technical_keys:
                if key in self.config:
                    trader_config[key] = self.config[key]
            
            # Adjust position size based on portfolio allocation
            allocation = self.portfolio.get_allocation(symbol)
            trader_config['position_pct_normal'] = allocation * 0.5
            trader_config['position_pct_strong'] = allocation
            
            self.logger.info(f"   Config: allocation={allocation}, normal={trader_config['position_pct_normal']}")
            
            # Create trader
            trader = LiveScalpingTrader(trader_config)
            
            # Connect
            self.logger.info(f"   Connecting trader for {symbol}...")
            if not trader.trader.connect():
                self.logger.error(f"❌ Failed to connect trader for {symbol}")
                return None
            
            # Set leverage
            try:
                trader.trader.set_leverage(symbol, trader.leverage)
                self.logger.info(f"✅ {symbol}: Leverage set to {trader.leverage}x")
            except Exception as e:
                self.logger.warning(f"⚠️ {symbol}: Could not set leverage: {e}")
            
            # Sync state
            self.logger.info(f"   Syncing state for {symbol}...")
            trader.sync_state_with_exchange()
            
            # Initialize symbol state
            self.symbol_states[symbol] = {
                'executing': False,
                'last_trade_time': 0
            }
            
            self.logger.info(f"✅ Created trader for {symbol}")
            return trader
            
        except Exception as e:
            self.logger.error(f"❌ Error creating trader for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def add_symbol_to_trading(self, symbol: str):
        """Add a symbol to active trading"""
        try:
            self.logger.info(f"📊 Adding {symbol} to trading...")
            
            if symbol in self.blacklist:
                self.logger.warning(f"⚠️ {symbol} is blacklisted, skipping")
                return False
            
            if symbol in self.traders:
                self.logger.info(f"📊 {symbol} already being traded")
                return True
            
            if not self.portfolio.can_open_position(symbol):
                self.logger.warning(f"⚠️ Cannot add {symbol}: Portfolio limit reached")
                return False
            
            # Create trader
            self.logger.info(f"   Creating trader for {symbol}...")
            trader = self.create_trader_for_symbol(symbol)
            if trader is None:
                self.logger.error(f"❌ Failed to create trader for {symbol}")
                return False
            
            # Add to portfolio and traders
            self.traders[symbol] = trader
            self.portfolio.add_symbol(symbol)
            
            self.logger.info(f"🎯 Added {symbol} to trading portfolio (Total: {len(self.traders)})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error adding {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def remove_symbol_from_trading(self, symbol: str, reason: str = "Removed"):
        """Remove a symbol from active trading"""
        try:
            if symbol not in self.traders:
                return
            
            trader = self.traders[symbol]
            
            # Close any open position
            if trader.in_position:
                self.logger.info(f"🔄 Closing position for {symbol}: {reason}")
                trader.close_position(reason)
            
            # Release allocated balance
            self.release_balance(symbol)
            
            # Disconnect trader
            trader.trader.disconnect()
            
            # Remove from tracking
            del self.traders[symbol]
            self.portfolio.remove_symbol(symbol)
            
            # Clean up symbol state
            if symbol in self.symbol_states:
                del self.symbol_states[symbol]
            
            self.logger.info(f"❌ Removed {symbol} from trading: {reason}")
            
        except Exception as e:
            self.logger.error(f"❌ Error removing {symbol}: {e}")
    
    def wait_for_execution_slot(self) -> bool:
        """SOLUTION: Ensure proper spacing between executions"""
        current_time = time.time()
        time_since_last = current_time - self.last_execution_time
        
        if time_since_last < self.min_execution_delay:
            wait_time = self.min_execution_delay - time_since_last
            self.logger.info(f"⏳ Waiting {wait_time:.1f}s before next execution...")
            time.sleep(wait_time)
        
        self.last_execution_time = time.time()
        return True
    
    def process_signal_for_symbol(self, symbol: str, signal: Dict):
        """Process a trading signal for a specific symbol - WITH LOCK"""
        try:
            if symbol not in self.traders:
                self.logger.debug(f"⚠️ {symbol} not in active traders, skipping")
                return False
            
            if not self.can_trade_more():
                self.logger.warning(f"⚠️ Daily trade limit reached ({self.max_daily_trades})")
                return False
            
            # SOLUTION: Use execution lock
            with self.execution_lock:
                # Mark symbol as executing
                if symbol in self.symbol_states:
                    self.symbol_states[symbol]['executing'] = True
                
                # Wait for execution slot
                self.wait_for_execution_slot()
                
                self.logger.info(f"🔒 Processing {symbol} with lock (exclusive execution)")
                
                trader = self.traders[symbol]
                
                # Log the signal being processed
                direction = "LONG" if signal.get('decision', 0) > 0 else "SHORT"
                self.logger.info(f"⚡ Processing {symbol}: {direction} signal {signal.get('decision', 0):.3f}")
                
                # Process signal
                old_position_status = trader.in_position
                old_position_side = trader.position_side
                
                # Execute with proper error handling
                try:
                    trader.process_signal(signal)
                    
                    # Wait for settlement
                    time.sleep(3)  # Give time for orders to settle
                    
                except Exception as e:
                    self.logger.error(f"❌ Error executing trade for {symbol}: {e}")
                    return False
                
                # Check if any action was taken
                action_taken = False
                if trader.in_position != old_position_status:
                    action_taken = True
                    self.daily_trade_count += 1
                    self.logger.info(f"📊 Trade executed for {symbol}! Daily trades: {self.daily_trade_count}/{self.max_daily_trades}")
                elif trader.in_position and trader.position_side != old_position_side:
                    action_taken = True
                    self.daily_trade_count += 1
                    self.logger.info(f"📊 Position flipped for {symbol}! Daily trades: {self.daily_trade_count}/{self.max_daily_trades}")
                
                # Update symbol state
                if symbol in self.symbol_states:
                    self.symbol_states[symbol]['executing'] = False
                    self.symbol_states[symbol]['last_trade_time'] = time.time()
                
                self.logger.info(f"🔓 Released lock for {symbol}")
                
                return action_taken
            
        except Exception as e:
            self.logger.error(f"❌ Error processing signal for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            
            # Ensure state is cleaned up
            if symbol in self.symbol_states:
                self.symbol_states[symbol]['executing'] = False
            
            return False
    
    def update_portfolio_from_screening(self, top_signals: List[Dict]):
        """Update portfolio based on screening results - SEQUENTIAL"""
        try:
            self.logger.info(f"🔄 PORTFOLIO UPDATE: Processing {len(top_signals)} signals")
            
            # Get current symbols
            current_symbols = set(self.traders.keys())
            self.logger.info(f"   Current symbols: {current_symbols}")
            
            # Get new top symbols
            new_top_symbols = {signal['symbol'] for signal in top_signals}
            self.logger.info(f"   New top symbols: {new_top_symbols}")
            
            # Remove symbols no longer in top list (only if not in position)
            symbols_to_remove = current_symbols - new_top_symbols
            for symbol in symbols_to_remove:
                trader = self.traders.get(symbol)
                if trader and not trader.in_position:
                    self.logger.info(f"   Removing {symbol} (not in top list)")
                    self.remove_symbol_from_trading(symbol, "No longer in top signals")
            
            # Add new symbols - ONE BY ONE
            added_count = 0
            for signal in top_signals:
                symbol = signal['symbol']
                if symbol not in current_symbols:
                    self.logger.info(f"   Adding new symbol: {symbol}")
                    
                    # Check if we can add more
                    if not self.portfolio.can_open_position(symbol):
                        self.logger.info(f"   Portfolio full, skipping {symbol}")
                        continue
                    
                    success = self.add_symbol_to_trading(symbol)
                    
                    if success:
                        added_count += 1
                        self.logger.info(f"✅ Successfully added {symbol}")
                        
                        # SOLUTION: Process signal with lock
                        self.logger.info(f"🎯 Processing immediate signal for {symbol}")
                        self.process_signal_for_symbol(symbol, signal)
                        
                    else:
                        self.logger.error(f"❌ Failed to add {symbol}")
            
            self.logger.info(f"📊 Portfolio update completed:")
            self.logger.info(f"   Added: {added_count} symbols")
            self.logger.info(f"   Total active: {len(self.traders)} symbols")
            self.logger.info(f"   Active symbols: {list(self.traders.keys())}")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating portfolio: {e}")
            import traceback
            traceback.print_exc()
    
    def run_screening_cycle(self):
        """Run one screening cycle"""
        try:
            if not self.screening_enabled:
                self.logger.info("⚠️ Screening disabled, skipping")
                return
            
            self.logger.info("🔍 Running screening cycle...")
            
            # Run screening
            top_signals = self.screener.screen_crypto()
            
            if top_signals:
                self.logger.info(f"🏆 Found {len(top_signals)} opportunities:")
                for i, signal in enumerate(top_signals[:5]):
                    direction = "LONG" if signal['decision'] > 0 else "SHORT"
                    self.logger.info(f"  {i+1}. {signal['symbol']}: {signal['decision']:+.3f} ({direction})")
                
                # Update portfolio based on results
                self.logger.info("🔄 Updating portfolio from screening results...")
                self.update_portfolio_from_screening(top_signals)
            else:
                self.logger.warning("⚠️ No opportunities found in screening")
            
            self.last_screening = time.time()
            self.logger.info("✅ Screening cycle completed")
            
        except Exception as e:
            self.logger.error(f"❌ Screening cycle error: {e}")
            import traceback
            traceback.print_exc()
    
    def run_trading_cycle(self):
        """Run one trading cycle for all active symbols - SEQUENTIAL EXECUTION"""
        try:
            self.logger.info(f"⚡ Trading cycle start - {len(self.traders)} active traders")
            
            if not self.traders:
                self.logger.info("📊 No active traders - waiting for screening to find opportunities")
                next_screening_minutes = (self.screening_interval - (time.time() - self.last_screening)) // 60
                self.logger.info(f"💡 Next screening in {next_screening_minutes:.0f} minutes")
                self.show_portfolio_status()
                return
            
            # SOLUTION: Process symbols SEQUENTIALLY, not parallel
            active_symbols = list(self.traders.keys())
            self.logger.info(f"📊 Processing signals sequentially for: {active_symbols}")
            
            processed = 0
            actions_taken = 0
            
            # Process ONE BY ONE
            for symbol in active_symbols:
                try:
                    # Skip if symbol is already executing
                    if symbol in self.symbol_states and self.symbol_states[symbol].get('executing', False):
                        self.logger.warning(f"⚠️ {symbol} is already executing, skipping")
                        continue
                    
                    # Get signal for this symbol
                    self.logger.debug(f"📡 Getting signal for {symbol}...")
                    signal = self.screener.get_signal_for_symbol(symbol)
                    
                    if signal:
                        # Process with lock
                        action_taken = self.process_signal_for_symbol(symbol, signal)
                        if action_taken:
                            actions_taken += 1
                    
                    processed += 1
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to process {symbol}: {e}")
                    continue
            
            # Summary of cycle
            self.logger.info(f"📊 Trading cycle completed:")
            self.logger.info(f"   Processed: {processed} symbols")
            self.logger.info(f"   Actions taken: {actions_taken}")
            self.logger.info(f"   Daily trades: {self.daily_trade_count}/{self.max_daily_trades}")
            
            # Show portfolio status
            self.show_portfolio_status()
            
        except Exception as e:
            self.logger.error(f"❌ Trading cycle error: {e}")
            import traceback
            traceback.print_exc()
            self.logger.info("🔄 Continuing despite trading cycle error...")
    
    def show_portfolio_status(self):
        """Show current portfolio status"""
        try:
            total_positions = 0
            position_details = []
            
            for symbol, trader in self.traders.items():
                if trader.in_position:
                    total_positions += 1
                    current_price = trader.trader.get_current_price(symbol)
                    
                    # Calculate P&L
                    if trader.position_side == 'LONG':
                        pnl_pct = ((current_price - trader.entry_price) / trader.entry_price) * 100
                    else:
                        pnl_pct = ((trader.entry_price - current_price) / trader.entry_price) * 100
                    
                    position_details.append({
                        'symbol': symbol,
                        'side': trader.position_side,
                        'entry': trader.entry_price,
                        'current': current_price,
                        'pnl_pct': pnl_pct,
                        'size': trader.position_size
                    })
            
            # Show balance allocation
            total_balance = self.screener.trader.get_balance()
            total_allocated = sum(self.allocated_balance.values())
            
            self.logger.info(f"💰 Balance Status:")
            self.logger.info(f"   Total: ${total_balance:.2f}")
            self.logger.info(f"   Allocated: ${total_allocated:.2f}")
            self.logger.info(f"   Available: ${total_balance - total_allocated:.2f}")
            
            if total_positions > 0:
                self.logger.info(f"💼 Active Positions ({total_positions}):")
                for pos in position_details:
                    pnl_color = "🟢" if pos['pnl_pct'] > 0 else "🔴"
                    self.logger.info(f"  {pos['symbol']:12s} | {pos['side']} | "
                                   f"${pos['current']:.5f} | {pnl_color}{pos['pnl_pct']:+.2f}%")
            else:
                self.logger.info("💼 No active positions")
            
        except Exception as e:
            self.logger.error(f"❌ Portfolio status error: {e}")
    
    def verify_position_mode(self):
        """Ensure account is in One-Way mode"""
        try:
            # Check current mode through screener client
            if self.screener.trader.client:
                position_mode = self.screener.trader.client.futures_get_position_mode()
                
                if position_mode['dualSidePosition']:
                    self.logger.warning("⚠️ Account in Hedge Mode, switching to One-Way Mode...")
                    try:
                        self.screener.trader.client.futures_change_position_mode(dualSidePosition=False)
                        self.logger.info("✅ Switched to One-Way Mode successfully")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to switch position mode: {e}")
                        self.logger.warning("⚠️ Please manually set account to One-Way Mode in Binance")
                else:
                    self.logger.info("✅ Account already in One-Way Mode")
                    
        except Exception as e:
            self.logger.error(f"❌ Error checking position mode: {e}")
    
    def health_check(self):
        """Comprehensive health check for all traders - ENHANCED VERSION"""
        try:
            self.logger.info("🏥 Running health check...")
            
            issues_found = 0
            
            # Check each trader
            for symbol, trader in list(self.traders.items()):  # Use list() to avoid dict modification during iteration
                try:
                    # Ensure trader is connected
                    if not trader.trader.test_connection():
                        self.logger.error(f"❌ {symbol}: Connection lost!")
                        # Try to reconnect
                        if not trader.trader.connect():
                            self.logger.error(f"❌ {symbol}: Failed to reconnect, removing from trading")
                            self.remove_symbol_from_trading(symbol, "Connection Lost")
                            issues_found += 1
                            continue
                    
                    # Get actual position with retry
                    actual_position = None
                    for attempt in range(3):
                        try:
                            actual_position = trader.trader.get_position(symbol)
                            break
                        except Exception as e:
                            if attempt < 2:
                                time.sleep(1)
                                continue
                            else:
                                raise
                    
                    if actual_position is None:
                        self.logger.error(f"❌ {symbol}: Could not get position data")
                        issues_found += 1
                        continue
                    
                    # Check position sync
                    if actual_position['size'] > 0 and not trader.in_position:
                        self.logger.warning(f"⚠️ {symbol}: Exchange has position but trader doesn't know!")
                        self.logger.info(f"   Syncing state for {symbol}...")
                        trader.sync_state_with_exchange()
                        issues_found += 1
                    
                    elif actual_position['size'] == 0 and trader.in_position:
                        self.logger.warning(f"⚠️ {symbol}: Trader thinks has position but exchange doesn't!")
                        trader.cleanup_and_reset_state("Health Check - No Position")
                        issues_found += 1
                    
                    # Check for position without stop loss
                    if actual_position['size'] > 0:
                        open_orders = trader.trader.get_open_orders(symbol)
                        has_sl = any(o['type'] in ['STOP_MARKET', 'STOP_LOSS'] for o in open_orders)
                        
                        if not has_sl:
                            self.logger.error(f"❌ {symbol}: Position without stop loss!")
                            self.logger.info(f"🛡️ Placing emergency stop loss...")
                            
                            # Place emergency stop loss
                            try:
                                current_price = trader.trader.get_current_price(symbol)
                                if trader.position_side == 'LONG':
                                    emergency_sl = current_price * 0.98  # 2% stop loss
                                else:
                                    emergency_sl = current_price * 1.02
                                
                                formatted_sl = trader.format_price_for_api(emergency_sl)
                                sl_order = trader.place_stop_loss_order_with_verification(
                                    symbol=symbol,
                                    side=trader.position_side,
                                    quantity=int(actual_position['size']),
                                    stop_price=formatted_sl
                                )
                                trader.stop_loss_order_id = sl_order['order_id']
                                self.logger.info(f"✅ Emergency SL placed for {symbol}")
                            except Exception as e:
                                self.logger.error(f"❌ Failed to place emergency SL for {symbol}: {e}")
                                trader.close_position("No Stop Loss Protection")
                            
                            issues_found += 1
                    
                except Exception as e:
                    self.logger.error(f"❌ Health check failed for {symbol}: {e}")
                    issues_found += 1
            
            # Check orphan orders across all tracked symbols
            try:
                all_open_orders = self.screener.trader.client.futures_get_open_orders()
                tracked_symbols = set(self.traders.keys())
                
                for order in all_open_orders:
                    if order['symbol'] not in tracked_symbols:
                        self.logger.warning(f"⚠️ Orphan order found: {order['symbol']} {order['type']}")
                        # Cancel orphan order
                        try:
                            self.screener.trader.client.futures_cancel_order(
                                symbol=order['symbol'],
                                orderId=order['orderId']
                            )
                            self.logger.info(f"✅ Cancelled orphan order for {order['symbol']}")
                        except:
                            pass
                        issues_found += 1
                        
            except Exception as e:
                self.logger.error(f"❌ Failed to check orphan orders: {e}")
            
            self.logger.info(f"🏥 Health check completed: {issues_found} issues found")
            
        except Exception as e:
            self.logger.error(f"❌ Health check error: {e}")
            import traceback
            traceback.print_exc()
    
    async def run(self):
        """Main trading loop - ENHANCED WITH PROPER SEQUENCING"""
        self.logger.info("🚀 === STARTING MULTI-CRYPTO TRADING SYSTEM (ENHANCED) ===")
        self.logger.info(f"   Screening: {'Enabled' if self.screening_enabled else 'Disabled'}")
        self.logger.info(f"   Max Positions: {self.portfolio.max_positions}")
        self.logger.info(f"   Screening Interval: {self.screening_interval}s ({self.screening_interval//60}min)")
        self.logger.info(f"   Trading Interval: {self.trading_interval}s")
        self.logger.info(f"   Execution Protection: ENABLED (sequential processing)")
        
        # Connect screener
        self.logger.info("🔌 Connecting screener...")
        if not self.screener.trader.connect():
            self.logger.error("❌ Failed to connect screener")
            return
        self.logger.info("✅ Screener connected")
        
        # Verify position mode
        self.verify_position_mode()
        
        # Initial screening
        if self.screening_enabled:
            self.logger.info("🔍 Running INITIAL screening...")
            self.run_screening_cycle()
            self.logger.info(f"✅ Initial screening completed - Active traders: {len(self.traders)}")
        else:
            self.logger.info("⚠️ Screening disabled - no initial screening")
        
        self.logger.info("🔄 === STARTING MAIN TRADING LOOP ===")
        self.is_running = True
        
        loop_count = 0
        health_check_counter = 0
        
        try:
            while self.is_running:
                loop_count += 1
                cycle_start = time.time()

                time_since_screening = time.time() - self.last_screening
                is_screening_time = (self.screening_enabled and 
                                time_since_screening >= self.screening_interval)


                
                if is_screening_time:
                    # Screening Loop
                    print("\n" + "🔍"*30)
                    print(f"🔍 SCREENING LOOP #{loop_count} - {datetime.now().strftime('%H:%M:%S')}")
                    print("🔍"*30)
                    print(f"🔄 Running FULL CYCLE: Screening + Trading")
                    print(f"🎯 Looking for new opportunities...")
                else:
                    # Trading Only Loop
                    if loop_count % 5 == 0:  # Print header every 5 loops
                        print("\n" + "📈"*30)
                        print(f"📈 TRADING LOOP #{loop_count} - {datetime.now().strftime('%H:%M:%S')}")
                        print("📈"*30)
                    
                    # Show monitoring status
                    time_until_next = self.screening_interval - time_since_screening
                    minutes_until = time_until_next / 60
                    
                    if loop_count % 5 == 0:  # Show status every 5 loops
                        print(f"💤 MONITORING MODE")
                        print(f"📊 Active Positions: {len(self.traders)}")
                        print(f"⏱️  Next Screening: {minutes_until:.1f} minutes")
                        print(f"🔄 Checking existing positions only...")
                
                # ===== BAGIAN LAMA (tetap sama) =====
                self.logger.info(f"🔄 === LOOP ITERATION #{loop_count} ===")
                self.logger.info(f"   Active traders: {len(self.traders)}")
                self.logger.info(f"   Last screening: {(time.time() - self.last_screening)//60:.0f}min ago")
                
                # Health check every 5 cycles
                health_check_counter += 1
                if health_check_counter >= 5:
                    self.health_check()
                    health_check_counter = 0
                
                # Check if it's time for screening
                if (self.screening_enabled and 
                    time.time() - self.last_screening >= self.screening_interval):
                    
                    # ===== TAMBAHAN KECIL: Notifikasi =====
                    print(f"\n🔔 SCREENING TIME! Starting new screening cycle...")
                    
                    self.logger.info("🔍 Time for periodic screening...")
                    self.run_screening_cycle()
                
                # Run trading cycle
                self.logger.info("⚡ Running trading cycle...")
                self.run_trading_cycle()
                
                # Calculate sleep time (TETAP SAMA)
                cycle_duration = time.time() - cycle_start
                sleep_time = max(0, self.trading_interval - cycle_duration)
                
                self.logger.info(f"✅ Loop iteration #{loop_count} completed in {cycle_duration:.1f}s")
                
                if sleep_time > 0:
                    self.logger.info(f"💤 Sleeping for {sleep_time:.1f}s...")
                    await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.logger.info("\n⏹️ Stopping multi-crypto trader...")
        except Exception as e:
            self.logger.error(f"❌ Main loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup all resources - ENHANCED"""
        self.logger.info("🧹 === CLEANUP START ===")
        
        # Stop all traders
        trader_count = len(self.traders)
        for symbol in list(self.traders.keys()):
            try:
                self.remove_symbol_from_trading(symbol, "System Shutdown")
            except Exception as e:
                self.logger.error(f"Error removing {symbol}: {e}")
        
        self.logger.info(f"✅ Stopped {trader_count} traders")
        
        # Disconnect screener
        try:
            if hasattr(self.screener, 'trader') and self.screener.trader:
                self.screener.trader.disconnect()
                self.logger.info("✅ Screener disconnected")
        except Exception as e:
            self.logger.error(f"Error disconnecting screener: {e}")
        
        # Shutdown executor
        try:
            self.executor.shutdown(wait=True)
            self.logger.info("✅ Executor shutdown")
        except Exception as e:
            self.logger.error(f"Error shutting down executor: {e}")
        
        self.logger.info("🧹 === CLEANUP COMPLETE ===")
    
    def stop(self):
        """Stop the trading system"""
        self.is_running = False


def get_multi_trader_config():
    """Configuration for multi-crypto trader"""
    return {
        # API Configuration
        'api_key': 'EduyybaFGjUpSkR7q2J0HwHjHF6dB8TB5klAAUX8Ukum2Yz1jR2J8osZVXz9kxZC',
        'api_secret': 'QmAxhDG4QYxdrif38WyQ6uvGLv5OZvlGPIRBzdtFWry7adtRNzGFY8HlLkOSLOyY',
        'model_path': 'models/trading_lstm_20250701_233903.pth',
        
        # Trading Configuration
        'leverage': 20,
        'stop_loss_pct': 0.02,
        'use_take_profit': True,
        'tp1_percent': 0.003,
        'tp2_percent': 0.008,
        'tp1_size_ratio': 0.4,
        
        # Portfolio Management
        'max_positions': 5,  # Maximum simultaneous positions
        'position_pct_per_symbol': 0.15,  # 15% per symbol
        'total_balance_pct': 0.8,  # Use 80% of total balance
        
        # Screening Configuration
        'enable_screening': True,
        'screening_interval': 1800,  # 30 minutes
        'min_volume_24h': 30000000,  # $30M minimum
        'min_signal_strength': 0.6,
        'max_symbols': 8,
        
        # Risk Management
        'max_daily_trades': 30,
        'trading_interval': 60,  # 1 minute
        
        # Filters
        'exclude_stable': True,
        'min_price': 0.01,
        'max_price': 1000,
        'blacklist': ['BTCUSDT'],
        
        # Technical Analysis Configuration
        'use_technical_levels': True,
        'lookback_candles': 200,
        'sr_merge_threshold': 0.001,
        'max_risk_technical': 0.02,
        'min_rr_ratio': 1.5,
        'level_update_interval': 300,
        'fallback_to_fixed': True
    }


if __name__ == "__main__":
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get configuration
    config = get_multi_trader_config()
    
    # Create and run trader
    trader = MultiCryptoTrader(config)
    
    try:
        asyncio.run(trader.run())
    except KeyboardInterrupt:
        print("\n⏹️ Stopping...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()