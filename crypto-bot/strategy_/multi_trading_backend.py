# multi_trading_backend.py
import asyncio
import websockets
import json
import threading
import logging
import time
import socket
from datetime import datetime
from typing import Dict, Optional, List
import logging
from flask import Flask, render_template_string
from flask_cors import CORS
import signal  # ADD THIS
import os
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import your existing modules
from multi_crypto_trader import MultiCryptoTrader, get_multi_trader_config
from crypto_screener import CryptoScreener
from features_binance import get_live_lstm_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiTradingMonitorServer:
    """WebSocket server for real-time multi-crypto trading monitoring"""
    
    def __init__(self, trader_config: Dict, port: int = 8766):
        self.trader_config = trader_config
        self.port = port
        self.clients = set()
        self.multi_trader = None
        self.is_running = False
        
        # Store latest data for all symbols
        self.latest_data = {
            'signals': {},      # symbol -> signal
            'positions': {},    # symbol -> position
            'prices': {},       # symbol -> price
            'features': {}      # symbol -> features
        }
        
        # Screening data
        self.latest_screening = {
            'top_signals': [],
            'timestamp': None,
            'portfolio_status': {}
        }
        
        # ADD THIS LINE:
        self.last_screening_time = 0
        
        # Flask app for serving HTML
        self.app = Flask(__name__)
        CORS(self.app)
        self.flask_port = 5000
        
        # Logger
        self.logger = logging.getLogger(__name__)

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        self.logger.info("\n⏹️ Shutdown signal received...")
        try:
            # Set flag to stop
            self.is_running = False
            
            # Close all WebSocket connections
            for client in list(self.clients):
                try:
                    asyncio.create_task(client.close())
                except:
                    pass
            
            # Stop multi trader if exists
            if self.multi_trader:
                self.multi_trader.stop()
            
            self.logger.info("✅ Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
        finally:
            # Force exit after 2 seconds
            import threading
            threading.Timer(2.0, lambda: os._exit(0)).start()
        
    def verify_connections(self):
        """Verify all connections are healthy"""
        try:
            # Check Flask server
            try:
                import requests
                response = requests.get(f'http://localhost:{self.flask_port}/health', timeout=5)
                if response.status_code == 200:
                    self.logger.info("✅ Flask server healthy")
                else:
                    self.logger.error("❌ Flask server unhealthy")
            except:
                self.logger.error("❌ Flask server not responding")
            
            # Check WebSocket clients
            if self.clients:
                self.logger.info(f"✅ {len(self.clients)} WebSocket clients connected")
            else:
                self.logger.warning("⚠️ No WebSocket clients connected")
                
        except Exception as e:
            self.logger.error(f"❌ Connection verification error: {e}")

    def check_port_available(self, port):
        """Check if port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return True
        except OSError:
            return False
    
    async def register(self, websocket):
        """Register new client"""
        self.clients.add(websocket)
        self.logger.info(f"✅ Client connected. Total clients: {len(self.clients)}")
        
        # Send configuration first
        try:
            config_data = {
                'type': 'config',
                'max_positions': self.trader_config.get('max_positions', 5),
                'screening_enabled': self.trader_config.get('enable_screening', True),
                'screening_interval': self.trader_config.get('screening_interval', 1800)
            }
            await websocket.send(json.dumps(config_data))
            self.logger.info("📤 Sent config to client")
            
            # Send initial data if available
            if self.latest_data['signals'] or self.latest_screening['top_signals']:
                initial_data = {
                    'type': 'initial',
                    'data': {
                        'signals': self.latest_data['signals'],
                        'positions': self.latest_data['positions'],
                        'prices': self.latest_data['prices'],
                        'screening': self.latest_screening,
                        'portfolio_status': self.get_portfolio_status()
                    }
                }
                await websocket.send(json.dumps(initial_data))
                self.logger.info("📤 Sent initial data to client")
                
        except Exception as e:
            self.logger.error(f"❌ Error sending initial data: {e}")
    
    async def unregister(self, websocket):
        """Unregister client"""
        self.clients.discard(websocket)
        self.logger.info(f"❌ Client disconnected. Total clients: {len(self.clients)}")
    
    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients"""
        if not self.clients:
            self.logger.debug("No clients connected to broadcast to")
            return
            
        self.logger.info(f"📤 Broadcasting to {len(self.clients)} clients")
        message_str = json.dumps(message)
        
        # Send to all clients
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message_str)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                self.logger.error(f"❌ Error broadcasting to client: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        for client in disconnected:
            self.clients.discard(client)
    
    async def fast_price_update_loop(self):
        """Ultra fast price updates"""
        while self.is_running:
            try:
                if self.multi_trader and self.multi_trader.traders:
                    updates = {}
                    
                    # Get all prices in one API call
                    tickers = self.multi_trader.screener.trader.client.futures_ticker()
                    ticker_dict = {t['symbol']: float(t['lastPrice']) for t in tickers}
                    
                    # Update for active symbols only
                    for symbol in self.multi_trader.traders.keys():
                        if symbol in ticker_dict:
                            price = ticker_dict[symbol]
                            self.latest_data['prices'][symbol] = price
                            updates[symbol] = price
                    
                    if updates:
                        await self.broadcast({
                            'type': 'price_batch_update',
                            'prices': updates,
                            'timestamp': datetime.now().isoformat()
                        })
                        
            except Exception as e:
                self.logger.error(f"Fast price update error: {e}")
            
            await asyncio.sleep(0.5)

    def clean_for_json(self, obj):
        """Clean object for JSON serialization"""
        import pandas as pd
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self.clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.clean_for_json(item) for item in obj]
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        if not self.multi_trader:
            return {}
        
        try:
            portfolio_status = self.multi_trader.portfolio.get_status()
            
            # Add position details
            position_details = []
            total_pnl = 0
            
            for symbol, trader in self.multi_trader.traders.items():
                if trader.in_position:
                    try:
                        current_price = trader.trader.get_current_price(symbol)
                        
                        # Calculate P&L
                        if trader.position_side == 'LONG':
                            pnl_pct = ((current_price - trader.entry_price) / trader.entry_price) * 100
                        else:
                            pnl_pct = ((trader.entry_price - current_price) / trader.entry_price) * 100
                        
                        total_pnl += pnl_pct
                        
                        position_details.append({
                            'symbol': symbol,
                            'side': trader.position_side,
                            'entry_price': float(trader.entry_price),
                            'current_price': float(current_price),
                            'size': float(trader.position_size),
                            'pnl_pct': float(pnl_pct),
                            'pyramid_level': trader.pyramid_count
                        })
                    except Exception as e:
                        self.logger.warning(f"Error getting status for {symbol}: {e}")
            
            portfolio_status.update({
                'positions': position_details,
                'total_pnl': total_pnl,
                'daily_trades': self.multi_trader.daily_trade_count,
                'max_daily_trades': self.multi_trader.max_daily_trades
            })
            
            return portfolio_status
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio status: {e}")
            return {}
    
    def cleanup_orphan_orders(self):
        """Clean up orphan orders for all symbols"""
        try:
            if not self.multi_trader or not self.multi_trader.traders:
                return
                
            self.logger.info("🧹 Running orphan order cleanup...")
            cleaned_count = 0
            
            for symbol, trader in self.multi_trader.traders.items():
                try:
                    # Get actual position
                    actual_position = trader.trader.get_position(symbol)
                    open_orders = trader.trader.get_open_orders(symbol)
                    
                    # If no position but has orders, clean them
                    if actual_position['size'] == 0 and open_orders:
                        self.logger.warning(f"⚠️ {symbol}: Found {len(open_orders)} orphan orders")
                        
                        for order in open_orders:
                            try:
                                trader.trader.cancel_order(symbol, order['orderId'])
                                self.logger.info(f"   ✅ Cancelled orphan {order['type']} order {order['orderId']}")
                                cleaned_count += 1
                                time.sleep(0.1)  # Avoid rate limiting
                            except Exception as e:
                                self.logger.error(f"   ❌ Failed to cancel {order['orderId']}: {e}")
                                
                except Exception as e:
                    self.logger.error(f"❌ Cleanup error for {symbol}: {e}")
                    
            if cleaned_count > 0:
                self.logger.info(f"✅ Cleaned {cleaned_count} orphan orders")
                
        except Exception as e:
            self.logger.error(f"❌ Orphan cleanup error: {e}")

    def process_screening_cycle(self):
        """Process screening and update data"""
        try:
            if not self.multi_trader or not self.multi_trader.screening_enabled:
                return None
            
            self.logger.info("🔍 Processing screening cycle...")
            
            # Get top signals from screener
            top_signals = self.multi_trader.screener.screen_crypto()

            if top_signals:
                self.logger.info(f"🎯 Executing trades for {len(top_signals)} signals...")
                self.multi_trader.update_portfolio_from_screening(top_signals)
                
            # Update screening data
            self.latest_screening = {
                'top_signals': self.clean_for_json(top_signals),
                'timestamp': datetime.now().isoformat(),
                'portfolio_status': self.get_portfolio_status()
            }
            
            # Prepare screening update message
            screening_update = {
                'type': 'screening_update',
                'timestamp': datetime.now().isoformat(),
                'data': self.latest_screening
            }
            
            self.logger.info(f"✅ Screening completed: {len(top_signals)} opportunities found")
            return screening_update
            
        except Exception as e:
            self.logger.error(f"❌ Error in screening cycle: {e}")
            return None
    
    def process_trading_cycle(self):
        """Process trading cycle for all active symbols - ENHANCED WITH BETTER UPDATES"""
        try:
            if not self.multi_trader or not self.multi_trader.traders:
                return None
            
            self.logger.info("⚡ Processing trading cycle...")
            
            # Get signals for all active symbols
            active_symbols = list(self.multi_trader.traders.keys())
            signals = self.multi_trader.screener.get_quick_signals(active_symbols)
            
            # Update data for each symbol
            updated_symbols = []
            
            for symbol in active_symbols:  # Process ALL active symbols, not just ones with signals
                trader = self.multi_trader.traders.get(symbol)
                
                if not trader:
                    continue
                
                try:
                    # Get current price
                    current_price = trader.trader.get_current_price(symbol)
                    
                    # Get actual position from exchange
                    actual_position = trader.trader.get_position(symbol)
                    
                    # Update signal data if available
                    signal_data = next((s for s in signals if s['symbol'] == symbol), None)
                    if signal_data:
                        clean_signal = self.clean_for_json(signal_data)
                        self.latest_data['signals'][symbol] = clean_signal
                    else:
                        # Create default signal if not available
                        self.latest_data['signals'][symbol] = {
                            'symbol': symbol,
                            'decision': 0.0,
                            'action': 'HOLD',
                            'confidence': 0.0,
                            'timestamp': datetime.now().isoformat()
                        }
                    
                    # Update price
                    self.latest_data['prices'][symbol] = float(current_price)
                    
                    # Update position data with ACTUAL exchange data
                    if actual_position['size'] > 0:
                        # Calculate actual P&L
                        if actual_position['side'] == 'LONG':
                            pnl_pct = ((current_price - actual_position['entry_price']) / actual_position['entry_price']) * 100
                        else:
                            pnl_pct = ((actual_position['entry_price'] - current_price) / actual_position['entry_price']) * 100
                        
                        self.latest_data['positions'][symbol] = {
                            'in_position': True,
                            'side': actual_position['side'],
                            'entry_price': float(actual_position['entry_price']),
                            'stop_loss': float(trader.stop_loss_price) if trader.stop_loss_price else 0.0,
                            'size': float(actual_position['size']),
                            'pyramid_level': trader.pyramid_count,
                            'current_price': float(current_price),
                            'pnl_pct': float(pnl_pct),
                            'margin_used': float(actual_position.get('margin_used', 0))
                        }
                    else:
                        # No position
                        self.latest_data['positions'][symbol] = {
                            'in_position': False,
                            'side': None,
                            'entry_price': 0.0,
                            'stop_loss': 0.0,
                            'size': 0.0,
                            'pyramid_level': 0,
                            'current_price': float(current_price),
                            'pnl_pct': 0.0,
                            'margin_used': 0.0
                        }
                    
                    # Get features for this symbol
                    try:
                        features_df = get_live_lstm_features(symbol, limit=1, client=trader.trader.client)
                        if not features_df.empty:
                            latest_features = features_df.iloc[-1]
                            self.latest_data['features'][symbol] = {
                                'rsi_norm_1h': float(latest_features['rsi_norm_1h']),
                                'macd_norm_1h': float(latest_features['macd_norm_1h']),
                                'rsi_norm_30m': float(latest_features['rsi_norm_30m']),
                                'bb_position_30m': float(latest_features['bb_position_30m']),
                                'macd_norm_5m': float(latest_features['macd_norm_5m']),
                                'adx_norm_5m': float(latest_features['adx_norm_5m']),
                                'momentum_convergence': float(latest_features['momentum_convergence'])
                            }
                    except Exception as fe:
                        self.logger.warning(f"Failed to get features for {symbol}: {fe}")
                    
                    updated_symbols.append(symbol)
                    
                    # Process signal in trader if available
                    if signal_data:
                        trader.process_signal(signal_data)
                        
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")
            
            # Prepare trading update message with ALL data
            trading_update = {
                'type': 'trading_update',
                'timestamp': datetime.now().isoformat(),
                'data': {
                    'updated_symbols': updated_symbols,
                    'signals': self.latest_data['signals'],  # Send ALL signals
                    'positions': self.latest_data['positions'],  # Send ALL positions
                    'prices': self.latest_data['prices'],  # Send ALL prices
                    'features': self.latest_data['features'],  # Send ALL features
                    'portfolio_status': self.get_portfolio_status()
                }
            }
            
            self.logger.info(f"✅ Trading cycle completed: {len(updated_symbols)} symbols updated")
            return trading_update
            
        except Exception as e:
            self.logger.error(f"❌ Error in trading cycle: {e}")
            return None
    
    async def trading_loop(self):
        """Main trading loop with WebSocket broadcasting"""
        # Initialize multi-trader
        self.logger.info("🚀 Initializing multi-crypto trader...")
        self.multi_trader = MultiCryptoTrader(self.trader_config)
        
        # Connect screener
        if not self.multi_trader.screener.trader.connect():
            self.logger.error("❌ Failed to connect to Binance")
            return
        
        self.logger.info(f"🚀 Multi-crypto trading monitor started")
        
        # Counters for different cycles
        screening_counter = 0
        trading_counter = 0
        cleanup_counter = 0  # NEW: Add cleanup counter
        
        # Intervals (in seconds)
        trading_interval = self.trader_config.get('trading_interval', 1)
        screening_interval = self.trader_config.get('screening_interval', 1800)
        cleanup_interval = 120  # NEW: Cleanup every 2 minutes
        
        try:
            # Initial screening
            if self.multi_trader.screening_enabled:
                self.logger.info("🚀 Starting initial screening...")
                screening_update = self.process_screening_cycle()
                if screening_update:
                    await self.broadcast(screening_update)
                    self.logger.info("📤 Broadcasted screening update") 
                else:
                    self.logger.warning("⚠️ No screening update to broadcast")

                self.logger.info("🚀 Starting initial trading cycle...")
                trading_update = self.process_trading_cycle()
                if trading_update:
                    self.logger.info("📤 Broadcasted trading update")
            
            while self.is_running:
                cycle_start = time.time()
                
                self.logger.info(f"🔄 === LOOP ITERATION ===")
                self.logger.info(f"   Active traders: {len(self.multi_trader.traders) if self.multi_trader else 0}")
                self.logger.info(f"   Last screening: {(time.time() - self.last_screening_time if hasattr(self, 'last_screening_time') else 0)//60:.0f}min ago")
                
                # Check if it's time for screening
                screening_counter += trading_interval
                if (self.multi_trader.screening_enabled and 
                    screening_counter >= screening_interval):
                    
                    screening_update = self.process_screening_cycle()
                    if screening_update:
                        await self.broadcast(screening_update)
                    screening_counter = 0
                    self.last_screening_time = time.time()
                
                # NEW: Check if it's time for cleanup
                cleanup_counter += trading_interval
                if cleanup_counter >= cleanup_interval:
                    self.cleanup_orphan_orders()
                    cleanup_counter = 0
                
                # Run trading cycle
                trading_update = self.process_trading_cycle()
                if trading_update:
                    await self.broadcast(trading_update)
                
                # Calculate sleep time
                cycle_duration = time.time() - cycle_start
                sleep_time = max(0, trading_interval - cycle_duration)
                
                self.logger.info(f"✅ Loop iteration completed in {cycle_duration:.1f}s")
                self.logger.info(f"😴 Sleeping for {sleep_time:.1f}s until next cycle...")
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
        except Exception as e:
            self.logger.error(f"❌ Trading loop error: {e}")
        finally:
            # Cleanup
            if self.multi_trader:
                self.multi_trader.cleanup()
            self.logger.info("✅ Trading monitor stopped")
    
    async def handler(self, websocket, path=None):
        """WebSocket connection handler - FIXED VERSION"""
        client_id = id(websocket)
        self.logger.info(f"🔌 New connection attempt from client {client_id}")
        
        try:
            # Register client first
            await self.register(websocket)
            
            # Keep connection alive
            while True:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(websocket.recv(), timeout=60.0)
                    
                    if message == 'ping':
                        await websocket.send('pong')
                        self.logger.debug(f"🏓 Ping-pong from client {client_id}")
                    elif message:
                        try:
                            data = json.loads(message)
                            if data.get('type') == 'request_initial_data':
                                # Send initial data
                                await self.send_initial_data(websocket)
                        except json.JSONDecodeError:
                            self.logger.warning(f"Invalid JSON from client {client_id}: {message}")
                            
                except asyncio.TimeoutError:
                    # Send ping to check if client is alive
                    try:
                        await websocket.ping()
                        self.logger.debug(f"🏓 Sent ping to client {client_id}")
                    except:
                        break
                        
                except websockets.exceptions.ConnectionClosed:
                    self.logger.info(f"❌ Client {client_id} disconnected normally")
                    break
                    
                except Exception as e:
                    self.logger.error(f"❌ Handler error for client {client_id}: {e}")
                    break
                    
        finally:
            await self.unregister(websocket)
            self.logger.info(f"🔌 Client {client_id} cleanup completed")


    # ADD NEW METHOD after handler
    async def send_initial_data(self, websocket):
        """Send initial data to newly connected client"""
        try:
            initial_data = {
                'type': 'initial',
                'data': {
                    'signals': self.latest_data['signals'],
                    'positions': self.latest_data['positions'],
                    'prices': self.latest_data['prices'],
                    'screening': self.latest_screening,
                    'portfolio_status': self.get_portfolio_status()
                }
            }
            await websocket.send(json.dumps(initial_data))
            self.logger.info("📤 Sent initial data to client")
        except Exception as e:
            self.logger.error(f"❌ Error sending initial data: {e}")
    
    def find_html_file(self):
        """Find HTML file in possible locations"""
        possible_names = ['multi_trading_monitor.html', 'trading_monitor.html']
        possible_dirs = [
            os.getcwd(),
            os.path.dirname(__file__) if __file__ else '.',
            os.path.dirname(os.path.abspath(__file__)) if __file__ else '.',
            '.',  # Current directory
        ]
        
        self.logger.info(f"🔍 Searching for HTML file...")
        self.logger.info(f"   Current working directory: {os.getcwd()}")
        self.logger.info(f"   Script directory: {os.path.dirname(__file__) if __file__ else 'N/A'}")
        
        for directory in possible_dirs:
            self.logger.info(f"   Checking directory: {directory}")
            try:
                files_in_dir = os.listdir(directory)
                self.logger.info(f"      Files: {files_in_dir}")
                
                for filename in possible_names:
                    filepath = os.path.join(directory, filename)
                    self.logger.info(f"      Trying: {filepath}")
                    if os.path.exists(filepath):
                        self.logger.info(f"✅ Found HTML file: {filepath}")
                        return filepath
            except Exception as e:
                self.logger.warning(f"      Error accessing {directory}: {e}")
        
        self.logger.error("❌ HTML file not found in any location")
        return None

    def run_flask(self):
        """Run Flask server for HTML - OPTIMIZED STARTUP"""
        try:
            @self.app.route('/')
            def index():
                self.logger.info("📥 HTTP request received for /")
                try:
                    html_file = self.find_html_file()
                    self.logger.info(f"🔍 HTML file search result: {html_file}")
                    
                    if html_file:
                        try:
                            with open(html_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                                self.logger.info(f"✅ HTML file loaded successfully: {len(content)} chars")
                                return content
                        except Exception as e:
                            self.logger.error(f"❌ Error reading HTML file: {e}")
                            return f"<h1>Error reading HTML file: {str(e)}</h1>"
                    else:
                        self.logger.warning("⚠️ No HTML file found, returning fallback")
                        # Return a simple loading page that redirects
                        return """
                        <html>
                        <head>
                            <meta http-equiv="refresh" content="2">
                            <style>
                                body { font-family: Arial; padding: 20px; background: #1e1e1e; color: #d4d4d4; }
                                .loading { text-align: center; margin-top: 50px; }
                                .spinner { border: 4px solid #333; border-top: 4px solid #4CAF50; 
                                        border-radius: 50%; width: 40px; height: 40px; 
                                        animation: spin 1s linear infinite; margin: 20px auto; }
                                @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                            </style>
                        </head>
                        <body>
                            <div class="loading">
                                <h2>Multi-Crypto Trading Monitor</h2>
                                <div class="spinner"></div>
                                <p>Loading interface...</p>
                                <p><small>WebSocket: ws://localhost:{}</small></p>
                            </div>
                        </body>
                        </html>
                        """.format(self.port)
                except Exception as e:
                    self.logger.error(f"❌ Route handler error: {e}")
                    import traceback
                    traceback.print_exc()
                    return f"<h1>Route Error: {str(e)}</h1><pre>{traceback.format_exc()}</pre>"
            
            @self.app.route('/health')
            def health():
                return {"status": "ok", "message": "Multi-crypto trading monitor is running"}
            
            # Find available port
            port = self.flask_port
            while not self.check_port_available(port) and port < 5010:
                port += 1
            
            if port != self.flask_port:
                self.logger.info(f"⚠️ Port {self.flask_port} occupied, using port {port}")
                self.flask_port = port
            
            self.logger.info(f"🌐 Starting Flask server on 0.0.0.0:{self.flask_port}")
            
            # Use threaded=False to avoid conflicts
            self.app.run(host='0.0.0.0', port=self.flask_port, debug=False, threaded=False, use_reloader=False)
            
        except Exception as e:
            self.logger.error(f"❌ Flask server error: {e}")
    
    async def start(self):
        """Start WebSocket server and trading loop - FIXED VERSION"""
        self.is_running = True
        
        # Start Flask in separate thread
        self.logger.info("🌐 Starting Flask server...")
        flask_thread = threading.Thread(target=self.run_flask, daemon=True)
        flask_thread.start()
        
        # Wait for Flask to start
        await asyncio.sleep(2)
        
        # Check if port is available
        port = self.port
        while not self.check_port_available(port) and port < 8776:
            port += 1
        
        if port != self.port:
            self.logger.info(f"⚠️ Port {self.port} occupied, using port {port}")
            self.port = port
        
        # Start WebSocket server with better configuration
        self.logger.info(f"📡 Starting WebSocket server on port {self.port}")
        
        try:
            # Simple server setup
            async with websockets.serve(self.handler, "0.0.0.0", self.port) as server:
                self.logger.info(f"✅ WebSocket server started on ws://localhost:{self.port}")
                self.logger.info(f"✅ Web interface available at http://localhost:{self.flask_port}")
                
                # Run trading loop
                await asyncio.gather(
                    self.trading_loop(),
                    self.fast_price_update_loop()
                )
                
        except Exception as e:
            self.logger.error(f"❌ Failed to start WebSocket server: {e}")
            import traceback
            traceback.print_exc()
    
    def stop(self):
        """Stop the server"""
        self.is_running = False
        if self.multi_trader:
            self.multi_trader.stop()


if __name__ == "__main__":
    try:
        logger.info("🚀 Starting Multi-Crypto Trading Monitor Server...")
        config = get_multi_trader_config()
        server = MultiTradingMonitorServer(config)
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("\n⏹️ Stopping server...")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")