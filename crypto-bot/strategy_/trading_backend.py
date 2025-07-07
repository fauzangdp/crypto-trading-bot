# trading_backend.py
import asyncio
import websockets
import json
import threading
import time
import socket
from datetime import datetime
from typing import Dict, Optional
import logging
from flask import Flask, render_template_string
from flask_cors import CORS
import os
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import your existing modules
from live_trading import LiveScalpingTrader
from features_binance import get_live_lstm_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingMonitorServer:
    """WebSocket server for real-time trading monitoring"""
    
    def __init__(self, trader_config: Dict, port: int = 8766):
        self.trader_config = trader_config
        self.port = port
        self.clients = set()
        self.trader = None
        self.is_running = False
        
        # Store latest data
        self.latest_signal = None
        self.latest_features = None
        self.latest_position = None
        self.latest_price = None
        
        # Flask app for serving HTML
        self.app = Flask(__name__)
        CORS(self.app)
        self.flask_port = 5000
        
    def check_port_available(self, port):
        """Check if port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return True
        except OSError:
            return False
        
    def find_html_file(self):
        """Find HTML file in possible locations"""
        possible_names = ['trading_monitor.html', 'paste.txt']
        possible_dirs = [
            os.getcwd(),
            os.path.dirname(__file__) if __file__ else '.',
            os.path.dirname(os.path.abspath(__file__)) if __file__ else '.',
            '.',  # Current directory
        ]
        
        logger.info(f"🔍 Searching for HTML file...")
        logger.info(f"   Current working directory: {os.getcwd()}")
        logger.info(f"   Script directory: {os.path.dirname(__file__) if __file__ else 'N/A'}")
        
        for directory in possible_dirs:
            logger.info(f"   Checking directory: {directory}")
            try:
                files_in_dir = os.listdir(directory)
                logger.info(f"      Files: {files_in_dir}")
                
                for filename in possible_names:
                    filepath = os.path.join(directory, filename)
                    logger.info(f"      Trying: {filepath}")
                    if os.path.exists(filepath):
                        logger.info(f"✅ Found HTML file: {filepath}")
                        return filepath
            except Exception as e:
                logger.warning(f"      Error accessing {directory}: {e}")
        
        logger.error("❌ HTML file not found in any location")
        return None

    async def register(self, websocket):
        """Register new client"""
        self.clients.add(websocket)
        logger.info(f"✅ Client connected. Total clients: {len(self.clients)}")
        
        # Send configuration first
        try:
            config_data = {
                'type': 'config',
                'symbol': self.trader.symbol if self.trader else 'ETHUSDT',
                'leverage': self.trader.leverage if self.trader else 20
            }
            await websocket.send(json.dumps(config_data))
            logger.info("📤 Sent config to client")
            
            # Then send initial data if available
            if self.latest_signal:
                # IMPORTANT: Process a fresh cycle to get features_history
                update_data = self.process_trading_cycle()
                
                if update_data:
                    # Send the full update data as initial
                    initial_message = {
                        'type': 'initial',
                        'data': update_data['data']  # This includes features_history
                    }
                    await websocket.send(json.dumps(initial_message))
                    logger.info("📤 Sent initial data to client (with features_history)")
                else:
                    # Fallback to old method
                    initial_data = {
                        'type': 'initial',
                        'data': {
                            'signal': self.latest_signal,
                            'features': self.latest_features,
                            'position': self.latest_position,
                            'price': self.latest_price
                        }
                    }
                    await websocket.send(json.dumps(initial_data))
                    logger.info("📤 Sent initial data to client (without features_history)")
        except Exception as e:
            logger.error(f"❌ Error sending data: {e}")
    
    async def unregister(self, websocket):
        """Unregister client"""
        self.clients.discard(websocket)
        logger.info(f"❌ Client disconnected. Total clients: {len(self.clients)}")
    
    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients"""
        if not self.clients:
            logger.debug("No clients connected to broadcast to")
            return
            
        logger.info(f"📤 Broadcasting to {len(self.clients)} clients")
        message_str = json.dumps(message)
        
        # Send to all clients
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message_str)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"❌ Error broadcasting to client: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        for client in disconnected:
            self.clients.discard(client)
    
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

    def process_trading_cycle(self):
        """Process one trading cycle and get all data"""
        try:
            logger.info("🔄 Processing trading cycle...")
            
            # Get current price
            current_price = self.trader.trader.get_current_price(self.trader.symbol)
            self.latest_price = float(current_price)
            logger.info(f"💰 Current price: ${current_price}")
            
            # Get trading signal from model
            logger.info("🤖 Getting signal from model...")
            signal = self.trader.model.get_signal(self.trader.symbol)
            logger.info(f"🎯 Raw signal from model: {signal}")

            signal['timestamp'] = time.time()
            signal['generated_at'] = datetime.now().isoformat()
            
            # Log signal details
            if signal:
                logger.info(f"📊 Signal Details:")
                logger.info(f"   - Action: {signal.get('action', 'N/A')}")
                logger.info(f"   - Decision: {signal.get('decision', 'N/A')}")
                logger.info(f"   - Confidence: {signal.get('confidence', 'N/A')}")
                logger.info(f"   - Timestamp: {signal.get('timestamp', 'N/A')}")
            else:
                logger.warning("⚠️ No signal received from model!")
            
            # Clean signal data for JSON
            clean_signal = self.clean_for_json(signal)
            self.latest_signal = clean_signal
            logger.info(f"✅ Clean signal: {clean_signal}")
            
            # Get LSTM features - FULL HISTORICAL DATA
            logger.info("🔧 Getting LSTM features (200 historical points)...")
            features_df = get_live_lstm_features(self.trader.symbol, limit=200)
            
            # Get latest features for current signal
            latest_features = features_df.iloc[-1]
            self.latest_features = {
                'rsi_norm_1h': float(latest_features['rsi_norm_1h']),
                'macd_norm_1h': float(latest_features['macd_norm_1h']),
                'rsi_norm_30m': float(latest_features['rsi_norm_30m']),
                'bb_position_30m': float(latest_features['bb_position_30m']),
                'macd_norm_5m': float(latest_features['macd_norm_5m']),
                'adx_norm_5m': float(latest_features['adx_norm_5m']),
                'momentum_convergence': float(latest_features['momentum_convergence'])
            }
            
            # Prepare HISTORICAL features for chart (last 100 points for performance)
            historical_features = features_df.tail(100).copy()
            historical_data = []
            
            # Simple timestamp generation - use current time and work backwards
            current_time = int(time.time())
            
            for i, (idx, row) in enumerate(historical_features.iterrows()):
                # Calculate timestamp: current time minus minutes ago (reverse order)
                minutes_ago = len(historical_features) - i - 1
                timestamp = current_time - (minutes_ago * 60)
                
                historical_data.append({
                    'time': timestamp,
                    'rsi_norm_1h': float(row['rsi_norm_1h']),
                    'macd_norm_1h': float(row['macd_norm_1h']),
                    'rsi_norm_30m': float(row['rsi_norm_30m']),
                    'bb_position_30m': float(row['bb_position_30m']),
                    'macd_norm_5m': float(row['macd_norm_5m']),
                    'adx_norm_5m': float(row['adx_norm_5m']),
                    'momentum_convergence': float(row['momentum_convergence'])
                })
            
            logger.info(f"✅ Historical features prepared: {len(historical_data)} points")
            logger.info(f"📊 Sample historical data: {historical_data[-1] if historical_data else 'None'}")
            logger.info(f"📊 Historical data length: {len(historical_data)}")
            logger.info(f"📊 Historical data is valid: {len(historical_data) > 0}")
                        # ===== FETCH REAL POSITION FROM EXCHANGE =====
            try:
                logger.info("📊 Fetching position from exchange...")
                # Get actual position from Binance
                exchange_position = self.trader.trader.get_position(self.trader.symbol)
                logger.info(f"📊 Exchange position: {exchange_position}")
                
                # Update internal trader state
                if exchange_position.get('size', 0) > 0:
                    self.trader.in_position = True
                    self.trader.position_side = exchange_position.get('side')
                    self.trader.position_size = exchange_position.get('size')
                    self.trader.entry_price = exchange_position.get('entry_price', 0)
                    logger.info(f"✅ Position found: {self.trader.position_side} {self.trader.position_size}")
                else:
                    self.trader.in_position = False
                    self.trader.position_side = None
                    self.trader.position_size = 0
                    self.trader.entry_price = 0
                    self.trader.stop_loss_price = 0
                    logger.info("📊 No position found")
                    
            except Exception as e:
                logger.error(f"❌ Error fetching position: {e}")
            
            # Format position data for frontend
            self.latest_position = {
                'in_position': bool(self.trader.in_position),
                'side': str(self.trader.position_side) if self.trader.position_side else None,
                'entry_price': float(self.trader.entry_price) if self.trader.entry_price else 0.0,
                'stop_loss': float(self.trader.stop_loss_price) if self.trader.stop_loss_price else 0.0,
                'size': float(self.trader.position_size) if self.trader.position_size else 0.0
            }
            
            # Process signal in trader
            logger.info("⚡ Processing signal in trader...")
            try:
                self.trader.process_signal(signal)
                logger.info("✅ Signal processed by trader")
            except Exception as e:
                logger.error(f"❌ Error processing signal: {e}")
                import traceback
                traceback.print_exc()
            
            # Update position info again after processing signal
            try:
                exchange_position = self.trader.trader.get_position(self.trader.symbol)
                if exchange_position.get('size', 0) > 0:
                    self.trader.in_position = True
                    self.trader.position_side = exchange_position.get('side')
                    self.trader.position_size = exchange_position.get('size')
                    self.trader.entry_price = exchange_position.get('entry_price', 0)
                    
                    # Calculate stop loss if not set
                    if not self.trader.stop_loss_price and self.trader.entry_price:
                        if self.trader.position_side == 'LONG':
                            self.trader.stop_loss_price = self.trader.entry_price * (1 - self.trader.stop_loss_pct)
                        else:
                            self.trader.stop_loss_price = self.trader.entry_price * (1 + self.trader.stop_loss_pct)
                        
            except Exception as e:
                logger.error(f"❌ Error updating position after signal: {e}")
            
            # Final position data
            self.latest_position = {
                'in_position': bool(self.trader.in_position),
                'side': str(self.trader.position_side) if self.trader.position_side else None,
                'entry_price': float(self.trader.entry_price) if self.trader.entry_price else 0.0,
                'stop_loss': float(self.trader.stop_loss_price) if self.trader.stop_loss_price else 0.0,
                'size': float(self.trader.position_size) if self.trader.position_size else 0.0
            }
            
            candle_data = {
                'time': int(time.time()),
                'price': self.latest_price
            }


            # Create update data with clean types
            update_data = {
                'type': 'update',
                'timestamp': datetime.now().isoformat(),
                'data': {
                    'price': float(current_price),
                    'signal': clean_signal,
                    'features': self.latest_features,  # Current features
                    'features_history': historical_data,  # Historical features for chart
                    'position': self.latest_position,
                    'candle': candle_data,
                }
            }
            
            logger.info(f"📊 Historical data length: {len(historical_data)}")
            logger.info(f"📊 Features history in data: {len(update_data['data'].get('features_history', []))}")

            # Final JSON cleanup
            update_data = self.clean_for_json(update_data)
            
            logger.info(f"📊 Final position status: {self.latest_position}")
            logger.info(f"📊 Final signal status: {self.latest_signal}")
            logger.info("✅ Trading cycle completed")
            return update_data
            
        except Exception as e:
            logger.error(f"❌ Error in trading cycle: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def trading_loop(self):
        """Main trading loop with WebSocket broadcasting"""
        # Initialize trader
        logger.info("🚀 Initializing trader...")
        self.trader = LiveScalpingTrader(self.trader_config)
        
        # Connect to exchange
        if not self.trader.trader.connect():
            logger.error("❌ Failed to connect to Binance")
            return
        
        # Set leverage
        try:
            self.trader.trader.set_leverage(self.trader.symbol, self.trader.leverage)
            logger.info(f"✅ Leverage set to {self.trader.leverage}x")
        except Exception as e:
            logger.warning(f"⚠️ Could not set leverage: {e}")
        
        logger.info(f"🚀 Trading monitor started for {self.trader.symbol}")
        
        try:
            # Process first cycle immediately
            update_data = self.process_trading_cycle()
            if update_data:
                await self.broadcast(update_data)
            
            while self.is_running:
                # Wait for next cycle
                await asyncio.sleep(60)
                
                # Process trading cycle
                update_data = self.process_trading_cycle()
                if update_data:
                    await self.broadcast(update_data)
                
        except Exception as e:
            logger.error(f"❌ Trading loop error: {e}")
        finally:
            # Cleanup
            if self.trader.in_position:
                logger.info("🔄 Closing position before exit...")
                self.trader.close_position("Shutdown")
            
            self.trader.trader.disconnect()
            logger.info("✅ Trading monitor stopped")
    
    async def handler(self, websocket, path=None):
        """WebSocket connection handler"""
        await self.register(websocket)
        try:
            async for message in websocket:
                if message == 'ping':
                    await websocket.send('pong')
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)
    
    def run_flask(self):
        """Run Flask server for HTML"""
        try:
            @self.app.route('/')
            def index():
                logger.info("📥 HTTP request received for /")
                try:
                    html_file = self.find_html_file()
                    logger.info(f"🔍 HTML file search result: {html_file}")
                    
                    if html_file:
                        try:
                            with open(html_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                                logger.info(f"✅ HTML file loaded successfully: {len(content)} chars")
                                return content
                        except Exception as e:
                            logger.error(f"❌ Error reading HTML file: {e}")
                            return f"<h1>Error reading HTML file: {str(e)}</h1>"
                    else:
                        logger.warning("⚠️ No HTML file found, returning fallback")
                        return """
                        <html>
                        <body style="font-family: Arial; padding: 20px; background: #1e1e1e; color: #d4d4d4;">
                            <h2>Trading Monitor - HTML File Not Found</h2>
                            <p>Please create 'trading_monitor.html' in the same directory as trading_backend.py</p>
                            <p>Current directory: <code>{}</code></p>
                            <p>Backend is running on: <a href="ws://localhost:{}">ws://localhost:{}</a></p>
                            <hr>
                            <h3>Debug Info:</h3>
                            <p>Script file: <code>{}</code></p>
                            <p>Files in directory: {}</p>
                        </body>
                        </html>
                        """.format(
                            os.getcwd(), 
                            self.port, 
                            self.port,
                            __file__,
                            str(os.listdir(os.path.dirname(__file__) if os.path.dirname(__file__) else '.'))
                        )
                except Exception as e:
                    logger.error(f"❌ Route handler error: {e}")
                    import traceback
                    traceback.print_exc()
                    return f"<h1>Route Error: {str(e)}</h1><pre>{traceback.format_exc()}</pre>"
            
            @self.app.route('/health')
            def health():
                logger.info("📥 Health check request")
                return {"status": "ok", "message": "Flask server is running"}
            
            # Test file finding before starting server
            logger.info("🔍 Pre-startup file check...")
            test_file = self.find_html_file()
            logger.info(f"📄 HTML file found: {test_file}")
            
            # Find available port
            port = self.flask_port
            while not self.check_port_available(port) and port < 5010:
                port += 1
            
            if port != self.flask_port:
                logger.info(f"⚠️ Port {self.flask_port} occupied, using port {port}")
                self.flask_port = port
            
            logger.info(f"🌐 Starting Flask server on 0.0.0.0:{self.flask_port}")
            logger.info(f"🌐 Access URLs:")
            logger.info(f"   - http://localhost:{self.flask_port}")
            logger.info(f"   - http://127.0.0.1:{self.flask_port}")
            logger.info(f"   - http://0.0.0.0:{self.flask_port}")
            
            self.app.run(host='0.0.0.0', port=self.flask_port, debug=False, threaded=True, use_reloader=False)
            
        except Exception as e:
            logger.error(f"❌ Flask server error: {e}")
            import traceback
            traceback.print_exc()
    
    async def start(self):
        """Start WebSocket server and trading loop"""
        self.is_running = True
        
        # Start Flask in separate thread
        logger.info("🌐 Starting Flask server...")
        flask_thread = threading.Thread(target=self.run_flask, daemon=True)
        flask_thread.start()
        
        # Wait a moment for Flask to start
        await asyncio.sleep(2)
        
        # Start WebSocket server
        logger.info(f"📡 Starting WebSocket server on port {self.port}")
        async with websockets.serve(self.handler, "localhost", self.port):
            logger.info(f"✅ WebSocket server started on ws://localhost:{self.port}")
            logger.info(f"✅ Web interface available at http://localhost:{self.flask_port}")
            
            # Run trading loop
            await self.trading_loop()
    
    def stop(self):
        """Stop the server"""
        self.is_running = False


def get_config():
    """Trading configuration"""
    return {
        'api_key': 'EduyybaFGjUpSkR7q2J0HwHjHF6dB8TB5klAAUX8Ukum2Yz1jR2J8osZVXz9kxZC',
        'api_secret': 'QmAxhDG4QYxdrif38WyQ6uvGLv5OZvlGPIRBzdtFWry7adtRNzGFY8HlLkOSLOyY',
        'symbol': 'PLUMEUSDT',
        'model_path': 'models/trading_lstm_20250701_233903.pth',
        'leverage': 50,
        'position_pct_normal': 0.04,
        'position_pct_strong': 0.08,
        'stop_loss_pct': 0.02,
        # Take Profit Configuration
        'use_take_profit': True,
        'tp1_percent': 0.003,         # 0.3% TP1
        'tp2_percent': 0.008,         # 0.8% TP2
        'tp1_size_ratio': 0.4,        # 60% position at TP1, 40% at TP2
    }


if __name__ == "__main__":
    try:
        logger.info("🚀 Starting Trading Monitor Server...")
        config = get_config()
        server = TradingMonitorServer(config)
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("\n⏹️ Stopping server...")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")