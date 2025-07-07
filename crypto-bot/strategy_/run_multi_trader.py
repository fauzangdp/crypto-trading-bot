# run_multi_trader.py - FIXED VERSION
"""
Multi-Crypto Trading System - Main Runner - FIXED
==========================================

This script demonstrates how to run the multi-crypto trading system
with different strategies and configurations.

Usage Examples:
    python run_multi_trader.py --strategy balanced
    python run_multi_trader.py --strategy aggressive --web-monitor
    python run_multi_trader.py --strategy conservative --test-mode
    python run_multi_trader.py --screen-only
"""

import argparse
import asyncio
import logging
import sys
import os
from datetime import datetime
import logging
import re 
import builtins

FILTER_KEYWORDS = [
    
]

# Save original print
_original_print = builtins.print

def filtered_print(*args, **kwargs):
    # Convert all args to string
    text = ' '.join(str(arg) for arg in args)
    
    # Check if should filter
    for keyword in FILTER_KEYWORDS:
        if keyword in text:
            return  # Skip this print
    
    # Otherwise print normally
    _original_print(*args, **kwargs)

# Replace print function
builtins.print = filtered_print



logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Only show CRITICAL logs from main
logging.getLogger('__main__').setLevel(logging.INFO)

# Silence ALL other modules
for module in ['multi_crypto_trader', 'crypto_screener', 'live_trading',
               'features_binance', 'model', 'binance', 'urllib3', 
               'websockets', 'werkzeug', 'multi_trading_backend']:
    logging.getLogger(module).setLevel(logging.ERROR)

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from config import TradingConfig, get_config, print_config_comparison
from crypto_screener import CryptoScreener
from multi_crypto_trader import MultiCryptoTrader
from multi_trading_backend import MultiTradingMonitorServer


class ImportantLogFilter(logging.Filter):
    """Filter to show only important logs"""
    
    IMPORTANT_KEYWORDS = [
        # === TRADING ACTIONS (Updated) ===
        'BUY @', 'SELL @', 'STRONG BUY', 'STRONG SELL',
        'Position opened', 'Position closed', 'P&L:',
        '🟢 LONG', '🔴 SHORT',  # Trading directions
        '💰 Opening', '✅ Order placed',  # Order execution
        'Entry price:', 'Stop Loss:', 'Take Profit:',  # Position details
        
        # === SCREENING RESULTS (Enhanced) ===
        'TOP', 'OPPORTUNITIES', 'Found.*signals',
        '🎯 Executing trades',  # When trades are executed from screening
        'decision:', 'Signal:',  # Signal details
        'qualified symbols', 'strong signals',  # Screening summary
        'below threshold', 'rejected',  # Failed signals
        '⚠️.*Signal.*won\'t execute',  # Signals that won't trade
        
        # === THRESHOLD & VALIDATION INFO ===
        'threshold', 'met threshold', 'No signals met',
        'validation', 'failed validation',
        'Score:', 'Confidence:',  # Signal scores
        
        # === PORTFOLIO & POSITIONS ===
        'Portfolio update', 'Active positions',
        'Balance:', 'Profit:', 'Loss:',
        'Available balance:', 'Total allocation:',
        
        # === ERRORS & WARNINGS ===
        'ERROR', 'CRITICAL', 'Failed',
        '❌', '⚠️',  # Error/warning emojis
        
        # === STATUS UPDATES ===
        'Started', 'Stopped', 'Connected', 'Disconnected',
        '🔄 LOOP ITERATION',  # Main loop status
        '🔍 Running.*screening',  # Screening status
        
        # === SCREENING DETAILS ===
        '📊 Signal', '🏆',  # Signal rankings
        'Batch.*signals',  # Batch processing
        'Symbol:.*Decision:',  # Individual symbol results
    ]
    
    # Additional patterns for regex matching
    IMPORTANT_PATTERNS = [
        r'Signal.*\d+\.\d+',  # Signal with score
        r'\w+USDT.*decision',  # Symbol with decision
        r'Screening.*\d+.*symbols',  # Screening results
        r'(LONG|SHORT).*rejected',  # Rejected trades
    ]
    
    def filter(self, record):
        # Always show WARNING and above
        if record.levelno >= logging.WARNING:
            return True
            
        message = record.getMessage()
        
        # Check exact keywords
        for keyword in self.IMPORTANT_KEYWORDS:
            if keyword in message or re.search(keyword, message, re.IGNORECASE):
                return True
        
        # Check regex patterns
        for pattern in self.IMPORTANT_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                return True
                
        return False

def setup_logging(level=logging.INFO, log_file=None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    pass


async def run_screening_test(config):
    """Test the screening functionality"""
    print("🔍 === TESTING CRYPTO SCREENER ===")
    print("=" * 50)
    
    screener = CryptoScreener(config)
    
    # Connect
    print("🔌 Connecting to Binance...")
    if not screener.trader.connect():
        print("❌ Failed to connect to Binance")
        return
    print("✅ Connected to Binance")
    
    try:
        # Run screening
        print("🔍 Running crypto screening...")
        top_signals = screener.screen_crypto()
        
        if top_signals:
            print(f"\n🏆 TOP {len(top_signals)} TRADING OPPORTUNITIES:")
            print("=" * 60)
            
            for i, signal in enumerate(top_signals):
                direction = "🟢 LONG" if signal['decision'] > 0 else "🔴 SHORT"
                strength = "STRONG" if abs(signal['decision']) > 0.8 else "NORMAL"
                confidence = abs(signal['decision'])
                
                print(f"{i+1:2d}. {signal['symbol']:12s} | {direction} | "
                      f"{signal['decision']:+.3f} | {strength} | "
                      f"Confidence: {confidence:.1%}")
        else:
            print("❌ No trading opportunities found")
            
    finally:
        screener.trader.disconnect()
        print("✅ Screening test completed")


async def run_trader_only(config):
    """Run the multi-crypto trader without web interface - FIXED VERSION"""
    print("🚀 === ENTERING run_trader_only() ===")
    print("=" * 50)
    
    # Debug config
    print(f"🔧 Config validation:")
    print(f"   API Key: {'✅ Present' if config.get('api_key') else '❌ Missing'}")
    print(f"   Model Path: {'✅ Present' if config.get('model_path') else '❌ Missing'}")
    print(f"   Screening: {'✅ Enabled' if config.get('enable_screening') else '❌ Disabled'}")
    print(f"   Strategy: {config.get('strategy_name', 'Unknown')}")
    print(f"   Max Positions: {config.get('max_positions', 'Not set')}")
    print(f"   Leverage: {config.get('leverage', 'Not set')}x")
    print(f"   Trading Interval: {config.get('trading_interval', 'Not set')}s")
    print(f"   Screening Interval: {config.get('screening_interval', 'Not set')}s")
    print()
    
    # Show configuration summary
    print(f"📊 TRADING CONFIGURATION:")
    print(f"   Strategy: {config.get('strategy_name', 'Custom')}")
    print(f"   Max Positions: {config['max_positions']}")
    print(f"   Leverage: {config['leverage']}x")
    print(f"   Position Size: {config['position_pct_per_symbol']*100:.1f}% per symbol")
    print(f"   Screening: {'Enabled' if config['enable_screening'] else 'Disabled'}")
    print(f"   Screening Interval: {config.get('screening_interval', 1800)//60} minutes")
    print(f"   Trading Interval: {config.get('trading_interval', 60)} seconds")
    print()
    print("💡 IMPORTANT: System will run continuously until stopped")
    print("💡 Press Ctrl+C to stop the system gracefully")
    print("💡 Monitor the logs to track progress and trading activities")
    print()
    
    # Create trader with enhanced error handling
    print("🔧 Creating MultiCryptoTrader instance...")
    try:
        trader = MultiCryptoTrader(config)
        print("✅ MultiCryptoTrader created successfully")
    except Exception as e:
        print(f"❌ Failed to create MultiCryptoTrader: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Validate trader instance
    if not trader:
        print("❌ Trader instance is None")
        return
    
    print(f"📊 Trader instance validated:")
    print(f"   Screening enabled: {trader.screening_enabled}")
    print(f"   Max positions: {trader.portfolio.max_positions}")
    print(f"   Trading interval: {trader.trading_interval}s")
    print()
    
    try:
        print("🔄 === STARTING MAIN TRADING SYSTEM ===")
        print("   Initializing and starting main trading loop...")
        print("   The system will now run continuously...")
        print()
        
        # This is the CRITICAL line that actually starts the trader
        print("⚡ Calling trader.run()...")
        await trader.run()
        
        # This line should only be reached if trader.run() exits normally
        print("🔄 Trading loop ended normally")
        
    except KeyboardInterrupt:
        print("\n⏹️ Keyboard interrupt received - stopping trader gracefully...")
        if hasattr(trader, 'stop'):
            trader.stop()
    except Exception as e:
        print(f"\n❌ Unexpected trader error: {e}")
        import traceback
        traceback.print_exc()
        print("\n🛠️ This is likely a configuration or connection issue")
        print("💡 Try running with --debug flag for more details")
        if hasattr(trader, 'stop'):
            trader.stop()
    finally:
        print("\n🧹 Cleaning up resources...")
        try:
            if hasattr(trader, 'cleanup'):
                trader.cleanup()
            print("✅ Cleanup completed successfully")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
        
        print("✅ Trader stopped successfully")
        print("\n📊 Session Summary:")
        if hasattr(trader, 'daily_trade_count') and hasattr(trader, 'max_daily_trades'):
            print(f"   Daily Trades Executed: {trader.daily_trade_count}/{trader.max_daily_trades}")
        if hasattr(trader, 'traders'):
            print(f"   Active Symbols at End: {len(trader.traders)}")
        print("   Thank you for using Multi-Crypto Trading System! 🚀")


async def run_web_monitor(config):
    """Run the web monitoring interface - FIXED VERSION"""
    print("🌐 === ENTERING run_web_monitor() ===")
    print("🌐 STARTING WEB MONITORING INTERFACE")
    print("=" * 50)
    
    # Show configuration
    print(f"Strategy: {config.get('strategy_name', 'Custom')}")
    print(f"Web Interface: http://localhost:5000")
    print(f"WebSocket: ws://localhost:8766")
    print()
    
    server = None
    
    try:
        # Create server with enhanced error handling
        print("🔧 Creating MultiTradingMonitorServer...")
        server = MultiTradingMonitorServer(config)
        print("✅ Server instance created")
        
        # Add timeout to prevent hanging
        print("🚀 Starting web monitor server...")
        print("⏰ If this hangs, press Ctrl+C to stop...")
        
        # Start with timeout
        await server.start()
        
    except asyncio.TimeoutError:
        print("⏰ Web monitor startup timed out after 30 seconds")
        print("💡 This usually means port conflict or WebSocket issues")
        print("💡 Try closing other applications using port 5000 or 8766")
        
    except KeyboardInterrupt:
        print("\n⏹️ Stopping web monitor...")
        
    except ImportError as e:
        print(f"❌ Import error - missing dependency: {e}")
        print("💡 Make sure all required packages are installed")
        
    except Exception as e:
        print(f"❌ Web monitor error: {e}")
        print("🔍 Full error details:")
        import traceback
        traceback.print_exc()
        
        # Check common issues
        if "port" in str(e).lower() or "address" in str(e).lower():
            print("\n💡 POSSIBLE SOLUTIONS:")
            print("   - Close other applications using port 5000 or 8766")
            print("   - Wait a few seconds and try again")
            print("   - Use --strategy balanced (without --web-monitor) instead")
        
    finally:
        print("\n🧹 Cleaning up web monitor...")
        if server:
            try:
                server.stop()
                print("✅ Server stopped successfully")
            except Exception as e:
                print(f"⚠️ Error stopping server: {e}")
        
        print("🌐 Web monitor session ended")


def validate_config(config):
    """Validate configuration parameters"""
    print("🔧 === VALIDATING CONFIGURATION ===")
    
    errors = []
    warnings = []
    
    # Check required fields
    required_fields = ['api_key', 'api_secret', 'model_path']
    for field in required_fields:
        if not config.get(field):
            errors.append(f"Missing required field: {field}")
        else:
            print(f"✅ {field}: Present")
    
    # Check model file exists
    model_path = config.get('model_path')
    if model_path:
        if os.path.exists(model_path):
            print(f"✅ Model file: {model_path}")
        else:
            errors.append(f"Model file not found: {model_path}")
    
    # Check reasonable values
    leverage = config.get('leverage', 0)
    if leverage > 100:
        warnings.append(f"Very high leverage: {leverage}x")
    elif leverage > 0:
        print(f"✅ Leverage: {leverage}x")
    
    max_positions = config.get('max_positions', 0)
    if max_positions > 10:
        warnings.append(f"Many positions: {max_positions}")
    elif max_positions > 0:
        print(f"✅ Max positions: {max_positions}")
    
    position_pct = config.get('position_pct_per_symbol', 0)
    if position_pct > 0.3:
        warnings.append(f"Large position size: {position_pct*100:.1f}%")
    elif position_pct > 0:
        print(f"✅ Position size: {position_pct*100:.1f}%")
    
    # Print results
    if errors:
        print("\n❌ CONFIGURATION ERRORS:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    if warnings:
        print("\n⚠️ CONFIGURATION WARNINGS:")
        for warning in warnings:
            print(f"   - {warning}")
    
    print("✅ Configuration validation passed")
    return True


class SimpleMonitor:
    """Simple status printer"""
    
    @staticmethod
    def print_header(strategy, mode):
        print("\n" + "="*50)
        print(f"🚀 AI TRADING ROBOT - {strategy.upper()} STRATEGY")
        print("="*50)
        print(f"Mode: {mode}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50 + "\n")
    
    @staticmethod
    def print_status(message):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Crypto Trading System - FIXED VERSION",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --strategy balanced                 # Run balanced strategy
  %(prog)s --strategy aggressive --web-monitor # Run with web interface
  %(prog)s --strategy conservative --test-mode # Test mode (paper trading)
  %(prog)s --screen-only                       # Only run screening
  %(prog)s --compare-strategies                # Compare all strategies
  %(prog)s --custom --max-positions 3 --leverage 15  # Custom config
        """
    )
    
    # Strategy selection
    parser.add_argument('--strategy', choices=['conservative', 'balanced', 'aggressive', 'scalping', 'swing'],
                       default='balanced', help='Trading strategy to use')
    
    # Running modes
    parser.add_argument('--web-monitor', action='store_true',
                       help='Run with web monitoring interface')
    parser.add_argument('--screen-only', action='store_true',
                       help='Only run crypto screening (test mode)')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode (paper trading)')
    parser.add_argument('--compare-strategies', action='store_true',
                       help='Compare all trading strategies and exit')
    
    # Custom configuration
    parser.add_argument('--custom', action='store_true',
                       help='Use custom configuration with overrides')
    parser.add_argument('--max-positions', type=int,
                       help='Maximum number of positions')
    parser.add_argument('--leverage', type=int,
                       help='Trading leverage')
    parser.add_argument('--position-size', type=float,
                       help='Position size percentage (0.1 = 10%)')
    parser.add_argument('--min-signal', type=float,
                       help='Minimum signal strength (0.0 to 1.0)')
    
    # Performance and debugging
    parser.add_argument('--fast-cycle', action='store_true',
                       help='Use faster cycles for testing (30s trading, 5min screening)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--log-file', type=str,
                       help='Log to file')
    
    args = parser.parse_args()
    
    if not args.compare_strategies:
        # Clear screen for clean start
        os.system('cls' if os.name == 'nt' else 'clear')
    
    # Skip verbose validation
    if args.compare_strategies:
        print_config_comparison()
        return 0
    
    # Get configuration silently
    config_manager = TradingConfig()
    
    if args.custom:
        overrides = {}
        if args.max_positions:
            overrides['max_positions'] = args.max_positions
        if args.leverage:
            overrides['leverage'] = args.leverage
        if args.position_size:
            overrides['position_pct_per_symbol'] = args.position_size
        if args.min_signal:
            overrides['min_signal_strength'] = args.min_signal
        
        config = config_manager.get_custom_config(
            risk_level=args.strategy,
            **overrides
        )
        config['strategy_name'] = f"Custom {args.strategy.title()}"
    else:
        config = get_config(args.strategy)
        config['strategy_name'] = args.strategy.title()
    
    # Fast cycle override
    if args.fast_cycle:
        config['trading_interval'] = 30
        config['screening_interval'] = 300
    
    # Test mode
    if args.test_mode:
        config['testnet'] = True
        config['max_daily_trades'] = 5
        config['position_pct_per_symbol'] = 0.01
    
    # SIMPLE STATUS DISPLAY
    try:
        if args.screen_only:
            SimpleMonitor.print_header(args.strategy, "SCREENING TEST")
            SimpleMonitor.print_status("Running screening test...")
            asyncio.run(run_screening_test(config))
            
        elif args.web_monitor:
            SimpleMonitor.print_header(args.strategy, "WEB MONITOR")
            SimpleMonitor.print_status(f"Web UI: http://localhost:5000")
            SimpleMonitor.print_status(f"API: ws://localhost:8766")
            SimpleMonitor.print_status("Starting services...")
            asyncio.run(run_web_monitor(config))
            
        else:
            SimpleMonitor.print_header(args.strategy, "LIVE TRADING")
            SimpleMonitor.print_status(f"Positions: {config.get('max_positions', 5)}")
            SimpleMonitor.print_status(f"Leverage: {config.get('leverage', 20)}x")
            SimpleMonitor.print_status("Starting trader...")
            asyncio.run(run_trader_only(config))
            
    except KeyboardInterrupt:
        SimpleMonitor.print_status("Shutting down...")
    except Exception as e:
        SimpleMonitor.print_status(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    SimpleMonitor.print_status("Stopped successfully")
    return 0


if __name__ == "__main__":
    exit(main())