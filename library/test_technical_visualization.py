# test_technical_visualization.py
"""
Test and Visualize Technical Levels with Resistance Check Logic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from library.binance_connector import BinanceTrader
from library.sr_fibo import SupportResistanceAnalyzer, FibonacciCalculator, DynamicLevelSelector

# Configuration
API_KEY = "EduyybaFGjUpSkR7q2J0HwHjHF6dB8TB5klAAUX8Ukum2Yz1jR2J8osZVXz9kxZC"
API_SECRET = "QmAxhDG4QYxdrif38WyQ6uvGLv5OZvlGPIRBzdtFWry7adtRNzGFY8HlLkOSLOyY"

# Resistance check configuration (same as live_trading.py)
RESISTANCE_PROXIMITY_THRESHOLD = 0.003  # 0.3%
SUPPORT_PROXIMITY_THRESHOLD = 0.003     # 0.3%
BREAKOUT_CONFIRMATION_PCT = 0.002       # 0.2%


def fetch_data(symbol='ETHUSDT', interval='5m', limit=200):
    """Fetch real data from Binance"""
    print(f"📊 Fetching {symbol} data...")
    
    trader = BinanceTrader(API_KEY, API_SECRET, testnet=False)
    if not trader.connect():
        print("❌ Failed to connect to Binance")
        return None
        
    try:
        # Get klines
        klines = trader.client.futures_klines(
            symbol=symbol,
            interval=interval,
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
            df[col] = pd.to_numeric(df[col])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        trader.disconnect()
        print(f"✅ Fetched {len(df)} candles")
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        trader.disconnect()
        return None


def plot_candlestick(ax, df, last_n=100):
    """Plot candlestick chart"""
    df_plot = df.tail(last_n)
    
    for idx, (timestamp, row) in enumerate(df_plot.iterrows()):
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        
        # Candle color
        color = 'green' if c >= o else 'red'
        alpha = 0.8
        
        # Draw wick
        ax.plot([idx, idx], [l, h], color=color, linewidth=1, alpha=alpha)
        
        # Draw body
        body_height = abs(c - o)
        body_bottom = min(o, c)
        
        rect = Rectangle((idx - 0.3, body_bottom), 0.6, body_height,
                        facecolor=color, edgecolor=color, alpha=alpha)
        ax.add_patch(rect)
    
    ax.set_xlim(-1, len(df_plot))
    return df_plot


def visualize_resistance_check_logic(df, symbol='ETHUSDT'):
    """Visualize the resistance check logic for buy signals"""
    print("\n🎯 Testing Resistance Check Logic...")
    
    # Create analyzer
    sr_analyzer = SupportResistanceAnalyzer(lookback_period=100, merge_threshold=0.001)
    
    # Get all levels
    levels = sr_analyzer.get_all_levels(df)
    current_price = levels['current_price']
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Main chart
    ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=2, rowspan=2)
    
    # Plot candlesticks
    df_plot = plot_candlestick(ax1, df, last_n=100)
    
    # Plot current price
    ax1.axhline(y=current_price, color='blue', linestyle='-', linewidth=2, 
                label=f'Current: ${current_price:.5f}')
    
    # Get nearest resistance
    resistances = levels['resistance']
    if resistances:
        nearest_resistance = resistances[0]
        resistance_price = nearest_resistance['price']
        
        # Plot resistance level
        ax1.axhline(y=resistance_price, color='red', linestyle='-', linewidth=3,
                   label=f'Nearest Resistance: ${resistance_price:.5f}')
        
        # Plot proximity zone (where buy signals would be rejected)
        proximity_zone_bottom = resistance_price * (1 - RESISTANCE_PROXIMITY_THRESHOLD)
        proximity_zone_top = resistance_price
        
        # Fill proximity zone
        ax1.fill_between(range(len(df_plot)), proximity_zone_bottom, proximity_zone_top,
                        color='red', alpha=0.2, label=f'No-Buy Zone ({RESISTANCE_PROXIMITY_THRESHOLD*100:.1f}%)')
        
        # Add text annotations
        ax1.text(len(df_plot) * 0.7, proximity_zone_bottom, 
                'Buy Signals REJECTED Here', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.5),
                color='white', fontsize=10, ha='center')
        
        # Breakout zone
        breakout_zone_bottom = resistance_price
        breakout_zone_top = resistance_price * (1 + BREAKOUT_CONFIRMATION_PCT)
        
        ax1.fill_between(range(len(df_plot)), breakout_zone_bottom, breakout_zone_top,
                        color='yellow', alpha=0.3, label=f'Breakout Confirmation Zone ({BREAKOUT_CONFIRMATION_PCT*100:.1f}%)')
        
        # Confirmed breakout zone
        confirmed_breakout = resistance_price * (1 + BREAKOUT_CONFIRMATION_PCT)
        ax1.axhline(y=confirmed_breakout, color='green', linestyle='--', linewidth=2,
                   label=f'Confirmed Breakout: ${confirmed_breakout:.5f}')
        
        ax1.text(len(df_plot) * 0.7, confirmed_breakout * 1.001, 
                'Buy Signals ALLOWED Here', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.5),
                color='white', fontsize=10, ha='center')
    
    # Add other S/R levels (faded)
    for i, support in enumerate(levels['support'][:3]):
        ax1.axhline(y=support['price'], color='green', linestyle=':', 
                   linewidth=1, alpha=0.5)
    for i, resistance in enumerate(levels['resistance'][1:4]):
        ax1.axhline(y=resistance['price'], color='red', linestyle=':', 
                   linewidth=1, alpha=0.5)
    
    ax1.set_title(f'{symbol} - Resistance Check Logic Visualization', 
                 fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    
    # Test different price scenarios
    ax2 = plt.subplot2grid((4, 2), (2, 0), colspan=2)
    
    if resistances:
        resistance_price = resistances[0]['price']
        
        # Define test scenarios
        test_prices = [
            ('Below Safe', current_price * 0.995, 'green'),
            ('Near Resistance', resistance_price * 0.998, 'red'),
            ('At Resistance', resistance_price, 'red'),
            ('False Breakout', resistance_price * 1.001, 'yellow'),
            ('Confirmed Breakout', resistance_price * 1.003, 'green')
        ]
        
        x_pos = range(len(test_prices))
        colors = [color for _, _, color in test_prices]
        prices = [price for _, price, _ in test_prices]
        labels = [label for label, _, _ in test_prices]
        
        bars = ax2.bar(x_pos, prices, color=colors, alpha=0.7)
        
        # Add horizontal line for resistance
        ax2.axhline(y=resistance_price, color='red', linestyle='--', linewidth=2)
        ax2.axhline(y=current_price, color='blue', linestyle='--', linewidth=1)
        
        # Add labels
        for i, (bar, label, price) in enumerate(zip(bars, labels, prices)):
            height = bar.get_height()
            
            # Determine if buy allowed
            distance_to_resistance = (resistance_price - price) / resistance_price
            
            if price < resistance_price * (1 - RESISTANCE_PROXIMITY_THRESHOLD):
                decision = "✅ BUY ALLOWED"
                decision_color = 'green'
            elif price < resistance_price:
                decision = "❌ BUY REJECTED"
                decision_color = 'red'
            elif price < resistance_price * (1 + BREAKOUT_CONFIRMATION_PCT):
                decision = "⚠️ WAIT CONFIRM"
                decision_color = 'orange'
            else:
                decision = "✅ BREAKOUT BUY"
                decision_color = 'green'
            
            ax2.text(bar.get_x() + bar.get_width()/2., height * 1.001,
                    f'${price:.5f}\n{decision}', 
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=decision_color, alpha=0.3))
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=45)
        ax2.set_ylabel('Price ($)', fontsize=12)
        ax2.set_title('Buy Signal Decision at Different Price Levels', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
    
    # Signal type comparison
    ax3 = plt.subplot2grid((4, 2), (3, 0))
    
    signal_types = ['Normal Buy', 'Strong Buy', 'Normal Sell', 'Strong Sell']
    signal_decisions = []
    
    if resistances and current_price < resistances[0]['price']:
        distance_pct = (resistances[0]['price'] - current_price) / resistances[0]['price']
        
        # Normal Buy
        if distance_pct <= RESISTANCE_PROXIMITY_THRESHOLD:
            signal_decisions.append(('REJECTED', 'red'))
        else:
            signal_decisions.append(('ALLOWED', 'green'))
        
        # Strong Buy
        if distance_pct <= RESISTANCE_PROXIMITY_THRESHOLD * 0.5:
            signal_decisions.append(('REJECTED', 'red'))
        elif distance_pct <= RESISTANCE_PROXIMITY_THRESHOLD:
            signal_decisions.append(('CAUTION', 'orange'))
        else:
            signal_decisions.append(('ALLOWED', 'green'))
        
        # Sells always allowed
        signal_decisions.append(('ALLOWED', 'green'))
        signal_decisions.append(('ALLOWED', 'green'))
    else:
        signal_decisions = [('ALLOWED', 'green')] * 4
    
    # Create color-coded table
    cell_colors = []
    cell_text = []
    
    for signal, (decision, color) in zip(signal_types, signal_decisions):
        cell_text.append([signal, decision])
        if color == 'green':
            cell_colors.append(['lightgray', 'lightgreen'])
        elif color == 'red':
            cell_colors.append(['lightgray', 'lightcoral'])
        else:
            cell_colors.append(['lightgray', 'lightyellow'])
    
    table = ax3.table(cellText=cell_text, cellColours=cell_colors,
                     colLabels=['Signal Type', 'Decision'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    ax3.axis('off')
    ax3.set_title('Signal Type Behavior Near Resistance', fontsize=14, fontweight='bold')
    
    # Summary text
    ax4 = plt.subplot2grid((4, 2), (3, 1))
    ax4.axis('off')
    
    if resistances:
        resistance_distance = ((resistances[0]['price'] - current_price) / current_price) * 100
        
        summary_text = f"""
    📊 RESISTANCE CHECK SUMMARY
    {'='*35}
    
    Current Price: ${current_price:.5f}
    Nearest Resistance: ${resistances[0]['price']:.5f}
    Distance: {resistance_distance:.2f}%
    
    Proximity Threshold: {RESISTANCE_PROXIMITY_THRESHOLD*100:.1f}%
    Breakout Confirm: {BREAKOUT_CONFIRMATION_PCT*100:.1f}%
    
    Current Status:
    """
        
        if resistance_distance < 0:  # Above resistance
            if abs(resistance_distance) >= BREAKOUT_CONFIRMATION_PCT * 100:
                summary_text += "✅ CONFIRMED BREAKOUT\nAll buy signals allowed"
            else:
                summary_text += "⚠️ UNCONFIRMED BREAKOUT\nWait for confirmation"
        elif resistance_distance <= RESISTANCE_PROXIMITY_THRESHOLD * 100:
            summary_text += "❌ TOO CLOSE TO RESISTANCE\nBuy signals rejected"
        else:
            summary_text += "✅ SAFE DISTANCE\nAll signals allowed"
    else:
        summary_text = "No resistance levels found"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    return levels


def simulate_trading_decisions(df, symbol='ETHUSDT'):
    """Simulate trading decisions with resistance check"""
    print("\n🎯 Simulating Trading Decisions...")
    
    # Create analyzer
    sr_analyzer = SupportResistanceAnalyzer(lookback_period=100, merge_threshold=0.001)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot last 50 candles
    df_plot = plot_candlestick(ax1, df, last_n=50)
    
    # Get S/R levels for each candle
    decisions = []
    
    for i in range(len(df_plot)):
        # Get data up to this point
        historical_data = df.iloc[:df.index.get_loc(df_plot.index[i])+1]
        
        # Get levels
        levels = sr_analyzer.get_all_levels(historical_data)
        current_price = historical_data['close'].iloc[-1]
        
        # Simulate different signal types
        for signal_type, signal_strength in [('BUY', 0.75), ('STRONG_BUY', 0.95)]:
            decision = check_resistance_proximity_logic(
                current_price, 
                signal_type, 
                levels['resistance'],
                RESISTANCE_PROXIMITY_THRESHOLD,
                BREAKOUT_CONFIRMATION_PCT
            )
            
            if signal_type == 'BUY' and decision['can_trade']:
                # Plot buy signal
                ax1.scatter(i, current_price * 0.999, marker='^', 
                          color='green', s=100, zorder=5,
                          label='Buy Allowed' if i == 0 else "")
            elif signal_type == 'BUY' and not decision['can_trade']:
                # Plot rejected buy
                ax1.scatter(i, current_price * 0.999, marker='x', 
                          color='red', s=100, zorder=5,
                          label='Buy Rejected' if i == 0 else "")
    
    # Plot S/R levels at last candle
    current_levels = sr_analyzer.get_all_levels(df_plot)
    
    for resistance in current_levels['resistance'][:3]:
        ax1.axhline(y=resistance['price'], color='red', linestyle='--', 
                   linewidth=1, alpha=0.7)
    
    for support in current_levels['support'][:3]:
        ax1.axhline(y=support['price'], color='green', linestyle='--', 
                   linewidth=1, alpha=0.7)
    
    ax1.set_title(f'{symbol} - Trading Decision Simulation', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Volume
    colors = ['green' if df_plot['close'].iloc[i] >= df_plot['open'].iloc[i] else 'red' 
              for i in range(len(df_plot))]
    ax2.bar(range(len(df_plot)), df_plot['volume'], color=colors, alpha=0.5)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def check_resistance_proximity_logic(current_price, signal_type, resistances, 
                                   proximity_threshold, breakout_threshold):
    """Simulate the resistance check logic"""
    
    if signal_type not in ['BUY', 'STRONG_BUY']:
        return {'can_trade': True, 'reason': 'Not a buy signal'}
    
    if not resistances:
        return {'can_trade': True, 'reason': 'No resistance found'}
    
    nearest_resistance = resistances[0]
    resistance_price = nearest_resistance['price']
    distance_pct = (resistance_price - current_price) / resistance_price
    
    # Above resistance (breakout)
    if current_price > resistance_price:
        breakout_distance = (current_price - resistance_price) / resistance_price
        if breakout_distance >= breakout_threshold:
            return {'can_trade': True, 'reason': 'Confirmed breakout'}
        else:
            return {'can_trade': False, 'reason': 'Unconfirmed breakout'}
    
    # Below resistance
    if distance_pct <= proximity_threshold:
        if signal_type == 'STRONG_BUY' and distance_pct > proximity_threshold * 0.5:
            return {'can_trade': True, 'reason': 'Strong signal override'}
        else:
            return {'can_trade': False, 'reason': 'Too close to resistance'}
    
    return {'can_trade': True, 'reason': 'Safe distance from resistance'}


def run_resistance_check_analysis(symbol='ETHUSDT'):
    """Run complete resistance check analysis"""
    print(f"\n{'='*60}")
    print(f"🚀 RESISTANCE CHECK ANALYSIS FOR {symbol}")
    print(f"{'='*60}")
    
    # Fetch data
    df = fetch_data(symbol)
    if df is None:
        return
    
    # Run visualizations
    levels = visualize_resistance_check_logic(df, symbol)
    simulate_trading_decisions(df, symbol)
    
    return levels


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
        run_resistance_check_analysis(symbol)
    else:
        # Default: Run for ETHUSDT
        run_resistance_check_analysis('ETHUSDT')
        
        print("\n💡 USAGE:")
        print("python test_technical_visualization.py              # Default ETHUSDT")
        print("python test_technical_visualization.py BTCUSDT      # Specific symbol")
        print("python test_technical_visualization.py SOLUSDT      # Another symbol")