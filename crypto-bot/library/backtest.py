# library/backtest.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SimpleBacktest:
    """Simple backtesting engine for trading signals"""
    
    def __init__(self, initial_capital=10000, commission=0.001):
        """
        Initialize backtester
        
        Args:
            initial_capital: starting capital
            commission: trading commission (0.1% default)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.reset()
    
    def reset(self):
        """Reset backtester state"""
        self.capital = self.initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = [self.initial_capital]
    
    def run_backtest(self, predictions, actual_prices, atr_values):
        """
        Run backtest on predictions
        
        Args:
            predictions: model predictions (direction, confidence, tp1-3, sl, timing, risk)
            actual_prices: actual price data
            atr_values: ATR values for denormalizing targets
            
        Returns:
            results dictionary
        """
        self.reset()

        print(f"\n=== DEBUG BACKTEST ===")
        print(f"Total predictions: {len(predictions)}")

        
        
        # Check direction distribution
        directions = predictions[:, 0]
        print(f"Direction range: {directions.min():.3f} to {directions.max():.3f}")
        print(f"Direction mean: {directions.mean():.3f}")
        
        # Count potential trades
        high_conf = predictions[:, 1] >= 0.6
        strong_dir = np.abs(predictions[:, 0]) >= 0.05  # Check your threshold
        valid_trades = high_conf & strong_dir
        print(f"High confidence (>=0.6): {high_conf.sum()}")
        print(f"Strong direction (>=0.05): {strong_dir.sum()}")
        print(f"Valid trades: {valid_trades.sum()}")
        
        # Sample first few valid signals
        valid_idx = np.where(valid_trades)[0][:5]
        for idx in valid_idx:
            print(f"  Signal {idx}: Dir={predictions[idx,0]:.3f}, Conf={predictions[idx,1]:.3f}")
        
        for i in range(len(predictions)):
            # Get prediction
            
            direction = predictions[i][0]
            confidence = predictions[i][1]
            confidence = np.clip(confidence, 0, 1)
            tp1_atr = predictions[i][2]
            tp2_atr = predictions[i][3]
            tp3_atr = predictions[i][4]
            sl_atr = predictions[i][5]
            sl_atr = np.clip(sl_atr, 0.3, 1.0)
            
            # Skip if low confidence or no signal
            if confidence < 0.05 or abs(direction) < 0.01:  # Turun confidence ke 0.5
                self.equity_curve.append(self.capital)
                continue
            if i < 10:  # First 10 only
                print(f"{i}: Dir={direction:.3f}, Conf={confidence:.3f}, Skip={confidence < 0.2 or abs(direction) < 0.01}")
            
            # Convert ATR units to price
            current_price = actual_prices[i]
            current_atr = atr_values[i]
            
            if direction > 0:  # BUY
                tp1 = current_price + (tp1_atr * current_atr)
                tp2 = current_price + (tp2_atr * current_atr)
                tp3 = current_price + (tp3_atr * current_atr)
                sl = current_price - (sl_atr * current_atr)
                trade_type = 'BUY'
            else:  # SELL
                tp1 = current_price - (tp1_atr * current_atr)
                tp2 = current_price - (tp2_atr * current_atr)
                tp3 = current_price - (tp3_atr * current_atr)
                sl = current_price + (sl_atr * current_atr)
                trade_type = 'SELL'

            if i < 5:  # Debug first 5 trades only
                print(f"\n=== TRADE DEBUG #{i} ===")
                print(f"Direction: {direction:.3f}")
                print(f"Current Price: {current_price:.2f}")
                print(f"Trade Type: {trade_type}")
                print(f"TP1: {tp1:.2f} (Distance: {abs(tp1-current_price):.2f})")
                print(f"SL: {sl:.2f} (Distance: {abs(sl-current_price):.2f})")
                
            # Sanity check
            if trade_type == 'BUY':
                if tp1 <= current_price:
                    print("ERROR: BUY TP1 is below entry!")
                if sl >= current_price:
                    print("ERROR: BUY SL is above entry!")
            else:  # SELL
                if tp1 >= current_price:
                    print("ERROR: SELL TP1 is above entry!")
                if sl <= current_price:
                    print("ERROR: SELL SL is below entry!")
            
            # Simulate trade
            trade_result = self._simulate_trade(
                i, actual_prices, trade_type, current_price,
                tp1, tp2, tp3, sl, confidence
            )
            
            if trade_result:
                self.trades.append(trade_result)
            
            self.equity_curve.append(self.capital)
        
        return self._calculate_metrics()
    
    def _simulate_trade(self, start_idx, prices, trade_type, entry, tp1, tp2, tp3, sl, confidence=1.0):
        """Simulate a single trade"""
        position_size = self.capital * 0.2 * confidence  # Risk 2% per trade

        # Look forward maximum 20 candles
        for j in range(start_idx + 1, min(start_idx + 20, len(prices))):
            current_price = prices[j]

            if trade_type == 'BUY':
                # Check stop loss
                if current_price <= sl:
                    exit_price = sl
                    profit = position_size * ((exit_price - entry) / entry)
                    self.capital += profit * (1 - self.commission)
                    return {
                        'type': trade_type,
                        'entry': entry,
                        'exit': exit_price,
                        'profit': profit,
                        'result': 'LOSS'
                    }
                # Check take profit
                if current_price >= tp1:
                    exit_price = tp1
                    profit = position_size * ((exit_price - entry) / entry)
                    self.capital += profit * (1 - self.commission)
                    return {
                        'type': trade_type,
                        'entry': entry,
                        'exit': exit_price,
                        'profit': profit,
                        'result': 'WIN'
                    }
            else:  # SELL
                # Check stop loss
                if current_price >= sl:
                    exit_price = sl
                    profit = position_size * ((entry - exit_price) / entry)
                    self.capital += profit * (1 - self.commission)
                    return {
                        'type': trade_type,
                        'entry': entry,
                        'exit': exit_price,
                        'profit': profit,
                        'result': 'LOSS'
                    }
                # Check take profit
                if current_price <= tp1:
                    exit_price = tp1
                    profit = position_size * ((entry - exit_price) / entry)
                    self.capital += profit * (1 - self.commission)
                    return {
                        'type': trade_type,
                        'entry': entry,
                        'exit': exit_price,
                        'profit': profit,
                        'result': 'WIN'
                    }
        # Trade didn't close within 20 candles
        return None
    
    def _calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'final_capital': self.initial_capital
            }
        
        # Win rate
        wins = sum(1 for t in self.trades if t['result'] == 'WIN')
        win_rate = wins / len(self.trades) * 100
        
        # Profit factor
        gross_profit = sum(t['profit'] for t in self.trades if t['profit'] > 0)
        gross_loss = abs(sum(t['profit'] for t in self.trades if t['profit'] < 0))

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float('inf')
        else:
            profit_factor = 0.0 
        
        # Total return
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        
        # Max drawdown
        peak = self.initial_capital
        max_dd = 0
        for value in self.equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        if self.trades:
            # Analyze trades
            wins = [t for t in self.trades if t['result'] == 'WIN']
            losses = [t for t in self.trades if t['result'] == 'LOSS']
            
            if wins:
                avg_win = np.mean([t['profit'] for t in wins])
                print(f"\nAverage Win: ${avg_win:.2f}")
            if losses:
                avg_loss = np.mean([abs(t['profit']) for t in losses])
                print(f"Average Loss: ${avg_loss:.2f}")
                
            # Risk/Reward
            if wins and losses:
                rr_ratio = avg_win / avg_loss
                print(f"Risk/Reward Ratio: 1:{rr_ratio:.2f}")
        return {
            'total_trades': len(self.trades),
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'total_return': round(total_return, 2),
            'max_drawdown': round(max_dd, 2),
            'final_capital': round(self.capital, 2)
        }
    
    def plot_results(self):
        """Plot equity curve"""
        plt.figure(figsize=(12, 6))
        
        # Equity curve
        plt.subplot(2, 1, 1)
        plt.plot(self.equity_curve)
        plt.title('Equity Curve')
        plt.ylabel('Capital')
        plt.grid(True)
        
        # Drawdown
        plt.subplot(2, 1, 2)
        peak = self.initial_capital
        drawdown = []
        for value in self.equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            drawdown.append(dd)
        
        plt.fill_between(range(len(drawdown)), drawdown, alpha=0.3, color='red')
        plt.title('Drawdown %')
        plt.ylabel('Drawdown %')
        plt.xlabel('Trades')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


def backtest_model(model, X_test, Y_test, prices_test, atr_test):
    """
    Quick function to backtest a model
    
    Args:
        model: trained model
        X_test: test features
        Y_test: test labels (for comparison)
        prices_test: actual prices
        atr_test: ATR values
        
    Returns:
        backtest results
    """
    # Get predictions
    predictions = model.predict(X_test)

    print(f"Sample predictions (first 5):")
    for i in range(min(5, len(predictions))):
        print(f"Dir: {predictions[i][0]:.3f}, Conf: {predictions[i][1]:.3f}")
    
    # Run backtest
    backtester = SimpleBacktest()
    results = backtester.run_backtest(predictions, prices_test, atr_test)
    
    # Print results
    print("\n=== Backtest Results ===")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']}%")
    print(f"Profit Factor: {results['profit_factor']}")
    print(f"Total Return: {results['total_return']}%")
    print(f"Max Drawdown: {results['max_drawdown']}%")
    print(f"Final Capital: ${results['final_capital']}")
    
    # Plot
    backtester.plot_results()
    
    return results, backtester


# Usage example
if __name__ == "__main__":
    # Create dummy data for testing
    np.random.seed(42)
    
    # Dummy predictions
    predictions = np.random.randn(100, 8)
    predictions[:, 0] = np.tanh(predictions[:, 0])  # Direction
    predictions[:, 1] = np.abs(predictions[:, 1]) / 2  # Confidence
    predictions[:, 2:5] = np.abs(predictions[:, 2:5])  # TPs
    predictions[:, 5] = np.abs(predictions[:, 5])  # SL
    
    # Dummy prices
    prices = 50000 + np.cumsum(np.random.randn(100) * 100)
    
    # Dummy ATR
    atr = np.ones(100) * 100
    
    # Run backtest
    bt = SimpleBacktest()
    results = bt.run_backtest(predictions, prices, atr)
    print(results)