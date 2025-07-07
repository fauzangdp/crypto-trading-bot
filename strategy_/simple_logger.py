import logging
from datetime import datetime

class TradingLogger:
    """Simplified logger for important events only"""
    
    def __init__(self):
        self.start_time = datetime.now()
        
    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color codes
        colors = {
            "INFO": "\033[0m",      # Default
            "TRADE": "\033[92m",    # Green
            "SIGNAL": "\033[93m",   # Yellow
            "ERROR": "\033[91m",    # Red
            "PROFIT": "\033[96m",   # Cyan
        }
        
        color = colors.get(level, colors["INFO"])
        reset = "\033[0m"
        
        print(f"{color}[{timestamp}] {message}{reset}")
    
    def trade(self, message):
        self.log(f"💰 {message}", "TRADE")
        
    def signal(self, message):
        self.log(f"📊 {message}", "SIGNAL")
        
    def error(self, message):
        self.log(f"❌ {message}", "ERROR")
        
    def profit(self, message):
        self.log(f"💎 {message}", "PROFIT")