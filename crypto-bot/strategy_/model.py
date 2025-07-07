import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

from library.binance_connector import generate_trading_features
from features_binance import create_lstm_features, get_live_lstm_features



torch.serialization.add_safe_globals([
    np.dtypes.Int64DType,
    np.dtypes.Float64DType,
    np.dtypes.Float32DType
])

torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])


# ==========================================
# 🎛️ MODEL CONFIGURATION CENTER
# ==========================================
class ModelConfig:
    """Centralized configuration untuk easy adjustment"""
    
    # === ARCHITECTURE SETTINGS ===
    INPUT_SIZE = 7
    HIDDEN_SIZE = 128           # Increased from 32 to 128
    NUM_LAYERS = 3              # Increased from 2 to 3
    DROPOUT_LSTM = 0.3          # Increased from 0.2
    DROPOUT_FC = 0.4            # Increased from 0.3
    
    # === TRAINING HYPERPARAMETERS ===
    LEARNING_RATE = 0.0015       # Increased from 0.0005
    SEQUENCE_LENGTH = 60        # Keep same
    BATCH_SIZE = 64             # Increased from 32
    EPOCHS = 100                # Increased from 50
    
    # === OPTIMIZER SETTINGS ===
    OPTIMIZER_TYPE = 'AdamW'    # Changed from Adam to AdamW
    WEIGHT_DECAY = 0.01         # L2 regularization
    SCHEDULER_FACTOR = 0.7      # LR reduction factor
    SCHEDULER_PATIENCE = 8      # Reduced from 10
    SCHEDULER_MIN_LR = 1e-6
    
    # === TRAINING CONTROL ===
    EARLY_STOP_PATIENCE = 18    # Reduced from 20
    GRADIENT_CLIP = 0.7         # Reduced from 1.0
    VALIDATION_SPLIT = 0.15     # Reduced from 0.2 (more training data)
    
    # === LOSS FUNCTION WEIGHTS ===
    EXTREME_WEIGHT = 5.5        # Reduced from 3.0
    NEUTRAL_WEIGHT = 1.5        # Increased from 0.5
    FALSE_EXTREME_PENALTY = 0.3  # Reduced from 0.3
    NEUTRAL_PENALTY = 0.0       # Disabled from -0.1
    
    # === MODEL BEHAVIOR ===
    USE_CONFIDENCE_SCALING = False  # Disable confidence multiplication
    ENABLE_AGGRESSIVE_LOSS = True   # Use enhanced loss function

    # === ZONE THRESHOLD SETTINGS === 
    NEUTRAL_HIGH = 0.6         # Diperlebar dari 0.3
    NEUTRAL_LOW = -0.6          # Diperlebar dari -0.3
    BUY_THRESHOLD = 0.7         # Sesuaikan dengan neutral_high
    SELL_THRESHOLD = -0.7       # Sesuaikan dengan neutral_low
    
    # === LOSS FUNCTION WEIGHTS ===
    EXTREME_WEIGHT = 5.0        # Turunkan dari 5.0
    NEUTRAL_WEIGHT = 0.7       # Naikkan dari 0.7
    NEUTRAL_ACCURACY_BONUS = 0.5  # BARU!
    FALSE_EXTREME_PENALTY = 0.5   # Naikkan dari 0.5
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n🎛️ MODEL CONFIGURATION:")
        print("=" * 40)
        print(f"Architecture:")
        print(f"  Hidden Size: {cls.HIDDEN_SIZE}")
        print(f"  Layers: {cls.NUM_LAYERS}")
        print(f"  Dropouts: LSTM={cls.DROPOUT_LSTM}, FC={cls.DROPOUT_FC}")
        print(f"\nTraining:")
        print(f"  Learning Rate: {cls.LEARNING_RATE}")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Epochs: {cls.EPOCHS}")
        print(f"  Optimizer: {cls.OPTIMIZER_TYPE}")
        print(f"\nLoss Weights:")
        print(f"  Extreme: {cls.EXTREME_WEIGHT}")
        print(f"  Neutral: {cls.NEUTRAL_WEIGHT}")
        print(f"  False Extreme Penalty: {cls.FALSE_EXTREME_PENALTY}")
        print("=" * 40)


class TradingDataset(Dataset):
    """Dataset for trading sequences"""
    def __init__(self, X, y_decision, seq_len=60):
        self.X = torch.FloatTensor(X)
        self.y_decision = torch.FloatTensor(y_decision)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.X) - self.seq_len + 1
    
    def __getitem__(self, idx):
        X = self.X[idx:idx + self.seq_len]
        y_decision = self.y_decision[idx + self.seq_len - 1]
        
        return X, y_decision
    


class ImprovedLSTM(nn.Module):
    """Enhanced LSTM with configurable parameters"""
    def __init__(self, input_size=None, hidden_size=None, num_layers=None):
        super().__init__()
        
        # Use config if parameters not provided
        input_size = input_size or ModelConfig.INPUT_SIZE
        hidden_size = hidden_size or ModelConfig.HIDDEN_SIZE
        num_layers = num_layers or ModelConfig.NUM_LAYERS
        
        # Enhanced LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=ModelConfig.DROPOUT_LSTM if num_layers > 1 else 0
        )
        
        # Enhanced head architecture
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 32)
        self.dropout1 = nn.Dropout(ModelConfig.DROPOUT_FC)
        self.dropout2 = nn.Dropout(ModelConfig.DROPOUT_FC * 0.7)
        
        # Decision head
        self.decision_head = nn.Linear(32, 1)
        
        # Confidence head (optional)
        self.confidence_head = nn.Linear(32, 1) if ModelConfig.USE_CONFIDENCE_SCALING else None
        
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        # Enhanced processing
        x = torch.relu(self.fc1(last_output))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Decision output
        decision = torch.tanh(self.decision_head(x)).squeeze()
        
        if ModelConfig.USE_CONFIDENCE_SCALING and self.confidence_head is not None:
            # Confidence scaling (optional)
            confidence = torch.sigmoid(self.confidence_head(x)).squeeze()
            final_decision = decision * confidence
        else:
            # Direct decision output
            final_decision = decision
        
        return final_decision


class TradingModel:
    """Enhanced Trading Model with centralized configuration"""
    
    def __init__(self, seq_len=None, hidden_size=None, lr=None):
        # Use config defaults if not provided
        self.seq_len = seq_len or ModelConfig.SEQUENCE_LENGTH
        hidden_size = hidden_size or ModelConfig.HIDDEN_SIZE
        lr = lr or ModelConfig.LEARNING_RATE
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Device: {self.device}")
        
        # Print configuration
        ModelConfig.print_config()
        
        self.scaler = StandardScaler()
        
        # Enhanced model
        self.model = ImprovedLSTM(
            input_size=ModelConfig.INPUT_SIZE,
            hidden_size=hidden_size,
            num_layers=ModelConfig.NUM_LAYERS
        ).to(self.device)
        
        # Enhanced optimizer
        if ModelConfig.OPTIMIZER_TYPE == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=lr,
                weight_decay=ModelConfig.WEIGHT_DECAY
            )
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Enhanced scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=ModelConfig.SCHEDULER_FACTOR,
            patience=ModelConfig.SCHEDULER_PATIENCE,
            min_lr=ModelConfig.SCHEDULER_MIN_LR
        )
        
        # Enhanced loss function
        self.criterion_decision = self._create_balanced_loss()
        
        # Training history
        self.training_history = {
            'train_loss': [], 'val_loss': [], 
            'train_mse': [], 'val_mse': [],
            'train_mae': [], 'val_mae': [],
            'val_r2': []
        }
        
        self.optimal_thresholds = None
        self.best_model_state = None
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"🧠 Model parameters: {total_params:,}")
    
    def _create_balanced_loss(self):
        """Enhanced loss function dengan neutral zone incentive"""
        def balanced_mse_loss(pred, target):
            # Basic MSE
            mse = torch.mean((pred - target) ** 2)
            
            if not ModelConfig.ENABLE_AGGRESSIVE_LOSS:
                return mse
            
            # Enhanced weights
            weights = torch.ones_like(target)
            
            # UPDATED: Gunakan ModelConfig thresholds
            extreme_mask = torch.abs(target) > 0.7
            neutral_mask = (torch.abs(target) >= ModelConfig.NEUTRAL_LOW) & \
                        (torch.abs(target) <= ModelConfig.NEUTRAL_HIGH)
            
            weights[extreme_mask] = ModelConfig.EXTREME_WEIGHT
            weights[neutral_mask] = ModelConfig.NEUTRAL_WEIGHT
            
            weighted_mse = torch.mean(weights * (pred - target) ** 2)
            
            # False extreme penalty
            pred_extreme_mask = torch.abs(pred) > 0.5
            target_neutral_mask = (torch.abs(target) < ModelConfig.NEUTRAL_HIGH)
            false_extreme_penalty = ModelConfig.FALSE_EXTREME_PENALTY * torch.mean(
                (pred[pred_extreme_mask & target_neutral_mask]) ** 2
            ) if torch.any(pred_extreme_mask & target_neutral_mask) else 0
            
            # === BARU: Neutral Accuracy Bonus ===
            pred_neutral_mask = (torch.abs(pred) >= ModelConfig.NEUTRAL_LOW) & \
                            (torch.abs(pred) <= ModelConfig.NEUTRAL_HIGH)
            target_neutral_mask = (torch.abs(target) >= ModelConfig.NEUTRAL_LOW) & \
                                (torch.abs(target) <= ModelConfig.NEUTRAL_HIGH)
            
            # Reward accurate neutral predictions
            correct_neutral = pred_neutral_mask & target_neutral_mask
            neutral_bonus = -ModelConfig.NEUTRAL_ACCURACY_BONUS * torch.mean(
                correct_neutral.float()
            ) if torch.any(correct_neutral) else 0
            
            return weighted_mse + false_extreme_penalty + neutral_bonus
        
        return balanced_mse_loss
    
    def prepare_data(self, features_df, labels_df, test_size=None):
        """Enhanced data preparation"""
        test_size = test_size or ModelConfig.VALIDATION_SPLIT
        
        print("\n📊 Preparing data...")
        
        # Extract data
        X = features_df.values
        y_decision = labels_df['decision'].values
        
        print(f"   Total samples: {len(X)}")
        print(f"   Decision range: [{y_decision.min():.3f}, {y_decision.max():.3f}]")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train/val split
        split_idx = int(len(X_scaled) * (1 - test_size))
        X_train = X_scaled[:split_idx]
        X_val = X_scaled[split_idx:]
        y_decision_train = y_decision[:split_idx]
        y_decision_val = y_decision[split_idx:]
        
        # Create datasets
        train_dataset = TradingDataset(X_train, y_decision_train, self.seq_len)
        val_dataset = TradingDataset(X_val, y_decision_val, self.seq_len)
        
        print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        print(f"   Train/Val split: {(1-test_size)*100:.1f}%/{test_size*100:.1f}%")
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset, epochs=None, batch_size=None):
        """Enhanced training loop with better monitoring"""
        epochs = epochs or ModelConfig.EPOCHS
        batch_size = batch_size or ModelConfig.BATCH_SIZE
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"\n🚀 Enhanced training for {epochs} epochs...")
        print(f"   📊 Batch size: {batch_size}")
        print(f"   🎯 Early stopping patience: {ModelConfig.EARLY_STOP_PATIENCE}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            train_preds = []
            train_targets = []
            
            for X, y_dec in train_loader:
                X = X.to(self.device)
                y_dec = y_dec.to(self.device)
                
                # Forward
                dec_pred = self.model(X)
                loss = self.criterion_decision(dec_pred, y_dec)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), ModelConfig.GRADIENT_CLIP)
                self.optimizer.step()
                
                train_losses.append(loss.item())
                train_preds.extend(dec_pred.detach().cpu().numpy())
                train_targets.extend(y_dec.cpu().numpy())
            
            # Validation
            self.model.eval()
            val_losses = []
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for X, y_dec in val_loader:
                    X = X.to(self.device)
                    y_dec = y_dec.to(self.device)
                    
                    dec_pred = self.model(X)
                    loss = self.criterion_decision(dec_pred, y_dec)
                    
                    val_losses.append(loss.item())
                    val_preds.extend(dec_pred.cpu().numpy())
                    val_targets.extend(y_dec.cpu().numpy())
            
            # Enhanced metrics
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            
            train_mse = mean_squared_error(train_targets, train_preds)
            val_mse = mean_squared_error(val_targets, val_preds)
            train_mae = mean_absolute_error(train_targets, train_preds)
            val_mae = mean_absolute_error(val_targets, val_preds)
            val_r2 = r2_score(val_targets, val_preds)
            
            # Prediction range analysis
            val_preds_np = np.array(val_preds)
            val_targets_np = np.array(val_targets)
            pred_range = f"[{val_preds_np.min():.3f}, {val_preds_np.max():.3f}]"
            target_range = f"[{val_targets_np.min():.3f}, {val_targets_np.max():.3f}]"
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_mse'].append(train_mse)
            self.training_history['val_mse'].append(val_mse)
            self.training_history['train_mae'].append(train_mae)
            self.training_history['val_mae'].append(val_mae)
            self.training_history['val_r2'].append(val_r2)
            
            # Learning rate scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Enhanced progress reporting
            if (epoch + 1) % 5 == 0:  # More frequent reporting
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Loss - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
                print(f"  MAE - Train: {train_mae:.4f}, Val: {val_mae:.4f}")
                print(f"  R² - Val: {val_r2:.3f}")
                print(f"  Ranges - Pred: {pred_range}, Target: {target_range}")
                print(f"  LR: {current_lr:.2e}")
                
                # Prediction distribution analysis
                neutral_ratio = np.mean(np.abs(val_preds_np) < 0.3)
                extreme_ratio = np.mean(np.abs(val_preds_np) > 0.7)
                print(f"  Pred distribution - Neutral: {neutral_ratio:.2f}, Extreme: {extreme_ratio:.2f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= ModelConfig.EARLY_STOP_PATIENCE:
                    print(f"\n⛔ Early stopping at epoch {epoch+1}")
                    self.model.load_state_dict(self.best_model_state)
                    break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        print(f"\n✅ Enhanced training completed!")
        print(f"   Best Val Loss: {best_val_loss:.4f}")
        print(f"   Final Val R²: {val_r2:.3f}")
    
    def optimize_thresholds(self, features_df, labels_df):
        """Placeholder for compatibility"""
        print("\n🎯 Threshold optimization not needed for regression")
        self.optimal_thresholds = None
    
    def evaluate_trading_performance(self, predictions, true_labels):
        """Enhanced evaluation for regression"""
        pred = predictions['decision'].values
        true = true_labels['decision'].values
        
        mse = mean_squared_error(true, pred)
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)
        
        # Direction accuracy
        direction_acc = np.mean(np.sign(pred) == np.sign(true))
        
        # Zone accuracy
        def get_zone(val):
            if val > ModelConfig.BUY_THRESHOLD:      # Gunakan config
                return 1  # Buy
            elif val < ModelConfig.SELL_THRESHOLD:   # Gunakan config
                return -1  # Sell
            else:
                return 0  # Neutral
        
        pred_zones = [get_zone(v) for v in pred]
        true_zones = [get_zone(v) for v in true]
        zone_accuracy = np.mean(np.array(pred_zones) == np.array(true_zones))
        
        print(f"\n📊 Enhanced Regression Metrics:")
        print(f"   MSE: {mse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   R²: {r2:.3f}")
        print(f"   Direction Accuracy: {direction_acc:.3f}")
        print(f"   Zone Accuracy: {zone_accuracy:.3f}")
        print(f"   Prediction Range: [{pred.min():.3f}, {pred.max():.3f}]")
        print(f"   Target Range: [{true.min():.3f}, {true.max():.3f}]")
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_acc,
            'zone_accuracy': zone_accuracy
        }
    
    def predict(self, features_df):
        """Simple prediction WITHOUT post-processing"""
        self.model.eval()
        
        X = self.scaler.transform(features_df.values)
        dataset = TradingDataset(X, np.zeros(len(X)), self.seq_len)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        all_decisions = []
        
        with torch.no_grad():
            for X, _ in loader:
                X = X.to(self.device)
                dec_pred = self.model(X)
                all_decisions.extend(dec_pred.cpu().numpy())
        
        # NO POST-PROCESSING - let model learn proper distribution
        results = pd.DataFrame({
            'decision': all_decisions,
            'sl': np.zeros(len(all_decisions))
        })
        
        if len(features_df) >= self.seq_len:
            # Predictions are for FUTURE timestamps
            try:
                results.index = features_df.index[self.seq_len-1:self.seq_len-1+len(results)]
            except ValueError as e:
                print(f"⚠️ Index assignment failed: {e}")
                print(f"🔍 Features length: {len(features_df)}, Results length: {len(results)}")
                # Use default integer index instead
                pass
        
        return results
    
    def _apply_signal_filter(self, results):
        """No filtering - keep it simple"""
        return results
    
    def save_model(self, path):
        """Save model with enhanced configuration"""
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler': self.scaler,
            'seq_len': self.seq_len,
            'training_history': self.training_history,
            'optimal_thresholds': self.optimal_thresholds,
            # Enhanced model config
            'model_config': {
                'seq_len': self.seq_len,
                'hidden_size': self.model.lstm.hidden_size,
                'num_layers': self.model.lstm.num_layers,
                'input_size': self.model.lstm.input_size,
                'model_class': self.model.__class__.__name__,
                'config_snapshot': {
                    'HIDDEN_SIZE': ModelConfig.HIDDEN_SIZE,
                    'NUM_LAYERS': ModelConfig.NUM_LAYERS,
                    'LEARNING_RATE': ModelConfig.LEARNING_RATE,
                    'BATCH_SIZE': ModelConfig.BATCH_SIZE,
                    'USE_CONFIDENCE_SCALING': ModelConfig.USE_CONFIDENCE_SCALING
                }
            }
        }, path)
        print(f"💾 Enhanced model saved to: {path}")
    
    def load_model(self, path):
        """Load model with backward compatibility"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.scaler = checkpoint['scaler']
        self.seq_len = checkpoint['seq_len']
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        if 'optimal_thresholds' in checkpoint:
            self.optimal_thresholds = checkpoint['optimal_thresholds']
        
        # Print loaded model info if available
        if 'model_config' in checkpoint and 'config_snapshot' in checkpoint['model_config']:
            config = checkpoint['model_config']['config_snapshot']
            print(f"📂 Model loaded from: {path}")
            print(f"   Saved config - Hidden: {config.get('HIDDEN_SIZE', 'N/A')}, "
                  f"Layers: {config.get('NUM_LAYERS', 'N/A')}, "
                  f"LR: {config.get('LEARNING_RATE', 'N/A')}")
        else:
            print(f"📂 Model loaded from: {path}")


class LiveTradingModel(TradingModel):
    """Simple Live Trading Model for real-time predictions"""
    
    def __init__(self, model_path=None):
        """Initialize with optional model loading"""
        super().__init__()
        
        if model_path:
            self.load_model(model_path)
            print(f"✅ Model loaded from: {model_path}")
    
    def get_signal(self, symbol='ETHUSDT', limit=200, client=None):
        """
        Get trading signal from live market
        
        Args:
            symbol: Trading symbol
            limit: Number of candles to analyze
            client: Optional shared Binance client
        
        Returns:
            dict: {
                'action': 'BUY'/'SELL'/'HOLD',
                'confidence': 0.0-1.0,
                'decision': raw value (-1 to 1),
                'timestamp': datetime
            }
        """
        try:
            # Get live features (ONLY ONCE with shared client)
            from features_binance import get_live_lstm_features
            features = get_live_lstm_features(symbol, limit, client=client)
            
            # Make prediction
            predictions = self.predict(features)
            
            # Get latest signal
            latest = predictions.iloc[-1]
            decision = latest['decision']
            
            # Determine action
            if decision > ModelConfig.BUY_THRESHOLD:
                action = 'BUY'
                confidence = min((decision - ModelConfig.BUY_THRESHOLD) / (1 - ModelConfig.BUY_THRESHOLD), 1.0)
            elif decision < ModelConfig.SELL_THRESHOLD:
                action = 'SELL'
                confidence = min((ModelConfig.SELL_THRESHOLD - decision) / (1 + ModelConfig.SELL_THRESHOLD), 1.0)
            else:
                action = 'HOLD'
                confidence = 1 - (abs(decision) / max(ModelConfig.BUY_THRESHOLD, abs(ModelConfig.SELL_THRESHOLD)))
            
            return {
                'symbol': symbol,
                'action': action,
                'confidence': round(confidence, 3),
                'decision': round(decision, 3),
                'timestamp': features.index[-1]
            }
            
        except Exception as e:
            print(f"Error getting signal for {symbol}: {e}")
            return None
    
    def should_trade(self, symbol='ETHUSDT', min_confidence=0.3):
        """
        Simple decision: should we trade now?
        
        Returns:
            tuple: (should_trade: bool, action: str, confidence: float)
        """
        signal = self.get_signal(symbol)
        
        if signal['action'] != 'HOLD' and signal['confidence'] >= min_confidence:
            return True, signal['action'], signal['confidence']
        else:
            return False, 'HOLD', signal['confidence']
    
    def get_position_size(self, balance, signal_confidence, max_risk=0.02):
        """
        Calculate position size based on confidence
        
        Args:
            balance: Account balance
            signal_confidence: Signal confidence (0-1)
            max_risk: Maximum risk per trade (default 2%)
        
        Returns:
            float: Position size as percentage of balance
        """
        # Scale position size with confidence
        base_size = max_risk
        position_size = base_size * signal_confidence
        
        # Ensure minimum and maximum limits
        position_size = max(0.01, min(position_size, max_risk))
        
        return round(position_size, 3)

# ==========================================
# 🎛️ QUICK CONFIGURATION PRESETS
# ==========================================

class QuickConfigs:
    """Predefined configurations for different scenarios"""
    
    @staticmethod
    def conservative():
        """Conservative settings for stable training"""
        ModelConfig.HIDDEN_SIZE = 64
        ModelConfig.NUM_LAYERS = 2
        ModelConfig.LEARNING_RATE = 0.0005
        ModelConfig.BATCH_SIZE = 32
        ModelConfig.EXTREME_WEIGHT = 1.5
        ModelConfig.FALSE_EXTREME_PENALTY = 0.05
    
    @staticmethod
    def aggressive():
        """Aggressive settings for better extreme prediction"""
        ModelConfig.HIDDEN_SIZE = 256
        ModelConfig.NUM_LAYERS = 4
        ModelConfig.LEARNING_RATE = 0.002
        ModelConfig.BATCH_SIZE = 128
        ModelConfig.EXTREME_WEIGHT = 3.0
        ModelConfig.FALSE_EXTREME_PENALTY = 0.2
    
    @staticmethod
    def balanced():
        """Balanced settings (default enhanced)"""
        ModelConfig.HIDDEN_SIZE = 128
        ModelConfig.NUM_LAYERS = 3
        ModelConfig.LEARNING_RATE = 0.001
        ModelConfig.BATCH_SIZE = 64
        ModelConfig.EXTREME_WEIGHT = 2.0
        ModelConfig.FALSE_EXTREME_PENALTY = 0.1


# USAGE EXAMPLE:
"""
# Apply preset configuration
QuickConfigs.aggressive()  # or conservative() or balanced()

# Or manually adjust specific parameters
ModelConfig.HIDDEN_SIZE = 256
ModelConfig.LEARNING_RATE = 0.002
ModelConfig.USE_CONFIDENCE_SCALING = False

# Create model with current configuration
model = TradingModel()
"""