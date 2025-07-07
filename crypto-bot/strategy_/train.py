import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import torch

# Add path untuk import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from library.data_utils import DataLoader
from features import create_lstm_features
from model import TradingModel,ModelConfig



def debug_training_pairs(model, features_df, labels_df):
    """
    SUPER MINIMAL - hanya info kritis, TIDAK PRINT dalam loop
    """
    print("\n🔍 CRITICAL ALIGNMENT CHECK:")
    
    # Create dataset
    from model import TradingDataset
    X = features_df.values
    y_decision = labels_df['decision'].values
    dataset = TradingDataset(X, y_decision, model.seq_len)
    
    # Check HANYA 1 sample untuk timing
    X_sample, y_sample = dataset[0]
    
    feature_start_time = features_df.index[0]
    feature_end_time = features_df.index[model.seq_len - 1]
    label_time = features_df.index[model.seq_len - 1]
    
    print(f"   Timing: Features {feature_start_time.strftime('%H:%M')}→{feature_end_time.strftime('%H:%M')}, Label {label_time.strftime('%H:%M')}={y_sample:.3f}")
    
    # Quick stats TANPA loop
    all_labels = y_decision[model.seq_len-1:]  # Labels yang benar-benar digunakan
    
    positives = (all_labels > 0.5).sum()
    negatives = (all_labels < -0.5).sum()
    neutrals = (np.abs(all_labels) <= 0.5).sum()
    
    print(f"   Labels: Positive={positives}, Negative={negatives}, Neutral={neutrals}")
    print(f"   Range: [{all_labels.min():.3f}, {all_labels.max():.3f}]")
    print(f"   Dataset size: {len(dataset)} pairs")
    
    # CRITICAL CHECK: Is training seeing extreme values?
    strong_positives = (all_labels > 0.8).sum()
    strong_negatives = (all_labels < -0.8).sum()
    print(f"   🎯 STRONG signals: Buy>{strong_positives}, Sell>{strong_negatives}")
    
    if strong_positives == 0 and strong_negatives == 0:
        print(f"   ❌ NO STRONG SIGNALS in training data!")
    else:
        print(f"   ✅ Strong signals present")
    
    print("="*50)

# ==========================================
# 🎛️ TRAINING CONFIGURATION CENTER
# ==========================================
class TrainingConfig:
    """Centralized training configuration untuk easy adjustment"""
    
    # === DATA SETTINGS ===
    SYMBOL = 'btc'
    DATA_LIMIT = 10000
    
    # === MODEL INITIALIZATION ===
    # Note: Jika None, akan menggunakan default dari ModelConfig di model.py
    SEQ_LEN = None          # Default: 60 (dari ModelConfig.SEQUENCE_LENGTH)
    HIDDEN_SIZE = None      # Default: 128 (dari ModelConfig.HIDDEN_SIZE)  
    LEARNING_RATE = None    # Default: 0.001 (dari ModelConfig.LEARNING_RATE)
    
    # === TRAINING PROCESS ===
    EPOCHS = None           # Default: 100 (dari ModelConfig.EPOCHS)
    BATCH_SIZE = None       # Default: 64 (dari ModelConfig.BATCH_SIZE)
    TEST_SIZE = None        # Default: 0.15 (dari ModelConfig.VALIDATION_SPLIT)
    
    # === LEGACY TRAINING OVERRIDE (untuk backward compatibility) ===
    # Hanya digunakan jika ingin override default model config
    LEGACY_SEQ_LEN = 120
    LEGACY_HIDDEN_SIZE = 64
    LEGACY_LR = 0.00001
    LEGACY_EPOCHS = 200
    LEGACY_BATCH_SIZE = 16
    USE_LEGACY_PARAMS = False  # Set True untuk gunakan legacy parameters

    # === SIGNAL THRESHOLDS ===
    STRONG_BUY = 0.6
    BUY = ModelConfig.BUY_THRESHOLD        # 0.4 (dari ModelConfig)
    NEUTRAL_HIGH = ModelConfig.NEUTRAL_HIGH # 0.4
    NEUTRAL_LOW = ModelConfig.NEUTRAL_LOW   # -0.4
    SELL = ModelConfig.SELL_THRESHOLD      # -0.4
    STRONG_SELL = -0.6
    
    # === MODEL NAMING ===
    MODEL_NAME = 'trading_lstm'
    
    @classmethod
    def print_config(cls):
        """Print current training configuration"""
        print("\n🎛️ TRAINING CONFIGURATION:")
        print("=" * 40)
        print(f"Data Settings:")
        print(f"  Symbol: {cls.SYMBOL}")
        print(f"  Data Limit: {cls.DATA_LIMIT}")
        print(f"\nModel Parameters:")
        if cls.USE_LEGACY_PARAMS:
            print(f"  🔴 LEGACY MODE ENABLED")
            print(f"  Seq Length: {cls.LEGACY_SEQ_LEN}")
            print(f"  Hidden Size: {cls.LEGACY_HIDDEN_SIZE}")
            print(f"  Learning Rate: {cls.LEGACY_LR}")
            print(f"  Epochs: {cls.LEGACY_EPOCHS}")
            print(f"  Batch Size: {cls.LEGACY_BATCH_SIZE}")
        else:
            print(f"  🟢 USING MODEL CONFIG DEFAULTS")
            print(f"  Seq Length: {cls.SEQ_LEN or 'Auto (ModelConfig)'}")
            print(f"  Hidden Size: {cls.HIDDEN_SIZE or 'Auto (ModelConfig)'}")
            print(f"  Learning Rate: {cls.LEARNING_RATE or 'Auto (ModelConfig)'}")
            print(f"  Epochs: {cls.EPOCHS or 'Auto (ModelConfig)'}")
            print(f"  Batch Size: {cls.BATCH_SIZE or 'Auto (ModelConfig)'}")
        print(f"\nModel Naming:")
        print(f"  Base Name: {cls.MODEL_NAME}")
        print("=" * 40)


class TrainingVisualizer:
    """Class untuk visualisasi hasil training"""
    
    def __init__(self):
        self.fig_size = (20, 15)
        # Use default style or available style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            plt.style.use('default')
        
    def create_training_report(self, model, predictions, true_labels, save_path='training_report.png'):
        """Create comprehensive training report visualization for regression"""
        
        print("\n📊 Creating training report visualization...")
        
        # Create figure dengan subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Predictions vs Actual Scatter
        ax1 = plt.subplot(3, 3, 1)
        self._plot_predictions_scatter(predictions['decision'], true_labels['decision'], ax1)
        
        # 2. Residuals Distribution
        ax2 = plt.subplot(3, 3, 2)
        self._plot_residuals_distribution(predictions['decision'], true_labels['decision'], ax2)
        
        # 3. Decision Distribution Comparison
        ax3 = plt.subplot(3, 3, 3)
        self._plot_decision_distribution_regression(predictions['decision'], true_labels['decision'], ax3)
        
        # 4. Predictions vs Actual Timeline
        ax4 = plt.subplot(3, 3, 4)
        self._plot_predictions_timeline(predictions['decision'], true_labels['decision'], ax4)
        
        # 5. Error over Time
        ax5 = plt.subplot(3, 3, 5)
        self._plot_error_timeline(predictions['decision'], true_labels['decision'], ax5)
        
        # 6. Performance Metrics
        ax6 = plt.subplot(3, 3, 6)
        self._plot_performance_metrics_regression(predictions, true_labels, ax6)
        
        # 7. Training Loss History
        ax7 = plt.subplot(3, 3, 7)
        self._plot_training_history(model, ax7)
        
        # 8. Q-Q Plot
        ax8 = plt.subplot(3, 3, 8)
        self._plot_qq_plot(predictions['decision'], true_labels['decision'], ax8)
        
        # 9. Prediction Zones Analysis
        ax9 = plt.subplot(3, 3, 9)
        self._plot_prediction_zones(predictions['decision'], true_labels['decision'], ax9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training report saved to: {save_path}")
        
        plt.show()
        
    def _plot_predictions_scatter(self, y_pred, y_true, ax):
        """Plot predictions vs actual scatter plot"""
        ax.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        # Add regression line
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        ax.plot([min_val, max_val], p([min_val, max_val]), "g-", label='Regression Line')
        
        ax.set_xlabel('Actual Decision Value')
        ax.set_ylabel('Predicted Decision Value')
        ax.set_title('Predictions vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_residuals_distribution(self, y_pred, y_true, ax):
        """Plot residuals distribution"""
        residuals = y_pred - y_true
        
        ax.hist(residuals, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(residuals.mean(), color='red', linestyle='--', 
                   label=f'Mean: {residuals.mean():.4f}')
        ax.axvline(0, color='black', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('Residual (Predicted - Actual)')
        ax.set_ylabel('Frequency')
        ax.set_title('Residuals Distribution')
        ax.legend()
        
    def _plot_decision_distribution_regression(self, y_pred, y_true, ax):
        """Plot decision value distributions"""
        # Create violin plot
        data = pd.DataFrame({
            'Actual': y_true,
            'Predicted': y_pred
        })
        
        data_melted = data.melt(var_name='Type', value_name='Decision Value')
        sns.violinplot(data=data_melted, x='Type', y='Decision Value', ax=ax)
        
        ax.set_title('Decision Value Distributions')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.3, label='Buy')
        ax.axhline(y=-1, color='red', linestyle='--', alpha=0.3, label='Sell')
        ax.legend()
        
    def _plot_predictions_timeline(self, y_pred, y_true, ax):
        """Plot predictions timeline"""
        # Sample last 200 points for visibility
        sample_size = min(200, len(y_pred))
        
        ax.plot(y_true[-sample_size:].values, label='Actual', alpha=0.7, linewidth=2)
        ax.plot(y_pred[-sample_size:].values, label='Predicted', alpha=0.7, linewidth=2)
        
        # Add horizontal lines for reference
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.3)
        ax.axhline(y=-1, color='red', linestyle='--', alpha=0.3)
        
        ax.set_title('Predictions Timeline (Last 200 points)')
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Decision Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_error_timeline(self, y_pred, y_true, ax):
        """Plot error over time"""
        errors = np.abs(y_pred - y_true)
        
        # Moving average of errors
        window = min(50, len(errors) // 10)
        ma_errors = pd.Series(errors).rolling(window=window).mean()
        
        ax.plot(errors, alpha=0.3, label='Absolute Error')
        ax.plot(ma_errors, color='red', label=f'{window}-period MA')
        
        ax.set_title('Prediction Error Over Time')
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Absolute Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_performance_metrics_regression(self, predictions, true_labels, ax):
        """Plot performance metrics for regression"""
        # Calculate metrics
        y_pred = predictions['decision'].values
        y_true = true_labels['decision'].values
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Direction accuracy
        direction_acc = np.mean(np.sign(y_pred) == np.sign(y_true))
        
        # Zone accuracy (how well we predict buy/sell/neutral zones)
        def get_zone(val):
            if val > ModelConfig.BUY_THRESHOLD:
                return 1  # Buy zone
            elif val < ModelConfig.SELL_THRESHOLD:
                return -1  # Sell zone
            else:
                return 0  # Neutral zone
        
        pred_zones = [get_zone(v) for v in y_pred]
        true_zones = [get_zone(v) for v in y_true]
        zone_acc = np.mean(np.array(pred_zones) == np.array(true_zones))
        
        # Create metrics text
        metrics_text = f"""Regression Performance:
        
MSE: {mse:.4f}
MAE: {mae:.4f}
R²: {r2:.3f}

Direction Accuracy: {direction_acc:.3f}
Zone Accuracy: {zone_acc:.3f}

Total Samples: {len(predictions)}

Value Ranges:
Actual: [{y_true.min():.3f}, {y_true.max():.3f}]
Predicted: [{y_pred.min():.3f}, {y_pred.max():.3f}]
        """
        
        ax.text(0.1, 0.5, metrics_text, fontsize=12, 
                verticalalignment='center', 
                transform=ax.transAxes)
        ax.axis('off')
        ax.set_title('Performance Summary')
        
    def _plot_training_history(self, model, ax):
        """Plot training history"""
        if hasattr(model, 'training_history') and model.training_history['train_loss']:
            epochs = range(1, len(model.training_history['train_loss']) + 1)
            
            # Create twin axis for R2
            ax2 = ax.twinx()
            
            # Plot losses
            ax.plot(epochs, model.training_history['train_loss'], 'b-', label='Train Loss')
            ax.plot(epochs, model.training_history['val_loss'], 'r-', label='Val Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (MSE)')
            ax.legend(loc='upper left')
            
            # Plot R2 if available
            if 'val_r2' in model.training_history:
                ax2.plot(epochs, model.training_history['val_r2'], 'g--', label='Val R²')
                ax2.set_ylabel('R² Score')
                ax2.legend(loc='upper right')
            
            ax.set_title('Training History')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No training history available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Training History')
            ax.axis('off')
    
    def _plot_qq_plot(self, y_pred, y_true, ax):
        """Plot Q-Q plot to check normality of residuals"""
        residuals = y_pred - y_true
        residuals_sorted = np.sort(residuals)
        
        # Generate theoretical quantiles
        n = len(residuals)
        theoretical_quantiles = np.array([np.percentile(residuals, i/(n+1)*100) for i in range(1, n+1)])
        
        ax.scatter(theoretical_quantiles, residuals_sorted, alpha=0.5)
        
        # Add reference line
        min_val = min(theoretical_quantiles.min(), residuals_sorted.min())
        max_val = max(theoretical_quantiles.max(), residuals_sorted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')
        ax.set_title('Q-Q Plot of Residuals')
        ax.grid(True, alpha=0.3)
        
    def _plot_prediction_zones(self, y_pred, y_true, ax):
        """Analyze predictions by zones (buy/neutral/sell)"""
        def get_zone_label(val):
            if val > 0.33:
                return 'Buy'
            elif val < -0.33:
                return 'Sell'
            else:
                return 'Neutral'
        
        # Create zone labels
        pred_zones = [get_zone_label(v) for v in y_pred]
        true_zones = [get_zone_label(v) for v in y_true]
        
        # Count zone predictions
        zone_matrix = pd.crosstab(
            pd.Series(true_zones, name='Actual'),
            pd.Series(pred_zones, name='Predicted')
        )
        
        # Plot heatmap
        sns.heatmap(zone_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Zone Prediction Matrix')




def find_latest_model():
    """Find latest model in models directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_learning_dir = os.path.dirname(script_dir)
    model_dir = os.path.join(test_learning_dir, 'models')
    
    if not os.path.exists(model_dir):
        return None
        
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        return None
        
    # Sort by modification time
    latest_model = max(model_files, key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))
    return os.path.join(model_dir, latest_model)


def ask_training_mode():
    """Ask user for training mode"""
    print("\n🤔 Select training mode:")
    print("1. Train from scratch (new model)")
    print("2. Continue training from latest model")
    print("3. Continue training from specific model")
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            return int(choice)
        print("❌ Invalid choice. Please enter 1, 2, or 3.")


def load_existing_model(model_path=None):
    """Load existing model for continued training"""
    if model_path is None:
        model_path = find_latest_model()
        
    if model_path is None or not os.path.exists(model_path):
        print("❌ No model found!")
        return None
        
    print(f"📂 Loading model from: {model_path}")
    
    # Initialize model and load state
    model = TradingModel()
    model.load_model(model_path)
    
    print("✅ Model loaded successfully!")
    return model


def load_and_prepare_data(symbol=None, limit=None):
    """Load data dan prepare features & labels using config"""
    symbol = symbol or TrainingConfig.SYMBOL
    limit = limit or TrainingConfig.DATA_LIMIT
    
    print(f"📊 Loading data for {symbol.upper()}...")
    
    # 1. Load multi-timeframe data
    loader = DataLoader()
    multi_data = loader.load_data(
        symbol=symbol,
        timeframes=['1m', '5m', '30m', '1h'],
        limit=limit,
        auto_align=True,
        alignment_mode='current_only'
    )
    
    # 2. Create LSTM features
    print("\n🔧 Creating LSTM features...")
    features_df = create_lstm_features(multi_data)
    
    # IMPORTANT: Round features timestamp to minute
    features_df.index = features_df.index.round('1min')
    # Remove timezone if exists
    if features_df.index.tz is not None:
        features_df.index = features_df.index.tz_localize(None)
    print(f"   Features timestamp rounded to minute (no timezone)")
    
    # 3. Load labels from CSV
    print("\n📋 Loading labels from CSV...")
    
    # Find label file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    label_dir = os.path.join(script_dir, '..', 'database_learning')
    label_files = [f for f in os.listdir(label_dir) if f.startswith('label_1m_') and f.endswith('.csv')]
    
    if not label_files:
        raise FileNotFoundError("No label files found in database_learning folder")
    
    # Use most recent label file
    label_file = sorted(label_files)[-1]
    label_path = os.path.join(label_dir, label_file)
    print(f"   Using label file: {label_file}")
    
    # Load labels with automatic timestamp parsing
    print(f"   Loading {label_path}...")
    
    # First, peek at the file to see timestamp format
    try:
        with open(label_path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig handles BOM
            header = f.readline().strip()
            first_row = f.readline().strip()
            print(f"   CSV Header: {header}")
            print(f"   First row: {first_row}")
    except Exception as e:
        print(f"   Warning: Could not peek at file: {e}")
    
    # Load with automatic parsing
    try:
        labels_df = pd.read_csv(label_path, index_col='timestamp', parse_dates=True, encoding='utf-8-sig')
    except Exception as e:
        print(f"   Failed to parse timestamps automatically: {e}")
        print("   Trying manual parsing...")
        
        # Manual approach
        labels_df = pd.read_csv(label_path, encoding='utf-8-sig')
        
        # Show sample of timestamp column
        print(f"   Sample timestamps: {labels_df['timestamp'].head(3).tolist()}")
        
        # Try to parse with pandas' smart parser
        labels_df['timestamp'] = pd.to_datetime(labels_df['timestamp'])
        labels_df.set_index('timestamp', inplace=True)
    
    # IMPORTANT: Round labels timestamp to minute
    labels_df.index = labels_df.index.round('1min')
    # Remove timezone
    labels_df.index = labels_df.index.tz_localize(None)
    print(f"   Labels timestamp rounded to minute (no timezone)")
    
    # Add dummy SL column for compatibility
    if 'SL' not in labels_df.columns:
        labels_df['SL'] = 0.0
    
    print(f"   Labels loaded: {len(labels_df)} rows")
    print(f"   Labels date range: {labels_df.index.min()} to {labels_df.index.max()}")
    print(f"   Decision value range: [{labels_df['decision'].min():.3f}, {labels_df['decision'].max():.3f}]")
    
    # 4. Debug info before alignment
    print("\n🔍 Debug Info Before Alignment:")
    print(f"   Features shape: {features_df.shape}")
    print(f"   Features date range: {features_df.index.min()} to {features_df.index.max()}")
    print(f"   Features sample timestamps: {features_df.index[:3].tolist()}")
    print(f"   Labels shape: {labels_df.shape}")
    print(f"   Labels sample timestamps: {labels_df.index[:3].tolist()}")
    
    # 5. Align features and labels
    print("\n🔄 Aligning features and labels...")
    
    # Get common timestamps
    common_index = features_df.index.intersection(labels_df.index)
    print(f"   Common timestamps found: {len(common_index)}")
    
    if len(common_index) == 0:
        print("\n❌ No common timestamps found!")
        print("   Attempting alternative alignment method...")
        
        # Alternative: Convert to string and back to ensure same format
        features_df.index = pd.to_datetime(
            features_df.index.strftime('%Y-%m-%d %H:%M:00')
        )
        labels_df.index = pd.to_datetime(
            labels_df.index.strftime('%Y-%m-%d %H:%M:00')
        )
        
        # Try again
        common_index = features_df.index.intersection(labels_df.index)
        print(f"   After reformatting: {len(common_index)} common timestamps")
        
        if len(common_index) == 0:
            # Last resort: Check if there's a time offset
            time_diff = features_df.index[0] - labels_df.index[0]
            print(f"   Time difference between datasets: {time_diff}")
            raise ValueError("No common timestamps between features and labels even after alignment attempts!")
    
    # Align data using common timestamps
    features_aligned = features_df.loc[common_index]
    labels_aligned = labels_df.loc[common_index][['decision', 'SL']]
    
    # 6. Final validation
    print(f"\n✅ Final aligned dataset:")
    print(f"   Total samples: {len(features_aligned)}")
    print(f"   Date range: {features_aligned.index.min()} to {features_aligned.index.max()}")
    print(f"   Decision value distribution:")
    print(f"     Strong Buy (>0.8): {(labels_aligned['decision'] > 0.8).sum()}")
    print(f"     Buy (0.3 to 0.8): {((labels_aligned['decision'] > 0.3) & (labels_aligned['decision'] <= 0.8)).sum()}")
    print(f"     Neutral (-0.3 to 0.3): {((labels_aligned['decision'] >= -0.3) & (labels_aligned['decision'] <= 0.3)).sum()}")
    print(f"     Sell (-0.8 to -0.3): {((labels_aligned['decision'] >= -0.8) & (labels_aligned['decision'] < -0.3)).sum()}")
    print(f"     Strong Sell (<-0.8): {(labels_aligned['decision'] < -0.8).sum()}")
    
    # Check for any remaining issues
    if len(features_aligned) == 0:
        raise ValueError("Aligned dataset is empty!")
    
    if features_aligned.shape[0] != labels_aligned.shape[0]:
        raise ValueError(f"Features and labels have different lengths after alignment: {features_aligned.shape[0]} vs {labels_aligned.shape[0]}")
    
    return features_aligned, labels_aligned


def train_model(features_df, labels_df, model=None, model_name=None):
    """Enhanced train model using training config"""
    model_name = model_name or TrainingConfig.MODEL_NAME

    
    
    print("\n🚀 Initializing LSTM Model...")
    
    # Print training configuration
    TrainingConfig.print_config()
    
    if model is None:
        print(f"   Creating new model...")
        
        if TrainingConfig.USE_LEGACY_PARAMS:
            # Legacy mode - override model config
            print("   🔴 Using LEGACY parameters")
            model = TradingModel(
                seq_len=TrainingConfig.LEGACY_SEQ_LEN,
                hidden_size=TrainingConfig.LEGACY_HIDDEN_SIZE,
                lr=TrainingConfig.LEGACY_LR
            )
        else:
            # Modern mode - use model config defaults with optional overrides
            print("   🟢 Using MODEL CONFIG defaults")
            model = TradingModel(
                seq_len=TrainingConfig.SEQ_LEN,
                hidden_size=TrainingConfig.HIDDEN_SIZE,
                lr=TrainingConfig.LEARNING_RATE
            )
    else:
        print("   Using existing model for continued training")
        
    print(f"🔍 Quick check - Seq len: {model.seq_len}, Features: {features_df.shape}, Labels: {labels_df.shape}")
    print(f"🔍 Label range: [{labels_df['decision'].min():.3f}, {labels_df['decision'].max():.3f}]")
    
    # Prepare datasets with config
    train_dataset, val_dataset = model.prepare_data(
        features_df, 
        labels_df,
        test_size=TrainingConfig.TEST_SIZE
    )
    
    # Train model with config
    if TrainingConfig.USE_LEGACY_PARAMS:
        # Legacy training parameters
        model.train(
            train_dataset,
            val_dataset,
            epochs=TrainingConfig.LEGACY_EPOCHS,
            batch_size=TrainingConfig.LEGACY_BATCH_SIZE
        )
    else:
        # Modern training - use model config defaults with optional overrides
        model.train(
            train_dataset,
            val_dataset,
            epochs=TrainingConfig.EPOCHS,
            batch_size=TrainingConfig.BATCH_SIZE
        )
    
    print("\n✅ Training completed")

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))  # strategy/
    test_learning_dir = os.path.dirname(script_dir)  # test_learning_lstm/
    models_dir = os.path.join(test_learning_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, f"{model_name}_{timestamp}.pth")
    
    # Save model
    model.save_model(model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    return model, model_path


def evaluate_model(model, features_df, labels_df):
    """Evaluate model performance"""
    
    print("\n📊 Evaluating model...")
    
    # Make predictions
    predictions = model.predict(features_df)
    
    # Align with true labels (accounting for sequence length)
    seq_len = model.seq_len  # CHANGED from model.sequence_length
    true_labels = labels_df.iloc[seq_len:]  # Skip sequence length completely
    predictions = predictions.iloc[:-1] 
    
    # IMPORTANT: Reset index to avoid comparison issues
    predictions = predictions.reset_index(drop=True)
    true_labels = true_labels.reset_index(drop=True)
    
    # Debug info
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   True labels shape: {true_labels.shape}")
    
    # Validate shapes match
    if len(predictions) != len(true_labels):
        print(f"   ⚠️ Length mismatch: predictions={len(predictions)}, labels={len(true_labels)}")
        # Trim to minimum length
        min_len = min(len(predictions), len(true_labels))
        predictions = predictions.iloc[:min_len]
        true_labels = true_labels.iloc[:min_len]
        print(f"   Trimmed to: {min_len} samples")
    
    # Calculate regression metrics
    y_pred = predictions['decision'].values
    y_true = true_labels['decision'].values
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Direction accuracy
    direction_acc = np.mean(np.sign(y_pred) == np.sign(y_true))
    
    print(f"\n📈 Performance Metrics:")
    print(f"   MSE: {mse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   R²: {r2:.3f}")
    print(f"   Direction Accuracy: {direction_acc:.3f}")
    
    # Value range analysis
    print(f"\n   Value Ranges:")
    print(f"   Actual: [{y_true.min():.3f}, {y_true.max():.3f}]")
    print(f"   Predicted: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
    
    # Zone analysis
    def get_zone(val):
        if val > 0.33:
            return 'Buy'
        elif val < -0.33:
            return 'Sell'
        else:
            return 'Neutral'
    
    pred_zones = [get_zone(v) for v in y_pred]
    true_zones = [get_zone(v) for v in y_true]
    
    zone_accuracy = np.mean(np.array(pred_zones) == np.array(true_zones))
    print(f"   Zone Accuracy: {zone_accuracy:.3f}")
    
    print(f"\n   Zone Distribution:")
    print(f"   Actual - Buy: {true_zones.count('Buy')}, Neutral: {true_zones.count('Neutral')}, Sell: {true_zones.count('Sell')}")
    print(f"   Predicted - Buy: {pred_zones.count('Buy')}, Neutral: {pred_zones.count('Neutral')}, Sell: {pred_zones.count('Sell')}")

    if hasattr(model, 'evaluate_trading_performance'):
        print("\n📊 Evaluating trading performance...")
        trading_metrics = model.evaluate_trading_performance(predictions, true_labels)
    
    return predictions, true_labels


def ask_continue_training():
    """Ask if user wants to continue training"""
    print("\n❓ Do you want to continue training with current model? (y/n)")
    choice = input().strip().lower()
    return choice == 'y'


def main():
    """Main training pipeline with centralized configuration"""
    
    print("🎯 LSTM Trading Model Training (Enhanced with Config)")
    print("=" * 60)
    
    try:
        # Ask training mode
        mode = ask_training_mode()
        
        model = None
        
        if mode == 2:  # Continue from latest
            model = load_existing_model()
            if model is None:
                print("⚠️ No model found. Starting from scratch...")
                model = None
                
        elif mode == 3:  # Continue from specific
            # Check if models directory exists
            script_dir = os.path.dirname(os.path.abspath(__file__))
            test_learning_dir = os.path.dirname(script_dir)
            models_dir = os.path.join(test_learning_dir, 'models')
            if not os.path.exists(models_dir):
                print(f"❌ Models directory not found: {models_dir}")
                print("⚠️ Starting from scratch...")
                model = None
            else:
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
                if model_files:
                    print("\nAvailable models:")
                    for i, mf in enumerate(model_files):
                        print(f"{i+1}. {mf}")
                    
                    try:
                        choice = int(input("Select model number: ")) - 1
                        if 0 <= choice < len(model_files):
                            selected_file = model_files[choice]
                            model_path = os.path.join(models_dir, selected_file)
                            model = load_existing_model(model_path)
                        else:
                            print("❌ Invalid selection. Starting from scratch...")
                            model = None
                    except ValueError:
                        print("❌ Invalid input. Starting from scratch...")
                        model = None
                else:
                    print("⚠️ No models found. Starting from scratch...")
                    model = None
        
        # Training loop
        while True:
            # 1. Load and prepare data using config
            features, labels = load_and_prepare_data()
            
            # 2. Train model using config
            model, model_path = train_model(features, labels, model)
            
            # 3. Evaluate model
            predictions, true_labels = evaluate_model(model, features, labels)
            
            # 4. Save predictions with error handling
            print("\n💾 Saving predictions...")
            try:
                predictions.to_csv('predictions.csv')
                print("   Predictions saved to: predictions.csv")
            except PermissionError:
                # Use timestamp if original fails
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                fallback_path = f'predictions_{timestamp}.csv'
                predictions.to_csv(fallback_path)
                print(f"   Predictions saved to: {fallback_path}")
                print("   💡 Original file was locked, used timestamped filename")
            except Exception as e:
                print(f"   ⚠️ Could not save predictions: {e}")
                print("   Continuing without saving predictions...")
            
            # 5. Create visualization report
            print("\n📊 Creating comprehensive training report...")
            try:
                visualizer = TrainingVisualizer()
                report_path = f'training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                visualizer.create_training_report(
                    model, predictions, true_labels,
                    save_path=report_path
                )
                print(f"   Report saved to: {report_path}")
                
            except Exception as e:
                print(f"   ⚠️ Could not create training report: {e}")
                print("   Continuing without visualization...")
            
            # 6. Ask if continue training
            if not ask_continue_training():
                break
            
            print("\n🔄 Continuing training with current model...")
        
        print("\n✅ Training pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ==========================================
# 🎛️ QUICK TRAINING PRESETS
# ==========================================

class TrainingPresets:
    """Predefined training configurations for different scenarios"""
    
    @staticmethod
    def legacy_mode():
        """Original training settings"""
        TrainingConfig.USE_LEGACY_PARAMS = True
        TrainingConfig.LEGACY_SEQ_LEN = 120
        TrainingConfig.LEGACY_HIDDEN_SIZE = 64
        TrainingConfig.LEGACY_LR = 0.00001
        TrainingConfig.LEGACY_EPOCHS = 200
        TrainingConfig.LEGACY_BATCH_SIZE = 16
    
    @staticmethod
    def modern_mode():
        """Use enhanced model config defaults"""
        TrainingConfig.USE_LEGACY_PARAMS = False
        # All parameters will use ModelConfig defaults from model.py
    
    @staticmethod
    def fast_training():
        """Quick training for testing"""
        TrainingConfig.USE_LEGACY_PARAMS = False
        TrainingConfig.EPOCHS = 50
        TrainingConfig.BATCH_SIZE = 128
        TrainingConfig.DATA_LIMIT = 5000
    
    @staticmethod
    def intensive_training():
        """Intensive training for best results"""
        TrainingConfig.USE_LEGACY_PARAMS = False
        TrainingConfig.EPOCHS = 300
        TrainingConfig.BATCH_SIZE = 32
        TrainingConfig.DATA_LIMIT = 20000


if __name__ == "__main__":
    # Quick setup examples:
    # TrainingPresets.legacy_mode()      # Use original settings
    # TrainingPresets.modern_mode()      # Use enhanced settings (default)
    # TrainingPresets.fast_training()    # Quick testing
    # TrainingPresets.intensive_training()  # Best results
    
    # Or manual configuration:
    # TrainingConfig.DATA_LIMIT = 15000
    # TrainingConfig.SYMBOL = 'eth'
    
    main()