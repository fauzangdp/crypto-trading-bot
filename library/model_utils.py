# lib/model_utils.py

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
import os
import json
from datetime import datetime

class ModelBuilder:
    """Build LSTM models with various configurations"""
    
    @staticmethod
    def lstm_layer(units, return_sequences=False, dropout=0.2):
        """
        Create LSTM layer with dropout
        """
        def layer(x):
            x = layers.LSTM(units, return_sequences=return_sequences)(x)
            if dropout > 0:
                x = layers.Dropout(dropout)(x)
            return x
        return layer
    
    @staticmethod
    def attention_layer(name='attention'):
        """
        Simple attention mechanism
        """
        def layer(inputs):
            # Calculate attention scores
            scores = layers.Dense(1, activation='tanh', name=f'{name}_scores')(inputs)
            weights = layers.Softmax(axis=1, name=f'{name}_weights')(scores)
            
            # Apply attention
            context = layers.Multiply(name=f'{name}_context')([inputs, weights])
            attended = layers.GlobalAveragePooling1D(name=f'{name}_sum')(context)
            
            return attended
        return layer
    
    @staticmethod
    def build_lstm_model(input_shape, output_size=8, use_attention=True):
        """
        Build complete LSTM model
        
        Args:
            input_shape: (sequence_length, features) e.g. (30, 32)
            output_size: number of outputs (default 8)
            use_attention: whether to use attention mechanism
        """
        inputs = layers.Input(shape=input_shape)
        
        # First LSTM
        x = ModelBuilder.lstm_layer(64, return_sequences=True)(inputs)
        
        # Second LSTM
        x = ModelBuilder.lstm_layer(32, return_sequences=use_attention)(x)
        
        # Attention (optional)
        if use_attention:
            x = ModelBuilder.attention_layer()(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output
        outputs = layers.Dense(output_size)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

class CustomLosses:
    """Custom loss functions for trading"""
    
    @staticmethod
    def directional_loss(y_true, y_pred):
        """
        Focus on getting direction right
        """
        # Extract direction (first output)
        direction_true = y_true[:, 0]
        direction_pred = y_pred[:, 0]
        
        # Direction loss (MSE)
        direction_loss = tf.keras.losses.mse(direction_true, direction_pred)
        
        # Other outputs loss
        other_loss = tf.keras.losses.mse(y_true[:, 1:], y_pred[:, 1:])
        
        # Weight direction more
        return direction_loss * 2.0 + other_loss
    
    @staticmethod
    def weighted_loss(weights=[0.3, 0.2, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05]):
        """
        Different weights for different outputs
        """
        def loss(y_true, y_pred):
            total_loss = 0
            for i, weight in enumerate(weights):
                total_loss += weight * tf.keras.losses.mse(y_true[:, i], y_pred[:, i])
            return total_loss
        return loss
    
    @staticmethod
    def profit_aware_loss(y_true, y_pred):
        """
        Penalize wrong direction more when confidence is high
        """
        direction_true = y_true[:, 0]
        direction_pred = y_pred[:, 0]
        confidence_pred = tf.nn.sigmoid(y_pred[:, 1])  # Ensure 0-1
        
        # Direction error
        direction_error = tf.abs(direction_true - direction_pred)
        
        # Penalize more when confident but wrong
        weighted_error = direction_error * (1 + confidence_pred)
        
        # Add other losses
        other_loss = tf.keras.losses.mse(y_true[:, 2:], y_pred[:, 2:])
        
        return tf.reduce_mean(weighted_error) + other_loss

class ModelManager:
    """Handle model saving, loading, and versioning"""
    
    def __init__(self, base_path='models'):
        """
        Initialize model manager
        
        Args:
            base_path: directory to save models
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def get_model_path(self, name='model'):
        """Generate unique model path with timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(self.base_path, f'{name}_{timestamp}')
    
    def save_model(self, model, metrics=None, name='model'):
        """
        Save model with metadata
        
        Args:
            model: Keras model
            metrics: dict of performance metrics
            name: model name
        """
        model_path = self.get_model_path(name)
        os.makedirs(model_path, exist_ok=True)
        
        # Save model
        model.save(os.path.join(model_path, 'model.h5'))
        
        # Save config
        config = {
            'timestamp': datetime.now().isoformat(),
            'input_shape': model.input_shape[1:],
            'output_shape': model.output_shape[1:],
            'total_params': model.count_params(),
            'metrics': metrics or {}
        }
        
        with open(os.path.join(model_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to: {model_path}")
        return model_path
    
    def load_model(self, model_path):
        """Load model and config"""
        # Load model
        model = tf.keras.models.load_model(
            os.path.join(model_path, 'model.h5'),
            compile=False
        )
        
        # Load config
        with open(os.path.join(model_path, 'config.json'), 'r') as f:
            config = json.load(f)
        
        return model, config
    
    def autosave_callback(self, name='model', monitor='val_loss'):
        """
        Create autosave callback for training
        
        Args:
            name: model name prefix
            monitor: metric to monitor
        """
        filepath = os.path.join(
            self.base_path, 
            f'{name}_best_{{epoch:02d}}_{{val_loss:.4f}}.h5'
        )
        
        return callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )

# Convenience functions
def create_callbacks(model_name='model', patience=10):
    """Create standard callbacks for training"""
    mm = ModelManager()
    
    callback_list = [
        # Auto-save best model
        mm.autosave_callback(model_name),
        
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience//2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    return callback_list


# Usage example
if __name__ == "__main__": 
    # Build model
    model = ModelBuilder.build_lstm_model(
        input_shape=(30, 32),
        output_size=8,
        use_attention=True
    )
    
    # Compile with custom loss
    model.compile(
        optimizer='adam',
        loss=CustomLosses.directional_loss,
        metrics=['mae']
    )
    
    # Model summary
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks('test_model')
    
    # Save model
    mm = ModelManager()
    mm.save_model(model, metrics={'test_accuracy': 0.85})