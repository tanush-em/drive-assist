
# lstm_models.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMDriverClassifier:
    def __init__(self, sequence_length=10, n_features=10):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None

    def create_sequences(self, data, targets=None, sequence_length=None):
        """Create sequences for LSTM training"""
        if sequence_length is None:
            sequence_length = self.sequence_length

        sequences = []
        labels = []

        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
            if targets is not None:
                labels.append(targets[i + sequence_length - 1])

        if targets is not None:
            return np.array(sequences), np.array(labels)
        return np.array(sequences)

    def build_stage1_model(self):
        """Build Stage 1 - Low-Level Driving Event Classification LSTM"""
        logger.info("Building Stage 1 LSTM model...")

        model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),
            LSTM(64, return_sequences=True, dropout=0.2),
            LSTM(32, dropout=0.2),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(3, activation='softmax')  # Accelerating, Braking, Cruising
        ])

        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )


        self.model = model
        return model

    def build_stage2_model(self, n_driving_styles=3):
        """Build Stage 2 - High-Level Driving Style Classification LSTM"""
        logger.info("Building Stage 2 LSTM model...")

        model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),
            LSTM(128, return_sequences=True, dropout=0.2),
            LSTM(64, return_sequences=True, dropout=0.2),
            LSTM(32, dropout=0.2),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(n_driving_styles, activation='softmax')  # Eco, Aggressive, Balanced
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the LSTM model"""
        logger.info("Training LSTM model...")

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        model_checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True
        )

        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )

        return self.history

    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        predictions = self.model.predict(X)
        return predictions

    def save_model(self, filepath):
        """Save trained model"""
        if self.model is not None:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model"""
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")

class TemporalAggregator:
    """Aggregate Stage 1 events into larger intervals"""

    def __init__(self, interval_size=30):
        self.interval_size = interval_size  # seconds

    def aggregate_events(self, events, timestamps):
        """Group events into larger time intervals"""
        logger.info(f"Aggregating events into {self.interval_size}s intervals...")

        # Convert to DataFrame for easier processing
        df = pd.DataFrame({
            'event': events,
            'timestamp': timestamps
        })

        # Create time intervals
        df['interval'] = (df['timestamp'] // self.interval_size).astype(int)

        # Aggregate events per interval
        aggregated = df.groupby('interval').agg({
            'event': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Cruising'
        }).reset_index()

        # Calculate additional features
        event_counts = df.groupby('interval')['event'].value_counts().unstack(fill_value=0)
        aggregated = aggregated.merge(event_counts, on='interval', how='left')

        return aggregated

class TuningMapRecommender:
    """Stage 3 - Generate ECU tuning recommendations"""

    def __init__(self):
        self.tuning_profiles = {
            'Eco': {
                'fuel_efficiency_boost': 15,
                'ignition_timing_adj': -2,
                'throttle_response': 'smooth',
                'recommendations': [
                    'Optimize fuel injection timing for efficiency',
                    'Reduce aggressive throttle mapping',
                    'Enable eco-mode features'
                ]
            },
            'Aggressive': {
                'performance_boost': 10,
                'ignition_timing_adj': 3,
                'throttle_response': 'sharp',
                'recommendations': [
                    'Advance ignition timing for performance',
                    'Increase throttle sensitivity',
                    'Optimize turbo boost pressure'
                ]
            },
            'Balanced': {
                'balance_factor': 1.0,
                'ignition_timing_adj': 0,
                'throttle_response': 'moderate',
                'recommendations': [
                    'Maintain balanced performance-efficiency tune',
                    'Moderate throttle mapping',
                    'Standard ignition timing'
                ]
            }
        }

    def generate_recommendations(self, driving_style, current_params):
        """Generate tuning recommendations based on driving style"""
        logger.info(f"Generating tuning recommendations for {driving_style} driver...")

        profile = self.tuning_profiles.get(driving_style, self.tuning_profiles['Balanced'])

        recommendations = {
            'driving_style': driving_style,
            'tuning_profile': profile,
            'current_parameters': current_params,
            'suggested_changes': {}
        }

        # Calculate suggested parameter changes
        if driving_style == 'Eco':
            recommendations['suggested_changes'] = {
                'base_fuel_adjustment': -5,
                'ignition_timing_adjustment': profile['ignition_timing_adj'],
                'throttle_map': 'eco_optimized'
            }
        elif driving_style == 'Aggressive':
            recommendations['suggested_changes'] = {
                'base_fuel_adjustment': 3,
                'ignition_timing_adjustment': profile['ignition_timing_adj'],
                'throttle_map': 'performance_optimized'
            }
        else:
            recommendations['suggested_changes'] = {
                'base_fuel_adjustment': 0,
                'ignition_timing_adjustment': 0,
                'throttle_map': 'standard'
            }

        return recommendations
