import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
import joblib
import logging

from data_preprocessing import ECUDataPreprocessor
from lstm_models import LSTMDriverClassifier, TemporalAggregator, TuningMapRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriverStyleClassificationPipeline:
    def __init__(self):
        self.preprocessor = ECUDataPreprocessor()
        self.stage1_model = None
        self.stage2_model = None
        self.aggregator = TemporalAggregator()
        self.recommender = TuningMapRecommender()
        self.label_encoder = LabelEncoder()

    def load_and_prepare_data(self, data_path):
        """Load and prepare ECU data"""
        logger.info(f"Loading data from {data_path}")

        # Load data
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV.")

        # Preprocess data
        features, patterns, processed_df = self.preprocessor.fit_transform(df)

        return features, patterns, processed_df

    def create_synthetic_labels(self, processed_df):
        """Create labels for training (in real scenario, these would come from expert labeling)"""
        logger.info("Creating synthetic labels for training...")

        # Stage 1 labels: Event classification (Accelerating, Braking, Cruising)
        event_labels = []
        for _, row in processed_df.iterrows():
            if row['Event'] == 'Accelerating':
                event_labels.append(0)
            elif row['Event'] == 'Braking':
                event_labels.append(1)
            else:
                event_labels.append(2)  # Cruising

        # Stage 2 labels: Driving style classification
        style_labels = []
        for _, row in processed_df.iterrows():
            rpm_std = row['RPM_std']
            fuel_mean = row['BaseFuel_mean']

            if rpm_std > 50 and fuel_mean > 360:
                style_labels.append(0)  # Aggressive
            elif rpm_std < 30 and fuel_mean < 350:
                style_labels.append(1)  # Eco
            else:
                style_labels.append(2)  # Balanced

        return np.array(event_labels), np.array(style_labels)

    def train_stage1_model(self, features, event_labels, sequence_length=10):
        """Train Stage 1: Low-Level Driving Event Classification"""
        logger.info("Training Stage 1 model...")

        # Initialize model
        self.stage1_model = LSTMDriverClassifier(
            sequence_length=sequence_length,
            n_features=features.shape[1]
        )

        # Build model
        self.stage1_model.build_stage1_model()

        # Create sequences
        X_seq, y_seq = self.stage1_model.create_sequences(features, event_labels)

        # Convert labels to categorical
        y_categorical = to_categorical(y_seq, num_classes=3)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_seq, y_categorical, test_size=0.2, random_state=42
        )

        # Train model
        history = self.stage1_model.train_model(X_train, y_train, X_val, y_val)

        # Save model
        self.stage1_model.save_model('models/stage1_model.h5')

        return history

    def train_stage2_model(self, features, style_labels, sequence_length=10):
        """Train Stage 2: High-Level Driving Style Classification"""
        logger.info("Training Stage 2 model...")

        # Initialize model
        self.stage2_model = LSTMDriverClassifier(
            sequence_length=sequence_length,
            n_features=features.shape[1]
        )

        # Build model
        self.stage2_model.build_stage2_model(n_driving_styles=3)

        # Create sequences
        X_seq, y_seq = self.stage2_model.create_sequences(features, style_labels)

        # Convert labels to categorical
        y_categorical = to_categorical(y_seq, num_classes=3)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_seq, y_categorical, test_size=0.2, random_state=42
        )

        # Train model
        history = self.stage2_model.train_model(X_train, y_train, X_val, y_val)

        # Save model
        self.stage2_model.save_model('models/stage2_model.h5')

        return history

    def full_pipeline_training(self, data_path):
        """Run complete training pipeline"""
        logger.info("Starting full pipeline training...")

        # Load and prepare data
        features, patterns, processed_df = self.load_and_prepare_data(data_path)

        # Create labels
        event_labels, style_labels = self.create_synthetic_labels(processed_df)

        # Train Stage 1 model
        stage1_history = self.train_stage1_model(features, event_labels)

        # Train Stage 2 model
        stage2_history = self.train_stage2_model(features, style_labels)

        # Save preprocessor
        joblib.dump(self.preprocessor, 'preprocessor.pkl')

        logger.info("Training completed successfully!")

        return {
            'stage1_history': stage1_history,
            'stage2_history': stage2_history,
            'patterns': patterns
        }

def main():
    """Main training function"""
    # Initialize pipeline
    pipeline = DriverStyleClassificationPipeline()

    # Run training (using actual labeled data)
    results = pipeline.full_pipeline_training('data/labeled_20180713-home2mimos.csv')

    print("Training completed!")
    print(f"Detected driving patterns: {results['patterns']}")

if __name__ == "__main__":
    main()
