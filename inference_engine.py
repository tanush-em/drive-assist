
# inference_engine.py
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from datetime import datetime
import logging

from data_preprocessing import ECUDataPreprocessor
from lstm_models import LSTMDriverClassifier, TuningMapRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriverStyleInferenceEngine:
    def __init__(self):
        self.stage1_model = None
        self.stage2_model = None
        self.preprocessor = None
        self.recommender = TuningMapRecommender()
        self.sequence_buffer = []
        self.sequence_length = 10

    def load_models(self, stage1_path='stage1_model.h5', stage2_path='stage2_model.h5', preprocessor_path='preprocessor.pkl'):
        """Load trained models and preprocessor"""
        logger.info("Loading trained models...")

        try:
            # Load models
            self.stage1_model = tf.keras.models.load_model(stage1_path)
            self.stage2_model = tf.keras.models.load_model(stage2_path)

            # Load preprocessor
            self.preprocessor = joblib.load(preprocessor_path)

            logger.info("Models loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    def preprocess_realtime_data(self, ecu_data):
        """Preprocess real-time ECU data"""
        try:
            # Convert to DataFrame if it's a dict
            if isinstance(ecu_data, dict):
                df = pd.DataFrame([ecu_data])
            else:
                df = ecu_data.copy()

            # Basic preprocessing
            df = self.preprocessor.remove_faulty_readings(df)
            df = self.preprocessor.synchronize_timestamps(df)

            # Extract features
            features = df[['RPM', 'Load', 'BaseFuel', 'BaseIgnitionTiming', 
                          'LambdaSensor', 'BatteryVolt', 'MAPSource']].values

            # Handle missing values
            features = np.nan_to_num(features, nan=0.0)

            # Scale features
            if hasattr(self.preprocessor, 'scaler'):
                features = self.preprocessor.scaler.transform(features)

            return features

        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return None

    def predict_driving_event(self, features):
        """Predict driving event using Stage 1 model"""
        if self.stage1_model is None:
            logger.error("Stage 1 model not loaded")
            return None

        try:
            # Ensure we have a sequence
            if len(features.shape) == 1:
                features = features.reshape(1, -1)

            # Add to sequence buffer
            self.sequence_buffer.append(features[0])

            # Keep only the last sequence_length items
            if len(self.sequence_buffer) > self.sequence_length:
                self.sequence_buffer.pop(0)

            # Make prediction only if we have enough data
            if len(self.sequence_buffer) == self.sequence_length:
                sequence = np.array(self.sequence_buffer).reshape(1, self.sequence_length, -1)
                prediction = self.stage1_model.predict(sequence, verbose=0)

                # Convert prediction to class
                predicted_class = np.argmax(prediction[0])
                confidence = np.max(prediction[0])

                event_names = ['Accelerating', 'Braking', 'Cruising']

                return {
                    'event': event_names[predicted_class],
                    'confidence': float(confidence),
                    'probabilities': prediction[0].tolist()
                }

        except Exception as e:
            logger.error(f"Error in event prediction: {e}")
            return None

    def predict_driving_style(self, features):
        """Predict driving style using Stage 2 model"""
        if self.stage2_model is None:
            logger.error("Stage 2 model not loaded")
            return None

        try:
            # Ensure we have a sequence
            if len(features.shape) == 1:
                features = features.reshape(1, -1)

            # Use the same sequence buffer
            if len(self.sequence_buffer) == self.sequence_length:
                sequence = np.array(self.sequence_buffer).reshape(1, self.sequence_length, -1)
                prediction = self.stage2_model.predict(sequence, verbose=0)

                # Convert prediction to class
                predicted_class = np.argmax(prediction[0])
                confidence = np.max(prediction[0])

                style_names = ['Aggressive', 'Eco', 'Balanced']

                return {
                    'style': style_names[predicted_class],
                    'confidence': float(confidence),
                    'probabilities': prediction[0].tolist()
                }

        except Exception as e:
            logger.error(f"Error in style prediction: {e}")
            return None

    def generate_tuning_recommendations(self, driving_style, current_ecu_params):
        """Generate ECU tuning recommendations"""
        return self.recommender.generate_recommendations(driving_style, current_ecu_params)

    def process_realtime_data(self, ecu_data):
        """Complete real-time processing pipeline"""
        logger.info("Processing real-time ECU data...")

        # Preprocess data
        features = self.preprocess_realtime_data(ecu_data)
        if features is None:
            return None

        # Predict driving event
        event_prediction = self.predict_driving_event(features)

        # Predict driving style
        style_prediction = self.predict_driving_style(features)

        # Generate recommendations if style is predicted
        recommendations = None
        if style_prediction:
            current_params = {
                'rpm': ecu_data.get('RPM', 0),
                'load': ecu_data.get('Load', 0),
                'fuel': ecu_data.get('BaseFuel', 0),
                'ignition_timing': ecu_data.get('BaseIgnitionTiming', 0)
            }
            recommendations = self.generate_tuning_recommendations(
                style_prediction['style'], current_params
            )

        # Compile results
        result = {
            'timestamp': datetime.now().isoformat(),
            'event_prediction': event_prediction,
            'style_prediction': style_prediction,
            'tuning_recommendations': recommendations,
            'input_data': ecu_data
        }

        return result

# Example usage
def main():
    """Example of using the inference engine"""
    # Initialize inference engine
    engine = DriverStyleInferenceEngine()

    # Load models
    models_loaded = engine.load_models()
    if not models_loaded:
        print("Could not load models. Please train models first.")
        return

    # Simulate real-time ECU data
    sample_ecu_data = {
        'Timestamp': '54:58.4',
        'BaseFuel': 355,
        'BaseIgnitionTiming': 913,
        'LambdaSensor': 1143,
        'RPM': 1500,
        'Load': 13216,
        'BatteryVolt': 12.6,
        'MAPSource': 87
    }

    # Process data
    result = engine.process_realtime_data(sample_ecu_data)

    if result:
        print("Real-time Analysis Results:")
        print(f"Driving Event: {result['event_prediction']}")
        print(f"Driving Style: {result['style_prediction']}")
        print(f"Recommendations: {result['tuning_recommendations']}")
    else:
        print("Could not process data")

if __name__ == "__main__":
    main()
