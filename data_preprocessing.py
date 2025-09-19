
# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ECUDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None

    def remove_faulty_readings(self, df):
        """Remove rows with faulty sensor readings"""
        logger.info("Removing faulty sensor readings...")
        initial_shape = df.shape

        # Remove rows where critical sensors have impossible values
        df = df[df['RPM'] >= 0]  # RPM cannot be negative
        df = df[df['RPM'] <= 8000]  # Reasonable max RPM
        df = df[df['Load'] >= 0]  # Load cannot be negative
        df = df[df['BatteryVolt'] > 10]  # Battery voltage should be reasonable
        df = df[df['BatteryVolt'] < 16]

        logger.info(f"Removed {initial_shape[0] - df.shape[0]} faulty readings")
        return df

    def synchronize_timestamps(self, df):
        """Synchronize timestamps and calibrate sensors"""
        logger.info("Synchronizing timestamps...")

        # Convert timestamp to seconds from start
        df = df.copy()
        df['timestamp_seconds'] = pd.to_datetime(df['Timestamp'], format='%M:%S.%f').dt.second + \
                                 pd.to_datetime(df['Timestamp'], format='%M:%S.%f').dt.microsecond / 1e6

        # Sort by timestamp
        df = df.sort_values('timestamp_seconds')

        return df

    def create_driving_events(self, df, window_size=5):
        """Create driving event timeline"""
        logger.info("Creating driving event timeline...")

        events = []
        for i in range(len(df) - window_size + 1):
            window = df.iloc[i:i+window_size]

            # Calculate deltas
            rpm_delta = window['RPM'].iloc[-1] - window['RPM'].iloc[0]
            load_delta = window['Load'].iloc[-1] - window['Load'].iloc[0] if not window['Load'].isna().all() else 0

            # Classify event
            if rpm_delta > 50:
                event = 'Accelerating'
            elif rpm_delta < -50:
                event = 'Braking'
            else:
                event = 'Cruising'

            events.append(event)

        # Pad events to match dataframe length
        events.extend(['Cruising'] * (window_size - 1))

        df = df.copy()
        df['Event'] = events
        return df

    def aggregate_temporal_features(self, df, interval_seconds=30):
        """Group events into larger intervals for analysis"""
        logger.info(f"Aggregating features over {interval_seconds}s intervals...")

        df = df.copy()
        df['interval'] = (df['timestamp_seconds'] // interval_seconds).astype(int)

        # Aggregate features per interval
        agg_features = df.groupby('interval').agg({
            'RPM': ['mean', 'std', 'max', 'min'],
            'Load': ['mean', 'std'],
            'BaseFuel': ['mean', 'std'],
            'BaseIgnitionTiming': ['mean'],
            'LambdaSensor': ['mean'],
            'Event': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Cruising'
        }).reset_index()

        # Flatten column names
        agg_features.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg_features.columns.values]

        return agg_features

    def extract_driving_patterns(self, df):
        """Detect driving patterns for high-level classification"""
        logger.info("Extracting driving patterns...")

        patterns = {}

        # Acceleration patterns
        patterns['avg_acceleration_intensity'] = df['RPM_std'].mean()
        patterns['max_acceleration'] = df['RPM_max'].max() - df['RPM_mean'].mean()
        patterns['acceleration_frequency'] = (df['Event'] == 'Accelerating').sum() / len(df)

        # Braking patterns
        patterns['braking_frequency'] = (df['Event'] == 'Braking').sum() / len(df)

        # Fuel efficiency patterns
        patterns['avg_fuel_consumption'] = df['BaseFuel_mean'].mean()
        patterns['fuel_variation'] = df['BaseFuel_std'].mean()

        # Overall style classification
        if patterns['acceleration_frequency'] > 0.4 and patterns['avg_acceleration_intensity'] > 50:
            style = 'Aggressive'
        elif patterns['acceleration_frequency'] < 0.2 and patterns['avg_acceleration_intensity'] < 30:
            style = 'Eco'
        else:
            style = 'Balanced'

        patterns['driving_style'] = style

        return patterns

    def prepare_features_for_ml(self, df):
        """Prepare features for machine learning"""
        logger.info("Preparing features for machine learning...")

        # Select numerical features
        numerical_features = [
            'RPM_mean', 'RPM_std', 'RPM_max', 'RPM_min',
            'Load_mean', 'Load_std',
            'BaseFuel_mean', 'BaseFuel_std',
            'BaseIgnitionTiming_mean',
            'LambdaSensor_mean'
        ]

        # Handle missing values
        df_clean = df[numerical_features].fillna(method='forward').fillna(method='backward')

        # Store feature columns
        self.feature_columns = numerical_features

        return df_clean

    def fit_transform(self, df):
        """Complete preprocessing pipeline"""
        logger.info("Starting complete preprocessing pipeline...")

        # Step 1: Remove faulty readings
        df = self.remove_faulty_readings(df)

        # Step 2: Synchronize timestamps
        df = self.synchronize_timestamps(df)

        # Step 3: Create driving events
        df = self.create_driving_events(df)

        # Step 4: Aggregate temporal features
        agg_df = self.aggregate_temporal_features(df)

        # Step 5: Prepare features for ML
        features = self.prepare_features_for_ml(agg_df)

        # Step 6: Scale features
        scaled_features = self.scaler.fit_transform(features)

        # Step 7: Extract patterns for labeling
        patterns = self.extract_driving_patterns(agg_df)

        return scaled_features, patterns, agg_df
