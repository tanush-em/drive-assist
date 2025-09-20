
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

        # Only remove rows with clearly impossible values, be more lenient
        df = df[df['RPM'] >= 0]  # RPM cannot be negative
        df = df[df['RPM'] <= 10000]  # More generous max RPM
        # Don't filter Load as it might have different scales
        # Don't filter BatteryVoltage as it might be missing or in different units

        logger.info(f"Removed {initial_shape[0] - df.shape[0]} faulty readings")
        return df

    def synchronize_timestamps(self, df):
        """Synchronize timestamps and calibrate sensors"""
        logger.info("Synchronizing timestamps...")

        # Convert timestamp to seconds from start
        df = df.copy()
        
        # Try multiple timestamp formats
        timestamp_formats = [
            '%H:%M:%S.%f',  # "07:54:58.422"
            '%M:%S.%f',     # "54:58.4" 
            '%H:%M:%S',     # "07:54:58"
            '%M:%S'         # "54:58"
        ]
        
        parsed_successfully = False
        
        for fmt in timestamp_formats:
            try:
                if ':' in str(df['Timestamp'].iloc[0]):
                    parsed_time = pd.to_datetime(df['Timestamp'], format=fmt, errors='coerce')
                    
                    # Check if parsing was successful (no NaT values)
                    if not parsed_time.isna().all():
                        if fmt.startswith('%H'):
                            # Full time format
                            df['timestamp_seconds'] = (parsed_time.dt.hour * 3600 + 
                                                     parsed_time.dt.minute * 60 + 
                                                     parsed_time.dt.second + 
                                                     parsed_time.dt.microsecond / 1e6)
                        else:
                            # Minutes:seconds format
                            df['timestamp_seconds'] = (parsed_time.dt.minute * 60 + 
                                                     parsed_time.dt.second + 
                                                     parsed_time.dt.microsecond / 1e6)
                        parsed_successfully = True
                        logger.info(f"Successfully parsed timestamps using format: {fmt}")
                        break
            except (ValueError, AttributeError):
                continue
        
        # Fallback to row index if all parsing attempts failed
        if not parsed_successfully:
            logger.warning("Could not parse timestamps with any known format, using row index")
            df['timestamp_seconds'] = df.index * 0.1  # Assume 10 Hz sampling

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
        # Handle NaN values in timestamp_seconds
        df = df.dropna(subset=['timestamp_seconds'])
        df['interval'] = (df['timestamp_seconds'] // interval_seconds).astype(int)

        # Aggregate features per interval
        agg_features = df.groupby('interval').agg({
            'RPM': ['mean', 'std', 'max', 'min'],
            'Load': ['mean', 'std'],
            'BaseFuel': ['mean', 'std'],
            'IgnitionTiming': ['mean'],
            'LambdaSensor1': ['mean'],
            'Event': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Cruising'
        }).reset_index()

        # Flatten column names
        agg_features.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg_features.columns.values]
        
        # Rename the Event column back to a simple name
        if 'Event_<lambda>' in agg_features.columns:
            agg_features = agg_features.rename(columns={'Event_<lambda>': 'Event'})

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
            'IgnitionTiming_mean',
            'LambdaSensor1_mean'
        ]

        # Handle missing values
        logger.info(f"Available columns in aggregated data: {df.columns.tolist()}")
        logger.info(f"Looking for features: {numerical_features}")
        
        # Check which features actually exist
        available_features = [col for col in numerical_features if col in df.columns]
        logger.info(f"Found features: {available_features}")
        
        if not available_features:
            logger.error("No features found! Check column names.")
            return df.iloc[:0]  # Return empty dataframe
            
        df_clean = df[available_features].ffill().bfill()

        # Store feature columns
        self.feature_columns = available_features

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
