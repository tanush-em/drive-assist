import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
import joblib
import logging
import os
import glob
from datetime import datetime

from data_preprocessing import ECUDataPreprocessor
from lstm_models import LSTMDriverClassifier, TemporalAggregator, TuningMapRecommender

# Setup logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment_log.txt', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DriverStyleClassificationPipeline:
    def __init__(self):
        self.preprocessor = ECUDataPreprocessor()
        self.stage1_model = None
        self.stage2_model = None
        self.aggregator = TemporalAggregator()
        self.recommender = TuningMapRecommender()
        self.label_encoder = LabelEncoder()
        self.all_features = []
        self.all_event_labels = []
        self.all_style_labels = []
        self.file_stats = {}

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

    def load_all_data_files(self, data_directory='data/'):
        """Load and process all CSV files in the data directory"""
        logger.info(f"Starting to load all data files from {data_directory}")
        
        # Get all CSV files
        csv_files = glob.glob(os.path.join(data_directory, '*.csv'))
        csv_files.sort()  # Sort for consistent processing order
        
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        total_files = len(csv_files)
        successful_files = 0
        
        for i, file_path in enumerate(csv_files, 1):
            try:
                logger.info(f"Processing file {i}/{total_files}: {os.path.basename(file_path)}")
                
                # Load and prepare data for this file
                features, patterns, processed_df = self.load_and_prepare_data(file_path)
                
                # Create labels for this file
                event_labels, style_labels = self.create_synthetic_labels(processed_df)
                
                # Store data
                self.all_features.append(features)
                self.all_event_labels.append(event_labels)
                self.all_style_labels.append(style_labels)
                
                # Store file statistics
                self.file_stats[os.path.basename(file_path)] = {
                    'samples': len(features),
                    'patterns': patterns,
                    'event_distribution': {
                        'Accelerating': np.sum(event_labels == 0),
                        'Braking': np.sum(event_labels == 1),
                        'Cruising': np.sum(event_labels == 2)
                    },
                    'style_distribution': {
                        'Aggressive': np.sum(style_labels == 0),
                        'Eco': np.sum(style_labels == 1),
                        'Balanced': np.sum(style_labels == 2)
                    }
                }
                
                successful_files += 1
                logger.info(f"Successfully processed {os.path.basename(file_path)} - {len(features)} samples")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
        
        # Combine all data
        if successful_files > 0:
            logger.info(f"Combining data from {successful_files} successful files")
            
            # Concatenate all features and labels
            self.combined_features = np.vstack(self.all_features)
            self.combined_event_labels = np.concatenate(self.all_event_labels)
            self.combined_style_labels = np.concatenate(self.all_style_labels)
            
            logger.info(f"Total combined dataset: {len(self.combined_features)} samples")
            logger.info(f"Feature shape: {self.combined_features.shape}")
            logger.info(f"Event labels distribution: {np.bincount(self.combined_event_labels)}")
            logger.info(f"Style labels distribution: {np.bincount(self.combined_style_labels)}")
            
            return True
        else:
            logger.error("No files were successfully processed!")
            return False

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

    def train_stage1_model(self, features=None, event_labels=None, sequence_length=10):
        """Train Stage 1: Low-Level Driving Event Classification"""
        logger.info("Training Stage 1 model...")
        
        # Use combined data if no specific data provided
        if features is None:
            features = self.combined_features
            event_labels = self.combined_event_labels
            logger.info(f"Using combined dataset for Stage 1: {features.shape[0]} samples")

        # Initialize model
        self.stage1_model = LSTMDriverClassifier(
            sequence_length=sequence_length,
            n_features=features.shape[1]
        )

        # Build model
        self.stage1_model.build_stage1_model()

        # Create sequences
        X_seq, y_seq = self.stage1_model.create_sequences(features, event_labels)
        logger.info(f"Created sequences for Stage 1: {X_seq.shape}")

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

    def train_stage2_model(self, features=None, style_labels=None, sequence_length=10):
        """Train Stage 2: High-Level Driving Style Classification"""
        logger.info("Training Stage 2 model...")
        
        # Use combined data if no specific data provided
        if features is None:
            features = self.combined_features
            style_labels = self.combined_style_labels
            logger.info(f"Using combined dataset for Stage 2: {features.shape[0]} samples")

        # Initialize model
        self.stage2_model = LSTMDriverClassifier(
            sequence_length=sequence_length,
            n_features=features.shape[1]
        )

        # Build model
        self.stage2_model.build_stage2_model(n_driving_styles=3)

        # Create sequences
        X_seq, y_seq = self.stage2_model.create_sequences(features, style_labels)
        logger.info(f"Created sequences for Stage 2: {X_seq.shape}")

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

    def full_pipeline_training(self, data_directory='data/', single_file=None):
        """Run complete training pipeline on all data files"""
        start_time = datetime.now()
        logger.info(f"Starting full pipeline training at {start_time}")
        
        if single_file:
            # Train on single file (original behavior)
            logger.info(f"Training on single file: {single_file}")
            features, patterns, processed_df = self.load_and_prepare_data(single_file)
            event_labels, style_labels = self.create_synthetic_labels(processed_df)
            
            stage1_history = self.train_stage1_model(features, event_labels)
            stage2_history = self.train_stage2_model(features, style_labels)
            
            file_stats = {single_file: {'patterns': patterns}}
        else:
            # Train on all files
            logger.info(f"Training on all files in directory: {data_directory}")
            
            # Load all data files
            success = self.load_all_data_files(data_directory)
            if not success:
                logger.error("Failed to load data files. Exiting.")
                return None
            
            # Train models on combined data
            stage1_history = self.train_stage1_model()
            stage2_history = self.train_stage2_model()
            
            file_stats = self.file_stats

        # Save preprocessor
        joblib.dump(self.preprocessor, 'preprocessor.pkl')
        logger.info("Preprocessor saved to preprocessor.pkl")

        end_time = datetime.now()
        training_duration = end_time - start_time
        
        logger.info("="*50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"Training duration: {training_duration}")
        logger.info(f"Total samples processed: {len(self.combined_features) if hasattr(self, 'combined_features') else 'N/A'}")
        logger.info("Models saved to:")
        logger.info("  - models/stage1_model.h5")
        logger.info("  - models/stage2_model.h5")
        logger.info("  - preprocessor.pkl")
        logger.info("="*50)

        return {
            'stage1_history': stage1_history,
            'stage2_history': stage2_history,
            'file_stats': file_stats,
            'training_duration': training_duration,
            'total_samples': len(self.combined_features) if hasattr(self, 'combined_features') else 0
        }

def main():
    """Main training function"""
    logger.info("Starting Driver Style Classification Training")
    logger.info("="*60)
    
    # Initialize pipeline
    pipeline = DriverStyleClassificationPipeline()

    # Run training on all data files
    results = pipeline.full_pipeline_training(data_directory='data/')

    if results:
        logger.info("Training Summary:")
        logger.info(f"  - Total files processed: {len(results['file_stats'])}")
        logger.info(f"  - Total samples: {results['total_samples']}")
        logger.info(f"  - Training duration: {results['training_duration']}")
        
        # Log file statistics
        logger.info("\nFile Processing Summary:")
        for filename, stats in results['file_stats'].items():
            logger.info(f"  {filename}: {stats['samples']} samples")
    else:
        logger.error("Training failed!")

if __name__ == "__main__":
    main()
