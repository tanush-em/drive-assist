# 🚗 Driver Style Classification & ECU Optimization System

A comprehensive machine learning system that analyzes driver behavior from ECU (Engine Control Unit) data and provides personalized ECU tuning recommendations. The system uses a two-stage LSTM neural network architecture to classify driving events and styles, enabling intelligent automotive performance optimization.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
  - [Windows Setup](#windows-setup)
  - [macOS Setup](#macos-setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Format](#data-format)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements an intelligent driver analysis system that:

1. **Analyzes ECU Data**: Processes real-time sensor data from vehicle ECUs
2. **Classifies Driving Behavior**: Uses LSTM networks to identify driving events and styles
3. **Provides ECU Tuning**: Generates personalized recommendations for engine optimization
4. **Real-time Dashboard**: Interactive web interface for monitoring and analysis

### Key Capabilities

- **Real-time Analysis**: Process ECU data streams in real-time
- **Historical Analysis**: Analyze large datasets of driving behavior
- **Driving Style Classification**: Identify Eco, Balanced, and Aggressive driving patterns
- **ECU Tuning Recommendations**: Generate personalized engine optimization suggestions
- **Interactive Dashboard**: Modern web interface with comprehensive visualizations

## ✨ Features

### 🔬 Machine Learning Pipeline
- **Two-Stage LSTM Architecture**: 
  - Stage 1: Low-level driving event classification (Accelerating, Braking, Cruising)
  - Stage 2: High-level driving style classification (Eco, Balanced, Aggressive)
- **Temporal Aggregation**: Groups events into meaningful time intervals
- **Feature Engineering**: Extracts relevant patterns from raw ECU data

### 📊 Data Processing
- **ECU Data Preprocessing**: Handles sensor calibration and timestamp synchronization
- **Faulty Reading Removal**: Filters out impossible sensor values
- **Feature Scaling**: Normalizes data for optimal model performance

### 🎛️ ECU Optimization
- **Personalized Tuning**: Generates recommendations based on individual driving patterns
- **Performance Profiles**: Eco, Balanced, and Aggressive tuning configurations
- **Parameter Adjustments**: Specific recommendations for fuel mapping, ignition timing, etc.

### 🌐 Web Dashboard
- **Real-time Monitoring**: Live analysis of driving behavior
- **Historical Analysis**: Upload and analyze CSV datasets
- **Interactive Visualizations**: Charts, graphs, and performance metrics
- **ECU Recommendations**: Detailed tuning suggestions with expected benefits

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   ECU Data      │───▶│  Preprocessing   │───▶│  LSTM Models    │
│   (Real-time)   │    │  & Feature Eng.  │    │  (Stage 1 & 2)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Dashboard     │◀───│  Tuning Engine   │◀───│  Predictions    │
│   (Streamlit)   │    │  & Recommender   │    │  & Classifications│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Components

1. **Data Preprocessing Module** (`data_preprocessing.py`)
   - ECU data cleaning and normalization
   - Feature extraction and aggregation
   - Driving event detection

2. **LSTM Models** (`lstm_models.py`)
   - Stage 1: Event classification model
   - Stage 2: Style classification model
   - Temporal aggregator and tuning recommender

3. **Training Pipeline** (`train_model.py`)
   - Complete training workflow
   - Model validation and saving
   - Hyperparameter configuration

4. **Inference Engine** (`inference_engine.py`)
   - Real-time prediction pipeline
   - Model loading and inference
   - Recommendation generation

5. **Web Dashboard** (`dashboard.py`)
   - Streamlit-based interactive interface
   - Real-time monitoring and analysis
   - Historical data visualization

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Windows Setup

#### Method 1: Using pip and virtual environment (Recommended)

1. **Clone the repository**:
   ```cmd
   git clone https://github.com/yourusername/FYP-vishnu.git
   cd FYP-vishnu
   ```

2. **Create a virtual environment**:
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```cmd
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```cmd
   python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
   ```

#### Method 2: Using Anaconda/Miniconda

1. **Create a conda environment**:
   ```cmd
   conda create -n driver-analysis python=3.9
   conda activate driver-analysis
   ```

2. **Install TensorFlow**:
   ```cmd
   conda install tensorflow
   ```

3. **Install remaining dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```

### macOS Setup

#### Method 1: Using pip and virtual environment (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/FYP-vishnu.git
   cd FYP-vishnu
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **For Apple Silicon (M1/M2) Macs**, use the optimized TensorFlow:
   ```bash
   pip install tensorflow-macos tensorflow-metal
   ```

#### Method 2: Using Homebrew and pyenv

1. **Install pyenv** (if not already installed):
   ```bash
   brew install pyenv
   ```

2. **Install Python 3.9**:
   ```bash
   pyenv install 3.9.16
   pyenv local 3.9.16
   ```

3. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Troubleshooting Installation

#### Common Issues

1. **TensorFlow installation fails on Windows**:
   ```cmd
   pip install --upgrade pip
   pip install tensorflow==2.13.0
   ```

2. **Memory issues during training**:
   - Reduce batch size in `train_model.py`
   - Use smaller sequence lengths
   - Ensure sufficient RAM (8GB+ recommended)

3. **CUDA/GPU issues**:
   - Install CUDA toolkit for Windows
   - For macOS, use CPU-only TensorFlow or MPS backend

## 📖 Usage

### 1. Training Models

First, train the LSTM models with your ECU data:

```bash
python train_model.py
```

This will:
- Load and preprocess the training data from `data/labeled_20180713-home2mimos.csv`
- Train both Stage 1 and Stage 2 models
- Save models to the `models/` directory
- Generate a preprocessor for data normalization

### 2. Running the Dashboard

Start the interactive web dashboard:

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

### 3. Real-time Analysis

1. **Load Models**: Click "Load Models" in the dashboard
2. **Input ECU Data**: Enter real-time sensor values
3. **Analyze**: Click "Analyze Driving Style"
4. **View Results**: See predictions and recommendations

### 4. Historical Data Analysis

1. Upload a CSV file with ECU data
2. View data statistics and visualizations
3. Analyze driving patterns and trends

## 📁 Project Structure

```
FYP-vishnu/
├── 📁 data/                          # Training datasets
│   ├── labeled_20180713-home2mimos.csv
│   ├── labeled_20180713-mimos2home.csv
│   └── ... (other labeled datasets)
├── 📁 models/                        # Trained ML models
│   ├── stage1_model.h5              # Stage 1 LSTM model
│   ├── stage2_model.h5              # Stage 2 LSTM model
│   └── best_model.h5                # Best performing model
├── 📁 env/                          # Python virtual environment
├── 📄 dashboard.py                  # Streamlit web dashboard
├── 📄 inference_engine.py           # Real-time inference engine
├── 📄 lstm_models.py               # LSTM model definitions
├── 📄 data_preprocessing.py         # Data preprocessing pipeline
├── 📄 train_model.py               # Model training script
├── 📄 preprocessor.pkl             # Saved preprocessor
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Package setup configuration
└── 📄 README.md                    # This file
```

## 📊 Data Format

### ECU Data Schema

The system expects CSV files with the following columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Timestamp` | String | Time stamp in HH:MM:SS.mmm format | "07:54:58.422" |
| `RPM` | Integer | Engine RPM | 1500 |
| `Load` | Float | Engine load percentage | 13216.0 |
| `BaseFuel` | Integer | Base fuel injection amount | 355 |
| `IgnitionTiming` | Integer | Ignition timing advance | 913 |
| `LambdaSensor1` | Integer | Lambda sensor reading | 1143 |
| `BatteryVoltage` | Float | Battery voltage | 12.6 |
| `MAPSource` | Integer | MAP sensor reading | 87 |

### Sample Data

```csv
Timestamp,RPM,Load,BaseFuel,IgnitionTiming,LambdaSensor1,BatteryVoltage,MAPSource
07:54:58.422,1500,13216.0,355,913,1143,12.6,87
07:54:58.522,1550,13450.0,360,915,1145,12.7,88
07:54:58.622,1600,13680.0,365,918,1147,12.8,89
```

## 🔧 API Reference

### DriverStyleClassificationPipeline

Main training pipeline class.

```python
from train_model import DriverStyleClassificationPipeline

# Initialize pipeline
pipeline = DriverStyleClassificationPipeline()

# Run complete training
results = pipeline.full_pipeline_training('data/your_data.csv')
```

### DriverStyleInferenceEngine

Real-time inference engine.

```python
from inference_engine import DriverStyleInferenceEngine

# Initialize engine
engine = DriverStyleInferenceEngine()

# Load trained models
engine.load_models()

# Process real-time data
result = engine.process_realtime_data(ecu_data_dict)
```

### ECUDataPreprocessor

Data preprocessing utilities.

```python
from data_preprocessing import ECUDataPreprocessor

# Initialize preprocessor
preprocessor = ECUDataPreprocessor()

# Process data
features, patterns, processed_df = preprocessor.fit_transform(df)
```

## 🧪 Model Performance

### Training Metrics

- **Stage 1 (Event Classification)**: 
  - Accuracy: ~85-90%
  - Classes: Accelerating, Braking, Cruising

- **Stage 2 (Style Classification)**:
  - Accuracy: ~80-85%
  - Classes: Eco, Balanced, Aggressive

### Model Architecture

**Stage 1 LSTM**:
- Input: 10 timesteps × 10 features
- LSTM layers: 64 → 32 units
- Dense layers: 16 → 3 units (softmax)

**Stage 2 LSTM**:
- Input: 10 timesteps × 10 features  
- LSTM layers: 128 → 64 → 32 units
- Dense layers: 32 → 16 → 3 units (softmax)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

1. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. Run tests:
   ```bash
   python -m pytest tests/
   ```

3. Format code:
   ```bash
   black *.py
   flake8 *.py
   ```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ECU data collection and labeling methodology
- LSTM architecture design principles
- Streamlit dashboard framework
- TensorFlow/Keras machine learning platform

## 📞 Support

For questions and support:

- Create an issue in the GitHub repository
- Contact: [your.email@example.com]
- Documentation: [Link to detailed docs]

---

**Note**: This system is designed for educational and research purposes. Always consult with automotive professionals before making ECU modifications to your vehicle.
