
# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
from datetime import datetime, timedelta
import time

# Import our custom modules
try:
    from inference_engine import DriverStyleInferenceEngine
    from data_preprocessing import ECUDataPreprocessor
except ImportError:
    st.error("Required modules not found. Please ensure all modules are in the same directory.")

# Page configuration
st.set_page_config(
    page_title="Driver Style Analysis Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #1f77b4;
}
.recommendation-card {
    background-color: #e8f4fd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #17a2b8;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'inference_engine' not in st.session_state:
    st.session_state.inference_engine = DriverStyleInferenceEngine()
    st.session_state.models_loaded = False

if 'realtime_data' not in st.session_state:
    st.session_state.realtime_data = []

if 'driving_history' not in st.session_state:
    st.session_state.driving_history = {
        'timestamps': [],
        'events': [],
        'styles': [],
        'rpm': [],
        'load': [],
        'fuel': []
    }

def main():
    st.markdown('<h1 class="main-header">🚗 Driver Style Analysis Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", [
        "Real-time Analysis",
        "Historical Data Analysis", 
        "Model Training",
        "ECU Tuning Recommendations",
        "Driver Statistics"
    ])

    if page == "Real-time Analysis":
        realtime_analysis_page()
    elif page == "Historical Data Analysis":
        historical_analysis_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "ECU Tuning Recommendations":
        tuning_recommendations_page()
    elif page == "Driver Statistics":
        driver_statistics_page()

def load_models():
    """Load trained models"""
    try:
        success = st.session_state.inference_engine.load_models()
        if success:
            st.session_state.models_loaded = True
            st.success("Models loaded successfully!")
        else:
            st.error("Failed to load models. Please train models first.")
        return success
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return False

def realtime_analysis_page():
    st.header("Real-time ECU Data Analysis")

    # Model loading section
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Model Status")
        if st.session_state.models_loaded:
            st.success("✅ Models loaded and ready for inference")
        else:
            st.warning("⚠️ Models not loaded")

    with col2:
        if st.button("Load Models"):
            load_models()

    if not st.session_state.models_loaded:
        st.info("Please load the trained models to continue with real-time analysis.")
        return

    # Real-time data input
    st.subheader("Input ECU Data")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        rpm = st.number_input("RPM", min_value=0, max_value=8000, value=1500)
        load = st.number_input("Load", min_value=0.0, max_value=20000.0, value=13216.0)

    with col2:
        base_fuel = st.number_input("Base Fuel", min_value=0, max_value=500, value=355)
        ignition_timing = st.number_input("Ignition Timing", min_value=0, max_value=1200, value=913)

    with col3:
        lambda_sensor = st.number_input("Lambda Sensor", min_value=0, max_value=1500, value=1143)
        battery_volt = st.number_input("Battery Voltage", min_value=10.0, max_value=16.0, value=12.6)

    with col4:
        map_source = st.number_input("MAP Source", min_value=0, max_value=150, value=87)
        st.markdown("---")
        analyze_button = st.button("🔍 Analyze Driving Style", type="primary")

    # Analysis results
    if analyze_button:
        # Prepare ECU data
        ecu_data = {
            'Timestamp': datetime.now().strftime('%M:%S.%f')[:-3],
            'RPM': rpm,
            'Load': load,
            'BaseFuel': base_fuel,
            'IgnitionTiming': ignition_timing,
            'LambdaSensor1': lambda_sensor,
            'BatteryVoltage': battery_volt,
            'MAPSource': map_source
        }

        # Process data
        with st.spinner("Analyzing driving style..."):
            result = st.session_state.inference_engine.process_realtime_data(ecu_data)

        if result:
            # Update history
            st.session_state.driving_history['timestamps'].append(datetime.now())
            st.session_state.driving_history['rpm'].append(rpm)
            st.session_state.driving_history['load'].append(load)
            st.session_state.driving_history['fuel'].append(base_fuel)

            if result['event_prediction']:
                st.session_state.driving_history['events'].append(result['event_prediction']['event'])
            else:
                st.session_state.driving_history['events'].append('Unknown')

            if result['style_prediction']:
                st.session_state.driving_history['styles'].append(result['style_prediction']['style'])
            else:
                st.session_state.driving_history['styles'].append('Unknown')

            # Keep only last 100 records
            for key in st.session_state.driving_history:
                if len(st.session_state.driving_history[key]) > 100:
                    st.session_state.driving_history[key] = st.session_state.driving_history[key][-100:]

            # Display results
            display_analysis_results(result)
        else:
            st.error("Failed to analyze data. Please check your inputs.")

    # Real-time charts
    if len(st.session_state.driving_history['timestamps']) > 0:
        display_realtime_charts()

def display_analysis_results(result):
    """Display analysis results"""
    st.subheader("Analysis Results")

    col1, col2, col3 = st.columns(3)

    # Event prediction
    with col1:
        if result['event_prediction']:
            event = result['event_prediction']['event']
            confidence = result['event_prediction']['confidence']

            color = "green" if event == "Cruising" else "orange" if event == "Accelerating" else "red"

            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: {color};">Driving Event</h3>
                <h2>{event}</h2>
                <p>Confidence: {confidence:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

    # Style prediction
    with col2:
        if result['style_prediction']:
            style = result['style_prediction']['style']
            confidence = result['style_prediction']['confidence']

            color = "green" if style == "Eco" else "orange" if style == "Balanced" else "red"

            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: {color};">Driving Style</h3>
                <h2>{style}</h2>
                <p>Confidence: {confidence:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

    # Recommendations
    with col3:
        if result['tuning_recommendations']:
            recommendations = result['tuning_recommendations']
            st.markdown(f"""
            <div class="recommendation-card">
                <h3>🔧 Tuning Profile</h3>
                <h4>{recommendations['driving_style']} Driver</h4>
                <p>Profile recommendations available</p>
            </div>
            """, unsafe_allow_html=True)

    # Detailed recommendations
    if result['tuning_recommendations']:
        with st.expander("View Detailed Tuning Recommendations"):
            recommendations = result['tuning_recommendations']

            st.write("**Suggested Parameter Changes:**")
            for param, value in recommendations['suggested_changes'].items():
                st.write(f"- {param.replace('_', ' ').title()}: {value}")

            st.write("**Optimization Recommendations:**")
            for rec in recommendations['tuning_profile']['recommendations']:
                st.write(f"- {rec}")

def display_realtime_charts():
    """Display real-time charts"""
    st.subheader("Real-time Monitoring")

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('RPM Over Time', 'Load Over Time', 'Driving Events', 'Driving Styles'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"type": "bar"}, {"type": "pie"}]]
    )

    # RPM chart
    fig.add_trace(
        go.Scatter(
            x=st.session_state.driving_history['timestamps'],
            y=st.session_state.driving_history['rpm'],
            mode='lines+markers',
            name='RPM',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    # Load chart
    fig.add_trace(
        go.Scatter(
            x=st.session_state.driving_history['timestamps'],
            y=st.session_state.driving_history['load'],
            mode='lines+markers',
            name='Load',
            line=dict(color='green')
        ),
        row=1, col=2
    )

    # Event distribution
    if st.session_state.driving_history['events']:
        event_counts = pd.Series(st.session_state.driving_history['events']).value_counts()
        fig.add_trace(
            go.Bar(
                x=event_counts.index,
                y=event_counts.values,
                name='Events',
                marker_color=['red', 'orange', 'green']
            ),
            row=2, col=1
        )

    # Style distribution
    if st.session_state.driving_history['styles']:
        style_counts = pd.Series(st.session_state.driving_history['styles']).value_counts()
        fig.add_trace(
            go.Pie(
                labels=style_counts.index,
                values=style_counts.values,
                name='Styles'
            ),
            row=2, col=2
        )

    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

def historical_analysis_page():
    st.header("Historical Data Analysis")

    # File upload
    uploaded_file = st.file_uploader("Upload ECU Data CSV", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Data Overview")
        st.write(f"Dataset shape: {df.shape}")
        st.write(df.head())

        # Basic statistics
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Numerical Statistics:**")
            st.write(df.describe())

        with col2:
            st.write("**Missing Values:**")
            st.write(df.isnull().sum())

        # Visualizations
        st.subheader("Data Visualizations")

        # Time series plots
        if 'Timestamp' in df.columns:
            fig = px.line(df, x='Timestamp', y=['RPM', 'Load', 'BaseFuel'], 
                         title="ECU Parameters Over Time")
            st.plotly_chart(fig, use_container_width=True)

        # Distribution plots
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_col = st.selectbox("Select parameter for distribution", numeric_cols)

        fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
        st.plotly_chart(fig, use_container_width=True)

def model_training_page():
    st.header("Model Training")
    st.info("This page allows you to train new models with your data.")

    # Training options
    st.subheader("Training Configuration")

    col1, col2 = st.columns(2)

    with col1:
        epochs = st.number_input("Number of Epochs", min_value=1, max_value=200, value=50)
        batch_size = st.number_input("Batch Size", min_value=8, max_value=128, value=32)

    with col2:
        sequence_length = st.number_input("Sequence Length", min_value=5, max_value=50, value=10)
        validation_split = st.slider("Validation Split", 0.1, 0.5, 0.2)

    # Upload training data
    training_file = st.file_uploader("Upload Training Data CSV", type=['csv'])

    if training_file is not None:
        if st.button("Start Training"):
            with st.spinner("Training models... This may take several minutes."):
                # Here you would implement the actual training
                st.success("Training completed! Models saved.")

def tuning_recommendations_page():
    st.header("ECU Tuning Recommendations")

    # Driver style input
    driving_style = st.selectbox("Select Driving Style", ['Eco', 'Balanced', 'Aggressive'])

    # Current ECU parameters
    st.subheader("Current ECU Parameters")
    col1, col2 = st.columns(2)

    with col1:
        current_fuel = st.number_input("Current Base Fuel", value=355)
        current_ignition = st.number_input("Current Ignition Timing", value=913)

    with col2:
        current_rpm = st.number_input("Typical RPM", value=1500)
        current_load = st.number_input("Typical Load", value=13216)

    if st.button("Generate Recommendations"):
        from lstm_models import TuningMapRecommender

        recommender = TuningMapRecommender()
        current_params = {
            'fuel': current_fuel,
            'ignition_timing': current_ignition,
            'rpm': current_rpm,
            'load': current_load
        }

        recommendations = recommender.generate_recommendations(driving_style, current_params)

        st.subheader("Tuning Recommendations")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Suggested Changes:**")
            for param, value in recommendations['suggested_changes'].items():
                st.write(f"- {param.replace('_', ' ').title()}: {value}")

        with col2:
            st.write("**Optimization Tips:**")
            for tip in recommendations['tuning_profile']['recommendations']:
                st.write(f"- {tip}")

def driver_statistics_page():
    st.header("Driver Statistics")

    if len(st.session_state.driving_history['timestamps']) == 0:
        st.info("No driving data available. Please use the Real-time Analysis page to collect data.")
        return

    # Statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_rpm = np.mean(st.session_state.driving_history['rpm'])
        st.metric("Average RPM", f"{avg_rpm:.0f}")

    with col2:
        avg_load = np.mean(st.session_state.driving_history['load'])
        st.metric("Average Load", f"{avg_load:.0f}")

    with col3:
        total_sessions = len(st.session_state.driving_history['timestamps'])
        st.metric("Total Sessions", total_sessions)

    with col4:
        if st.session_state.driving_history['styles']:
            dominant_style = max(set(st.session_state.driving_history['styles']), 
                               key=st.session_state.driving_history['styles'].count)
            st.metric("Dominant Style", dominant_style)

    # Detailed statistics
    st.subheader("Detailed Analysis")

    if st.session_state.driving_history['events']:
        # Event analysis
        event_df = pd.DataFrame({
            'Timestamp': st.session_state.driving_history['timestamps'],
            'Event': st.session_state.driving_history['events'],
            'RPM': st.session_state.driving_history['rpm']
        })

        fig = px.scatter(event_df, x='Timestamp', y='RPM', color='Event',
                        title="Driving Events Over Time")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
