
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
    background-color: #2d3748;
    color: #ffffff;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #1f77b4;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.recommendation-card {
    background-color: #1a202c;
    color: #ffffff;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #17a2b8;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
        "Driver Analysis & ECU Optimization"
    ])

    if page == "Real-time Analysis":
        realtime_analysis_page()
    elif page == "Historical Data Analysis":
        historical_analysis_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Driver Analysis & ECU Optimization":
        driver_analysis_and_optimization_page()

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
    
    # Add explanation about sequence requirement
    st.info("🧠 **How it works**: The AI models use LSTM (Long Short-Term Memory) networks that analyze patterns over time. You need to run **10 analyses** to build a complete driving pattern sequence before getting accurate predictions.")

    # Model loading section
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Model Status")
        if st.session_state.models_loaded:
            st.success("✅ Models loaded and ready for inference")
            # Show current sequence status
            if hasattr(st.session_state.inference_engine, 'sequence_buffer'):
                current_seq = len(st.session_state.inference_engine.sequence_buffer)
                required_seq = st.session_state.inference_engine.sequence_length
                st.write(f"📊 Current sequence: {current_seq}/{required_seq} data points")
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

    # Control buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("🗑️ Clear History", help="Clear all collected driving data"):
            for key in st.session_state.driving_history:
                st.session_state.driving_history[key] = []
            # Also clear sequence buffer
            if hasattr(st.session_state.inference_engine, 'sequence_buffer'):
                st.session_state.inference_engine.sequence_buffer = []
            st.success("All data cleared!")
            st.rerun()
    
    with col_btn2:
        if st.button("⚡ Quick Fill Sequence", help="Automatically fill sequence for testing"):
            if st.session_state.models_loaded:
                with st.spinner("Building sequence..."):
                    # Fill with varied data to build sequence
                    for i in range(10):
                        test_data = {
                            'Timestamp': f"00:{i:02d}.0",
                            'RPM': rpm + i * 50,
                            'Load': load + i * 100,
                            'BaseFuel': base_fuel,
                            'IgnitionTiming': ignition_timing,
                            'LambdaSensor1': lambda_sensor,
                            'BatteryVoltage': battery_volt,
                            'MAPSource': map_source
                        }
                        st.session_state.inference_engine.process_realtime_data(test_data)
                st.success("✅ Sequence filled! Next analysis will give predictions.")
                st.rerun()
            else:
                st.error("Please load models first!")
    
    with col_btn3:
        st.write("")  # Empty column for spacing

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
            
        # Show sequence progress
        if result:
            sequence_length = len(st.session_state.inference_engine.sequence_buffer)
            required_length = st.session_state.inference_engine.sequence_length
            
            if sequence_length < required_length:
                progress = sequence_length / required_length
                st.progress(progress, text=f"Building driving pattern sequence: {sequence_length}/{required_length} analyses completed")
                st.info(f"ℹ️ The AI needs {required_length - sequence_length} more data points to make accurate predictions. Keep analyzing!")
            else:
                st.success("✅ Sequence complete! AI can now make predictions.")
                
            # Debug output (temporary)
            with st.expander("Debug Info (Click to expand)"):
                st.write(f"Sequence buffer length: {sequence_length}")
                st.write(f"Event prediction: {result.get('event_prediction', 'None')}")
                st.write(f"Style prediction: {result.get('style_prediction', 'None')}")

        if result:
            # Update history
            st.session_state.driving_history['timestamps'].append(datetime.now())
            st.session_state.driving_history['rpm'].append(rpm)
            st.session_state.driving_history['load'].append(load)
            st.session_state.driving_history['fuel'].append(base_fuel)

            if result['event_prediction'] and 'event' in result['event_prediction']:
                st.session_state.driving_history['events'].append(result['event_prediction']['event'])
            else:
                st.session_state.driving_history['events'].append('Unknown')

            if result['style_prediction'] and 'style' in result['style_prediction']:
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

    fig.update_layout(
        height=600, 
        showlegend=True,
        title_font_color='white',
        font_color='white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            font_color='white'
        )
    )
    # Update all subplots for better visibility
    fig.update_xaxes(
        gridcolor='rgba(255,255,255,0.2)',
        color='white'
    )
    fig.update_yaxes(
        gridcolor='rgba(255,255,255,0.2)',
        color='white'
    )
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
            fig.update_layout(
                title_font_color='white',
                font_color='white',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    gridcolor='rgba(255,255,255,0.2)',
                    color='white'
                ),
                yaxis=dict(
                    gridcolor='rgba(255,255,255,0.2)',
                    color='white'
                ),
                legend=dict(
                    font_color='white'
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        # Distribution plots
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_col = st.selectbox("Select parameter for distribution", numeric_cols)

        fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
        fig.update_layout(
            title_font_color='white',
            font_color='white',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.2)',
                color='white'
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.2)',
                color='white'
            )
        )
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

def driver_analysis_and_optimization_page():
    st.header("🏁 Driver Analysis & ECU Optimization")
    
    # Check if we have driving data
    has_data = len(st.session_state.driving_history['timestamps']) > 0
    
    if not has_data:
        st.info("📊 No driving data available yet. Use the Real-time Analysis page to collect driving data first.")
        st.markdown("---")
        st.subheader("🔧 Manual ECU Tuning Recommendations")
        manual_tuning_section()
        return
    
    # === DRIVER PROFILE ANALYSIS ===
    st.subheader("👤 Your Driving Profile")
    
    # Calculate comprehensive statistics
    stats = calculate_comprehensive_stats()
    
    # Display driver profile cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Driving Style</h3>
            <h2 style="color: {get_style_color(stats['dominant_style'])}">{stats['dominant_style']}</h2>
            <p>{stats['style_confidence']:.0f}% confidence</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚡ Performance Score</h3>
            <h2 style="color: {get_performance_color(stats['performance_score'])}">{stats['performance_score']:.0f}/100</h2>
            <p>Based on efficiency & control</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🌱 Eco Rating</h3>
            <h2 style="color: {get_eco_color(stats['eco_score'])}">{stats['eco_score']:.0f}/100</h2>
            <p>Fuel efficiency rating</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Consistency</h3>
            <h2 style="color: {get_consistency_color(stats['consistency_score'])}">{stats['consistency_score']:.0f}/100</h2>
            <p>Driving pattern stability</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # === DRIVING BEHAVIOR ANALYSIS ===
    st.subheader("📊 Driving Behavior Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Event distribution pie chart
        if st.session_state.driving_history['events']:
            event_counts = pd.Series(st.session_state.driving_history['events']).value_counts()
            fig_events = px.pie(
                values=event_counts.values, 
                names=event_counts.index,
                title="Driving Events Distribution",
                color_discrete_map={
                    'Cruising': '#2E8B57',
                    'Accelerating': '#FF6347', 
                    'Braking': '#4169E1',
                    'Unknown': '#FF6347'
                }
            )
            # Update text colors for better visibility
            fig_events.update_traces(
                textfont_size=14,
                textfont_color='white',
                textposition='inside'
            )
            fig_events.update_layout(
                title_font_color='white',
                font_color='white',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_events, use_container_width=True)
    
    with col2:
        # RPM vs Load scatter plot
        fig_scatter = px.scatter(
            x=st.session_state.driving_history['rpm'],
            y=st.session_state.driving_history['load'],
            color=st.session_state.driving_history['styles'],
            title="RPM vs Load Pattern",
            labels={'x': 'RPM', 'y': 'Load', 'color': 'Driving Style'},
            color_discrete_map={
                'Eco': '#2E8B57', 
                'Balanced': '#FFD700', 
                'Aggressive': '#FF4500',
                'Unknown': '#888888'
            }
        )
        # Update text colors for better visibility
        fig_scatter.update_layout(
            title_font_color='white',
            font_color='white',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.2)',
                color='white'
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.2)',
                color='white'
            ),
            legend=dict(
                font_color='white'
            )
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # === PERFORMANCE TRENDS ===
    st.subheader("📈 Performance Trends")
    
    # Time series analysis
    if len(st.session_state.driving_history['timestamps']) > 1:
        trend_df = pd.DataFrame({
            'Time': st.session_state.driving_history['timestamps'],
            'RPM': st.session_state.driving_history['rpm'],
            'Load': st.session_state.driving_history['load'],
            'Fuel': st.session_state.driving_history['fuel']
        })
        
        fig_trends = make_subplots(
            rows=2, cols=2,
            subplot_titles=('RPM Trend', 'Load Trend', 'Fuel Consumption', 'Driving Efficiency'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # RPM trend
        fig_trends.add_trace(
            go.Scatter(x=trend_df['Time'], y=trend_df['RPM'], 
                      mode='lines+markers', name='RPM', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Load trend
        fig_trends.add_trace(
            go.Scatter(x=trend_df['Time'], y=trend_df['Load'], 
                      mode='lines+markers', name='Load', line=dict(color='green')),
            row=1, col=2
        )
        
        # Fuel consumption
        fig_trends.add_trace(
            go.Scatter(x=trend_df['Time'], y=trend_df['Fuel'], 
                      mode='lines+markers', name='Fuel', line=dict(color='red')),
            row=2, col=1
        )
        
        # Efficiency metric (RPM/Fuel ratio)
        efficiency = np.array(trend_df['RPM']) / (np.array(trend_df['Fuel']) + 1)
        fig_trends.add_trace(
            go.Scatter(x=trend_df['Time'], y=efficiency, 
                      mode='lines+markers', name='Efficiency', line=dict(color='purple')),
            row=2, col=2
        )
        
        fig_trends.update_layout(
            height=500, 
            showlegend=False,
            title_font_color='white',
            font_color='white',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        # Update all subplots for better visibility
        fig_trends.update_xaxes(
            gridcolor='rgba(255,255,255,0.2)',
            color='white'
        )
        fig_trends.update_yaxes(
            gridcolor='rgba(255,255,255,0.2)',
            color='white'
        )
        st.plotly_chart(fig_trends, use_container_width=True)
    
    st.markdown("---")
    
    # === PERSONALIZED ECU RECOMMENDATIONS ===
    st.subheader("🔧 Personalized ECU Optimization")
    
    # Generate recommendations based on actual driving data
    recommendations = generate_personalized_recommendations(stats)
    
    # Display recommendations in tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Recommended Settings", "📊 Expected Benefits", "⚠️ Important Notes"])
    
    with tab1:
        display_ecu_recommendations(recommendations, stats)
    
    with tab2:
        display_expected_benefits(recommendations, stats)
    
    with tab3:
        display_tuning_warnings_and_tips()

def manual_tuning_section():
    """Manual tuning section for when no driving data is available"""
    st.write("Since no driving data is available, you can still get ECU tuning recommendations by manually selecting your driving style:")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        driving_style = st.selectbox("Select Your Driving Style", ['Eco', 'Balanced', 'Aggressive'])
        
        st.write("**Style Descriptions:**")
        style_descriptions = {
            'Eco': "🌱 Focus on fuel efficiency, smooth acceleration, minimal aggressive maneuvers",
            'Balanced': "⚖️ Mix of performance and efficiency, moderate acceleration, occasional spirited driving", 
            'Aggressive': "🏁 Performance-focused, frequent hard acceleration, track or sport driving"
        }
        st.info(style_descriptions[driving_style])
    
    with col2:
        if st.button("Generate Recommendations", type="primary"):
            from lstm_models import TuningMapRecommender
            recommender = TuningMapRecommender()
            
            # Use default parameters
            current_params = {
                'fuel': 355,
                'ignition_timing': 913,
                'rpm': 1500,
                'load': 13216
            }
            
            recommendations = recommender.generate_recommendations(driving_style, current_params)
            
            st.subheader("Tuning Recommendations")
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.write("**Suggested Changes:**")
                for param, value in recommendations['suggested_changes'].items():
                    st.write(f"• {param.replace('_', ' ').title()}: {value}")
            
            with col_b:
                st.write("**Optimization Tips:**")
                for tip in recommendations['tuning_profile']['recommendations']:
                    st.write(f"• {tip}")

def calculate_comprehensive_stats():
    """Calculate comprehensive driving statistics"""
    history = st.session_state.driving_history
    
    # Basic statistics
    avg_rpm = np.mean(history['rpm'])
    avg_load = np.mean(history['load'])
    avg_fuel = np.mean(history['fuel'])
    
    # Style analysis
    if history['styles']:
        style_counts = pd.Series(history['styles']).value_counts()
        dominant_style = style_counts.index[0]
        style_confidence = (style_counts.iloc[0] / len(history['styles'])) * 100
    else:
        dominant_style = "Unknown"
        style_confidence = 0
    
    # Performance score (based on RPM efficiency and load management)
    rpm_efficiency = max(0, 100 - (abs(avg_rpm - 2000) / 20))  # Optimal around 2000 RPM
    load_efficiency = max(0, 100 - (avg_load / 200))  # Lower load is better
    performance_score = (rpm_efficiency + load_efficiency) / 2
    
    # Eco score (fuel efficiency)
    fuel_efficiency = max(0, 100 - ((avg_fuel - 300) / 5))  # Lower fuel consumption is better
    eco_score = min(100, fuel_efficiency)
    
    # Consistency score (lower standard deviation is better)
    rpm_consistency = max(0, 100 - (np.std(history['rpm']) / 10))
    load_consistency = max(0, 100 - (np.std(history['load']) / 100))
    consistency_score = (rpm_consistency + load_consistency) / 2
    
    return {
        'dominant_style': dominant_style,
        'style_confidence': style_confidence,
        'performance_score': performance_score,
        'eco_score': eco_score,
        'consistency_score': consistency_score,
        'avg_rpm': avg_rpm,
        'avg_load': avg_load,
        'avg_fuel': avg_fuel,
        'total_sessions': len(history['timestamps'])
    }

def get_style_color(style):
    """Get color for driving style"""
    colors = {'Eco': '#2E8B57', 'Balanced': '#FFD700', 'Aggressive': '#FF4500'}
    return colors.get(style, '#666666')

def get_performance_color(score):
    """Get color based on performance score"""
    if score >= 80: return '#2E8B57'
    elif score >= 60: return '#FFD700'
    else: return '#FF6347'

def get_eco_color(score):
    """Get color based on eco score"""
    if score >= 75: return '#228B22'
    elif score >= 50: return '#32CD32'
    else: return '#FF6347'

def get_consistency_color(score):
    """Get color based on consistency score"""
    if score >= 70: return '#4169E1'
    elif score >= 50: return '#6495ED'
    else: return '#FF6347'

def generate_personalized_recommendations(stats):
    """Generate personalized recommendations based on actual driving data"""
    from lstm_models import TuningMapRecommender
    
    recommender = TuningMapRecommender()
    
    # Create current parameters from actual data
    current_params = {
        'fuel': stats['avg_fuel'],
        'ignition_timing': 913,  # Default value
        'rpm': stats['avg_rpm'],
        'load': stats['avg_load']
    }
    
    # Get base recommendations
    recommendations = recommender.generate_recommendations(stats['dominant_style'], current_params)
    
    # Add personalized adjustments based on performance scores
    personalized_adjustments = []
    
    if stats['eco_score'] < 50:
        personalized_adjustments.append("Consider reducing fuel map richness by 2-3% for better efficiency")
    
    if stats['performance_score'] < 60:
        personalized_adjustments.append("Optimize ignition timing for your typical RPM range")
    
    if stats['consistency_score'] < 50:
        personalized_adjustments.append("Focus on smoother throttle response mapping")
    
    recommendations['personalized_adjustments'] = personalized_adjustments
    recommendations['stats'] = stats
    
    return recommendations

def display_ecu_recommendations(recommendations, stats):
    """Display ECU tuning recommendations"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎛️ Suggested Parameter Changes:**")
        for param, value in recommendations['suggested_changes'].items():
            if isinstance(value, (int, float)):
                if value > 0:
                    st.success(f"• {param.replace('_', ' ').title()}: +{value}")
                elif value < 0:
                    st.warning(f"• {param.replace('_', ' ').title()}: {value}")
                else:
                    st.info(f"• {param.replace('_', ' ').title()}: No change needed")
            else:
                st.info(f"• {param.replace('_', ' ').title()}: {value}")
    
    with col2:
        st.markdown("**💡 Optimization Strategy:**")
        for tip in recommendations['tuning_profile']['recommendations']:
            st.write(f"• {tip}")
    
    # Personalized adjustments
    if recommendations.get('personalized_adjustments'):
        st.markdown("**🎯 Personalized Recommendations (Based on Your Data):**")
        for adjustment in recommendations['personalized_adjustments']:
            st.info(f"• {adjustment}")

def display_expected_benefits(recommendations, stats):
    """Display expected benefits from tuning"""
    st.markdown("**Expected Improvements:**")
    
    style = stats['dominant_style']
    
    if style == 'Eco':
        st.success("🌱 **Fuel Efficiency**: 5-8% improvement in fuel economy")
        st.info("⚡ **Throttle Response**: Smoother, more linear power delivery")
        st.info("🔧 **Engine Load**: Reduced stress on engine components")
    elif style == 'Aggressive':
        st.success("🏁 **Performance**: 10-15% improvement in power output")
        st.info("⚡ **Acceleration**: Sharper throttle response and faster spool-up")
        st.warning("⛽ **Fuel Consumption**: May increase by 5-10%")
    else:  # Balanced
        st.success("⚖️ **Overall**: 5-10% improvement in both performance and efficiency")
        st.info("🎯 **Drivability**: Better balance between power and economy")
    
    # Performance predictions based on current stats
    st.markdown("**Predicted Score Improvements:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        new_perf = min(100, stats['performance_score'] + 15)
        st.metric("Performance Score", f"{new_perf:.0f}", f"+{new_perf - stats['performance_score']:.0f}")
    
    with col2:
        eco_change = 10 if style == 'Eco' else (-5 if style == 'Aggressive' else 5)
        new_eco = max(0, min(100, stats['eco_score'] + eco_change))
        st.metric("Eco Score", f"{new_eco:.0f}", f"{new_eco - stats['eco_score']:+.0f}")
    
    with col3:
        new_consistency = min(100, stats['consistency_score'] + 8)
        st.metric("Consistency", f"{new_consistency:.0f}", f"+{new_consistency - stats['consistency_score']:.0f}")

def display_tuning_warnings_and_tips():
    """Display important warnings and tips"""
    st.warning("⚠️ **Important Safety Notes:**")
    st.write("• Always backup your original ECU map before making changes")
    st.write("• Start with conservative adjustments and gradually fine-tune")
    st.write("• Monitor engine parameters closely after tuning changes")
    st.write("• Consider professional dyno tuning for optimal results")
    
    st.info("💡 **Pro Tips:**")
    st.write("• Log more driving data for increasingly accurate recommendations")
    st.write("• Different weather conditions may require slight adjustments")
    st.write("• Regular maintenance ensures optimal tuning performance")
    st.write("• Consider upgrading supporting modifications (intake, exhaust) for best results")

if __name__ == "__main__":
    main()
