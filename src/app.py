"""
Ticket Price Prediction Dashboard.
This Streamlit app provides an interactive interface for ticket price analysis and prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from models.xgboost import TicketPriceModel
from visualization.plots import (
    create_feature_importance_plot,
    create_prediction_plot,
    create_price_trend_plot
)
from utils.helpers import (
    calculate_price_metrics,
    calculate_savings_metrics,
    get_purchase_recommendation,
    format_currency,
    format_percentage
)

# Set page config
st.set_page_config(
    page_title="Ticket Price Prediction Dashboard",
    page_icon="🎫",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stPlotlyChart {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🎫 Ticket Price Prediction Dashboard")
st.markdown("""
This executive dashboard provides data-driven insights into ticket pricing trends and optimal purchase timing.
Explore model performance, feature importance, and price predictions across different ticket categories.
""")

# Load data
@st.cache_data
def load_data():
    """Load the prepared price prediction data."""
    data_path = Path("data/processed/prepared_price_prediction.csv")
    if not data_path.exists():
        st.error("Data file not found. Please ensure the data file exists at the correct location.")
        return None
    return pd.read_csv(data_path)

# Cache model training
@st.cache_resource
def train_models():
    """Train the floor and non-floor ticket models."""
    print("Training models (this will only happen once)...")
    df = load_data()
    if df is None:
        return None
    
    # Split data into floor and non-floor tickets
    df_floor = df[df['is_ga_floor'] == 1].copy()
    df_non_floor = df[df['is_ga_floor'] == 0].copy()
    
    # Train models
    floor_model = TicketPriceModel(model_type='floor')
    non_floor_model = TicketPriceModel(model_type='non_floor')
    
    floor_results = floor_model.train(df_floor)
    non_floor_results = non_floor_model.train(df_non_floor)
    
    return {
        'floor': floor_results,
        'non_floor': non_floor_results
    }

# Load and prepare data
with st.spinner('Loading and preparing data...'):
    models = train_models()
    df = load_data()

if models is None or df is None:
    st.error("Failed to load data or train models. Please check the data files and try again.")
    st.stop()

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Model Performance",
    "Feature Analysis",
    "Price Predictions",
    "Entry Price Dynamics Analysis",
    "Savings Calculator"
])

# Model Performance Tab
with tab1:
    st.header("Model Performance Metrics")
    st.markdown("""
    ### Model Accuracy and Reliability
    These metrics demonstrate the model's ability to predict ticket prices accurately across different categories.
    """)
    
    # Display metrics in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Floor Tickets Model")
        metrics = models['floor']['metrics']
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Root Mean Square Error (RMSE)</h4>
            <p style='font-size: 24px; color: #1E3A8A;'>{format_currency(metrics['rmse_mean'])}</p>
            <p style='color: #6B7280;'>Average prediction error in dollars</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class='metric-card'>
            <h4>R² Score</h4>
            <p style='font-size: 24px; color: #1E3A8A;'>{metrics['r2_mean']:.4f}</p>
            <p style='color: #6B7280;'>Proportion of variance explained by the model</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Entry Price Model")
        metrics = models['non_floor']['metrics']
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Root Mean Square Error (RMSE)</h4>
            <p style='font-size: 24px; color: #1E3A8A;'>{format_currency(metrics['rmse_mean'])}</p>
            <p style='color: #6B7280;'>Average prediction error in dollars</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class='metric-card'>
            <h4>R² Score</h4>
            <p style='font-size: 24px; color: #1E3A8A;'>{metrics['r2_mean']:.4f}</p>
            <p style='color: #6B7280;'>Proportion of variance explained by the model</p>
        </div>
        """, unsafe_allow_html=True)

# Feature Analysis Tab
with tab2:
    st.header("Feature Importance Analysis")
    st.markdown("""
    ### Key Factors Influencing Ticket Prices
    This analysis identifies the most significant factors affecting ticket prices across different categories.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Floor Tickets")
        fig_floor = create_feature_importance_plot(
            models['floor']['feature_importance'],
            "Feature Importance - Floor Tickets"
        )
        st.plotly_chart(fig_floor, use_container_width=True)
    
    with col2:
        st.markdown("### Entry Price Tickets")
        fig_non_floor = create_feature_importance_plot(
            models['non_floor']['feature_importance'],
            "Feature Importance - Entry Price Tickets"
        )
        st.plotly_chart(fig_non_floor, use_container_width=True)

# Price Predictions Tab
with tab3:
    st.header("Price Predictions")
    st.markdown("""
    ### Model Prediction Accuracy
    This visualization compares predicted prices against actual prices, demonstrating the model's reliability.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Floor Tickets")
        test_data = models['floor']['test_data']
        fig_floor_pred = create_prediction_plot(
            test_data['y'],
            test_data['predictions'],
            "Price Prediction Accuracy - Floor Tickets"
        )
        st.plotly_chart(fig_floor_pred, use_container_width=True)
    
    with col2:
        st.markdown("### Entry Price Tickets")
        test_data = models['non_floor']['test_data']
        fig_non_floor_pred = create_prediction_plot(
            test_data['y'],
            test_data['predictions'],
            "Price Prediction Accuracy - Entry Price Tickets"
        )
        st.plotly_chart(fig_non_floor_pred, use_container_width=True)

# Entry Price Dynamics Analysis Tab
with tab4:
    st.header("Entry Price Dynamics Analysis")
    st.markdown("""
    ### Price Evolution and Optimal Purchase Timing
    This analysis helps identify the best time to purchase tickets based on historical price patterns and trends.
    """)
    
    # Get unique event names from non-floor tickets
    event_names = sorted(df[df['is_ga_floor'] == 0]['event_name'].unique())
    
    # Create filters
    col1, col2 = st.columns(2)
    
    with col1:
        selected_event = st.selectbox(
            "Select Event",
            event_names,
            index=0 if 'Billy Joel' in event_names else None
        )
    
    with col2:
        days_until_event = st.slider(
            "Days Until Event",
            min_value=1,
            max_value=30,
            value=7,
            step=1
        )
    
    # Filter data based on selection
    event_data = df[(df['event_name'] == selected_event) & (df['is_ga_floor'] == 0)].copy()
    
    if len(event_data) == 0:
        st.error(f"No data available for {selected_event}. Please select a different event.")
    else:
        # Calculate price trends
        price_trend = event_data.groupby('days_until_event')['price'].agg(['mean', 'std', 'count']).reset_index()
        price_trend['lower_bound'] = price_trend['mean'] - price_trend['std']
        price_trend['upper_bound'] = price_trend['mean'] + price_trend['std']
        
        # Create price trend plot
        fig_trend = create_price_trend_plot(
            price_trend,
            f"Price Evolution Analysis - {selected_event}"
        )
        
        # Add vertical lines for selected and best days
        best_day = price_trend.loc[price_trend['mean'].idxmin()]
        fig_trend.add_vline(
            x=days_until_event,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Selected: {days_until_event} days",
            annotation_position="top right"
        )
        fig_trend.add_vline(
            x=best_day['days_until_event'],
            line_dash="dash",
            line_color="green",
            annotation_text=f"Best Day: {int(best_day['days_until_event'])} days",
            annotation_position="top left"
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Calculate and display metrics
        price_metrics = calculate_price_metrics(event_data, days_until_event)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Average Price",
                format_currency(price_metrics['mean_price']),
                f"Range: {format_currency(price_metrics['min_price'])} - {format_currency(price_metrics['max_price'])}"
            )
        
        with col2:
            st.metric(
                "Price Volatility",
                format_percentage(price_metrics['volatility']),
                "Based on historical data"
            )
        
        with col3:
            st.metric(
                "Best Day to Buy",
                f"{int(best_day['days_until_event'])} days before event",
                f"Average price: {format_currency(best_day['mean'])}"
            )
        
        # Display recommendations
        recommendation = get_purchase_recommendation(
            price_metrics['mean_price'],
            best_day['mean'],
            days_until_event,
            int(best_day['days_until_event']),
            price_metrics['volatility']
        )
        
        if recommendation['recommendation'] == 'buy':
            st.success(f"""
            🎯 **Recommended Strategy**: {recommendation['reason']}
            - {recommendation['details']}
            - Confidence: {recommendation['confidence'].title()}
            """)
        else:
            st.warning(f"""
            ⚠️ **Recommended Strategy**: {recommendation['reason']}
            - {recommendation['details']}
            - Confidence: {recommendation['confidence'].title()}
            """)

# Savings Calculator Tab
with tab5:
    st.header("Ticket Purchase Optimization Calculator")
    st.markdown("""
    ### Smart Purchase Decision Tool
    This calculator helps optimize ticket purchases by analyzing potential savings and risks based on historical data.
    """)
    
    # Create input form
    with st.form("ticket_input_form"):
        st.subheader("Enter Ticket Details")
        
        # Add model selection
        model_type = st.radio(
            "Select Ticket Type",
            ["10% Cheapest Tickets (Entry Price)", "Floor Tickets"],
            horizontal=True
        )
        
        # Get unique event names based on selected model type
        if model_type == "Floor Tickets":
            floor_events = df[df['is_ga_floor'] == 1]['event_name'].unique()
            event_names = sorted(floor_events)
            if len(event_names) == 0:
                st.warning("No floor tickets available in the dataset.")
                event_names = ["No floor tickets available"]
        else:
            event_names = sorted(df[df['is_ga_floor'] == 0]['event_name'].unique())
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_event = st.selectbox(
                "Event Name",
                event_names,
                index=0 if 'Harry Styles' in event_names else None,
                key="savings_calc_event"
            )
        
        with col2:
            current_price = st.number_input(
                "Current Ticket Price ($)",
                min_value=0.0,
                max_value=10000.0,
                value=150.0,
                step=10.0,
                format="%.2f"
            )
        
        with col3:
            days_until_event = st.number_input(
                "Days Until Event",
                min_value=1,
                max_value=365,
                value=7,
                step=1
            )
        
        submitted = st.form_submit_button("Analyze Purchase Strategy")
    
    if submitted:
        if model_type == "Floor Tickets" and selected_event == "No floor tickets available":
            st.error("Please select a different ticket type as no floor tickets are available.")
        else:
            # Filter data based on selected model type
            is_floor = model_type == "Floor Tickets"
            event_data = df[(df['event_name'] == selected_event) & (df['is_ga_floor'] == is_floor)].copy()
            
            if len(event_data) > 0:
                # Calculate price trends
                price_trend = event_data.groupby('days_until_event')['price'].agg(['mean', 'std']).reset_index()
                
                # Find baseline price (furthest day from event)
                baseline_day = price_trend['days_until_event'].max()
                baseline_data = event_data[event_data['days_until_event'] == baseline_day]
                
                if len(baseline_data) > 0:
                    baseline_price = baseline_data['price'].mean()
                    
                    # Calculate metrics
                    savings_metrics = calculate_savings_metrics(current_price, baseline_price)
                    price_metrics = calculate_price_metrics(event_data, days_until_event)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Potential Savings per Ticket",
                            format_currency(savings_metrics['potential_savings']),
                            f"{format_percentage(savings_metrics['savings_percentage'])} vs. baseline"
                        )
                    
                    with col2:
                        st.metric(
                            "Risk of Price Increase",
                            format_percentage(price_metrics['volatility']),
                            "Based on price volatility"
                        )
                    
                    with col3:
                        st.metric(
                            "ROI vs. Baseline",
                            format_percentage(savings_metrics['roi']),
                            "Cost savings potential"
                        )
                    
                    # Create price trend visualization
                    st.subheader("Price Trend Analysis")
                    
                    fig_trend = create_price_trend_plot(
                        price_trend,
                        f"Price Trend Analysis - {selected_event} ({model_type})"
                    )
                    
                    # Add current price point
                    fig_trend.add_trace(go.Scatter(
                        x=[days_until_event],
                        y=[current_price],
                        mode='markers',
                        name='Your Price',
                        marker=dict(
                            color='red',
                            size=12,
                            symbol='star'
                        ),
                        hovertemplate="Your Price: $%{y:.2f}<extra></extra>"
                    ))
                    
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # Display recommendations
                    best_day = price_trend.loc[price_trend['mean'].idxmin()]
                    recommendation = get_purchase_recommendation(
                        current_price,
                        best_day['mean'],
                        days_until_event,
                        int(best_day['days_until_event']),
                        price_metrics['volatility']
                    )
                    
                    if recommendation['recommendation'] == 'buy':
                        st.success(f"""
                        🎯 **Recommended Strategy**: {recommendation['reason']}
                        - {recommendation['details']}
                        - Confidence: {recommendation['confidence'].title()}
                        """)
                    else:
                        st.warning(f"""
                        ⚠️ **Recommended Strategy**: {recommendation['reason']}
                        - {recommendation['details']}
                        - Confidence: {recommendation['confidence'].title()}
                        """)
                else:
                    st.error("Insufficient historical data for analysis. Please try a different event.")
            else:
                st.error("No data available for the selected event. Please try a different event.")

# Add footer
st.markdown("""
    <div style='text-align: center; color: #6B7280; padding: 2rem;'>
        <hr style='border: 1px solid #E5E7EB;'>
        <p style='font-size: 14px;'>Dara Sheehan | Ticket Price Prediction Dashboard</p>
    </div>
""", unsafe_allow_html=True) 