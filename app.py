import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.models.dashboard_xgboost_models import prepare_features, build_xgboost_model_stratified, build_xgboost_model_tscv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Ticket Price Prediction Dashboard",
    page_icon="🎫",
    layout="wide"
)

# Add custom CSS for consistent styling
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
    df = pd.read_csv('prepared_price_prediction.csv')
    return df

# Cache model training
@st.cache_resource
def train_models():
    print("Training models (this will only happen once)...")
    df = load_data()
    
    # Split data into floor and non-floor tickets
    df_floor = df[df['is_ga_floor'] == 1].copy()
    df_non_floor = df[df['is_ga_floor'] == 0].copy()
    
    # Prepare features
    df_model_floor = prepare_features(df_floor)
    df_model_non_floor = prepare_features(df_non_floor)
    
    # For non-floor tickets, add event_strata for stratified model
    # (reproduce logic from old stratified model)
    high_popularity_events = df_model_non_floor.groupby('event_name')['price'].mean().nlargest(10).index
    df_model_non_floor['event_strata'] = df_model_non_floor['event_name'].apply(
        lambda x: 'High' if x in high_popularity_events else 'Low'
    )
    
    # Train models
    model_floor, X_test_floor, y_test_floor, feature_importance_floor = build_xgboost_model_tscv(df_model_floor)
    model_non_floor, X_test_non_floor, y_test_non_floor, feature_importance_non_floor = build_xgboost_model_stratified(df_model_non_floor)
    
    # Calculate predictions for test data
    y_pred_floor = model_floor.predict(xgb.DMatrix(X_test_floor))
    y_pred_non_floor = model_non_floor.predict(xgb.DMatrix(X_test_non_floor))
    
    # Calculate metrics
    rmse_floor = np.sqrt(mean_squared_error(y_test_floor, y_pred_floor))
    r2_floor = r2_score(y_test_floor, y_pred_floor)
    rmse_non_floor = np.sqrt(mean_squared_error(y_test_non_floor, y_pred_non_floor))
    r2_non_floor = r2_score(y_test_non_floor, y_pred_non_floor)
    
    return {
        'floor': {
            'model': model_floor,
            'data': df_floor,
            'feature_importance': feature_importance_floor,
            'metrics': {
                'rmse': rmse_floor,
                'r2': r2_floor
            },
            'test_data': {
                'X': X_test_floor,
                'y': y_test_floor,
                'predictions': y_pred_floor
            }
        },
        'non_floor': {
            'model': model_non_floor,
            'data': df_non_floor,
            'feature_importance': feature_importance_non_floor,
            'metrics': {
                'rmse': rmse_non_floor,
                'r2': r2_non_floor
            },
            'test_data': {
                'X': X_test_non_floor,
                'y': y_test_non_floor,
                'predictions': y_pred_non_floor
            }
        }
    }

# Load and prepare data
with st.spinner('Loading and preparing data...'):
    models = train_models()
    df = load_data()

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Model Performance", 
    "Feature Analysis", 
    "Price Predictions", 
    "Entry Price Dynamics Analysis",
    "Savings Calculator"
])

# Common layout settings for all plots
def get_common_layout(title, x_title, y_title):
    return dict(
        title=dict(
            text=title,
            font=dict(size=24, color='#1E3A8A'),
            x=0.5,
            y=0.95
        ),
        xaxis=dict(
            title=dict(
                text=x_title,
                font=dict(size=14, color='#4B5563')
            ),
            showgrid=True,
            gridcolor='#E5E7EB',
            zerolinecolor='#E5E7EB'
        ),
        yaxis=dict(
            title=dict(
                text=y_title,
                font=dict(size=14, color='#4B5563')
            ),
            showgrid=True,
            gridcolor='#E5E7EB',
            zerolinecolor='#E5E7EB',
            tickprefix='$',
            tickformat='.2f'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50),
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#E5E7EB',
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Rockwell",
            bordercolor='#E5E7EB'
        )
    )

with tab1:
    st.header("Model Performance Metrics")
    st.markdown("""
    ### Model Accuracy and Reliability
    These metrics demonstrate the model's ability to predict ticket prices accurately across different categories.
    """)
    
    # Display metrics in columns with enhanced styling
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Floor Tickets Model")
        st.markdown("""
        <div class='metric-card'>
            <h4>Root Mean Square Error (RMSE)</h4>
            <p style='font-size: 24px; color: #1E3A8A;'>${:.2f}</p>
            <p style='color: #6B7280;'>Average prediction error in dollars</p>
        </div>
        """.format(models['floor']['metrics']['rmse']), unsafe_allow_html=True)
        st.markdown("""
        <div class='metric-card'>
            <h4>R² Score</h4>
            <p style='font-size: 24px; color: #1E3A8A;'>{:.4f}</p>
            <p style='color: #6B7280;'>Proportion of variance explained by the model</p>
        </div>
        """.format(models['floor']['metrics']['r2']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Entry Price Model")
        st.markdown("""
        <div class='metric-card'>
            <h4>Root Mean Square Error (RMSE)</h4>
            <p style='font-size: 24px; color: #1E3A8A;'>${:.2f}</p>
            <p style='color: #6B7280;'>Average prediction error in dollars</p>
        </div>
        """.format(models['non_floor']['metrics']['rmse']), unsafe_allow_html=True)
        st.markdown("""
        <div class='metric-card'>
            <h4>R² Score</h4>
            <p style='font-size: 24px; color: #1E3A8A;'>{:.4f}</p>
            <p style='color: #6B7280;'>Proportion of variance explained by the model</p>
        </div>
        """.format(models['non_floor']['metrics']['r2']), unsafe_allow_html=True)

with tab2:
    st.header("Feature Importance Analysis")
    st.markdown("""
    ### Key Factors Influencing Ticket Prices
    This analysis identifies the most significant factors affecting ticket prices across different categories.
    """)
    
    # Create feature importance plots with enhanced styling
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Floor Tickets")
        fig_floor = px.bar(
            models['floor']['feature_importance'].sort_values('mean_importance', ascending=True).tail(10),
            orientation='h',
            title='Top 10 Features - Floor Tickets',
            labels={'mean_importance': 'Feature Importance', 'index': 'Feature'},
            color='mean_importance',
            color_continuous_scale='Blues'
        )
        fig_floor.update_layout(
            **get_common_layout(
                "Feature Importance - Floor Tickets",
                "Feature Importance Score",
                "Feature"
            )
        )
        st.plotly_chart(fig_floor, use_container_width=True)
    
    with col2:
        st.markdown("### Entry Price Tickets")
        fig_non_floor = px.bar(
            models['non_floor']['feature_importance'].sort_values('mean_importance', ascending=True).tail(10),
            orientation='h',
            title='Top 10 Features - Entry Price Tickets',
            labels={'mean_importance': 'Feature Importance', 'index': 'Feature'},
            color='mean_importance',
            color_continuous_scale='Blues'
        )
        fig_non_floor.update_layout(
            **get_common_layout(
                "Feature Importance - Entry Price Tickets",
                "Feature Importance Score",
                "Feature"
            )
        )
        st.plotly_chart(fig_non_floor, use_container_width=True)

with tab3:
    st.header("Price Predictions")
    st.markdown("""
    ### Model Prediction Accuracy
    This visualization compares predicted prices against actual prices, demonstrating the model's reliability.
    """)
    
    # Create actual vs predicted plots with enhanced styling
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Floor Tickets")
        fig_floor_pred = px.scatter(
            x=models['floor']['test_data']['y'],
            y=models['floor']['test_data']['predictions'],
            labels={'x': 'Actual Price ($)', 'y': 'Predicted Price ($)'},
            title='Actual vs Predicted Prices - Floor Tickets',
            trendline="ols"
        )
        fig_floor_pred.add_trace(
            go.Scatter(
                x=[models['floor']['test_data']['y'].min(), models['floor']['test_data']['y'].max()],
                y=[models['floor']['test_data']['y'].min(), models['floor']['test_data']['y'].max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            )
        )
        fig_floor_pred.update_layout(
            **get_common_layout(
                "Price Prediction Accuracy - Floor Tickets",
                "Actual Price ($)",
                "Predicted Price ($)"
            )
        )
        st.plotly_chart(fig_floor_pred, use_container_width=True)
    
    with col2:
        st.markdown("### Entry Price Tickets")
        fig_non_floor_pred = px.scatter(
            x=models['non_floor']['test_data']['y'],
            y=models['non_floor']['test_data']['predictions'],
            labels={'x': 'Actual Price ($)', 'y': 'Predicted Price ($)'},
            title='Actual vs Predicted Prices - Entry Price Tickets',
            trendline="ols"
        )
        fig_non_floor_pred.add_trace(
            go.Scatter(
                x=[models['non_floor']['test_data']['y'].min(), models['non_floor']['test_data']['y'].max()],
                y=[models['non_floor']['test_data']['y'].min(), models['non_floor']['test_data']['y'].max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            )
        )
        fig_non_floor_pred.update_layout(
            **get_common_layout(
                "Price Prediction Accuracy - Entry Price Tickets",
                "Actual Price ($)",
                "Predicted Price ($)"
            )
        )
        st.plotly_chart(fig_non_floor_pred, use_container_width=True)

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
    
    # Filter data based on selection (only non-floor tickets)
    event_data = df[(df['event_name'] == selected_event) & (df['is_ga_floor'] == 0)].copy()
    
    if len(event_data) == 0:
        st.error(f"No data available for {selected_event}. Please select a different event.")
    else:
        # Create price trend plot with enhanced styling
        st.subheader(f"Price Evolution Analysis: {selected_event}")
        
        # Calculate average prices by days until event
        price_trend = event_data.groupby('days_until_event')['price'].agg(['mean', 'std', 'count']).reset_index()
        
        if len(price_trend) == 0:
            st.error("No price data available for the selected event. Please try a different event.")
        else:
            price_trend['lower_bound'] = price_trend['mean'] - price_trend['std']
            price_trend['upper_bound'] = price_trend['mean'] + price_trend['std']
            
            # Find the best day to buy (day with lowest average price)
            best_day = price_trend.loc[price_trend['mean'].idxmin()]
            worst_day = price_trend.loc[price_trend['mean'].idxmax()]
            
            # Calculate price volatility
            price_volatility = price_trend['std'].mean()
            avg_price = price_trend['mean'].mean()
            volatility_percentage = (price_volatility / avg_price) * 100
            
            # Create the plot
            fig_trend = go.Figure()
            
            # Add mean price line
            fig_trend.add_trace(go.Scatter(
                x=price_trend['days_until_event'],
                y=price_trend['mean'].round(2),
                mode='lines+markers',
                name='Average Entry Price',
                line=dict(color='blue', width=2),
                hovertemplate="Days Until Event: %{x}<br>Average Entry Price: $%{y:.2f}<extra></extra>"
            ))
            
            # Add confidence interval
            fig_trend.add_trace(go.Scatter(
                x=price_trend['days_until_event'],
                y=price_trend['upper_bound'].round(2),
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False,
                hovertemplate="Upper Bound: $%{y:.2f}<extra></extra>"
            ))
            
            fig_trend.add_trace(go.Scatter(
                x=price_trend['days_until_event'],
                y=price_trend['lower_bound'].round(2),
                mode='lines',
                name='Lower Bound',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0,100,255,0.2)',
                showlegend=False,
                hovertemplate="Lower Bound: $%{y:.2f}<extra></extra>"
            ))
            
            # Add vertical line for selected days
            fig_trend.add_vline(
                x=days_until_event,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Selected: {days_until_event} days",
                annotation_position="top right"
            )
            
            # Add vertical line for best day
            fig_trend.add_vline(
                x=best_day['days_until_event'],
                line_dash="dash",
                line_color="green",
                annotation_text=f"Best Day: {int(best_day['days_until_event'])} days",
                annotation_position="top left"
            )
            
            # Update plot styling
            fig_trend.update_layout(
                **get_common_layout(
                    f"Price Evolution Analysis - {selected_event}",
                    "Days Until Event",
                    "Entry Price ($)"
                )
            )
            
            # Add annotations for key points
            fig_trend.add_annotation(
                x=best_day['days_until_event'],
                y=best_day['mean'],
                text=f"Best Price: ${best_day['mean']:.2f}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )
            
            # Display the plot with enhanced styling
            st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': True})
            
            # Display recommendations with enhanced styling
            st.markdown("### Purchase Recommendations")
            
            # Find the closest available day to the selected day
            available_days = price_trend['days_until_event'].values
            closest_day = min(available_days, key=lambda x: abs(x - days_until_event))
            
            # Calculate potential savings
            current_price = price_trend[price_trend['days_until_event'] == closest_day]['mean'].iloc[0]
            best_price = best_day['mean']
            potential_savings = current_price - best_price
            savings_percentage = (potential_savings / current_price) * 100
            
            # Create three columns for recommendations
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Best Day to Buy",
                    f"{int(best_day['days_until_event'])} days before event",
                    f"Average price: ${best_price:.2f}"
                )
            
            with col2:
                st.metric(
                    "Potential Savings",
                    f"${potential_savings:.2f}",
                    f"{savings_percentage:.1f}% savings"
                )
            
            with col3:
                st.metric(
                    "Price Volatility",
                    f"{volatility_percentage:.1f}%",
                    "Based on historical data"
                )
            
            # Display detailed recommendations
            if closest_day != days_until_event:
                st.info(f"Note: Showing data for {closest_day} days before event (closest available to your selection of {days_until_event} days)")
            
            if closest_day > best_day['days_until_event']:
                st.success(f"""
                🎯 **Recommended Strategy**: Consider waiting to purchase.
                - Best price typically occurs {int(best_day['days_until_event'])} days before the event
                - You could save ${potential_savings:.2f} ({savings_percentage:.1f}%) by waiting
                - Price volatility is {volatility_percentage:.1f}%, indicating moderate price fluctuations
                """)
            elif closest_day == best_day['days_until_event']:
                st.success(f"""
                🎯 **Recommended Strategy**: This is the optimal time to buy!
                - You're at the historically best price point
                - Average price at this point: ${best_price:.2f}
                - Price volatility is {volatility_percentage:.1f}%, indicating moderate price fluctuations
                """)
            else:
                st.warning(f"""
                ⚠️ **Recommended Strategy**: Consider purchasing now.
                - You're past the optimal buying window
                - Prices tend to increase as the event approaches
                - Current price is ${current_price:.2f} vs. best price of ${best_price:.2f}
                """)
            
            # Display statistics with enhanced styling
            st.markdown("### Price Statistics")
            selected_data = event_data[event_data['days_until_event'] == closest_day]
            
            if len(selected_data) > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    <div class='metric-card'>
                        <h4>Average Entry Price</h4>
                        <p style='font-size: 24px; color: #1E3A8A;'>${:.2f}</p>
                    </div>
                    """.format(selected_data['price'].mean()), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class='metric-card'>
                        <h4>Lowest Available Price</h4>
                        <p style='font-size: 24px; color: #1E3A8A;'>${:.2f}</p>
                    </div>
                    """.format(selected_data['price'].min()), unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div class='metric-card'>
                        <h4>Highest Entry Price</h4>
                        <p style='font-size: 24px; color: #1E3A8A;'>${:.2f}</p>
                    </div>
                    """.format(selected_data['price'].max()), unsafe_allow_html=True)
            else:
                st.warning(f"No price data available for {closest_day} days before the event.")

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
            # Filter for events that have floor tickets
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
            if model_type == "Floor Tickets":
                event_data = df[(df['event_name'] == selected_event) & (df['is_ga_floor'] == 1)].copy()
            else:
                event_data = df[(df['event_name'] == selected_event) & (df['is_ga_floor'] == 0)].copy()
            
            if len(event_data) > 0:
                # Get available days for the selected event
                available_days = sorted(event_data['days_until_event'].unique())
                
                if len(available_days) > 0:
                    # Find the closest available day for baseline comparison
                    baseline_day = max(available_days)
                    baseline_data = event_data[event_data['days_until_event'] == baseline_day]
                    
                    if len(baseline_data) > 0:
                        baseline_price = baseline_data['price'].mean()
                        
                        # Calculate metrics using user input price
                        potential_savings = baseline_price - current_price
                        savings_percentage = (potential_savings / baseline_price) * 100 if baseline_price != 0 else 0
                        
                        # Calculate price volatility for risk assessment
                        price_volatility = event_data['price'].std()
                        availability_risk = (price_volatility / current_price) * 100 if current_price != 0 else 0
                        
                        # Calculate ROI vs baseline
                        roi_vs_baseline = (baseline_price - current_price) / baseline_price * 100 if baseline_price != 0 else 0
                        
                        # Display metrics in a clean layout
                        st.subheader("Purchase Analysis Results")
                        
                        # Create three columns for metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Potential Savings per Ticket",
                                f"${potential_savings:.2f}",
                                f"{savings_percentage:.1f}% vs. baseline"
                            )
                        
                        with col2:
                            st.metric(
                                "Risk of Price Increase",
                                f"{availability_risk:.1f}%",
                                "Based on price volatility"
                            )
                        
                        with col3:
                            st.metric(
                                "ROI vs. Baseline",
                                f"{roi_vs_baseline:.1f}%",
                                "Cost savings potential"
                            )
                        
                        # Create price trend visualization
                        st.subheader("Price Trend Analysis")
                        
                        # Calculate price trends
                        price_trend = event_data.groupby('days_until_event')['price'].agg(['mean', 'std']).reset_index()
                        price_trend['lower_bound'] = price_trend['mean'] - price_trend['std']
                        price_trend['upper_bound'] = price_trend['mean'] + price_trend['std']
                        
                        # Create the plot
                        fig_trend = go.Figure()
                        
                        # Add mean price line
                        fig_trend.add_trace(go.Scatter(
                            x=price_trend['days_until_event'],
                            y=price_trend['mean'],
                            mode='lines+markers',
                            name='Average Price',
                            line=dict(color='blue', width=2),
                            hovertemplate="Days Until Event: %{x}<br>Average Price: $%{y:.2f}<extra></extra>"
                        ))
                        
                        # Add confidence interval
                        fig_trend.add_trace(go.Scatter(
                            x=price_trend['days_until_event'],
                            y=price_trend['upper_bound'],
                            mode='lines',
                            name='Upper Bound',
                            line=dict(width=0),
                            showlegend=False,
                            hovertemplate="Upper Bound: $%{y:.2f}<extra></extra>"
                        ))
                        
                        fig_trend.add_trace(go.Scatter(
                            x=price_trend['days_until_event'],
                            y=price_trend['lower_bound'],
                            mode='lines',
                            name='Lower Bound',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor='rgba(0,100,255,0.2)',
                            showlegend=False,
                            hovertemplate="Lower Bound: $%{y:.2f}<extra></extra>"
                        ))
                        
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
                        
                        # Update plot styling
                        fig_trend.update_layout(
                            **get_common_layout(
                                f"Price Trend Analysis - {selected_event} ({model_type})",
                                "Days Until Event",
                                "Price ($)"
                            )
                        )
                        
                        # Add annotations for key points
                        fig_trend.add_annotation(
                            x=days_until_event,
                            y=current_price,
                            text=f"Your Price: ${current_price:.2f}",
                            showarrow=True,
                            arrowhead=1,
                            ax=0,
                            ay=-40
                        )
                        
                        st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': True})
                        
                        # Add recommendations with enhanced styling
                        st.markdown("### Purchase Analysis Results")
                        
                        # Find the closest available day for baseline comparison
                        baseline_day = max(available_days)
                        baseline_data = event_data[event_data['days_until_event'] == baseline_day]
                        
                        if len(baseline_data) > 0:
                            baseline_price = baseline_data['price'].mean()
                            
                            # Calculate potential savings
                            potential_savings = baseline_price - current_price
                            savings_percentage = (potential_savings / baseline_price) * 100 if baseline_price != 0 else 0
                            
                            # Calculate price volatility for risk assessment
                            price_volatility = event_data['price'].std()
                            availability_risk = (price_volatility / current_price) * 100 if current_price != 0 else 0
                            
                            # Calculate ROI vs baseline
                            roi_vs_baseline = (baseline_price - current_price) / baseline_price * 100 if baseline_price != 0 else 0
                            
                            if potential_savings > 0:
                                st.success(f"""
                                🎯 **Recommended Strategy**: Consider purchasing tickets now.
                                - Potential savings of ${potential_savings:.2f} per ticket
                                - {savings_percentage:.1f}% lower than baseline price
                                - Risk of price increase: {availability_risk:.1f}%
                                """)
                            else:
                                st.warning(f"""
                                ⚠️ **Recommended Strategy**: Consider waiting.
                                - Current prices are ${abs(potential_savings):.2f} higher than baseline
                                - Risk of price increase: {availability_risk:.1f}%
                                - Monitor prices for potential drops
                                """)
                        else:
                            st.error("Insufficient historical data for analysis. Please try a different event.")
                    else:
                        st.error("Insufficient historical data for analysis. Please try a different event.")
                else:
                    st.error("No historical price data available for this event. Please try a different event.")
            else:
                st.error("No data available for the selected event. Please try a different event.")

# Add footer with enhanced styling
st.markdown("""
    <div style='text-align: center; color: #6B7280; padding: 2rem;'>
        <hr style='border: 1px solid #E5E7EB;'>
        <p style='font-size: 14px;'>Dara Sheehan | Ticket Price Prediction Dashboard</p>
    </div>
""", unsafe_allow_html=True) 