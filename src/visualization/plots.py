"""
Visualization utilities for the ticket price prediction dashboard.
This module provides functions for creating various plots and visualizations.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any

def get_common_layout(title: str, x_title: str, y_title: str) -> Dict[str, Any]:
    """
    Get common layout settings for plots.
    
    Args:
        title (str): Plot title
        x_title (str): X-axis title
        y_title (str): Y-axis title
        
    Returns:
        Dict[str, Any]: Layout settings
    """
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

def create_feature_importance_plot(
    feature_importance: pd.DataFrame,
    title: str = "Feature Importance"
) -> go.Figure:
    """
    Create a feature importance plot.
    
    Args:
        feature_importance (pd.DataFrame): Feature importance data
        title (str): Plot title
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = px.bar(
        feature_importance.sort_values('Importance', ascending=True).tail(10),
        orientation='h',
        title=title,
        labels={'Importance': 'Feature Importance', 'Feature': 'Feature'},
        color='Importance',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        **get_common_layout(
            title,
            "Feature Importance Score",
            "Feature"
        )
    )
    
    return fig

def create_prediction_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted Prices"
) -> go.Figure:
    """
    Create a scatter plot of actual vs predicted prices.
    
    Args:
        y_true (np.ndarray): Actual prices
        y_pred (np.ndarray): Predicted prices
        title (str): Plot title
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = px.scatter(
        x=y_true,
        y=y_pred,
        labels={'x': 'Actual Price ($)', 'y': 'Predicted Price ($)'},
        title=title,
        trendline="ols"
    )
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red')
        )
    )
    
    fig.update_layout(
        **get_common_layout(
            title,
            "Actual Price ($)",
            "Predicted Price ($)"
        )
    )
    
    return fig

def create_price_trend_plot(
    price_data: pd.DataFrame,
    title: str = "Price Evolution Analysis"
) -> go.Figure:
    """
    Create a price trend plot with confidence intervals.
    
    Args:
        price_data (pd.DataFrame): Price trend data
        title (str): Plot title
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = go.Figure()
    
    # Add mean price line
    fig.add_trace(go.Scatter(
        x=price_data['days_until_event'],
        y=price_data['mean'].round(2),
        mode='lines+markers',
        name='Average Price',
        line=dict(color='blue', width=2),
        hovertemplate="Days Until Event: %{x}<br>Average Price: $%{y:.2f}<extra></extra>"
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=price_data['days_until_event'],
        y=price_data['upper_bound'].round(2),
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False,
        hovertemplate="Upper Bound: $%{y:.2f}<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=price_data['days_until_event'],
        y=price_data['lower_bound'].round(2),
        mode='lines',
        name='Lower Bound',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0,100,255,0.2)',
        showlegend=False,
        hovertemplate="Lower Bound: $%{y:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        **get_common_layout(
            title,
            "Days Until Event",
            "Price ($)"
        )
    )
    
    return fig 