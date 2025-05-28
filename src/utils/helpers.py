"""
Helper functions for the ticket price prediction project.
This module provides utility functions used across the project.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

def calculate_price_metrics(
    price_data: pd.DataFrame,
    days_until_event: int
) -> Dict[str, float]:
    """
    Calculate price metrics for a given number of days until event.
    
    Args:
        price_data (pd.DataFrame): Price data
        days_until_event (int): Days until event
        
    Returns:
        Dict[str, float]: Price metrics
    """
    # Find the closest available day
    available_days = sorted(price_data['days_until_event'].unique())
    closest_day = min(available_days, key=lambda x: abs(x - days_until_event))
    
    # Get data for the closest day
    day_data = price_data[price_data['days_until_event'] == closest_day]
    
    if len(day_data) == 0:
        return {
            'mean_price': 0.0,
            'min_price': 0.0,
            'max_price': 0.0,
            'std_price': 0.0,
            'volatility': 0.0
        }
    
    mean_price = day_data['price'].mean()
    min_price = day_data['price'].min()
    max_price = day_data['price'].max()
    std_price = day_data['price'].std()
    volatility = (std_price / mean_price) * 100 if mean_price != 0 else 0
    
    return {
        'mean_price': mean_price,
        'min_price': min_price,
        'max_price': max_price,
        'std_price': std_price,
        'volatility': volatility
    }

def calculate_savings_metrics(
    current_price: float,
    baseline_price: float
) -> Dict[str, float]:
    """
    Calculate savings metrics.
    
    Args:
        current_price (float): Current ticket price
        baseline_price (float): Baseline price for comparison
        
    Returns:
        Dict[str, float]: Savings metrics
    """
    if baseline_price == 0:
        return {
            'potential_savings': 0.0,
            'savings_percentage': 0.0,
            'roi': 0.0
        }
    
    potential_savings = baseline_price - current_price
    savings_percentage = (potential_savings / baseline_price) * 100
    roi = (potential_savings / current_price) * 100 if current_price != 0 else 0
    
    return {
        'potential_savings': potential_savings,
        'savings_percentage': savings_percentage,
        'roi': roi
    }

def get_purchase_recommendation(
    current_price: float,
    best_price: float,
    days_until_event: int,
    best_day: int,
    volatility: float
) -> Dict[str, Any]:
    """
    Generate purchase recommendation based on price analysis.
    
    Args:
        current_price (float): Current ticket price
        best_price (float): Best historical price
        days_until_event (int): Days until event
        best_day (int): Best day to buy (days before event)
        volatility (float): Price volatility percentage
        
    Returns:
        Dict[str, Any]: Purchase recommendation
    """
    if current_price <= best_price:
        return {
            'recommendation': 'buy',
            'confidence': 'high',
            'reason': 'Current price is at or below historical best price',
            'details': f'Current price (${current_price:.2f}) is at or below the historical best price (${best_price:.2f})'
        }
    
    if days_until_event <= best_day:
        return {
            'recommendation': 'buy',
            'confidence': 'medium',
            'reason': 'Past optimal buying window',
            'details': f'Best buying window was {best_day} days before event. Current price: ${current_price:.2f}'
        }
    
    if volatility > 20:
        return {
            'recommendation': 'wait',
            'confidence': 'medium',
            'reason': 'High price volatility',
            'details': f'Price volatility is {volatility:.1f}%. Consider waiting for price stabilization.'
        }
    
    return {
        'recommendation': 'wait',
        'confidence': 'high',
        'reason': 'Before optimal buying window',
        'details': f'Best buying window is {best_day} days before event. Current price: ${current_price:.2f}'
    }

def format_currency(value: float) -> str:
    """
    Format a number as currency.
    
    Args:
        value (float): Number to format
        
    Returns:
        str: Formatted currency string
    """
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    """
    Format a number as percentage.
    
    Args:
        value (float): Number to format
        
    Returns:
        str: Formatted percentage string
    """
    return f"{value:.1f}%" 