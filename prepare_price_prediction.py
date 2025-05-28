import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

def calculate_price_changes(df, days_to_check=7, price_drop_threshold=0.10):
    """
    Calculate if price dropped by threshold within days_to_check
    Returns: DataFrame with new target variable
    """
    print("\nCalculating price changes...")
    
    # Convert timestamp to datetime with UTC=True to handle mixed timezones
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # Sort by event_name and timestamp
    df = df.sort_values(['event_name', 'timestamp'])
    
    # Initialize target column
    df['should_wait'] = 0
    
    # Process each event separately to reduce memory usage
    for event in tqdm(df['event_name'].unique()):
        # Get event data
        event_mask = df['event_name'] == event
        event_df = df[event_mask].copy()
        
        # Create a shifted price series for comparison
        event_df['next_price'] = event_df['price'].shift(-1)
        event_df['next_timestamp'] = event_df['timestamp'].shift(-1)
        
        # Calculate time difference in days
        event_df['days_diff'] = (event_df['next_timestamp'] - event_df['timestamp']).dt.total_seconds() / (24 * 3600)
        
        # Calculate price change
        event_df['price_change'] = (event_df['next_price'] - event_df['price']) / event_df['price']
        
        # Mark as should wait if price dropped by threshold within days_to_check
        wait_mask = (event_df['days_diff'] <= days_to_check) & (event_df['price_change'] <= -price_drop_threshold)
        df.loc[event_mask, 'should_wait'] = wait_mask.astype(int)
    
    return df

def create_features(df):
    """
    Create relevant features for the model
    """
    print("\nCreating features...")
    
    # Time-based features
    df['event_date'] = pd.to_datetime(df['event_date'], utc=True)
    df['days_until_event'] = (df['event_date'] - df['timestamp']).dt.days
    df['is_weekend'] = df['event_date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Price-based features
    df['price_per_day'] = df['price'] / df['days_until_event']
    
    # Event popularity features
    print("Calculating event popularity...")
    event_counts = df['event_name'].value_counts()
    df['event_popularity'] = df['event_name'].map(event_counts)
    
    # Zone-based features
    print("Creating zone-based features...")
    zone_dummies = pd.get_dummies(df['standardized_zone'], prefix='zone')
    df = pd.concat([df, zone_dummies], axis=1)
    
    # Binary flag for General Admission Floor
    df['is_ga_floor'] = (df['standardized_zone'] == 'General Admission Floor').astype(int)
    
    return df

def main():
    # Load the data
    print("Loading data...")
    file_path = 'Msg_Floor_10Prc.csv'
    df = pd.read_csv(file_path)
    print("\nColumns in the DataFrame:")
    print(df.columns.tolist())
    
    # Calculate price changes and create target variable
    df = calculate_price_changes(df, days_to_check=3, price_drop_threshold=0.10)
    
    # Create features
    df = create_features(df)
    
    # Print class distribution
    wait_ratio = df['should_wait'].mean()
    print(f"\nClass Distribution:")
    print(f"Wait (1): {wait_ratio:.2%}")
    print(f"Buy Now (0): {(1-wait_ratio):.2%}")
    
    # Save prepared data
    print("\nSaving prepared data...")
    df.to_csv('prepared_price_prediction.csv', index=False)
    print("Prepared data saved to 'prepared_price_prediction.csv'")
    
    # Print sample of the prepared data
    print("\nSample of prepared data:")
    print(df[['event_name', 'timestamp', 'price', 'should_wait', 'days_until_event', 'is_ga_floor']].head())

if __name__ == "__main__":
    main() 