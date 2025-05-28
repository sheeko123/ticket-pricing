import pandas as pd
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
import pytz

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_processed_data(file_path: str = "processed_data.csv") -> Optional[pd.DataFrame]:
    """
    Load the processed ticket data from CSV.
    
    Args:
        file_path (str): Path to the processed data CSV file
        
    Returns:
        pd.DataFrame: Loaded DataFrame or None if loading fails
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path, parse_dates=['event_date'])
        logger.info(f"Successfully loaded {len(df)} rows")
        return df
    except FileNotFoundError:
        logger.error(f"Could not find file: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None

def standardize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize timestamps to EST timezone and ensure consistent format.
    
    Args:
        df (pd.DataFrame): DataFrame with timestamp column
        
    Returns:
        pd.DataFrame: DataFrame with standardized timestamps
    """
    # Convert timestamp to datetime and set to EST
    est = pytz.timezone('US/Eastern')
    
    # Convert listing timestamp to datetime and localize to EST
    df['listing_time'] = pd.to_datetime(df['timestamp'], format="%m-%d-%y / %I:%M %p")
    df['listing_time'] = df['listing_time'].dt.tz_localize(est, ambiguous='NaT')
    
    # Ensure event_date is also in EST
    df['event_date'] = df['event_date'].dt.tz_localize(est, ambiguous='NaT')
    
    # Remove any rows with NaT (ambiguous DST times)
    df = df.dropna(subset=['listing_time', 'event_date'])
    
    return df

def calculate_days_until_event(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate days until event with proper handling of same-day listings.
    Preserves negative values for non-same-day listings for inspection.
    
    Args:
        df (pd.DataFrame): DataFrame with listing_time and event_date columns
        
    Returns:
        pd.DataFrame: DataFrame with days_until_event column
    """
    try:
        # Calculate time difference in seconds
        time_diff = (df["event_date"] - df["listing_time"]).dt.total_seconds()
        
        # Convert to days and round down
        df["days_until_event"] = (time_diff / (24 * 3600)).astype(int)
        
        # Handle only same-day listings (where time_diff is negative but within 24 hours)
        same_day_mask = (time_diff < 0) & (time_diff > -24 * 3600)
        if same_day_mask.any():
            logger.warning(f"Found {same_day_mask.sum()} same-day listings")
            df.loc[same_day_mask, "days_until_event"] = 0
            
        # Log other negative values for inspection
        other_negative = (df["days_until_event"] < 0) & ~same_day_mask
        if other_negative.any():
            logger.warning(f"Found {other_negative.sum()} listings with negative days that need inspection")
            # Log some examples for inspection
            sample_negative = df[other_negative].sample(min(5, other_negative.sum()))
            logger.warning("Sample of listings with negative days:")
            for _, row in sample_negative.iterrows():
                logger.warning(f"Event: {row['event_name']}, Date: {row['event_date']}, "
                             f"Listing: {row['listing_time']}, Days: {row['days_until_event']}")
            
        return df
        
    except Exception as e:
        logger.error(f"Error calculating days until event: {str(e)}")
        raise

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer temporal features from the ticket data.
    
    Args:
        df (pd.DataFrame): Input DataFrame with timestamp and event_date columns
        
    Returns:
        pd.DataFrame: DataFrame with added features
    """
    try:
        # Create a copy to avoid modifying the original
        df_features = df.copy()
        
        # Standardize timestamps
        logger.info("Standardizing timestamps to EST...")
        df_features = standardize_timestamps(df_features)
        
        # Calculate days until event
        logger.info("Calculating days until event...")
        df_features = calculate_days_until_event(df_features)
        
        # Add temporal features
        logger.info("Adding temporal features...")
        df_features["day_of_week"] = df_features["event_date"].dt.dayofweek  # 0=Monday
        df_features["is_weekend_event"] = df_features["day_of_week"].isin([4,5,6]).astype(int)
        
        # Add month and year features
        df_features["event_month"] = df_features["event_date"].dt.month
        df_features["event_year"] = df_features["event_date"].dt.year
        
        logger.info("Feature engineering completed successfully")
        return df_features
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise

def main():
    """Main function to demonstrate the feature engineering process."""
    # Load the data
    df = load_processed_data()
    if df is None:
        return
    
    # Engineer features
    try:
        df_features = engineer_features(df)
        
        # Display some basic statistics
        print("\nFeature Statistics:")
        print(f"Total rows: {len(df_features)}")
        print("\nDays until event statistics:")
        print(df_features["days_until_event"].describe())
        print("\nWeekend events percentage:")
        print(f"{df_features['is_weekend_event'].mean()*100:.1f}%")
        
        # Save the enhanced dataset
        output_path = "processed_data_with_features.csv"
        df_features.to_csv(output_path, index=False)
        logger.info(f"Saved enhanced dataset to {output_path}")
        
        # Filter and save MSG data
        msg_data = df_features[df_features['venue'].str.lower() == 'msg'].copy()
        msg_output_path = "msg_with_features.csv"
        msg_data.to_csv(msg_output_path, index=False)
        logger.info(f"Saved MSG data ({len(msg_data)} rows) to {msg_output_path}")
        
        # Save 10% random sample
        sample_data = df_features.sample(frac=0.1, random_state=42)
        sample_output_path = "sample_10_percent.csv"
        sample_data.to_csv(sample_output_path, index=False)
        logger.info(f"Saved 10% sample ({len(sample_data)} rows) to {sample_output_path}")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")

if __name__ == "__main__":
    main()
