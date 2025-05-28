import pandas as pd
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_display_negative_days():
    """Load the processed data and display cases with negative days until event."""
    try:
        # Load the data
        df = pd.read_csv("processed_data_with_features.csv")
        
        # Filter for negative days
        negative_cases = df[df['days_until_event'] < 0].copy()
        
        # Sort by days_until_event to see most negative first
        negative_cases = negative_cases.sort_values('days_until_event')
        
        # Select columns for display
        display_df = negative_cases[[
            'event_name', 
            'venue', 
            'timestamp',  # Using original timestamp
            'event_date',
            'days_until_event'
        ]]
        
        # Set display options
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)
        
        print(f"\nFound {len(negative_cases)} cases with negative days until event:")
        print("=" * 120)
        print(display_df.to_string(index=False))
        print("=" * 120)
        
        # Print summary statistics
        print("\nSummary of negative days:")
        print(negative_cases['days_until_event'].describe())
        
        # Print venue distribution
        print("\nDistribution by venue:")
        print(negative_cases['venue'].value_counts())
        
    except Exception as e:
        logger.error(f"Error loading or displaying data: {str(e)}")

if __name__ == "__main__":
    load_and_display_negative_days() 