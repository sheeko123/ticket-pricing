import pandas as pd
import numpy as np

# Read the CSV file
file_path = 'etl/msg/msg_filtered_tickets_cleaned.csv'

try:
    # Read the entire CSV file
    df = pd.read_csv(file_path)
    
    # Function to get lowest 10% of tickets for each event
    def get_lowest_10_percent(group):
        return group.nsmallest(int(len(group) * 0.1), 'price')
    
    # Collect DataFrames for floor/GA and lowest 10% tickets
    floor_ga_list = []
    lowest_10_list = []
    
    for event, group in df.groupby('event_name'):
        # Floor/GA tickets
        floor_ga = group[group['standardized_zone'].str.contains('Floor|General Admission', case=False, na=False)]
        floor_ga_list.append(floor_ga)
        # Lowest 10% tickets
        lowest_10 = get_lowest_10_percent(group)
        lowest_10_list.append(lowest_10)
    
    # Concatenate and remove duplicates
    floor_ga_df = pd.concat(floor_ga_list)
    lowest_10_df = pd.concat(lowest_10_list)
    combined_df = pd.concat([floor_ga_df, lowest_10_df]).drop_duplicates()
    
    # Save to CSV
    combined_df.to_csv('Msg_Floor_10Prc.csv', index=False)
    print("Saved combined floor/GA and lowest 10% tickets to 'Msg_Floor_10Prc.csv'.")
    
    # Print a sample of the resulting ticket listings
    print("\nSample of combined ticket listings:")
    print(combined_df.head(20))
    
except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
except Exception as e:
    print(f"An error occurred: {str(e)}") 