import pandas as pd

def analyze_missing_zones(file_path):
    """
    Analyze and display rows with missing zone values
    """
    print(f"Loading data from {file_path}...")
    
    # Load the data
    df = pd.read_csv(file_path)
    
    # Print available columns
    print("\nAvailable columns in the dataset:")
    print(df.columns.tolist())
    
    # Find rows with missing zone values
    missing_zones = df[df['zone'].isna()]
    
    print(f"\nFound {len(missing_zones)} rows with missing zone values")
    
    if len(missing_zones) > 0:
        print("\nSample of rows with missing zones:")
        print("-" * 80)
        # Display all columns for these rows
        print(missing_zones.head(10))
        
        # Print some basic statistics about these rows
        print("\nPrice statistics for rows with missing zones:")
        if 'price' in df.columns:
            print(missing_zones['price'].describe())
        
        # Check if there are any patterns in the missing data
        print("\nEvent distribution for missing zones:")
        if 'event_name' in df.columns:
            print(missing_zones['event_name'].value_counts().head())
        elif 'event' in df.columns:
            print(missing_zones['event'].value_counts().head())

if __name__ == "__main__":
    file_path = "msg_entertainment_events.csv"
    analyze_missing_zones(file_path) 