import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def standardize_zone(zone_name):
    if pd.isna(zone_name):
        return 'Unknown'
        
    zone_name_lower = str(zone_name).lower()
    
    # Handle common patterns and typos
    if 'barstool' in zone_name_lower or 'bar stool' in zone_name_lower:
        return 'Barstool Seating'
    elif 'ga floor' in zone_name_lower or 'floor' in zone_name_lower:
        return 'General Admission Floor'
    elif 'chase bridge' in zone_name_lower or 'bridges' in zone_name_lower:
        return 'Chase Bridge'
    elif 'vip' in zone_name_lower or 'gold vip' in zone_name_lower:
        return 'VIP Packages'
    elif 'club' in zone_name_lower:
        return 'Club'
    elif 'suite' in zone_name_lower:
        return 'Suites'
    elif 'pit' in zone_name_lower:
        return 'Pit'
    elif 'lounge' in zone_name_lower:
        return 'Lounge'
    elif 'riser' in zone_name_lower:
        return 'Riser'
    elif 'level' in zone_name_lower:
        # Standardize "Level X00" to "X00 Level"
        parts = str(zone_name).split()
        if parts[0].lower() == 'level' and parts[1].isdigit():
            return f"{parts[1]} Level"
        return zone_name
    return zone_name

# def extract_section_value(section_value):
#     """Extract and standardize section value."""
#     if pd.isna(section_value):
#         return np.nan
    
#     section_str = str(section_value).strip().upper()
    
#     # If it's just a number, return it
#     if section_str.isdigit():
#         return float(section_str)
    
#     # If it's just letters, convert to number (A=1, B=2, etc.)
#     if section_str.isalpha():
#         return sum(ord(c) - ord('A') + 1 for c in section_str)
    
#     # If it's a mix of letters and numbers, prioritize letters
#     letters = ''.join(c for c in section_str if c.isalpha())
#     if letters:
#         return sum(ord(c) - ord('A') + 1 for c in letters)
    
#     return np.nan

# def extract_row_value(row_value):
#     """Extract and standardize row value."""
#     if pd.isna(row_value):
#         return np.nan
    
#     row_str = str(row_value).strip().upper()
    
#     # If it's just a number, return it
#     if row_str.isdigit():
#         return float(row_str)
    
#     # If it's just letters, convert to number (A=1, B=2, etc.)
#     if row_str.isalpha():
#         return sum(ord(c) - ord('A') + 1 for c in row_str)
    
#     # If it's a mix of letters and numbers, try to extract numbers first
#     numbers = ''.join(c for c in row_str if c.isdigit())
#     if numbers:
#         return float(numbers)
    
#     # If no numbers, use letters
#     letters = ''.join(c for c in row_str if c.isalpha())
#     if letters:
#         return sum(ord(c) - ord('A') + 1 for c in letters)
    
#     return np.nan

# def calculate_seat_scores(df):
#     """
#     Calculate seat scores based on price, row, and section quality.
#     Returns the dataframe with additional score columns.
#     """
#     # Convert section and row to numeric values
#     df['section_numeric'] = df['section'].apply(extract_section_value)
#     df['row_numeric'] = df['row'].apply(extract_row_value)
    
#     # Step 1: Calculate min price per group (zone-section-row)
#     df['min_price'] = df.groupby(['zone', 'section', 'row'])['price'].transform('min')
#     df['price_ratio'] = df['price'] / df['min_price']

#     # Step 2: Row score (lower row = better)
#     df['max_row_per_section'] = df.groupby(['zone', 'section'])['row_numeric'].transform('max')
#     df['row_score'] = (df['max_row_per_section'] - df['row_numeric'] + 1) / df['max_row_per_section']
    
#     # Step 3: Section score (lower section = better)
#     df['max_section_per_zone'] = df.groupby(['zone'])['section_numeric'].transform('max')
#     df['section_score'] = (df['max_section_per_zone'] - df['section_numeric'] + 1) / df['max_section_per_zone']
    
#     # Fill NaN values with middle values
#     df['row_score'] = df['row_score'].fillna(0.5)
#     df['section_score'] = df['section_score'].fillna(0.5)

#     # Step 4: Section rank (by avg price)
#     df['section_avg_price'] = df.groupby(['zone', 'section'])['price'].transform('mean')
#     df['section_rank'] = df.groupby(['zone'])['section_avg_price'].rank(pct=True)

#     # Step 5: Seat score (weighted average)
#     df['seat_score'] = (
#         0.4 * df['price_ratio'] + 
#         0.3 * df['row_score'] + 
#         0.2 * df['section_score'] +
#         0.1 * df['section_rank']
#     )

#     # Normalize to 0-10 scale for interpretability
#     df['seat_score'] = (df['seat_score'] - df['seat_score'].min()) / \
#                        (df['seat_score'].max() - df['seat_score'].min()) * 10
    
#     # Drop temporary columns
#     df = df.drop(['section_numeric', 'row_numeric'], axis=1)
    
#     return df

# def analyze_zones(csv_path):
#     # Read the CSV file
#     df = pd.read_csv(csv_path)
    
#     # Analyze event listings
#     print("\nEvent Listing Analysis:")
#     print("-" * 50)
#     event_counts = df['event_name'].value_counts()
    
#     # Print all events with less than 1000 listings
#     print("\nEvents with fewer than 1000 listings:")
#     print("{:<60} | {:>10}".format('Event Name', 'Listings'))
#     print('-' * 73)
#     for event, count in event_counts.items():
#         if count < 1000:
#             print("{:<60} | {:>10,}".format(str(event)[:60], count))
    
#     # Print total number of events
#     print(f"\nTotal number of events: {len(event_counts)}")
#     print(f"Events with < 1000 listings: {sum(event_counts < 1000)}")
#     print("-" * 50)
    
#     # Print all unique zone values
#     print("\nAll Unique Zone Values:")
#     print("-" * 50)
#     unique_zones = df['zone'].unique()
#     # Convert all values to strings for sorting
#     unique_zones = [str(z) if pd.notna(z) else 'Unknown' for z in unique_zones]
#     for zone in sorted(unique_zones):
#         print(zone)
#     print("-" * 50)
    
#     # Print sample of section and row values
#     print("\nSample Section and Row Values:")
#     print("-" * 50)
#     print("\nUnique Sections:")
#     # Convert to strings and handle NaN values
#     sections = [str(s) if pd.notna(s) else 'Unknown' for s in df['section'].unique()]
#     print(sorted(sections))
#     print("\nUnique Rows:")
#     # Convert to strings and handle NaN values
#     rows = [str(r) if pd.notna(r) else 'Unknown' for r in df['row'].unique()]
#     print(sorted(rows))
#     print("-" * 50)
    
#     # Standardize zones
#     df['standardized_zone'] = df['zone'].apply(standardize_zone)
    
#     # Calculate seat scores
#     df = calculate_seat_scores(df)
    
#     # Count occurrences of each standardized zone
#     zone_counts = df['standardized_zone'].value_counts()
    
#     # Group small categories into 'Other'
#     threshold = 100  # Adjust this threshold as needed
#     small_categories = zone_counts[zone_counts < threshold]
#     large_categories = zone_counts[zone_counts >= threshold]
    
#     # Create final dictionary
#     final = large_categories.to_dict()
#     if len(small_categories) > 0:
#         final['Other'] = small_categories.sum()
    
#     # Sort by count descending
#     sorted_final = dict(sorted(final.items(), key=lambda item: item[1], reverse=True))
    
#     # Print zone analysis results
#     print("\nZone Analysis Results (After Standardization):")
#     print("{:<25} | {:>10}".format('Zone', 'Listings'))
#     print('-' * 38)
#     for zone, count in sorted_final.items():
#         print("{:<25} | {:>10,}".format(zone, count))
    
#     # Print sample of seat scores
#     print("\nSample Seat Scores (first 5 rows):")
#     print(df[['zone', 'section', 'row', 'price', 'seat_score']].head())
    
#     # Create price distribution boxplots
#     print("\nGenerating price distribution boxplots...")
    
#     # Set up the plot style
#     plt.style.use('ggplot')  # Using ggplot style instead of seaborn
#     plt.figure(figsize=(15, 8))
    
#     # Create boxplot
#     sns.boxplot(data=df, x='standardized_zone', y='price', hue='event_name')
    
#     # Customize the plot
#     plt.title('Price Distribution by Zone and Event')
#     plt.xlabel('Zone')
#     plt.ylabel('Price')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
    
#     # Save the plot
#     plt.savefig('price_distribution.png')
#     print("Price distribution plot saved as 'price_distribution.png'")
    
#     # Save results to CSV
#     output_path = 'msg_seat_scores.csv'
#     df.to_csv(output_path, index=False)
#     print(f"\nFull results saved to {output_path}")
    
#     return df

# if __name__ == "__main__":
#     csv_path = "msg_entertainment_events.csv"
#     analyze_zones(csv_path)

#Step 1: Calculate Baseline Prices
# Group by zone, section, and row, then compute:

# min_price: Minimum price in the group.

# price_ratio: price / min_price (normalized to baseline).

# Step 2: Encode Row Quality
# Lower row numbers (e.g., Row 1) are typically better. Convert rows to a score:

# python
# max_row = max(row_numbers_in_section)
# row_score = (max_row - row_number + 1) / max_row  # Normalized to 0-1
# Step 3: Encode Section Quality
# Rank sections by their average price within a zone:

# python
# section_avg_price = df.groupby(['zone', 'section'])['price'].transform('mean')
# section_rank = section_avg_price.rank(pct=True)  # 0-1 percentile
# Step 4: Combine Features into Seat Score
# Use weighted features to create a composite score:

# Seat Score
# =
# (
# 0.5
# ×
# Seat Score=(0.5×

def save_standardized_zones(csv_path):
    """
    Read the CSV file, standardize zones, and save the results to a new CSV file.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Standardize zones
    df['standardized_zone'] = df['zone'].apply(standardize_zone)
    
    # Count occurrences of each standardized zone
    zone_counts = df['standardized_zone'].value_counts()
    
    # Group small categories into 'Other'
    threshold = 100  # Adjust this threshold as needed
    small_categories = zone_counts[zone_counts < threshold]
    large_categories = zone_counts[zone_counts >= threshold]
    
    # Create final dictionary
    final = large_categories.to_dict()
    if len(small_categories) > 0:
        final['Other'] = small_categories.sum()
    
    # Sort by count descending
    sorted_final = dict(sorted(final.items(), key=lambda item: item[1], reverse=True))
    
    # Print zone analysis results
    print("\nZone Analysis Results (After Standardization):")
    print("{:<25} | {:>10}".format('Zone', 'Listings'))
    print('-' * 38)
    for zone, count in sorted_final.items():
        print("{:<25} | {:>10,}".format(zone, count))
    
    # Save results to CSV
    output_path = 'msg_standardized_zones.csv'
    df.to_csv(output_path, index=False)
    print(f"\nFull results saved to {output_path}")
    
    return df

if __name__ == "__main__":
    csv_path = "msg_entertainment_events.csv"
    save_standardized_zones(csv_path)
