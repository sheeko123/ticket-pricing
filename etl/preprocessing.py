import os
import pandas as pd
from datetime import datetime
import json

# --------------------------------------------
# 1. Load JSON Data & Extract Event Metadata
# --------------------------------------------
def load_json_events(base_path="listings"):
    all_events = []
    
    for venue in os.listdir(base_path):
        venue_path = os.path.join(base_path, venue)
        if not os.path.isdir(venue_path): continue
        
        for file_name in os.listdir(venue_path):
            if not file_name.lower().endswith(".json"): continue
            json_file = os.path.join(venue_path, file_name)
            file_base, _ = os.path.splitext(file_name)
            
            if os.path.exists(json_file):
                # Extract event date and name from filename
                event_date_str = file_base[-6:]
                event_name = file_base[:-6]
                event_date = datetime.strptime(event_date_str, "%y%m%d")
                
                # Load JSON data
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        listings = json.load(f)
                except UnicodeDecodeError:
                    try:
                        with open(json_file, 'r', encoding='latin-1') as f:
                            listings = json.load(f)
                    except Exception as e:
                        print(f"Error reading {json_file}: {str(e)}")
                        continue
                
                df = pd.DataFrame(listings)
                df["event_date"] = event_date
                df["venue"] = venue
                df["event_name"] = event_name
                all_events.append(df)
    
    return pd.concat(all_events, ignore_index=True)

# Load all data
df = load_json_events()
print("\nDataFrame Info:")
print(df.info())
print("\nDataFrame Shape (rows, columns):")
print(df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())

# Save to CSV for quick loading later
print("\nSaving to CSV...")
df.to_csv("processed_data.csv", index=False)
print("Data saved to processed_data.csv")