import pandas as pd

try:
    df = pd.read_csv('processed_data_with_features.csv')
    print("Columns in the file:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
except Exception as e:
    print(f"Error reading file: {str(e)}") 