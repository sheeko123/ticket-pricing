import pandas as pd
import numpy as np
from datetime import datetime
import sys

def load_and_validate_data(file_path):
    """
    Load and validate the CSV data with comprehensive checks
    """
    print(f"Loading data from {file_path}...")
    
    try:
        # Load the data
        df = pd.read_csv(file_path)
        print(f"\nSuccessfully loaded data with {len(df)} rows and {len(df.columns)} columns")
        
        # Basic information
        print("\nDataset Information:")
        print("-" * 50)
        print(df.info())
        
        # Check for missing values
        print("\nMissing Values Analysis:")
        print("-" * 50)
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        missing_info = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage': missing_percentage
        })
        print(missing_info[missing_info['Missing Values'] > 0])
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        print(f"\nNumber of duplicate rows: {duplicates}")
        
        # Basic statistics for numeric columns
        print("\nNumeric Columns Statistics:")
        print("-" * 50)
        print(df.describe())
        
        # Check for potential data quality issues
        print("\nData Quality Checks:")
        print("-" * 50)
        
        # Check for negative values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            neg_values = (df[col] < 0).sum()
            if neg_values > 0:
                print(f"Column '{col}' has {neg_values} negative values")
        
        # Check for date columns and validate them
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        for col in date_cols:
            try:
                pd.to_datetime(df[col])
                print(f"Column '{col}' contains valid dates")
            except:
                print(f"Column '{col}' contains invalid dates")
        
        return df
    
    except Exception as e:
        print(f"Error loading or validating data: {str(e)}")
        return None

if __name__ == "__main__":
    # You can specify the file path as a command line argument
    file_path = sys.argv[1] if len(sys.argv) > 1 else "etl\msg\msg_filtered_tickets_cleaned.csv"
    
    df = load_and_validate_data(file_path)
    
    if df is not None:
        # Check for rows with zero or negative quantity
        invalid_quantity = df[df['quantity'] <= 0]
        print("\nRows with zero or negative quantity:")
        print(f"Found {len(invalid_quantity)} rows with quantity <= 0")
        if len(invalid_quantity) > 0:
            print("\nSample of rows with invalid quantity:")
            print(invalid_quantity[['timestamp', 'zone', 'section', 'price', 'event_name', 'quantity']].head())
        
        # Remove rows with zero or negative quantity
        df_cleaned = df[df['quantity'] > 0].copy()
        print(f"\nRemoved {len(df) - len(df_cleaned)} rows with invalid quantity")
        print(f"Remaining rows: {len(df_cleaned)}")
        
        # Save cleaned data to new CSV
        output_path = file_path.replace('.csv', '_cleaned.csv')
        df_cleaned.to_csv(output_path, index=False)
        print(f"\nSaved cleaned data to: {output_path}")
        print("\nData validation completed successfully!")