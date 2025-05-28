import pandas as pd

# Load the filtered tickets file
file_path = 'Msg_Floor_10Prc.csv'

def main():
    try:
        df = pd.read_csv(file_path)
        print("Sample of value counts for each artist (event_name) in the filtered dataset:")
        print(df['event_name'].value_counts())
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 