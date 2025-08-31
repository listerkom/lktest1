import numpy as np # Import the NumPy library, useful for numerical operations on arrays
import pandas as pd # Import the Pandas library, useful for data manipulation and analysis
import os # Import the os module, useful for interacting with the operating system
from sklearn.model_selection import train_test_split # Import the train_test_split function from scikit-learn, useful for splitting datasets
import yaml # Import the PyYAML library, useful for parsing YAML files


# Function to load data from a given URL into a pandas DataFrame
def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url) # Attempt to read a CSV file from the URL into a DataFrame
        return df # Return the DataFrame if read successfully
    except pd.errors.ParserError as e: # Specific error if the CSV parsing fails
        print(f"Error: Failed to parse the CSV file from {data_url}.") # Print an error message
        print(e) # Print the specific parsing error
        raise # Rethrow the exception to handle it elsewhere if needed
    except Exception as e: # Catch any other unexpected exceptions/errors
        print(f"Error: An unexpected error occurred while loading the data.") # Print generic error message
        print(e)
        raise


# Function to preprocess the loaded data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['tweet_id'], inplace=True) # Remove the 'tweet_id' column, modifies df in place
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])] # Filter rows where 'sentiment' is either 'happiness' or 'sadness'
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True) # Replace 'happiness' with 1 and 'sadness' with 0 in the 'sentiment' column
        return final_df # Return the preprocessed DataFrame
    except KeyError as e:
        print(f"Error: Missing column {e} in the dataframe.")
        raise
    except Exception as e:
        print(f"Error: An unexpected error occurred during preprocessing.")
        print(e)
        raise



# Function to save the train and test data splits to CSV files at a specified path
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        data_path = os.path.join(data_path, 'raw') # Append 'raw' folder to the given path
        os.makedirs(data_path, exist_ok=True) # Create the directory if it doesn't exist
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False) # Save the training data to 'train.csv' without the index
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the data.")
        print(e)
        raise


# Main function coordinating the data loading, preprocessing, splitting, and saving processes
def main():
    try:
        df = load_data(data_url='https://raw.githubusercontent.com/entbappy/Branching-tutorial/refs/heads/master/tweet_emotions.csv')
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)
        save_data(train_data, test_data, data_path='data')
    except Exception as e:
        print(f"Error: {e}")
        print("Failed to complete the data ingestion process.")


# This block ensures the main function runs only when script is executed directly (not imported as a module)
if __name__ == '__main__':
    main()
