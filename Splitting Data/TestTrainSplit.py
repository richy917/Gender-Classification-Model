import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load the CSV file into a DataFrame
df = pd.read_csv(r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Feature Engineering\combined_messages.csv')

# Split the DataFrame into training and testing sets with different splits
splits = [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3)]  # Specify the splits here
random_state = 42  # Set a random state for reproducibility

for train_ratio, test_ratio in splits:
    # Split the DataFrame into training and testing sets
    train_df, test_df = train_test_split(df, test_size=test_ratio, train_size=train_ratio, random_state=random_state)

    # Define the output folder for saving the split datasets
    output_folder = r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Splitting Data'

    # Save the split datasets to new CSV files
    train_filename = os.path.join(output_folder, f'Splitting Data_{int(train_ratio * 100)}.csv')
    test_filename = os.path.join(output_folder, f'Splitting Data_{int(test_ratio * 100)}.csv')

    train_df.to_csv(train_filename, index=False)
    test_df.to_csv(test_filename, index=False)
