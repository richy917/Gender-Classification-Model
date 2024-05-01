import pandas as pd

# Define the file paths
file_paths = [
    r"C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Feature Engineering\messages_contains_words.csv",
    r"C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Feature Engineering\messages_exact_ascii.csv",
    r"C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Feature Engineering\messages_features_binary.csv",
    r"C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Feature Engineering\messages_longtext.csv",
    r"C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Feature Engineering\messages_with_sentiment_labels.csv",
    r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Data Preprocessing\output_withgender.csv',
]

# Initialize an empty list to store dataframes
dfs = []

# Read and append each CSV file to the list
for file_path in file_paths:
    df = pd.read_csv(file_path)
    dfs.append(df)

# Concatenate all dataframes horizontally
combined_df = pd.concat(dfs, axis=1)

# Drop the 'Message' column if it exists
if 'Message' in combined_df.columns:
    combined_df.drop('Message', axis=1, inplace=True)

# Save the combined dataframe to a new CSV file
output_file_path = r"combined_messages.csv"
combined_df.to_csv(output_file_path, index=False)

print("Combined CSV file 'combined_messages.csv' saved successfully.")
