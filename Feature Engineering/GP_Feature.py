import pandas as pd
import re

# Load the original CSV file with messages
file_path = r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Data Preprocessing\output_withnames.csv'
df = pd.read_csv(file_path)

# Define the suffixes of interest for gender-preferential features
suffixes = ['able', 'al', 'ful', 'ible', 'ic', 'ive', 'less', 'ly', 'ous', 'sorry']

# Define a function to check for the presence of gender-preferential features in a message
def has_gender_preferential_features(text):
    words = re.findall(r'\b\w+\b', str(text).lower())  # Tokenize the text into words
    for word in words:
        if any(word.endswith(suffix) for suffix in suffixes):
            return 1
    return 0

# Apply the function to each message and create a new column 'has_features'
df['has_features'] = df['Message'].apply(has_gender_preferential_features)

# Save the messages and binary feature values to a new CSV file
output_file_path = "messages_features_binary.csv"
df[['Message', 'has_features']].to_csv(output_file_path, index=False)

print("Output CSV file 'messages_features_binary.csv' saved successfully.")
