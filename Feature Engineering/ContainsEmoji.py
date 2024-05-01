import pandas as pd

# Load the ASCII words from Ascii.txt
ascii_file_path = r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Feature Engineering\Ascii.txt'
with open(ascii_file_path, 'r') as ascii_file:
    ascii_words = {word.strip() for word in ascii_file.readlines()}  # Using a set for faster lookups

# Read the CSV file containing messages
file_path = r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Data Preprocessing\output_withnames.csv'
df = pd.read_csv(file_path)

# Drop rows with missing values in the 'Message' column
df = df.dropna(subset=['Message'])

# Define a function to check for exact ASCII words in messages
def has_exact_ascii_word(message):
    words_in_message = set(message.split())
    if any(word in ascii_words for word in words_in_message):
        return 1
    return 0

# Apply the function to each message in the DataFrame
df['has_exact_ascii_word'] = df['Message'].apply(has_exact_ascii_word)

# Save the modified DataFrame to a new CSV file
output_file_path = r'messages_exact_ascii.csv'
df[['Message', 'has_exact_ascii_word']].to_csv(output_file_path, index=False)

print("Output CSV file 'messages_exact_ascii.csv' saved successfully.")
