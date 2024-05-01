import pandas as pd

# Load the messages CSV file
messages_file_path = r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Data Preprocessing\output_withnames.csv'
df_messages = pd.read_csv(messages_file_path)

# Load the text file containing words separated by commas
words_file_path = r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Feature Engineering\Word Classes.txt'
with open(words_file_path, 'r') as file:
    words_list = file.read().split(',')

# Function to check if a message contains any of the words from the list
def contains_words(message):
    if isinstance(message, str):  # Check if the message is a string
        for word in words_list:
            if word.strip() in message:
                return 1
    return 0

# Apply the function to the 'message' column and create a new column 'contains_words'
df_messages['contains_words'] = df_messages['Message'].apply(contains_words)

# Save the messages along with the 'contains_words' column to a new CSV file
output_file_path = "messages_contains_words.csv"
df_messages[['Message', 'contains_words']].to_csv(output_file_path, index=False)

print("Output CSV file 'messages_contains_words.csv' saved successfully.")
