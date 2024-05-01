import pandas as pd

# Load the original CSV file with messages
file_path = r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Data Preprocessing\output_withnames.csv'
df = pd.read_csv(file_path)

# Define a function to label messages as 1 if they are over 20 words long, and 0 otherwise
def label_long_text(text):
    word_count = len(str(text).split())
    if word_count > 20:
        return 1
    else:
        return 0

# Apply the label_long_text function to the 'Message' column and create a new column 'is_long_text'
df['is_long_text'] = df['Message'].apply(label_long_text)

# Save the messages and the 'is_long_text' labels to a new CSV file
output_file_path = "messages_longtext.csv"
df[['Message', 'is_long_text']].to_csv(output_file_path, index=False)

print("Output CSV file 'messages_longtext.csv' saved successfully.")
