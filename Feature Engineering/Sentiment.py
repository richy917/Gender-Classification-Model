import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the CSV file containing messages
file_path = r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Data Preprocessing\output_withnames.csv'
df = pd.read_csv(file_path)

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define a function to calculate sentiment labels using VADER
def get_sentiment_label(message):
    if isinstance(message, str):  # Check if message is a string
        sentiment = analyzer.polarity_scores(message)
        if sentiment['compound'] > 0:  # Positive sentiment
            return 1
        else:  # Negative or neutral sentiment
            return 0
    else:
        return -1  # Return -1 for non-string values (optional, you can handle this case differently)

# Apply the sentiment analysis function to each message
df['sentiment_label'] = df['Message'].apply(get_sentiment_label)

# Save the DataFrame with sentiment labels to a new CSV file
output_file_path = "messages_with_sentiment_labels.csv"
df[['Message', 'sentiment_label']].to_csv(output_file_path, index=False)

print("Sentiment labels added and CSV file saved successfully.")
