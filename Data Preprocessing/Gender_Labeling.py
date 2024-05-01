import pandas as pd

# Read female names from female.txt
with open(r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Data Preprocessing\female.txt', 'r') as f_female:
    female_names = [name.strip() for name in f_female.readlines()]

# Read male names from male.txt
with open(r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Data Preprocessing\male.txt', 'r') as f_male:
    male_names = [name.strip() for name in f_male.readlines()]

# Read output.csv into a DataFrame
df_output = pd.read_csv(r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Data Extraction\output.csv')

# Check the actual column names in df_output
print(df_output.columns)

# Define a function to label names as 1 for female and 0 for male
def label_gender(name):
    if name in female_names:
        return 1  # Female
    elif name in male_names:
        return 0  # Male
    else:
        return -1  # Unknown gender, you can handle this case as needed

# Apply the label_gender function to the correct column in df_output
df_output['gender_label'] = df_output['User'].apply(label_gender)

# Exclude rows with gender label -1 (unknown gender)
df_labeled = df_output[df_output['gender_label'] != -1]

# Create a new DataFrame with only the correct columns
df_labeled = df_labeled[['Message', 'gender_label']]

# Save the labeled DataFrame to a new CSV file
output_file_path = r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Data Preprocessing\output_withgender.csv'
df_labeled.to_csv(output_file_path, index=False)

print(f"Gender-labeled data saved to {output_file_path} successfully.")
