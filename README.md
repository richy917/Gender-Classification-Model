# Gender-Classification-Model
# Overview 
This repository contains the code and resources for an unsupervised machine learning approach to gender-classification based on Discord chat conversations. 
The dataset used in this project was extracted from the conference paper by Keerthana, et al. titled "DISCO: a dataset of Discord chat conversations for software engineering research."

# Dataset Preparation 
The original XML file was converted into a CSV format using the DiscordMessageExtractor.py python script which leverages the Element Tree XML API from Python. The csv file output is then inputted 
into the Gender_Labeling.py file. This utilizes female.txt and male.txt and labels the data with the associated gender. 

# Feature Engineering
Each python file in the engineering folder acts upon the previous output and creates a new csv file with the corresponding features extracted. After running all five files, CSV Combiner.py combines all the csv's together. 

# Results 
The results of the gender-classification-model, including ROC plots were generated for visual representation and comparison. This model achieved an AUC score of 0.54. 

# Usage 
1. Clone repository to local machine.
2. Set up python environment using the 'requirements.txt' file.
3. Run the scripts in the process described above

# Reference 
For more details on Gender-Classification-Model, the user is refered to the full paper. 

