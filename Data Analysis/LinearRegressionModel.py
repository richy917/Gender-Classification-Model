import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score
from itertools import combinations
import numpy as np

# Load the training CSV file for the Linear Regression Classifier into a DataFrame
train_df_lr = pd.read_csv(r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Splitting Data\Splitting Data_70.csv')

# Load the testing CSV file for the Linear Regression Classifier into a DataFrame
test_df_lr = pd.read_csv(r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Splitting Data\Splitting Data_30.csv')

# Drop NaN values from the DataFrames
train_df_lr.dropna(inplace=True)
test_df_lr.dropna(inplace=True)

# Exclude the last feature from the features to test for the Linear Regression Classifier
features_to_test_lr = train_df_lr.columns[0:-1]  # Exclude the first and last columns

# Initialize the Linear Regression Classifier
clf_lr = LogisticRegression(random_state=42)

# Initialize KFold cross-validation with 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize confusion matrix
conf_matrix = np.zeros((2, 2), dtype=int)

# Open a text file to write the output
with open('linear_regression_output.txt', 'w') as file:

    # Iterate through each feature combination and test its impact on F1, Sn, Sp, and AUC for the Linear Regression Classifier
    combination_num = 1
    for i in range(1, len(features_to_test_lr) + 1):
        # Generate combinations of features for the Linear Regression Classifier
        feature_combinations_lr = combinations(features_to_test_lr, i)

        for selected_features_lr in feature_combinations_lr:
            selected_features_lr = list(selected_features_lr)

            # Initialize lists to store F1, Sn, Sp, and AUC scores for each fold
            f1_scores_lr = []
            sn_scores_lr = []
            sp_scores_lr = []
            auc_scores_lr = []

            # Perform KFold cross-validation
            for train_index, test_index in kf.split(train_df_lr):
                X_train_lr, X_test_lr = train_df_lr.iloc[train_index][selected_features_lr], train_df_lr.iloc[test_index][selected_features_lr]
                y_train_lr, y_test_lr = train_df_lr.iloc[train_index]['gender_label'], train_df_lr.iloc[test_index]['gender_label']

                clf_lr.fit(X_train_lr, y_train_lr)
                y_pred_proba_lr = clf_lr.predict_proba(X_test_lr)[:, 1]
                y_pred_lr = clf_lr.predict(X_test_lr)

                # Calculate F1, Sn, Sp, and AUC scores for the current fold
                f1_lr = f1_score(y_test_lr, y_pred_lr)
                sn_lr = recall_score(y_test_lr, y_pred_lr)
                sp_lr = precision_score(y_test_lr, y_pred_lr)
                auc_lr = roc_auc_score(y_test_lr, y_pred_proba_lr)

                # Append F1, Sn, Sp, and AUC scores to lists
                f1_scores_lr.append(f1_lr)
                sn_scores_lr.append(sn_lr)
                sp_scores_lr.append(sp_lr)
                auc_scores_lr.append(auc_lr)

                # Update confusion matrix for the current fold
                conf_matrix += confusion_matrix(y_test_lr, y_pred_lr)

            # Calculate the mean F1, Sn, Sp, and AUC scores for the current feature combination
            mean_f1_lr = sum(f1_scores_lr) / len(f1_scores_lr)
            mean_sn_lr = sum(sn_scores_lr) / len(sn_scores_lr)
            mean_sp_lr = sum(sp_scores_lr) / len(sp_scores_lr)
            mean_auc_lr = sum(auc_scores_lr) / len(auc_scores_lr)

            # Write F1, Sn, Sp, and AUC scores for the current feature combination to the text file
            file.write(f"Linear Regression - Feature combination {combination_num}: {selected_features_lr}\n")
            file.write(f"F1 Score: {mean_f1_lr:.2f}\n")
            file.write(f"Sensitivity (Sn): {mean_sn_lr:.2f}\n")
            file.write(f"Specificity (Sp): {mean_sp_lr:.2f}\n")
            file.write(f"AUC Score: {mean_auc_lr:.2f}\n\n")

            combination_num += 1

    # Write confusion matrix to the text file
    file.write("\nConfusion Matrix for Linear Regression Classifier:\n")
    file.write(str(conf_matrix))
