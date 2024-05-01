import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score
from itertools import combinations
import numpy as np

# Load the training CSV file for the Random Forest Classifier into a DataFrame
train_df_rf = pd.read_csv(r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Splitting Data\Splitting Data_70.csv')

# Load the testing CSV file for the Random Forest Classifier into a DataFrame
test_df_rf = pd.read_csv(r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Splitting Data\Splitting Data_30.csv')

# Drop NaN values from the DataFrames
train_df_rf.dropna(inplace=True)
test_df_rf.dropna(inplace=True)

# Exclude the last feature from the features to test for the Random Forest Classifier
features_to_test_rf = train_df_rf.columns[0:-1]  # Exclude the first and last columns

# Initialize the Random Forest Classifier
clf_rf = RandomForestClassifier(random_state=42)

# Initialize KFold cross-validation with 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize confusion matrix
conf_matrix = np.zeros((2, 2), dtype=int)

# Iterate through each feature combination and test its impact on F1, Sn, Sp, and AUC for the Random Forest Classifier
combination_num = 1
for i in range(1, len(features_to_test_rf) + 1):
    # Generate combinations of features for the Random Forest Classifier
    feature_combinations_rf = combinations(features_to_test_rf, i)

    for selected_features_rf in feature_combinations_rf:
        selected_features_rf = list(selected_features_rf)

        # Initialize lists to store F1, Sn, Sp, and AUC scores for each fold
        f1_scores_rf = []
        sn_scores_rf = []
        sp_scores_rf = []
        auc_scores_rf = []

        # Perform KFold cross-validation
        for train_index, test_index in kf.split(train_df_rf):
            X_train_rf, X_test_rf = train_df_rf.iloc[train_index][selected_features_rf], train_df_rf.iloc[test_index][selected_features_rf]
            y_train_rf, y_test_rf = train_df_rf.iloc[train_index]['gender_label'], train_df_rf.iloc[test_index]['gender_label']

            clf_rf.fit(X_train_rf, y_train_rf)
            y_pred_proba_rf = clf_rf.predict_proba(X_test_rf)[:, 1]
            y_pred_rf = clf_rf.predict(X_test_rf)

            # Calculate F1, Sn, Sp, and AUC scores for the current fold
            f1_rf = f1_score(y_test_rf, y_pred_rf)
            sn_rf = recall_score(y_test_rf, y_pred_rf)
            sp_rf = precision_score(y_test_rf, y_pred_rf)
            auc_rf = roc_auc_score(y_test_rf, y_pred_proba_rf)

            # Append F1, Sn, Sp, and AUC scores to lists
            f1_scores_rf.append(f1_rf)
            sn_scores_rf.append(sn_rf)
            sp_scores_rf.append(sp_rf)
            auc_scores_rf.append(auc_rf)

            # Update confusion matrix for the current fold
            conf_matrix += confusion_matrix(y_test_rf, y_pred_rf)

        # Calculate the mean F1, Sn, Sp, and AUC scores for the current feature combination
        mean_f1_rf = sum(f1_scores_rf) / len(f1_scores_rf)
        mean_sn_rf = sum(sn_scores_rf) / len(sn_scores_rf)
        mean_sp_rf = sum(sp_scores_rf) / len(sp_scores_rf)
        mean_auc_rf = sum(auc_scores_rf) / len(auc_scores_rf)

        # Output F1, Sn, Sp, and AUC scores for the current feature combination
        print(f"Random Forest - Feature combination {combination_num}: {selected_features_rf}")
        print(f"F1 Score: {mean_f1_rf:.2f}")
        print(f"Sensitivity (Sn): {mean_sn_rf:.2f}")
        print(f"Specificity (Sp): {mean_sp_rf:.2f}")
        print(f"AUC Score: {mean_auc_rf:.2f}\n")

        combination_num += 1

# Output confusion matrix
print("\nConfusion Matrix for Random Forest Classifier:")
print(conf_matrix)
