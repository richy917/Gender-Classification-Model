import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score
from itertools import combinations
import numpy as np

# Load the training CSV file for the Decision Tree Classifier into a DataFrame
train_df_dt = pd.read_csv(r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Splitting Data\Splitting Data_70.csv')

# Load the testing CSV file for the Decision Tree Classifier into a DataFrame
test_df_dt = pd.read_csv(r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Splitting Data\Splitting Data_30.csv')

# Drop NaN values from the DataFrames
train_df_dt.dropna(inplace=True)
test_df_dt.dropna(inplace=True)

# Exclude the last feature from the features to test for the Decision Tree Classifier
features_to_test_dt = train_df_dt.columns[0:-1]  # Exclude the first and last columns

# Initialize the Decision Tree Classifier
clf_dt = DecisionTreeClassifier(random_state=42)

# Initialize KFold cross-validation with 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize confusion matrix
conf_matrix = np.zeros((2, 2), dtype=int)

# Iterate through each feature combination and test its impact on F1, Sn, Sp, and AUC for the Decision Tree Classifier
combination_num = 1
for i in range(1, len(features_to_test_dt) + 1):
    # Generate combinations of features for the Decision Tree Classifier
    feature_combinations_dt = combinations(features_to_test_dt, i)

    for selected_features_dt in feature_combinations_dt:
        selected_features_dt = list(selected_features_dt)

        # Initialize lists to store F1, Sn, Sp, and AUC scores for each fold
        f1_scores_dt = []
        sn_scores_dt = []
        sp_scores_dt = []
        auc_scores_dt = []

        # Perform KFold cross-validation
        for train_index, test_index in kf.split(train_df_dt):
            X_train_dt, X_test_dt = train_df_dt.iloc[train_index][selected_features_dt], train_df_dt.iloc[test_index][selected_features_dt]
            y_train_dt, y_test_dt = train_df_dt.iloc[train_index]['gender_label'], train_df_dt.iloc[test_index]['gender_label']

            clf_dt.fit(X_train_dt, y_train_dt)
            y_pred_proba_dt = clf_dt.predict_proba(X_test_dt)[:, 1]
            y_pred_dt = clf_dt.predict(X_test_dt)

            # Calculate F1, Sn, Sp, and AUC scores for the current fold
            f1_dt = f1_score(y_test_dt, y_pred_dt)
            sn_dt = recall_score(y_test_dt, y_pred_dt)
            sp_dt = precision_score(y_test_dt, y_pred_dt)
            auc_dt = roc_auc_score(y_test_dt, y_pred_proba_dt)

            # Append F1, Sn, Sp, and AUC scores to lists
            f1_scores_dt.append(f1_dt)
            sn_scores_dt.append(sn_dt)
            sp_scores_dt.append(sp_dt)
            auc_scores_dt.append(auc_dt)

            # Update confusion matrix for the current fold
            conf_matrix += confusion_matrix(y_test_dt, y_pred_dt)

        # Calculate the mean F1, Sn, Sp, and AUC scores for the current feature combination
        mean_f1_dt = sum(f1_scores_dt) / len(f1_scores_dt)
        mean_sn_dt = sum(sn_scores_dt) / len(sn_scores_dt)
        mean_sp_dt = sum(sp_scores_dt) / len(sp_scores_dt)
        mean_auc_dt = sum(auc_scores_dt) / len(auc_scores_dt)

        # Output F1, Sn, Sp, and AUC scores for the current feature combination
        print(f"Decision Tree - Feature combination {combination_num}: {selected_features_dt}")
        print(f"F1 Score: {mean_f1_dt:.2f}")
        print(f"Sensitivity (Sn): {mean_sn_dt:.2f}")
        print(f"Specificity (Sp): {mean_sp_dt:.2f}")
        print(f"AUC Score: {mean_auc_dt:.2f}\n")

        combination_num += 1

# Output confusion matrix
print("\nConfusion Matrix for Decision Tree Classifier:")
print(conf_matrix)
