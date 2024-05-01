import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import numpy as np

# Load the training CSV files for each classifier into DataFrames, excluding the "message" column
train_df_dt = pd.read_csv(r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Splitting Data\Splitting Data_70.csv')
train_df_rf = pd.read_csv(r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Splitting Data\Splitting Data_70.csv')
train_df_lr = pd.read_csv(r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Splitting Data\Splitting Data_70.csv')

# Load the testing CSV files for each classifier into DataFrames, excluding the "message" column
test_df_dt = pd.read_csv(r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Splitting Data\Splitting Data_30.csv')
test_df_rf = pd.read_csv(r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Splitting Data\Splitting Data_30.csv')
test_df_lr = pd.read_csv(r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Splitting Data\Splitting Data_30.csv')

# Drop NaN values from the DataFrames
train_df_dt.dropna(inplace=True)
train_df_rf.dropna(inplace=True)
train_df_lr.dropna(inplace=True)
test_df_dt.dropna(inplace=True)
test_df_rf.dropna(inplace=True)
test_df_lr.dropna(inplace=True)

# Exclude the last feature from the features to test for each classifier
features_to_test_dt = train_df_dt.columns[0:-1]  # Exclude the first and last columns
features_to_test_rf = train_df_rf.columns[0:-1]  # Exclude the first and last columns
features_to_test_lr = train_df_lr.columns[0:-1]  # Exclude the first and last columns


# Initialize classifiers
clf_dt = DecisionTreeClassifier(random_state=42)
clf_rf = RandomForestClassifier(random_state=42)
clf_lr = LogisticRegression(random_state=42)

# Initialize KFold cross-validation with 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store AUC scores and true labels, predicted probabilities for each classifier
auc_scores_dt = []
true_labels_dt = []
predicted_probabilities_dt = []
auc_scores_rf = []
true_labels_rf = []
predicted_probabilities_rf = []
auc_scores_lr = []
true_labels_lr = []
predicted_probabilities_lr = []

# Iterate through each feature combination and test its impact on AUC for each classifier
for i in range(1, len(features_to_test_dt) + 1):
    # Generate combinations of features for each classifier
    feature_combinations_dt = combinations(features_to_test_dt, i)
    feature_combinations_rf = combinations(features_to_test_rf, i)
    feature_combinations_lr = combinations(features_to_test_lr, i)

    for selected_features_dt, selected_features_rf, selected_features_lr in zip(feature_combinations_dt,
                                                                                 feature_combinations_rf,
                                                                                 feature_combinations_lr):
        selected_features_dt = list(selected_features_dt)
        selected_features_rf = list(selected_features_rf)
        selected_features_lr = list(selected_features_lr)

        # Decision Tree Classifier
        X_train_dt = train_df_dt[selected_features_dt]
        y_train_dt = train_df_dt['gender_label']
        X_test_dt = test_df_dt[selected_features_dt]
        y_test_dt = test_df_dt['gender_label']

        clf_dt.fit(X_train_dt, y_train_dt)
        y_pred_proba_dt = clf_dt.predict_proba(X_test_dt)[:, 1]
        auc_score_dt = roc_auc_score(y_test_dt, y_pred_proba_dt)
        auc_scores_dt.append(auc_score_dt)
        true_labels_dt.extend(y_test_dt)
        predicted_probabilities_dt.extend(y_pred_proba_dt)

        # Random Forest Classifier
        X_train_rf = train_df_rf[selected_features_rf]
        y_train_rf = train_df_rf['gender_label']
        X_test_rf = test_df_rf[selected_features_rf]
        y_test_rf = test_df_rf['gender_label']

        clf_rf.fit(X_train_rf, y_train_rf)
        y_pred_proba_rf = clf_rf.predict_proba(X_test_rf)[:, 1]
        auc_score_rf = roc_auc_score(y_test_rf, y_pred_proba_rf)
        auc_scores_rf.append(auc_score_rf)
        true_labels_rf.extend(y_test_rf)
        predicted_probabilities_rf.extend(y_pred_proba_rf)

        # Logistic Regression Classifier
        X_train_lr = train_df_lr[selected_features_lr]
        y_train_lr = train_df_lr['gender_label']
        X_test_lr = test_df_lr[selected_features_lr]
        y_test_lr = test_df_lr['gender_label']

        clf_lr.fit(X_train_lr, y_train_lr)
        y_pred_proba_lr = clf_lr.predict_proba(X_test_lr)[:, 1]
        auc_score_lr = roc_auc_score(y_test_lr, y_pred_proba_lr)
        auc_scores_lr.append(auc_score_lr)
        true_labels_lr.extend(y_test_lr)
        predicted_probabilities_lr.extend(y_pred_proba_lr)

# Compute ROC curves for each classifier
fpr_dt, tpr_dt, _ = roc_curve(true_labels_dt, predicted_probabilities_dt)
fpr_rf, tpr_rf, _ = roc_curve(true_labels_rf, predicted_probabilities_rf)
fpr_lr, tpr_lr, _ = roc_curve(true_labels_lr, predicted_probabilities_lr)

# Plot ROC curves for each classifier
plt.figure(figsize=(8, 6))
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_scores_dt[-1]:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_scores_rf[-1]:.2f})')
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_scores_lr[-1]:.2f})')

# Plot the random guess line
plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random Guess')

# Set labels and title
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curves for Classifiers 70/30')
plt.legend()
plt.grid(True)
plt.show()

# Output mean accuracy and AUC scores for each feature combination for each classifier (optional)
combination_num_dt = 1
combination_num_rf = 1
combination_num_lr = 1
for i in range(1, len(features_to_test_dt) + 1):
    feature_combinations_dt = combinations(features_to_test_dt, i)
    feature_combinations_rf = combinations(features_to_test_rf, i)
    feature_combinations_lr = combinations(features_to_test_lr, i)

    for selected_features_dt, selected_features_rf, selected_features_lr in zip(feature_combinations_dt,
                                                                                 feature_combinations_rf,
                                                                                 feature_combinations_lr):
        selected_features_dt = list(selected_features_dt)
        selected_features_rf = list(selected_features_rf)
        selected_features_lr = list(selected_features_lr)

        print(f'Decision Tree - Feature combination {combination_num_dt}: {selected_features_dt}')
        print(f'AUC with feature combination {combination_num_dt}: {auc_scores_dt[combination_num_dt-1]:.2f}')
        print('-' * 30)
        combination_num_dt += 1

        print(f'Random Forest - Feature combination {combination_num_rf}: {selected_features_rf}')
        print(f'AUC with feature combination {combination_num_rf}: {auc_scores_rf[combination_num_rf-1]:.2f}')
        print('-' * 30)
        combination_num_rf += 1

        print(f'Logistic Regression - Feature combination {combination_num_lr}: {selected_features_lr}')
        print(f'AUC with feature combination {combination_num_lr}: {auc_scores_lr[combination_num_lr-1]:.2f}')
        print('-' * 30)
        combination_num_lr += 1

