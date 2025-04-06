import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.mixture import GaussianMixture

# Your existing code for data loading and preprocessing stays the same

# After preprocessing and train-test split, add the GMM implementation

# Original SMOTE section
print("Before Resampling")
print(pd.Series(Y_train).value_counts())

# applying SMOTE
smote = SMOTE(random_state=42)
X_train_smote, Y_train_smote = smote.fit_resample(X_train_preprocessed, Y_train)

# after SMOTE
print("After SMOTE")
print(pd.Series(Y_train_smote).value_counts())

# GMM-based oversampling
def gmm_oversampling(X, y, minority_class=1, n_components=3):
    """
    Performs oversampling using Gaussian Mixture Models.
    
    Parameters:
    X: Feature matrix
    y: Target labels
    minority_class: The class to oversample
    n_components: Number of GMM components
    
    Returns:
    X_resampled, y_resampled: Resampled dataset
    """
    # Separate majority and minority classes
    X_minority = X[y == minority_class]
    X_majority = X[y != minority_class]
    
    # Count samples to determine how many to generate
    n_minority = X_minority.shape[0]
    n_majority = X_majority.shape[0]
    n_to_generate = n_majority - n_minority
    
    # Fit GMM to the minority class
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_minority)
    
    # Generate new samples
    X_new, _ = gmm.sample(n_to_generate)
    
    # Combine with original data
    X_resampled = np.vstack((X, X_new))
    y_resampled = np.hstack((y, np.full(n_to_generate, minority_class)))
    
    return X_resampled, y_resampled

# Apply GMM oversampling
X_train_gmm, Y_train_gmm = gmm_oversampling(X_train_preprocessed, Y_train, minority_class=0)

print("After GMM Oversampling")
print(pd.Series(Y_train_gmm).value_counts())

# You'll need to update your model evaluation section to include GMM results

# Modified evaluation function to accept a dataset name parameter
def evaluatemodel(model, X_train, Y_train, X_test, Y_test, model_name, dataset_name=""):
    full_model_name = f"{model_name} ({dataset_name})" if dataset_name else model_name
    
    # train
    model.fit(X_train, Y_train)
    # predict
    y_pred = model.predict(X_test)
    
    # metrics
    acc = accuracy_score(Y_test, y_pred)
    prec = precision_score(Y_test, y_pred)
    rec = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    
    print(f"\n{full_model_name} Performance:")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {f1}")
    
    # Cross validation
    model_cv = model.__class__(**model.get_params())
    cv_scores = cross_val_score(model_cv, X_train, Y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV Score: {np.mean(cv_scores)}")
    
    # Confusion Matrix
    cm = confusion_matrix(Y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {full_model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'confusion_matrix_{full_model_name.replace(" ", "_").lower()}.png')
    plt.close()
    
    # ROC curve
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(Y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {full_model_name}')
        plt.legend(loc="lower right")
        plt.savefig(f'roc_curve_{full_model_name.replace(" ", "_").lower()}.png')
        plt.close()
        
    # classification report
    print("\nClassification Report:")
    print(classification_report(Y_test, y_pred))
    
    return model, acc, prec, rec, f1

# Update your model evaluation section to include GMM
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}

results = {}

print("____________________Models Performance on Original Data___________________________")
original_results = {}

for name, model in models.items():
    model, acc, prec, rec, f1 = evaluatemodel(model, X_train_preprocessed, Y_train, X_test_preprocessed, Y_test, name, "original")
    original_results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1Score': f1}
    
print("___________________Models Performance after SMOTE____________________")
smote_results = {}
for name, model in models.items():
    model, acc, prec, rec, f1 = evaluatemodel(model, X_train_smote, Y_train_smote, X_test_preprocessed, Y_test, name, "SMOTE")
    smote_results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1Score": f1}

print("___________________Models Performance after GMM Oversampling____________________")
gmm_results = {}
for name, model in models.items():
    model, acc, prec, rec, f1 = evaluatemodel(model, X_train_gmm, Y_train_gmm, X_test_preprocessed, Y_test, name, "GMM")
    gmm_results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1Score": f1}
    
original_df = pd.DataFrame(original_results)
smote_df = pd.DataFrame(smote_results)
gmm_df = pd.DataFrame(gmm_results)

print("Original Data Results")
print(original_df)
print("SMOTE-Resampled Data Results")
print(smote_df)
print("GMM-Resampled Data Results")
print(gmm_df)

# Update the visualization code to include GMM results
metrics = ['Accuracy', 'Precision', 'Recall', 'F1Score']
model_names = list(models.keys())

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    original_values = [original_results[model][metric] for model in model_names]
    smote_values = [smote_results[model][metric] for model in model_names]
    gmm_values = [gmm_results[model][metric] for model in model_names]
    
    x = np.arange(len(model_names))
    width = 0.25

    axes[i].bar(x - width, original_values, width, label='Original')
    axes[i].bar(x, smote_values, width, label='SMOTE')
    axes[i].bar(x + width, gmm_values, width, label='GMM')

    axes[i].set_title(f'Comparison of {metric}')
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(model_names)
    axes[i].set_ylim(0, 1)
    axes[i].legend()
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('metrics_comparison.png')
plt.close()

# Update the conclusion section to include GMM results
print("______________Conclusion_________________")
print("Best model based on:")

best_original_f1 = max(original_results.items(), key=lambda x: x[1]['F1Score'])
best_smote_f1 = max(smote_results.items(), key=lambda x: x[1]['F1Score'])
best_gmm_f1 = max(gmm_results.items(), key=lambda x: x[1]['F1Score'])

best_original_acc = max(original_results.items(), key=lambda x: x[1]['Accuracy'])
best_smote_acc = max(smote_results.items(), key=lambda x: x[1]['Accuracy'])
best_gmm_acc = max(gmm_results.items(), key=lambda x: x[1]['Accuracy'])

best_original_prec = max(original_results.items(), key=lambda x: x[1]['Precision'])
best_smote_prec = max(smote_results.items(), key=lambda x: x[1]['Precision'])
best_gmm_prec = max(gmm_results.items(), key=lambda x: x[1]['Precision'])

best_original_rec = max(original_results.items(), key=lambda x: x[1]['Recall'])
best_smote_rec = max(smote_results.items(), key=lambda x: x[1]['Recall'])
best_gmm_rec = max(gmm_results.items(), key=lambda x: x[1]['Recall'])

print("_______________BASED ON PRECISION_______________")
print(f"Original data: {best_original_prec[0]} with Precision of {best_original_prec[1]['Precision']}")
print(f"SMOTE data: {best_smote_prec[0]} with Precision of {best_smote_prec[1]['Precision']}")
print(f"GMM data: {best_gmm_prec[0]} with Precision of {best_gmm_prec[1]['Precision']}")

print("_______________BASED ON F1 SCORE_______________")
print(f"Original data: {best_original_f1[0]} with F1 Score of {best_original_f1[1]['F1Score']}")
print(f"SMOTE data: {best_smote_f1[0]} with F1 Score of {best_smote_f1[1]['F1Score']}")
print(f"GMM data: {best_gmm_f1[0]} with F1 Score of {best_gmm_f1[1]['F1Score']}")

print("_______________BASED ON ACCURACY_______________")
print(f"Original data: {best_original_acc[0]} with Accuracy of {best_original_acc[1]['Accuracy']}")
print(f"SMOTE data: {best_smote_acc[0]} with Accuracy of {best_smote_acc[1]['Accuracy']}")
print(f"GMM data: {best_gmm_acc[0]} with Accuracy of {best_gmm_acc[1]['Accuracy']}")

print("_______________BASED ON RECALL_______________")
print(f"Original data: {best_original_rec[0]} with Recall of {best_original_rec[1]['Recall']}")
print(f"SMOTE data: {best_smote_rec[0]} with Recall of {best_smote_rec[1]['Recall']}")
print(f"GMM data: {best_gmm_rec[0]} with Recall of {best_gmm_rec[1]['Recall']}")

# Find overall best model across all resampling techniques
all_f1_scores = [
    ("Original " + best_original_f1[0], best_original_f1[1]['F1Score']),
    ("SMOTE " + best_smote_f1[0], best_smote_f1[1]['F1Score']),
    ("GMM " + best_gmm_f1[0], best_gmm_f1[1]['F1Score'])
]

overall_best = max(all_f1_scores, key=lambda x: x[1])
print("\n_______________OVERALL BEST MODEL_______________")
print(f"The best overall model is {overall_best[0]} with F1 Score of {overall_best[1]}")
