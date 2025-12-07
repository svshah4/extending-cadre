import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import pickle
from utils import load_dataset, split_dataset

# Load data
print("Loading dataset...")
dataset, _ = load_dataset(input_dir='data/input', repository='gdsc', drug_id=-1)
train_set, test_set = split_dataset(dataset, ratio=0.8)

# Prepare features (use binary gene expression directly)
X_train = train_set['exp_bin']  # (676, 3000)
y_train = train_set['tgt']      # (676, 260)
X_test = test_set['exp_bin']    # (170, 3000)
y_test = test_set['tgt']        # (170, 260)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Train separate logistic regression for each drug
results = []
for drug_idx in range(y_train.shape[1]):
    y_train_drug = y_train[:, drug_idx]
    y_test_drug = y_test[:, drug_idx]
    
    # Train logistic regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train_drug)
    
    # Predict
    y_pred_proba = lr.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    auc = roc_auc_score(y_test_drug, y_pred_proba)
    f1 = f1_score(y_test_drug, y_pred)
    acc = accuracy_score(y_test_drug, y_pred)
    
    results.append({'drug': drug_idx, 'auc': auc, 'f1': f1, 'acc': acc})
    
    if drug_idx % 50 == 0:
        print(f"Processed {drug_idx}/{y_train.shape[1]} drugs...")

# Aggregate results
df_results = pd.DataFrame(results)
print("\n=== LOGISTIC REGRESSION BASELINE ===")
print(f"Mean AUC: {df_results['auc'].mean():.1%}")
print(f"Mean F1: {df_results['f1'].mean():.1%}")
print(f"Mean Accuracy: {df_results['acc'].mean():.1%}")

# Save results
with open('data/output/cf/baseline_results.pkl', 'wb') as f:
    pickle.dump(df_results, f)