import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from utils import load_dataset, split_dataset
import pickle

print("Loading dataset...")
dataset, _ = load_dataset(input_dir='data/input', repository='gdsc', drug_id=-1)
train_set, test_set = split_dataset(dataset, ratio=0.8)

X_train = train_set['exp_bin']
y_train = train_set['tgt']
X_test = test_set['exp_bin']
y_test = test_set['tgt']

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# --- simulate training steps like CADRE ---
training_steps = [10, 50, 100, 300, 1000, 3000, 5000, 10000]

lr_progression = []

print("\n=== TRAINING LOGISTIC REGRESSION OVER MULTIPLE ITERATIONS ===")

for max_iter in training_steps:
    print(f"Training with max_iter={max_iter}...")

    auc_list, f1_list, acc_list = [], [], []

    # Train separate LR for each drug
    for drug_idx in range(y_train.shape[1]):
        y_train_drug = y_train[:, drug_idx]
        y_test_drug = y_test[:, drug_idx]

        lr = LogisticRegression(max_iter=max_iter, random_state=42)
        lr.fit(X_train, y_train_drug)

        # Predict probabilities
        y_pred_proba = lr.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Metrics
        try:
            auc = roc_auc_score(y_test_drug, y_pred_proba)
        except:
            auc = np.nan

        try:
            f1 = f1_score(y_test_drug, y_pred)
        except:
            f1 = np.nan

        acc = accuracy_score(y_test_drug, y_pred)

        auc_list.append(auc)
        f1_list.append(f1)
        acc_list.append(acc)

    # Aggregate for this step
    lr_progression.append({
        "iter": max_iter,
        "test_auc": float(np.nanmean(auc_list)),
        "test_f1": float(np.nanmean(f1_list)),
        "test_acc": float(np.nanmean(acc_list))
    })

    print(f"AUC={np.nanmean(auc_list):.2%}, F1={np.nanmean(f1_list):.2%}, ACC={np.nanmean(acc_list):.2%}")

# Save curve
with open("data/output/cf/lr_learning_curve.pkl", "wb") as f:
    pickle.dump(lr_progression, f)

print("\nSaved learning curve ➜ data/output/cf/lr_learning_curve.pkl")
