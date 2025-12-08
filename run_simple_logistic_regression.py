import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from utils import load_dataset, split_dataset
import pickle
import os

print("Loading dataset...")
dataset, _ = load_dataset(input_dir='data/input', repository='gdsc', drug_id=-1)
train_set, test_set = split_dataset(dataset, ratio=0.8)

X_train = train_set['exp_bin']
y_train = train_set['tgt']
X_test = test_set['exp_bin']
y_test = test_set['tgt']

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Number of drugs: {y_train.shape[1]}")

# Create output directory if it doesn't exist
os.makedirs("data/output/cf", exist_ok=True)

# Training iterations to simulate learning curve
training_steps = [10, 50, 100, 300, 1000, 3000, 5000, 10000]

lr_progression = []

print("\n=== TRAINING LOGISTIC REGRESSION OVER MULTIPLE ITERATIONS ===")

for max_iter in training_steps:
    print(f"\nTraining with max_iter={max_iter}...")
    auc_list, f1_list, acc_list = [], [], []
    
    # Train separate LR for each drug
    for drug_idx in range(y_train.shape[1]):
        y_train_drug = y_train[:, drug_idx]
        y_test_drug = y_test[:, drug_idx]
        
        # Skip drugs with only one class in training or test
        if len(np.unique(y_train_drug)) < 2 or len(np.unique(y_test_drug)) < 2:
            continue
        
        # Train logistic regression
        lr = LogisticRegression(max_iter=max_iter, random_state=42, solver='lbfgs')
        lr.fit(X_train, y_train_drug)
        
        # Predict probabilities
        y_pred_proba = lr.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        try:
            auc = roc_auc_score(y_test_drug, y_pred_proba)
            auc_list.append(auc)
        except:
            pass  # Skip if AUC calculation fails
        
        try:
            f1 = f1_score(y_test_drug, y_pred, zero_division=0)
            f1_list.append(f1)
        except:
            pass  # Skip if F1 calculation fails
        
        try:
            acc = accuracy_score(y_test_drug, y_pred)
            acc_list.append(acc)
        except:
            pass  # Skip if accuracy calculation fails
    
    # Aggregate metrics for this iteration
    mean_auc = np.nanmean(auc_list) if auc_list else 0.0
    mean_f1 = np.nanmean(f1_list) if f1_list else 0.0
    mean_acc = np.nanmean(acc_list) if acc_list else 0.0
    
    lr_progression.append({
        "iter": max_iter,
        "test_auc": float(mean_auc),
        "test_f1": float(mean_f1),
        "test_acc": float(mean_acc),
        "n_valid_drugs": len(auc_list)
    })
    
    print(f"  Valid drugs: {len(auc_list)}/{y_train.shape[1]}")
    print(f"  AUC={mean_auc:.4f} ({mean_auc*100:.2f}%)")
    print(f"  F1={mean_f1:.4f} ({mean_f1*100:.2f}%)")
    print(f"  ACC={mean_acc:.4f} ({mean_acc*100:.2f}%)")

# Convert to DataFrame for easier plotting
lr_df = pd.DataFrame(lr_progression)

# Save both pickle and CSV
pickle_path = "data/output/cf/lr_learning_curve.pkl"
csv_path = "data/output/cf/lr_learning_curve.csv"

with open(pickle_path, "wb") as f:
    pickle.dump(lr_progression, f)

lr_df.to_csv(csv_path, index=False)

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print(f"Saved pickle ➜ {pickle_path}")
print(f"Saved CSV ➜ {csv_path}")
print("="*60)

# Print final results
print("\n=== FINAL RESULTS (max_iter=10000) ===")
final_results = lr_progression[-1]
print(f"Test AUC: {final_results['test_auc']:.4f} ({final_results['test_auc']*100:.2f}%)")
print(f"Test F1: {final_results['test_f1']:.4f} ({final_results['test_f1']*100:.2f}%)")
print(f"Test Accuracy: {final_results['test_acc']:.4f} ({final_results['test_acc']*100:.2f}%)")

# Print summary table
print("\n=== LEARNING CURVE SUMMARY ===")
print(lr_df.to_string(index=False))
