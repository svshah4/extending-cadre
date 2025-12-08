import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from utils import load_dataset, split_dataset
import pickle
import os
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings (expected for low max_iter values)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

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

# Map to CADRE-style training steps
# CADRE trains for 147,628 iterations total
# We'll sample at similar checkpoints to CADRE
cadre_checkpoints = [0, 7268, 14536, 22128, 29396, 44256, 59116, 73652, 88512, 103372, 118232, 133092, 147628]

# Map CADRE steps to sklearn max_iter (approximate)
# Since LR converges much faster, we use a scaling factor
# Assume ~100 sklearn iterations ≈ 10k CADRE steps (rough estimate)
lr_max_iters = [1, 10, 20, 30, 40, 60, 80, 100, 120, 140, 160, 180, 200]

lr_progression = []

print("\n=== TRAINING LOGISTIC REGRESSION AT CADRE CHECKPOINTS ===")

for cadre_step, max_iter in zip(cadre_checkpoints, lr_max_iters):
    print(f"\nCADRE Step {cadre_step:,} → LR max_iter={max_iter}...")
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
            pass
        
        try:
            f1 = f1_score(y_test_drug, y_pred, zero_division=0)
            f1_list.append(f1)
        except:
            pass
        
        try:
            acc = accuracy_score(y_test_drug, y_pred)
            acc_list.append(acc)
        except:
            pass
    
    # Aggregate metrics
    mean_auc = np.nanmean(auc_list) if auc_list else 0.0
    mean_f1 = np.nanmean(f1_list) if f1_list else 0.0
    mean_acc = np.nanmean(acc_list) if acc_list else 0.0
    
    lr_progression.append({
        "cadre_step": cadre_step,
        "lr_max_iter": max_iter,
        "test_auc": float(mean_auc),
        "test_f1": float(mean_f1),
        "test_acc": float(mean_acc),
        "n_valid_drugs": len(auc_list)
    })
    
    print(f"  [{cadre_step:,}] | Valid drugs: {len(auc_list)}/{y_train.shape[1]}")
    print(f"  [{cadre_step:,}] | AUC={mean_auc*100:.1f}%, F1={mean_f1*100:.1f}%, ACC={mean_acc*100:.1f}%")

# Convert to DataFrame
lr_df = pd.DataFrame(lr_progression)

# Save results
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
print("\n=== FINAL RESULTS ===")
final_results = lr_progression[-1]
print(f"Logistic Regression (max_iter={final_results['lr_max_iter']}):")
print(f"  Test AUC: {final_results['test_auc']*100:.2f}%")
print(f"  Test F1:  {final_results['test_f1']*100:.2f}%")
print(f"  Test ACC: {final_results['test_acc']*100:.2f}%")

print(f"\nCADRE (from paper, 147,628 steps):")
print(f"  Test AUC: 83.8%")
print(f"  Test F1:  64.2%")
print(f"  Test ACC: 78.6%")

print(f"\nImprovement (CADRE over LR):")
print(f"  AUC: +{83.8 - final_results['test_auc']*100:.1f} percentage points")
print(f"  F1:  +{64.2 - final_results['test_f1']*100:.1f} percentage points")

# Print summary table
print("\n=== LEARNING CURVE SUMMARY ===")
print(lr_df.to_string(index=False))
