import pickle
import matplotlib.pyplot as plt

with open('data/output/cf/logs0_100k.pkl', 'rb') as f:
    logs = pickle.load(f)

iterations = logs['iter']
test_auc = [a * 100 for a in logs['auc']]
train_auc = [a * 100 for a in logs['auc_train']]

print(f"Training stopped at: {iterations[-1]:,} iterations")
print(f"Final Test AUC: {test_auc[-1]:.1f}%")
print(f"Final Train AUC: {train_auc[-1]:.1f}%")
print(f"Paper AUC: 83.4%")
print(f"Achievement: {(test_auc[-1]/83.4)*100:.1f}% of paper\n")

# Learning curve
plt.figure(figsize=(10, 6))
plt.plot(iterations, test_auc, 'o-', linewidth=2.5, markersize=7, 
         label='Test AUC', color='#2E86AB')
plt.plot(iterations, train_auc, 's-', linewidth=2.5, markersize=7, 
         label='Train AUC', color='#A23B72')
plt.axhline(y=83.4, linestyle='--', linewidth=2.5, 
            color='#F18F01', label='Paper (83.4%)')

plt.xlabel('Training Iterations', fontsize=14, fontweight='bold')
plt.ylabel('AUC (%)', fontsize=14, fontweight='bold')
plt.title('CADRE Training Progress', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: learning_curve.png")
plt.close()

print("\nPlots ready!")
