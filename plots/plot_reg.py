import matplotlib.pyplot as plt
import numpy as np

# ===============================
# Logistic Regression Learning Curve (aligned with CADRE steps)
# ===============================
lr_cadre_steps = np.array([
    0, 7268, 14536, 22128, 29396,
    44256, 59116, 73652, 88512,
    103372, 118232, 133092, 147628
])

lr_auc = np.array([
    0.582205, 0.641551, 0.626208, 0.621063, 0.618937,
    0.617504, 0.617220, 0.617172, 0.617173,
    0.617173, 0.617173, 0.617173, 0.617173
])

lr_f1 = np.array([
    0.145442, 0.330101, 0.316922, 0.312873, 0.309002,
    0.309417, 0.308400, 0.308333, 0.308333,
    0.308333, 0.308333, 0.308333, 0.308333
])

lr_acc = np.array([
    0.789007, 0.780877, 0.776493, 0.779128, 0.778310,
    0.778401, 0.777720, 0.777811, 0.777811,
    0.777811, 0.777811, 0.777811, 0.777811
])

# ===============================
# CADRE Learning Curve (AUC only)
# ===============================
cadre_iters = np.array([
    0, 14536, 29396, 44256, 59116,
    73652, 88512, 103372, 118232,
    133092, 147628
])

cadre_auc = np.array([
    0.522, 0.735, 0.792, 0.814, 0.824,
    0.835, 0.839, 0.838, 0.837,
    0.837, 0.838
])

# ===============================
# Plotting
# ===============================
plt.figure(figsize=(9, 5))

# Logistic Regression curve
plt.plot(lr_cadre_steps, lr_auc * 100, marker='o', label="Logistic Regression (AUC)", linewidth=2)

# CADRE curve
plt.plot(cadre_iters, cadre_auc * 100, marker='o', label="CADRE (AUC)", linewidth=2)

plt.xlabel("Training Iterations", fontsize=12)
plt.ylabel("Test AUC (%)", fontsize=12)
plt.title("Learning Curve Comparison: Logistic Regression vs CADRE", fontsize=13)

plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("learning_curve_comparison.png", dpi=300)
plt.show()

