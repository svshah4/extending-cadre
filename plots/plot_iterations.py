import re
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. Paste your raw logs for each run inside these variables
# -------------------------------------------------------------

log_73k = """
[0,0] | tst f1:40.8, auc:51.6 | trn f1:44.6, auc:52.3, loss:5.902
[7,268] | tst f1:49.2, auc:64.2 | trn f1:48.6, auc:62.0, loss:1.377
[14,536] | tst f1:55.3, auc:73.8 | trn f1:53.9, auc:70.5, loss:0.624
[22,128] | tst f1:58.0, auc:77.3 | trn f1:62.3, auc:80.3, loss:0.506
[29,396] | tst f1:59.2, auc:79.1 | trn f1:65.2, auc:82.8, loss:0.469
[36,664] | tst f1:59.8, auc:80.2 | trn f1:66.5, auc:84.0, loss:0.453
[44,256] | tst f1:60.8, auc:80.9 | trn f1:67.3, auc:85.0, loss:0.437
[51,524] | tst f1:61.3, auc:81.5 | trn f1:68.0, auc:85.5, loss:0.428
[59,116] | tst f1:62.0, auc:81.9 | trn f1:68.6, auc:86.0, loss:0.422
[66,384] | tst f1:63.3, auc:82.4 | trn f1:69.1, auc:86.5, loss:0.415
[73,652] | tst f1:63.2, auc:82.5 | trn f1:69.3, auc:86.7, loss:0.412
"""

log_22k = """
[0,0] | tst f1:36.3, auc:48.9 | trn f1:33.2, auc:48.4 loss:6.335
[1,348] | tst f1:45.7, auc:62.2 | trn f1:43.0, auc:55.5 loss:3.817
[3,20] | tst f1:45.7, auc:62.7, acc:62.6 | trn f1:49.7, auc:64.9, acc:64.4 loss:0.724
[4,368] | tst f1:47.0, auc:64.0, acc:63.6 | trn f1:49.9, auc:66.0, acc:65.0 loss:0.719
[6,40] | tst f1:46.5, auc:64.0, acc:63.5 | trn f1:50.5, auc:66.4, acc:65.4 loss:0.716
[7,388] | tst f1:47.4, auc:64.9, acc:64.2 | trn f1:50.9, auc:66.8, acc:65.7 loss:0.721
[9,60] | tst f1:47.2, auc:64.9, acc:64.6 | trn f1:51.0, auc:67.5, acc:66.4 loss:0.707
[10,408] | tst f1:49.0, auc:67.8, acc:67.0 | trn f1:51.6, auc:67.0, acc:67.2 loss:0.658
[12,80] | tst f1:51.6, auc:71.4, acc:70.1 | trn f1:55.3, auc:73.6, acc:71.0 loss:0.583
[13,428] | tst f1:53.6, auc:73.4, acc:72.1 | trn f1:58.6, auc:77.1, acc:74.2 loss:0.547
[15,100] | tst f1:55.3, auc:75.1, acc:73.4 | trn f1:61.0, auc:78.9, acc:76.0 loss:0.524
[16,448] | tst f1:55.9, auc:76.0, acc:74.2 | trn f1:62.6, auc:80.5, acc:77.6 loss:0.504
[18,120] | tst f1:56.6, auc:76.5, acc:74.5 | trn f1:64.2, auc:81.4, acc:78.5 loss:0.495
[19,468] | tst f1:57.5, auc:77.0, acc:75.4 | trn f1:64.8, auc:82.0, acc:78.9 loss:0.484
[21,140] | tst f1:58.0, auc:77.5, acc:75.6 | trn f1:65.1, auc:82.4, acc:79.2 loss:0.479
[22,488] | tst f1:57.8, auc:77.9, acc:75.7 | trn f1:65.5, auc:82.9, acc:79.6 loss:0.471
[24,160] | tst f1:58.1, auc:78.0, acc:75.8 | trn f1:66.2, auc:83.2, acc:80.0 loss:0.468
[25,508] | tst f1:58.3, auc:78.3, acc:76.0 | trn f1:66.4, auc:83.6, acc:80.1 loss:0.462
[27,180] | tst f1:58.7, auc:78.3, acc:76.2 | trn f1:66.5, auc:83.5, acc:80.2 loss:0.462
[28,528] | tst f1:58.7, auc:78.6, acc:76.1 | trn f1:66.4, auc:83.7, acc:80.1 loss:0.460
[30,200] | tst f1:58.3, auc:78.5, acc:76.0 | trn f1:66.8, auc:83.8, acc:80.3 loss:0.459
[31,548] | tst f1:58.6, auc:78.7, acc:76.1 | trn f1:66.8, auc:84.0, acc:80.5 loss:0.455
"""

log_14k = """
[0,0] | tst f1:41.3, auc:50.5 | trn f1:39.4, auc:49.8 loss:6.002
[1,348] | tst f1:47.9, auc:62.5 | trn f1:43.2, auc:57.1 loss:3.453
[3,20] | tst f1:48.3, auc:62.6, acc:63.4 | trn f1:49.3, auc:65.4, acc:65.7 loss:0.724
[4,368] | tst f1:47.6, auc:63.0, acc:63.7 | trn f1:49.7, auc:66.2, acc:66.5 loss:0.719
[6,40] | tst f1:48.7, auc:64.1, acc:64.4 | trn f1:50.1, auc:66.9, acc:66.9 loss:0.712
[7,388] | tst f1:49.2, auc:64.4, acc:64.8 | trn f1:50.3, auc:66.9, acc:67.3 loss:0.710
[9,60] | tst f1:51.8, auc:69.6, acc:68.1 | trn f1:51.8, auc:70.5, acc:69.0 loss:0.611
[10,408] | tst f1:54.0, auc:72.2, acc:70.4 | trn f1:55.8, auc:75.3, acc:73.0 loss:0.561
[12,80] | tst f1:54.9, auc:73.6, acc:71.6 | trn f1:58.3, auc:77.4, acc:75.0 loss:0.540
[13,428] | tst f1:56.1, auc:74.8, acc:72.6 | trn f1:60.5, auc:79.3, acc:76.7 loss:0.514
[15,100] | tst f1:56.8, auc:75.5, acc:73.2 | trn f1:61.5, auc:80.0, acc:77.5 loss:0.507
[16,448] | tst f1:57.8, auc:76.0, acc:73.9 | trn f1:62.2, auc:80.8, acc:78.2 loss:0.495
[18,120] | tst f1:57.0, auc:76.2, acc:73.6 | trn f1:63.1, auc:81.4, acc:78.7 loss:0.486
[19,468] | tst f1:58.0, auc:76.6, acc:74.2 | trn f1:63.6, auc:81.8, acc:78.9 loss:0.484
[21,140] | tst f1:57.7, auc:76.8, acc:74.0 | trn f1:63.8, auc:82.0, acc:79.2 loss:0.479
"""

log_147k = """
[0,0] | tst f1:40.2, auc:52.2 | trn f1:33.4, auc:51.3 loss:5.837
[14,536] | tst f1:54.5, auc:73.5 | trn f1:50.6, auc:64.8 loss:1.041
[0,0] | tst f1:40.2, auc:52.2, acc:52.0 | trn f1:33.4, auc:51.3, acc:51.9 loss:5.837
[14,536] | tst f1:54.5, auc:73.5, acc:71.0 | trn f1:50.6, auc:64.8, acc:66.7 loss:1.041
[29,396] | tst f1:59.5, auc:79.2, acc:75.9 | trn f1:62.0, auc:80.0, acc:77.4 loss:0.507
[44,256] | tst f1:61.4, auc:81.4, acc:77.2 | trn f1:65.8, auc:83.4, acc:80.2 loss:0.459
[59,116] | tst f1:62.5, auc:82.4, acc:77.8 | trn f1:67.3, auc:84.9, acc:81.2 loss:0.437
[73,652] | tst f1:65.3, auc:83.5, acc:78.4 | trn f1:68.3, auc:86.0, acc:81.7 loss:0.421
[88,512] | tst f1:64.6, auc:83.9, acc:78.6 | trn f1:68.8, auc:86.7, acc:81.9 loss:0.411
[103,372] | tst f1:64.3, auc:83.8, acc:78.6 | trn f1:68.9, auc:87.1, acc:82.0 loss:0.406
[118,232] | tst f1:63.8, auc:83.7, acc:78.5 | trn f1:69.0, auc:87.3, acc:82.1 loss:0.404
[133,92] | tst f1:64.0, auc:83.7, acc:78.5 | trn f1:69.3, auc:87.5, acc:82.2 loss:0.401
[147,628] | tst f1:64.2, auc:83.8, acc:78.6 | trn f1:69.6, auc:87.8, acc:82.4 loss:0.397
"""

# -------------------------------------------------------------
# 2. Parse logs into DataFrames
# -------------------------------------------------------------
def parse_log(log_string, label):
    pattern = r"\[(\d+),(\d+)\] \| tst f1:(.*?), auc:(.*?)[, ]+acc*:*.*?\| trn f1:(.*?), auc:(.*?),.*?loss:(.*)"
    simple_pattern = r"\[(\d+),(\d+)\] \| tst f1:(.*?), auc:(.*?) \| trn f1:(.*?), auc:(.*?),.*?loss:(.*)"

    rows = []
    for line in log_string.split("\n"):
        line = line.strip()
        if not line: 
            continue

        m = re.search(pattern, line)
        if not m:
            m = re.search(simple_pattern, line)
        if not m:
            continue

        step = int(m.group(1))
        test_f1 = float(m.group(3))
        test_auc = float(m.group(4))
        train_f1 = float(m.group(5))
        train_auc = float(m.group(6))
        loss = float(m.group(7))

        rows.append({
            "step": step,
            "test_f1": test_f1,
            "test_auc": test_auc,
            "train_f1": train_f1,
            "train_auc": train_auc,
            "loss": loss,
            "run": label
        })

    return pd.DataFrame(rows)


# -------------------------------------------------------------
# 3. Build full dataset
# -------------------------------------------------------------
df = pd.concat([
    parse_log(log_73k,  "73k_iters"),
    parse_log(log_22k,  "22k_iters"),
    parse_log(log_14k,  "14k_iters"),
    parse_log(log_147k, "147k_iters"),
], ignore_index=True)

df = df.sort_values(["run", "step"])
df = df.groupby("run").apply(lambda g: g[g["step"] > g["step"].min()]).reset_index(drop=True)



# -------------------------------------------------------------
# 4. Plotting with sorted legend + log-scale x
# -------------------------------------------------------------
def plot_metric(metric, ylabel):
    plt.figure(figsize=(8, 5))

    # Extract numeric iteration count from run label
    def extract_iters(label):
        m = re.search(r"(\d+)", label)
        return int(m.group(1)) * 1000 if m else 0

    # Sort groups by iteration count
    grouped = sorted(df.groupby("run"), key=lambda x: extract_iters(x[0]))

    # Plot
    for label, d in grouped:
        plt.plot(d["step"], d[metric], marker="o", label=label)

    plt.xscale("log")
    plt.xlabel("Training step (log scale)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Training Steps")
    plt.legend(title="Runs (low → high iterations)", loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------
# 5. Generate plots
# -------------------------------------------------------------
plot_metric("test_auc", "Test AUC")
plot_metric("test_f1", "Test F1 Score")
plot_metric("train_auc", "Train AUC")
plot_metric("loss", "Train Loss")

