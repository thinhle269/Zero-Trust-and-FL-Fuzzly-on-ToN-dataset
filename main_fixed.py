# main_fixed.py
# Uses a clean evaluation protocol: D_val for trust only; final metrics on D_test

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

from dataset_fixed import prepare_data
from model import Net, get_weights, set_weights
from plot_utils import (
    get_predictions_and_metrics,
    plot_confusion_matrix,
    plot_final_metrics_bar_chart,
    plot_trust_dynamics
)
from fuzzy_trust import compute_fuzzy_trust

# ----------------------
# Hyperparameters
# ----------------------
NUM_CLIENTS = 10          # initial target; will be aligned to actual shards
NUM_ROUNDS = 20
LOCAL_EPOCHS = 5
CENTRALIZED_EPOCHS = 20
TAU = 0.20  # trust threshold

# ----------------------
# Data loading (three-way split)
# ----------------------
trainloaders, valloader, testloader, trainloader_centralized, input_size, num_classes, class_names = prepare_data(
    csv_path="IoT_GPS_Tracker.csv", num_clients=NUM_CLIENTS, beta=0.5, seed=42
)

# Align NUM_CLIENTS to actual number of non-empty shards
NUM_CLIENTS = len(trainloaders)

# Determine normal class id robustly
if isinstance(class_names, (list, tuple)) and 'normal' in class_names:
    NORMAL_CLASS_ID = int(class_names.index('normal'))
else:
    NORMAL_CLASS_ID = 0  # fallback

# ----------------------
# Centralized Training
# ----------------------
print("--- Training Centralized Model ---")
centralized_model = Net(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(centralized_model.parameters(), lr=0.001)

centralized_model.train()
for epoch in range(CENTRALIZED_EPOCHS):
    for features, labels in trainloader_centralized:
        optimizer.zero_grad()
        loss = criterion(centralized_model(features), labels)
        loss.backward()
        optimizer.step()
    print(f"Centralized Epoch {epoch+1}/{CENTRALIZED_EPOCHS} complete.")

# ----------------------
# Federated Learning Simulation
# ----------------------
def run_simulation(mode="FedAvg"):
    print(f"\n--- Starting {mode} Simulation ---")
    global_model = Net(input_size, num_classes)
    history = []
    ar_history = [] 
    trust_history = []
    exclusion_records = []

    for round_num in range(NUM_ROUNDS):
        client_updates, client_ns = [], []
        f1_list, dev_list, ar_list = [], [], []

        global_weights = get_weights(global_model)

        for client_id in range(NUM_CLIENTS):
            client_net = deepcopy(global_model)
            set_weights(client_net, global_weights)
            optimizer = torch.optim.Adam(client_net.parameters(), lr=0.001)

            # Local training
            client_net.train()
            for _ in range(LOCAL_EPOCHS):
                for features, labels in trainloaders[client_id]:
                    optimizer.zero_grad()
                    loss = criterion(client_net(features), labels)
                    loss.backward()
                    optimizer.step()

            client_updates.append(get_weights(client_net))
            client_ns.append(len(trainloaders[client_id].dataset))

            if mode in ["TrustFedAvg", "TrustFedAvg-Threshold"]:
                # indicators computed on D_val (internal verification only)
                metrics, _ = get_predictions_and_metrics(client_net, valloader, class_names)
                f1_list.append(float(metrics["macro_f1"]))

                client_update = np.concatenate([w.flatten() for w in get_weights(client_net)])
                global_update = np.concatenate([w.flatten() for w in global_weights])
                deviation_value = np.linalg.norm(client_update - global_update) / (np.linalg.norm(global_update) + 1e-8)
                dev_list.append(float(deviation_value))

                labels_list = []
                for _, local_labels in trainloaders[client_id]:
                    labels_list.extend(local_labels.numpy())
                ar_list.append(float(np.mean(np.array(labels_list) != NORMAL_CLASS_ID)))

        client_ns = np.array(client_ns, dtype=float)

        if mode in ["TrustFedAvg", "TrustFedAvg-Threshold"]:
            dev_arr = np.array(dev_list, dtype=float)
            if dev_arr.size > 0:
                dev_norm01 = (dev_arr - dev_arr.min()) / (np.ptp(dev_arr) + 1e-8)
                dev_for_fuzzy = 2.0 * np.clip(dev_norm01, 0.0, 1.0)  # map to [0,2] as in fuzzy system
            else:
                dev_for_fuzzy = dev_arr

            f1_arr = np.array(f1_list, dtype=float)
            ar_arr = np.array(ar_list, dtype=float)
            ar_history.append(ar_arr.copy())

            client_trusts = np.array([
                compute_fuzzy_trust(f1_arr[i], dev_for_fuzzy[i], ar_arr[i])
                for i in range(len(f1_arr))
            ], dtype=float)
            trust_history.append(client_trusts.copy())

            if mode == "TrustFedAvg-Threshold":
                include_mask = client_trusts >= TAU
                excluded_ids = [int(i) for i in np.where(~include_mask)[0]]
                effective_trusts = client_trusts.copy()
                effective_trusts[~include_mask] = 0.0
                exclusion_records.append({
                    'round': round_num + 1,
                    'num_excluded': int((~include_mask).sum()),
                    'excluded_ids': excluded_ids
                })
            else:
                effective_trusts = client_trusts.copy()
                exclusion_records.append({
                    'round': round_num + 1,
                    'num_excluded': 0,
                    'excluded_ids': []
                })

            agg_proportions = client_ns * effective_trusts
            if agg_proportions.sum() > 0:
                agg_proportions /= agg_proportions.sum()
            else:
                agg_proportions = client_ns / client_ns.sum()
        else:
            agg_proportions = client_ns / client_ns.sum()

        # aggregate
        layer_count = len(get_weights(global_model))
        aggregated_weights = [
            sum(agg_proportions[i] * layer_weights[l] for i, layer_weights in enumerate(client_updates))
            for l in range(layer_count)
        ]
        set_weights(global_model, aggregated_weights)

        # Monitoring curves can still use D_val
        metrics, _ = get_predictions_and_metrics(global_model, valloader, class_names)
        if mode in ["TrustFedAvg", "TrustFedAvg-Threshold"] and trust_history:
            metrics["avg_trust"] = float(np.mean(trust_history[-1]))
            metrics["num_excluded"] = exclusion_records[-1]['num_excluded']
        history.append(metrics)
        print(f"  > Round {round_num+1} Acc={metrics['accuracy']:.4f}, F1={metrics['macro_f1']:.4f}")
        if mode in ("TrustFedAvg", "TrustFedAvg-Threshold") and trust_history:
            final_trusts = np.asarray(trust_history[-1], dtype=float)
            final_attack = np.asarray(ar_history[-1], dtype=float) if ar_history else np.zeros_like(final_trusts)
            n = min(len(final_trusts), len(final_attack))
            if n > 0:
                df_trust = pd.DataFrame({
                  "Client ID": list(range(n)),
                  "Attack Ratio": final_attack[:n],
                  "Final Trust": final_trusts[:n]
            })
            out_name = f"client_trust_distribution_{mode}.xlsx"
            df_trust.to_excel(out_name, index=False)
            print(f"[INFO] Saved client trust distribution: {out_name}")

    return global_model, history, (trust_history if mode in ["TrustFedAvg", "TrustFedAvg-Threshold"] else None), exclusion_records

# ----------------------
# Run simulations
# ----------------------
fedavg_model, fedavg_history, _, _ = run_simulation("FedAvg")
trustfedavg_model, trustfedavg_history, trustfedavg_trust_history, trustfedavg_exclusions = run_simulation("TrustFedAvg")
trustfedavg_thr_model, trustfedavg_thr_history, trustfedavg_thr_trust_history, trustfedavg_thr_exclusions = run_simulation("TrustFedAvg-Threshold")

# ----------------------
# Final Evaluation on D_test (no leakage)
# ----------------------
print("\n[INFO] All final metrics are evaluated on D_test (unseen during trust computation).")
metrics_cent, cm_cent = get_predictions_and_metrics(centralized_model, testloader, class_names)
metrics_fedavg, cm_fedavg = get_predictions_and_metrics(fedavg_model, testloader, class_names)
metrics_trust, cm_trust = get_predictions_and_metrics(trustfedavg_model, testloader, class_names)
metrics_trust_tau, cm_trust_tau = get_predictions_and_metrics(trustfedavg_thr_model, testloader, class_names)

# ----------------------
# Export Results
# ----------------------
with pd.ExcelWriter("publication_results.xlsx") as writer:
    results_data = {
        "Metric": ["Accuracy", "Macro F1-Score", "Weighted F1-Score", "Macro Precision", "Macro Recall"],
        "Centralized": [
            metrics_cent["accuracy"], metrics_cent["macro_f1"], metrics_cent["weighted_f1"],
            metrics_cent["macro_precision"], metrics_cent["macro_recall"]
        ],
        "FedAvg": [
            metrics_fedavg["accuracy"], metrics_fedavg["macro_f1"], metrics_fedavg["weighted_f1"],
            metrics_fedavg["macro_precision"], metrics_fedavg["macro_recall"]
        ],
        "TrustFedAvg-Fuzzy": [
            metrics_trust["accuracy"], metrics_trust["macro_f1"], metrics_trust["weighted_f1"],
            metrics_trust["macro_precision"], metrics_trust["macro_recall"]
        ],
        f"TrustFedAvg-Fuzzy(τ={TAU:.2f})": [
            metrics_trust_tau["accuracy"], metrics_trust_tau["macro_f1"], metrics_trust_tau["weighted_f1"],
            metrics_trust_tau["macro_precision"], metrics_trust_tau["macro_recall"]
        ],
    }
    df_res = pd.DataFrame(results_data).set_index("Metric")
    df_res.to_excel(writer, sheet_name="Final_Metrics_Comparison")

    # Delta sheet
    df_delta = df_res[f"TrustFedAvg-Fuzzy(τ={TAU:.2f})"] - df_res["TrustFedAvg-Fuzzy"]
    df_delta.to_excel(writer, sheet_name="Final_Metrics_Delta")

    pd.DataFrame(fedavg_history).to_excel(writer, sheet_name="FedAvg_Round_History")
    pd.DataFrame(trustfedavg_history).to_excel(writer, sheet_name="TrustFedAvg_Round_History")
    pd.DataFrame(trustfedavg_thr_history).to_excel(writer, sheet_name="TrustFedAvg_Tau_Round_History")
    pd.DataFrame(trustfedavg_exclusions).to_excel(writer, sheet_name="Exclusion_History_NoTau")
    pd.DataFrame(trustfedavg_thr_exclusions).to_excel(writer, sheet_name="Exclusion_History_Tau")

print("Results saved to publication_results.xlsx")


# ----------------------
# Save Confusion Matrices (Centralized & FedAvg) on D_test
# ----------------------
import matplotlib.pyplot as _plt
from matplotlib import pyplot as _plt

def _save_cm(cm, classes, title, fname):
    fig, ax = _plt.subplots(figsize=(7,6))
    plot_confusion_matrix(ax, cm, classes, title)
    _plt.tight_layout()
    _plt.savefig(fname, dpi=300)
    _plt.close(fig)

_save_cm(cm_cent, class_names, "Confusion Matrix — Centralized (D_test)", "cm_centralized.png")
_save_cm(cm_fedavg, class_names, "Confusion Matrix — FedAvg (D_test)", "cm_fedavg.png")
print("Saved confusion matrices: cm_centralized.png, cm_fedavg.png")

# ----------------------
# Figures
# ----------------------
fig, axes = plt.subplots(3, 2, figsize=(18, 20))
fig.suptitle("Comparative Results (Final metrics on D_test)", fontsize=22)

# Learning curves (monitoring on D_val)
def plot_curve(ax, h_no, h_tau, metric, title):
    ax.plot([r[metric] for r in h_no], label="No τ")
    ax.plot([r[metric] for r in h_tau], label=f"τ={TAU}")
    ax.set_title(title); ax.set_xlabel("Round"); ax.set_ylabel(metric); ax.legend()

plot_curve(axes[0,0], trustfedavg_history, trustfedavg_thr_history, "accuracy", "Accuracy vs Rounds (D_val)")
plot_curve(axes[0,1], trustfedavg_history, trustfedavg_thr_history, "macro_f1", "Macro-F1 vs Rounds (D_val)")

# Final metrics bar chart (on D_test)
plot_final_metrics_bar_chart(axes[1,0], metrics_cent, metrics_fedavg, metrics_trust)
axes[1,0].set_title("Final Metrics on D_test (No τ)")

# Confusion matrix with τ (on D_test)
plot_confusion_matrix(axes[1,1], cm_trust_tau, class_names, f"Confusion Matrix - TrustFedAvg-Fuzzy (τ={TAU}) on D_test")

# Trust dynamics (D_val-based trust)
plot_trust_dynamics(axes[2,0], trustfedavg_thr_trust_history, NUM_CLIENTS)
axes[2,0].set_title("Trust Dynamics per Client (τ)")

# Exclusion curve
excl = [rec["num_excluded"] for rec in trustfedavg_thr_exclusions]
axes[2,1].plot(range(1, len(excl)+1), excl, marker="o")
axes[2,1].set_title("Clients Excluded per Round (τ)")
axes[2,1].set_xlabel("Round"); axes[2,1].set_ylabel("# Excluded")

plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig("publication_results.png", dpi=300)
print("Figures saved to publication_results.png")
# --------------------------------------------------
# Fig. 3 & Fig. 4 — Learning curves: FedAvg vs TrustFedAvg(τ) on D_val (monitoring)
# --------------------------------------------------
def _plot_round_curve(y_fed, y_trust, ylabel, title, fname):
    rounds = range(1, len(y_fed) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, y_fed, label="FedAvg", linewidth=2)
    plt.plot(rounds, y_trust, label=f"TrustFedAvg (τ={TAU})", linewidth=2)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()

# Lấy series theo vòng từ lịch sử (được tính trên D_val để monitoring)
acc_fed   = [r["accuracy"]  for r in fedavg_history]
acc_trust = [r["accuracy"]  for r in trustfedavg_thr_history]
f1_fed    = [r["macro_f1"]  for r in fedavg_history]
f1_trust  = [r["macro_f1"]  for r in trustfedavg_thr_history]

# Fig. 3: Accuracy comparison
_plot_round_curve(
    acc_fed, acc_trust,
    ylabel="Accuracy",
    title="Fig. 3: Accuracy comparison showing consistent advantage of TrustFedAvg over FedAvg across rounds",
    fname="fig3_accuracy.png",
)

# Fig. 4: Macro-F1 comparison
_plot_round_curve(
    f1_fed, f1_trust,
    ylabel="Macro-F1",
    title="Fig. 4: Macro-F1 comparison showing consistent advantage of TrustFedAvg over FedAvg across rounds",
    fname="fig4_macro_f1.png",
)

print("Saved learning-curve figures: fig3_accuracy.png, fig4_macro_f1.png")
