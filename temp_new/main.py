# main.py (TrustFedAvg-Fuzzy with threshold-based exclusion + Section 6 outputs)
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

from dataset import prepare_data
from model import Net, get_weights, set_weights
from plot_utils import (
    get_predictions_and_metrics,
    plot_confusion_matrix,
    plot_final_metrics_bar_chart,
    plot_trust_dynamics
)
from fuzzy_trust import compute_fuzzy_trust

# ----------------------
# Hyperparameters / Config
# ----------------------
NUM_CLIENTS = 10
NUM_ROUNDS = 20
LOCAL_EPOCHS = 5
CENTRALIZED_EPOCHS = 20
TAU = 0.20  # Trust threshold for hard exclusion

# ----------------------
# Data loading
# ----------------------
trainloaders, valloader, trainloader_centralized, input_size, num_classes, class_names = prepare_data(num_clients=NUM_CLIENTS)

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
# Federated Learning Simulations
# ----------------------
def run_simulation(mode="FedAvg"):
    print(f"\n--- Starting {mode} Simulation ---")
    
    global_model = Net(input_size, num_classes)
    history = []
    trust_history = []         # per round: np.array(T_k(t)) for k=1..K
    exclusion_records = []     # list of dict(round, num_excluded, excluded_ids)

    for round_num in range(NUM_ROUNDS):
        print(f"\n{mode} Round {round_num + 1}/{NUM_ROUNDS}")
        
        client_updates = []
        client_ns = []
        f1_list, dev_list, ar_list = [], [], []  # fuzzy inputs

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
            
            # Collect indicators for trust
            if mode in ("TrustFedAvg", "TrustFedAvg-Threshold"):
                # 1) Validation Macro-F1 on D_val
                metrics, _ = get_predictions_and_metrics(client_net, valloader, class_names)
                f1_list.append(float(metrics["macro_f1"]))
                # 2) Normalized deviation
                client_update = np.concatenate([w.flatten() for w in get_weights(client_net)])
                global_update = np.concatenate([w.flatten() for w in global_weights])
                deviation_value = np.linalg.norm(client_update - global_update) / (np.linalg.norm(global_update) + 1e-8)
                dev_list.append(float(deviation_value))
                # 3) Attack ratio
                labels_list = []
                for _, local_labels in trainloaders[client_id]:
                    labels_list.extend(local_labels.numpy())
                labels_arr = np.array(labels_list)
                attack_ratio = float(np.mean(labels_arr != NORMAL_CLASS_ID))
                ar_list.append(attack_ratio)

        client_ns = np.array(client_ns, dtype=float)
        
        if mode in ("TrustFedAvg", "TrustFedAvg-Threshold"):
            # Normalize deviation to [0,2]
            dev_arr = np.array(dev_list, dtype=float)
            if dev_arr.size > 0:
                dev_norm01 = (dev_arr - dev_arr.min()) / (np.ptp(dev_arr) + 1e-8)  # np.ptp per NumPy 2.x
                dev_for_fuzzy = 2.0 * np.clip(dev_norm01, 0.0, 1.0)
            else:
                dev_for_fuzzy = dev_arr

            f1_arr = np.array(f1_list, dtype=float)
            ar_arr = np.array(ar_list, dtype=float)

            client_trusts = np.array([
                compute_fuzzy_trust(f1_arr[i], dev_for_fuzzy[i], ar_arr[i])
                for i in range(len(f1_arr))
            ], dtype=float)
            trust_history.append(client_trusts.copy())

            # Threshold-based exclusion
            if mode == "TrustFedAvg-Threshold":
                include_mask = client_trusts >= TAU
                excluded_ids = [int(i) for i in np.where(~include_mask)[0]]
                num_excluded = int((~include_mask).sum())
                effective_trusts = client_trusts.copy()
                effective_trusts[~include_mask] = 0.0
                exclusion_records.append({
                    'round': round_num + 1,
                    'num_excluded': num_excluded,
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
                # if all excluded, fall back to size-weighted
                agg_proportions = client_ns / client_ns.sum()
        else:
            agg_proportions = client_ns / client_ns.sum()

        # Aggregate
        layer_count = len(get_weights(global_model))
        aggregated_weights = [
            sum(agg_proportions[i] * layer_weights[l] for i, layer_weights in enumerate(client_updates))
            for l in range(layer_count)
        ]
        set_weights(global_model, aggregated_weights)

        # Round metrics (global model)
        metrics, _ = get_predictions_and_metrics(global_model, valloader, class_names)
        if mode in ("TrustFedAvg", "TrustFedAvg-Threshold") and len(trust_history) > 0:
            metrics["avg_trust"] = float(np.mean(trust_history[-1]))
            metrics["num_excluded"] = exclusion_records[-1]['num_excluded']
        history.append(metrics)
        print(f"  > Round {round_num + 1} Acc: {metrics['accuracy']:.4f}, Macro F1: {metrics['macro_f1']:.4f}"
              + (f", Excluded: {exclusion_records[-1]['num_excluded']}" if mode in ("TrustFedAvg","TrustFedAvg-Threshold") else ""))

    return global_model, history, (trust_history if mode in ("TrustFedAvg", "TrustFedAvg-Threshold") else None), exclusion_records

# ----------------------
# Run simulations
# ----------------------
fedavg_model, fedavg_history, _, _ = run_simulation(mode="FedAvg")
trustfedavg_model, trustfedavg_history, trustfedavg_trust_history, trustfedavg_exclusions = run_simulation(mode="TrustFedAvg")
trustfedavg_thr_model, trustfedavg_thr_history, trustfedavg_thr_trust_history, trustfedavg_thr_exclusions = run_simulation(mode="TrustFedAvg-Threshold")

# ----------------------
# Final Evaluation
# ----------------------
print("\n--- Final Evaluation ---")
metrics_cent, cm_cent = get_predictions_and_metrics(centralized_model, valloader, class_names)
metrics_fedavg, cm_fedavg = get_predictions_and_metrics(fedavg_model, valloader, class_names)
metrics_trustfedavg, cm_trustfedavg = get_predictions_and_metrics(trustfedavg_model, valloader, class_names)
metrics_trustfedavg_thr, cm_trustfedavg_thr = get_predictions_and_metrics(trustfedavg_thr_model, valloader, class_names)

# ----------------------
# Export Results to Excel (+ Section 6 delta sheet)
# ----------------------
print("\n--- Exporting Results to Excel ---")
with pd.ExcelWriter("publication_results.xlsx") as writer:
    # a) Final metrics table
    results_data = {
        "Metric": ["Accuracy", "Macro F1-Score", "Weighted F1-Score", "Macro Precision", "Macro Recall"],
        "Centralized": [
            metrics_cent["accuracy"], metrics_cent["macro_f1"], metrics_cent["weighted_f1"],
            metrics_cent["macro_precision"], metrics_cent["macro_recall"],
        ],
        "FedAvg": [
            metrics_fedavg["accuracy"], metrics_fedavg["macro_f1"], metrics_fedavg["weighted_f1"],
            metrics_fedavg["macro_precision"], metrics_fedavg["macro_recall"],
        ],
        "TrustFedAvg-Fuzzy": [
            metrics_trustfedavg["accuracy"], metrics_trustfedavg["macro_f1"], metrics_trustfedavg["weighted_f1"],
            metrics_trustfedavg["macro_precision"], metrics_trustfedavg["macro_recall"],
        ],
        f"TrustFedAvg-Fuzzy(τ={TAU:.2f})": [
            metrics_trustfedavg_thr["accuracy"], metrics_trustfedavg_thr["macro_f1"], metrics_trustfedavg_thr["weighted_f1"],
            metrics_trustfedavg_thr["macro_precision"], metrics_trustfedavg_thr["macro_recall"],
        ],
    }
    df_final = pd.DataFrame(results_data).set_index("Metric")
    df_final.to_excel(writer, sheet_name="Final_Metrics_Comparison")

    # b) Delta sheet (τ – no-τ) for Section 6
    cols = ["Accuracy","Macro F1-Score","Weighted F1-Score","Macro Precision","Macro Recall"]
    deltas = {
        m: df_final.loc[m, f"TrustFedAvg-Fuzzy(τ={TAU:.2f})"] - df_final.loc[m, "TrustFedAvg-Fuzzy"]
        for m in cols
    }
    df_delta = pd.DataFrame.from_dict(deltas, orient="index", columns=[f"Δ (τ - no τ)"]).rename_axis("Metric")
    df_delta.to_excel(writer, sheet_name="Final_Metrics_Delta")

    # c) Round histories
    df_fed = pd.DataFrame(fedavg_history); df_fed.index += 1; df_fed.index.name = "Round"
    df_fed.to_excel(writer, sheet_name="FedAvg_Round_History")

    df_trust = pd.DataFrame(trustfedavg_history); df_trust.index += 1; df_trust.index.name = "Round"
    df_trust.to_excel(writer, sheet_name="TrustFedAvg_Round_History")

    df_trust_thr = pd.DataFrame(trustfedavg_thr_history); df_trust_thr.index += 1; df_trust_thr.index.name = "Round"
    df_trust_thr.to_excel(writer, sheet_name="TrustFedAvg_Tau_Round_History")

    # d) Client-level trust distribution (final) for both variants
    attack_ratios = []
    for client_id in range(NUM_CLIENTS):
        labels_list = []
        for _, local_labels in trainloaders[client_id]:
            labels_list.extend(local_labels.numpy())
        labels_arr = np.array(labels_list)
        attack_ratio = float(np.mean(labels_arr != NORMAL_CLASS_ID))
        attack_ratios.append(attack_ratio)

    final_trusts = trustfedavg_trust_history[-1] if trustfedavg_trust_history else []
    final_trusts_thr = trustfedavg_thr_trust_history[-1] if trustfedavg_thr_trust_history else []

    df_client_trust = pd.DataFrame({
        "Client_ID": list(range(NUM_CLIENTS)),
        "Attack_Ratio": attack_ratios,
        "Final_Trust": final_trusts if len(final_trusts)>0 else [np.nan]*NUM_CLIENTS,
        "Final_Trust_tau": final_trusts_thr if len(final_trusts_thr)>0 else [np.nan]*NUM_CLIENTS,
    })
    df_client_trust.to_excel(writer, sheet_name="Client_Trust_Distribution", index=False)

    # e) Trust history matrices
    df_trust_history = pd.DataFrame(trustfedavg_trust_history, index=range(1, len(trustfedavg_trust_history)+1))
    df_trust_history.index.name =
