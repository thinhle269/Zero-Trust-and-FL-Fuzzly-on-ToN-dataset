# main.py  

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

from dataset import prepare_data
from model import Net, get_weights, set_weights
from plot_utils import (
    get_predictions_and_metrics,
    plot_confusion_matrix,
    plot_learning_curves,
    plot_final_metrics_bar_chart,
    plot_trust_dynamics
)
from fuzzy_trust import compute_fuzzy_trust

# 1. Hyperparameters
NUM_CLIENTS = 10
NUM_ROUNDS = 20
LOCAL_EPOCHS = 5
CENTRALIZED_EPOCHS = 20

# 2. Data loading
trainloaders, valloader, trainloader_centralized, input_size, num_classes, class_names = prepare_data(num_clients=NUM_CLIENTS)

# Determine normal class id robustly
if isinstance(class_names, (list, tuple)) and 'normal' in class_names:
    NORMAL_CLASS_ID = int(class_names.index('normal'))
else:
    NORMAL_CLASS_ID = 0  # fallback

# 3. Centralized Training
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

# 4. Federated Learning Simulations
def run_simulation(mode="FedAvg"):
    print(f"\n--- Starting {mode} Simulation ---")
    
    global_model = Net(input_size, num_classes)
    history = []
    trust_history = []  # store trust per client per round

    for round_num in range(NUM_ROUNDS):
        print(f"\n{mode} Round {round_num + 1}/{NUM_ROUNDS}")
        
        client_updates = []
        client_ns = []
        # temp holders for fuzzy inputs
        f1_list, dev_list, ar_list = [], [], []

        global_weights = get_weights(global_model)

        for client_id in tqdm(range(NUM_CLIENTS), desc=f"  Training clients"):
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
            
            # For TrustFedAvg, collect indicators (compute trust later after normalization)
            if mode == "TrustFedAvg":
                # 1. Validation performance
                metrics, _ = get_predictions_and_metrics(client_net, valloader, class_names)
                f1_list.append(float(metrics["macro_f1"]))  # in [0,1]

                # 2. Raw deviation (will normalize across clients)
                client_update = np.concatenate([w.flatten() for w in get_weights(client_net)])
                global_update = np.concatenate([w.flatten() for w in global_weights])
                deviation_value = np.linalg.norm(client_update - global_update) / (np.linalg.norm(global_update) + 1e-8)
                dev_list.append(float(deviation_value))

                # 3. Attack ratio (anomaly score) from local labels
                labels_list = []
                for fts, local_labels in trainloaders[client_id]:
                    labels_list.extend(local_labels.numpy())
                labels_arr = np.array(labels_list)
                attack_ratio = float(np.mean(labels_arr != NORMAL_CLASS_ID))
                ar_list.append(attack_ratio)

        client_ns = np.array(client_ns, dtype=float)
        
        if mode == "TrustFedAvg":
            # Normalize deviation per round to [0,1] then scale to [0,2] (universe of deviation)
            dev_arr = np.array(dev_list, dtype=float)
            if dev_arr.size > 0:
                dev_norm01 = (dev_arr - dev_arr.min()) / (np.ptp(dev_arr) + 1e-8)
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

            agg_proportions = client_ns * client_trusts
            if agg_proportions.sum() > 0:
                agg_proportions /= agg_proportions.sum()
            else:
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
        if mode == "TrustFedAvg" and len(trust_history) > 0:
            metrics["avg_trust"] = float(np.mean(trust_history[-1]))
        history.append(metrics)
        print(f"  > Round {round_num + 1} Accuracy: {metrics['accuracy']:.4f}, Macro F1: {metrics['macro_f1']:.4f}")

    return global_model, history, (trust_history if mode=="TrustFedAvg" else None)

# Run simulations
fedavg_model, fedavg_history, _ = run_simulation(mode="FedAvg")
trustfedavg_model, trustfedavg_history, trustfedavg_trust_history = run_simulation(mode="TrustFedAvg")

# 5. Final Evaluation
print("\n--- Final Evaluation ---")
metrics_cent, cm_cent = get_predictions_and_metrics(centralized_model, valloader, class_names)
metrics_fedavg, cm_fedavg = get_predictions_and_metrics(fedavg_model, valloader, class_names)
metrics_trustfedavg, cm_trustfedavg = get_predictions_and_metrics(trustfedavg_model, valloader, class_names)

# 6. Export Results to Excel
print("\n--- Exporting Results to Excel ---")
with pd.ExcelWriter("publication_results.xlsx") as writer:
    # a) Final metrics
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
    }
    pd.DataFrame(results_data).set_index("Metric").to_excel(writer, sheet_name="Final_Metrics_Comparison")

    # b) Round histories
    df_fed = pd.DataFrame(fedavg_history)
    df_fed.index = df_fed.index + 1
    df_fed.index.name = "Round"
    df_fed.to_excel(writer, sheet_name="FedAvg_Round_History")

    df_trust = pd.DataFrame(trustfedavg_history)
    df_trust.index = df_trust.index + 1
    df_trust.index.name = "Round"
    df_trust.to_excel(writer, sheet_name="TrustFedAvg_Round_History")

    # c) Client-level trust distribution (final)
    # recompute attack ratio per client for reporting
    attack_ratios = []
    for client_id in range(NUM_CLIENTS):
        labels_list = []
        for features, local_labels in trainloaders[client_id]:
            labels_list.extend(local_labels.numpy())
        labels_arr = np.array(labels_list)
        attack_ratio = float(np.mean(labels_arr != NORMAL_CLASS_ID))
        attack_ratios.append(attack_ratio)

    final_trusts = trustfedavg_trust_history[-1] if trustfedavg_trust_history else []
    df_client_trust = pd.DataFrame({
        "Client_ID": list(range(NUM_CLIENTS)),
        "Attack_Ratio": attack_ratios,
        "Final_Trust": final_trusts if len(final_trusts)>0 else [np.nan]*NUM_CLIENTS
    })
    df_client_trust.to_excel(writer, sheet_name="Client_Trust_Distribution", index=False)

    # d) Trust history (per round per client)
    df_trust_history = pd.DataFrame(trustfedavg_trust_history, index=range(1, len(trustfedavg_trust_history)+1))
    df_trust_history.index.name = "Round"
    df_trust_history.to_excel(writer, sheet_name="Trust_History")

print("Results saved to publication_results.xlsx")

# 7. Final Figures
print("\n--- Generating Final Figures ---")
fig, axes = plt.subplots(3, 3, figsize=(28, 20))
fig.suptitle('Model Comparison and Trust Dynamics', fontsize=24)

# Row 1: Learning curves and bar chart
plot_learning_curves(axes[0, 0], fedavg_history, trustfedavg_history, 'accuracy', 'FL Accuracy vs. Rounds')
plot_learning_curves(axes[0, 1], fedavg_history, trustfedavg_history, 'macro_f1', 'FL Macro F1-Score vs. Rounds')
plot_final_metrics_bar_chart(axes[0, 2], metrics_cent, metrics_fedavg, metrics_trustfedavg)

# Row 2: Confusion matrices
plot_confusion_matrix(axes[1, 0], cm_cent, class_names, "Confusion Matrix - Centralized")
plot_confusion_matrix(axes[1, 1], cm_fedavg, class_names, "Confusion Matrix - FedAvg")
plot_confusion_matrix(axes[1, 2], cm_trustfedavg, class_names, "Confusion Matrix - TrustFedAvg-Fuzzy")

# Row 3: Trust dynamics per client
plot_trust_dynamics(axes[2, 0], trustfedavg_trust_history, NUM_CLIENTS)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("publication_results.png", dpi=300)
print("Simulation finished. All figures saved to publication_results.png")
