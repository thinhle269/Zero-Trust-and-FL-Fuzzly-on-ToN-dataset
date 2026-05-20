# sensitivity_tau.py
# Ablation / sensitivity analysis for the trust threshold tau in TrustFedAvg(tau).
#
# For tau in {0.1, 0.2, ..., 0.9} we re-run TrustFedAvg-Threshold with the same
# data partition and the same seed, and record the resulting D_test metrics.
# The goal is to demonstrate that tau = 0.20 is the *justified* operating point,
# not an arbitrary choice (per Section 5 of the paper).
#
# Outputs:
#   results/tau_sensitivity.xlsx               - per-tau metrics + summary
#   figures/tau_sensitivity_macroF1_recall.png - main ablation curve
#   figures/tau_sensitivity_acc_wf1.png        - accuracy / weighted F1 view
#   figures/tau_sensitivity_exclusion.png      - avg #excluded vs tau
#   figures/tau_sensitivity_combined.png       - 2x2 publication figure

import os
import time
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset import prepare_data
from model import Net, get_weights, set_weights
from plot_utils import get_predictions_and_metrics
from fuzzy_trust import compute_fuzzy_trust

# ---------------- Configuration ----------------
NUM_CLIENTS = 10
NUM_ROUNDS = 20
LOCAL_EPOCHS = 5
CENTRALIZED_EPOCHS = 20
SEED = 42

CSV_PATH = "../IoT_GPS_Tracker.csv"

# Sample fraction lets the sweep finish in a reasonable wall-clock time while
# preserving the qualitative ordering across tau values.  Set to None (or 1.0)
# to run the full data sweep that exactly matches the paper's settings.
SAMPLE_FRAC = 0.15
TRAIN_BATCH = 128
VAL_BATCH = 256

TAU_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PAPER_TAU = 0.20

RESULTS_DIR = "results"
FIG_DIR = "figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def _stratified_subsample(X, y, frac, rng):
    if frac is None or frac >= 1.0:
        return X, y
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = max(1, int(round(len(idx) * frac)))
        keep.append(idx[:n])
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return X[keep], y[keep]


def _prepare_data_for_sweep(num_clients, seed, sample_frac, train_batch, val_batch):
    """Wrap prepare_data() with stratified subsampling and configurable batch size."""
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset
    from dataset import load_and_engineer, dirichlet_partition_indices
    from sklearn.model_selection import train_test_split

    X, y, classes, feature_cols = load_and_engineer(CSV_PATH)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    if sample_frac is not None and sample_frac < 1.0:
        rng = np.random.RandomState(seed)
        X_train, y_train = _stratified_subsample(X_train, y_train, sample_frac, rng)
        X_val, y_val = _stratified_subsample(X_val, y_val, sample_frac, rng)
        X_test, y_test = _stratified_subsample(X_test, y_test, sample_frac, rng)

    sc = StandardScaler().fit(X_train)
    X_train = sc.transform(X_train)
    X_val = sc.transform(X_val)
    X_test = sc.transform(X_test)

    idx_clients = dirichlet_partition_indices(y_train, num_clients=num_clients,
                                              beta=0.5, seed=seed)
    non_empty = [idx for idx in idx_clients if len(idx) > 0]
    trainloaders = []
    for idx in non_empty:
        ds = TensorDataset(torch.tensor(X_train[idx], dtype=torch.float32),
                           torch.tensor(y_train[idx], dtype=torch.long))
        trainloaders.append(DataLoader(ds, batch_size=train_batch, shuffle=True))

    valloader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                         torch.tensor(y_val, dtype=torch.long)),
                           batch_size=val_batch)
    testloader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                          torch.tensor(y_test, dtype=torch.long)),
                            batch_size=val_batch)
    trainloader_centralized = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=train_batch, shuffle=True
    )

    return (trainloaders, valloader, testloader, trainloader_centralized,
            len(feature_cols), len(classes), classes)


# ---------------- One simulation with a given tau ----------------
def run_trust_fedavg(tau, trainloaders, valloader, input_size, num_classes,
                     class_names, normal_class_id, seed=SEED, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    criterion = nn.CrossEntropyLoss()
    global_model = Net(input_size, num_classes)

    history = []
    trust_history = []
    exclusion_records = []
    num_clients = len(trainloaders)

    for t in range(NUM_ROUNDS):
        client_updates, client_ns = [], []
        f1_list, dev_list, ar_list = [], [], []
        global_weights = get_weights(global_model)

        for k in range(num_clients):
            client_net = deepcopy(global_model)
            set_weights(client_net, global_weights)
            opt = torch.optim.Adam(client_net.parameters(), lr=0.001)
            client_net.train()
            for _ in range(LOCAL_EPOCHS):
                for feats, lbls in trainloaders[k]:
                    opt.zero_grad()
                    loss = criterion(client_net(feats), lbls)
                    loss.backward()
                    opt.step()

            client_updates.append(get_weights(client_net))
            client_ns.append(len(trainloaders[k].dataset))

            m, _ = get_predictions_and_metrics(client_net, valloader, class_names)
            f1_list.append(float(m["macro_f1"]))
            cu = np.concatenate([w.flatten() for w in get_weights(client_net)])
            gu = np.concatenate([w.flatten() for w in global_weights])
            dev = np.linalg.norm(cu - gu) / (np.linalg.norm(gu) + 1e-8)
            dev_list.append(float(dev))
            labs = []
            for _, ll in trainloaders[k]:
                labs.extend(ll.numpy())
            ar_list.append(float(np.mean(np.array(labs) != normal_class_id)))

        client_ns = np.array(client_ns, dtype=float)
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

        mask = client_trusts >= tau
        excluded_ids = [int(i) for i in np.where(~mask)[0]]
        eff_trusts = client_trusts.copy()
        eff_trusts[~mask] = 0.0
        exclusion_records.append({
            "round": t + 1,
            "num_excluded": int((~mask).sum()),
            "excluded_ids": excluded_ids,
        })

        props = client_ns * eff_trusts
        if props.sum() > 0:
            props = props / props.sum()
        else:
            props = client_ns / client_ns.sum()

        L = len(get_weights(global_model))
        new_weights = [
            sum(props[i] * client_updates[i][iL] for i in range(num_clients))
            for iL in range(L)
        ]
        set_weights(global_model, new_weights)

        met, _ = get_predictions_and_metrics(global_model, valloader, class_names)
        met["num_excluded"] = exclusion_records[-1]["num_excluded"]
        history.append(met)
        if verbose:
            print(f"  Round {t+1}/{NUM_ROUNDS} Acc={met['accuracy']:.4f} "
                  f"F1={met['macro_f1']:.4f} Excl={met['num_excluded']}")

    return global_model, history, trust_history, exclusion_records


def run_fedavg_only(trainloaders, valloader, input_size, num_classes,
                    class_names, seed=SEED, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    criterion = nn.CrossEntropyLoss()
    global_model = Net(input_size, num_classes)
    num_clients = len(trainloaders)

    for t in range(NUM_ROUNDS):
        client_updates, client_ns = [], []
        global_weights = get_weights(global_model)
        for k in range(num_clients):
            client_net = deepcopy(global_model)
            set_weights(client_net, global_weights)
            opt = torch.optim.Adam(client_net.parameters(), lr=0.001)
            client_net.train()
            for _ in range(LOCAL_EPOCHS):
                for feats, lbls in trainloaders[k]:
                    opt.zero_grad()
                    loss = criterion(client_net(feats), lbls)
                    loss.backward()
                    opt.step()
            client_updates.append(get_weights(client_net))
            client_ns.append(len(trainloaders[k].dataset))
        client_ns = np.array(client_ns, dtype=float)
        props = client_ns / client_ns.sum()
        L = len(get_weights(global_model))
        new_weights = [
            sum(props[i] * client_updates[i][iL] for i in range(num_clients))
            for iL in range(L)
        ]
        set_weights(global_model, new_weights)
        if verbose and (t + 1) % 5 == 0:
            met, _ = get_predictions_and_metrics(global_model, valloader, class_names)
            print(f"  FedAvg round {t+1}/{NUM_ROUNDS} acc={met['accuracy']:.4f}")
    return global_model


# ---------------- Plots ----------------
def _annotate(ax, x, y, label, color="tab:red"):
    ax.axvline(x, color=color, linestyle="--", alpha=0.7, label=label)
    ax.scatter([x], [y], color=color, zorder=5, s=110,
               edgecolor="black", linewidth=1.2)


def _plot_main(df, fed, optimal_tau, save_path):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(df["tau"], df["macro_f1"], marker="o", linewidth=2.0,
            label="Macro-F1 (TrustFedAvg(tau))", color="tab:blue")
    ax.plot(df["tau"], df["macro_recall"], marker="s", linewidth=2.0,
            label="Macro-Recall (TrustFedAvg(tau))", color="tab:green")
    ax.axhline(fed["macro_f1"], color="tab:blue", linestyle=":", alpha=0.6,
               label=f"FedAvg Macro-F1 = {fed['macro_f1']:.3f}")
    ax.axhline(fed["macro_recall"], color="tab:green", linestyle=":", alpha=0.6,
               label=f"FedAvg Macro-Recall = {fed['macro_recall']:.3f}")
    f1_opt = df.loc[df["tau"] == optimal_tau, "macro_f1"].values[0]
    _annotate(ax, optimal_tau, f1_opt, f"Selected tau = {optimal_tau:.2f}")
    ax.set_xlabel("Trust threshold tau")
    ax.set_ylabel("Score")
    ax.set_title("Sensitivity of TrustFedAvg(tau) - Macro-F1 and Macro-Recall vs tau")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def _plot_acc(df, fed, optimal_tau, save_path):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(df["tau"], df["accuracy"], marker="o", linewidth=2.0,
            label="Accuracy", color="tab:purple")
    ax.plot(df["tau"], df["weighted_f1"], marker="s", linewidth=2.0,
            label="Weighted F1", color="tab:orange")
    ax.axhline(fed["accuracy"], color="tab:purple", linestyle=":", alpha=0.6,
               label=f"FedAvg Accuracy = {fed['accuracy']:.3f}")
    acc_opt = df.loc[df["tau"] == optimal_tau, "accuracy"].values[0]
    _annotate(ax, optimal_tau, acc_opt, f"Selected tau = {optimal_tau:.2f}")
    ax.set_xlabel("Trust threshold tau")
    ax.set_ylabel("Score")
    ax.set_title("Sensitivity of TrustFedAvg(tau) - Accuracy and Weighted-F1 vs tau")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def _plot_exclusion(df, optimal_tau, save_path):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(df["tau"].astype(str), df["avg_excluded_per_round"],
                  color="tab:gray", edgecolor="black",
                  label="Average clients excluded per round")
    opt_pos = list(df["tau"]).index(optimal_tau)
    bars[opt_pos].set_color("tab:red")
    ax.set_xlabel("Trust threshold tau")
    ax.set_ylabel("Average #excluded clients per round")
    ax.set_title("Exclusion rate vs tau - higher tau evicts more clients")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    for idx, v in enumerate(df["avg_excluded_per_round"]):
        ax.text(idx, v + 0.05, f"{v:.1f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def _plot_combined(df, fed, optimal_tau, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(df["tau"], df["macro_f1"], marker="o", linewidth=2.0, color="tab:blue",
            label="TrustFedAvg(tau)")
    ax.axhline(fed["macro_f1"], color="tab:blue", linestyle=":", alpha=0.6,
               label=f"FedAvg = {fed['macro_f1']:.3f}")
    f1_opt = df.loc[df["tau"] == optimal_tau, "macro_f1"].values[0]
    _annotate(ax, optimal_tau, f1_opt, f"Selected tau = {optimal_tau:.2f}")
    ax.set_title("(a) Macro-F1 vs tau"); ax.set_xlabel("tau"); ax.set_ylabel("Macro-F1")
    ax.grid(True, linestyle=":", alpha=0.6); ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.plot(df["tau"], df["macro_recall"], marker="s", linewidth=2.0, color="tab:green",
            label="TrustFedAvg(tau)")
    ax.axhline(fed["macro_recall"], color="tab:green", linestyle=":", alpha=0.6,
               label=f"FedAvg = {fed['macro_recall']:.3f}")
    r_opt = df.loc[df["tau"] == optimal_tau, "macro_recall"].values[0]
    _annotate(ax, optimal_tau, r_opt, f"Selected tau = {optimal_tau:.2f}")
    ax.set_title("(b) Macro-Recall vs tau"); ax.set_xlabel("tau"); ax.set_ylabel("Macro-Recall")
    ax.grid(True, linestyle=":", alpha=0.6); ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.plot(df["tau"], df["accuracy"], marker="o", linewidth=2.0,
            label="Accuracy", color="tab:purple")
    ax.plot(df["tau"], df["weighted_f1"], marker="s", linewidth=2.0,
            label="Weighted F1", color="tab:orange")
    acc_opt = df.loc[df["tau"] == optimal_tau, "accuracy"].values[0]
    _annotate(ax, optimal_tau, acc_opt, f"Selected tau = {optimal_tau:.2f}")
    ax.set_title("(c) Accuracy & Weighted-F1 vs tau"); ax.set_xlabel("tau"); ax.set_ylabel("Score")
    ax.grid(True, linestyle=":", alpha=0.6); ax.legend(fontsize=9)

    ax = axes[1, 1]
    bars = ax.bar(df["tau"].astype(str), df["avg_excluded_per_round"],
                  color="tab:gray", edgecolor="black")
    opt_pos = list(df["tau"]).index(optimal_tau)
    bars[opt_pos].set_color("tab:red")
    ax.set_title("(d) Average clients excluded per round vs tau")
    ax.set_xlabel("tau"); ax.set_ylabel("# excluded")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    for idx, v in enumerate(df["avg_excluded_per_round"]):
        ax.text(idx, v + 0.05, f"{v:.1f}", ha="center", fontsize=9)

    fig.suptitle("Sensitivity analysis of TrustFedAvg(tau) - tau = 0.20 is the chosen operating point",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


# ---------------- Main ----------------
def main():
    print(f"Preparing data (sample_frac={SAMPLE_FRAC}, train_batch={TRAIN_BATCH})...")
    (trainloaders, valloader, testloader, _,
     input_size, num_classes, class_names) = _prepare_data_for_sweep(
         num_clients=NUM_CLIENTS, seed=SEED, sample_frac=SAMPLE_FRAC,
         train_batch=TRAIN_BATCH, val_batch=VAL_BATCH)

    normal_id = class_names.index("normal") if "normal" in class_names else 0
    print(f"Clients={len(trainloaders)} sizes={[len(t.dataset) for t in trainloaders]}")

    print("\n[1/2] Running FedAvg baseline (no trust mechanism)...")
    t0 = time.time()
    fed_model = run_fedavg_only(trainloaders, valloader, input_size, num_classes,
                                class_names, seed=SEED, verbose=True)
    fed_metrics, _ = get_predictions_and_metrics(fed_model, testloader, class_names)
    print(f"  FedAvg test  Acc={fed_metrics['accuracy']:.4f}  "
          f"MacroF1={fed_metrics['macro_f1']:.4f}  "
          f"MacroR={fed_metrics['macro_recall']:.4f}  ({time.time()-t0:.0f}s)")

    rows = []
    excl_table = {}
    for tau in TAU_GRID:
        t0 = time.time()
        print(f"\n[2/2] Sweep tau = {tau:.2f}")
        model, history, _, excl = run_trust_fedavg(
            tau, trainloaders, valloader, input_size, num_classes,
            class_names, normal_id, seed=SEED, verbose=False
        )
        metrics, _ = get_predictions_and_metrics(model, testloader, class_names)
        avg_excl = float(np.mean([r["num_excluded"] for r in excl]))
        max_excl = int(np.max([r["num_excluded"] for r in excl]))
        rows.append({
            "tau": tau,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "avg_excluded_per_round": avg_excl,
            "max_excluded_per_round": max_excl,
        })
        excl_table[tau] = [r["num_excluded"] for r in excl]
        print(f"  tau={tau:.2f}  Acc={metrics['accuracy']:.4f}  "
              f"MacroF1={metrics['macro_f1']:.4f}  "
              f"MacroR={metrics['macro_recall']:.4f}  "
              f"avgExcl={avg_excl:.2f}  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df_excl = pd.DataFrame(excl_table)
    df_excl.index = [f"Round {r+1}" for r in range(NUM_ROUNDS)]

    fed_row = pd.DataFrame([{
        "tau": "FedAvg (no trust)",
        "accuracy": fed_metrics["accuracy"],
        "macro_f1": fed_metrics["macro_f1"],
        "weighted_f1": fed_metrics["weighted_f1"],
        "macro_precision": fed_metrics["macro_precision"],
        "macro_recall": fed_metrics["macro_recall"],
        "avg_excluded_per_round": 0.0,
        "max_excluded_per_round": 0,
    }])
    df_save = pd.concat([df, fed_row], ignore_index=True)

    summary = pd.DataFrame({
        "Criterion": ["Best Macro-F1", "Best Macro-Recall", "Best Accuracy",
                      "Paper's tau (chosen)"],
        "tau": [df.loc[df["macro_f1"].idxmax(), "tau"],
                df.loc[df["macro_recall"].idxmax(), "tau"],
                df.loc[df["accuracy"].idxmax(), "tau"],
                PAPER_TAU],
        "Macro_F1": [df["macro_f1"].max(),
                     df.loc[df["macro_recall"].idxmax(), "macro_f1"],
                     df.loc[df["accuracy"].idxmax(), "macro_f1"],
                     df.loc[df["tau"] == PAPER_TAU, "macro_f1"].values[0]],
        "Macro_Recall": [df.loc[df["macro_f1"].idxmax(), "macro_recall"],
                         df["macro_recall"].max(),
                         df.loc[df["accuracy"].idxmax(), "macro_recall"],
                         df.loc[df["tau"] == PAPER_TAU, "macro_recall"].values[0]],
        "Accuracy": [df.loc[df["macro_f1"].idxmax(), "accuracy"],
                     df.loc[df["macro_recall"].idxmax(), "accuracy"],
                     df["accuracy"].max(),
                     df.loc[df["tau"] == PAPER_TAU, "accuracy"].values[0]],
    })

    xlsx = os.path.join(RESULTS_DIR, "tau_sensitivity.xlsx")
    with pd.ExcelWriter(xlsx) as writer:
        df_save.to_excel(writer, sheet_name="Tau_Sensitivity", index=False)
        df_excl.to_excel(writer, sheet_name="Exclusions_per_Round")
        summary.to_excel(writer, sheet_name="Summary", index=False)
    print(f"\nWrote {xlsx}")

    _plot_main(df, fed_metrics, PAPER_TAU,
               os.path.join(FIG_DIR, "tau_sensitivity_macroF1_recall.png"))
    _plot_acc(df, fed_metrics, PAPER_TAU,
              os.path.join(FIG_DIR, "tau_sensitivity_acc_wf1.png"))
    _plot_exclusion(df, PAPER_TAU,
                    os.path.join(FIG_DIR, "tau_sensitivity_exclusion.png"))
    _plot_combined(df, fed_metrics, PAPER_TAU,
                   os.path.join(FIG_DIR, "tau_sensitivity_combined.png"))

    print("\n=== TAU SENSITIVITY ===")
    print(df.round(4).to_string(index=False))
    print("\n=== JUSTIFICATION TABLE ===")
    print(summary.round(4).to_string(index=False))
    print(f"\nFigures saved under {FIG_DIR}/")


if __name__ == "__main__":
    main()
