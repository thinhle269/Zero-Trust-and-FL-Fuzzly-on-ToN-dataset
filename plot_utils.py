# plot_utils.py (Nội dung mới)

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(ax, cm, classes, title):
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes, title=title,
           ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

def get_predictions_and_metrics(net, dataloader, class_names):
    net.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in dataloader:
            outputs = net(features)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0
    )
    metrics = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
    }
    cm = confusion_matrix(all_labels, all_preds)
    return metrics, cm

def plot_learning_curves(ax, history_fedavg, history_trustfedavg, metric, title):
    rounds = range(1, len(history_fedavg) + 1)
    fedavg_scores = [h[metric] for h in history_fedavg]
    trustfedavg_scores = [h[metric] for h in history_trustfedavg]
    
    ax.plot(rounds, fedavg_scores, marker='o', linestyle='-', label='FedAvg')
    ax.plot(rounds, trustfedavg_scores, marker='s', linestyle='--', label='TrustFedAvg')
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    ax.set_xticks(rounds)

def plot_final_metrics_bar_chart(ax, metrics_cent, metrics_fedavg, metrics_trustfedavg):
    labels = ['Accuracy', 'Macro F1', 'W. F1', 'Precision', 'Recall']
    cent_scores = [metrics_cent[k] for k in ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall']]
    fedavg_scores = [metrics_fedavg[k] for k in ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall']]
    trustfedavg_scores = [metrics_trustfedavg[k] for k in ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall']]

    x = np.arange(len(labels))
    width = 0.25
    
    rects1 = ax.bar(x - width, cent_scores, width, label='Centralized')
    rects2 = ax.bar(x, fedavg_scores, width, label='FedAvg')
    rects3 = ax.bar(x + width, trustfedavg_scores, width, label='TrustFedAvg')

    ax.set_ylabel('Scores')
    ax.set_title('Final Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(rects1, padding=3, fmt='%.3f', rotation=90)
    ax.bar_label(rects2, padding=3, fmt='%.3f', rotation=90)
    ax.bar_label(rects3, padding=3, fmt='%.3f', rotation=90)
    ax.set_ylim(0, 1.15)
   
def plot_trust_dynamics(ax, trust_history, num_clients):
    rounds = range(1, len(trust_history) + 1)

    for client_id in range(num_clients):
        client_scores = [round_trusts[client_id] for round_trusts in trust_history]
        ax.plot(rounds, client_scores, marker='o', linestyle='-',
                label=f'Client {client_id}', alpha=0.7)

    ax.set_title("Trust Dynamics per Client")
    ax.set_xlabel("Round")
    ax.set_ylabel("Trust Score")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=':', alpha=0.6)
    if num_clients <= 10:  # chỉ hiển thị legend nếu ít client
        ax.legend()
