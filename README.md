# Operationalizing Zero Trust in Federated Learning

Reference implementation and reproduction code for the paper

> **Operationalizing Zero Trust in Federated Learning: A Fuzzy Approach to Client Trustworthiness in IoT Security**
> Thinh V. Le, Huan T. Tran, Samia Bouzefrane.

The repository contains a clean implementation of the **TrustFedAvg** framework, the dataset preprocessing pipeline for the *IoT GPS Tracker* subset of ToN-IoT, and scripts that reproduce the quantitative results reported in the paper.

---

## Abstract (paper)

Federated Learning (FL) has emerged as a vital paradigm for training machine-learning models on decentralised Internet-of-Things (IoT) data while preserving privacy. However, conventional FL frameworks remain vulnerable to unreliable or malicious clients that can compromise the integrity of the global model. To address this challenge we propose **TrustFedAvg**, a framework that embeds Zero-Trust security principles directly into the federated aggregation process. The approach operationalises the *"never trust, always verify"* philosophy by requiring each client to transmit its local model weights together with compact metadata — validation performance, update deviation, anomaly ratio, and dataset size. A fuzzy inference system computes a dynamic trust score from these indicators and regulates each client's influence on the global model. Evaluated on the *IoT GPS Tracker* subset of ToN-IoT, the framework achieves a **macro-F1 improvement of 2.6 %** over FedAvg and demonstrates strong resilience against poisoned updates, better recall of minority attack classes, and stable convergence under non-IID data.

---

## Key contributions

1. **Zero-Trust at the algorithmic layer.** Trust is enforced *inside* the federated aggregation step, not only at the network perimeter.
2. **Dynamic fuzzy trust evaluation.** A Mamdani fuzzy inference system combines (i) server-side validation Macro-F1, (ii) the normalised deviation of the client update from the global model, and (iii) the anomaly (attack) ratio of the client's local data, producing an interpretable trust score in `[0, 1]`.
3. **Dual soft / hard regulation.** Updates are softly down-weighted in proportion to their fuzzy trust, while a strict cut-off rule removes persistently low-trust clients from aggregation. Excluded clients are *re-admissible* once their behaviour improves, satisfying the continuous-verification principle of Zero-Trust.
4. **Empirical validation on ToN-IoT.** Robustness against poisoning, improved minority-class recall, stable convergence under non-IID partitions, and a scalability study with K = 10 , 50 or 100 clients, already test.

---

## Repository structure

```
new_update/
├── main.py                # Reproduces the paper's main 10-client experiment
├── run_all.py             # One-shot launcher: main + verification against Table 4
├── dataset.py             # Loading, Haversine speed, 70/15/15 split, Dirichlet partition
├── model.py               # MLP classifier  (5  ->  64  ->  32  ->  C)
├── fuzzy_trust.py         # Mamdani fuzzy inference for the trust score
├── plot_utils.py          # Metrics and plotting helpers
├── sensitivity_tau.py     # Robustness study of the trust cut-off
├── requirements.txt
├── figures/               # Output figures
└── results/               # Output Excel files
```

---

## Dataset

The experiments use the **IoT GPS Tracker** subset of the ToN-IoT corpus, available at:

<https://research.unsw.edu.au/projects/toniot-datasets>

Download `IoT_GPS_Tracker.csv` and place it one directory above `new_update/` (or change `CSV_PATH` in `main.py`).

### Pre-processing pipeline (`dataset.py`)

| Step | Description |
|-----|-------------|
| Timestamp unification | `date + time` parsed via `pandas.to_datetime`; rows with parse errors dropped. |
| Haversine speed       | `speed_mps` computed from consecutive GPS points and time deltas. |
| Temporal encodings    | `hour`, `weekday` extracted from the unified timestamp. |
| Splits                | 70 % train / 15 % validation / 15 % test, stratified by class. |
| Scaling               | `StandardScaler` fitted on train only, applied to val / test. |
| Federated partition   | Non-IID label-skew via Dirichlet(α = 0.5) over the train split. |

The validation set is used **only** by the server for the Zero-Trust verification step. All final metrics are reported on the held-out test set, so no information from the trust mechanism leaks into the reported numbers.

---

## Model

A lightweight Multi-Layer Perceptron:

```
Input (5)  ->  Linear(64)  ->  ReLU  ->  Linear(32)  ->  ReLU  ->  Linear(C)
```

Trained with Adam (`lr = 1e-3`), cross-entropy loss, batch size 32 for training and 64 for validation.

---

## Fuzzy trust system (`fuzzy_trust.py`)

| Variable          | Linguistic terms         | Universe |
|------------------|--------------------------|----------|
| Validation F1     | Low / Medium / High      | `[0, 1]` |
| Update deviation  | Small / Moderate / Large | `[0, 2]` |
| Anomaly ratio     | Benign / Suspicious / Malicious | `[0, 1]` |
| Trust output      | Low / Medium / High      | `[0, 1]` |

All membership functions are triangular; defuzzification uses the centroid method. Out-of-universe inputs trigger an interpretable weighted-heuristic fallback.

---

## Installation

```bash
git clone <repo-url>
cd <repo>/new_update
pip install -r requirements.txt
```

Dependencies: PyTorch, scikit-learn, scikit-fuzzy, pandas, numpy, matplotlib, openpyxl.

---

## Reproducing the paper

```bash
# (1) full reproduction of the 10-client experiment
python main.py

# (2) reproduction + automatic check against Table 4
python run_all.py
```

`run_all.py` prints a side-by-side comparison between the regenerated metrics and the paper's Table 4, plus the absolute differences and a max-delta check.

### Expected metrics on D_test (paper Table 4)

| Metric            | Centralized | FedAvg | TrustFedAvg (soft) | TrustFedAvg (strict) |
|-------------------|-------------|--------|--------------------|----------------------|
| Accuracy          | 0.9924      | 0.9739 | 0.9828             | 0.9917               |
| Macro F1          | 0.9439      | 0.8189 | 0.7602             | 0.8764               |
| Weighted F1       | 0.9924      | 0.9722 | 0.9801             | 0.9914               |
| Macro Precision   | 0.9316      | 0.9790 | 0.8554             | 0.9709               |
| Macro Recall      | 0.9569      | 0.7419 | 0.7128             | 0.8752               |

The strict TrustFedAvg variant matches the centralized accuracy upper bound while improving minority-class recall by **+13 points** over FedAvg.

### Expected exclusion dynamics (paper Table 6)

| Round  | # Excluded | Excluded client IDs |
|--------|------------|---------------------|
| 1      | 4          | `[1, 3, 4, 6]`      |
| 2 – 17 | 3          | `[1, 3, 6]`         |
| 18 – 20| 2          | `[1, 6]`            |

The pattern shows the framework's adaptive behaviour: clients are excluded when their trust is persistently low and re-admitted once it recovers.

---

## Outputs

After running `main.py`:

- `results/publication_results.xlsx` — Final_Metrics_Comparison, per-round histories, exclusion logs.
- `figures/publication_results.png` — composite figure with learning curves, confusion matrix, trust dynamics, and per-round exclusion counts.

---

## License

Code released for research and reproducibility. Please cite the paper if you use this implementation.

## Citation

```bibtex
@article{le2026operationalizing,
  title   = {Operationalizing Zero Trust in Federated Learning: A Fuzzy Approach to Client Trustworthiness in IoT Security},
  author  = {Le, Thinh V. and Tran, Huan T. and Bouzefrane, Samia},
  year    = {2026}
}
```

## Acknowledgements

The authors acknowledge the support of time and facilities from HCM City University of Technology and Education.
