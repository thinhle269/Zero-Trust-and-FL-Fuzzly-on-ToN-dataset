# run_all.py
# One-shot launcher that:
#   1) Reproduces the paper's Table 4 / Table 6 results (main.py)
#   2) Runs the tau sensitivity / ablation analysis (sensitivity_tau.py)
#   3) Prints a side-by-side comparison so you can verify the reproduction
#
# Usage:
#   python run_all.py
#   python run_all.py --skip-main          # only run the sensitivity sweep
#   python run_all.py --skip-sensitivity   # only reproduce main results

import argparse
import os
import subprocess
import sys
import time
import pandas as pd

PAPER_TABLE4 = pd.DataFrame({
    "Metric": ["Accuracy", "Macro F1-Score", "Weighted F1-Score",
               "Macro Precision", "Macro Recall"],
    "Centralized":            [0.9924, 0.9439, 0.9924, 0.9316, 0.9569],
    "FedAvg":                 [0.9739, 0.8189, 0.9722, 0.9790, 0.7419],
    "TrustFedAvg-Fuzzy":      [0.9828, 0.7602, 0.9801, 0.8554, 0.7128],
    "TrustFedAvg-Fuzzy(t=0.20)":[0.9917, 0.8764, 0.9914, 0.9709, 0.8752],
}).set_index("Metric")

PAPER_TABLE6 = (
    "Round | Num excluded | Excluded ids\n"
    "  1   |      4       | [1, 3, 4, 6]\n"
    " 2-17 |      3       | [1, 3, 6]\n"
    "18-20 |      2       | [1, 6]\n"
)


def _run(cmd, cwd):
    print(f"\n>>> {cmd}  (cwd={cwd})")
    t0 = time.time()
    proc = subprocess.run(cmd, shell=True, cwd=cwd)
    print(f">>> finished in {time.time()-t0:.1f}s with exit code {proc.returncode}")
    if proc.returncode != 0:
        raise SystemExit(f"Command failed: {cmd}")


def _compare_with_paper():
    xlsx = os.path.join("results", "publication_results.xlsx")
    if not os.path.exists(xlsx):
        print(f"[compare] {xlsx} not found - skipping comparison.")
        return

    df_run = pd.read_excel(xlsx, sheet_name="Final_Metrics_Comparison").set_index("Metric")
    print("\n=== PAPER vs RE-RUN (Final_Metrics_Comparison on D_test) ===")
    print("\nPaper Table 4:")
    print(PAPER_TABLE4.round(4).to_string())
    print("\nThis run:")
    print(df_run.round(4).to_string())

    rename = {
        "Centralized": "Centralized",
        "FedAvg": "FedAvg",
        "TrustFedAvg-Fuzzy": "TrustFedAvg-Fuzzy",
    }
    tau_col = [c for c in df_run.columns if "0.20" in c]
    if tau_col:
        rename[tau_col[0]] = "TrustFedAvg-Fuzzy(t=0.20)"
    df_aligned = df_run.rename(columns=rename)
    common = [c for c in PAPER_TABLE4.columns if c in df_aligned.columns]
    if common:
        diff = df_aligned[common] - PAPER_TABLE4[common]
        print("\nAbsolute difference (re-run minus paper):")
        print(diff.round(4).to_string())
        max_abs = float(diff.abs().to_numpy().max())
        print(f"\nMax |delta| across {len(common)} columns x 5 metrics = {max_abs:.4f}")
        if max_abs < 0.02:
            print("[OK] Re-run closely matches the paper (max delta < 0.02).")
        elif max_abs < 0.05:
            print("[OK] Re-run is reasonably close to the paper (max delta < 0.05).")
        else:
            print("[NOTE] Re-run differs by more than 0.05 - check seed / partitioning.")

    print("\nPaper Table 6 (exclusion dynamics):")
    print(PAPER_TABLE6)
    try:
        excl = pd.read_excel(xlsx, sheet_name="Exclusion_History_Tau")
        print("This run's exclusion history (tau=0.20):")
        print(excl.to_string(index=False))
    except Exception as e:
        print(f"[NOTE] Could not read Exclusion_History_Tau: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-main", action="store_true",
                        help="Skip main.py (Table 4 reproduction)")
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="Skip sensitivity_tau.py (tau ablation)")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))

    if not args.skip_main:
        print("=" * 70)
        print("Step 1/2 - Reproducing paper Table 4 (main.py)")
        print("    NUM_CLIENTS=10, NUM_ROUNDS=20, LOCAL_EPOCHS=5, TAU=0.20")
        print("=" * 70)
        _run(f"{sys.executable} -u main.py", cwd=here)
        _compare_with_paper()

    if not args.skip_sensitivity:
        print("\n" + "=" * 70)
        print("Step 2/2 - tau sensitivity / ablation (sensitivity_tau.py)")
        print("    Sweeps tau in {0.1, 0.2, ..., 0.9} to show tau=0.20 is the")
        print("    justified operating point, not an arbitrary choice.")
        print("=" * 70)
        _run(f"{sys.executable} -u sensitivity_tau.py", cwd=here)

    print("\nAll done. See results/ and figures/ for outputs.")


if __name__ == "__main__":
    main()
