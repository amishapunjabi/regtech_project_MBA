"""
run_pipeline.py
────────────────────────────────────────────────────────────────
Master script that runs the full RegTech pipeline end-to-end:

  Step 1: Generate synthetic dataset
  Step 2: Feature engineering (VCI, TPS, RTR, GRE, PGDS)
  Step 3: Train ML models (Random Forest + Isolation Forest)
  Step 4: Generate SAR reports for top flagged transactions

After this completes, launch the dashboard with:
  streamlit run dashboard/app.py
────────────────────────────────────────────────────────────────
"""

import sys
import os
import time

# ── Path setup ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

def step(n, title):
    print(f"\n{'='*60}")
    print(f"  STEP {n}: {title}")
    print(f"{'='*60}")

def main():
    start = time.time()
    print("\n" + "█"*60)
    print("  RegTech AML Pipeline — Full Run")
    print("  BITS Pilani MBA FinTech | Amisha Punjabi | 2024MB53031")
    print("█"*60)

    # ── Step 1: Synthetic Data Generation ──────────────────────
    step(1, "Synthetic Data Generation")
    os.chdir(BASE_DIR)
    from data.synthetic_data_generator import generate_full_dataset
    customers, txn_df = generate_full_dataset()

    # ── Step 2: Feature Engineering ────────────────────────────
    step(2, "Feature Engineering")
    from utils.feature_engineering import build_feature_set
    featured_df = build_feature_set(sample_size=3000)

    # ── Step 3: ML Model Training & Evaluation ─────────────────
    step(3, "ML Model Training & Evaluation")
    from models.risk_model import run_full_pipeline
    run_full_pipeline()

    # ── Step 4: SAR Generation ──────────────────────────────────
    step(4, "SAR Report Generation")
    from sar.sar_generator import generate_all_sars
    generate_all_sars(n_sars=20)

    # ── Summary ─────────────────────────────────────────────────
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  ✅ Pipeline complete in {elapsed:.1f} seconds")
    print(f"{'='*60}")
    print("\n  Output files generated:")
    output_dir = os.path.join(BASE_DIR, "output_data")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"    📄 output_data/{f}  ({size/1024:.1f} KB)")

    sar_dir = os.path.join(output_dir, "sar_reports")
    if os.path.exists(sar_dir):
        n_sars = len([f for f in os.listdir(sar_dir) if f.endswith(".txt")])
        print(f"    📁 output_data/sar_reports/  ({n_sars} SAR files)")

    print(f"\n  🚀 To launch the dashboard, run:")
    print(f"     streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    main()
