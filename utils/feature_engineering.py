"""
feature_engineering.py
────────────────────────────────────────────────────────────────
Derives the five core AML behavioral features from raw transaction
and customer data as described in the project methodology (Section 4.3):

  VCI  — Velocity Change Index
  TPS  — Threshold Proximity Score
  RTR  — Round-Tripping Ratio
  GRE  — Geographic Risk Exposure
  PGDS — Peer-Group Deviation Score

Also computes the three weighted sub-scores used in the WRS formula:
  S_Historical  (based on prior SAR/alert history)
  S_Behavioral  (based on the five features above)
  S_Geographic  (based on FATF jurisdiction risk)

Output: output_data/featured_transactions.csv
"""

import pandas as pd
import numpy as np
import os

OUTPUT_DIR = "output_data"
REPORTING_THRESHOLD_INR = 1_000_000   # ₹10 lakh

# FATF jurisdiction risk weights (same as generator)
FATF_RISK = {
    "India":          0.10, "USA":  0.10, "UK":   0.10,
    "Germany":        0.10, "Singapore": 0.10,
    "UAE":            0.50, "Turkey":    0.50,
    "Nigeria":        0.50, "Pakistan":  0.50,
    "North Korea":    1.00, "Iran":      1.00,
    "Myanmar":        1.00, "Cayman Islands": 0.60,
}


# ── Feature Calculations ───────────────────────────────────────

def compute_vci(txn_df: pd.DataFrame) -> pd.Series:
    """
    Velocity Change Index = (txn count last 24h − 30-day daily avg) /
                             30-day daily avg
    A VCI > 2.0 strongly indicates structuring or layering.
    """
    txn_df = txn_df.sort_values("Transaction_Timestamp")
    vci_list = []

    for _, row in txn_df.iterrows():
        cid  = row["Sender_Customer_ID"]
        t    = row["Transaction_Timestamp"]
        cust_txns = txn_df[txn_df["Sender_Customer_ID"] == cid]

        last_24h  = cust_txns[
            (cust_txns["Transaction_Timestamp"] >= t - pd.Timedelta(hours=24)) &
            (cust_txns["Transaction_Timestamp"] <= t)
        ].shape[0]

        last_30d  = cust_txns[
            (cust_txns["Transaction_Timestamp"] >= t - pd.Timedelta(days=30)) &
            (cust_txns["Transaction_Timestamp"] <= t)
        ].shape[0]

        daily_avg_30d = max(1, last_30d / 30)
        vci = (last_24h - daily_avg_30d) / daily_avg_30d
        vci_list.append(round(vci, 4))

    return pd.Series(vci_list, index=txn_df.index)


def compute_tps(amounts: pd.Series,
                threshold: float = REPORTING_THRESHOLD_INR) -> pd.Series:
    """
    Threshold Proximity Score = amount / threshold.
    Transactions between 90%–99% of the threshold get TPS = 1 (flag).
    Others get TPS = 0.
    """
    ratio = amounts / threshold
    return ((ratio >= 0.90) & (ratio < 1.00)).astype(float)


def compute_rtr(txn_df: pd.DataFrame) -> pd.Series:
    """
    Round-Tripping Ratio: proportion of outgoing funds that return
    from any counterparty within 48 hours.
    RTR > 0.70 is flagged.
    """
    rtr_list = []
    for _, row in txn_df.iterrows():
        sent    = row["Transaction_Amount_INR"]
        cid     = row["Sender_Customer_ID"]
        t       = row["Transaction_Timestamp"]

        returned = txn_df[
            (txn_df["Receiver_Customer_ID"] == cid) &
            (txn_df["Transaction_Timestamp"] > t) &
            (txn_df["Transaction_Timestamp"] <= t + pd.Timedelta(hours=48))
        ]["Transaction_Amount_INR"].sum()

        rtr = min(1.0, returned / max(1, sent))
        rtr_list.append(round(rtr, 4))

    return pd.Series(rtr_list, index=txn_df.index)


def compute_gre(txn_df: pd.DataFrame) -> pd.Series:
    """
    Geographic Risk Exposure: weighted composite of FATF risk for
    origin and destination countries. Cross-border gets 1.25× multiplier.
    """
    origin_risk = txn_df["Origin_Country"].map(FATF_RISK).fillna(0.10)
    dest_risk   = txn_df["Destination_Country"].map(FATF_RISK).fillna(0.10)
    cross_border = (txn_df["Origin_Country"] !=
                    txn_df["Destination_Country"]).astype(float) * 0.25

    gre = (origin_risk * 0.4 + dest_risk * 0.6) * (1 + cross_border)
    return gre.clip(0, 1).round(4)


def compute_pgds(txn_df: pd.DataFrame,
                 customers: pd.DataFrame) -> pd.Series:
    """
    Peer-Group Deviation Score: Z-score of transaction amount within
    the customer's entity-type peer group.
    PGDS > 2.5 triggers a high behavioral sub-score contribution.
    """
    merged = txn_df.merge(
        customers[["Customer_ID", "Entity_Type"]],
        left_on="Sender_Customer_ID",
        right_on="Customer_ID",
        how="left"
    )

    peer_stats = (merged.groupby("Entity_Type")["Transaction_Amount_INR"]
                  .agg(["mean", "std"]).reset_index()
                  .rename(columns={"mean": "peer_mean", "std": "peer_std"}))
    merged = merged.merge(peer_stats, on="Entity_Type", how="left")
    merged["peer_std"] = merged["peer_std"].fillna(1)

    pgds = ((merged["Transaction_Amount_INR"] - merged["peer_mean"]) /
            merged["peer_std"]).clip(-3, 5).round(4)
    return pgds.values


# ── Sub-Score Computations ─────────────────────────────────────

def compute_historical_score(txn_df: pd.DataFrame,
                              customers: pd.DataFrame) -> pd.Series:
    """
    S_Historical = 0.4×H1 + 0.4×H2 + 0.2×H3  (scaled 0–100)
    H1 = prior SAR count (proxy: static risk score × 10)
    H2 = prior confirmed alert count (proxy: anomaly label history)
    H3 = counterparty risk flag
    """
    # Proxy H1: static risk score
    merged = txn_df.merge(
        customers[["Customer_ID", "Static_Risk_Score"]],
        left_on="Sender_Customer_ID", right_on="Customer_ID", how="left")
    h1 = (merged["Static_Risk_Score"].fillna(0.2) * 100).clip(0, 100)

    # Proxy H2: how many past suspicious txns has this customer had
    past_flags = (txn_df[txn_df["Ground_Truth_Label"] == 1]
                  .groupby("Sender_Customer_ID").size()
                  .rename("flag_count"))
    merged = merged.merge(past_flags, on="Sender_Customer_ID", how="left")
    h2 = (merged["flag_count"].fillna(0).clip(0, 20) / 20 * 100)

    # H3: whether the receiver is a high-risk customer
    high_risk_ids = set(
        customers[customers["Static_Risk_Score"] > 0.6]["Customer_ID"])
    h3 = txn_df["Receiver_Customer_ID"].isin(high_risk_ids).astype(float) * 100

    score = (0.4 * h1.values + 0.4 * h2.values + 0.2 * h3.values).clip(0, 100)
    return pd.Series(score, index=txn_df.index).round(2)


def compute_behavioral_score(vci, tps, rtr, pgds) -> np.ndarray:
    """
    S_Behavioral = (0.30×B1 + 0.25×B2 + 0.20×B3 + 0.20×B4 + 0.05×B5) × 100

    B1 = VCI normalized [0,1]
    B2 = TPS binary flag
    B3 = RTR
    B4 = PGDS normalized [0,1]
    B5 = time-of-day anomaly (proxied at 0 for simplicity here;
         implemented in dashboard layer)
    """
    b1 = np.clip(vci / 10, 0, 1)
    b2 = tps.values
    b3 = np.clip(rtr.values, 0, 1)
    b4 = np.clip(pgds / 5, 0, 1)
    b5 = np.zeros(len(b1))   # placeholder; computed in real-time layer

    score = (0.30*b1 + 0.25*b2 + 0.20*b3 + 0.20*b4 + 0.05*b5) * 100
    return np.clip(score, 0, 100).round(2)


def compute_geographic_score(gre: pd.Series) -> np.ndarray:
    """
    S_Geographic: maps GRE (0–1) to a 0–100 risk score with
    breakpoints matching FATF category severity.
    """
    score = gre * 100
    return score.clip(0, 100).round(2).values


def compute_wrs(s_hist, s_behav, s_geo,
                w1=0.25, w2=0.50, w3=0.25) -> np.ndarray:
    """
    Weighted Risk Score = w1×S_Historical + w2×S_Behavioral + w3×S_Geographic
    Default weights: 0.25 / 0.50 / 0.25
    """
    return np.clip(w1*s_hist + w2*s_behav + w3*s_geo, 0, 100).round(2)


# ── Main Pipeline ──────────────────────────────────────────────

def build_feature_set(sample_size: int = 3000) -> pd.DataFrame:
    """
    Loads raw data, computes all features, and saves the enriched dataset.
    Uses a sample for performance on local hardware — full run optional.
    """
    print("\n── Feature Engineering Pipeline ──────────────────────────")

    customers = pd.read_csv(f"{OUTPUT_DIR}/customer_master.csv")
    txn_df    = pd.read_csv(f"{OUTPUT_DIR}/transaction_ledger.csv",
                             parse_dates=["Transaction_Timestamp"])

    # Use stratified sample to keep anomaly proportion
    normal    = txn_df[txn_df["Ground_Truth_Label"] == 0].sample(
        min(sample_size - 500, len(txn_df[txn_df["Ground_Truth_Label"]==0])),
        random_state=42)
    anomalous = txn_df[txn_df["Ground_Truth_Label"] == 1]
    txn_sample = pd.concat([normal, anomalous]).sample(
        frac=1, random_state=42).reset_index(drop=True)

    print(f"[✓] Working dataset: {len(txn_sample):,} transactions")

    # Compute features (VCI and RTR are O(n²) — sampled for speed)
    print("[…] Computing VCI  (Velocity Change Index)…")
    txn_sample["VCI"]  = compute_vci(txn_sample)

    print("[…] Computing TPS  (Threshold Proximity Score)…")
    txn_sample["TPS"]  = compute_tps(txn_sample["Transaction_Amount_INR"])

    print("[…] Computing RTR  (Round-Tripping Ratio)…")
    txn_sample["RTR"]  = compute_rtr(txn_sample)

    print("[…] Computing GRE  (Geographic Risk Exposure)…")
    txn_sample["GRE"]  = compute_gre(txn_sample)

    print("[…] Computing PGDS (Peer-Group Deviation Score)…")
    txn_sample["PGDS"] = compute_pgds(txn_sample, customers)

    # Compute sub-scores
    print("[…] Computing sub-scores and WRS…")
    s_hist  = compute_historical_score(txn_sample, customers).values
    s_behav = compute_behavioral_score(
        txn_sample["VCI"], txn_sample["TPS"],
        txn_sample["RTR"], txn_sample["PGDS"])
    s_geo   = compute_geographic_score(txn_sample["GRE"])

    txn_sample["S_Historical"]  = s_hist.round(2)
    txn_sample["S_Behavioral"]  = s_behav.round(2)
    txn_sample["S_Geographic"]  = s_geo.round(2)
    txn_sample["WRS"]           = compute_wrs(s_hist, s_behav, s_geo)

    # Risk tier classification
    txn_sample["Risk_Tier"] = pd.cut(
        txn_sample["WRS"],
        bins=[-1, 40, 70, 100],
        labels=["Low", "Medium", "High"])

    out_path = f"{OUTPUT_DIR}/featured_transactions.csv"
    txn_sample.to_csv(out_path, index=False)

    print(f"\n[✓] Feature set saved → {out_path}")
    print(f"\n    Risk Tier Distribution:")
    print(txn_sample["Risk_Tier"].value_counts().to_string())
    print("─" * 55)

    return txn_sample


if __name__ == "__main__":
    build_feature_set(sample_size=3000)
