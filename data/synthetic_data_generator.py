"""
synthetic_data_generator.py
────────────────────────────────────────────────────────────────
Generates a realistic synthetic FinTech AML dataset calibrated to
Indian regulatory thresholds (PMLA / RBI) and FATF typologies.

Produces:
  output_data/customer_master.csv      — 200 synthetic customer profiles
  output_data/transaction_ledger.csv   — ~10,000 transaction records

Anomaly types injected:
  1. Structuring       — multiple txns just below ₹10L threshold
  2. Layering          — rapid multi-hop transfers across accounts
  3. Round-Tripping    — funds leave and return within 48 hours
  4. Velocity Spike    — sudden surge in transaction frequency
  5. Geographic Risk   — transactions to FATF Grey/Black list countries
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os

fake = Faker("en_IN")
np.random.seed(42)
random.seed(42)

# ── Constants ──────────────────────────────────────────────────
REPORTING_THRESHOLD_INR = 1_000_000   # ₹10 lakh (PMLA Rule 3)
OUTPUT_DIR = "output_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FATF jurisdiction risk weights
FATF_RISK = {
    "India":          0.10,
    "USA":            0.10,
    "UK":             0.10,
    "Germany":        0.10,
    "Singapore":      0.10,
    "UAE":            0.50,   # Grey List (as of 2024)
    "Turkey":         0.50,   # Grey List
    "Nigeria":        0.50,   # Grey List
    "Pakistan":       0.50,   # Grey List
    "North Korea":    1.00,   # Black List
    "Iran":           1.00,   # Black List
    "Myanmar":        1.00,   # Black List
    "Cayman Islands": 0.60,   # High-risk jurisdiction
}

PAYMENT_CHANNELS = ["UPI", "IMPS", "NEFT", "RTGS", "Card", "SWIFT"]
ENTITY_TYPES     = ["Individual", "SME", "Corporate", "PEP"]


# ── 1. CUSTOMER MASTER TABLE ───────────────────────────────────

def generate_customers(n: int = 200) -> pd.DataFrame:
    """
    Creates n synthetic customer profiles with static risk scores
    derived from KYC attributes (entity type, geography, PEP status).
    """
    records = []
    for i in range(1, n + 1):
        entity_type  = random.choices(
            ENTITY_TYPES, weights=[55, 25, 15, 5])[0]
        country      = random.choices(
            list(FATF_RISK.keys()),
            weights=[60,5,5,5,5,5,5,2,2,1,1,1,3])[0]
        geo_risk     = FATF_RISK[country]

        # Static risk score: blend of entity type + geography
        base_risk = {
            "Individual": 0.15,
            "SME":        0.25,
            "Corporate":  0.20,
            "PEP":        0.75,
        }[entity_type]
        static_risk = round(min(1.0, base_risk + geo_risk * 0.4 +
                                random.uniform(-0.05, 0.05)), 2)

        # Typical transaction behaviour per entity type
        avg_count, avg_amount = {
            "Individual": (random.randint(5,  30),  random.randint(5_000,   200_000)),
            "SME":        (random.randint(20, 100), random.randint(50_000,  800_000)),
            "Corporate":  (random.randint(50, 300), random.randint(200_000, 5_000_000)),
            "PEP":        (random.randint(3,  20),  random.randint(100_000, 2_000_000)),
        }[entity_type]

        records.append({
            "Customer_ID":           f"C{i:04d}",
            "Name":                  fake.name(),
            "Entity_Type":           entity_type,
            "Registration_Country":  country,
            "Static_Risk_Score":     static_risk,
            "Account_Open_Date":     fake.date_between(
                                        start_date="-3y",
                                        end_date="-3m"),
            "Avg_Monthly_Txn_Count": avg_count,
            "Avg_Txn_Amount_INR":    avg_amount,
            "KYC_Verified":          random.choice([True, True, True, False]),
            "Occupation":            fake.job(),
            "City":                  fake.city(),
        })

    df = pd.DataFrame(records)
    df.to_csv(f"{OUTPUT_DIR}/customer_master.csv", index=False)
    print(f"[✓] Generated {len(df)} customer profiles → customer_master.csv")
    return df


# ── 2. NORMAL TRANSACTION GENERATOR ───────────────────────────

def _random_timestamp(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def generate_normal_transactions(customers: pd.DataFrame,
                                 n_per_customer_avg: int = 40) -> list[dict]:
    """
    Generates baseline normal transactions for every customer.
    Amounts, channels, and timestamps follow realistic distributions.
    """
    txns = []
    txn_id = 1
    start_dt = datetime(2024, 1, 1)
    end_dt   = datetime(2026, 1, 1)
    customer_ids = customers["Customer_ID"].tolist()

    for _, cust in customers.iterrows():
        n_txns = max(5, int(np.random.normal(
            cust["Avg_Monthly_Txn_Count"] * 24,   # 24-month window
            cust["Avg_Monthly_Txn_Count"] * 5)))

        for _ in range(n_txns):
            amount = max(100, int(np.random.lognormal(
                np.log(cust["Avg_Txn_Amount_INR"]), 0.6)))
            channel = random.choices(
                PAYMENT_CHANNELS,
                weights=[40, 25, 15, 5, 10, 5])[0]
            dest_country = random.choices(
                list(FATF_RISK.keys()),
                weights=[70,4,4,4,4,3,2,2,2,1,1,1,2])[0]
            receiver = random.choice(
                [c for c in customer_ids if c != cust["Customer_ID"]])

            txns.append({
                "Transaction_ID":      f"T{txn_id:07d}",
                "Sender_Customer_ID":  cust["Customer_ID"],
                "Receiver_Customer_ID": receiver,
                "Transaction_Amount_INR": amount,
                "Transaction_Timestamp":  _random_timestamp(start_dt, end_dt),
                "Payment_Channel":     channel,
                "Origin_Country":      cust["Registration_Country"],
                "Destination_Country": dest_country,
                "Ground_Truth_Label":  0,
                "Anomaly_Type":        "None",
            })
            txn_id += 1

    return txns


# ── 3. ANOMALY INJECTION ───────────────────────────────────────

def inject_structuring(customers: pd.DataFrame,
                       n_cases: int = 30) -> list[dict]:
    """
    STRUCTURING: Multiple transactions just below the ₹10L PMLA threshold.
    Targets high-risk customers. Each case = 3–8 rapid sub-threshold txns.
    """
    txns = []
    txn_id = 9_000_001
    high_risk = customers[customers["Static_Risk_Score"] > 0.4]
    targets = high_risk.sample(min(n_cases, len(high_risk)),
                               random_state=42)

    for _, cust in targets.iterrows():
        base_time = _random_timestamp(
            datetime(2024, 6, 1), datetime(2025, 12, 1))
        n_splits  = random.randint(3, 8)

        for j in range(n_splits):
            # Amount: 90–99% of ₹10L threshold (classic structuring signature)
            amount = random.randint(
                int(REPORTING_THRESHOLD_INR * 0.90),
                int(REPORTING_THRESHOLD_INR * 0.99))

            txns.append({
                "Transaction_ID":      f"T{txn_id:07d}",
                "Sender_Customer_ID":  cust["Customer_ID"],
                "Receiver_Customer_ID": f"C{random.randint(1,200):04d}",
                "Transaction_Amount_INR": amount,
                "Transaction_Timestamp":  base_time + timedelta(hours=j*2),
                "Payment_Channel":     random.choice(["UPI", "IMPS", "NEFT"]),
                "Origin_Country":      cust["Registration_Country"],
                "Destination_Country": "India",
                "Ground_Truth_Label":  1,
                "Anomaly_Type":        "Structuring",
            })
            txn_id += 1

    print(f"[✓] Injected {len(txns)} structuring transactions ({n_cases} cases)")
    return txns


def inject_layering(customers: pd.DataFrame,
                    n_cases: int = 20) -> list[dict]:
    """
    LAYERING: Large funds move rapidly through a chain of accounts
    to obscure origin. Each case = 4–6 rapid sequential hops.
    """
    txns = []
    txn_id = 9_100_001
    customer_ids = customers["Customer_ID"].tolist()
    targets = customers[
        customers["Entity_Type"].isin(["Corporate", "PEP"])
    ].sample(min(n_cases, len(customers[
        customers["Entity_Type"].isin(["Corporate", "PEP"])])),
        random_state=1)

    for _, cust in targets.iterrows():
        base_time  = _random_timestamp(
            datetime(2024, 3, 1), datetime(2025, 10, 1))
        amount     = random.randint(500_000, 5_000_000)
        chain_len  = random.randint(4, 6)
        chain      = random.sample(customer_ids, chain_len)

        for j in range(len(chain) - 1):
            txns.append({
                "Transaction_ID":      f"T{txn_id:07d}",
                "Sender_Customer_ID":  chain[j],
                "Receiver_Customer_ID": chain[j+1],
                "Transaction_Amount_INR": int(amount * random.uniform(0.85, 1.0)),
                "Transaction_Timestamp":  base_time + timedelta(minutes=j*45),
                "Payment_Channel":     random.choice(["SWIFT", "NEFT", "RTGS"]),
                "Origin_Country":      random.choice(["UAE", "Cayman Islands", "India"]),
                "Destination_Country": random.choice(["India", "UAE", "Singapore"]),
                "Ground_Truth_Label":  1,
                "Anomaly_Type":        "Layering",
            })
            txn_id += 1

    print(f"[✓] Injected {len(txns)} layering transactions ({n_cases} cases)")
    return txns


def inject_round_tripping(customers: pd.DataFrame,
                          n_cases: int = 20) -> list[dict]:
    """
    ROUND-TRIPPING: Funds leave an account and return from a related
    party within 48 hours — signature of circular placement schemes.
    """
    txns = []
    txn_id = 9_200_001
    customer_ids = customers["Customer_ID"].tolist()

    for i in range(n_cases):
        sender   = random.choice(customer_ids)
        receiver = random.choice([c for c in customer_ids if c != sender])
        amount   = random.randint(200_000, 3_000_000)
        base_time = _random_timestamp(
            datetime(2024, 4, 1), datetime(2025, 11, 1))

        # Outgoing leg
        txns.append({
            "Transaction_ID":      f"T{txn_id:07d}",
            "Sender_Customer_ID":  sender,
            "Receiver_Customer_ID": receiver,
            "Transaction_Amount_INR": amount,
            "Transaction_Timestamp":  base_time,
            "Payment_Channel":     random.choice(["IMPS", "NEFT"]),
            "Origin_Country":      "India",
            "Destination_Country": "India",
            "Ground_Truth_Label":  1,
            "Anomaly_Type":        "Round-Tripping",
        })
        txn_id += 1

        # Return leg within 48 hours (70–100% of original amount)
        txns.append({
            "Transaction_ID":      f"T{txn_id:07d}",
            "Sender_Customer_ID":  receiver,
            "Receiver_Customer_ID": sender,
            "Transaction_Amount_INR": int(amount * random.uniform(0.70, 1.00)),
            "Transaction_Timestamp":  base_time + timedelta(
                hours=random.randint(1, 47)),
            "Payment_Channel":     random.choice(["IMPS", "UPI"]),
            "Origin_Country":      "India",
            "Destination_Country": "India",
            "Ground_Truth_Label":  1,
            "Anomaly_Type":        "Round-Tripping",
        })
        txn_id += 1

    print(f"[✓] Injected {len(txns)} round-tripping transactions ({n_cases} cases)")
    return txns


def inject_velocity_spike(customers: pd.DataFrame,
                          n_cases: int = 25) -> list[dict]:
    """
    VELOCITY SPIKE: A customer makes an unusually high number of
    transactions in a 24-hour window — far above their 30-day average.
    """
    txns = []
    txn_id = 9_300_001
    customer_ids = customers["Customer_ID"].tolist()
    targets = customers[
        customers["Avg_Monthly_Txn_Count"] < 30
    ].sample(min(n_cases, 80), random_state=2)

    for _, cust in targets.iterrows():
        base_time = _random_timestamp(
            datetime(2024, 5, 1), datetime(2025, 12, 1))
        # Spike: 10–20x the customer's normal daily rate
        n_spike = random.randint(
            int(cust["Avg_Monthly_Txn_Count"] * 10),
            int(cust["Avg_Monthly_Txn_Count"] * 20))
        n_spike = max(15, min(n_spike, 60))

        for j in range(n_spike):
            txns.append({
                "Transaction_ID":      f"T{txn_id:07d}",
                "Sender_Customer_ID":  cust["Customer_ID"],
                "Receiver_Customer_ID": random.choice(customer_ids),
                "Transaction_Amount_INR": random.randint(
                    5_000, int(REPORTING_THRESHOLD_INR * 0.5)),
                "Transaction_Timestamp":  base_time + timedelta(
                    minutes=j * random.randint(5, 30)),
                "Payment_Channel":     "UPI",
                "Origin_Country":      cust["Registration_Country"],
                "Destination_Country": "India",
                "Ground_Truth_Label":  1,
                "Anomaly_Type":        "Velocity-Spike",
            })
            txn_id += 1

    print(f"[✓] Injected {len(txns)} velocity-spike transactions ({n_cases} cases)")
    return txns


def inject_geographic_risk(customers: pd.DataFrame,
                           n_cases: int = 25) -> list[dict]:
    """
    GEOGRAPHIC RISK: Transactions to/from FATF Black List or high-risk
    jurisdictions (Iran, North Korea, Myanmar, Cayman Islands).
    """
    txns = []
    txn_id = 9_400_001
    HIGH_RISK_COUNTRIES = ["Iran", "North Korea", "Myanmar", "Cayman Islands"]
    customer_ids = customers["Customer_ID"].tolist()

    for i in range(n_cases):
        sender = random.choice(customer_ids)
        txns.append({
            "Transaction_ID":      f"T{txn_id:07d}",
            "Sender_Customer_ID":  sender,
            "Receiver_Customer_ID": random.choice(customer_ids),
            "Transaction_Amount_INR": random.randint(50_000, 2_000_000),
            "Transaction_Timestamp":  _random_timestamp(
                datetime(2024, 1, 1), datetime(2026, 1, 1)),
            "Payment_Channel":     "SWIFT",
            "Origin_Country":      "India",
            "Destination_Country": random.choice(HIGH_RISK_COUNTRIES),
            "Ground_Truth_Label":  1,
            "Anomaly_Type":        "Geographic-Risk",
        })
        txn_id += 1

    print(f"[✓] Injected {len(txns)} geographic-risk transactions ({n_cases} cases)")
    return txns


# ── 4. ASSEMBLE AND SAVE ───────────────────────────────────────

def generate_full_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n── Generating Synthetic RegTech Dataset ──────────────────")
    customers = generate_customers(200)

    all_txns = generate_normal_transactions(customers, n_per_customer_avg=40)
    all_txns += inject_structuring(customers,     n_cases=30)
    all_txns += inject_layering(customers,        n_cases=20)
    all_txns += inject_round_tripping(customers,  n_cases=20)
    all_txns += inject_velocity_spike(customers,  n_cases=25)
    all_txns += inject_geographic_risk(customers, n_cases=25)

    txn_df = pd.DataFrame(all_txns)
    txn_df["Transaction_Timestamp"] = pd.to_datetime(
        txn_df["Transaction_Timestamp"])
    txn_df = txn_df.sample(frac=1, random_state=42).reset_index(drop=True)

    txn_df.to_csv(f"{OUTPUT_DIR}/transaction_ledger.csv", index=False)

    normal_count    = (txn_df["Ground_Truth_Label"] == 0).sum()
    suspicious_count = (txn_df["Ground_Truth_Label"] == 1).sum()
    print(f"\n[✓] Total transactions : {len(txn_df):,}")
    print(f"    Normal             : {normal_count:,} "
          f"({normal_count/len(txn_df)*100:.1f}%)")
    print(f"    Suspicious         : {suspicious_count:,} "
          f"({suspicious_count/len(txn_df)*100:.1f}%)")
    print(f"\n    Anomaly breakdown:")
    print(txn_df[txn_df["Ground_Truth_Label"]==1]
          ["Anomaly_Type"].value_counts().to_string())
    print(f"\n[✓] Saved → {OUTPUT_DIR}/transaction_ledger.csv")
    print("─" * 55)

    return customers, txn_df


if __name__ == "__main__":
    generate_full_dataset()
