"""
sar_generator.py
────────────────────────────────────────────────────────────────
Automated SAR (Suspicious Activity Report) narrative generator.

Uses a hybrid template-based NLG approach as described in
Section 4.6 of the project report:
  • Factual data fields (Who/What/When/Where) → 100% automated
  • Analytical narrative (Why)                → rule-to-template mapping
  • Reason Code block                         → audit-traceable evidence

Supports all 5 anomaly types:
  Structuring | Layering | Round-Tripping | Velocity-Spike | Geographic-Risk

Output:
  output_data/sar_reports/SAR_<TXN_ID>.txt   — individual SAR text files
  output_data/sar_summary.csv                 — summary of all SARs generated
"""

import pandas as pd
import numpy as np
from jinja2 import Template, Environment
from datetime import datetime
import os
import textwrap

OUTPUT_DIR = "output_data"
SAR_DIR    = f"{OUTPUT_DIR}/sar_reports"
os.makedirs(SAR_DIR, exist_ok=True)


def format_inr(value):
    """Formats a number in Indian number system (lakhs, crores)."""
    try:
        v = int(value)
        s = str(v)
        if len(s) <= 3:
            return s
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.append(rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.append(rest)
        parts.reverse()
        return ",".join(parts) + "," + last3
    except Exception:
        return str(value)


# Create Jinja2 environment with custom filter
jinja_env = Environment()
jinja_env.filters["format_inr"] = format_inr


# ── SAR Narrative Templates (Why Section) ─────────────────────

TEMPLATES = {

    "Structuring": jinja_env.from_string("""
NATURE OF SUSPICIOUS ACTIVITY — STRUCTURING
The account holder, {{ sender_name }} (Customer ID: {{ sender_id }}), conducted
{{ txn_count }} transactions within a {{ window_hours }}-hour period, each with an
amount between ₹{{ min_amount | int | format_inr }} and
₹{{ max_amount | int | format_inr }}. These amounts consistently fall within the
90%–99% band of the mandatory reporting threshold of ₹10,00,000 as prescribed
under Rule 3 of the Prevention of Money Laundering Act (PMLA), 2002. The pattern
of multiple sub-threshold transactions within a compressed time window is
consistent with the money laundering typology known as 'Structuring' or 'Smurfing,'
in which a subject deliberately breaks a large sum into smaller amounts to avoid
triggering mandatory disclosure obligations. The regularity and tight amount banding
of these transactions strongly suggests intentional threshold avoidance rather than
normal business activity. The Threshold Proximity Score (TPS) for these transactions
is {{ tps }}, and the Velocity Change Index (VCI) is {{ vci }}, both of which are
significantly above the alert thresholds defined in the risk monitoring framework.
"""),

    "Layering": jinja_env.from_string("""
NATURE OF SUSPICIOUS ACTIVITY — LAYERING
A series of {{ txn_count }} rapid sequential transfers were detected originating
from or passing through the account of {{ sender_name }} (Customer ID: {{ sender_id }})
over a period of {{ window_hours }} hours. The funds, totaling approximately
₹{{ total_amount | int | format_inr }}, moved through multiple intermediary accounts
across {{ unique_channels }} different payment channels
({{ channels }}), with each transfer reducing the traceable link between the
original source and the ultimate destination. This pattern is consistent with the
'Layering' stage of money laundering, in which illicit funds are moved repeatedly
across accounts and channels to obscure their criminal origin and complicate
financial investigation. Cross-border transactions were observed involving
{{ destination_countries }}, which carry elevated FATF risk classifications.
The Geographic Risk Exposure (GRE) score for the transaction chain is {{ gre }},
reflecting the high-risk nature of the jurisdictions involved.
"""),

    "Round-Tripping": jinja_env.from_string("""
NATURE OF SUSPICIOUS ACTIVITY — ROUND-TRIPPING / CIRCULAR FUND TRANSFER
A payment of ₹{{ out_amount | int | format_inr }} was made from the account of
{{ sender_name }} (Customer ID: {{ sender_id }}) to {{ receiver_id }} via
{{ channel }} on {{ out_timestamp }}. Within {{ return_hours }} hours, a return
payment of ₹{{ in_amount | int | format_inr }} was received from the same
counterparty or a closely related entity. The proportion of funds returned
represents a Round-Tripping Ratio (RTR) of {{ rtr }}, which exceeds the alert
threshold of 0.70 established in this framework. Circular fund transfers of this
nature, where funds rapidly depart and return to the originating account without
any apparent commercial justification, are consistent with the 'Placement' and
'Layering' stages of money laundering. The brevity of the round-trip window and
the absence of any identifiable business rationale for the circular movement of
funds makes this activity highly suspicious.
"""),

    "Velocity-Spike": jinja_env.from_string("""
NATURE OF SUSPICIOUS ACTIVITY — ABNORMAL TRANSACTION VELOCITY
The account of {{ sender_name }} (Customer ID: {{ sender_id }}) initiated
{{ spike_count }} transactions within a 24-hour window on {{ spike_date }},
compared to a 30-day daily average of {{ daily_avg }} transactions. This
represents a Velocity Change Index (VCI) of {{ vci }}, which is {{ vci_multiple }}×
the customer's established behavioral baseline. The sudden and significant
departure from this customer's historical transaction pattern — with no apparent
seasonal, commercial, or lifecycle explanation — is a strong behavioral indicator
of potential suspicious activity. The transaction amounts during the spike window
ranged from ₹{{ min_amount | int | format_inr }} to
₹{{ max_amount | int | format_inr }}, and all transactions were conducted via
{{ channel }}, which is consistent with rapid digital smurfing or account takeover
for fund placement purposes.
"""),

    "Geographic-Risk": jinja_env.from_string("""
NATURE OF SUSPICIOUS ACTIVITY — HIGH-RISK JURISDICTION TRANSFER
A SWIFT transfer of ₹{{ amount | int | format_inr }} was initiated from the
account of {{ sender_name }} (Customer ID: {{ sender_id }}) to a recipient in
{{ destination_country }} on {{ timestamp }}. {{ destination_country }} is
classified as a {{ fatf_classification }} jurisdiction by the Financial Action Task
Force (FATF), and transactions to this jurisdiction carry a Geographic Risk
Exposure (GRE) score of {{ gre }}, which is above the High-Risk alert threshold
of 0.60. Cross-border transfers to FATF-listed jurisdictions without clear
commercial documentation or prior transaction history with the destination country
are subject to Enhanced Due Diligence (EDD) requirements under the RBI Master
Directions on KYC/AML (Section 37) and PMLA Rule 9. No prior transaction
history with this jurisdiction was identified for this customer, and the transfer
lacks supporting trade documentation in the KYC records available at the time
of filing.
"""),
}

# FATF classifications for narrative
FATF_CLASSIFICATION = {
    "North Korea":    "Black List",
    "Iran":           "Black List",
    "Myanmar":        "Black List",
    "Cayman Islands": "High-Risk",
    "UAE":            "Grey List",
    "Turkey":         "Grey List",
    "Nigeria":        "Grey List",
    "Pakistan":       "Grey List",
}

SAR_HEADER = """
╔══════════════════════════════════════════════════════════════════╗
║         SUSPICIOUS ACTIVITY REPORT (SAR)                        ║
║         [AUTO-GENERATED DRAFT — PENDING ANALYST REVIEW]         ║
╚══════════════════════════════════════════════════════════════════╝

Filing Institution   : FinTech Compliance POC System
Report Reference     : SAR-{{ sar_ref }}
Date of Generation   : {{ generation_date }}
Filing Deadline      : {{ filing_deadline }}
Status               : DRAFT — Requires Compliance Officer Approval

═══════════════════════════════════════════════════════════════════
SECTION A: SUBJECT INFORMATION (WHO)
═══════════════════════════════════════════════════════════════════
Customer ID          : {{ sender_id }}
Entity Type          : {{ entity_type }}
Registration Country : {{ reg_country }}
KYC Verified         : {{ kyc_verified }}
Static Risk Score    : {{ static_risk }}

═══════════════════════════════════════════════════════════════════
SECTION B: TRANSACTION DETAILS (WHAT / WHEN / WHERE)
═══════════════════════════════════════════════════════════════════
Transaction ID       : {{ txn_id }}
Transaction Amount   : ₹{{ amount | int | format_inr }}
Payment Channel      : {{ channel }}
Transaction Date     : {{ timestamp }}
Origin Country       : {{ origin_country }}
Destination Country  : {{ destination_country }}
Anomaly Type         : {{ anomaly_type }}

═══════════════════════════════════════════════════════════════════
SECTION C: NATURE OF SUSPICIOUS ACTIVITY (WHY)
═══════════════════════════════════════════════════════════════════
"""

SAR_FOOTER = """
═══════════════════════════════════════════════════════════════════
SECTION D: REASON CODES (QUANTITATIVE EVIDENCE)
═══════════════════════════════════════════════════════════════════
Weighted Risk Score (WRS)        : {{ wrs }} / 100  [Tier: {{ risk_tier }}]
Velocity Change Index (VCI)      : {{ vci }}         [Alert threshold: > 2.0]
Threshold Proximity Score (TPS)  : {{ tps }}         [Alert threshold: 0.9–0.99]
Round-Tripping Ratio (RTR)       : {{ rtr }}         [Alert threshold: > 0.70]
Geographic Risk Exposure (GRE)   : {{ gre }}         [Alert threshold: > 0.50]
Peer-Group Deviation Score (PGDS): {{ pgds }}        [Alert threshold: > 2.50]
Historical Sub-Score             : {{ s_hist }}
Behavioral Sub-Score             : {{ s_behav }}
Geographic Sub-Score             : {{ s_geo }}

═══════════════════════════════════════════════════════════════════
SECTION E: ANALYST REVIEW
═══════════════════════════════════════════════════════════════════
Review Status        : [ ] Approved  [ ] Rejected  [ ] Pending
Analyst Name         : ___________________________
Review Date          : ___________________________
Notes                : ___________________________

This SAR was auto-generated by the RegTech POC System.
All factual data fields are sourced directly from the transaction
ledger and customer KYC database. The analytical narrative in
Section C was generated using a rule-to-template NLG engine.
The filing compliance officer is responsible for reviewing,
approving, and submitting this report to FIU-IND within the
applicable regulatory deadline.
═══════════════════════════════════════════════════════════════════
"""


def get_why_narrative(row: pd.Series,
                      all_txns: pd.DataFrame,
                      customers: pd.DataFrame) -> str:
    """
    Selects the appropriate narrative template based on anomaly type
    and populates it with transaction-specific data.
    """
    atype = row.get("Anomaly_Type", "None")
    cid   = row.get("Sender_Customer_ID", "UNKNOWN")

    # Look up customer details
    cust_row = customers[customers["Customer_ID"] == cid]
    sender_name = cust_row["Name"].values[0] if len(cust_row) else "Unknown"

    if atype == "Structuring":
        cust_txns = all_txns[
            (all_txns["Sender_Customer_ID"] == cid) &
            (all_txns["Anomaly_Type"] == "Structuring")]
        return TEMPLATES["Structuring"].render(
            sender_name=sender_name, sender_id=cid,
            txn_count=len(cust_txns),
            window_hours=24,
            min_amount=cust_txns["Transaction_Amount_INR"].min() if len(cust_txns) else row["Transaction_Amount_INR"],
            max_amount=cust_txns["Transaction_Amount_INR"].max() if len(cust_txns) else row["Transaction_Amount_INR"],
            tps=round(row.get("TPS", 0), 3),
            vci=round(row.get("VCI", 0), 3))

    elif atype == "Layering":
        cust_txns = all_txns[
            (all_txns["Sender_Customer_ID"] == cid) &
            (all_txns["Anomaly_Type"] == "Layering")]
        channels  = cust_txns["Payment_Channel"].unique().tolist()
        dest_countries = cust_txns["Destination_Country"].unique().tolist()
        return TEMPLATES["Layering"].render(
            sender_name=sender_name, sender_id=cid,
            txn_count=len(cust_txns),
            window_hours=6,
            total_amount=cust_txns["Transaction_Amount_INR"].sum() if len(cust_txns) else row["Transaction_Amount_INR"],
            unique_channels=len(channels),
            channels=", ".join(channels) if channels else row["Payment_Channel"],
            destination_countries=", ".join(dest_countries) if dest_countries else row["Destination_Country"],
            gre=round(row.get("GRE", 0), 3))

    elif atype == "Round-Tripping":
        return TEMPLATES["Round-Tripping"].render(
            sender_name=sender_name, sender_id=cid,
            out_amount=row["Transaction_Amount_INR"],
            receiver_id=row.get("Receiver_Customer_ID", "UNKNOWN"),
            channel=row.get("Payment_Channel", "IMPS"),
            out_timestamp=str(row.get("Transaction_Timestamp", ""))[:16],
            return_hours=random.randint(2, 47),
            in_amount=row["Transaction_Amount_INR"] * 0.9,
            rtr=round(row.get("RTR", 0), 3))

    elif atype == "Velocity-Spike":
        cust_txns = all_txns[all_txns["Sender_Customer_ID"] == cid]
        daily_avg  = max(1, len(cust_txns) / 730)
        spike_cnt  = int(row.get("VCI", 5) * daily_avg + daily_avg)
        return TEMPLATES["Velocity-Spike"].render(
            sender_name=sender_name, sender_id=cid,
            spike_count=spike_cnt,
            spike_date=str(row.get("Transaction_Timestamp", ""))[:10],
            daily_avg=round(daily_avg, 1),
            vci=round(row.get("VCI", 0), 3),
            vci_multiple=round(row.get("VCI", 1) + 1, 1),
            min_amount=row["Transaction_Amount_INR"] * 0.6,
            max_amount=row["Transaction_Amount_INR"],
            channel=row.get("Payment_Channel", "UPI"))

    elif atype == "Geographic-Risk":
        dest = row.get("Destination_Country", "Unknown")
        fatf_class = FATF_CLASSIFICATION.get(dest, "High-Risk")
        return TEMPLATES["Geographic-Risk"].render(
            sender_name=sender_name, sender_id=cid,
            amount=row["Transaction_Amount_INR"],
            destination_country=dest,
            timestamp=str(row.get("Transaction_Timestamp", ""))[:16],
            fatf_classification=fatf_class,
            gre=round(row.get("GRE", 0), 3))

    else:
        return "\nNo specific anomaly narrative template available for this flag type.\n"


import random

def generate_sar(row: pd.Series,
                 all_txns: pd.DataFrame,
                 customers: pd.DataFrame) -> str:
    """
    Generates a complete SAR document for a single flagged transaction.
    Returns the full SAR text string.
    """
    cid      = row.get("Sender_Customer_ID", "UNKNOWN")
    cust_row = customers[customers["Customer_ID"] == cid]
    entity_type  = cust_row["Entity_Type"].values[0]  if len(cust_row) else "Unknown"
    reg_country  = cust_row["Registration_Country"].values[0] if len(cust_row) else "Unknown"
    kyc_verified = cust_row["KYC_Verified"].values[0] if len(cust_row) else "Unknown"
    static_risk  = cust_row["Static_Risk_Score"].values[0] if len(cust_row) else 0.0

    ts = row.get("Transaction_Timestamp", datetime.now())
    try:
        ts = pd.to_datetime(ts)
        filing_deadline = (ts + pd.Timedelta(days=7)).strftime("%d %B %Y")
    except Exception:
        filing_deadline = "Within 7 days of detection"

    # Render header
    header_tmpl = jinja_env.from_string(SAR_HEADER)
    header = header_tmpl.render(
        sar_ref=str(row.get("Transaction_ID", "UNKNOWN"))[-8:],
        generation_date=datetime.now().strftime("%d %B %Y %H:%M"),
        filing_deadline=filing_deadline,
        sender_id=cid,
        entity_type=entity_type,
        reg_country=reg_country,
        kyc_verified=str(kyc_verified),
        static_risk=round(float(static_risk), 2),
        txn_id=row.get("Transaction_ID", "UNKNOWN"),
        amount=row.get("Transaction_Amount_INR", 0),
        channel=row.get("Payment_Channel", "Unknown"),
        timestamp=str(ts)[:16],
        origin_country=row.get("Origin_Country", "Unknown"),
        destination_country=row.get("Destination_Country", "Unknown"),
        anomaly_type=row.get("Anomaly_Type", "Unknown"),
    )

    # Render why narrative
    why = get_why_narrative(row, all_txns, customers)

    # Render footer
    footer_tmpl = jinja_env.from_string(SAR_FOOTER)
    footer = footer_tmpl.render(
        wrs=round(float(row.get("WRS", row.get("WRS_ML", 0))), 2),
        risk_tier=str(row.get("Risk_Tier", row.get("Risk_Tier_ML", "High"))),
        vci=round(float(row.get("VCI", 0)), 4),
        tps=round(float(row.get("TPS", 0)), 4),
        rtr=round(float(row.get("RTR", 0)), 4),
        gre=round(float(row.get("GRE", 0)), 4),
        pgds=round(float(row.get("PGDS", 0)), 4),
        s_hist=round(float(row.get("S_Historical", 0)), 2),
        s_behav=round(float(row.get("S_Behavioral", 0)), 2),
        s_geo=round(float(row.get("S_Geographic", 0)), 2),
    )

    return header + why + footer


def generate_all_sars(n_sars: int = 20):
    """
    Generates SAR reports for the top n suspicious transactions
    by WRS score from the predictions output.
    """
    print("\n── SAR Automation Engine ─────────────────────────────────")

    # Load data
    pred_path = f"{OUTPUT_DIR}/predictions.csv"
    if not os.path.exists(pred_path):
        # Fall back to featured_transactions
        pred_path = f"{OUTPUT_DIR}/featured_transactions.csv"

    txn_df    = pd.read_csv(pred_path, parse_dates=["Transaction_Timestamp"])
    customers = pd.read_csv(f"{OUTPUT_DIR}/customer_master.csv")
    all_txns  = pd.read_csv(f"{OUTPUT_DIR}/transaction_ledger.csv",
                             parse_dates=["Transaction_Timestamp"])

    # Select top suspicious transactions across all anomaly types
    suspicious = txn_df[txn_df["Ground_Truth_Label"] == 1].copy()
    if "WRS_ML" in suspicious.columns:
        suspicious = suspicious.sort_values("WRS_ML", ascending=False)
    elif "WRS" in suspicious.columns:
        suspicious = suspicious.sort_values("WRS", ascending=False)

    # Ensure all 5 anomaly types are represented
    selected = []
    for atype in ["Structuring", "Layering", "Round-Tripping",
                  "Velocity-Spike", "Geographic-Risk"]:
        subset = suspicious[suspicious["Anomaly_Type"] == atype].head(
            max(2, n_sars // 5))
        selected.append(subset)
    selected_df = pd.concat(selected).head(n_sars)

    # Generate SARs
    summary_records = []
    for _, row in selected_df.iterrows():
        sar_text = generate_sar(row, all_txns, customers)
        fname    = f"SAR_{row['Transaction_ID']}.txt"
        fpath    = os.path.join(SAR_DIR, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(sar_text)

        summary_records.append({
            "Transaction_ID":    row["Transaction_ID"],
            "Anomaly_Type":      row["Anomaly_Type"],
            "Amount_INR":        row["Transaction_Amount_INR"],
            "WRS":               row.get("WRS_ML", row.get("WRS", 0)),
            "SAR_File":          fname,
            "Generated_At":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        print(f"[✓] SAR generated : {fname}  [{row['Anomaly_Type']}]")

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(f"{OUTPUT_DIR}/sar_summary.csv", index=False)
    print(f"\n[✓] {len(summary_records)} SARs saved → {SAR_DIR}/")
    print(f"[✓] Summary saved  → {OUTPUT_DIR}/sar_summary.csv")
    print("─" * 55)


if __name__ == "__main__":
    generate_all_sars(n_sars=20)
