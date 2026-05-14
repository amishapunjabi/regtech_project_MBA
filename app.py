"""
dashboard/app.py
────────────────────────────────────────────────────────────────
RegTech Compliance Dashboard — Streamlit Application

Three views as described in Annexure 4 of the project report:
  View 1: Executive Summary        (Chief Compliance Officer)
  View 2: Investigator Workbench   (Compliance Analyst)
  View 3: Trend Analysis           (Risk Manager)

Run:  streamlit run dashboard/app.py
────────────────────────────────────────────────────────────────
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime, timedelta

# ── Path setup ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, "output_data")
SAR_DIR    = os.path.join(OUTPUT_DIR, "sar_reports")

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="RegTech Compliance Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded")

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #1e1e1e; color: #ffffff; }
    .metric-card {
        background: #2d2d2d; border-radius: 10px;
        padding: 20px; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        border-left: 5px solid #1F3864;
        color: #ffffff;
    }
    .metric-value { font-size: 2.2em; font-weight: bold; color: #ffffff; }
    .metric-label { font-size: 0.9em; color: #cccccc; margin-top: 4px; }
    .high-risk   { border-left-color: #ff6b6b !important; }
    .medium-risk { border-left-color: #ffa726 !important; }
    .low-risk    { border-left-color: #66bb6a !important; }
    .sar-box {
        background: #2d2d2d; border: 1px solid #555555;
        border-radius: 8px; padding: 15px;
        font-family: monospace; font-size: 0.75em;
        white-space: pre-wrap; max-height: 500px;
        overflow-y: auto; color: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #3d3d3d; border-radius: 6px 6px 0 0;
        font-weight: 600; color: #ffffff;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1F3864 !important; color: white !important;
    }
    /* Additional dark theme overrides */
    .stMarkdown, .stText, .stHeader, .stSubheader { color: #ffffff; }
    .stDataFrame { background-color: #2d2d2d; color: #ffffff; }
    .stTable { background-color: #2d2d2d; color: #ffffff; }
    .stSidebar { background-color: #252526; color: #ffffff; }
    .stSidebar .stMarkdown { color: #ffffff; }
    .stSelectbox, .stMultiselect, .stSlider { color: #ffffff; }
    .stButton button { background-color: #1F3864; color: #ffffff; }
</style>
""", unsafe_allow_html=True)


# ── Data Loaders ───────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_data():
    """Loads all data files with fallback handling."""
    data = {}

    # Predictions (post-model) or featured transactions (pre-model)
    for fname in ["predictions.csv", "featured_transactions.csv"]:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["Transaction_Timestamp"])
            # Normalize WRS column name
            if "WRS_ML" in df.columns and "WRS" not in df.columns:
                df["WRS"] = df["WRS_ML"]
            if "Risk_Tier_ML" in df.columns and "Risk_Tier" not in df.columns:
                df["Risk_Tier"] = df["Risk_Tier_ML"]
            data["transactions"] = df
            break

    if "transactions" not in data:
        st.error("⚠️ No transaction data found. Run the pipeline first.")
        st.stop()

    # Customers
    cust_path = os.path.join(OUTPUT_DIR, "customer_master.csv")
    if os.path.exists(cust_path):
        data["customers"] = pd.read_csv(cust_path)

    # SAR Summary
    sar_path = os.path.join(OUTPUT_DIR, "sar_summary.csv")
    if os.path.exists(sar_path):
        data["sar_summary"] = pd.read_csv(sar_path)

    # Model evaluation
    eval_path = os.path.join(OUTPUT_DIR, "model_evaluation.csv")
    if os.path.exists(eval_path):
        data["evaluation"] = pd.read_csv(eval_path)

    return data


def load_sar_text(txn_id: str) -> str:
    """Loads a SAR text file by transaction ID."""
    path = os.path.join(SAR_DIR, f"SAR_{txn_id}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "SAR report not yet generated for this transaction."


# ── Sidebar ────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.image("https://img.icons8.com/color/96/shield.png", width=60)
    st.sidebar.title("🛡️ RegTech POC")
    st.sidebar.markdown("**Amisha Punjabi | 2024MB53031**")
    st.sidebar.markdown("MBA FinTech — BITS Pilani")
    st.sidebar.divider()

    view = st.sidebar.radio(
        "Select Dashboard View",
        ["📊 Executive Summary",
         "🔍 Investigator Workbench",
         "📈 Trend Analysis"],
        index=0)

    st.sidebar.divider()
    st.sidebar.markdown("**Risk Tier Filter**")
    tiers = st.sidebar.multiselect(
        "Show tiers:",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"])

    st.sidebar.markdown("**Anomaly Type Filter**")
    anomaly_types = st.sidebar.multiselect(
        "Show anomaly types:",
        ["Structuring", "Layering", "Round-Tripping",
         "Velocity-Spike", "Geographic-Risk", "None"],
        default=["Structuring", "Layering", "Round-Tripping",
                 "Velocity-Spike", "Geographic-Risk"])

    return view, tiers, anomaly_types


# ── View 1: Executive Summary ──────────────────────────────────

def render_executive_summary(df: pd.DataFrame, data: dict):
    st.title("📊 Executive Summary")
    st.markdown("*Chief Compliance Officer view — Overall compliance health at a glance*")
    st.divider()

    # KPI Scorecards (F-Pattern: top row)
    suspicious = df[df["Ground_Truth_Label"] == 1]
    high_risk  = df[df["Risk_Tier"] == "High"]
    med_risk   = df[df["Risk_Tier"] == "Medium"]

    sar_count  = len(data.get("sar_summary", pd.DataFrame()))
    avg_wrs    = df["WRS"].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card high-risk">
            <div class="metric-value">{len(high_risk)}</div>
            <div class="metric-label">🔴 High-Risk Alerts</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card medium-risk">
            <div class="metric-value">{len(med_risk)}</div>
            <div class="metric-label">🟡 Medium-Risk Alerts</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{sar_count}</div>
            <div class="metric-label">📄 SARs Generated</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_wrs:.1f}</div>
            <div class="metric-label">📊 Avg Portfolio WRS</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Anomaly breakdown + Risk tier donut
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("🔎 Anomaly Type Breakdown")
        anom_counts = (suspicious["Anomaly_Type"]
                       .value_counts().reset_index())
        anom_counts.columns = ["Anomaly Type", "Count"]
        fig_bar = px.bar(
            anom_counts, x="Count", y="Anomaly Type",
            orientation="h", color="Anomaly Type",
            color_discrete_sequence=px.colors.qualitative.Bold,
            template="plotly_white")
        fig_bar.update_layout(
            showlegend=False, height=320,
            yaxis_title="", xaxis_title="Transaction Count",
            margin=dict(l=0, r=0, t=20, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.subheader("📊 Risk Tier Distribution")
        tier_counts = df["Risk_Tier"].value_counts().reset_index()
        tier_counts.columns = ["Risk Tier", "Count"]
        fig_donut = px.pie(
            tier_counts, values="Count", names="Risk Tier",
            hole=0.55,
            color="Risk Tier",
            color_discrete_map={
                "High": "#d32f2f",
                "Medium": "#f57c00",
                "Low": "#388e3c"},
            template="plotly_white")
        fig_donut.update_layout(
            height=320, margin=dict(l=0, r=0, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2))
        st.plotly_chart(fig_donut, use_container_width=True)

    # Row 3: Transaction amount distribution + WRS heatmap
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("💰 Transaction Amount Distribution by Risk Tier")
        fig_box = px.box(
            df[df["Risk_Tier"].notna()],
            x="Risk_Tier", y="Transaction_Amount_INR",
            color="Risk_Tier",
            color_discrete_map={
                "High":"#d32f2f","Medium":"#f57c00","Low":"#388e3c"},
            template="plotly_white",
            category_orders={"Risk_Tier": ["High","Medium","Low"]})
        fig_box.update_layout(
            height=320, showlegend=False,
            yaxis_title="Transaction Amount (₹)",
            xaxis_title="Risk Tier",
            margin=dict(l=0, r=0, t=20, b=20))
        st.plotly_chart(fig_box, use_container_width=True)

    with col_b:
        st.subheader("🌍 Geographic Risk — Top Destination Countries")
        geo_risk = (df[df["Ground_Truth_Label"]==1]
                    .groupby("Destination_Country")["GRE"]
                    .mean().reset_index()
                    .sort_values("GRE", ascending=False)
                    .head(10))
        geo_risk.columns = ["Country", "Avg GRE"]
        fig_geo = px.bar(
            geo_risk, x="Avg GRE", y="Country",
            orientation="h",
            color="Avg GRE",
            color_continuous_scale="Reds",
            template="plotly_white")
        fig_geo.update_layout(
            height=320, showlegend=False,
            margin=dict(l=0, r=0, t=20, b=20),
            coloraxis_showscale=False)
        st.plotly_chart(fig_geo, use_container_width=True)

    # Model performance panel (if available)
    if "evaluation" in data:
        st.divider()
        st.subheader("🤖 ML Model Performance")
        ev = data["evaluation"].iloc[0]
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("AUC-ROC", f"{ev.get('AUC_ROC',0):.4f}")
        e2.metric("Precision", f"{ev.get('Precision_Suspicious',0):.4f}")
        e3.metric("Recall", f"{ev.get('Recall_Suspicious',0):.4f}")
        e4.metric("False Discovery Rate", f"{ev.get('FDR',0):.4f}")
        e5.metric("True Positives", int(ev.get("True_Positives", 0)))


# ── View 2: Investigator Workbench ─────────────────────────────

def render_investigator_workbench(df: pd.DataFrame,
                                  customers: pd.DataFrame,
                                  tiers: list, anomaly_types: list):
    st.title("🔍 Investigator's Workbench")
    st.markdown("*Compliance Analyst view — Alert queue and SAR generation*")
    st.divider()

    # Filter panel
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        min_wrs = st.slider("Minimum WRS Score", 0, 100, 40)
    with col_f2:
        channel_filter = st.multiselect(
            "Payment Channel",
            df["Payment_Channel"].unique().tolist(),
            default=df["Payment_Channel"].unique().tolist())
    with col_f3:
        flag_only = st.checkbox("Show Suspicious Only", value=True)

    # Apply filters
    filtered = df.copy()
    if flag_only:
        filtered = filtered[filtered["Ground_Truth_Label"] == 1]
    filtered = filtered[
        (filtered["WRS"] >= min_wrs) &
        (filtered["Risk_Tier"].isin(tiers)) &
        (filtered["Anomaly_Type"].isin(anomaly_types)) &
        (filtered["Payment_Channel"].isin(channel_filter))
    ].sort_values("WRS", ascending=False)

    st.markdown(f"**{len(filtered):,} alerts matching filters**")

    # Alert queue table
    display_cols = ["Transaction_ID", "Sender_Customer_ID",
                    "Transaction_Amount_INR", "Payment_Channel",
                    "Anomaly_Type", "WRS", "Risk_Tier",
                    "Origin_Country", "Destination_Country",
                    "Transaction_Timestamp"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    # Colour code rows by risk tier
    
    def highlight_risk(row):
        bg_color = {"High":"#ffebee","Medium":"#fff3e0","Low":"#e8f5e9"}.get(
            str(row.get("Risk_Tier","")), "#2d2d2d")
        return [f"background-color: {bg_color}; color: black"] * len(row)

    st.dataframe(
        filtered[display_cols].head(100)
        .style.apply(highlight_risk, axis=1),
        use_container_width=True, height=350)

    st.divider()

    # Alert detail + SAR generator
    st.subheader("📋 Alert Detail & SAR Generator")
    selected_txn = st.selectbox(
        "Select Transaction ID to investigate:",
        filtered["Transaction_ID"].head(50).tolist()
        if len(filtered) > 0 else ["No alerts"])

    if selected_txn and selected_txn != "No alerts":
        row = filtered[filtered["Transaction_ID"] == selected_txn].iloc[0]

        col_d1, col_d2 = st.columns([1, 1])

        with col_d1:
            st.markdown("**Transaction Details**")
            detail_data = {
                "Field": ["Transaction ID", "Amount (₹)", "Channel",
                           "Anomaly Type", "WRS Score", "Risk Tier",
                           "Origin", "Destination", "Timestamp"],
                "Value": [
                    row.get("Transaction_ID",""),
                    f"₹{row.get('Transaction_Amount_INR',0):,.0f}",
                    row.get("Payment_Channel",""),
                    row.get("Anomaly_Type",""),
                    f"{row.get('WRS',0):.2f} / 100",
                    str(row.get("Risk_Tier","")),
                    row.get("Origin_Country",""),
                    row.get("Destination_Country",""),
                    str(row.get("Transaction_Timestamp",""))[:16],
                ]
            }
            st.table(pd.DataFrame(detail_data))

        with col_d2:
            st.markdown("**Risk Score Breakdown**")
            scores = {
                "S_Historical": row.get("S_Historical", 0),
                "S_Behavioral": row.get("S_Behavioral", 0),
                "S_Geographic": row.get("S_Geographic", 0),
            }
            fig_gauge = go.Figure()
            for label, val in scores.items():
                fig_gauge.add_trace(go.Bar(
                    x=[label], y=[val],
                    marker_color=["#1F3864","#2E4A8F","#4472C4"],
                    text=[f"{val:.1f}"], textposition="auto"))
            fig_gauge.update_layout(
                height=250, showlegend=False,
                yaxis=dict(range=[0,100], title="Score (0–100)"),
                margin=dict(l=0,r=0,t=10,b=0),
                template="plotly_white")
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("**Feature Indicators**")
            feat_df = pd.DataFrame({
                "Feature": ["VCI","TPS","RTR","GRE","PGDS"],
                "Value":   [round(row.get(f,0),4)
                            for f in ["VCI","TPS","RTR","GRE","PGDS"]],
                "Alert?":  [
                    "🔴" if row.get("VCI",0) > 2.0 else "🟢",
                    "🔴" if row.get("TPS",0) >= 0.9 else "🟢",
                    "🔴" if row.get("RTR",0) > 0.7 else "🟢",
                    "🔴" if row.get("GRE",0) > 0.5 else "🟢",
                    "🔴" if row.get("PGDS",0) > 2.5 else "🟢",
                ]
            })
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

        # Customer profile
        cust_row = customers[
            customers["Customer_ID"] == row.get("Sender_Customer_ID")]
        if len(cust_row):
            st.markdown("**Customer Risk Profile**")
            c = cust_row.iloc[0]
            cp1, cp2, cp3, cp4 = st.columns(4)
            cp1.metric("Entity Type",  c["Entity_Type"])
            cp2.metric("Static Risk",  f"{c['Static_Risk_Score']:.2f}")
            cp3.metric("Country",      c["Registration_Country"])
            cp4.metric("KYC Verified", str(c["KYC_Verified"]))

        # SAR Generator button
        st.divider()
        if st.button("📄 Generate / View SAR Report", type="primary"):
            sar_text = load_sar_text(selected_txn)
            st.markdown("**Auto-Generated SAR Draft** *(Pending analyst approval)*")
            st.markdown(f'<div class="sar-box">{sar_text}</div>',
                        unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download SAR (.txt)",
                data=sar_text,
                file_name=f"SAR_{selected_txn}.txt",
                mime="text/plain")


# ── View 3: Trend Analysis ─────────────────────────────────────

def render_trend_analysis(df: pd.DataFrame, data: dict):
    st.title("📈 Trend Analysis")
    st.markdown("*Risk Manager view — Evolving risk exposure and system performance*")
    st.divider()

    df["Month"] = df["Transaction_Timestamp"].dt.to_period("M").astype(str)
    df["Week"]  = df["Transaction_Timestamp"].dt.to_period("W").astype(str)

    # Row 1: Monthly alert volume by tier + anomaly type over time
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📅 Monthly Alert Volume by Risk Tier")
        suspicious = df[df["Ground_Truth_Label"]==1]
        monthly = (suspicious.groupby(["Month","Risk_Tier"])
                   .size().reset_index(name="Count"))
        fig_monthly = px.bar(
            monthly, x="Month", y="Count", color="Risk_Tier",
            color_discrete_map={
                "High":"#d32f2f","Medium":"#f57c00","Low":"#388e3c"},
            barmode="stack", template="plotly_white")
        fig_monthly.update_layout(
            height=350, xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=-0.4),
            margin=dict(l=0,r=0,t=20,b=60))
        st.plotly_chart(fig_monthly, use_container_width=True)

    with col2:
        st.subheader("📊 WRS Distribution Over Time")
        fig_violin = px.violin(
            df[df["Ground_Truth_Label"]==1],
            x="Anomaly_Type", y="WRS", color="Anomaly_Type",
            box=True, template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Bold)
        fig_violin.update_layout(
            height=350, showlegend=False,
            xaxis_tickangle=-20,
            margin=dict(l=0,r=0,t=20,b=60))
        st.plotly_chart(fig_violin, use_container_width=True)

    # Row 2: Feature correlation heatmap + Channel distribution
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🔗 Feature Correlation Matrix")
        feat_cols = ["VCI","TPS","RTR","GRE","PGDS",
                     "S_Historical","S_Behavioral","WRS"]
        feat_cols = [c for c in feat_cols if c in df.columns]
        corr = df[feat_cols].corr().round(2)
        fig_heat = px.imshow(
            corr, text_auto=True, color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, template="plotly_white",
            aspect="auto")
        fig_heat.update_layout(
            height=380, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig_heat, use_container_width=True)

    with col4:
        st.subheader("💳 Payment Channel Risk Profile")
        channel_risk = (df[df["Ground_Truth_Label"]==1]
                        .groupby("Payment_Channel")
                        .agg(Count=("WRS","count"),
                             Avg_WRS=("WRS","mean"))
                        .reset_index()
                        .sort_values("Avg_WRS", ascending=False))
        fig_ch = px.scatter(
            channel_risk, x="Count", y="Avg_WRS",
            size="Count", color="Payment_Channel",
            text="Payment_Channel",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            template="plotly_white")
        fig_ch.update_traces(textposition="top center")
        fig_ch.update_layout(
            height=380, showlegend=False,
            xaxis_title="Number of Suspicious Transactions",
            yaxis_title="Average WRS",
            margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig_ch, use_container_width=True)

    # Row 3: Entity type risk profile + SAR summary
    st.divider()
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("👤 Customer Entity Type Risk Profile")
        if "customers" in data:
            ent_risk = (data["customers"]
                        .groupby("Entity_Type")["Static_Risk_Score"]
                        .agg(["mean","count"])
                        .reset_index())
            ent_risk.columns = ["Entity Type","Avg Risk Score","Count"]
            fig_ent = px.bar(
                ent_risk, x="Entity Type", y="Avg Risk Score",
                color="Entity Type", text="Count",
                color_discrete_sequence=px.colors.qualitative.Safe,
                template="plotly_white")
            fig_ent.update_layout(
                height=300, showlegend=False,
                margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig_ent, use_container_width=True)

    with col6:
        st.subheader("📄 SAR Generation Summary")
        if "sar_summary" in data:
            sar_df = data["sar_summary"]
            fig_sar = px.bar(
                sar_df.groupby("Anomaly_Type").size().reset_index(
                    name="SAR Count"),
                x="Anomaly_Type", y="SAR Count",
                color="Anomaly_Type",
                color_discrete_sequence=px.colors.qualitative.Bold,
                template="plotly_white")
            fig_sar.update_layout(
                height=300, showlegend=False,
                margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig_sar, use_container_width=True)
        else:
            st.info("Run sar_generator.py to populate SAR summary.")

    # Model evaluation metrics (if available)
    if "evaluation" in data:
        st.divider()
        st.subheader("🤖 ML Model Evaluation Metrics")
        ev = data["evaluation"].iloc[0]
        cols = st.columns(5)
        metrics = [
            ("AUC-ROC",    f"{ev.get('AUC_ROC',0):.4f}"),
            ("Precision",  f"{ev.get('Precision_Suspicious',0):.4f}"),
            ("Recall",     f"{ev.get('Recall_Suspicious',0):.4f}"),
            ("FDR",        f"{ev.get('FDR',0):.4f}"),
            ("True Pos",   str(int(ev.get("True_Positives",0)))),
        ]
        for col, (label, val) in zip(cols, metrics):
            col.metric(label, val)


# ── Main App ───────────────────────────────────────────────────

def main():
    data = load_data()
    df   = data["transactions"]
    customers = data.get("customers", pd.DataFrame())

    view, tiers, anomaly_types = render_sidebar()

    # Apply sidebar filters to main dataset
    df_filtered = df[
        df["Risk_Tier"].isin(tiers) |
        (df["Risk_Tier"].isna())]

    if "📊 Executive Summary" in view:
        render_executive_summary(df_filtered, data)

    elif "🔍 Investigator" in view:
        render_investigator_workbench(
            df_filtered, customers, tiers, anomaly_types)

    elif "📈 Trend" in view:
        render_trend_analysis(df_filtered, data)

    # Footer
    st.sidebar.divider()
    st.sidebar.markdown(
        "*RegTech POC System | BITS Pilani MBA FinTech*")
    st.sidebar.markdown(
        "*Data: Synthetic — No real PII used*")


if __name__ == "__main__":
    main()
