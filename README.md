# 🛡️ RegTech AML Compliance System — POC

**Project:** Role of RegTech in Enhancing Risk Monitoring, Automating SAR Reports and Visualization of Compliance Dashboard in FinTech Firms
**Student:** Amisha Punjabi | 2024MB53031 | MBA FinTech — BITS Pilani
**Supervisor:** Branham Charles | Infosys Limited, Pune

---

## 📁 Project Structure

```
regtech_project/
│
├── data/
│   └── synthetic_data_generator.py   ← Generates customer + transaction data
│
├── utils/
│   └── feature_engineering.py        ← Computes VCI, TPS, RTR, GRE, PGDS, WRS
│
├── models/
│   └── risk_model.py                 ← Trains Random Forest + Isolation Forest
│
├── sar/
│   └── sar_generator.py              ← Auto-generates SAR narrative reports
│
├── dashboard/
│   └── app.py                        ← Streamlit compliance dashboard
│
├── output_data/                      ← Generated after running pipeline
│   ├── customer_master.csv
│   ├── transaction_ledger.csv
│   ├── featured_transactions.csv
│   ├── predictions.csv
│   ├── model_evaluation.csv
│   ├── sar_summary.csv
│   └── sar_reports/                  ← Individual SAR .txt files
│
├── models/                           ← Saved ML model files (.pkl)
├── run_pipeline.py                   ← Master runner script
└── requirements.txt
```

---

## ⚙️ Setup Instructions (VS Code)

### Step 1 — Install Python
Ensure Python 3.10 or higher is installed.
Check: open VS Code terminal and run:
```
python --version
```

### Step 2 — Open the Project in VS Code
1. Open VS Code
2. Go to **File → Open Folder**
3. Select the `regtech_project` folder

### Step 3 — Create a Virtual Environment
In the VS Code terminal (`` Ctrl+` `` to open):

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear in the terminal prompt.

### Step 4 — Install All Dependencies
```bash
pip install -r requirements.txt
```
This installs: pandas, numpy, scikit-learn, imbalanced-learn,
streamlit, plotly, faker, jinja2, scipy, joblib, matplotlib.

---

## 🚀 Running the Project

### Option A — Run Full Pipeline (Recommended for first run)

This runs all 4 steps in sequence:
1. Generates synthetic data
2. Computes features
3. Trains ML models
4. Generates SAR reports

```bash
python run_pipeline.py
```

Total runtime: approximately 3–8 minutes depending on hardware.

### Option B — Run Each Step Individually

**Step 1: Generate synthetic data**
```bash
python data/synthetic_data_generator.py
```
Output: `output_data/customer_master.csv` and `output_data/transaction_ledger.csv`

**Step 2: Feature engineering**
```bash
python utils/feature_engineering.py
```
Output: `output_data/featured_transactions.csv`

**Step 3: Train ML models**
```bash
python models/risk_model.py
```
Output: `models/random_forest_model.pkl`, `models/isolation_forest_model.pkl`,
`output_data/predictions.csv`, `output_data/model_evaluation.csv`

**Step 4: Generate SAR reports**
```bash
python sar/sar_generator.py
```
Output: `output_data/sar_reports/SAR_*.txt` and `output_data/sar_summary.csv`

---

## 📊 Launching the Dashboard

After the pipeline completes, launch the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

This will automatically open `http://localhost:8501` in your browser.

**Dashboard Views:**
- **📊 Executive Summary** — KPI scorecards, anomaly breakdown, geographic risk
- **🔍 Investigator Workbench** — Alert queue, transaction details, SAR generator
- **📈 Trend Analysis** — Monthly trends, feature correlations, channel risk profiles

---

## ☁️ Deploying to Streamlit Cloud (Free)

1. Push the project to a GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Initial RegTech POC"
   git remote add origin https://github.com/YOUR_USERNAME/regtech-poc.git
   git push -u origin main
   ```

2. Go to [share.streamlit.io](https://share.streamlit.io)

3. Sign in with GitHub

4. Click **New app** and fill in:
   - **Repository:** `YOUR_USERNAME/regtech-poc`
   - **Branch:** `main`
   - **Main file path:** `dashboard/app.py`

5. Click **Deploy**

> ⚠️ **Important:** Before deploying, run the pipeline locally once so that the
> `output_data/` folder and all CSV files are committed to GitHub along with the code.
> Streamlit Cloud does not run the pipeline automatically — it only serves the dashboard.

---

## 🧪 Anomaly Types and What to Expect

| Anomaly Type     | Description                                               | Key Feature Triggered |
|------------------|-----------------------------------------------------------|-----------------------|
| Structuring      | Multiple txns just below ₹10L threshold                   | TPS (0.90–0.99)       |
| Layering         | Rapid multi-hop transfers across accounts                 | VCI, GRE              |
| Round-Tripping   | Funds leave and return within 48 hours                    | RTR (> 0.70)          |
| Velocity-Spike   | Sudden surge in transaction count vs. 30-day average     | VCI (> 2.0)           |
| Geographic-Risk  | Transfers to FATF Black/Grey list countries via SWIFT     | GRE (> 0.50)          |

---

## 📐 Risk Score Formula (WRS)

```
WRS = (0.25 × S_Historical) + (0.50 × S_Behavioral) + (0.25 × S_Geographic)

S_Behavioral = (0.30×VCI + 0.25×TPS + 0.20×RTR + 0.20×PGDS + 0.05×TOD) × 100
S_Geographic = GRE × 100
```

Risk Tiers:
- **High Risk:** WRS 71–100 → Immediate investigation required
- **Medium Risk:** WRS 41–70 → Weekly review queue
- **Low Risk:** WRS 0–40 → No immediate action

---

## 📋 SAR Reports

Auto-generated SAR drafts are saved as `.txt` files in `output_data/sar_reports/`.
Each SAR covers the mandatory 5W framework:
- **Who** — Customer identity and KYC details
- **What** — Transaction instruments and channels
- **When** — Timestamps and temporal patterns
- **Where** — Geographic origin and destination
- **Why** — Specific behavioral indicators and reason codes

All SARs require compliance officer review before filing with FIU-IND.

---

## ⚠️ Data Privacy Notice

This system uses 100% synthetic data generated by the custom Python engine.
No real personally identifiable information (PII) is used at any stage.
The synthetic data is calibrated to Indian FinTech statistical distributions
but does not represent or reference any real individual or organization.
This project complies with India's DPDP Act, 2023.

---

## 📚 References

- Kamolov, A. (2025). The RegTech revolution. CAIJITMF.
- Nie, Liu, & Wang (2025). AI applications in AML. arXiv.
- Naik et al. (2025). Co-Investigator AI. arXiv.
- Arner, Barberis & Buckley (2019). The RegTech Book. Wiley.
- FATF (2023). AML Typology Reports. fatf-gafi.org
- RBI Master Directions on KYC/AML (2016, updated 2023).
