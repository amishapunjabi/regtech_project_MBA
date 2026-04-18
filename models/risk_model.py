"""
risk_model.py
────────────────────────────────────────────────────────────────
Trains and evaluates the ensemble ML risk scoring model:
  • Random Forest Classifier   (supervised — labeled training data)
  • Isolation Forest           (unsupervised — detects novel patterns)

Combined into a single Weighted Risk Score (WRS) using the
formula from Annexure 5 of the project report.

Outputs:
  models/random_forest_model.pkl
  models/isolation_forest_model.pkl
  output_data/model_evaluation.csv
  output_data/predictions.csv
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve,
                             ConfusionMatrixDisplay)
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

OUTPUT_DIR = "output_data"
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Feature columns used for ML training
FEATURE_COLS = ["VCI", "TPS", "RTR", "GRE", "PGDS",
                "S_Historical", "S_Behavioral", "S_Geographic",
                "Transaction_Amount_INR"]


def load_featured_data() -> pd.DataFrame:
    path = f"{OUTPUT_DIR}/featured_transactions.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Run feature_engineering.py first to generate featured_transactions.csv")
    df = pd.read_csv(path, parse_dates=["Transaction_Timestamp"])
    print(f"[✓] Loaded {len(df):,} feature-enriched transactions")
    return df


def prepare_training_data(df: pd.DataFrame):
    """
    Splits into features/labels, applies MinMax scaling,
    and uses SMOTE to balance the class distribution.
    """
    X = df[FEATURE_COLS].fillna(0)
    y = df["Ground_Truth_Label"]

    print(f"\n    Class distribution before SMOTE:")
    print(f"      Normal     : {(y==0).sum():,}")
    print(f"      Suspicious : {(y==1).sum():,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # SMOTE: oversample minority class in training set only
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_sc, y_train)

    print(f"\n    Class distribution after SMOTE:")
    print(f"      Normal     : {(y_train_bal==0).sum():,}")
    print(f"      Suspicious : {(y_train_bal==1).sum():,}")

    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    return X_train_bal, X_test_sc, y_train_bal, y_test, X_test, scaler


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Random Forest with 5-fold cross-validation.
    n_estimators=200, class_weight='balanced' as additional guard.
    """
    print("\n── Training Random Forest Classifier ─────────────────────")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1)

    # 5-fold CV on training data
    cv_scores = cross_val_score(rf, X_train, y_train,
                                cv=StratifiedKFold(5),
                                scoring="f1", n_jobs=-1)
    print(f"[✓] 5-fold CV F1 scores : {cv_scores.round(3)}")
    print(f"    Mean CV F1          : {cv_scores.mean():.3f} "
          f"(±{cv_scores.std():.3f})")

    rf.fit(X_train, y_train)
    joblib.dump(rf, f"{MODEL_DIR}/random_forest_model.pkl")
    print(f"[✓] Model saved → {MODEL_DIR}/random_forest_model.pkl")
    return rf


def train_isolation_forest(X_train_unscaled: pd.DataFrame) -> IsolationForest:
    """
    Isolation Forest for unsupervised anomaly detection.
    Detects novel patterns not seen during supervised training.
    contamination = estimated anomaly fraction in data.
    """
    print("\n── Training Isolation Forest (Unsupervised) ──────────────")
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.08,
        random_state=42,
        n_jobs=-1)

    iso.fit(X_train_unscaled[FEATURE_COLS].fillna(0))
    joblib.dump(iso, f"{MODEL_DIR}/isolation_forest_model.pkl")
    print(f"[✓] Model saved → {MODEL_DIR}/isolation_forest_model.pkl")
    return iso


def evaluate_models(rf, iso, X_test_sc, X_test_raw, y_test,
                    df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Generates predictions, blends RF + Isolation Forest scores,
    and evaluates against the held-out test set.
    """
    print("\n── Model Evaluation ──────────────────────────────────────")

    # Random Forest probabilities
    rf_proba = rf.predict_proba(X_test_sc)[:, 1]

    # Isolation Forest anomaly scores (convert: -1=anomaly, 1=normal → [0,1])
    iso_scores_raw = iso.decision_function(X_test_raw[FEATURE_COLS].fillna(0))
    iso_proba = 1 - (iso_scores_raw - iso_scores_raw.min()) / \
                    (iso_scores_raw.max() - iso_scores_raw.min() + 1e-9)

    # Ensemble blend: 70% RF + 30% Isolation Forest
    ensemble_proba = 0.70 * rf_proba + 0.30 * iso_proba

    # Threshold at 0.5 for binary classification
    y_pred = (ensemble_proba >= 0.50).astype(int)

    # Metrics
    print("\n    Classification Report (Ensemble):")
    print(classification_report(y_test, y_pred,
                                target_names=["Normal", "Suspicious"]))

    auc = roc_auc_score(y_test, ensemble_proba)
    print(f"    AUC-ROC Score : {auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fdr = round(fp / max(1, fp + tp), 4)
    print(f"\n    Confusion Matrix:")
    print(f"      True Negatives  (Normal correctly flagged as Normal)    : {tn}")
    print(f"      False Positives (Normal incorrectly flagged Suspicious) : {fp}")
    print(f"      False Negatives (Suspicious missed)                     : {fn}")
    print(f"      True Positives  (Suspicious correctly detected)         : {tp}")
    print(f"\n    False Discovery Rate (FDR) : {fdr:.4f} "
          f"({fdr*100:.1f}% false positives in alerts)")

    # Feature importance
    fi = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False)
    print(f"\n    Top Feature Importances (Random Forest):")
    print(fi.to_string(index=False))

    # Build predictions dataframe
    test_indices = X_test_raw.index
    result_df = df_test.loc[test_indices].copy()
    result_df["RF_Probability"]       = rf_proba.round(4)
    result_df["IsoForest_Score"]      = iso_proba.round(4)
    result_df["Ensemble_Probability"] = ensemble_proba.round(4)
    result_df["ML_Prediction"]        = y_pred
    result_df["WRS_ML"]               = (ensemble_proba * 100).round(2)

    result_df["Risk_Tier_ML"] = pd.cut(
        result_df["WRS_ML"],
        bins=[-1, 40, 70, 100],
        labels=["Low", "Medium", "High"])

    result_df.to_csv(f"{OUTPUT_DIR}/predictions.csv", index=False)
    print(f"\n[✓] Predictions saved → {OUTPUT_DIR}/predictions.csv")

    # Save evaluation summary
    eval_summary = pd.DataFrame([{
        "AUC_ROC": round(auc, 4),
        "FDR": fdr,
        "True_Positives": int(tp),
        "False_Positives": int(fp),
        "True_Negatives": int(tn),
        "False_Negatives": int(fn),
        "Precision_Suspicious": round(tp / max(1, tp+fp), 4),
        "Recall_Suspicious":    round(tp / max(1, tp+fn), 4),
    }])
    eval_summary.to_csv(f"{OUTPUT_DIR}/model_evaluation.csv", index=False)

    print("─" * 55)
    return result_df


def run_full_pipeline():
    df = load_featured_data()
    print("\n── Preparing Training Data ───────────────────────────────")
    X_tr, X_te_sc, y_tr, y_te, X_te_raw, scaler = prepare_training_data(df)

    rf  = train_random_forest(X_tr, y_tr)
    iso = train_isolation_forest(df)

    # Test set raw dataframe (unscaled, for Isolation Forest)
    _, X_test_raw = train_test_split(
        df, test_size=0.20, random_state=42,
        stratify=df["Ground_Truth_Label"])

    evaluate_models(rf, iso, X_te_sc, X_test_raw, y_te, df)


if __name__ == "__main__":
    run_full_pipeline()
