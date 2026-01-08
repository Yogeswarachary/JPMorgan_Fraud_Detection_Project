import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import io
import requests
from io import BytesIO

st.set_page_config(page_title="JP Morgan Fraud Detection", layout="wide")

# GitHub raw base (update branch/user/repo if needed)
GITHUB_RAW = "https://raw.githubusercontent.com/Yogeswarachary/JPMorgan_Fraud_Detection_Project/main/Capstone_Deployment/"

@st.cache_resource
def load_from_github(url):
    """Download and cache bytes from GitHub raw."""
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return io.BytesIO(resp.content)
    except Exception as e:
        st.error(f"Failed to load {url}: {e}")
        st.stop()

@st.cache_resource
def load_models_and_meta():
    # INLINE hybrid_pipeline.py - no external file needed
    class HybridFraudPipeline:
        def __init__(self, scaler, dt, cb, xgb, kmeans, iso, hybrid_cb):
            self.scaler = scaler
            self.dt = dt
            self.cb = cb
            self.xgb = xgb
            self.kmeans = kmeans
            self.iso = iso
            self.hybrid_cb = hybrid_cb
        
        def preprocess(self, X):
            if 'isFraud' in X.columns:
                X = X.drop(columns=['isFraud'])
            X_scaled = self.scaler.transform(X)
            dt_pred = self.dt.predict(X_scaled).reshape(-1, 1)
            cb_pred = self.cb.predict(X_scaled).reshape(-1, 1)
            xgb_pred = self.xgb.predict(X_scaled).reshape(-1, 1)
            clusters = self.kmeans.predict(X_scaled).reshape(-1, 1)
            anomaly = self.iso.decision_function(X_scaled).reshape(-1, 1)
            X_hybrid = np.hstack([X_scaled, dt_pred, cb_pred, xgb_pred, clusters, anomaly])
            return X_hybrid
        
        def predict(self, X):
            X_hybrid = self.preprocess(X)
            return self.hybrid_cb.predict(X_hybrid)
    
    # Load PKLs only
    pkl_files = ['scaler.pkl', 'dt.pkl', 'cb.pkl', 'xgb.pkl', 'kmeans.pkl', 'iso.pkl', 'hybrid_cb.pkl', 'train_columns.pkl']
    models = {}
    for pkl in pkl_files:
        pkl_url = GITHUB_RAW + pkl
        pkl_bytes = load_from_github(pkl_url).read()
        models[pkl] = pickle.load(BytesIO(pkl_bytes))
    
    scaler, dt, cb, xgb, kmeans, iso, hybrid_cb, train_columns = [models[f] for f in pkl_files]
    pipeline = HybridFraudPipeline(scaler, dt, cb, xgb, kmeans, iso, hybrid_cb)
    return pipeline, train_columns

pipeline, TRAIN_COLUMNS = load_models_and_meta()
SAMPLE_CSV_URL = GITHUB_RAW + "JPMorgan_Data_1000Rows_Sample.csv" 
# https://raw.githubusercontent.com/Yogeswarachary/JPMorgan_Fraud_Detection_Project/main/Capstone_Deployment/JPMorgan_Data_1000Rows_Sample.csv

RAW_FEATURES = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
                'nameDest', 'oldbalanceDest', 'newbalanceDest']
TYPE_CATEGORIES = ['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']

def feature_engineering(df):
    df = df.copy()
    for col in ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
        df[col] = np.log1p(df[col])
    df['orig_balance_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['dest_balance_diff'] = df['newbalanceDest'] - df['oldbalanceDest']
    df['orig_zero_balance_flag'] = ((df['oldbalanceOrg'] == 0) & (df['newbalanceOrig'] == 0)).astype(int)
    df['dest_zero_balance_flag'] = ((df['oldbalanceDest'] == 0) & (df['newbalanceDest'] == 0)).astype(int)
    df['transfer_to_zero_dest'] = ((df['type'] == 'TRANSFER') & (df['newbalanceDest'] == 0)).astype(int)
    df['cashout_from_zero_orig'] = ((df['type'] == 'CASH-OUT') & (df['oldbalanceOrg'] == 0)).astype(int)
    df['amount_to_orig_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['amount_to_dest_balance_ratio'] = df['amount'] / (df['oldbalanceDest'] + 1)
    df['suspicious_flag'] = ((df['amount'] > 100000) & (df['oldbalanceOrg'] == 0) & 
                             (df['newbalanceOrig'] == 0) & (df['type'] == 'TRANSFER')).astype(int)
    for t in TYPE_CATEGORIES:
        df[f'type_{t}'] = (df['type'] == t).astype(int)
    df = df.drop(columns=['type', 'nameOrig', 'nameDest'])
    return df

def prepare_for_model(df_raw):
    df_clean = df_raw[RAW_FEATURES].copy()
    if 'isFlaggedFraud' in df_clean.columns:
        df_clean = df_clean.drop(columns=['isFlaggedFraud'])
    df_fe = feature_engineering(df_clean)
    df_aligned = df_fe.reindex(columns=TRAIN_COLUMNS, fill_value=0)
    return df_aligned

st.title("🛡️ JP Morgan Chase Fraud Detection")
st.markdown("Loads models/data directly from GitHub. Hybrid ML predictions (CatBoost + stacking).")

mode = st.radio("Choose input mode:", ["Single Transaction", "Batch from CSV"])

if mode == "Single Transaction":
    default_row = pd.DataFrame({
        'step': [1], 'type': ['PAYMENT'], 'amount': [1000.0],
        'nameOrig': ['C123456789'], 'oldbalanceOrg': [0.0], 'newbalanceOrig': [0.0],
        'nameDest': ['M123456789'], 'oldbalanceDest': [0.0], 'newbalanceDest': [0.0]
    })
    edited = st.data_editor(
        default_row, num_rows="fixed",
        column_config={"type": st.column_config.SelectboxColumn(options=TYPE_CATEGORIES)},
        use_container_width=True
    )
    df_input_raw = edited.copy()
    
    if st.button("🔍 Predict Fraud"):
        try:
            X_input = prepare_for_model(df_input_raw)
            pred = pipeline.predict(X_input)[0]
            proba = float(pipeline.hybrid_cb.predict_proba(X_input)[0, 1])
            if pred == 1:
                st.error("🚨 **Transaction is FRAUD.**")
            else:
                st.success("✅ **Transaction is NOT Fraud.**")
            st.metric("Fraud Probability", f"{proba:.1%}")
        except Exception as e:
            st.warning(f"Prediction error: {e}")

else:  # Batch
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        df_batch_raw = pd.read_csv(uploaded)
    else:
        try:
            sample_resp = requests.get(SAMPLE_CSV_URL)
            sample_resp.raise_for_status()
            df_batch_raw = pd.read_csv(BytesIO(sample_resp.content))
            st.info("✅ Loaded sample CSV from GitHub.")
        except Exception as e:
            st.error(f"Failed to load sample CSV from GitHub: {e}. Upload manually.")
            st.stop()
    
    st.dataframe(df_batch_raw.head())
    
    if st.button("🚀 Run Batch Prediction"):
        try:
            X_batch = prepare_for_model(df_batch_raw)
            preds = pipeline.predict(X_batch)
            df_result = df_batch_raw.copy()
            df_result['fraud_prediction'] = preds
            fraud_count = int((df_result['fraud_prediction'] == 1).sum())
            st.success(f"✅ Predicted {fraud_count} frauds out of {len(df_result)}.")
            st.dataframe(df_result.head(10))
            csv_out = df_result.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv_out, "fraud_predictions.csv", "text/csv")
        except Exception as e:
            st.warning(f"Batch error: {e}")

st.markdown("---")
st.markdown("*Trained on 6.3M rows | All assets from GitHub | 0 FN*")


