import streamlit as st
import pandas as pd
import psycopg2
import time
import warnings

# Suppress pandas SQL warning for raw psycopg2 connections
warnings.filterwarnings('ignore', category=UserWarning)

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Real-Time Fraud Monitoring Engine")

def fetch_live_data():
    conn = psycopg2.connect(host="localhost", database="fraud_audit_log", user="admin", password="adminpassword")
    df = pd.read_sql("SELECT id, amount, risk_score, decision, shap_reasons FROM audit_log ORDER BY id DESC LIMIT 100", conn)
    conn.close()
    return df

df = fetch_live_data()

if not df.empty:
    # Calculate top-level metrics
    total_txns = len(df)
    flagged_txns = len(df[df['decision'] != 'approve'])
    avg_risk = df['risk_score'].mean()

    # Display KPI metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Recent Transactions Processed", total_txns)
    col2.metric("Flagged for Review / Blocked", flagged_txns)
    col3.metric("Average Risk Score", f"{avg_risk:.4f}")
    
    # Display the raw audit log
    st.dataframe(df, use_container_width=True)
    
    # Auto-refresh the dashboard every 2 seconds to simulate a live monitoring environment
    time.sleep(2)
    st.rerun()
else:
    st.warning("Awaiting live data. Ensure your Kafka producer, FastAPI server, and consumer are running.")