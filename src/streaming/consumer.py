import json
import requests
import threading
from kafka import KafkaConsumer
from concurrent.futures import ThreadPoolExecutor
import psycopg2
from psycopg2 import pool

# Initialize a thread-safe database connection pool
db_pool = psycopg2.pool.SimpleConnectionPool(
    1, 20, host="localhost", database="fraud_audit_log", user="admin", password="adminpassword"
)

# Setup schema using a single connection before spawning threads
conn = db_pool.getconn()
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS audit_log")
cursor.execute("""
    CREATE TABLE audit_log (
        id SERIAL PRIMARY KEY,
        amount FLOAT,
        true_class FLOAT,
        risk_score FLOAT,
        decision VARCHAR(50),
        shap_reasons TEXT,
        model_version VARCHAR(50),
        threshold FLOAT
    )
""")
conn.commit()
cursor.close()
db_pool.putconn(conn)

consumer = KafkaConsumer(
    'fraud_transactions',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

print("Listening for streaming transactions with multithreaded processing...")

def process_message(txn):
    features = [txn[f'V{i}'] for i in range(1, 29)] + [txn['Amount']]
    
    try:
        response = requests.post("http://localhost:8000/score", json={"features": features})
        result = response.json()
        
        # Thread-safe database insert
        local_conn = db_pool.getconn()
        local_cursor = local_conn.cursor()
        
        local_cursor.execute(
            """INSERT INTO audit_log 
               (amount, true_class, risk_score, decision, shap_reasons, model_version, threshold) 
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (txn['Amount'], txn.get('Class', 0), result['score'], result['decision'], 
             str(result['top_shap_reasons']), result['model_version'], result['threshold'])
        )
        local_conn.commit()
        
        local_cursor.close()
        db_pool.putconn(local_conn)
        
        thread_id = threading.current_thread().name
        print(f"[{thread_id}] Scored Txn: ${txn['Amount']:.2f} | Risk: {result['score']:.4f} | Action: {result['decision'].upper()}")
        
    except Exception as e:
        print(f"Error processing transaction: {e}")

# Process the stream using a thread pool of 5 concurrent workers
with ThreadPoolExecutor(max_workers=5) as executor:
    for message in consumer:
        executor.submit(process_message, message.value)