import pandas as pd
from kafka import KafkaProducer
import json
import time

# Connect to the local Kafka container
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Load the test set to use as our live data stream
test_data = pd.read_csv(r'E:\FraudDetectionProject\data\processed\test.csv')

print("Starting to stream transactions to Kafka...")
topic = 'fraud_transactions'

# Replay transactions one by one
for index, row in test_data.iterrows():
    transaction = row.to_dict()
    producer.send(topic, transaction)
    print(f"Sent transaction {index} to Kafka.")
    
    # Pause for half a second to simulate real-time flow
    time.sleep(0.5)