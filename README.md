# Real-Time Fraud Detection Pipeline

A bank-style, real-time card fraud detection system with an analyst dashboard and a customer-facing scam-awareness flow. Runs locally with one command.

> **Status:** Work in progress. See the [Roadmap](#roadmap) for what is done and what is next.

## Why this project

Most fraud projects stop at a trained model in a notebook. This one covers the full path: streaming ingestion, a scoring API, rules combined with ML, explainable decisions, an audit trail, monitoring, and the human side (analysts and customers).

## Architecture

```
Transaction producer -> Kafka topic -> FastAPI scoring service (XGBoost + rules)
                                              |
                                              v
                        decision (approve / review / block)
                                              |
                     +------------------------+------------------------+
                     v                                                 v
             Postgres audit log                      Streamlit analyst dashboard
                                                     + customer awareness view
```

_Add an architecture diagram image here once ready: `docs/architecture.png`_

## Features

- **Real-time scoring:** Kafka consumer scores each transaction and returns a decision tier.
- **Hybrid decisions:** XGBoost model combined with a pluggable rules engine (high amount, burst of transactions, new country).
- **Explainability:** top 3 SHAP reasons returned with every score.
- **Audit log:** every decision stored with score, model version, threshold and reasons.
- **Analyst dashboard:** review queue, alert trends, SHAP explanations, feedback buttons, threshold slider (fraud caught vs. customer friction).
- **Drift monitoring:** PSI check on score and feature distributions.
- **Customer awareness view:** plain-language explanation for held transactions, an "Did you share your OTP or CVV?" prompt, and an "already shared details" guide (block card, call bank, report to 1930 and cybercrime.gov.in).

## Tech stack

Python, XGBoost, SHAP, MLflow, FastAPI, Kafka, Postgres, Streamlit, Docker Compose, pytest, Locust, GitHub Actions.

## Quick start

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
docker compose up
```

Then open:

- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

_Target: a stranger can clone and see it working in 5 minutes._

## API example

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "demo-001", "amount": 249.90, "...": "..."}'
```

Example response:

```json
{
  "transaction_id": "demo-001",
  "score": 0.87,
  "decision": "review",
  "reasons": ["High amount", "Unusual time", "Feature V14"],
  "customer_message": "We held this payment for your safety. Did you share your OTP or CVV with anyone?"
}
```

## Results

| Metric | Value |
| --- | --- |
| PR-AUC (test set) | _TBD_ |
| Recall at 1% FPR | _TBD_ |
| Baseline (logistic regression) PR-AUC | _TBD_ |
| Throughput (txns/sec) | _TBD_ |
| p95 scoring latency | _TBD_ ms |
| Test coverage | _TBD_ % |

Data is split by time (oldest 70% train, next 15% validation, newest 15% test) to avoid leaking future patterns into training.

## Project structure

```
.
├── data/                  # dataset notes (data itself is not committed)
├── notebooks/             # EDA
├── src/
│   ├── model/             # training, evaluation, SHAP
│   ├── api/               # FastAPI scoring service
│   ├── rules/             # rules engine (one class per rule)
│   ├── streaming/         # Kafka producer and consumer
│   ├── awareness/         # customer-facing awareness module (kept separable)
│   └── dashboard/         # Streamlit app
├── tests/                 # pytest unit and API tests
├── docs/                  # architecture diagram, model card, debugging log
├── docker-compose.yml
└── README.md
```

## Testing

```bash
pytest
locust -f tests/load/locustfile.py   # load test
```

CI runs linting, tests and the Docker build on every push via GitHub Actions.

## Roadmap

- [ ] Week 1: Data, EDA, time-based split, logistic regression baseline
- [ ] Week 2: XGBoost, MLflow tracking, SHAP, cost-based threshold
- [ ] Week 3: FastAPI scoring service, rules engine, tests
- [ ] Week 4: Kafka and Postgres streaming, audit log
- [ ] Week 5: Dashboard, drift monitoring, customer awareness view
- [ ] Week 6: CI/CD, deployment, model card

## Privacy and security

- Uses the public Kaggle credit card fraud dataset (anonymised features). No real card data.
- Identifiers are hashed and raw card data is never written to logs.
- See `docs/model_card.md` for training data, limitations and retraining plan.

## Limitations

- The dataset is anonymised and heavily imbalanced (about 0.17% fraud), so results will not transfer directly to a real bank.
- A "no alert" result does not mean a transaction is safe.
- Awareness messaging is general guidance for India (1930, cybercrime.gov.in). Verify helpline details against official sources.

## Debugging log

Notes on simulated incidents (for example, consumer lag) and how they were diagnosed live in `docs/debugging_log.md`.

## AI tools

AI assistants were used for boilerplate and test generation. All output was reviewed and tested.
