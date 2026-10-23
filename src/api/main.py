from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import joblib
import numpy as np
import uvicorn
from abc import ABC, abstractmethod

app = FastAPI(title="Fraud Detection API")

# Load artifacts
model = xgb.XGBClassifier()
model.load_model(r'E:\FraudDetectionProject\src\models\xgb_model.json')
explainer = joblib.load(r'E:\FraudDetectionProject\src\models\shap_explainer.pkl')
MODEL_VERSION = "xgb-v1.0"
OPTIMAL_THRESHOLD = 0.0226

# --- Strategy Pattern: Rules Engine ---
class FraudRule(ABC):
    @abstractmethod
    def evaluate(self, features: list, score: float) -> str:
        pass

class ModelScoreRule(FraudRule):
    def evaluate(self, features: list, score: float) -> str:
        if score > 0.80:
            return "block"
        elif score > OPTIMAL_THRESHOLD:
            return "review"
        return "approve"

class HighAmountRule(FraudRule):
    def evaluate(self, features: list, score: float) -> str:
        amount = features[-1] # Amount is the last feature
        if amount > 5000.0:
            return "review"
        return "approve"

class RulesEngine:
    def __init__(self, rules: list[FraudRule]):
        self.rules = rules

    def evaluate_all(self, features: list, score: float) -> str:
        decisions = [rule.evaluate(features, score) for rule in self.rules]
        if "block" in decisions:
            return "block"
        if "review" in decisions:
            return "review"
        return "approve"

# Initialize engine with our independent rule classes
engine = RulesEngine(rules=[ModelScoreRule(), HighAmountRule()])

# --- API Endpoints ---
class Transaction(BaseModel):
    features: list[float]

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/score")
def score_transaction(txn: Transaction):
    x_input = np.array(txn.features).reshape(1, -1)
    score = float(model.predict_proba(x_input)[0, 1])
    
    # Execute pluggable rules
    decision = engine.evaluate_all(txn.features, score)
        
    shap_vals = explainer.shap_values(x_input)[0]
    top_indices = np.argsort(np.abs(shap_vals))[-3:][::-1]
    top_reasons = [f"Feature {idx}: {shap_vals[idx]:.4f}" for idx in top_indices]
    
    return {
        "score": round(score, 4),
        "decision": decision,
        "top_shap_reasons": top_reasons,
        "model_version": MODEL_VERSION,
        "threshold": OPTIMAL_THRESHOLD
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)