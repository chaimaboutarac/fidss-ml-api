from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Roche FIDSS AI Engine")

# Load pre-trained model artifacts
vectorizer = joblib.load("tfidf_vectorizer.joblib")
model_cat = joblib.load("linear_svc_category.joblib")
model_act = joblib.load("linear_svc_actionability.joblib")

class InsightRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "FIDSS AI Engine is active and running"}

@app.post("/predict")
def predict_insight(payload: InsightRequest):
    text = payload.text
    if not text.strip():
        return {"predicted_category": "Commercial, Sales & Field Operations", "predicted_actionability": 0}

    vec = vectorizer.transform([text])
    pred_cat = model_cat.predict(vec)[0]
    pred_act = int(model_act.predict(vec)[0])

    return {
        "predicted_category": pred_cat,
        "predicted_actionability": pred_act
    }
