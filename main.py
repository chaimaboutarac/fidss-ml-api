import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Roche FIDSS AI Engine")

# Get absolute directory path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

# Load pre-trained model artifacts dynamically
vectorizer = joblib.load(get_path("tfidf_vectorizer.joblib"))
model_cat = joblib.load(get_path("linear_svc_category.joblib"))
model_act = joblib.load(get_path("linear_svc_actionability.joblib"))

class InsightRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "FIDSS AI Engine is active and running"}

@app.post("/predict")
def predict_insight(payload: InsightRequest):
    text = payload.text
    if not text or not text.strip():
        return {
            "predicted_category": "Commercial, Sales & Field Operations",
            "predicted_actionability": 0
        }

    vec = vectorizer.transform([text])
    pred_cat = model_cat.predict(vec)[0]
    pred_act = int(model_act.predict(vec)[0])

    return {
        "predicted_category": pred_cat,
        "predicted_actionability": pred_act
    }
