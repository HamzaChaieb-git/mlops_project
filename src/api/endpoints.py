# src/api/endpoints.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.xgboost
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, Any, Optional
from src.utils.metrics import ModelMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="API for customer churn prediction",
    version="1.0.0"
)

# Initialize model monitor
monitor = ModelMonitor()

class PredictionInput(BaseModel):
    total_day_minutes: float
    customer_service_calls: int
    international_plan: str
    total_intl_minutes: float
    total_intl_calls: int
    total_eve_minutes: float
    number_vmail_messages: int
    voice_mail_plan: str

class PredictionOutput(BaseModel):
    churn_prediction: bool
    churn_probability: float
    model_version: str
    prediction_time: datetime

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Make churn prediction."""
    try:
        start_time = datetime.now()
        
        # Load model from MLflow
        model = mlflow.xgboost.load_model("models:/churn_model/Production")
        
        # Prepare input data
        features = pd.DataFrame([input_data.dict()])
        
        # Make prediction
        prediction_prob = model.predict_proba(features)[0][1]
        prediction = bool(prediction_prob >= 0.5)
        
        # Calculate latency
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Log prediction
        monitor.log_prediction(
            features=input_data.dict(),
            prediction=prediction,
            probability=float(prediction_prob),
            model_version="1.0.0",
            latency_ms=latency_ms
        )
        
        return PredictionOutput(
            churn_prediction=prediction,
            churn_probability=float(prediction_prob),
            model_version="1.0.0",
            prediction_time=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain_model():
    """Retrain the model with new data."""
    try:
        from src.models.training import ModelTrainer
        from src.data.preprocessing import DataPreprocessor
        
        # Prepare data
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(
            "data/churn-bigml-80.csv",
            "data/churn-bigml-20.csv"
        )
        
        # Train model
        trainer = ModelTrainer()
        model = trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        
        return {
            "status": "success",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get model performance metrics."""
    try:
        metrics = monitor.get_prediction_metrics()
        return {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)