# src/models/training.py
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import classification_report, confusion_matrix
import logging
import yaml
from pathlib import Path
import joblib
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self.load_config(config_path)
        self.model = None
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load model configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Loaded model configuration")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise
            
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
        """Train the XGBoost model and log with MLflow."""
        try:
            with mlflow.start_run() as run:
                logger.info(f"Started MLflow run: {run.info.run_id}")
                
                # Get model parameters from config
                params = self.config['model_params']
                
                # Log parameters
                mlflow.log_params(params)
                
                # Train model
                self.model = xgb.XGBClassifier(**params)
                self.model.fit(X_train, y_train)
                
                # Log model
                mlflow.xgboost.log_model(
                    self.model,
                    "model",
                    registered_model_name="churn_model"
                )
                
                logger.info("Model training completed")
                return self.model
                
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
            
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate the model and log metrics to MLflow."""
        try:
            with mlflow.start_run(run_id=mlflow.active_run().info.run_id):
                # Make predictions
                y_pred = self.model.predict(X_test)
                
                # Calculate metrics
                report = classification_report(y_test, y_pred, output_dict=True)
                
                # Log metrics
                metrics = {
                    "accuracy": report['accuracy'],
                    "precision": report['weighted avg']['precision'],
                    "recall": report['weighted avg']['recall'],
                    "f1_score": report['weighted avg']['f1-score']
                }
                mlflow.log_metrics(metrics)
                
                # Log confusion matrix as a figure
                cm = confusion_matrix(y_test, y_pred)
                mlflow.log_text(str(cm), "confusion_matrix.txt")
                
                logger.info(f"Model evaluation completed. Metrics: {metrics}")
                return metrics
                
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
            
    def save_model(self, path: str = "models/model.joblib") -> None:
        """Save the trained model."""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, path: str = "models/model.joblib") -> xgb.XGBClassifier:
        """Load a trained model."""
        try:
            self.model = joblib.load(path)
            logger.info(f"Model loaded from {path}")
            return self.model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("churn_prediction")
    
    # Import preprocessor
    from src.data.preprocessing import DataPreprocessor
    
    # Prepare data
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        "data/churn-bigml-80.csv",
        "data/churn-bigml-20.csv"
    )
    
    # Train and evaluate model
    trainer = ModelTrainer()
    model = trainer.train(X_train, y_train)
    metrics = trainer.evaluate(X_test, y_test)
    
    # Save model
    trainer.save_model()