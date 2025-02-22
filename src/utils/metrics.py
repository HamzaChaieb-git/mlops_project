# src/utils/metrics.py
from elasticsearch import Elasticsearch
import mlflow
from datetime import datetime, timedelta
import logging
import numpy as np
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self, es_host: str = "http://localhost:9200"):
        self.es = Elasticsearch(es_host)
        self.mlflow_client = mlflow.tracking.MlflowClient()
        
    def log_prediction(self, 
                      features: Dict[str, Any],
                      prediction: bool,
                      probability: float,
                      model_version: str,
                      latency_ms: float) -> None:
        """Log prediction details to Elasticsearch."""
        try:
            doc = {
                "timestamp": datetime.now().isoformat(),
                "features": features,
                "prediction": prediction,
                "probability": probability,
                "model_version": model_version,
                "latency_ms": latency_ms
            }
            
            self.es.index(index="churn-predictions", document=doc)
            logger.info("Prediction logged successfully")
        except Exception as e:
            logger.error(f"Failed to log prediction: {str(e)}")
            
    def get_prediction_metrics(self, time_window_minutes: int = 60) -> Optional[Dict[str, Any]]:
        """Get prediction metrics for a specific time window."""
        try:
            query = {
                "query": {
                    "range": {
                        "timestamp": {
                            "gte": f"now-{time_window_minutes}m"
                        }
                    }
                },
                "aggs": {
                    "avg_probability": {"avg": {"field": "probability"}},
                    "avg_latency": {"avg": {"field": "latency_ms"}},
                    "predictions": {
                        "terms": {"field": "prediction"}
                    }
                }
            }
            
            result = self.es.search(index="churn-predictions", body=query)
            
            return {
                "total_predictions": result["hits"]["total"]["value"],
                "avg_probability": result["aggregations"]["avg_probability"]["value"],
                "avg_latency_ms": result["aggregations"]["avg_latency"]["value"],
                "prediction_distribution": {
                    bucket["key"]: bucket["doc_count"]
                    for bucket in result["aggregations"]["predictions"]["buckets"]
                }
            }
        except Exception as e:
            logger.error(f"Failed to get prediction metrics: {str(e)}")
            return None
            
    def check_data_drift(self, 
                        reference_data: pd.DataFrame,
                        current_data: pd.DataFrame,
                        threshold: float = 0.1) -> Dict[str, Any]:
        """Check for data drift using statistical tests."""
        try:
            drift_detected = False
            drift_features = []
            
            for column in reference_data.columns:
                if reference_data[column].dtype in ['int64', 'float64']:
                    # Calculate mean and std difference
                    mean_diff = abs(reference_data[column].mean() - current_data[column].mean())
                    std_diff = abs(reference_data[column].std() - current_data[column].std())
                    
                      if mean_diff > threshold * reference_data[column].std():
                drift_detected = True
                drift_features.append({
                    "feature": column,
                    "mean_difference": mean_diff,
                    "std_difference": std_diff
                })
                
            return {
                "drift_detected": drift_detected,
                "drift_features": drift_features,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error checking data drift: {str(e)}")
            return {"error": str(e)}

    def monitor_model_performance(self) -> Dict[str, Any]:
        """Monitor model performance metrics over time."""
        try:
            # Get recent predictions
            recent_metrics = self.get_prediction_metrics(60)  # Last hour
            historical_metrics = self.get_prediction_metrics(1440)  # Last 24 hours
            
            performance_metrics = {
                "recent": recent_metrics,
                "historical": historical_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
            # Log to MLflow
            with mlflow.start_run(run_name="monitoring"):
                mlflow.log_metrics({
                    "recent_avg_latency": recent_metrics["avg_latency_ms"],
                    "historical_avg_latency": historical_metrics["avg_latency_ms"]
                })
            
            return performance_metrics
        
        except Exception as e:
            logger.error(f"Error monitoring performance: {str(e)}")
            return {"error": str(e)}

    def create_monitoring_dashboard(self) -> Dict[str, Any]:
        """Create monitoring dashboard configuration for Kibana."""
        dashboard_config = {
            "dashboard": {
                "title": "Churn Model Monitoring",
                "panels": [
                    {
                        "title": "Prediction Distribution",
                        "type": "pie",
                        "index_pattern": "churn-predictions",
                        "field": "prediction"
                    },
                    {
                        "title": "Average Latency Over Time",
                        "type": "line",
                        "index_pattern": "churn-predictions",
                        "field": "latency_ms",
                        "interval": "1h"
                    },
                    {
                        "title": "Prediction Volume",
                        "type": "metric",
                        "index_pattern": "churn-predictions",
                        "field": "timestamp",
                        "interval": "1h"
                    }
                ]
            }
        }
        return dashboard_config

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the monitoring functionality
    monitor = ModelMonitor()
    
    # Test prediction logging
    test_prediction = {
        "features": {"feature1": 1.0},
        "prediction": True,
        "probability": 0.8,
        "model_version": "1.0.0",
        "latency_ms": 100.0
    }
    
    monitor.log_prediction(
        test_prediction["features"],
        test_prediction["prediction"],
        test_prediction["probability"],
        test_prediction["model_version"],
        test_prediction["latency_ms"]
    )
    
    # Get metrics
    metrics = monitor.get_prediction_metrics()
    logger.info(f"Current metrics: {metrics}")