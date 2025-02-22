# run.py
import uvicorn
import argparse
import os
from dotenv import load_dotenv
import logging.config
import yaml
import mlflow

def setup_logging():
    """Setup logging configuration"""
    with open('config/logging_config.yaml', 'rt') as f:
        config = yaml.safe_load(f)
    
    # Ensure log directory exists
    os.makedirs('logs', exist_ok=True)
    
    logging.config.dictConfig(config)

def setup_mlflow():
    """Setup MLflow configuration"""
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment("churn_prediction")

def main():
    """Main application runner"""
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Setup MLflow
    setup_mlflow()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run MLOps Application')
    parser.add_argument('--host', default=os.getenv('API_HOST', '0.0.0.0'))
    parser.add_argument('--port', type=int, default=int(os.getenv('API_PORT', 8000)))
    parser.add_argument('--reload', action='store_true')
    args = parser.parse_args()
    
    logger.info(f"Starting application on {args.host}:{args.port}")
    
    # Run FastAPI application
    uvicorn.run(
        "src.api.endpoints:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_config="config/logging_config.yaml"
    )

if __name__ == "__main__":
    main()
