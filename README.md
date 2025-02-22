# Customer Churn Prediction MLOps Pipeline

## Project Overview
This project implements a complete MLOps pipeline for customer churn prediction using XGBoost. It includes data preprocessing, model training, API deployment, monitoring, and continuous integration/deployment.

## Features
- Automated data preprocessing pipeline
- XGBoost model training with MLflow tracking
- FastAPI REST API for model serving
- Docker containerization
- Elasticsearch & Kibana monitoring
- Complete CI/CD pipeline with Jenkins
- Comprehensive testing suite

## Project Structure
```
mlops_project/
├── src/
│   ├── data/          # Data preprocessing
│   ├── models/        # Model training
│   ├── api/           # FastAPI implementation
│   └── utils/         # Utilities and monitoring
├── tests/             # Test suite
├── config/            # Configuration files
├── docker/            # Docker configurations
├── notebooks/         # Jupyter notebooks
└── docs/             # Documentation
```

## Setup Instructions

### Local Development
1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configurations
```

4. Run the application:
```bash
python run.py
```

### Docker Deployment
1. Build and run with Docker Compose:
```bash
docker-compose up -d
```

2. Access services:
- FastAPI docs: http://localhost:8000/docs
- MLflow UI: http://localhost:5000
- Kibana: http://localhost:5601

## API Endpoints
- `/predict`: Make churn predictions
- `/retrain`: Retrain the model
- `/metrics`: Get model performance metrics
- `/health`: Health check endpoint

## Testing
Run tests with pytest:
```bash
pytest tests/
```

## Monitoring
Access monitoring dashboards:
1. Kibana: http://localhost:5601
2. MLflow: http://localhost:5000

## CI/CD Pipeline
The project includes a Jenkins pipeline that handles:
- Code quality checks
- Testing
- Docker image building
- Deployment
- Monitoring setup

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a pull request

## License
MIT License
