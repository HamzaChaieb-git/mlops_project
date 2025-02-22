import pytest
import pandas as pd
import numpy as np
import os
import yaml
from src.models.training import ModelTrainer
import mlflow

@pytest.fixture
def sample_config(tmp_path):
    """Create a sample configuration file."""
    config = {
        'model_params': {
            'objective': 'binary:logistic',
            'max_depth': 3,
            'learning_rate': 0.1,
            'n_estimators': 10,
            'random_state': 42
        }
    }
    
    config_path = tmp_path / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return str(config_path)

@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 100
    
    X_train = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randint(0, 2, n_samples)
    })
    
    y_train = np.random.randint(0, 2, n_samples)
    
    X_test = pd.DataFrame({
        'feature1': np.random.randn(20),
        'feature2': np.random.randn(20),
        'feature3': np.random.randint(0, 2, 20)
    })
    
    y_test = np.random.randint(0, 2, 20)
    
    return X_train, X_test, y_train, y_test

def test_model_trainer_initialization(sample_config):
    """Test ModelTrainer initialization."""
    trainer = ModelTrainer(sample_config)
    assert trainer.config is not None
    assert 'model_params' in trainer.config

def test_model_training(sample_config, sample_data):
    """Test model training process."""
    X_train, _, y_train, _ = sample_data
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("test_experiment")
    
    # Train model
    trainer = ModelTrainer(sample_config)
    model = trainer.train(X_train, y_train)
    
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')

def test_model_evaluation(sample_config, sample_data):
    """Test model evaluation process."""
    X_train, X_test, y_train, y_test = sample_data
    
    # Train and evaluate
    trainer = ModelTrainer(sample_config)
    model = trainer.train(X_train, y_train)
    metrics = trainer.evaluate(X_test, y_test)
    
    assert metrics is not None
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics

def test_model_save_load(sample_config, sample_data, tmp_path):
    """Test model saving and loading."""
    X_train, X_test, y_train, y_test = sample_data
    model_path = str(tmp_path / "model.joblib")
    
    # Train and save model
    trainer = ModelTrainer(sample_config)
    model = trainer.train(X_train, y_train)
    trainer.save_model(model_path)
    
    # Load model and verify
    assert os.path.exists(model_path)
    loaded_model = trainer.load_model(model_path)
    assert loaded_model is not None
    
    # Verify predictions match
    original_preds = model.predict(X_test)
    loaded_preds = loaded_model.predict(X_test)
    assert np.array_equal(original_preds, loaded_preds)

def test_mlflow_logging(sample_config, sample_data):
    """Test MLflow logging functionality."""
    X_train, X_test, y_train, y_test = sample_data
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("test_experiment")
    
    # Train and evaluate with MLflow logging
    trainer = ModelTrainer(sample_config)
    with mlflow.start_run():
        model = trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        
        # Verify run was created
        assert mlflow.active_run() is not None
        
        # Verify metrics were logged
        run = mlflow.active_run()
        assert len(mlflow.get_run(run.info.run_id).data.metrics) > 0
