import pytest
import pandas as pd
import numpy as np
from src.data.preprocessing import DataPreprocessor

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    train_data = pd.DataFrame({
        'Total day minutes': [100.0, 150.0],
        'Customer service calls': [2, 3],
        'International plan': ['yes', 'no'],
        'Total intl minutes': [10.0, 15.0],
        'Total intl calls': [3, 4],
        'Total eve minutes': [200.0, 250.0],
        'Number vmail messages': [25, 30],
        'Voice mail plan': ['yes', 'no'],
        'Churn': ['yes', 'no']
    })
    
    test_data = pd.DataFrame({
        'Total day minutes': [120.0],
        'Customer service calls': [2],
        'International plan': ['yes'],
        'Total intl minutes': [12.0],
        'Total intl calls': [3],
        'Total eve minutes': [220.0],
        'Number vmail messages': [27],
        'Voice mail plan': ['yes'],
        'Churn': ['no']
    })
    
    return train_data, test_data

def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    preprocessor = DataPreprocessor()
    assert len(preprocessor.selected_features) == 8
    assert preprocessor.label_encoders == {}

def test_data_loading(sample_data, tmp_path):
    """Test data loading functionality."""
    train_data, test_data = sample_data
    
    # Save sample data to temporary files
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    # Test loading
    preprocessor = DataPreprocessor()
    df_train, df_test = preprocessor.load_data(train_path, test_path)
    
    assert len(df_train) == 2
    assert len(df_test) == 1
    assert all(col in df_train.columns for col in preprocessor.selected_features)

def test_feature_preprocessing(sample_data):
    """Test feature preprocessing."""
    train_data, test_data = sample_data
    preprocessor = DataPreprocessor()
    
    # Process training data
    X_train = preprocessor.preprocess_features(train_data, is_training=True)
    assert X_train.shape[1] == len(preprocessor.selected_features)
    assert all(col in X_train.columns for col in preprocessor.selected_features)
    
    # Process test data
    X_test = preprocessor.preprocess_features(test_data, is_training=False)
    assert X_test.shape[1] == len(preprocessor.selected_features)

def test_target_preprocessing(sample_data):
    """Test target preprocessing."""
    train_data, test_data = sample_data
    preprocessor = DataPreprocessor()
    
    # Process training target
    y_train = preprocessor.preprocess_target(train_data, is_training=True)
    assert len(y_train) == len(train_data)
    assert isinstance(y_train, np.ndarray)
    
    # Process test target
    y_test = preprocessor.preprocess_target(test_data, is_training=False)
    assert len(y_test) == len(test_data)

def test_complete_pipeline(sample_data, tmp_path):
    """Test the complete preprocessing pipeline."""
    train_data, test_data = sample_data
    
    # Save sample data
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    # Run complete pipeline
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(train_path, test_path)
    
    # Assertions
    assert X_train.shape[1] == len(preprocessor.selected_features)
    assert X_test.shape[1] == len(preprocessor.selected_features)
    assert len(y_train) == len(train_data)
    assert len(y_test) == len(test_data)
