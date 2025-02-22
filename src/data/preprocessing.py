# src/data/preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.selected_features = [
            "Total day minutes",
            "Customer service calls",
            "International plan",
            "Total intl minutes",
            "Total intl calls",
            "Total eve minutes",
            "Number vmail messages",
            "Voice mail plan",
        ]
        
    def load_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test datasets."""
        try:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            logger.info(f"Loaded data: train={df_train.shape}, test={df_test.shape}")
            return df_train, df_test
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def preprocess_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Preprocess features including encoding categorical variables."""
        try:
            # Select features
            X = df[self.selected_features].copy()
            
            # Encode categorical features
            categorical_features = ["International plan", "Voice mail plan"]
            for feature in categorical_features:
                if is_training:
                    self.label_encoders[feature] = LabelEncoder()
                    X[feature] = self.label_encoders[feature].fit_transform(X[feature])
                else:
                    X[feature] = self.label_encoders[feature].transform(X[feature])
            
            logger.info(f"Preprocessed features shape: {X.shape}")
            return X
        except Exception as e:
            logger.error(f"Error preprocessing features: {str(e)}")
            raise
            
    def preprocess_target(self, df: pd.DataFrame, is_training: bool = True) -> pd.Series:
        """Preprocess target variable."""
        try:
            if is_training:
                self.label_encoders['target'] = LabelEncoder()
                y = self.label_encoders['target'].fit_transform(df['Churn'])
            else:
                y = self.label_encoders['target'].transform(df['Churn'])
            
            logger.info(f"Preprocessed target shape: {y.shape}")
            return y
        except Exception as e:
            logger.error(f"Error preprocessing target: {str(e)}")
            raise
            
    def prepare_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for training and testing."""
        # Load data
        df_train, df_test = self.load_data(train_path, test_path)
        
        # Preprocess features
        X_train = self.preprocess_features(df_train, is_training=True)
        X_test = self.preprocess_features(df_test, is_training=False)
        
        # Preprocess target
        y_train = self.preprocess_target(df_train, is_training=True)
        y_test = self.preprocess_target(df_test, is_training=False)
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the preprocessor
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        "data/churn-bigml-80.csv",
        "data/churn-bigml-20.csv"
    )
    
    logger.info("Data preprocessing completed successfully")