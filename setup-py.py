from setuptools import setup, find_packages

setup(
    name="churn_prediction",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "pandas>=2.1.3",
        "numpy>=1.26.2",
        "scikit-learn>=1.3.2",
        "xgboost>=2.0.2",
        "mlflow>=2.8.1",
        "elasticsearch>=8.11.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "joblib>=1.3.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "pylint>=3.0.2",
            "mypy>=1.7.1",
            "bandit>=1.7.5",
            "safety>=2.3.5",
            "pre-commit>=3.5.0",
        ]
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Customer Churn Prediction MLOps Pipeline",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/churn_prediction",
)
