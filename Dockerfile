# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/
COPY models/ ./models/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.api.endpoints:app", "--host", "0.0.0.0", "--port", "8000"]