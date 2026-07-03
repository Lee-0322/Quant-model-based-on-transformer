#!/bin/bash
set -e

echo "=========================================="
echo "Starting LSTM Prediction and Evaluation"
echo "=========================================="

cd /app/code/src

# Set environment variables
export APP_DIR=/app
export PYTHONPATH=/app/code/src:$PYTHONPATH

# Run prediction
echo "Running prediction..."
python predict.py

# Run evaluation
echo "Running evaluation..."
python test.py

echo ""
echo "Prediction and evaluation completed!"
echo "=========================================="
