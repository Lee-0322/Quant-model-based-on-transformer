#!/bin/bash
set -e

echo "=========================================="
echo "Starting LSTM Model Training"
echo "=========================================="

cd /app/code/src

# Set environment variables
export APP_DIR=/app
export PYTHONPATH=/app/code/src:$PYTHONPATH

# Run training
python train.py

echo ""
echo "Training completed!"
echo "=========================================="
