#!/bin/bash
set -e

echo "=========================================="
echo "Initializing LSTM Stock Prediction Model"
echo "=========================================="

# Create necessary directories
mkdir -p /app/model
mkdir -p /app/output
mkdir -p /app/temp/cache

echo "Directories created:"
echo "  - /app/model (model storage)"
echo "  - /app/output (results output)"
echo "  - /app/temp/cache (intermediate cache)"

# Set environment variables
export APP_DIR=/app
export PYTHONPATH=/app/code/src:$PYTHONPATH

echo ""
echo "Environment initialization completed!"
echo "=========================================="
