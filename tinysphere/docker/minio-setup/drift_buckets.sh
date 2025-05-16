#!/bin/bash

# This script creates the required MinIO buckets for drift detection and prediction images
# It should be run after the main setup.sh script

set -e

# Default values
MINIO_ROOT_USER=${MINIO_ROOT_USER:-minioadmin}
MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD:-minioadmin}
MINIO_HOST=${MINIO_HOST:-minio:9000}

# Required buckets for drift detection
DRIFT_BUCKET="drift"
DRIFT_MODELS_BUCKET="drift-models"

# Required bucket for prediction images
PREDICTION_IMAGES_BUCKET="prediction-images"

# Required bucket for operational logs
DATA_LOGS_BUCKET="data-logs"

# Create the MinIO client alias
echo "Creating MinIO client alias..."
mc alias set tinysphere http://${MINIO_HOST} ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}

# Create buckets for drift detection
echo "Creating drift detection buckets..."
mc mb --ignore-existing tinysphere/${DRIFT_BUCKET}
mc mb --ignore-existing tinysphere/${DRIFT_MODELS_BUCKET}

# Create bucket for prediction images
echo "Creating prediction images bucket..."
mc mb --ignore-existing tinysphere/${PREDICTION_IMAGES_BUCKET}

# Create bucket for operational logs
echo "Creating operational logs bucket..."
mc mb --ignore-existing tinysphere/${DATA_LOGS_BUCKET}

# Set policies to allow public read (useful for web UI)
echo "Setting bucket policies..."
mc policy set download tinysphere/${DRIFT_BUCKET}
mc policy set download tinysphere/${DRIFT_MODELS_BUCKET}
mc policy set download tinysphere/${PREDICTION_IMAGES_BUCKET}
mc policy set download tinysphere/${DATA_LOGS_BUCKET}

# Create sample directory structure
echo "Creating sample directory structure..."
mc mb --ignore-existing tinysphere/${DRIFT_BUCKET}/pending
mc mb --ignore-existing tinysphere/${DRIFT_BUCKET}/validated
mc mb --ignore-existing tinysphere/${DRIFT_BUCKET}/rejected

echo "Drift detection buckets configured successfully!"
echo "Prediction images bucket configured successfully!"
echo "Operational logs bucket configured successfully!"