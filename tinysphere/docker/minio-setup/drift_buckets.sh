#!/bin/bash

# This script creates the required MinIO buckets for drift detection
# It should be run after the main setup.sh script

set -e

# Default values
MINIO_ROOT_USER=${MINIO_ROOT_USER:-minioadmin}
MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD:-minioadmin}
MINIO_HOST=${MINIO_HOST:-minio:9000}

# Required buckets for drift detection
DRIFT_FEATURES_BUCKET="drift-features"
DRIFT_SAMPLES_BUCKET="drift-samples"
DRIFT_MODELS_BUCKET="drift-models"

# Create the MinIO client alias
echo "Creating MinIO client alias..."
mc alias set tinysphere http://${MINIO_HOST} ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}

# Create buckets for drift detection
echo "Creating drift detection buckets..."
mc mb --ignore-existing tinysphere/${DRIFT_FEATURES_BUCKET}
mc mb --ignore-existing tinysphere/${DRIFT_SAMPLES_BUCKET}
mc mb --ignore-existing tinysphere/${DRIFT_MODELS_BUCKET}

# Set policies to allow public read (useful for web UI)
echo "Setting bucket policies..."
mc policy set download tinysphere/${DRIFT_SAMPLES_BUCKET}
mc policy set download tinysphere/${DRIFT_FEATURES_BUCKET}
mc policy set download tinysphere/${DRIFT_MODELS_BUCKET}

# Create sample directory structure
echo "Creating sample directory structure..."
mc mb --ignore-existing tinysphere/${DRIFT_SAMPLES_BUCKET}/pending
mc mb --ignore-existing tinysphere/${DRIFT_SAMPLES_BUCKET}/validated
mc mb --ignore-existing tinysphere/${DRIFT_SAMPLES_BUCKET}/rejected

echo "Drift detection buckets configured successfully!"