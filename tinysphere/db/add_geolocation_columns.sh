#!/bin/bash

# Run SQL script to add geolocation columns
# This can be executed inside the Docker container or locally

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Database connection parameters
DB_HOST=${POSTGRES_HOST:-"postgres"}
DB_PORT=${POSTGRES_PORT:-"5432"}
DB_USER=${POSTGRES_USER:-"tinysphere"}
DB_PASS=${POSTGRES_PASSWORD:-"tinysphere"}
DB_NAME=${POSTGRES_DB:-"tinysphere"}

echo "Adding geolocation columns to devices table..."
PGPASSWORD=${DB_PASS} psql -h ${DB_HOST} -p ${DB_PORT} -U ${DB_USER} -d ${DB_NAME} -f ${SCRIPT_DIR}/add_geolocation_columns.sql

echo "Done!"