#!/bin/bash
# Script to add platform version and device model columns to the devices table

set -e  # Exit on error

# Use DATABASE_URL if set, otherwise fall back to default
DB_URL="${DATABASE_URL:-postgresql://postgres:postgres@localhost:5432/tinysphere}"

echo "Adding platform_version and device_model columns to devices table..."
psql "$DB_URL" -f add_platform_columns.sql

echo "Migration completed successfully!"