#!/usr/bin/env python3
"""
Script to check and display all available enum values in the PostgreSQL database.
This will help diagnose issues with missing enum values.
"""

import os
import sys
from sqlalchemy import create_engine, text
import json
from pprint import pprint

# Get configuration from environment or use defaults for development
DB_HOST = os.environ.get("POSTGRES_HOST", "localhost")
DB_PORT = os.environ.get("POSTGRES_PORT", "5432")
DB_NAME = os.environ.get("POSTGRES_DB", "tinysphere")
DB_USER = os.environ.get("POSTGRES_USER", "tinysphere")
DB_PASS = os.environ.get("POSTGRES_PASSWORD", "tinysphere")

# Construct connection string
db_url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

print(f"Connecting to database: {DB_HOST}:{DB_PORT}/{DB_NAME}")

# Connect to the database
try:
    engine = create_engine(db_url)
    connection = engine.connect()
    print("Successfully connected to the database")
except Exception as e:
    print(f"Error connecting to database: {e}")
    sys.exit(1)

# Function to check if an enum type exists
def enum_exists(name):
    query = text("""
    SELECT 1 FROM pg_type WHERE typname = :name
    """)
    
    result = connection.execute(query, {"name": name})
    return bool(result.scalar())

# Function to get all values for a specific enum
def get_enum_values(name):
    query = text("""
    SELECT enumlabel, enumsortorder
    FROM pg_enum
    JOIN pg_type ON pg_enum.enumtypid = pg_type.oid
    WHERE pg_type.typname = :name
    ORDER BY enumsortorder
    """)
    
    result = connection.execute(query, {"name": name})
    return [row[0] for row in result.fetchall()]

# List of all enum types to check
enum_types_to_check = [
    "drifttype",
    "driftstatus"
]

# Check each enum type
all_enums = {}

for enum_type in enum_types_to_check:
    if enum_exists(enum_type):
        enum_values = get_enum_values(enum_type)
        all_enums[enum_type] = enum_values
        print(f"Enum '{enum_type}' exists with values: {', '.join(enum_values)}")
    else:
        print(f"Enum '{enum_type}' does not exist in the database")
        all_enums[enum_type] = []

# Check which expected values might be missing
expected_values = {
    "drifttype": ["confidence", "distribution", "feature", "outlier", "custom", "unknown", "knn_distance"],
    "driftstatus": ["pending", "validated", "rejected", "resolved", "ignored"]
}

print("\nChecking for missing enum values:")
for enum_type, values in expected_values.items():
    if enum_type in all_enums:
        missing_values = [v for v in values if v not in all_enums[enum_type]]
        if missing_values:
            print(f"Missing values for '{enum_type}': {', '.join(missing_values)}")
        else:
            print(f"Enum '{enum_type}' has all expected values")
    else:
        print(f"Enum '{enum_type}' does not exist, can't check for missing values")

# Function to check values in a specific column of a table
def check_column_values(table, column):
    query = text(f"""
    SELECT DISTINCT {column} FROM {table}
    """)
    
    try:
        result = connection.execute(query)
        values = [row[0] for row in result.fetchall()]
        print(f"Values in {table}.{column}: {values}")
        return values
    except Exception as e:
        print(f"Error checking {table}.{column}: {e}")
        return []

# Check values in the drift_type column of the drift_events table
print("\nChecking actual values used in drift_events table:")
try:
    # First check if the table exists
    table_check = text("""
    SELECT 1 FROM information_schema.tables
    WHERE table_name = 'drift_events'
    """)
    
    if connection.execute(table_check).scalar():
        drift_types_used = check_column_values("drift_events", "drift_type")
        if drift_types_used:
            print(f"Total of {len(drift_types_used)} distinct drift_type values in use")
        else:
            print("No drift events found in the database")
    else:
        print("drift_events table does not exist in the database")
except Exception as e:
    print(f"Error checking drift_events table: {e}")

# Generate SQL to add missing enum values
if not all_enums or any(len(vals) == 0 for vals in all_enums.values()):
    print("\nPotential problem: Some enum types are missing!")
    print("Run the following SQL to create the required enums with all expected values:")
    
    for enum_type, values in expected_values.items():
        if enum_type not in all_enums or not all_enums[enum_type]:
            values_str = ", ".join([f"'{val}'" for val in values])
            print(f"\nCREATE TYPE {enum_type} AS ENUM ({values_str});")
else:
    # Generate SQL to add missing values to existing enums
    add_sql = []
    for enum_type, values in expected_values.items():
        if enum_type in all_enums:
            missing_values = [v for v in values if v not in all_enums[enum_type]]
            if missing_values:
                for val in missing_values:
                    add_sql.append(f"ALTER TYPE {enum_type} ADD VALUE IF NOT EXISTS '{val}';")
    
    if add_sql:
        print("\nSQL to add missing enum values:")
        for sql in add_sql:
            print(sql)

# Close the connection
connection.close()
print("\nDatabase connection closed")