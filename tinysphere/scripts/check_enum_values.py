#!/usr/bin/env python3
"""
Script to check actual values in the PostgreSQL enum type
"""

import sys
import os
from sqlalchemy import create_engine, text

# Get database URL from environment or use default
db_host = os.getenv("POSTGRES_HOST", "localhost")
db_port = os.getenv("POSTGRES_PORT", "5432")
db_user = os.getenv("POSTGRES_USER", "tinysphere")
db_pass = os.getenv("POSTGRES_PASSWORD", "tinysphere")
db_name = os.getenv("POSTGRES_DB", "tinysphere")

# Construct database URL
db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

# Connect to the database
engine = create_engine(db_url)

# Query to get enum values
with engine.connect() as conn:
    # Get enum values for drifttype
    drifttype_values = conn.execute(text("""
        SELECT enumlabel
        FROM pg_enum
        WHERE enumtypid = (
            SELECT oid 
            FROM pg_type 
            WHERE typname = 'drifttype'
        )
        ORDER BY enumsortorder
    """)).fetchall()
    
    # Get enum values for driftstatus
    driftstatus_values = conn.execute(text("""
        SELECT enumlabel
        FROM pg_enum
        WHERE enumtypid = (
            SELECT oid 
            FROM pg_type 
            WHERE typname = 'driftstatus'
        )
        ORDER BY enumsortorder
    """)).fetchall()
    
    # Print results
    print("=== DriftType Values ===")
    for val in drifttype_values:
        print(f"- '{val[0]}'")
    
    print("\n=== DriftStatus Values ===")
    for val in driftstatus_values:
        print(f"- '{val[0]}'")