import os
import sys

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Determine if we're in test mode
TEST_MODE = os.getenv("TEST_MODE", "").lower() in ("1", "true", "yes")

# Use SQLite for testing, PostgreSQL for production
if TEST_MODE:
    DATABASE_URL = "sqlite:///./test.db"
    print("Using SQLite test database")
else:
    # Try to connect to Docker container first, fall back to localhost
    db_host = os.getenv("POSTGRES_HOST", "postgres")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    db_user = os.getenv("POSTGRES_USER", "tinysphere")
    db_pass = os.getenv("POSTGRES_PASSWORD", "tinysphere")
    db_name = os.getenv("POSTGRES_DB", "tinysphere")
    
    DATABASE_URL = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

# Create engine with appropriate settings for SQLite if needed
if TEST_MODE:
    engine = create_engine(
        DATABASE_URL, 
        connect_args={"check_same_thread": False}  # Only needed for SQLite
    )
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()