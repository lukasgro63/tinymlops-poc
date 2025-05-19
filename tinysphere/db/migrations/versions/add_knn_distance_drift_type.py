"""Add KNN_DISTANCE drift type and ensure all required enum values exist

Revision ID: add_knn_distance_drift_type
Revises: rename_metadata_columns
Create Date: 2025-05-13 18:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
import logging

logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision = 'add_knn_distance_drift_type'
down_revision = 'rename_metadata_columns'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Get database connection
    conn = op.get_bind()
    
    # Check if the drifttype enum exists
    enum_check = conn.execute("SELECT 1 FROM pg_type WHERE typname = 'drifttype'").fetchone()
    if not enum_check:
        logger.warning("drifttype enum does not exist! Will create it with all values.")
        # Create the enum with all required values
        op.execute("""
        CREATE TYPE drifttype AS ENUM (
            'confidence', 
            'distribution', 
            'feature', 
            'outlier', 
            'custom', 
            'unknown', 
            'knn_distance'
        )
        """)
        logger.info("Created drifttype enum with all required values")
        return
        
    # If enum exists, check which values it has
    values_query = """
    SELECT enumlabel FROM pg_enum 
    JOIN pg_type ON pg_enum.enumtypid = pg_type.oid 
    WHERE pg_type.typname = 'drifttype'
    """
    existing_values = [row[0] for row in conn.execute(values_query).fetchall()]
    logger.info(f"Existing enum values: {existing_values}")
    
    # Make sure all required values exist
    required_values = [
        'confidence', 
        'distribution', 
        'feature', 
        'outlier', 
        'custom', 
        'unknown', 
        'knn_distance'
    ]
    
    # Add any missing values
    for value in required_values:
        if value not in existing_values:
            logger.info(f"Adding missing value '{value}' to drifttype enum")
            try:
                op.execute(f"ALTER TYPE drifttype ADD VALUE IF NOT EXISTS '{value}'")
                logger.info(f"Successfully added '{value}' to drifttype enum")
            except Exception as e:
                logger.error(f"Error adding '{value}' to drifttype enum: {e}")
                
    # Verify all values exist now
    final_values = [row[0] for row in conn.execute(values_query).fetchall()]
    logger.info(f"Final enum values after migration: {final_values}")
    
    # Make sure drift_events.drift_type allows NULL for existing rows
    # This is a safety step in case we need to update existing rows
    try:
        op.execute("ALTER TABLE drift_events ALTER COLUMN drift_type DROP NOT NULL")
        logger.info("Temporarily made drift_type column nullable")
        
        # Check for any NULL drift_type values and update them
        null_check = "SELECT COUNT(*) FROM drift_events WHERE drift_type IS NULL"
        null_count = conn.execute(null_check).scalar()
        
        if null_count > 0:
            logger.warning(f"Found {null_count} rows with NULL drift_type")
            
            # Update NULL values to a valid type (using metadata if available)
            update_sql = """
            UPDATE drift_events
            SET drift_type = 
                CASE 
                    WHEN event_metadata->>'original_drift_type' = 'knn_distance' THEN 'knn_distance'::drifttype
                    WHEN event_metadata->>'original_drift_type' IS NOT NULL THEN 
                        (event_metadata->>'original_drift_type')::drifttype
                    ELSE 'confidence'::drifttype
                END
            WHERE drift_type IS NULL
            """
            conn.execute(update_sql)
            logger.info(f"Updated {null_count} rows with NULL drift_type")
            
        # Make drift_type NOT NULL again
        op.execute("ALTER TABLE drift_events ALTER COLUMN drift_type SET NOT NULL")
        logger.info("Made drift_type column NOT NULL again")
        
    except Exception as e:
        logger.error(f"Error handling nullable drift_type: {e}")


def downgrade() -> None:
    # Cannot safely remove a value from an enum in PostgreSQL
    # Would need to create a new type, update the column, and drop the old type
    pass