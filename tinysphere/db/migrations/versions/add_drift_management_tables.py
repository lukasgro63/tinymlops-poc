"""Add drift management tables

Revision ID: add_drift_management_tables
Revises: add_notifications_table
Create Date: 2023-05-04 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_drift_management_tables'
down_revision = 'add_notifications_table'
branch_labels = None
depends_on = None


def upgrade():
    # Check if enums already exist to handle idempotent migrations
    conn = op.get_bind()
    
    # Check if driftstatus enum exists
    result_status = conn.execute(sa.text(
        "SELECT 1 FROM pg_type WHERE typname = 'driftstatus'"
    ))
    status_exists = bool(result_status.scalar())
    
    # Create drift_status enum if it doesn't exist
    if not status_exists:
        drift_status = sa.Enum('pending', 'validated', 'rejected', 'resolved', 'ignored', name='driftstatus')
        drift_status.create(conn)
    else:
        # If it exists, just reference it
        drift_status = sa.Enum('pending', 'validated', 'rejected', 'resolved', 'ignored', name='driftstatus', create_type=False)
    
    # Check if drifttype enum exists
    result_type = conn.execute(sa.text(
        "SELECT 1 FROM pg_type WHERE typname = 'drifttype'"
    ))
    type_exists = bool(result_type.scalar())
    
    # Create drift_type enum if it doesn't exist
    if not type_exists:
        drift_type = sa.Enum('confidence', 'distribution', 'feature', 'outlier', 'custom', 'unknown', 'knn_distance', name='drifttype')
        drift_type.create(conn)
    else:
        # If it exists, just reference it
        drift_type = sa.Enum('confidence', 'distribution', 'feature', 'outlier', 'custom', 'unknown', 'knn_distance', name='drifttype', create_type=False)
    
    # Create drift_events table
    op.create_table('drift_events',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('event_id', sa.String(length=255), nullable=False),
        sa.Column('device_id', sa.String(length=255), nullable=False),
        sa.Column('model_id', sa.String(length=255), nullable=True),
        sa.Column('drift_type', drift_type, nullable=False, server_default='unknown'),
        sa.Column('drift_score', sa.Float(), nullable=True),
        sa.Column('detector_name', sa.String(length=255), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('received_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('metrics_before', sa.JSON(), nullable=True),
        sa.Column('metrics_after', sa.JSON(), nullable=True),
        sa.Column('status', drift_status, nullable=False, server_default='pending'),
        sa.Column('resolved_at', sa.DateTime(), nullable=True),
        sa.Column('resolution_notes', sa.Text(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['device_id'], ['devices.device_id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('event_id')
    )
    op.create_index(op.f('ix_drift_events_device_id'), 'drift_events', ['device_id'], unique=False)
    op.create_index(op.f('ix_drift_events_event_id'), 'drift_events', ['event_id'], unique=True)
    op.create_index(op.f('ix_drift_events_id'), 'drift_events', ['id'], unique=False)
    op.create_index(op.f('ix_drift_events_status'), 'drift_events', ['status'], unique=False)
    op.create_index(op.f('ix_drift_events_timestamp'), 'drift_events', ['timestamp'], unique=False)

    # Create drift_samples table
    op.create_table('drift_samples',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('sample_id', sa.String(length=255), nullable=False),
        sa.Column('drift_event_id', sa.String(length=255), nullable=False),
        sa.Column('prediction', sa.String(length=255), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('drift_score', sa.Float(), nullable=True),
        sa.Column('feature_path', sa.String(length=512), nullable=True),
        sa.Column('raw_data_path', sa.String(length=512), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['drift_event_id'], ['drift_events.event_id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('sample_id')
    )
    op.create_index(op.f('ix_drift_samples_drift_event_id'), 'drift_samples', ['drift_event_id'], unique=False)
    op.create_index(op.f('ix_drift_samples_id'), 'drift_samples', ['id'], unique=False)
    op.create_index(op.f('ix_drift_samples_sample_id'), 'drift_samples', ['sample_id'], unique=True)

    # Create drift_validations table
    op.create_table('drift_validations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('validation_id', sa.String(length=255), nullable=False),
        sa.Column('drift_event_id', sa.String(length=255), nullable=False),
        sa.Column('drift_sample_id', sa.String(length=255), nullable=True),
        sa.Column('is_valid_drift', sa.Boolean(), nullable=True),
        sa.Column('true_label', sa.String(length=255), nullable=True),
        sa.Column('validated_by', sa.String(length=255), nullable=True),
        sa.Column('validated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('validation_notes', sa.Text(), nullable=True),
        sa.Column('is_acknowledged', sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column('acknowledged_at', sa.DateTime(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['drift_event_id'], ['drift_events.event_id'], ),
        sa.ForeignKeyConstraint(['drift_sample_id'], ['drift_samples.sample_id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('validation_id')
    )
    op.create_index(op.f('ix_drift_validations_drift_event_id'), 'drift_validations', ['drift_event_id'], unique=False)
    op.create_index(op.f('ix_drift_validations_drift_sample_id'), 'drift_validations', ['drift_sample_id'], unique=False)
    op.create_index(op.f('ix_drift_validations_id'), 'drift_validations', ['id'], unique=False)
    op.create_index(op.f('ix_drift_validations_validation_id'), 'drift_validations', ['validation_id'], unique=True)
    op.create_index(op.f('ix_drift_validations_validated_at'), 'drift_validations', ['validated_at'], unique=False)
    op.create_index(op.f('ix_drift_validations_is_acknowledged'), 'drift_validations', ['is_acknowledged'], unique=False)

    # Add relationship to devices table
    op.execute('ALTER TABLE devices ADD COLUMN IF NOT EXISTS drift_event_count INTEGER DEFAULT 0')


def downgrade():
    # Remove relationship from devices table
    op.execute('ALTER TABLE devices DROP COLUMN IF EXISTS drift_event_count')
    
    # Drop tables
    op.drop_index(op.f('ix_drift_validations_is_acknowledged'), table_name='drift_validations')
    op.drop_index(op.f('ix_drift_validations_validated_at'), table_name='drift_validations')
    op.drop_index(op.f('ix_drift_validations_validation_id'), table_name='drift_validations')
    op.drop_index(op.f('ix_drift_validations_id'), table_name='drift_validations')
    op.drop_index(op.f('ix_drift_validations_drift_sample_id'), table_name='drift_validations')
    op.drop_index(op.f('ix_drift_validations_drift_event_id'), table_name='drift_validations')
    op.drop_table('drift_validations')
    
    op.drop_index(op.f('ix_drift_samples_sample_id'), table_name='drift_samples')
    op.drop_index(op.f('ix_drift_samples_id'), table_name='drift_samples')
    op.drop_index(op.f('ix_drift_samples_drift_event_id'), table_name='drift_samples')
    op.drop_table('drift_samples')
    
    op.drop_index(op.f('ix_drift_events_timestamp'), table_name='drift_events')
    op.drop_index(op.f('ix_drift_events_status'), table_name='drift_events')
    op.drop_index(op.f('ix_drift_events_id'), table_name='drift_events')
    op.drop_index(op.f('ix_drift_events_event_id'), table_name='drift_events')
    op.drop_index(op.f('ix_drift_events_device_id'), table_name='drift_events')
    op.drop_table('drift_events')
    
    # Drop enum types - only if they exist
    conn = op.get_bind()
    
    # Check if tables that use these enums are gone
    try:
        # Check if drifttype is still used
        result_type = conn.execute(sa.text(
            "SELECT 1 FROM pg_type WHERE typname = 'drifttype'"
        ))
        if result_type.scalar():
            try:
                sa.Enum(name='drifttype').drop(conn, checkfirst=True)
            except Exception as e:
                # Log the error but continue
                import logging
                logging.getLogger(__name__).warning(f"Could not drop drifttype enum: {e}")
        
        # Check if driftstatus is still used
        result_status = conn.execute(sa.text(
            "SELECT 1 FROM pg_type WHERE typname = 'driftstatus'"
        ))
        if result_status.scalar():
            try:
                sa.Enum(name='driftstatus').drop(conn, checkfirst=True)
            except Exception as e:
                # Log the error but continue
                import logging
                logging.getLogger(__name__).warning(f"Could not drop driftstatus enum: {e}")
    except Exception as e:
        # Log any other error but continue
        import logging
        logging.getLogger(__name__).warning(f"Error during enum cleanup: {e}")