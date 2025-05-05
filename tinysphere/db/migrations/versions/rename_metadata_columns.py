"""Rename metadata columns to avoid SQLAlchemy reserved word

Revision ID: rename_metadata_columns
Revises: add_drift_management_tables
Create Date: 2023-05-05 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'rename_metadata_columns'
down_revision = 'add_drift_management_tables'
branch_labels = None
depends_on = None


def upgrade():
    # Rename metadata columns to avoid SQLAlchemy reserved word conflict
    op.alter_column('drift_events', 'metadata', new_column_name='event_metadata')
    op.alter_column('drift_samples', 'metadata', new_column_name='sample_metadata')
    op.alter_column('drift_validations', 'metadata', new_column_name='validation_metadata')


def downgrade():
    # Rename columns back to original
    op.alter_column('drift_events', 'event_metadata', new_column_name='metadata')
    op.alter_column('drift_samples', 'sample_metadata', new_column_name='metadata')
    op.alter_column('drift_validations', 'validation_metadata', new_column_name='metadata')