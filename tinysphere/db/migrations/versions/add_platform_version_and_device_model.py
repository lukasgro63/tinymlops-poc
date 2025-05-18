"""add platform version and device model fields

Revision ID: add_platform_version_and_device_model
Revises: add_knn_distance_drift_type
Create Date: 2025-05-18 20:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_platform_version_and_device_model'
down_revision = 'add_knn_distance_drift_type'
branch_labels = None
depends_on = None


def upgrade():
    # Add platform_version and device_model columns to devices table
    op.add_column('devices', sa.Column('platform_version', sa.String(255), nullable=True))
    op.add_column('devices', sa.Column('device_model', sa.String(255), nullable=True))


def downgrade():
    # Remove the columns
    op.drop_column('devices', 'platform_version')
    op.drop_column('devices', 'device_model')