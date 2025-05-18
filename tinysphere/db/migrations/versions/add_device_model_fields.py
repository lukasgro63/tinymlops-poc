"""add platform version and device model fields

Revision ID: add_device_model_fields
Revises: add_geolocation_fields
Create Date: 2025-05-18 20:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_device_model_fields'
down_revision = 'add_geolocation_fields'
branch_labels = None
depends_on = None


def upgrade():
    # Check if columns already exist to handle idempotent migrations
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = inspector.get_columns('devices')
    column_names = [col['name'] for col in columns]
    
    # Add platform_version column if it doesn't exist
    if 'platform_version' not in column_names:
        op.add_column('devices', sa.Column('platform_version', sa.String(255), nullable=True))
    
    # Add device_model column if it doesn't exist
    if 'device_model' not in column_names:
        op.add_column('devices', sa.Column('device_model', sa.String(255), nullable=True))


def downgrade():
    # Remove the columns
    op.drop_column('devices', 'platform_version')
    op.drop_column('devices', 'device_model')