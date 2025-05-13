"""Add KNN_DISTANCE drift type

Revision ID: add_knn_distance_drift_type
Revises: rename_metadata_columns
Create Date: 2025-05-13 18:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_knn_distance_drift_type'
down_revision = 'rename_metadata_columns'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # PostgreSQL way to add a new value to an enum type
    op.execute("ALTER TYPE drifttype ADD VALUE IF NOT EXISTS 'knn_distance'")


def downgrade() -> None:
    # Cannot safely remove a value from an enum in PostgreSQL
    # Would need to create a new type, update the column, and drop the old type
    pass