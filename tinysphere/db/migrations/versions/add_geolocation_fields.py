"""Add geolocation fields to devices table

Revision ID: add_geolocation_fields
Revises: add_knn_distance_drift_type
Create Date: 2025-05-17 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_geolocation_fields'
down_revision = 'add_knn_distance_drift_type'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Überprüfe, ob die Spalten bereits existieren
    conn = op.get_bind()
    
    # Abfragen, welche Spalten bereits existieren
    result = conn.execute(sa.text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'devices' AND column_name IN ('latitude', 'longitude', 'geo_accuracy')"
    ))
    
    existing_columns = {row[0] for row in result}
    
    # Füge nur Spalten hinzu, die noch nicht existieren
    if 'latitude' not in existing_columns:
        op.add_column('devices', sa.Column('latitude', sa.Float(), nullable=True))
    
    if 'longitude' not in existing_columns:
        op.add_column('devices', sa.Column('longitude', sa.Float(), nullable=True))
    
    if 'geo_accuracy' not in existing_columns:
        op.add_column('devices', sa.Column('geo_accuracy', sa.Float(), nullable=True))


def downgrade() -> None:
    # Remove geolocation columns from devices table
    conn = op.get_bind()
    
    # Prüfen, welche Spalten existieren, bevor wir versuchen, sie zu entfernen
    result = conn.execute(sa.text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'devices' AND column_name IN ('latitude', 'longitude', 'geo_accuracy')"
    ))
    
    existing_columns = {row[0] for row in result}
    
    # Entferne nur Spalten, die tatsächlich existieren
    if 'geo_accuracy' in existing_columns:
        op.drop_column('devices', 'geo_accuracy')
    
    if 'longitude' in existing_columns:
        op.drop_column('devices', 'longitude')
    
    if 'latitude' in existing_columns:
        op.drop_column('devices', 'latitude')