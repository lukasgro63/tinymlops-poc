"""Add notifications table

Revision ID: add_notifications_table
Revises: 
Create Date: 2023-05-03 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_notifications_table'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create notifications table with enum type reference
    # Da der ENUM-Typ möglicherweise schon von SQLAlchemy erstellt wurde,
    # verwenden wir hier die String-Repräsentation, um darauf zu verweisen
    try:
        # Wir überprüfen zuerst, ob die Tabelle bereits existiert
        conn = op.get_bind()
        result = conn.execute(sa.text(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'notifications'"
        ))
        
        table_exists = result.fetchone() is not None
        
        if not table_exists:
            # Die Tabelle existiert noch nicht, also erstellen wir sie
            # Wir verwenden postgresql.ENUM für die Spalte, um auf einen bestehenden Typ zu verweisen
            op.create_table('notifications',
                sa.Column('id', sa.Integer(), nullable=False),
                sa.Column('message', sa.String(length=512), nullable=False),
                sa.Column('notification_type', postgresql.ENUM('info', 'warning', 'error', 'success', name='notificationtype', create_type=False), 
                       nullable=False, server_default='info'),
                sa.Column('source', sa.String(length=100), nullable=True),
                sa.Column('source_id', sa.String(length=255), nullable=True),
                sa.Column('details', sa.JSON(), nullable=True),
                sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
                sa.Column('read', sa.Boolean(), nullable=False, server_default=sa.text('false')),
                sa.Column('read_at', sa.DateTime(), nullable=True),
                sa.PrimaryKeyConstraint('id')
            )
        else:
            # Die Tabelle existiert bereits, also machen wir nichts
            pass
            
    except Exception as e:
        # Im Fehlerfall geben wir eine klare Fehlermeldung aus
        import logging
        logging.error(f"Failed to create notifications table: {str(e)}")
        # Wir werfen den Fehler weiter
        raise
    
    # Indizes nur erstellen, wenn sie noch nicht existieren
    try:
        # Überprüfen, welche Indizes bereits existieren
        conn = op.get_bind()
        result = conn.execute(sa.text(
            "SELECT indexname FROM pg_indexes WHERE tablename = 'notifications'"
        ))
        existing_indexes = {row[0] for row in result}
        
        # Erstelle nur Indizes, die noch nicht existieren
        if 'ix_notifications_id' not in existing_indexes:
            op.create_index(op.f('ix_notifications_id'), 'notifications', ['id'], unique=False)
            
        if 'ix_notifications_created_at' not in existing_indexes:
            op.create_index(op.f('ix_notifications_created_at'), 'notifications', ['created_at'], unique=False)
            
        if 'ix_notifications_read' not in existing_indexes:
            op.create_index(op.f('ix_notifications_read'), 'notifications', ['read'], unique=False)
            
        if 'ix_notifications_source_id' not in existing_indexes:
            op.create_index(op.f('ix_notifications_source_id'), 'notifications', ['source_id'], unique=False)
    
    except Exception as e:
        logging.error(f"Failed to create indexes: {str(e)}")
        raise


def downgrade():
    op.drop_index(op.f('ix_notifications_source_id'), table_name='notifications')
    op.drop_index(op.f('ix_notifications_read'), table_name='notifications')
    op.drop_index(op.f('ix_notifications_created_at'), table_name='notifications')
    op.drop_index(op.f('ix_notifications_id'), table_name='notifications')
    op.drop_table('notifications')
    
    # Drop enum type
    sa.Enum(name='notificationtype').drop(op.get_bind())