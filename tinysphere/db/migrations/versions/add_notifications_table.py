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
    # Create enum type for notification_type
    notification_type = sa.Enum('info', 'warning', 'error', 'success', name='notificationtype')
    notification_type.create(op.get_bind())
    
    # Create notifications table
    op.create_table('notifications',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('message', sa.String(length=512), nullable=False),
        sa.Column('notification_type', notification_type, nullable=False, server_default='info'),
        sa.Column('source', sa.String(length=100), nullable=True),
        sa.Column('source_id', sa.String(length=255), nullable=True),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('read', sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column('read_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_notifications_id'), 'notifications', ['id'], unique=False)
    op.create_index(op.f('ix_notifications_created_at'), 'notifications', ['created_at'], unique=False)
    op.create_index(op.f('ix_notifications_read'), 'notifications', ['read'], unique=False)
    op.create_index(op.f('ix_notifications_source_id'), 'notifications', ['source_id'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_notifications_source_id'), table_name='notifications')
    op.drop_index(op.f('ix_notifications_read'), table_name='notifications')
    op.drop_index(op.f('ix_notifications_created_at'), table_name='notifications')
    op.drop_index(op.f('ix_notifications_id'), table_name='notifications')
    op.drop_table('notifications')
    
    # Drop enum type
    sa.Enum(name='notificationtype').drop(op.get_bind())