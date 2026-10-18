"""Create chunk_embeddings table with vector(384), foreign keys, and constraints.

Revision ID: 0004_create_chunk_embeddings_table
Revises: 0003_create_chunks_table
Create Date: 2026-10-18
"""

from typing import Sequence, Union

from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0004_create_embeddings_table"
down_revision: Union[str, None] = "0003_create_chunks_table"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create chunk_embeddings table, foreign keys, constraints, and indexes."""
    op.create_table(
        "chunk_embeddings",
        sa.Column("id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("model_id", sa.String(length=100), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chunk_embeddings"),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["chunks.id"],
            name="fk_chunk_embeddings_chunk_id_chunks",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_chunk_embeddings_document_id_documents",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "chunk_id",
            "model_id",
            name="uq_chunk_embeddings_chunk_model",
        ),
        sa.CheckConstraint(
            "dimension = 384",
            name="check_chunk_embedding_dimension_384",
        ),
    )
    op.create_index("ix_chunk_embeddings_chunk_id", "chunk_embeddings", ["chunk_id"], unique=False)
    op.create_index("ix_chunk_embeddings_document_id", "chunk_embeddings", ["document_id"], unique=False)
    op.create_index("ix_chunk_embeddings_model_id", "chunk_embeddings", ["model_id"], unique=False)
    op.create_index("ix_chunk_embeddings_created_at", "chunk_embeddings", ["created_at"], unique=False)


def downgrade() -> None:
    """Drop chunk_embeddings table and associated indexes."""
    op.drop_index("ix_chunk_embeddings_created_at", table_name="chunk_embeddings")
    op.drop_index("ix_chunk_embeddings_model_id", table_name="chunk_embeddings")
    op.drop_index("ix_chunk_embeddings_document_id", table_name="chunk_embeddings")
    op.drop_index("ix_chunk_embeddings_chunk_id", table_name="chunk_embeddings")
    op.drop_table("chunk_embeddings")
