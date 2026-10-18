"""Create chunks table with foreign key cascade, indexing, and structural constraints.

Revision ID: 0003_create_chunks_table
Revises: 0002_create_documents_table
Create Date: 2026-10-18
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0003_create_chunks_table"
down_revision: Union[str, None] = "0002_create_documents_table"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create chunks table, foreign keys, constraints, and indexes."""
    op.create_table(
        "chunks",
        sa.Column("id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("character_count", sa.Integer(), nullable=False),
        sa.Column("start_char_offset", sa.Integer(), nullable=False),
        sa.Column("end_char_offset", sa.Integer(), nullable=False),
        sa.Column("source_format", sa.String(length=100), nullable=False, server_default="text/plain"),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column(
            "page_numbers",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column("heading", sa.String(length=500), nullable=True),
        sa.Column("heading_level", sa.Integer(), nullable=True),
        sa.Column(
            "metadata",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chunks"),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_chunks_document_id_documents",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_chunk_index"),
        sa.CheckConstraint("chunk_index >= 0", name="check_chunk_index_non_negative"),
        sa.CheckConstraint("token_count >= 0", name="check_chunk_token_count_non_negative"),
        sa.CheckConstraint("character_count >= 0", name="check_chunk_character_count_non_negative"),
        sa.CheckConstraint("start_char_offset >= 0", name="check_chunk_start_char_offset_non_negative"),
        sa.CheckConstraint("end_char_offset >= start_char_offset", name="check_chunk_end_char_offset_valid"),
        sa.CheckConstraint(
            "end_char_offset - start_char_offset = character_count",
            name="check_chunk_offset_span_equals_character_count",
        ),
        sa.CheckConstraint("page_number IS NULL OR page_number >= 1", name="check_chunk_page_number_positive"),
        sa.CheckConstraint(
            "heading_level IS NULL OR (heading_level >= 1 AND heading_level <= 6)",
            name="check_chunk_heading_level_valid",
        ),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"], unique=False)
    op.create_index("ix_chunks_created_at", "chunks", ["created_at"], unique=False)


def downgrade() -> None:
    """Drop chunks table and associated indexes."""
    op.drop_index("ix_chunks_created_at", table_name="chunks")
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_table("chunks")
