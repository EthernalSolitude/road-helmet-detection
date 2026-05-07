"""initial violations table

Revision ID: 0001
Revises:
Create Date: 2026-05-07 00:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "violations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("video_name", sa.String(), nullable=True),
        sa.Column("track_id", sa.Integer(), nullable=True),
        sa.Column("frame_idx", sa.Integer(), nullable=True),
        sa.Column("bbox", sa.String(), nullable=True),
        sa.Column("ratio_no_helmet", sa.Float(), nullable=True),
        sa.Column("image_path", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_violations_video_name", "violations", ["video_name"])
    op.create_index("ix_violations_track_id", "violations", ["track_id"])


def downgrade() -> None:
    op.drop_index("ix_violations_track_id", table_name="violations")
    op.drop_index("ix_violations_video_name", table_name="violations")
    op.drop_table("violations")
