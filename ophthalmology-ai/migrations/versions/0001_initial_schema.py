"""Initial schema: users, patients, exams, images, models, analyses, results and audit log.

Revision ID: 0001_initial
Revises:
Create Date: 2026-08-31 23:16:44.817025
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "models",
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("modality", sa.String(length=16), nullable=False),
        sa.Column("task", sa.String(length=24), nullable=False),
        sa.Column("framework", sa.String(length=24), nullable=False),
        sa.Column("evidence_level", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("weights_sha256", sa.String(length=64), nullable=True),
        sa.Column("license_name", sa.String(length=120), nullable=True),
        sa.Column("license_url", sa.String(length=512), nullable=True),
        sa.Column("source_url", sa.String(length=512), nullable=True),
        sa.Column("commercial_use", sa.String(length=32), nullable=True),
        sa.Column(
            "input_spec", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True
        ),
        sa.Column(
            "output_spec", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True
        ),
        sa.Column(
            "reported_metrics",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_models")),
        sa.UniqueConstraint("model_id", "version", name="uq_models_model_id_version"),
    )
    with op.batch_alter_table("models", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_models_model_id"), ["model_id"], unique=False)

    op.create_table(
        "users",
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("full_name", sa.String(length=200), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
    )
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_users_email"), ["email"], unique=True)

    op.create_table(
        "audit_logs",
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actor_user_id", sa.Uuid(), nullable=True),
        sa.Column("actor_role", sa.String(length=32), nullable=True),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("resource_type", sa.String(length=64), nullable=True),
        sa.Column("resource_id", sa.String(length=64), nullable=True),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("client_ip_hash", sa.String(length=64), nullable=True),
        sa.Column("outcome", sa.String(length=16), nullable=False),
        sa.Column("meta", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["actor_user_id"], ["users.id"], name=op.f("fk_audit_logs_actor_user_id_users")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_audit_logs")),
    )
    with op.batch_alter_table("audit_logs", schema=None) as batch_op:
        batch_op.create_index(
            "ix_audit_logs_action_created_at", ["action", "created_at"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_audit_logs_actor_user_id"), ["actor_user_id"], unique=False
        )
        batch_op.create_index(batch_op.f("ix_audit_logs_created_at"), ["created_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_audit_logs_request_id"), ["request_id"], unique=False)
        batch_op.create_index(
            batch_op.f("ix_audit_logs_resource_id"), ["resource_id"], unique=False
        )

    op.create_table(
        "patients",
        sa.Column("external_ref", sa.String(length=64), nullable=False),
        sa.Column("birth_year", sa.Integer(), nullable=True),
        sa.Column("sex", sa.String(length=16), nullable=True),
        sa.Column("consent_research", sa.Boolean(), nullable=False),
        sa.Column(
            "clinical_context",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
        sa.Column("created_by_id", sa.Uuid(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "birth_year IS NULL OR birth_year > 1850", name=op.f("ck_patients_birth_year_plausible")
        ),
        sa.ForeignKeyConstraint(
            ["created_by_id"], ["users.id"], name=op.f("fk_patients_created_by_id_users")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_patients")),
        sa.UniqueConstraint("external_ref", name="uq_patients_external_ref"),
    )
    with op.batch_alter_table("patients", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_patients_external_ref"), ["external_ref"], unique=False
        )

    op.create_table(
        "exams",
        sa.Column("patient_id", sa.Uuid(), nullable=False),
        sa.Column("modality", sa.String(length=16), nullable=False),
        sa.Column("laterality", sa.String(length=16), nullable=False),
        sa.Column("acquired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("device_manufacturer", sa.String(length=120), nullable=True),
        sa.Column("device_model", sa.String(length=120), nullable=True),
        sa.Column(
            "acquisition_metadata",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
        sa.Column("created_by_id", sa.Uuid(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["created_by_id"], ["users.id"], name=op.f("fk_exams_created_by_id_users")
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["patients.id"],
            name=op.f("fk_exams_patient_id_patients"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_exams")),
    )
    with op.batch_alter_table("exams", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_exams_patient_id"), ["patient_id"], unique=False)

    op.create_table(
        "images",
        sa.Column("exam_id", sa.Uuid(), nullable=False),
        sa.Column("storage_bucket", sa.String(length=128), nullable=False),
        sa.Column("storage_key", sa.String(length=512), nullable=False),
        sa.Column("content_type", sa.String(length=64), nullable=False),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("num_frames", sa.Integer(), nullable=False),
        sa.Column(
            "pixel_spacing_um",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("original_extension", sa.String(length=16), nullable=True),
        sa.Column("uploaded_by_id", sa.Uuid(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["exam_id"], ["exams.id"], name=op.f("fk_images_exam_id_exams"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["uploaded_by_id"], ["users.id"], name=op.f("fk_images_uploaded_by_id_users")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_images")),
    )
    with op.batch_alter_table("images", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_images_checksum_sha256"), ["checksum_sha256"], unique=False
        )
        batch_op.create_index(batch_op.f("ix_images_exam_id"), ["exam_id"], unique=False)
        batch_op.create_index(
            "ix_images_exam_id_created_at", ["exam_id", "created_at"], unique=False
        )

    op.create_table(
        "analyses",
        sa.Column("exam_id", sa.Uuid(), nullable=False),
        sa.Column("image_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column(
            "requested_models",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "pipeline_config",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
        sa.Column(
            "quality_summary",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("software_version", sa.String(length=32), nullable=True),
        sa.Column("task_id", sa.String(length=128), nullable=True),
        sa.Column("created_by_id", sa.Uuid(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["created_by_id"], ["users.id"], name=op.f("fk_analyses_created_by_id_users")
        ),
        sa.ForeignKeyConstraint(
            ["exam_id"], ["exams.id"], name=op.f("fk_analyses_exam_id_exams"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["image_id"],
            ["images.id"],
            name=op.f("fk_analyses_image_id_images"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_analyses")),
    )
    with op.batch_alter_table("analyses", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_analyses_exam_id"), ["exam_id"], unique=False)
        batch_op.create_index(
            "ix_analyses_status_created_at", ["status", "created_at"], unique=False
        )

    op.create_table(
        "model_runs",
        sa.Column("analysis_id", sa.Uuid(), nullable=False),
        sa.Column("image_id", sa.Uuid(), nullable=True),
        sa.Column("model_record_id", sa.Uuid(), nullable=True),
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("model_version", sa.String(length=32), nullable=False),
        sa.Column("task", sa.String(length=24), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("device", sa.String(length=32), nullable=False),
        sa.Column("device_name", sa.String(length=128), nullable=True),
        sa.Column("precision", sa.String(length=8), nullable=False),
        sa.Column("batch_size", sa.Integer(), nullable=False),
        sa.Column("processing_time_ms", sa.Float(), nullable=True),
        sa.Column("vram_used_mb", sa.Float(), nullable=True),
        sa.Column("input_hash", sa.String(length=64), nullable=True),
        sa.Column("software_version", sa.String(length=32), nullable=True),
        sa.Column(
            "measurements", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True
        ),
        sa.Column(
            "quality", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True
        ),
        sa.Column(
            "warnings", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["analysis_id"],
            ["analyses.id"],
            name=op.f("fk_model_runs_analysis_id_analyses"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["image_id"],
            ["images.id"],
            name=op.f("fk_model_runs_image_id_images"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["model_record_id"], ["models.id"], name=op.f("fk_model_runs_model_record_id_models")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_runs")),
    )
    with op.batch_alter_table("model_runs", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_model_runs_analysis_id"), ["analysis_id"], unique=False
        )
        batch_op.create_index(batch_op.f("ix_model_runs_input_hash"), ["input_hash"], unique=False)
        batch_op.create_index(
            "ix_model_runs_model_id_created_at", ["model_id", "created_at"], unique=False
        )

    op.create_table(
        "reports",
        sa.Column("analysis_id", sa.Uuid(), nullable=False),
        sa.Column("format", sa.String(length=8), nullable=False),
        sa.Column(
            "payload", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True
        ),
        sa.Column("storage_bucket", sa.String(length=128), nullable=True),
        sa.Column("storage_key", sa.String(length=512), nullable=True),
        sa.Column("disclaimer_version", sa.String(length=16), nullable=False),
        sa.Column("created_by_id", sa.Uuid(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["analysis_id"],
            ["analyses.id"],
            name=op.f("fk_reports_analysis_id_analyses"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_id"], ["users.id"], name=op.f("fk_reports_created_by_id_users")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reports")),
    )
    with op.batch_alter_table("reports", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_reports_analysis_id"), ["analysis_id"], unique=False)

    op.create_table(
        "artifacts",
        sa.Column("model_run_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("storage_bucket", sa.String(length=128), nullable=False),
        sa.Column("storage_key", sa.String(length=512), nullable=False),
        sa.Column("content_type", sa.String(length=64), nullable=False),
        sa.Column("meta", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["model_run_id"],
            ["model_runs.id"],
            name=op.f("fk_artifacts_model_run_id_model_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_artifacts")),
    )
    with op.batch_alter_table("artifacts", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_artifacts_model_run_id"), ["model_run_id"], unique=False
        )

    op.create_table(
        "predictions",
        sa.Column("model_run_id", sa.Uuid(), nullable=False),
        sa.Column("label", sa.String(length=120), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("extra", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["model_run_id"],
            ["model_runs.id"],
            name=op.f("fk_predictions_model_run_id_model_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_predictions")),
    )
    with op.batch_alter_table("predictions", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_predictions_model_run_id"), ["model_run_id"], unique=False
        )

    op.create_table(
        "segmentations",
        sa.Column("model_run_id", sa.Uuid(), nullable=False),
        sa.Column("label", sa.String(length=120), nullable=False),
        sa.Column("storage_bucket", sa.String(length=128), nullable=False),
        sa.Column("storage_key", sa.String(length=512), nullable=False),
        sa.Column("content_type", sa.String(length=64), nullable=False),
        sa.Column("area_px", sa.Float(), nullable=True),
        sa.Column("area_ratio", sa.Float(), nullable=True),
        sa.Column(
            "measurements", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True
        ),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["model_run_id"],
            ["model_runs.id"],
            name=op.f("fk_segmentations_model_run_id_model_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_segmentations")),
    )
    with op.batch_alter_table("segmentations", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_segmentations_model_run_id"), ["model_run_id"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table("segmentations", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_segmentations_model_run_id"))

    op.drop_table("segmentations")
    with op.batch_alter_table("predictions", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_predictions_model_run_id"))

    op.drop_table("predictions")
    with op.batch_alter_table("artifacts", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_artifacts_model_run_id"))

    op.drop_table("artifacts")
    with op.batch_alter_table("reports", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_reports_analysis_id"))

    op.drop_table("reports")
    with op.batch_alter_table("model_runs", schema=None) as batch_op:
        batch_op.drop_index("ix_model_runs_model_id_created_at")
        batch_op.drop_index(batch_op.f("ix_model_runs_input_hash"))
        batch_op.drop_index(batch_op.f("ix_model_runs_analysis_id"))

    op.drop_table("model_runs")
    with op.batch_alter_table("analyses", schema=None) as batch_op:
        batch_op.drop_index("ix_analyses_status_created_at")
        batch_op.drop_index(batch_op.f("ix_analyses_exam_id"))

    op.drop_table("analyses")
    with op.batch_alter_table("images", schema=None) as batch_op:
        batch_op.drop_index("ix_images_exam_id_created_at")
        batch_op.drop_index(batch_op.f("ix_images_exam_id"))
        batch_op.drop_index(batch_op.f("ix_images_checksum_sha256"))

    op.drop_table("images")
    with op.batch_alter_table("exams", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_exams_patient_id"))

    op.drop_table("exams")
    with op.batch_alter_table("patients", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_patients_external_ref"))

    op.drop_table("patients")
    with op.batch_alter_table("audit_logs", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_audit_logs_resource_id"))
        batch_op.drop_index(batch_op.f("ix_audit_logs_request_id"))
        batch_op.drop_index(batch_op.f("ix_audit_logs_created_at"))
        batch_op.drop_index(batch_op.f("ix_audit_logs_actor_user_id"))
        batch_op.drop_index("ix_audit_logs_action_created_at")

    op.drop_table("audit_logs")
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_users_email"))

    op.drop_table("users")
    with op.batch_alter_table("models", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_models_model_id"))

    op.drop_table("models")
