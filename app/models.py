from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dimension: Mapped[str] = mapped_column(String(2), nullable=False, index=True)
    agree_pole: Mapped[str] = mapped_column(String(1), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="ai")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )


class Test(Base):
    __tablename__ = "tests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mode: Mapped[int] = mapped_column(Integer, nullable=False)
    target_count: Mapped[int] = mapped_column(Integer, nullable=False)
    extra_max: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="in_progress", index=True)

    resume_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    test_token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    resume_token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    resume_code_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)

    share_token_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True, index=True)
    share_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    result_type: Mapped[str | None] = mapped_column(String(4), nullable=True)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    answers: Mapped[list["Answer"]] = relationship(
        "Answer",
        back_populates="test",
        cascade="all, delete-orphan",
        order_by="Answer.answered_at",
    )


class TestItem(Base):
    __tablename__ = "test_items"
    __table_args__ = (
        UniqueConstraint("test_id", "position", name="uq_test_items_test_position"),
        UniqueConstraint("test_id", "question_id", name="uq_test_items_test_question"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    test_id: Mapped[int] = mapped_column(ForeignKey("tests.id"), nullable=False, index=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id"), nullable=False, index=True)
    is_extra: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class Answer(Base):
    __tablename__ = "answers"
    __table_args__ = (UniqueConstraint("test_id", "question_id", name="uq_answers_test_question"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    test_id: Mapped[int] = mapped_column(ForeignKey("tests.id"), nullable=False, index=True)
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id"), nullable=False, index=True)
    value: Mapped[int] = mapped_column(Integer, nullable=False)
    answered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)

    test: Mapped["Test"] = relationship("Test", back_populates="answers")
    question: Mapped["Question"] = relationship("Question")
