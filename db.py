from __future__ import annotations
from sqlalchemy import text  # add at top if not imported

def drop_all() -> None:
    """Dangerous: drop all tables in the current database."""
    Base.metadata.drop_all(bind=engine)

def reset_database() -> None:
    """Drop all tables and recreate an empty schema."""
    drop_all()
    init_db()

def migrate_drop_columns(columns_to_drop):
    """
    Rebuild 'results' table without specified columns (SQLite-friendly).
    Example: migrate_drop_columns(["source_file", "key_set"])
    """
    init_db()
    with get_session() as s:
        s.execute(text("""
        CREATE TABLE IF NOT EXISTS results_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id VARCHAR(64),
            subject_scores JSON,
            total_score INTEGER,
            evaluation_time DATETIME,
            confidence_score FLOAT,
            key_set VARCHAR(8),
            source_file VARCHAR(256)
        );
        """))
        keep_cols = ["id","student_id","subject_scores","total_score","evaluation_time","confidence_score","key_set","source_file"]
        keep_cols = [c for c in keep_cols if c.lower() not in {x.lower() for x in columns_to_drop}]
        cols_csv = ", ".join(keep_cols)
        s.execute(text(f"INSERT INTO results_new ({cols_csv}) SELECT {cols_csv} FROM results"))
        s.execute(text("DROP TABLE results"))
        s.execute(text("ALTER TABLE results_new RENAME TO results"))
        s.commit()

import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
from sqlalchemy import (
    create_engine,
    Integer,
    String,
    DateTime,
    Float,
    JSON,
    select,
    func,
    desc,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, Session

# Database file in project root
DB_PATH = os.path.join(os.path.dirname(__file__), "omr.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, future=True, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class Result(Base):
    __tablename__ = "results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(String(64), index=True)
    subject_scores: Mapped[Dict[str, int]] = mapped_column(JSON)
    total_score: Mapped[int] = mapped_column(Integer)
    evaluation_time: Mapped[datetime] = mapped_column(DateTime, index=True)
    confidence_score: Mapped[float] = mapped_column(Float)
    key_set: Mapped[Optional[str]] = mapped_column(String(8), default=None, index=True)
    source_file: Mapped[Optional[str]] = mapped_column(String(256), default=None)

def init_db() -> None:
    Base.metadata.create_all(bind=engine)

def get_session() -> Session:
    return SessionLocal()

def save_result(
    *,
    student_id: str,
    subject_scores: Dict[str, int],
    total_score: int,
    evaluation_time_str: str,
    confidence_score: float,
    key_set: Optional[str] = None,
    source_file: Optional[str] = None,
) -> int:
    """
    Persist a result to the database. Returns the new row id.
    evaluation_time_str must be like "YYYY-MM-DD HH:MM:SS".
    """
    init_db()
    # Accept either "YYYY-MM-DD HH:MM:SS" or ISO with seconds
    when: datetime
    try:
        when = datetime.strptime(evaluation_time_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        # Fallback to more flexible parse if needed
        try:
            when = datetime.fromisoformat(evaluation_time_str)
        except Exception:
            # Default to now if unparseable (keeps system running)
            when = datetime.utcnow()

    with get_session() as s:
        row = Result(
            student_id=student_id,
            subject_scores=subject_scores,
            total_score=total_score,
            evaluation_time=when,
            confidence_score=confidence_score,
            key_set=key_set,
            source_file=source_file,
        )
        s.add(row)
        s.commit()
        s.refresh(row)
        return row.id

def _fetch_all_results(limit: Optional[int] = None) -> List[Result]:
    init_db()
    with get_session() as s:
        stmt = select(Result).order_by(desc(Result.evaluation_time))
        if limit:
            stmt = stmt.limit(limit)
        rows = list(s.execute(stmt).scalars().all())
        return rows

def get_recent_dataframe(limit: int = 25) -> pd.DataFrame:
    """
    Returns a DataFrame of recent evaluations with columns:
    Student_ID, Total, Evaluation_Time, Confidence, Key_Set, plus per-subject columns.
    """
    rows = _fetch_all_results(limit=limit)
    if not rows:
        return pd.DataFrame()

    # Collect all subject names that appear in the data (union)
    subject_set = set()
    for r in rows:
        if isinstance(r.subject_scores, dict):
            subject_set.update(r.subject_scores.keys())
    subjects = sorted(subject_set)

    records: List[Dict[str, Any]] = []
    for r in rows:
        rec: Dict[str, Any] = {
            "Student_ID": r.student_id,
            "Total": r.total_score,
            "Evaluation_Time": r.evaluation_time.strftime("%Y-%m-%d %H:%M:%S") if r.evaluation_time else None,
            "Confidence": r.confidence_score,
            "Key_Set": r.key_set or "",
            "Source_File": r.source_file or "",
        }
        for s_name in subjects:
            rec[s_name] = int(r.subject_scores.get(s_name, 0)) if isinstance(r.subject_scores, dict) else 0
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    return df

def get_aggregates() -> Dict[str, Any]:
    """
    Compute aggregates for the dashboard:
      - total_students
      - average_total (0–100)
      - pass_rate (% with Total >= 50)
      - subject_averages: dict of subject -> mean (0–20)
      - grade_distribution: dict of grade bucket -> count
      - recent_df: pandas DataFrame (last 25)
    Returns an empty structure if there are no rows.
    """
    init_db()
    rows = _fetch_all_results(limit=None)
    if not rows:
        return {
            "total_students": 0,
            "average_total": 0.0,
            "pass_rate": 0.0,
            "subject_averages": {},
            "grade_distribution": {},
            "recent_df": pd.DataFrame(),
        }

    # Build a full DataFrame for computation
    df = get_recent_dataframe(limit=1000000)  # effectively all
    if df.empty:
        return {
            "total_students": 0,
            "average_total": 0.0,
            "pass_rate": 0.0,
            "subject_averages": {},
            "grade_distribution": {},
            "recent_df": pd.DataFrame(),
        }

    total_students = int(len(df))
    avg_total = float(df["Total"].mean()) if "Total" in df.columns else 0.0
    pass_rate = float((df["Total"] >= 50).mean() * 100.0) if "Total" in df.columns else 0.0

    # Subject columns are any numeric columns except known meta fields
    meta_cols = {"Student_ID", "Total", "Evaluation_Time", "Confidence", "Key_Set", "Source_File"}
    subject_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
    subject_averages = {c: float(df[c].mean()) for c in subject_cols}

    # Grade distribution by total percentage
    def grade_bucket(pct: float) -> str:
        if pct >= 90: return "A+ (90-100)"
        if pct >= 80: return "A (80-89)"
        if pct >= 70: return "B (70-79)"
        if pct >= 60: return "C (60-69)"
        if pct >= 50: return "D (50-59)"
        return "F (<50)"

    grade_counts: Dict[str, int] = {}
    if "Total" in df.columns:
        for v in df["Total"].tolist():
            bucket = grade_bucket(float(v))
            grade_counts[bucket] = grade_counts.get(bucket, 0) + 1

    # Recent evaluations table (most recent 25)
    recent_df = get_recent_dataframe(limit=25)

    return {
        "total_students": total_students,
        "average_total": avg_total,
        "pass_rate": pass_rate,
        "subject_averages": subject_averages,
        "grade_distribution": grade_counts,
        "recent_df": recent_df,
    }
