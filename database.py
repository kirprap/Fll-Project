import os
from datetime import datetime
from typing import Optional, Dict, Any, List, Union

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    Float,
    DateTime,
    LargeBinary,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import base64
from sqlalchemy import inspect, text

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Prefer the DATABASE_URL environment variable (set by docker-compose or .env).
# If it is missing or empty, fall back to a local SQLite file.
_DB_URL = os.getenv("DATABASE_URL")
if not _DB_URL:
    # Default SQLite database located in the container's /app directory.
    _DB_URL = "sqlite:///artifacts.db"

# SQLite requires a special ``check_same_thread`` flag; other DBMS do not.
# Add connection pooling for better performance
if _DB_URL.startswith("sqlite"):
    engine = create_engine(
        _DB_URL,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,  # Verify connections before using
        pool_recycle=3600,   # Recycle connections after 1 hour
    )
else:
    # PostgreSQL connection pooling
    engine = create_engine(
        _DB_URL,
        pool_size=10,           # Number of connections to maintain
        max_overflow=20,        # Additional connections when pool is full
        pool_pre_ping=True,     # Verify connections before using
        pool_recycle=3600,      # Recycle connections after 1 hour
        echo=False,             # Disable SQL logging for performance
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# ----------------------------------------------------------------------
# ORM Model
# ----------------------------------------------------------------------
class Artifact(Base):
    """Model for storing identified archaeological artifacts"""

    __tablename__ = "artifacts"

    id: int = Column(Integer, primary_key=True, index=True)

    # Core artifact information from AI analysis
    name: str = Column(String(500), nullable=False)
    value: Optional[str] = Column(String(200))
    age: Optional[str] = Column(String(300))
    description: Optional[str] = Column(Text)
    cultural_context: Optional[str] = Column(Text)
    material: Optional[str] = Column(String(500))
    function: Optional[str] = Column(Text)
    rarity: Optional[str] = Column(String(200))
    confidence: Optional[float] = Column(Float)

    # Image data
    image_data: Optional[bytes] = Column(LargeBinary)

    # Timestamps
    uploaded_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)
    analyzed_at: Optional[datetime] = Column(DateTime, default=datetime.utcnow)

    # Expert verification fields
    verification_status: str = Column(String(50), default="pending")
    verified_by: Optional[str] = Column(String(200))
    verified_at: Optional[datetime] = Column(DateTime)
    verification_comments: Optional[str] = Column(Text)

    # Detailed profile fields
    provenance: Optional[str] = Column(Text)
    historical_context: Optional[str] = Column(Text)
    references: Optional[str] = Column(Text)
    # Tags (comma-separated)
    tags: Optional[str] = Column(Text)

    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact to a plain‑dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "value": self.value,
            "age": self.age,
            "description": self.description,
            "cultural_context": self.cultural_context,
            "material": self.material,
            "function": self.function,
            "rarity": self.rarity,
            "confidence": self.confidence,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
            "analyzed_at": self.analyzed_at.isoformat() if self.analyzed_at else None,
            "verification_status": self.verification_status,
            "verified_by": self.verified_by,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "verification_comments": self.verification_comments,
            "provenance": self.provenance,
            "historical_context": self.historical_context,
            "references": self.references,
            "tags": self.tags,
        }


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def init_db() -> None:
    """Create all tables defined by the ORM models."""
    Base.metadata.create_all(bind=engine)
    # Ensure 'tags' column exists for existing databases
    try:
        inspector = inspect(engine)
        columns = [c["name"] for c in inspector.get_columns("artifacts")]
        if "tags" not in columns:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE artifacts ADD COLUMN tags TEXT"))
                conn.commit()
    except Exception:
        # Best-effort; ignore if not supported or already exists
        pass


@contextmanager
def get_db():
    """Yield a DB session and ensure proper cleanup/commit handling."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _normalize_tags_input(tags: Optional[Union[List[str], str]]) -> List[str]:
    """Return a de-duplicated list of trimmed tags, collapsing extra spaces."""
    if tags is None:
        return []

    if isinstance(tags, str):
        candidates: List[str] = [tags]
    else:
        candidates = [str(t) for t in tags if t is not None]

    seen = set()
    normalized: List[str] = []
    for item in candidates:
        fragments = item.replace("\n", ",").split(",")
        for fragment in fragments:
            cleaned = " ".join(fragment.strip().split())
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                normalized.append(cleaned)
    return normalized


def save_artifact(artifact_data: Dict[str, Any], image_bytes: bytes) -> int:
    """Persist a newly analysed artifact and return its primary key."""
    with get_db() as db:
        tags_list = _normalize_tags_input(artifact_data.get("tags"))
        artifact = Artifact(
            name=artifact_data.get("name", "Unknown"),
            value=artifact_data.get("value", "Unknown"),
            age=artifact_data.get("age", "Unknown"),
            description=artifact_data.get("description"),
            cultural_context=artifact_data.get("cultural_context"),
            material=artifact_data.get("material"),
            function=artifact_data.get("function"),
            rarity=artifact_data.get("rarity"),
            confidence=artifact_data.get("confidence", 0.0),
            image_data=image_bytes,
            analyzed_at=datetime.utcnow(),
            tags=",".join(tags_list) if tags_list else None,
        )
        db.add(artifact)
        db.flush()  # Obtain PK without committing twice
        artifact_id = artifact.id
        return artifact_id


def get_all_artifacts(
    limit: int = 100, offset: int = 0, include_images: bool = False
) -> List[Dict[str, Any]]:
    """Return a paginated list of artifacts; optionally embed base64 image data."""
    with get_db() as db:
        artifacts = (
            db.query(Artifact)
            .order_by(Artifact.uploaded_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )
        results: List[Dict[str, Any]] = []
        for artifact in artifacts:
            data = artifact.to_dict()
            if include_images and artifact.image_data:
                data["image_base64"] = base64.b64encode(artifact.image_data).decode(
                    "utf-8"
                )
            results.append(data)
        return results


def get_artifact_by_id(artifact_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a single artifact by its primary key."""
    with get_db() as db:
        artifact = db.query(Artifact).filter(Artifact.id == artifact_id).first()
        if not artifact:
            return None
        # Convert to dict and include image data to avoid detached instance errors
        data = artifact.to_dict()
        if artifact.image_data:
            data["image_data"] = artifact.image_data
        return data


def search_artifacts(query: str, limit: int = 50, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Search across textual columns and optionally filter by tags.

    If tags are provided, all tags must be present in the artifact's tag string.
    """
    with get_db() as db:
        pattern = f"%{query}%"
        q = db.query(Artifact).filter(
            (Artifact.name.ilike(pattern))
            | (Artifact.description.ilike(pattern))
            | (Artifact.cultural_context.ilike(pattern))
            | (Artifact.material.ilike(pattern))
        )
        tag_filters = _normalize_tags_input(tags)
        if tag_filters:
            for t in tag_filters:
                q = q.filter(Artifact.tags.ilike(f"%{t}%"))
        artifacts = q.order_by(Artifact.uploaded_at.desc()).limit(limit).all()
        return [a.to_dict() for a in artifacts]


def update_artifact_verification(
    artifact_id: int,
    status: str,
    verified_by: Optional[str] = None,
    comments: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Change verification fields for a given artifact."""
    with get_db() as db:
        artifact = db.query(Artifact).filter(Artifact.id == artifact_id).first()
        if not artifact:
            return None

        artifact.verification_status = status
        if verified_by:
            artifact.verified_by = verified_by
        if comments:
            artifact.verification_comments = comments
        if status.lower() == "verified":
            artifact.verified_at = datetime.utcnow()

        db.flush()
        return artifact.to_dict()


def update_artifact_tags(artifact_id: int, tags: Optional[Union[List[str], str]]) -> Optional[Dict[str, Any]]:
    """Update the tags for a given artifact and return its dict."""
    with get_db() as db:
        artifact = db.query(Artifact).filter(Artifact.id == artifact_id).first()
        if not artifact:
            return None
        tags_list = _normalize_tags_input(tags)
        artifact.tags = ",".join(tags_list) if tags_list else None
        db.flush()
        return artifact.to_dict()


def delete_artifact(artifact_id: int) -> bool:
    """Delete an artifact by id. Returns True if deleted."""
    with get_db() as db:
        artifact = db.query(Artifact).filter(Artifact.id == artifact_id).first()
        if not artifact:
            return False
        db.delete(artifact)
        db.flush()
        return True
