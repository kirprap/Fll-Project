import os
from datetime import datetime
from typing import Optional, Dict, Any, List

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
        }


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def init_db() -> None:
    """Create all tables defined by the ORM models."""
    Base.metadata.create_all(bind=engine)


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


def save_artifact(artifact_data: Dict[str, Any], image_bytes: bytes) -> int:
    """Persist a newly analysed artifact and return its primary key."""
    with get_db() as db:
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


def get_artifact_by_id(artifact_id: int) -> Optional[Artifact]:
    """Fetch a single artifact by its primary key."""
    with get_db() as db:
        return db.query(Artifact).filter(Artifact.id == artifact_id).first()


def search_artifacts(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Search across a handful of textual columns."""
    with get_db() as db:
        pattern = f"%{query}%"
        artifacts = (
            db.query(Artifact)
            .filter(
                (Artifact.name.ilike(pattern))
                | (Artifact.description.ilike(pattern))
                | (Artifact.cultural_context.ilike(pattern))
                | (Artifact.material.ilike(pattern))
            )
            .order_by(Artifact.uploaded_at.desc())
            .limit(limit)
            .all()
        )
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
