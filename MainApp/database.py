import base64
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Prefer the DATABASE_URL environment variable (set by docker-compose or .env).
# If it is missing or empty, fall back to a local SQLite file.
_DB_URL = os.getenv("DATABASE_URL")
if not _DB_URL:
    # Default SQLite database located in the project root directory
    _DB_DIR = os.path.dirname(os.path.abspath(__file__))
    _DB_PATH = os.path.join(_DB_DIR, "artifacts.db")
    _DB_URL = f"sqlite:///{_DB_PATH}"

# SQLite requires a special ``check_same_thread`` flag; other DBMS do not.
# Add connection pooling for better performance
if _DB_URL.startswith("sqlite"):
    engine = create_engine(
        _DB_URL,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,  # Verify connections before using
        pool_recycle=3600,  # Recycle connections after 1 hour
    )
else:
    # PostgreSQL connection pooling
    engine = create_engine(
        _DB_URL,
        pool_size=10,  # Number of connections to maintain
        max_overflow=20,  # Additional connections when pool is full
        pool_pre_ping=True,  # Verify connections before using
        pool_recycle=3600,  # Recycle connections after 1 hour
        echo=False,  # Disable SQL logging for performance
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
    thumbnail: Optional[bytes] = Column(LargeBinary)
    
    # 3D model data
    model_3d_data: Optional[bytes] = Column(LargeBinary)
    model_3d_format: Optional[str] = Column(String(10))  # e.g., "obj", "stl", "ply"

    # Timestamps
    uploaded_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)
    analyzed_at: Optional[datetime] = Column(DateTime, default=datetime.utcnow)
    updated_at: Optional[datetime] = Column(DateTime)

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

    # Form data (JSON string with physical measurements)
    form_data: Optional[str] = Column(Text)

    # Analysis tier used
    tier: Optional[str] = Column(String(50))

    # User who uploaded the artifact
    uploaded_by: Optional[str] = Column(String(200))

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
            "image_data": self.image_data,
            "thumbnail": self.thumbnail,
            "model_3d_data": self.model_3d_data,
            "model_3d_format": self.model_3d_format,
            "uploaded_at": self.uploaded_at.isoformat() + "Z"
            if self.uploaded_at
            else None,
            "analyzed_at": self.analyzed_at.isoformat() + "Z"
            if self.analyzed_at
            else None,
            "updated_at": self.updated_at.isoformat() + "Z"
            if self.updated_at
            else None,
            "verification_status": self.verification_status,
            "verified_by": self.verified_by,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "verification_comments": self.verification_comments,
            "provenance": self.provenance,
            "historical_context": self.historical_context,
            "references": self.references,
            "tags": self.tags,
            "form_data": self.form_data,
            "tier": self.tier,
            "uploaded_by": self.uploaded_by,
        }


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def init_db() -> None:
    """Create all tables defined by the ORM models."""
    Base.metadata.create_all(bind=engine)
    # Ensure required columns exist for existing databases
    try:
        inspector = inspect(engine)
        columns = [c["name"] for c in inspector.get_columns("artifacts")]
        with engine.connect() as conn:
            if "tags" not in columns:
                conn.execute(text("ALTER TABLE artifacts ADD COLUMN tags TEXT"))
            if "thumbnail" not in columns:
                conn.execute(text("ALTER TABLE artifacts ADD COLUMN thumbnail BLOB"))
            if "form_data" not in columns:
                conn.execute(text("ALTER TABLE artifacts ADD COLUMN form_data TEXT"))
            if "tier" not in columns:
                conn.execute(text("ALTER TABLE artifacts ADD COLUMN tier VARCHAR(50)"))
            if "updated_at" not in columns:
                conn.execute(text("ALTER TABLE artifacts ADD COLUMN updated_at DATETIME"))
            if "model_3d_data" not in columns:
                conn.execute(text("ALTER TABLE artifacts ADD COLUMN model_3d_data BLOB"))
            if "model_3d_format" not in columns:
                conn.execute(text("ALTER TABLE artifacts ADD COLUMN model_3d_format VARCHAR(10)"))
            if "uploaded_by" not in columns:
                conn.execute(text("ALTER TABLE artifacts ADD COLUMN uploaded_by VARCHAR(200)"))
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


def save_artifact(
    artifact_data: Dict[str, Any],
    image_bytes: bytes = None,
    thumbnail_bytes: bytes = None,
    model_3d_bytes: bytes = None,
) -> int:
    """Persist a newly analysed artifact and return its primary key."""
    with get_db() as db:
        tags_list = _normalize_tags_input(artifact_data.get("tags"))
        # Handle both old format (image_data in artifact_data) and new format (separate params)
        image_to_save = image_bytes if image_bytes else artifact_data.get("image_data")
        thumbnail_to_save = (
            thumbnail_bytes if thumbnail_bytes else artifact_data.get("thumbnail")
        )
        # Handle 3D model data
        model_3d_to_save = model_3d_bytes if model_3d_bytes else artifact_data.get("model_3d_data")

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
            image_data=image_to_save,
            thumbnail=thumbnail_to_save,
            model_3d_data=model_3d_to_save,
            model_3d_format=artifact_data.get("model_3d_format"),
            analyzed_at=datetime.utcnow(),
            tags=",".join(tags_list) if tags_list else None,
            form_data=artifact_data.get("form_data"),
            tier=artifact_data.get("tier"),
            uploaded_by=artifact_data.get("uploaded_by"),
        )
        db.add(artifact)
        db.flush()  # Obtain PK without committing twice
        artifact_id = artifact.id
        return artifact_id


def get_all_artifacts(
    limit: int = 100, offset: int = 0, include_images: bool = True
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
            # Convert binary fields to base64 strings for JSON serialization
            if include_images:
                if data.get("image_data") is not None:
                    data["image_data"] = base64.b64encode(data["image_data"]).decode(
                        "utf-8"
                    )
                if data.get("thumbnail") is not None:
                    data["thumbnail"] = base64.b64encode(data["thumbnail"]).decode(
                        "utf-8"
                    )
            else:
                # Remove binary fields if not including images
                data.pop("image_data", None)
                data.pop("thumbnail", None)
            results.append(data)
        return results


def get_artifact_by_id(artifact_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a single artifact by its primary key."""
    with get_db() as db:
        artifact = db.query(Artifact).filter(Artifact.id == artifact_id).first()
        if not artifact:
            return None
        # Convert to dict which now includes image_data and thumbnail
        data = artifact.to_dict()
        # Convert binary fields to base64 strings for JSON serialization
        if data.get("image_data") is not None:
            data["image_data"] = base64.b64encode(data["image_data"]).decode("utf-8")
        if data.get("thumbnail") is not None:
            data["thumbnail"] = base64.b64encode(data["thumbnail"]).decode("utf-8")
        return data


def search_artifacts(
    query: str,
    limit: int = 50,
    tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Search artifacts by keywords in text fields and optional tags."""

    keywords = [kw for kw in (query or "").split() if kw]
    tag_filters = _normalize_tags_input(tags)

    with get_db() as db:
        q = db.query(Artifact)

        if keywords:
            for kw in keywords:
                pattern = f"%{kw}%"
                q = q.filter(
                    Artifact.id.in_(
                        db.query(Artifact.id).filter(
                            (Artifact.name.ilike(pattern))
                            | (Artifact.description.ilike(pattern))
                            | (Artifact.cultural_context.ilike(pattern))
                            | (Artifact.material.ilike(pattern))
                            | (Artifact.tags.ilike(pattern))
                        )
                    )
                )
        else:
            # No keywords: still allow tag filtering without restricting base query
            q = q.filter(True)  # no-op filter for consistent chaining

        if tag_filters:
            for t in tag_filters:
                q = q.filter(Artifact.tags.ilike(f"%{t}%"))

        artifacts = q.order_by(Artifact.uploaded_at.desc()).limit(limit).all()
        results: List[Dict[str, Any]] = []
        for artifact in artifacts:
            data = artifact.to_dict()
            # Convert binary fields to base64 strings for JSON serialization
            if data.get("image_data") is not None:
                data["image_data"] = base64.b64encode(data["image_data"]).decode(
                    "utf-8"
                )
            if data.get("thumbnail") is not None:
                data["thumbnail"] = base64.b64encode(data["thumbnail"]).decode("utf-8")
            results.append(data)
        return results


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
        data = artifact.to_dict()
        # Convert binary fields to base64 strings for JSON serialization
        if data.get("image_data") is not None:
            data["image_data"] = base64.b64encode(data["image_data"]).decode("utf-8")
        if data.get("thumbnail") is not None:
            data["thumbnail"] = base64.b64encode(data["thumbnail"]).decode("utf-8")
        return data


def update_artifact_tags(
    artifact_id: int, tags: Optional[Union[List[str], str]]
) -> Optional[Dict[str, Any]]:
    """Update the tags for a given artifact and return its dict."""
    with get_db() as db:
        artifact = db.query(Artifact).filter(Artifact.id == artifact_id).first()
        if not artifact:
            return None
        tags_list = _normalize_tags_input(tags)
        artifact.tags = ",".join(tags_list) if tags_list else None
        db.flush()
        data = artifact.to_dict()
        # Convert binary fields to base64 strings for JSON serialization
        if data.get("image_data") is not None:
            data["image_data"] = base64.b64encode(data["image_data"]).decode("utf-8")
        if data.get("thumbnail") is not None:
            data["thumbnail"] = base64.b64encode(data["thumbnail"]).decode("utf-8")
        return data


def update_artifact(artifact_id: int, update_data: Dict[str, Any]) -> bool:
    """Update an artifact with the provided data. Returns True if updated."""
    with get_db() as db:
        artifact = db.query(Artifact).filter(Artifact.id == artifact_id).first()
        if not artifact:
            return False

        # Update fields that are provided
        if "name" in update_data:
            artifact.name = update_data["name"]
        if "description" in update_data:
            artifact.description = update_data["description"]
        if "tags" in update_data:
            tags_list = _normalize_tags_input(update_data["tags"])
            artifact.tags = ",".join(tags_list) if tags_list else None
        if "form_data" in update_data:
            artifact.form_data = update_data["form_data"]
        if "verification_status" in update_data:
            artifact.verification_status = update_data["verification_status"]
            if update_data["verification_status"] == "verified":
                artifact.verified_at = datetime.utcnow()

        artifact.updated_at = datetime.utcnow()
        db.flush()
        return True


def delete_artifact(artifact_id: int) -> bool:
    """Delete an artifact by id. Returns True if deleted."""
    with get_db() as db:
        artifact = db.query(Artifact).filter(Artifact.id == artifact_id).first()
        if not artifact:
            return False
        db.delete(artifact)
        db.flush()
        return True
