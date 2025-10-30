"""
Database models and operations for archaeological artifact storage
"""

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

SQLALCHEMY_DATABASE_URL = "sqlite:///artifacts.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


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
        """Convert artifact to dictionary"""

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
            "uploaded_at": self.uploaded_at.isoformat()
            if self.uploaded_at is not None
            else None,
            "analyzed_at": self.analyzed_at.isoformat()
            if self.analyzed_at is not None
            else None,
            "verification_status": self.verification_status,
            "verified_by": self.verified_by,
            "verified_at": self.verified_at.isoformat()
            if self.verified_at is not None
            else None,
            "verification_comments": self.verification_comments,
            "provenance": self.provenance,
            "historical_context": self.historical_context,
            "references": self.references,
        }


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db():
    """Get database session with automatic cleanup"""
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
    """Save an identified artifact to the database"""
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
        db.commit()
        db.refresh(artifact)
        return artifact.id


def get_all_artifacts(
    limit: int = 100, offset: int = 0, include_images: bool = False
) -> List[Dict[str, Any]]:
    """Get all artifacts with pagination"""
    with get_db() as db:
        artifacts = (
            db.query(Artifact)
            .order_by(Artifact.uploaded_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )

        results = []
        for artifact in artifacts:
            data = artifact.to_dict()
            if include_images and artifact.image_data is not None:
                data["image_base64"] = base64.b64encode(artifact.image_data).decode(
                    "utf-8"
                )
            results.append(data)
        return results


def get_artifact_by_id(artifact_id):
    """Get a specific artifact by ID"""
    with get_db() as db:
        artifact = db.query(Artifact).filter(Artifact.id == artifact_id).first()
        return artifact


def search_artifacts(query, limit=50):
    """Search artifacts by name, description, material, or cultural context"""
    with get_db() as db:
        search_pattern = f"%{query}%"
        artifacts = (
            db.query(Artifact)
            .filter(
                (Artifact.name.ilike(search_pattern))
                | (Artifact.description.ilike(search_pattern))
                | (Artifact.cultural_context.ilike(search_pattern))
                | (Artifact.material.ilike(search_pattern))
            )
            .order_by(Artifact.uploaded_at.desc())
            .limit(limit)
            .all()
        )
        return [artifact.to_dict() for artifact in artifacts]


def update_artifact_verification(
    artifact_id: int,
    status: str,
    verified_by: Optional[str] = None,
    comments: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Update artifact verification status"""

    with get_db() as db:
        artifact = db.query(Artifact).filter(Artifact.id == artifact_id).first()

        if artifact:
            artifact.verification_status = status

            artifact.verified_by = verified_by

            artifact.verified_at = datetime.utcnow()

            artifact.verification_comments = comments

            db.commit()
            db.refresh(artifact)

            return artifact.to_dict()

        return None


def update_artifact_profile(
    artifact_id: int,
    provenance: Optional[str] = None,
    historical_context: Optional[str] = None,
    references: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Update detailed artifact profile information"""

    with get_db() as db:
        artifact = db.query(Artifact).filter(Artifact.id == artifact_id).first()

        if artifact:
            if provenance is not None:
                artifact.provenance = provenance

            if historical_context is not None:
                artifact.historical_context = historical_context

            if references is not None:
                artifact.references = references

            db.commit()
            db.refresh(artifact)

            return artifact.to_dict()

        return None


def get_artifacts_by_verification_status(status, limit=50):
    """Get artifacts filtered by verification status"""
    with get_db() as db:
        artifacts = (
            db.query(Artifact)
            .filter(Artifact.verification_status == status)
            .order_by(Artifact.uploaded_at.desc())
            .limit(limit)
            .all()
        )
        return [artifact.to_dict() for artifact in artifacts]


def get_artifact_count():
    """Get total number of artifacts in database"""
    with get_db() as db:
        return db.query(Artifact).count()


# NEW helper for Streamlit sidebar / sample database
# NEW helper for Streamlit sidebar / sample database
def get_artifact_database():
    """
    Returns a dictionary of artifacts grouped by verification status
    so Streamlit can do: for category, artifacts in artifact_db.items()
    """
    grouped = {"Pending": [], "Verified": [], "Rejected": []}
    # fetch all artifacts once
    all_artifacts = get_all_artifacts(limit=500)

    for artifact in all_artifacts:
        status = artifact.get("verification_status", "pending").lower()
        if status == "verified":
            grouped["Verified"].append(artifact)
        elif status == "rejected":
            grouped["Rejected"].append(artifact)
        else:
            grouped["Pending"].append(artifact)

    return grouped


# initialize tables
init_db()
