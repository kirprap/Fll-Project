"""
Database models and operations for archaeological artifact storage
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

Base = declarative_base()

def get_database_engine():
    """Get database engine, raising error if DATABASE_URL is not configured"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is not configured. Please ensure PostgreSQL database is set up.")
    return create_engine(database_url)

def get_session_maker():
    """Get session maker for database operations"""
    engine = get_database_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Artifact(Base):
    """Model for storing identified archaeological artifacts"""
    __tablename__ = "artifacts"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Core artifact information from AI analysis
    name = Column(String(500), nullable=False)
    value = Column(String(200))
    age = Column(String(300))
    description = Column(Text)
    cultural_context = Column(Text)
    material = Column(String(500))
    function = Column(Text)
    rarity = Column(String(200))
    confidence = Column(Float)
    
    # Image data
    image_data = Column(LargeBinary)
    
    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    
    # Expert verification fields
    verification_status = Column(String(50), default="pending")
    verified_by = Column(String(200))
    verified_at = Column(DateTime)
    verification_comments = Column(Text)
    
    # Detailed profile fields
    provenance = Column(Text)
    historical_context = Column(Text)
    references = Column(Text)
    
    def to_dict(self):
        """Convert artifact to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'value': self.value,
            'age': self.age,
            'description': self.description,
            'cultural_context': self.cultural_context,
            'material': self.material,
            'function': self.function,
            'rarity': self.rarity,
            'confidence': self.confidence,
            'uploaded_at': self.uploaded_at.isoformat() if self.uploaded_at else None,
            'analyzed_at': self.analyzed_at.isoformat() if self.analyzed_at else None,
            'verification_status': self.verification_status,
            'verified_by': self.verified_by,
            'verified_at': self.verified_at.isoformat() if self.verified_at else None,
            'verification_comments': self.verification_comments,
            'provenance': self.provenance,
            'historical_context': self.historical_context,
            'references': self.references
        }


def init_db():
    """Initialize database tables"""
    engine = get_database_engine()
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db():
    """Get database session with automatic cleanup"""
    SessionLocal = get_session_maker()
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def save_artifact(artifact_data, image_bytes):
    """
    Save an identified artifact to the database
    
    Args:
        artifact_data: Dictionary with artifact information from AI analysis
        image_bytes: Binary image data
    
    Returns:
        The saved Artifact object
    """
    with get_db() as db:
        artifact = Artifact(
            name=artifact_data.get('name', 'Unknown'),
            value=artifact_data.get('value', 'Unknown'),
            age=artifact_data.get('age', 'Unknown'),
            description=artifact_data.get('description'),
            cultural_context=artifact_data.get('cultural_context'),
            material=artifact_data.get('material'),
            function=artifact_data.get('function'),
            rarity=artifact_data.get('rarity'),
            confidence=artifact_data.get('confidence', 0.0),
            image_data=image_bytes,
            analyzed_at=datetime.utcnow()
        )
        db.add(artifact)
        db.flush()
        db.refresh(artifact)
        return artifact


def get_all_artifacts(limit=100, offset=0, include_images=False):
    """Get all artifacts with pagination
    
    Args:
        limit: Maximum number of artifacts to return
        offset: Number of artifacts to skip
        include_images: If True, include base64-encoded images in results
    """
    with get_db() as db:
        artifacts = db.query(Artifact).order_by(
            Artifact.uploaded_at.desc()
        ).limit(limit).offset(offset).all()
        
        results = []
        for artifact in artifacts:
            data = artifact.to_dict()
            if include_images and artifact.image_data:
                import base64
                data['image_base64'] = base64.b64encode(artifact.image_data).decode('utf-8')
            results.append(data)
        return results


def get_artifact_by_id(artifact_id):
    """Get a specific artifact by ID"""
    with get_db() as db:
        artifact = db.query(Artifact).filter(Artifact.id == artifact_id).first()
        return artifact


def search_artifacts(query, limit=50):
    """
    Search artifacts by name, description, or cultural context
    
    Args:
        query: Search query string
        limit: Maximum number of results
    
    Returns:
        List of matching artifacts
    """
    with get_db() as db:
        search_pattern = f"%{query}%"
        artifacts = db.query(Artifact).filter(
            (Artifact.name.ilike(search_pattern)) |
            (Artifact.description.ilike(search_pattern)) |
            (Artifact.cultural_context.ilike(search_pattern)) |
            (Artifact.material.ilike(search_pattern))
        ).order_by(Artifact.uploaded_at.desc()).limit(limit).all()
        return [artifact.to_dict() for artifact in artifacts]


def update_artifact_verification(artifact_id, status, verified_by=None, comments=None):
    """
    Update artifact verification status
    
    Args:
        artifact_id: ID of the artifact
        status: Verification status (pending, verified, rejected)
        verified_by: Name of the expert who verified
        comments: Verification comments
    """
    with get_db() as db:
        artifact = db.query(Artifact).filter(Artifact.id == artifact_id).first()
        if artifact:
            artifact.verification_status = status
            artifact.verified_by = verified_by
            artifact.verified_at = datetime.utcnow()
            artifact.verification_comments = comments
            db.flush()
            return artifact
        return None


def update_artifact_profile(artifact_id, provenance=None, historical_context=None, references=None):
    """
    Update detailed artifact profile information
    
    Args:
        artifact_id: ID of the artifact
        provenance: Provenance information
        historical_context: Additional historical context
        references: References and citations
    """
    with get_db() as db:
        artifact = db.query(Artifact).filter(Artifact.id == artifact_id).first()
        if artifact:
            if provenance is not None:
                artifact.provenance = provenance
            if historical_context is not None:
                artifact.historical_context = historical_context
            if references is not None:
                artifact.references = references
            db.flush()
            return artifact
        return None


def get_artifacts_by_verification_status(status, limit=50):
    """Get artifacts filtered by verification status"""
    with get_db() as db:
        artifacts = db.query(Artifact).filter(
            Artifact.verification_status == status
        ).order_by(Artifact.uploaded_at.desc()).limit(limit).all()
        return [artifact.to_dict() for artifact in artifacts]


def get_artifact_count():
    """Get total number of artifacts in database"""
    with get_db() as db:
        return db.query(Artifact).count()
