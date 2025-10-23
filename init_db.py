"""
Initialize the database schema
Run this once to create the tables
"""

import os
import sys
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from database import Base
from config import DATABASE_URL

def init_db():
    """Initialize the database"""
    try:
        # Create engine
        engine = create_engine(DATABASE_URL)
        
        # Create database if it doesn't exist
        if not database_exists(engine.url):
            create_database(engine.url)
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        print("✅ Database initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing database: {str(e)}")
        return False

if __name__ == "__main__":
    print("Initializing database...")
    success = init_db()
    sys.exit(0 if success else 1)
