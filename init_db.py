"""
Initialize the database schema
Run this once to create the tables
"""

from database import init_db

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Database initialized successfully!")
