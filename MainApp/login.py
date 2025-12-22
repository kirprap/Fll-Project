"""
User authentication and management utilities.
This module provides database functions for user management and authentication.
The UI is now handled by the React frontend.
"""
import sqlite3
from datetime import datetime
import os
import bcrypt
import yaml
from yaml.loader import SafeLoader

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(PROJECT_DIR, "users.db")
DB =  os.path.join(PROJECT_DIR, "user.yaml")
# ----------------------------------------------------------------------
# Database Functions
# ----------------------------------------------------------------------
def init_user_db():
    """Initialize the users database and create tables if they don't exist."""
    with sqlite3.connect(DB_FILE, timeout=10) as conn:
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS users (
                        username TEXT PRIMARY KEY,
                        name TEXT,
                        hashed_password TEXT,
                        role TEXT,
                        email TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS audit_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        username TEXT,
                        action TEXT)""")
        # Load config and insert users from user.yaml if not in DB
        try:
            with open(DB) as file:
                config = yaml.load(file, Loader=SafeLoader)
            
            # Insert users from config.yaml if not in DB
            for username, details in config.get("credentials", {}).get("usernames", {}).items():
                c.execute("SELECT * FROM users WHERE username=?", (username,))
                if not c.fetchone():
                    # Hash password if it's not already hashed
                    password = details.get("password", "")
                    if not password.startswith("$2b$"):  # bcrypt hash prefix
                        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
                    else:
                        hashed_pw = password
                    
                    c.execute(
                        "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                        (
                            username,
                            details.get("name", ""),
                            hashed_pw,
                            details.get("role", "user"),
                            details.get("email", ""),
                        ),
                    )
        except FileNotFoundError:
            # user.yaml not found, skip initialization from config
            pass
        except Exception as e:
            # Log error but don't fail initialization
            print(f"Warning: Could not load users from user.yaml: {e}")
        
        conn.commit()


def log_action(username, action):
    """Log an action to the audit log."""
    with sqlite3.connect(DB_FILE, timeout=10) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO audit_logs (timestamp, username, action) VALUES (?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), username, action),
        )
        conn.commit()


def get_user_role(username):
    """Get the role of a user by username."""
    with sqlite3.connect(DB_FILE, timeout=10) as conn:
        c = conn.cursor()
        c.execute("SELECT role FROM users WHERE username=?", (username,))
        result = c.fetchone()
        return result[0] if result else None


def get_user_info(username):
    """Get user information (name, email, role) by username."""
    with sqlite3.connect(DB_FILE, timeout=10) as conn:
        c = conn.cursor()
        c.execute("SELECT name, email, role FROM users WHERE username=?", (username,))
        return c.fetchone()


def get_users():
    """Get all users from the database."""
    with sqlite3.connect(DB_FILE, timeout=10) as conn:
        c = conn.cursor()
        c.execute("SELECT username, name, role, email FROM users")
        return c.fetchall()


def add_user(username, name, password, role, email):
    """Add a new user to the database."""
    with sqlite3.connect(DB_FILE, timeout=10) as conn:
        c = conn.cursor()
        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        try:
            c.execute(
                "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                (username, name, hashed_pw, role, email),
            )
            conn.commit()
            log_action(username, f"User {username} created")
            return True
        except sqlite3.IntegrityError:
            return False


def reset_password(username, new_password):
    """Reset a user's password."""
    with sqlite3.connect(DB_FILE, timeout=10) as conn:
        c = conn.cursor()
        hashed_pw = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
        c.execute(
            "UPDATE users SET hashed_password=? WHERE username=?", (hashed_pw, username)
        )
        conn.commit()
        log_action(username, "Password reset")


def get_audit_logs():
    """Get the last 50 audit log entries."""
    with sqlite3.connect(DB_FILE, timeout=10) as conn:
        c = conn.cursor()
        c.execute(
            "SELECT timestamp, username, action FROM audit_logs ORDER BY id DESC LIMIT 50"
        )
        return c.fetchall()


# Initialize database on module import
init_user_db()
