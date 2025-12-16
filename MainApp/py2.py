import sqlite3
import bcrypt
import os

# Path to your users database
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(PROJECT_DIR, "users.db")

# Users from your YAML
users_to_add = [
    {"username": "admin", "name": "Admin User", "password": "Admin@123", "role": "admin", "email": "admin@example.com"},
    {"username": "user2", "name": "Field Engineer", "password": "abc", "role": "user", "email": "user2@example.com"},
    {"username": "user3", "name": "Guest User", "password": "abc", "role": "user", "email": "user3@example.com"}
]

# Connect to SQLite DB
with sqlite3.connect(DB_FILE) as conn:
    cursor = conn.cursor()

    for user in users_to_add:
        # Hash the password
        hashed_password = bcrypt.hashpw(user["password"].encode(), bcrypt.gensalt()).decode()

        # Check if user exists
        cursor.execute("SELECT username FROM users WHERE username=?", (user["username"],))
        exists = cursor.fetchone()

        if exists:
            print(f"⚠️ User '{user['username']}' exists. Updating password...")
            cursor.execute(
                "UPDATE users SET name=?, hashed_password=?, role=?, email=? WHERE username=?",
                (user["name"], hashed_password, user["role"], user["email"], user["username"])
            )
        else:
            print(f"✅ Creating user '{user['username']}'")
            cursor.execute(
                "INSERT INTO users (username, name, hashed_password, role, email) VALUES (?, ?, ?, ?, ?)",
                (user["username"], user["name"], hashed_password, user["role"], user["email"])
            )
    conn.commit()
print("All users added/updated successfully!")
