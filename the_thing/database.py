import sqlite3
import re
import hashlib
from datetime import datetime


STAFF_ID_PATTERN = r'^[A-Za-z]{2}\d{6}[A-Za-z]$'
DATABASE_NAME = "secure_database.db"

def create_connection():
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        return conn
    except Exception as e:
        message = f"Error connecting to database: {e}"
        print(message)
        return None

def create_table():
    conn = create_connection()
    if conn is None:
        return "Error connecting to database."
    try:  # hmmm
        query = f"""
        CREATE TABLE IF NOT EXISTS users (
            staff_id TEXT PRIMARY KEY,
            first_name TEXT,
            surname TEXT,
            password TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
        conn.execute(query)
        conn.commit()
        conn.close()
        message = "Table created successfully."
        print(message)
        return message
    except Exception as e:
        message = f"Error creating table: {e}"
        print(message)
        return message

def hash_password(password, staff_id):
    return hashlib.sha256(f"{password}{staff_id}".encode()).hexdigest()  # SHA-256 with staff_id for salt

def register_user(staff_id, first_name, surname, password, confirm_password):
    # Normalize staff_id to uppercase
    staff_id = staff_id.upper()
    if not re.match(STAFF_ID_PATTERN, staff_id):
        message = ("Invalid staff ID format. It should be two letters, followed by 6 numbers, and ending with a letter.")
        # print(message)
        return message
    if not first_name.isalpha():
        message = "Error: First name must contain only letters."
        # print(message)
        return message
    if not surname.isalpha():
        message = "Error: Surname must contain only letters."
        # print(message)
        return message

    # Enforce minimum password length
    if len(password) < 5:
        message = "Error: Password must be at least 5 characters long."
        # print(message)
        return message

    first_name = first_name.capitalize()
    surname = surname.capitalize()
    if password != confirm_password:
        message = "Error: Passwords do not match."
        # print(message)
        return message
    conn = create_connection()
    if conn is None:
        message = "Error connecting to database."
        # print(message)
        return message
    try:
        hashed_password = hash_password(password, staff_id)
        query = "INSERT INTO users (staff_id, first_name, surname, password) VALUES (?, ?, ?, ?);"
        conn.execute(query, (staff_id, first_name, surname, hashed_password))
        conn.commit()
        conn.close()
        message = "User registered successfully."
        # print(message)
        return message

    except sqlite3.IntegrityError as ie:   # Don't care
        message = f"Error: Staff ID '{staff_id}' already exists."
        # print(message)
        return message
    except Exception as e:
        message = f"Error registering user: {e}"
        # print(message)
        return message

def login_user(staff_id, password):
    # Normalize staff_id to uppercase
    staff_id = staff_id.upper()
    # Validate staff_id format
    if not re.match(STAFF_ID_PATTERN, staff_id):
        message = "Invalid staff ID format."
        # print(message)
        return message

    # Enforce minimum password length on login
    if len(password) < 5:
        message = "Error: Password must be at least 5 characters long."
        # print(message)
        return message

    conn = create_connection()
    if conn is None:
        message = "Error connecting to database."
        # print(message)
        return message

    try:
        query = "SELECT password, first_name, surname FROM users WHERE staff_id = ?;"
        cursor = conn.execute(query, (staff_id,))
        row = cursor.fetchone()
        if row is None:
            message = "Staff ID not found."
            # print(message)
            conn.close()
            return message

        stored_hash, first_name, surname = row
        hashed_password = hash_password(password, staff_id)
        if hashed_password == stored_hash:
            message = "Login successful."
            full_name = f"{first_name} {surname}"
            conn.close()
            # Return a tuple with the message, staff_id and full name
            return (message, staff_id, full_name)
        else:
            message = "Incorrect password."
            # print(message)
            conn.close()
            return message
    except Exception as e:
        message = f"Error during login: {e}"
        # print(message)
        conn.close()
        return message

def display_all_user_names():
    conn = create_connection()
    if conn is None:
        message = "Error connecting to database."
        print(message)
        return message
    try:
        query = "SELECT staff_id, first_name, surname FROM users;"
        cursor = conn.execute(query)
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            message = "No users found."
            print(message)
            return message
        for row in rows:
            staff_id, first_name, surname = row
            print(f"Staff ID: {staff_id} - Name: {first_name} {surname}")
        return rows
    except Exception as e:
        message = f"Error retrieving user names: {e}"
        print(message)
        return message

def delete_user(staff_id):
    # Normalize staff_id to uppercase
    staff_id = staff_id.upper()
    if not re.match(STAFF_ID_PATTERN, staff_id):
        message = "Invalid staff ID format."
        print(message)
        return message

    conn = create_connection()
    if conn is None:
        message = "Error connecting to database."
        print(message)
        return message
    try:
        query = "DELETE FROM users WHERE staff_id = ?;"
        cursor = conn.execute(query, (staff_id,))
        conn.commit()
        if cursor.rowcount == 0:
            message = f"No user found with staff ID {staff_id}."
        else:
            message = f"User with staff ID {staff_id} deleted successfully."
        conn.close()
        print(message)
        return message
    except Exception as e:
        message = f"Error deleting user: {e}"
        print(message)
        return message

def delete_all():
    conn = create_connection()
    if conn is None:
        message = "Error connecting to database."
        print(message)
        return message
    try:
        query = "DELETE FROM users;"
        conn.execute(query)
        conn.commit()
        conn.close()
        message = "All users deleted successfully."
        print(message)
        return message
    except Exception as e:
        message = f"Error deleting all users: {e}"
        print(message)
        return message


# if __name__ == "__main__":
#     delete_all()
#     create_table()
#     display_all_user_names()
