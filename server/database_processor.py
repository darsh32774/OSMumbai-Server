import psycopg2
import os
from typing import List, Tuple, Any

DB_HOST     = os.getenv('SUPABASE_DB_HOST')
DB_PORT     = os.getenv('SUPABASE_DB_PORT')
DB_NAME     = os.getenv('SUPABASE_DB_NAME')
DB_USER     = os.getenv('SUPABASE_DB_USER')
DB_PASSWORD = os.getenv('SUPABASE_DB_PASSWORD')

def _get_connection():
    """Opens and returns a new psycopg2 connection."""
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT]):
        raise RuntimeError("Database connection details are incomplete. Check environment variables.")
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )

def ensure_extensions():
    """
    Ensures required PostgreSQL extensions are installed.
    Called ONCE at server startup, not on every query.
    """
    conn = None
    cur = None
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        conn.commit()
        print("Database extensions verified (pg_trgm).")
    except psycopg2.Error as e:
        # Non-fatal — pg_trgm likely already exists; log and continue.
        print(f"Warning: Could not ensure pg_trgm extension: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def execute_query_raw(sql_query: str) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    cleaned = sql_query.strip().upper()
    if not cleaned.startswith(("SELECT", "WITH")):
        raise ValueError("SQL validation failed: only SELECT or WITH queries are allowed.")

    sql_query = sql_query.strip().rstrip(';')

    conn = None
    cur = None
    try:
        conn = _get_connection()
        cur = conn.cursor()

        print(f"Executing SQL: {sql_query}")
        cur.execute(sql_query)

        headers = [desc[0] for desc in cur.description] if cur.description else []
        rows = cur.fetchall()

        return headers, rows

    except psycopg2.Error as e:
        print(f"Database query execution failed: {e}")
        print(f"Failed query: {sql_query}")
        raise RuntimeError(f"Database query failed: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
