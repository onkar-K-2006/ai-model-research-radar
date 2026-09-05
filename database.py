import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    model_name TEXT NOT NULL
);
"""


class ChatDatabase:
    """Manages SQLite storage for the chat history using Airflow SqliteHook."""

    def __init__(
        self,
        conn_id: str = "chatbot_sqlite_db",
        db_path: str = "/home/jovyan/streamlit/chatbot.db",
    ):
        self.conn_id = conn_id
        self.db_path = db_path
        self.init_db()

    def get_conn(self):
        """Attempts to obtain a connection via SqliteHook, falling back to direct sqlite3."""
        try:
            from airflow.providers.sqlite.hooks.sqlite import SqliteHook

            hook = SqliteHook(sqlite_conn_id=self.conn_id)
            conn = hook.get_conn()
            return conn
        except Exception as e:
            logger.warning(
                "Could not connect via Airflow SqliteHook '%s' (%s). Fallback to file path: %s",
                self.conn_id,
                e,
                self.db_path,
            )
            return sqlite3.connect(self.db_path)

    def init_db(self):
        """Creates the chat_history table if it doesn't already exist."""
        try:
            conn = self.get_conn()
            cursor = conn.cursor()
            cursor.execute(CREATE_TABLE_SQL)
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error("Failed to initialize chat database: %s", e)

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        model_name: str,
        timestamp: Optional[str] = None,
    ):
        """Inserts a single chat message record."""
        if not timestamp:
            timestamp = datetime.utcnow().isoformat()
        try:
            conn = self.get_conn()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO chat_history (session_id, timestamp, role, content, model_name)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, timestamp, role, content, model_name),
            )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error("Failed to save message to database: %s", e)

    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves all messages for a given session sorted chronologically."""
        messages: List[Dict[str, Any]] = []
        try:
            conn = self.get_conn()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, session_id, timestamp, role, content, model_name
                FROM chat_history
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            )
            rows = cursor.fetchall()
            for r in rows:
                messages.append(
                    {
                        "id": r[0],
                        "session_id": r[1],
                        "timestamp": r[2],
                        "role": r[3],
                        "content": r[4],
                        "model_name": r[5],
                    }
                )
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error("Failed to fetch session messages: %s", e)
        return messages

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Retrieves all distinct session IDs with message counts and latest activity."""
        sessions: List[Dict[str, Any]] = []
        try:
            conn = self.get_conn()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT session_id, COUNT(*) as msg_count, MAX(timestamp) as last_active
                FROM chat_history
                GROUP BY session_id
                ORDER BY last_active DESC
                """
            )
            rows = cursor.fetchall()
            for r in rows:
                sessions.append(
                    {
                        "session_id": r[0],
                        "msg_count": r[1],
                        "last_active": r[2],
                    }
                )
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error("Failed to list sessions: %s", e)
        return sessions

    def delete_session(self, session_id: str):
        """Deletes all messages for a specific session."""
        try:
            conn = self.get_conn()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error("Failed to delete session %s: %s", session_id, e)

    def clear_all_history(self):
        """Clears all records in chat_history."""
        try:
            conn = self.get_conn()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_history")
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error("Failed to clear chat history: %s", e)

