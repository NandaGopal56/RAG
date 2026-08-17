# agents/shared/storage.py
#
# SQLite persistence layer — copied exactly from the original storage code.
# All function signatures and behaviour are preserved as-is.

import aiosqlite
import json
import asyncio
from datetime import datetime
from pathlib import Path

from agents.shared.logging import get_agent_logger, log_event

DB_PATH = Path(".storages") / "bot_data_v2.db"
logger = get_agent_logger("storage", "storage")


def _database_path() -> str:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return str(DB_PATH)


# -------------------------------------------------------------
#  DATABASE INIT
# -------------------------------------------------------------

async def init_db():
    async with aiosqlite.connect(_database_path()) as db:
        await db.execute("PRAGMA foreign_keys = ON;")

        await db.executescript("""
        CREATE TABLE IF NOT EXISTS threads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            sequence INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS ai_tool_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL,
            call_id TEXT,
            tool_input_json JSON NOT NULL,
            tool_output JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE
        );
        """)
        await db.commit()
        log_event(logger, "DB_INIT", path=_database_path())


# -------------------------------------------------------------
#  INTERNAL UTILITY
# -------------------------------------------------------------

async def _get_next_sequence(db, thread_id: int) -> int:
    row = await db.execute(
        "SELECT COALESCE(MAX(sequence), 0) + 1 FROM messages WHERE thread_id = ?",
        (thread_id,)
    )
    return (await row.fetchone())[0]


# -------------------------------------------------------------
#  THREAD OPERATIONS
# -------------------------------------------------------------

async def create_thread(title: str = "New Chat") -> int:
    async with aiosqlite.connect(_database_path()) as db:
        cur = await db.execute("INSERT INTO threads (title) VALUES (?)", (title,))
        await db.commit()
        return cur.lastrowid


# -------------------------------------------------------------
#  MESSAGE OPERATIONS
# -------------------------------------------------------------

async def add_message(thread_id: int, role: str, content: str = None) -> int:
    async with aiosqlite.connect(_database_path()) as db:
        seq = await _get_next_sequence(db, thread_id)
        cur = await db.execute("""
            INSERT INTO messages (thread_id, role, content, sequence)
            VALUES (?, ?, ?, ?)
        """, (thread_id, role, content, seq))
        await db.commit()
        return cur.lastrowid


# -------------------------------------------------------------
#  TOOL OPERATIONS
# -------------------------------------------------------------

async def add_tool_call(message_id: int, tool_input_json: dict, call_id: str = None) -> int:
    async with aiosqlite.connect(_database_path()) as db:
        cur = await db.execute("""
            INSERT INTO ai_tool_calls (message_id, call_id, tool_input_json)
            VALUES (?, ?, ?)
        """, (message_id, call_id, json.dumps(tool_input_json)))
        await db.commit()
        return cur.lastrowid


async def update_tool_result(message_id: int, call_id: str, tool_output: dict):
    async with aiosqlite.connect(_database_path()) as db:
        await db.execute("""
            UPDATE ai_tool_calls
            SET tool_output = ?
            WHERE message_id = ? AND call_id = ?
        """, (json.dumps(tool_output), message_id, call_id))
        await db.commit()


# -------------------------------------------------------------
#  CHAT RECONSTRUCTION
# -------------------------------------------------------------

async def get_full_thread(thread_id: int):
    async with aiosqlite.connect(_database_path()) as db:
        db.row_factory = aiosqlite.Row

        cursor = await db.execute("""
            SELECT
                messages.id AS message_id,
                messages.role,
                messages.content,
                messages.created_at AS message_created_at,
                messages.sequence AS message_sequence,
                ai_tool_calls.created_at AS tool_call_created_at,
                ai_tool_calls.call_id AS tool_call_id,
                ai_tool_calls.tool_input_json AS tool_tool_input_json,
                ai_tool_calls.tool_output AS tool_tool_output
            FROM messages
            LEFT JOIN ai_tool_calls ON messages.id = ai_tool_calls.message_id
            WHERE messages.thread_id = ?
            ORDER BY messages.created_at ASC, ai_tool_calls.created_at ASC
        """, (thread_id,))

        rows = await cursor.fetchall()

        messages_by_id = {}
        order = []
        for r in rows:
            message_id = r["message_id"]
            if message_id not in messages_by_id:
                messages_by_id[message_id] = {
                    "message_id": message_id,
                    "role":       r["role"],
                    "content":    r["content"],
                    "sequence":   r["message_sequence"],
                    "tool_calls": [],
                }
                order.append(message_id)

            if r["tool_call_id"] is not None:
                tool_input = r["tool_tool_input_json"]
                if tool_input:
                    try:
                        tool_input = json.loads(tool_input)
                    except json.JSONDecodeError:
                        pass
                messages_by_id[message_id]["tool_calls"].append({
                    "call_id":        r["tool_call_id"],
                    "tool_input_json": tool_input,
                    "tool_output":    r["tool_tool_output"],
                })

        return [messages_by_id[mid] for mid in order]