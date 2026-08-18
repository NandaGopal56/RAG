"""
Persistent SQLite checkpointer utilities for multi-turn agent conversations.

Provides:
  - get_checkpointer: Creates agent-specific AsyncSqliteSaver instances
  - load_previous_state: Restores prior state from checkpoint for a thread
  - merge_with_new_messages: Merges restored state with new incoming messages
"""

import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from agents.shared.logging import get_agent_logger

logger = get_agent_logger("checkpointer", "checkpointer")

_SAVER_LOOP: Optional[asyncio.AbstractEventLoop] = None
_SAVER_THREAD: Optional[threading.Thread] = None


def _get_saver_loop() -> asyncio.AbstractEventLoop:
    """Get or create a dedicated event loop for checkpoint saver initialization."""
    global _SAVER_LOOP, _SAVER_THREAD

    if _SAVER_LOOP is None:
        _SAVER_LOOP = asyncio.new_event_loop()
        _SAVER_THREAD = threading.Thread(target=_SAVER_LOOP.run_forever, daemon=True)
        _SAVER_THREAD.start()

    return _SAVER_LOOP


def get_checkpointer(agent_id: str) -> AsyncSqliteSaver:
    """Create an AsyncSqliteSaver for a specific agent."""
    storages_dir = Path(__file__).parent.parent.parent / ".storages"
    storages_dir.mkdir(exist_ok=True, parents=True)

    db_path = storages_dir / f"{agent_id}.db"

    async def _create_saver() -> AsyncSqliteSaver:
        conn = await aiosqlite.connect(str(db_path))
        return AsyncSqliteSaver(conn)

    loop = _get_saver_loop()
    future = asyncio.run_coroutine_threadsafe(_create_saver(), loop)
    saver = future.result()

    try:
        orig_aput = saver.aput

        async def logged_aput(config, checkpoint, metadata, new_versions):
            tid = config.get("configurable", {}).get("thread_id")
            logger.log(
                10,
                "CHECKPOINT_PUT agent=%s thread=%s checkpoint_id=%s",
                agent_id, tid, checkpoint.get("id"),
            )
            return await orig_aput(config, checkpoint, metadata, new_versions)

        saver.aput = logged_aput  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        orig_aput_writes = saver.aput_writes

        async def logged_aput_writes(config, writes, task_id, task_path: str = ""):
            tid = config.get("configurable", {}).get("thread_id")
            logger.log(
                10,
                "CHECKPOINT_PUT_WRITES agent=%s thread=%s write_count=%s task_id=%s",
                agent_id, tid, len(writes), task_id,
            )
            return await orig_aput_writes(config, writes, task_id, task_path)

        saver.aput_writes = logged_aput_writes  # type: ignore[attr-defined]
    except Exception:
        pass

    logger.info("CHECKPOINTER_READY agent=%s db_path=%s", agent_id, str(db_path))
    return saver


async def load_previous_state(
    graph,
    thread_id: str,
    agent_id: str,
) -> Optional[Dict[str, Any]]:
    """Load the last checkpoint/state for a given thread_id and agent."""
    from langgraph.types import RunnableConfig

    if not thread_id:
        return None

    try:
        cfg = RunnableConfig(configurable={"thread_id": thread_id})
        state = await graph.aget_state(cfg)
        found = bool(state and state.values)
        logger.info("STATE_LOAD agent=%s thread=%s found=%s", agent_id, thread_id, found)
        if state and state.values:
            return dict(state.values)
        return None
    except Exception as e:
        logger.warning("STATE_LOAD_ERROR agent=%s thread=%s error=%s", agent_id, thread_id, e)
        return None


def merge_with_new_messages(
    previous_state: Optional[Dict[str, Any]],
    new_state_values: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge a loaded previous checkpoint with new incoming state values."""
    if previous_state is None:
        logger.log(
            10,
            "STATE_MERGE previous=None new_message_count=%s",
            len(new_state_values.get("messages", [])),
        )
        return new_state_values

    merged = dict(previous_state)
    previous_messages = previous_state.get("messages", [])
    new_messages = new_state_values.get("messages", [])

    logger.log(
        10,
        "STATE_MERGE previous_message_count=%s new_message_count=%s",
        len(previous_messages), len(new_messages),
    )

    if new_messages:
        previous_ids = {
            getattr(m, "id", None) for m in previous_messages
            if hasattr(m, "id") and m.id
        }

        unique_new_messages = [
            m for m in new_messages
            if not (hasattr(m, "id") and m.id and m.id in previous_ids)
        ]

        merged["messages"] = list(previous_messages) + unique_new_messages
        logger.log(
            10,
            "STATE_MERGE_RESULT added_unique=%s total_messages=%s",
            len(unique_new_messages), len(merged["messages"]),
        )
    else:
        merged["messages"] = list(previous_messages)

    for key, value in new_state_values.items():
        if key != "messages":
            merged[key] = value

    return merged
