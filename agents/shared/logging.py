"""Agent-specific logging with automatic session mirroring.

Usage::

    from agents.shared.logging import get_agent_logger

    logger = get_agent_logger("supervisor", "nodes")
    logger.info("NODE_ENTER node=route_request thread_id=%s", tid)
    logger.error("INVOKE_ERROR agent=%s error=%s", agent, err)

Standard ``logger.info()`` / ``logger.error()`` / ``logger.debug()`` calls
are automatically mirrored to the active invocation session file when a
session context is active.
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Mapping, Optional

from shared.logging import LOGS_DIR, get_logger

SESSIONS_DIR = LOGS_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

STATE_TRUNCATE_LEN = 1200
_MESSAGE_LIST_KEYS = frozenset({"messages"})

_session_logger = get_logger("agents.session", log_file="session.log")
_invocation_ctx: ContextVar[Optional["InvocationContext"]] = ContextVar(
    "invocation_context", default=None
)
_active_session_handler: Optional[logging.Handler] = None


# ---------------------------------------------------------------------------
# Invocation context (ContextVar-based, propagates through async tasks)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InvocationContext:
    """Tracks one user-facing invocation across nested agent graphs."""

    invoke_id: str
    root_agent: str
    thread_id: str
    mode: str
    log_path: str


def get_invocation_context() -> Optional[InvocationContext]:
    """Return the active invocation context, if any."""
    return _invocation_ctx.get()


# ---------------------------------------------------------------------------
# Custom Logger that mirrors to the active session
# ---------------------------------------------------------------------------

class AgentLogger(logging.Logger):
    """Standard logger that automatically mirrors writes to the session log."""

    def _log_with_mirror(
        self,
        level: int,
        msg: str,
        args: tuple,
        **kwargs: Any,
    ) -> None:
        super()._log(level, msg, args, **kwargs)
        ctx = _invocation_ctx.get()
        if ctx is not None:
            if args:
                try:
                    rendered = msg % args
                except (TypeError, ValueError):
                    rendered = msg
            else:
                rendered = msg
            _session_logger.log(level, f"[{self.name}] {rendered}")

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log_with_mirror(logging.DEBUG, msg, args)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log_with_mirror(logging.INFO, msg, args)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log_with_mirror(logging.WARNING, msg, args)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log_with_mirror(logging.ERROR, msg, args)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log_with_mirror(logging.CRITICAL, msg, args)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log_with_mirror(logging.ERROR, msg, args)


# Keep a reference to the original factory so we can restore if needed
_original_factory = logging.getLogRecordFactory()


def _agent_log_record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
    return _original_factory(*args, **kwargs)


logging.setLogRecordFactory(_agent_log_record_factory)

# Register AgentLogger so logging.getLogger() returns our subclass
logging.setLoggerClass(AgentLogger)


# ---------------------------------------------------------------------------
# Session handler management
# ---------------------------------------------------------------------------

def _safe_path_token(value: str) -> str:
    return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _session_log_path(invoke_id: str, root_agent: str, thread_id: str) -> Path:
    filename = (
        f"{invoke_id}_{_safe_path_token(root_agent)}"
        f"_thread-{_safe_path_token(thread_id)}.log"
    )
    return SESSIONS_DIR / filename


def _attach_session_handler(path: Path) -> logging.Handler:
    global _active_session_handler

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    _session_logger.handlers.clear()
    _session_logger.addHandler(handler)
    _active_session_handler = handler
    return handler


def _detach_session_handler(handler: logging.Handler) -> None:
    global _active_session_handler

    _session_logger.removeHandler(handler)
    handler.close()
    _active_session_handler = None


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_agent_logger(
    agent_id: str,
    component: Optional[str] = None,
    *,
    level: int = logging.DEBUG,
) -> AgentLogger:
    """Return a logger for *agent_id* writing to that agent's aggregate log file."""
    agent_log_files: Dict[str, str] = {
        "supervisor": "supervisor.log",
        "personal": "personal.log",
        "deep_research": "deep_research.log",
        "tools": "tools.log",
        "checkpointer": "checkpointer.log",
        "storage": "storage.log",
        "client": "supervisor.log",
        "cli": "supervisor.log",
    }

    if component:
        name = f"agents.{agent_id}.{component}"
    else:
        name = f"agents.{agent_id}"
    log_file = agent_log_files.get(agent_id, f"{agent_id}.log")

    logger = get_logger(name, log_file=log_file, level=level)
    # Replace with our AgentLogger if not already one
    if not isinstance(logger, AgentLogger):
        agent_logger = AgentLogger(name, level=level)
        agent_logger.handlers = logger.handlers
        agent_logger.propagate = logger.propagate
        agent_logger.log_file_path = getattr(logger, "log_file_path", None)  # type: ignore[attr-defined]
        # Cache in the logger registry so future getLogger() calls return it
        logging.Logger.manager.loggerDict[name] = agent_logger  # type: ignore[attr-defined]
        return agent_logger
    return logger  # type: ignore[return-value]


def sanitize_state_for_log(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a trimmed copy of graph state safe for DEBUG dumps."""
    safe: Dict[str, Any] = {}

    for key, value in state.items():
        if key in _MESSAGE_LIST_KEYS:
            try:
                previews = []
                for msg in value:
                    content = getattr(msg, "content", str(msg))
                    words = str(content).split()
                    preview = " ".join(words[:10])
                    if len(words) > 10:
                        preview += "..."
                    previews.append(preview)

                safe[key] = previews
            except Exception:
                safe[key] = "[messages]"
            continue

        if isinstance(value, str) and len(value) > STATE_TRUNCATE_LEN:
            safe[key] = (
                value[:STATE_TRUNCATE_LEN]
                + f"... [truncated, total={len(value)}]"
            )
        else:
            safe[key] = value

    return safe


# ---------------------------------------------------------------------------
# Invocation session context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def invocation_session(
    root_agent: str,
    thread_id: str,
    *,
    mode: str = "invoke",
) -> AsyncIterator[str]:
    """Open a per-invocation session log that captures all mirrored agent events.

    One session file is created under ``.logs/sessions/`` for the full call
    tree. Nested supervisor -> personal/deep_research runs share the same
    session because the context propagates through async tasks.
    """
    invoke_id = uuid.uuid4().hex[:12]
    log_path = _session_log_path(invoke_id, root_agent, thread_id)
    handler = _attach_session_handler(log_path)

    ctx = InvocationContext(
        invoke_id=invoke_id,
        root_agent=root_agent,
        thread_id=thread_id,
        mode=mode,
        log_path=str(log_path),
    )
    token: Token = _invocation_ctx.set(ctx)
    gw = get_agent_logger("client", "gateway")

    try:
        gw.info(
            "SESSION_START invoke_id=%s agent=%s mode=%s log_path=%s",
            invoke_id, root_agent, mode, log_path,
        )
        yield invoke_id
    finally:
        gw.info("SESSION_END agent=%s mode=%s", root_agent, mode)
        _invocation_ctx.reset(token)
        _detach_session_handler(handler)


@asynccontextmanager
async def ensure_invocation_session(
    agent_id: str,
    thread_id: str,
    *,
    mode: str = "invoke",
) -> AsyncIterator[Optional[str]]:
    """Start a session only when one is not already active.

    Use at agent ``invoke``/``stream`` entry points so direct calls still get a
    session file, while nested subgraph work reuses the parent session.
    """
    existing = _invocation_ctx.get()
    if existing is not None:
        yield existing.invoke_id
        return

    async with invocation_session(agent_id, thread_id, mode=mode) as invoke_id:
        yield invoke_id
