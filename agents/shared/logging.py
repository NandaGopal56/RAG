"""Agent-specific logging helpers built on ``shared.logging``."""

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

# Per-agent aggregate log files (useful for debugging one component over time).
AGENT_LOG_FILES: Dict[str, str] = {
    "supervisor": "supervisor.log",
    "personal": "personal.log",
    "deep_research": "deep_research.log",
    "tools": "tools.log",
    "checkpointer": "checkpointer.log",
    "storage": "storage.log",
    "client": "supervisor.log",
    "cli": "supervisor.log",
}

_STATE_TRUNCATE_LEN = 1200
_MESSAGE_LIST_KEYS = frozenset({"messages"})

_session_logger = get_logger("agents.session", log_file="session.log")
_invocation_ctx: ContextVar[Optional["InvocationContext"]] = ContextVar(
    "invocation_context", default=None
)
_active_session_handler: Optional[logging.Handler] = None


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


def _enrich_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    ctx = _invocation_ctx.get()
    if ctx is None:
        return fields
    enriched = {
        "invoke_id": ctx.invoke_id,
        "root_agent": ctx.root_agent,
        "thread": ctx.thread_id,
    }
    enriched.update(fields)
    return enriched


def _format_event_message(event: str, fields: Mapping[str, Any]) -> str:
    parts = [f"EVENT={event}"]
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, dict):
            parts.append(f"{key}={_format_details(value)}")
        elif isinstance(value, (list, tuple)):
            parts.append(f"{key}={json.dumps(value, default=str)}")
        else:
            parts.append(f"{key}={value}")
    return " | ".join(parts)


def _emit(
    logger: logging.Logger,
    level: int,
    message: str,
    *,
    mirror_session: bool = True,
) -> None:
    """Write to the component logger and, when active, the invocation session file."""
    logger.log(level, message)
    if mirror_session and _invocation_ctx.get() is not None:
        _session_logger.log(level, f"[{logger.name}] {message}")


def get_agent_logger(
    agent_id: str,
    component: Optional[str] = None,
    *,
    level: int = logging.DEBUG,
) -> logging.Logger:
    """Return a logger for *agent_id* writing to that agent's aggregate log file."""
    if component:
        name = f"agents.{agent_id}.{component}"
    else:
        name = f"agents.{agent_id}"
    log_file = AGENT_LOG_FILES.get(agent_id, f"{agent_id}.log")
    return get_logger(name, log_file=log_file, level=level)


def _format_details(details: Mapping[str, Any]) -> str:
    if not details:
        return "{}"
    try:
        return json.dumps(dict(details), default=str, ensure_ascii=False)
    except Exception:
        return repr(dict(details))


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

        if isinstance(value, str) and len(value) > _STATE_TRUNCATE_LEN:
            safe[key] = (
                value[:_STATE_TRUNCATE_LEN]
                + f"... [truncated, total={len(value)}]"
            )
        else:
            safe[key] = value

    return safe


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
    gateway_logger = get_agent_logger("client", "gateway")

    try:
        _emit(
            gateway_logger,
            logging.INFO,
            _format_event_message(
                "SESSION_START",
                _enrich_fields(
                    {
                        "agent": root_agent,
                        "mode": mode,
                        "log_path": str(log_path),
                    }
                ),
            ),
        )
        yield invoke_id
    finally:
        _emit(
            gateway_logger,
            logging.INFO,
            _format_event_message("SESSION_END", _enrich_fields({"agent": root_agent, "mode": mode})),
        )
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


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit a structured log line to the component log and active session."""
    message = _format_event_message(event, _enrich_fields(dict(fields)))
    _emit(logger, level, message)


def log_error(
    logger: logging.Logger,
    event: str,
    **fields: Any,
) -> None:
    """Emit a structured ERROR log line with traceback info."""
    message = _format_event_message(event, _enrich_fields(dict(fields)))
    _emit(logger, logging.ERROR, message, mirror_session=True)


def log_node_enter(
    logger: logging.Logger,
    node: str,
    *,
    thread_id: Optional[str] = None,
    state: Optional[Mapping[str, Any]] = None,
    **context: Any,
) -> None:
    """Log the start of a graph node at INFO; dump input state at DEBUG."""
    tid = thread_id or context.pop("thread_id", None)
    log_event(
        logger,
        "NODE_ENTER",
        node=node,
        thread_id=tid,
        context=_format_details(context) if context else None,
    )
    if state is not None:
        log_node_state(logger, node, state, label="input_state")


def log_node_exit(
    logger: logging.Logger,
    node: str,
    *,
    thread_id: Optional[str] = None,
    result_keys: Optional[list[str]] = None,
    **summary: Any,
) -> None:
    """Log the end of a graph node with a short result summary."""
    fields: Dict[str, Any] = {"node": node, "thread_id": thread_id}
    if result_keys is not None:
        fields["result_keys"] = ",".join(result_keys)
    fields.update(summary)
    log_event(logger, "NODE_EXIT", **fields)


def log_node_state(
    logger: logging.Logger,
    node: str,
    state: Mapping[str, Any],
    *,
    label: str = "state",
) -> None:
    """Dump sanitized state at DEBUG."""
    safe = sanitize_state_for_log(state)
    try:
        payload = json.dumps(safe, default=str, ensure_ascii=False)
    except Exception:
        payload = repr(safe)
    message = f"STATE {node}.{label}: {payload}"
    _emit(logger, logging.DEBUG, message)


def log_route(
    logger: logging.Logger,
    source: str,
    target: str,
    *,
    reason: str = "",
    **details: Any,
) -> None:
    """Log a conditional-edge or router decision at INFO."""
    log_event(
        logger,
        "ROUTE",
        source=source,
        target=target,
        reason=reason or None,
        details=_format_details(details) if details else None,
    )


def log_branch(
    logger: logging.Logger,
    node: str,
    branch: str,
    **details: Any,
) -> None:
    """Log an in-node branch decision at DEBUG."""
    log_event(
        logger,
        "BRANCH",
        level=logging.DEBUG,
        node=node,
        branch=branch,
        details=_format_details(details) if details else None,
    )


def log_llm_result(
    logger: logging.Logger,
    node: str,
    label: str,
    payload: Mapping[str, Any],
) -> None:
    """Log parsed LLM JSON / structured output at DEBUG."""
    try:
        body = json.dumps(dict(payload), default=str, ensure_ascii=False)
    except Exception:
        body = repr(dict(payload))
    message = f"STATE {node}.{label}: {body}"
    _emit(logger, logging.DEBUG, message)


def log_invoke_start(
    logger: logging.Logger,
    agent_id: str,
    *,
    thread_id: str,
    mode: str = "invoke",
    task_preview: str = "",
    resumed: bool = False,
) -> None:
    """Log the start of an agent invoke/stream/resume cycle."""
    preview = task_preview
    if len(preview) > 200:
        preview = preview[:200] + "..."
    log_event(
        logger,
        "INVOKE_START",
        agent=agent_id,
        thread=thread_id,
        mode=mode,
        resumed=resumed,
        task=preview or None,
    )


def log_invoke_end(
    logger: logging.Logger,
    agent_id: str,
    *,
    thread_id: str,
    mode: str = "invoke",
    response_length: Optional[int] = None,
    interrupted: bool = False,
    **extra: Any,
) -> None:
    """Log the end of an agent invoke/stream/resume cycle."""
    log_event(
        logger,
        "INVOKE_END",
        agent=agent_id,
        thread=thread_id,
        mode=mode,
        response_length=response_length,
        interrupted=interrupted,
        details=_format_details(extra) if extra else None,
    )


def log_stream_update(
    logger: logging.Logger,
    agent_id: str,
    *,
    thread_id: str,
    node: str,
    update_keys: Optional[list[str]] = None,
) -> None:
    """Log a single streamed graph update at DEBUG."""
    log_event(
        logger,
        "STREAM_UPDATE",
        level=logging.DEBUG,
        agent=agent_id,
        thread=thread_id,
        node=node,
        update_keys=",".join(update_keys) if update_keys else None,
    )


def log_save(entity: str, action: str, **details: Any) -> None:
    """Log a persistence write (message, tool call, checkpoint, etc.)."""
    save_logger = get_agent_logger("storage", "saves")
    log_event(save_logger, "SAVE", entity=entity, action=action, **details)


def log_tool_call(tool_name: str, **kwargs: Any) -> None:
    """Standardized tool-call logging."""
    tool_logger = get_agent_logger("tools", "tools")
    log_event(
        tool_logger,
        "TOOL_CALL",
        tool=tool_name,
        args=_format_details(kwargs),
    )
